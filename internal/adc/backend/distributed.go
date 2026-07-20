package backend

// Row-partitioned multi-device backend for the limited-memory (Mode B) block-Lanczos
// driver. One sector's resident state — the ~4 live n×main Krylov panels and the
// block-sparse operator — is split across G sub-backends along the config (row/n)
// dimension, so a block that dwarfs one GPU (137 GB for melanin) fits when spread over
// a node's GPUs. It composes existing single-device backends (Gonum for host-side
// correctness tests, gpuBackend for scale), so the same code path is validated on CPU
// before it ever touches a GPU.
//
// WHY ROW-PARTITION. Every reduction the solver performs contracts the row dimension
// into a main×main (or scalar) result — α = Q_curᵀ·W, the CGS2 projection coefficients,
// the Gram, Dot/Nrm2 — so those become a local partial per device plus a tiny all-reduce.
// The operator is partitioned too. Only the mat-vec needs a real cross-device exchange:
// an operator block at group (row r, col c) reads in[group_c] on the device owning
// group_r. The caller supplies group-aligned partition boundaries so every block's row
// band lands on one device (see dip.PartitionBounds); the input slice may be remote and
// is gathered per apply — over NVLink (PeerCopier) when the sub-backends support it, else
// staged through the host.
//
// SHAPE INVARIANT. The Backend interface allocates by flat length (Alloc(int)), so a
// vector is classified as a row-partitioned panel iff its length is a multiple of n; the
// small orthogonalization scratch (≤ 2·main²) must therefore never reach n in length.
// NewDistributed enforces n > 2·main² so the two can never alias — trivially true at the
// production scale this exists for (melanin: n ≈ 14.75M ≫ 2.7M).

import "fmt"

// distBackend spreads the row dimension across subs. bound holds the G+1 partition
// boundaries (bound[0]=0, bound[G]=n); device d owns global rows [bound[d], bound[d+1]).
type distBackend struct {
	Gonum // inherit the host SymEig / any method not overridden below
	subs  []Backend
	n     int
	main  int
	bound []int
}

// NewDistributed builds a row-partitioned backend over subs (one per device), splitting n
// global rows at the supplied group-aligned boundaries. main is the 2h main-block size; it
// must satisfy n > 2·main² so small scratch buffers never alias a panel length (see the
// file comment). bounds must be ascending, start at 0, end at n, and have len(subs)+1
// entries.
func NewDistributed(subs []Backend, n, main int, bounds []int) (Backend, error) {
	if len(subs) < 1 {
		return nil, fmt.Errorf("distributed backend: need at least one sub-backend")
	}
	if len(bounds) != len(subs)+1 || bounds[0] != 0 || bounds[len(bounds)-1] != n {
		return nil, fmt.Errorf("distributed backend: bounds must be len(subs)+1, start 0, end n=%d (got %v)", n, bounds)
	}
	for i := 1; i < len(bounds); i++ {
		if bounds[i] < bounds[i-1] {
			return nil, fmt.Errorf("distributed backend: bounds not ascending: %v", bounds)
		}
	}
	if n <= 2*main*main {
		return nil, fmt.Errorf("distributed backend: shape invariant n>2·main² violated (n=%d, main=%d)", n, main)
	}
	enablePeers(subs)
	return &distBackend{subs: subs, n: n, main: main, bound: append([]int(nil), bounds...)}, nil
}

// enablePeers grants every peer-capable sub-backend NVLink read access to all the others,
// once at setup, so the mat-vec input gather can copy device-to-device instead of staging
// through the host. Backends that do not implement PeerCopier (e.g. Gonum) are left as-is
// and keep the host-staging fallback.
func enablePeers(subs []Backend) {
	for i, s := range subs {
		pc, ok := s.(PeerCopier)
		if !ok {
			continue
		}
		others := make([]Backend, 0, len(subs)-1)
		for j, o := range subs {
			if j != i {
				others = append(others, o)
			}
		}
		pc.EnablePeerAccess(others)
	}
}

func (b *distBackend) ndev() int        { return len(b.subs) }
func (b *distBackend) rowsOn(d int) int { return b.bound[d+1] - b.bound[d] }

// devOf returns the device index owning global row r (the largest d with bound[d] <= r).
func (b *distBackend) devOf(r int) int {
	for d := range b.ndev() {
		if r < b.bound[d+1] {
			return d
		}
	}
	return b.ndev() - 1
}

// distVec is a resident vector spread across the sub-backends. Two flavours:
//   - panel (repl=false): a row-partitioned grows×gcols column-major panel; part[d] holds
//     rowsOn(d)×gcols on device d, leading dimension rowsOn(d). grows == b.n for a full
//     panel; a RowRange view narrows it to a single device's band (see loc).
//   - replicated (repl=true): a small buffer duplicated identically on every device, so any
//     device can use it as a GEMM factor; part[d] is the full length on device d.
//
// loc, when non-nil, marks a single-device located sub-block (produced by RowRange): it
// names the owning device and the local row offset/height, which the operator apply uses to
// route the GEMM and gather a remote input.
type distVec struct {
	b            *distBackend
	part         []Vector // one per device (panel: the row-slice; replicated: the full copy)
	repl         bool
	grows, gcols int  // logical shape for Slice decoding (panel: grows=b.n unless located)
	loc          *loc // non-nil: a single-device row band
}

type loc struct {
	dev    int
	rowOff int // local row offset within the device's storage
	rows   int
	ld     int // leading dimension of part[dev] (= rowsOn(dev))
}

func (v distVec) Len() int {
	if v.repl {
		return v.part[0].Len()
	}
	return v.grows * v.gcols
}

// Slice resolves the column-major slicing the solver performs. On a replicated buffer it
// slices every copy identically. On a panel it distinguishes a column range (off and len
// both multiples of b.n) from a row band (a RowRange, landing on one device).
func (v distVec) Slice(off, n int) Vector {
	if v.repl {
		parts := make([]Vector, len(v.part))
		for d := range v.part {
			parts[d] = v.part[d].Slice(off, n)
		}
		return distVec{b: v.b, part: parts, repl: true, grows: 1, gcols: n}
	}
	N := v.b.n
	if off%N == 0 && n%N == 0 {
		// Column range [c0, c0+cols).
		c0, cols := off/N, n/N
		parts := make([]Vector, len(v.part))
		for d := range v.part {
			rd := v.b.rowsOn(d)
			parts[d] = v.part[d].Slice(c0*rd, cols*rd)
		}
		return distVec{b: v.b, part: parts, grows: N, gcols: cols}
	}
	// Row band (a RowRange): off = r0 (col 0, row r0), len = (cols-1)*N + rows with
	// 1 ≤ rows ≤ N. The band's column count is recovered from the length, NOT from the
	// panel's allocated gcols — a deflated block uses fewer columns than were allocated.
	r0 := off
	cols := (n-1)/N + 1
	rows := n - (cols-1)*N
	if rows <= 0 || rows > N || r0 < 0 || r0+rows > N {
		panic(fmt.Sprintf("distributed: unsupported slice off=%d n=%d (grows=%d gcols=%d)", off, n, v.grows, v.gcols))
	}
	d := v.b.devOf(r0)
	if r0+rows > v.b.bound[d+1] {
		panic(fmt.Sprintf("distributed: row band [%d,%d) crosses partition boundary %d — bounds must be group-aligned",
			r0, r0+rows, v.b.bound[d+1]))
	}
	rd := v.b.rowsOn(d)
	localOff := r0 - v.b.bound[d]
	// Enclosing contiguous span of the band within device d's column-major storage.
	span := (cols-1)*rd + rows
	parts := make([]Vector, len(v.part))
	parts[d] = v.part[d].Slice(localOff, span)
	return distVec{b: v.b, part: parts, grows: rows, gcols: cols,
		loc: &loc{dev: d, rowOff: 0, rows: rows, ld: rd}}
}

// --- memory management -------------------------------------------------------

// panelCols reports the column count if len is a row-partitioned panel (len % n == 0),
// else 0 (a replicated small buffer). The shape invariant (n > 2·main²) guarantees no
// small buffer is a multiple of n.
func (b *distBackend) panelCols(length int) int {
	if length%b.n == 0 {
		return length / b.n
	}
	return 0
}

func (b *distBackend) Alloc(length int) Vector {
	if cols := b.panelCols(length); cols > 0 {
		parts := make([]Vector, b.ndev())
		for d := range parts {
			parts[d] = b.subs[d].Alloc(b.rowsOn(d) * cols)
		}
		return distVec{b: b, part: parts, grows: b.n, gcols: cols}
	}
	parts := make([]Vector, b.ndev())
	for d := range parts {
		parts[d] = b.subs[d].Alloc(length)
	}
	return distVec{b: b, part: parts, repl: true, grows: 1, gcols: length}
}

func (b *distBackend) Upload(host Vec) Vector {
	if cols := b.panelCols(len(host)); cols > 0 {
		parts := make([]Vector, b.ndev())
		for d := range parts {
			rd := b.rowsOn(d)
			sub := make([]float64, rd*cols)
			// Gather device d's rows out of the global column-major host panel.
			for c := range cols {
				copy(sub[c*rd:(c+1)*rd], host[c*b.n+b.bound[d]:c*b.n+b.bound[d]+rd])
			}
			parts[d] = b.subs[d].Upload(sub)
		}
		return distVec{b: b, part: parts, grows: b.n, gcols: cols}
	}
	parts := make([]Vector, b.ndev())
	for d := range parts {
		parts[d] = b.subs[d].Upload(host)
	}
	return distVec{b: b, part: parts, repl: true, grows: 1, gcols: len(host)}
}

func (b *distBackend) Download(v Vector) Vec {
	dv := v.(distVec)
	if dv.repl {
		return b.subs[0].Download(dv.part[0])
	}
	cols := dv.gcols
	out := make([]float64, b.n*cols)
	for d := range b.ndev() {
		if dv.part[d] == nil {
			continue
		}
		rd := b.rowsOn(d)
		sub := b.subs[d].Download(dv.part[d])
		for c := range cols {
			copy(out[c*b.n+b.bound[d]:c*b.n+b.bound[d]+rd], sub[c*rd:(c+1)*rd])
		}
	}
	return out
}

func (b *distBackend) Zero(v Vector) {
	dv := v.(distVec)
	for d := range dv.part {
		if dv.part[d] != nil {
			b.subs[d].Zero(dv.part[d])
		}
	}
}

func (b *distBackend) Copy(dst, src Vector) {
	d, s := dst.(distVec), src.(distVec)
	for i := range d.part {
		if d.part[i] != nil && s.part[i] != nil {
			b.subs[i].Copy(d.part[i], s.part[i])
		}
	}
}

func (b *distBackend) Free(v Vector) {
	dv, ok := v.(distVec)
	if !ok {
		return
	}
	for d := range dv.part {
		if dv.part[d] != nil {
			b.subs[d].Free(dv.part[d])
		}
	}
}

// --- local BlockView reconstruction -----------------------------------------

// panelLocal reconstructs device d's slice of a row-partitioned panel operand as a
// stand-alone BlockView: rows = rowsOn(d), leading dimension rowsOn(d), the panel's
// column count carried through. The caller's global Ld (= b.n) is intentionally ignored.
func (b *distBackend) panelLocal(bv BlockView, d int) BlockView {
	dv := bv.V.(distVec)
	if dv.repl || dv.loc != nil {
		panic("distributed: expected a full row-partitioned panel operand")
	}
	return BlockView{V: dv.part[d], Rows: b.rowsOn(d), Cols: bv.Cols, Ld: b.rowsOn(d)}
}

// smallLocal returns device d's copy of a replicated small operand, unchanged in shape.
func (b *distBackend) smallLocal(bv BlockView, d int) BlockView {
	dv := bv.V.(distVec)
	if !dv.repl {
		panic("distributed: expected a replicated small operand")
	}
	return BlockView{V: dv.part[d], Rows: bv.Rows, Cols: bv.Cols, Ld: bv.Ld}
}

// --- BLAS-3 on panels --------------------------------------------------------

// Gemm handles the two shapes the Mode B driver emits. transA=true contracts the
// partitioned row dimension into a small main×main (or scalar-row) result: each device
// forms its partial and the results are summed and replicated on every device (a tiny
// all-reduce). transA=false is a panel update c = alpha·a·b + beta·c against a replicated
// small factor b, done locally on each device with no communication.
func (b *distBackend) Gemm(transA, transB bool, alpha float64, a, bb BlockView, beta float64, c BlockView) {
	if transA {
		cdv := c.V.(distVec)
		if !cdv.repl {
			panic("distributed Gemm(transA=true): output must be a replicated small buffer")
		}
		if beta != 0 {
			// The Mode B reduces (α, Gram, CGS2 projections) all overwrite; supporting
			// beta would require saving the pre-image before the per-device partials.
			panic("distributed Gemm(transA=true): beta!=0 unsupported")
		}
		// Each device computes its partial straight into its own copy of the small output,
		// honouring c.Ld (the buffer's leading dimension may exceed c.Rows — e.g. the CGS2
		// projection buffer — so a contiguous write would corrupt the strided layout).
		for d := range b.ndev() {
			cLocal := BlockView{V: cdv.part[d], Rows: c.Rows, Cols: c.Cols, Ld: c.Ld}
			b.subs[d].Gemm(true, transB, alpha, b.panelLocal(a, d), b.panelLocal(bb, d), 0, cLocal)
		}
		// All-reduce: sum the per-device buffers and replicate the total back to every device.
		// Gaps outside the c.Rows×c.Cols result region are never read by the consumer, so
		// summing whole buffers is safe.
		acc := b.subs[0].Download(cdv.part[0])
		for d := 1; d < b.ndev(); d++ {
			part := b.subs[d].Download(cdv.part[d])
			for i := range acc {
				acc[i] += part[i]
			}
		}
		for d := range b.ndev() {
			up := b.subs[d].Upload(acc)
			b.subs[d].Copy(cdv.part[d], up)
			b.subs[d].Free(up)
		}
		return
	}
	for d := range b.ndev() {
		b.subs[d].Gemm(false, transB, alpha, b.panelLocal(a, d), b.smallLocal(bb, d), beta, b.panelLocal(c, d))
	}
}

// AddPanel adds the full n×cols column-major host panel into dst (a row-partitioned panel),
// giving each device its own row band. It backs the matrix-free DIP satellite apply under
// -mgpu (dip/matfree_dist.go): the operator gathers the full input with Download, recomputes
// the satellite contribution on the host, and scatter-adds it here, so the satellite region
// never materializes on any device. dst must be a row-partitioned panel (not replicated).
func (b *distBackend) AddPanel(dst Vector, full []float64) {
	dv := dst.(distVec)
	if dv.repl {
		panic("distributed AddPanel: destination must be a row-partitioned panel")
	}
	cols := len(full) / b.n
	for d := range b.ndev() {
		if dv.part[d] == nil {
			continue
		}
		rd := b.rowsOn(d)
		band := make([]float64, rd*cols)
		for c := range cols {
			copy(band[c*rd:(c+1)*rd], full[c*b.n+b.bound[d]:c*b.n+b.bound[d]+rd])
		}
		up := b.subs[d].Upload(band)
		// The panel is column-major with leading dimension rowsOn(d), and may have more columns
		// allocated than this apply uses (the solver sizes panels to the max block width), so add
		// only into the first cols columns — their rd·cols storage is contiguous at the front.
		b.subs[d].Axpy(1, up, dv.part[d].Slice(0, rd*cols))
		b.subs[d].Free(up)
	}
}

// --- resident operator blocks ------------------------------------------------

// distMat is a block-sparse operator block held on the host, uploaded lazily to whichever
// device(s) apply it. An off-diagonal block is applied in both directions (into its row
// band and, transposed, into its column band), so it can end up resident on up to two
// devices — the devices owning those two group bands. Uploading only on demand is how the
// operator is partitioned (~operator/G per device) rather than replicated.
type distMat struct {
	b    *distBackend
	host Mat
	dev  []DeviceMat
}

func (m *distMat) Dims() (int, int) { return m.host.Rows, m.host.Cols }

func (m *distMat) on(d int) DeviceMat {
	if m.dev[d] == nil {
		m.dev[d] = m.b.subs[d].UploadMat(m.host)
	}
	return m.dev[d]
}

func (b *distBackend) UploadMat(m Mat) DeviceMat {
	return &distMat{b: b, host: m, dev: make([]DeviceMat, b.ndev())}
}

func (b *distBackend) FreeMat(m DeviceMat) {
	dm, ok := m.(*distMat)
	if !ok {
		return
	}
	for d := range dm.dev {
		if dm.dev[d] != nil {
			b.subs[d].FreeMat(dm.dev[d])
		}
	}
}

// gemmMatOne applies one operator block: c += op(a)·b, on the device owning c's row band.
// If b's row band lives on another device it is gathered there first (NVLink peer copy when
// available, else host-staged). a is uploaded to the output device on first use.
func (b *distBackend) gemmMatOne(transA bool, alpha float64, a DeviceMat, bb, c BlockView, beta float64) {
	dm := a.(*distMat)
	cdv := c.V.(distVec)
	if cdv.loc == nil {
		panic("distributed GemmMat: output is not a located row band")
	}
	do := cdv.loc.dev
	cLocal := BlockView{V: cdv.part[do], Rows: cdv.loc.rows, Cols: c.Cols, Ld: cdv.loc.ld}

	bdv := bb.V.(distVec)
	if bdv.loc == nil {
		panic("distributed GemmMat: input is not a located row band")
	}
	var bLocal BlockView
	if bdv.loc.dev == do {
		bLocal = BlockView{V: bdv.part[do], Rows: bdv.loc.rows, Cols: bb.Cols, Ld: bdv.loc.ld}
		b.subs[do].GemmMat(transA, alpha, dm.on(do), bLocal, beta, cLocal)
		return
	}
	// Remote input: compact its band onto the output device, contiguous (Ld=rows). Over
	// NVLink this is one peer copy; without peer access it stages through the host (the band
	// is the operator apply's one large cross-device mover, so the peer path is the point).
	di := bdv.loc.dev
	rows, cols, ld := bdv.loc.rows, bb.Cols, bdv.loc.ld
	var band Vector
	if pc, ok := b.subs[do].(PeerCopier); ok && pc.PeerAvailable(b.subs[di]) {
		// Drain the source device first: a peer read does not synchronize the source stream,
		// and the band may still be mid-write from an async panel kernel (unlike the host
		// path below, whose Download drains it implicitly). subs[di] is a *gpuBackend here
		// (PeerAvailable proved it), so it implements PeerCopier.Sync.
		b.subs[di].(PeerCopier).Sync()
		band = b.subs[do].Alloc(rows * cols)
		pc.PeerCopy2D(band, bdv.part[di], b.subs[di], rows, cols, ld)
	} else {
		span := b.subs[di].Download(bdv.part[di])
		compact := make([]float64, rows*cols)
		for cc := range cols {
			copy(compact[cc*rows:(cc+1)*rows], span[cc*ld:cc*ld+rows])
		}
		band = b.subs[do].Upload(compact)
	}
	bLocal = BlockView{V: band, Rows: rows, Cols: cols, Ld: rows}
	b.subs[do].GemmMat(transA, alpha, dm.on(do), bLocal, beta, cLocal)
	b.subs[do].Free(band)
}

func (b *distBackend) GemmMat(transA bool, alpha float64, a DeviceMat, bb BlockView, beta float64, c BlockView) {
	b.gemmMatOne(transA, alpha, a, bb, c, beta)
}

func (b *distBackend) GemmMatBatched(transA bool, alpha float64, a []DeviceMat, bb []BlockView, beta float64, c []BlockView) {
	for i := range a {
		b.gemmMatOne(transA, alpha, a[i], bb[i], c[i], beta)
	}
}

// --- unsupported outside the Mode B block path -------------------------------

func (b *distBackend) Axpy(float64, Vector, Vector) {
	panic("distributed backend: Axpy unsupported (Mode B block-Lanczos path only)")
}
func (b *distBackend) Dot(Vector, Vector) float64 {
	panic("distributed backend: Dot unsupported (Mode B block-Lanczos path only)")
}
func (b *distBackend) Nrm2(Vector) float64 {
	panic("distributed backend: Nrm2 unsupported (Mode B block-Lanczos path only)")
}
func (b *distBackend) Scal(float64, Vector) {
	panic("distributed backend: Scal unsupported (Mode B block-Lanczos path only)")
}
func (b *distBackend) AxpyDiag(Vector, Vector, Vector) {
	panic("distributed backend: AxpyDiag unsupported (Mode B block-Lanczos path only)")
}
func (b *distBackend) GemvN(float64, DeviceMat, Vector, Vector) {
	panic("distributed backend: GemvN unsupported (Mode B block-Lanczos path only)")
}
func (b *distBackend) GemvT(float64, DeviceMat, Vector, Vector) {
	panic("distributed backend: GemvT unsupported (Mode B block-Lanczos path only)")
}

var _ Backend = (*distBackend)(nil)
