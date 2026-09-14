package backend

// Row-partitioned multi-device backend for the limited-memory (Mode B) block-Lanczos
// driver. One sector's resident state — the ~4 live n×main Krylov panels and the
// block-sparse operator — is split across G sub-backends along the config (row/n)
// dimension, so a block that dwarfs one GPU (137 GB for the production system) fits when spread over
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
// production scale this exists for (n ≈ 14.75M ≫ 2.7M).

import (
	"fmt"
	"os"
	"runtime/debug"
	"sync"
)

// DistDeviceSymEig routes the -mgpu Rayleigh-Ritz eigensolve to a GPU instead of the host.
//
// distBackend embeds Gonum and, by default, inherits its SymEig — so under -mgpu the O(dim^3)
// projected eigensolve runs on the CPU while all eight GPUs sit idle. At production SIP's Krylov
// width (dim reaches 11,600, MaxBlocks*main = 200*58) that is ~1.9e12 flops on the host, and
// every sub-backend already carries a cuSOLVER SymEig that is never reached.
//
// It is OPT-IN, and deliberately so: cuSOLVER's dsyevd and the host LAPACK path are different
// implementations, so the eigenvalues agree only to rounding, not bit for bit. Flipping it
// silently would move every published line in the last digits with no record of why. Enable it
// with -mgpu-device-symeig when that trade is wanted and the run is re-validated.
//
// The projected matrix is small and replicated, not row-partitioned, so delegating to ONE
// sub-backend is the whole change: no reduction, no cross-device ordering question.
var DistDeviceSymEig bool

// SymEig runs the projected eigensolve on the first partition's device when DistDeviceSymEig is
// set, and otherwise on the host exactly as before (the embedded Gonum). See DistDeviceSymEig.
func (b *distBackend) SymEig(a Mat) ([]float64, Mat) {
	if !DistDeviceSymEig || len(b.subs) == 0 {
		return b.Gonum.SymEig(a)
	}
	return b.subs[0].SymEig(a)
}

// distBackend spreads the row dimension across subs. bound holds the G+1 partition
// boundaries (bound[0]=0, bound[G]=n); device d owns global rows [bound[d], bound[d+1]).
type distBackend struct {
	Gonum // inherit the host SymEig / any method not overridden below
	subs  []Backend
	n     int
	main  int
	bound []int
	// stage holds one reusable host buffer per device, shared by the three methods that move a
	// whole row-partitioned panel between the host and the devices — Download, DownloadInto and
	// AddPanel. Allocated in NewDistributed so no caller ever writes the outer slice; each call
	// grows only the entries it needs, to the size that call needs.
	//
	// Reuse is the point. be.Download hands back a fresh multi-hundred-MB slice per device per
	// call and AddPanel used to make one per device per apply, and Go returns large freed spans
	// to the OS only lazily, so the churn accumulates as RSS until the cgroup OOM-kills the job
	// (the 733 GB kill, jobs 14040959 / 14075367; see BufferedDownloader's doc in backend.go).
	//
	// stageMu serializes those three methods against each other. None of them takes a device
	// parameter — one call sweeps EVERY device and may re-make any stage[d] that is too small —
	// so distinct calls cannot be partitioned by device the way the per-device workers inside a
	// single call can (each of those touches only its own entry, which is what makes those loops
	// concurrent). In practice the callers are single-threaded — the checkpoint writer, and the
	// satellite fallback's Download-then-AddPanel — so the mutex is never contended; it is here
	// so that a future concurrent caller is merely serialized instead of silently overwriting a
	// neighbour's staging.
	stage   [][]float64
	stageMu sync.Mutex

	// band holds one reusable DEVICE scratch per device for the remote-input path of
	// gemmMatOne, which compacts a peer's row band onto the output device before the GEMM.
	//
	// It used to Alloc and Free that band per BLOCK. Both ends are expensive on the hot path:
	// Alloc is a cudaMalloc plus a devZero of the whole band, and cudaFree implicitly
	// synchronizes the device — so every remote-input block paid two whole-device drains (that
	// free, plus the source Sync above it) and an allocator round-trip, serially, on the dense
	// main/coupling path that the production DIP trace (job 14561251) put at ~1h52m of block 0's
	// 2h09m07s apply.
	//
	// Not zeroed on reuse, deliberately: PeerCopy2D writes the full rows×cols compact band
	// (dst pitch = rows, width = rows, height = cols), so every element read by the GEMM is
	// overwritten first. The non-peer host-staging fallback still allocates through Upload —
	// it is the slow path by construction and not worth a second mechanism.
	band   []Vector
	bandN  []int
	bandMu sync.Mutex
}

// ensureBand returns device dev's reusable remote-input scratch, at least n elements long.
// Caller must hold bandMu.
func (b *distBackend) ensureBand(dev, n int) Vector {
	if b.bandN[dev] < n {
		if b.band[dev] != nil {
			b.subs[dev].Free(b.band[dev])
		}
		b.band[dev] = b.subs[dev].Alloc(n)
		b.bandN[dev] = n
	}
	return b.band[dev]
}

// remoteCtx lets one GemmMatBatched call drain a source device once instead of once per
// remote-input block.
//
// The drain exists because a peer read does not synchronize the source stream. Repeating it
// per block is only necessary if something between two reads can dirty that source — and
// inside this call the only writes are the GEMMs, which land on OUTPUT bands. So a source
// device that is NOT also an output device in this call cannot be dirtied here and needs
// exactly one drain; a source that IS also an output device keeps the per-block drain, which
// costs nothing extra versus the old behaviour and needs no assumption about whether the
// caller's input and output panels alias.
type remoteCtx struct {
	synced []bool
	outDev []bool
}

func (r *remoteCtx) needSync(di int) bool {
	if r == nil {
		return true // single GemmMat: no call-level state, always drain
	}
	if r.outDev[di] || !r.synced[di] {
		r.synced[di] = true
		return true
	}
	return false
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
	return &distBackend{
		subs: subs, n: n, main: main,
		bound: append([]int(nil), bounds...),
		stage: make([][]float64, len(subs)),
		band:  make([]Vector, len(subs)),
		bandN: make([]int, len(subs)),
	}, nil
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

// goDevices runs body(d) for every partition concurrently, blocks until all of them have
// stopped, and then re-raises the lowest-index panic on the CALLER's goroutine. Every
// per-device fan-out in this file goes through it.
//
// SCHEDULING is the first reason it exists. Every call into a sub-backend blocks on a
// round-trip through that device's owning goroutine (gpuBackend.do), so a plain
// `for d := range b.ndev()` loop runs one GPU and idles the other seven for the whole
// sweep. That is the "summed per-device time equals wall time, i.e. no overlap at all"
// pathology job 14211868 measured for the mat-vec and dip/matfree_dist.go already fixed
// on its own side.
//
// PANICS are the second, and the reason this is a helper rather than a bare wg.Go loop.
// gpuBackend.do re-raises a device fault — a failed cudaMalloc, or a sticky
// cudaErrorLaunchFailure surfacing at the next checked call, as in the cudaError_t 719
// that killed job 14561251 — as a panic on whichever goroutine called it. Raised on a bare
// goroutine that panic has no path back to the solver: it aborts the process, so a fault
// 30 h into a production mat-vec dies with a goroutine dump and, unless the block happened to
// have reached cp.Every, no checkpoint. Recovering per device and re-raising on the
// caller's goroutine restores the unwind through ApplyBlock -> SolveLowMem, where the
// errInterrupted / checkpoint handling lives.
//
// Every device is allowed to stop before anything is re-raised: a sibling still writing to
// a buffer that an unwinding goroutine is about to free would be a use-after-free. The
// lowest device index wins so a reproducible fault reports reproducibly; the others are
// logged rather than lost.
//
// This is the backend-side twin of dip.goDevices (internal/adc/dip/matfree_dist.go),
// duplicated rather than shared because backend must stay importable on its own.
func goDevices(nd int, body func(d int)) {
	if nd == 1 {
		// One partition: no goroutine and no recover, so a fault keeps its original stack
		// instead of being re-raised with the worker frames already unwound.
		body(0)
		return
	}
	panics := make([]any, nd)
	stacks := make([][]byte, nd)
	var wg sync.WaitGroup
	for d := range nd {
		wg.Go(func() {
			defer func() {
				if r := recover(); r != nil {
					panics[d], stacks[d] = r, debug.Stack()
				}
			}()
			body(d)
		})
	}
	wg.Wait()

	first := -1
	for d := range nd {
		if panics[d] == nil {
			continue
		}
		if first < 0 {
			first = d
			continue
		}
		fmt.Fprintf(os.Stderr, "backend: partition %d also failed: %v\n%s\n", d, panics[d], stacks[d])
	}
	if first >= 0 {
		// Re-raise the original value, not a wrapper: any type-based handling upstream still
		// sees what the device backend raised. The partition and its worker stack — which the
		// re-panic's own stack no longer shows — go to stderr first.
		fmt.Fprintf(os.Stderr, "backend: partition %d failed:\n%s\n", first, stacks[first])
		panic(panics[first])
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

// Alloc fans out over the devices. Worker d allocates only parts[d], on its own sub-backend,
// and writes only its own element of parts — distinct words of a slice, no shared accumulator
// and no arithmetic at all, so this is bit-for-bit identical to the serial form.
//
// Worth doing because each sub-Alloc is a blocking round-trip through that device's owning
// goroutine AND, on the GPU backends, memsets the whole allocation (gpu_device.go Alloc calls
// devZero). A production DIP Krylov panel is ~17 GB per device, so serially this was eight
// sequential multi-GB cudaMalloc+cudaMemset pairs with seven H200s idle, once per panel.
func (b *distBackend) Alloc(length int) Vector {
	nd := b.ndev()
	parts := make([]Vector, nd)
	if cols := b.panelCols(length); cols > 0 {
		goDevices(nd, func(d int) { parts[d] = b.subs[d].Alloc(b.rowsOn(d) * cols) })
		return distVec{b: b, part: parts, grows: b.n, gcols: cols}
	}
	goDevices(nd, func(d int) { parts[d] = b.subs[d].Alloc(length) })
	return distVec{b: b, part: parts, repl: true, grows: 1, gcols: length}
}

// Upload fans out over the devices. Worker d READS only the [bound[d], bound[d+1]) row band of
// each column of host — the bands are disjoint and host is never written — and writes only its
// own staging slice and its own element of parts. Copies and one H2D transfer per device: no
// arithmetic, no shared accumulator, hence bit-for-bit identical to the serial form.
//
// Serially this was ndev sequential blocking H2D transfers of a whole panel band (~17 GB per
// device at production DIP scale) with the rest of the node idle.
//
// The per-device gather buffer stays a fresh make rather than the reusable `stage`: Upload is a
// setup-path call (seeding a panel), not a per-apply one, so holding a second ndev × band-sized
// host allocation resident for the run would cost more than the churn it saves.
func (b *distBackend) Upload(host Vec) Vector {
	nd := b.ndev()
	parts := make([]Vector, nd)
	if cols := b.panelCols(len(host)); cols > 0 {
		goDevices(nd, func(d int) {
			rd := b.rowsOn(d)
			sub := make([]float64, rd*cols)
			// Gather device d's rows out of the global column-major host panel.
			for c := range cols {
				copy(sub[c*rd:(c+1)*rd], host[c*b.n+b.bound[d]:c*b.n+b.bound[d]+rd])
			}
			parts[d] = b.subs[d].Upload(sub)
		})
		return distVec{b: b, part: parts, grows: b.n, gcols: cols}
	}
	goDevices(nd, func(d int) { parts[d] = b.subs[d].Upload(host) })
	return distVec{b: b, part: parts, repl: true, grows: 1, gcols: len(host)}
}

// Download materializes the global column-major panel on the host, fanning out over the
// devices and staging each device's band through the reusable per-device buffer.
//
// Concurrency is bit-for-bit free, for the same reason as DownloadInto below: worker d touches
// only stage[d] and writes only the [bound[d], bound[d+1]) row band of every column of out, and
// the bands are disjoint by construction. Copies only, no arithmetic, no shared accumulator.
// Serially this was ndev sequential blocking D2H transfers of a whole panel band (~17 GB per
// device at production DIP scale), once per satellite apply on the -mgpu fallback path.
//
// The RETURNED panel must stay a fresh allocation — the caller keeps it — but the per-device
// intermediates need not be, and they are the ones that churn: be.Download hands back a new
// multi-hundred-MB slice per device per call, which is the allocation pattern that OOM-killed
// the SIP runs at 733 GB RSS (jobs 14040959 / 14075367).
func (b *distBackend) Download(v Vector) Vec {
	dv := v.(distVec)
	if dv.repl {
		return b.subs[0].Download(dv.part[0])
	}
	cols := dv.gcols
	out := make([]float64, b.n*cols)
	b.stageMu.Lock()
	defer b.stageMu.Unlock()
	goDevices(b.ndev(), func(d int) {
		if dv.part[d] == nil {
			return
		}
		rd := b.rowsOn(d)
		var sub []float64
		if bd, ok := b.subs[d].(BufferedDownloader); ok {
			// Sized to the PART, not to rd*cols: a located row band (RowRange) holds a shorter
			// span than a full panel band, and staging it into a longer buffer would silently
			// read stale tail data where the allocating path slices out of range and panics.
			// Matching Download's own length keeps the two paths byte-for-byte equivalent.
			ln := dv.part[d].Len()
			if len(b.stage[d]) < ln {
				b.stage[d] = make([]float64, ln)
			}
			sub = b.stage[d][:ln]
			bd.DownloadInto(sub, dv.part[d])
		} else {
			sub = b.subs[d].Download(dv.part[d])
		}
		for c := range cols {
			copy(out[c*b.n+b.bound[d]:c*b.n+b.bound[d]+rd], sub[c*rd:(c+1)*rd])
		}
	})
	return out
}

// DownloadInto satisfies BufferedDownloader for the row-partitioned backend: the same global
// column-major panel Download materializes, but scattered into a caller-owned buffer.
//
// It MUST exist, and not merely as an optimization. distBackend embeds Gonum, so without this
// override the embedded Gonum.DownloadInto satisfies the interface — and then does v.(hostVec) on
// a distVec and panics. That is exactly how the production DIP probe (job 14158038) lost 40 h of work
// at its first checkpoint: the `be.(BufferedDownloader)` assertion in saveLowMem succeeded through
// the embedding, so the intended "fall back to the allocating Download path" never happened.
//
// Reusing the per-device staging buffers is the original motivation: be.Download returns a fresh
// slice per call, and a per-block checkpoint of the production system's panel is 274 GB, so the allocating path
// churns large spans in exactly the pattern that OOM-killed the SIP runs at 733 GB.
func (b *distBackend) DownloadInto(dst Vec, v Vector) {
	dv := v.(distVec)
	if dv.repl {
		if bd, ok := b.subs[0].(BufferedDownloader); ok {
			bd.DownloadInto(dst, dv.part[0])
			return
		}
		copy(dst, b.subs[0].Download(dv.part[0]))
		return
	}
	cols := dv.gcols
	if need := b.n * cols; len(dst) < need {
		panic(fmt.Sprintf("backend: DownloadInto dst too small (%d < %d)", len(dst), need))
	}
	// One device per goroutine. Each worker touches only its own stage[d] and writes only the
	// [bound[d], bound[d+1]) row band of every column of dst, and the bands are disjoint by
	// construction — so the workers share nothing. The download itself is a blocking round-trip
	// through that device's owning goroutine, so serially this cost ndev sequential transfers of
	// the whole basis every time the checkpoint writer ran. Copies only, no arithmetic and no
	// shared accumulator, so the fan-out is bit-for-bit free.
	//
	// Through goDevices rather than a bare wg.Go loop so that a device fault here — a failed
	// cudaMalloc growing the staging, a sticky launch failure surfacing at the D2H — unwinds to
	// the checkpoint writer that called this instead of aborting the process.
	b.stageMu.Lock()
	defer b.stageMu.Unlock()
	goDevices(b.ndev(), func(d int) {
		if dv.part[d] == nil {
			return
		}
		rd := b.rowsOn(d)
		var sub []float64
		if bd, ok := b.subs[d].(BufferedDownloader); ok {
			if len(b.stage[d]) < rd*cols {
				b.stage[d] = make([]float64, rd*cols)
			}
			sub = b.stage[d][:rd*cols]
			bd.DownloadInto(sub, dv.part[d])
		} else {
			sub = b.subs[d].Download(dv.part[d])
		}
		for c := range cols {
			copy(dst[c*b.n+b.bound[d]:c*b.n+b.bound[d]+rd], sub[c*rd:(c+1)*rd])
		}
	})
}

// Zero clears every device's band concurrently. Worker d writes only part[d], on its own
// sub-backend: disjoint device memory, no shared accumulator, no arithmetic — bit-for-bit
// identical to the serial form, only the issue order changes.
//
// This is the first thing every ApplyBlock does (dip/matvec.go applyBatches zeroes the output
// panel before accumulating the batches into it), and the panel it clears is ~17 GB per device
// at production DIP scale. Serially that was eight sequential multi-GB memsets — each a blocking
// round-trip through one device's owning goroutine — at the top of every mat-vec.
func (b *distBackend) Zero(v Vector) {
	dv := v.(distVec)
	goDevices(len(dv.part), func(d int) {
		if dv.part[d] != nil {
			b.subs[d].Zero(dv.part[d])
		}
	})
}

// Copy runs one device-to-device copy per partition, concurrently. Worker i reads only
// src.part[i] and writes only dst.part[i], both resident on sub-backend i: disjoint memory, no
// shared accumulator, no arithmetic — bit-for-bit free. Serially this was ndev sequential
// blocking D2D copies of a panel band (~17 GB per device at production DIP scale), which the
// Mode-B recurrence issues several times per Lanczos iteration.
func (b *distBackend) Copy(dst, src Vector) {
	dd, ss := dst.(distVec), src.(distVec)
	goDevices(len(dd.part), func(i int) {
		if dd.part[i] != nil && ss.part[i] != nil {
			b.subs[i].Copy(dd.part[i], ss.part[i])
		}
	})
}

// Free releases every device's band concurrently. Worker d frees only its own allocation on
// its own sub-backend — disjoint memory, no shared accumulator, nothing computed, so this is
// bit-for-bit free as well as race-free.
//
// It matters because cudaFree implicitly synchronizes its device, so each of these is a full
// device drain on top of the round-trip through the owning goroutine, and the Mode-B driver
// frees and reallocates ~17 GB-per-device panels between blocks.
func (b *distBackend) Free(v Vector) {
	dv, ok := v.(distVec)
	if !ok {
		return
	}
	goDevices(len(dv.part), func(d int) {
		if dv.part[d] != nil {
			b.subs[d].Free(dv.part[d])
		}
	})
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
		//
		// Issued concurrently, for the same reason the transfers below are and with a stronger
		// disjointness argument: device d reads only its own row band (panelLocal) and writes
		// only its own copy of the small output. This is the EXPENSIVE half of the reduce —
		// rowsOn(d)×main×main of arithmetic against the main²-sized transfers — so leaving it
		// serial idled ndev-1 devices through the whole contraction on a path the solver takes
		// several times per Lanczos iteration (alpha, Gram, every CGS2 projection). That is the
		// "summed per-device GEMM time equals wall time, i.e. no overlap at all" pathology job
		// 14211868 measured and dip/matfree_dist.go already fixed for the mat-vec.
		//
		// Bit-exact: the per-device partials are independent arithmetic on disjoint operands, so
		// only their issue order changes. The reduction that consumes them stays serial and in
		// ascending device order below.
		//
		// Through goDevices (see its doc) rather than a bare wg.Go loop: a device fault raised
		// inside one of these workers used to abort the process outright, with no path back to
		// the solver's checkpoint handling.
		nd := b.ndev()
		views := make([]BlockView, nd)
		aLocal := make([]BlockView, nd)
		bLocal := make([]BlockView, nd)
		for d := range nd {
			// Resolve on this goroutine: panelLocal panics on a mis-shaped operand, which is a
			// caller bug and should not surface from an anonymous goroutine.
			views[d] = BlockView{V: cdv.part[d], Rows: c.Rows, Cols: c.Cols, Ld: c.Ld}
			aLocal[d], bLocal[d] = b.panelLocal(a, d), b.panelLocal(bb, d)
		}
		goDevices(nd, func(d int) {
			b.subs[d].Gemm(true, transB, alpha, aLocal[d], bLocal[d], 0, views[d])
		})
		// All-reduce: sum the per-device buffers and replicate the total back to every device.
		// Gaps outside the c.Rows×c.Cols result region are never read by the consumer, so
		// summing whole buffers is safe.
		// Both halves of the all-reduce are issued concurrently too: every Download/Upload is a
		// blocking round-trip through that device's owning goroutine, and the devices are
		// independent, so serially this cost 2·ndev synchronous transfers on a path the solver
		// takes several times per Lanczos iteration (alpha, Gram, every CGS2 projection) — not
		// once per mat-vec.
		//
		// The ARITHMETIC stays serial and in ascending device order. Floating-point addition is
		// not associative, so the reduction order is a deliberate choice, not an incidental one:
		// summing as the partials happened to land would make results depend on transfer timing.
		// Only the transfers overlap. Download returns a fresh host copy on every backend, so
		// the partials do not alias device storage.
		parts := make([][]float64, nd)
		goDevices(nd, func(d int) { parts[d] = b.subs[d].Download(cdv.part[d]) })

		acc := parts[0]
		for d := 1; d < nd; d++ {
			for i := range acc {
				acc[i] += parts[d][i]
			}
		}

		// Replicate the total back. acc is read-only here, shared across the workers.
		goDevices(nd, func(d int) {
			up := b.subs[d].Upload(acc)
			b.subs[d].Copy(cdv.part[d], up)
			b.subs[d].Free(up)
		})
		return
	}
	// The local panel update: device d reads its own row band of a, its own copy of the
	// replicated small factor, and writes its own row band of c — wholly independent, no
	// communication. Concurrent for the same reason as the reduce above; each sub-Gemm is a
	// blocking round-trip through one device's owning goroutine, so a serial loop is ndev
	// sequential GEMMs where the hardware can run them at once. Operands resolved here rather
	// than in the workers so a mis-shaped one panics on the caller's goroutine, and issued
	// through goDevices so a device fault inside a worker unwinds to the caller too.
	nd := b.ndev()
	aLocal := make([]BlockView, nd)
	bLocal := make([]BlockView, nd)
	cLocal := make([]BlockView, nd)
	for d := range nd {
		aLocal[d], bLocal[d], cLocal[d] = b.panelLocal(a, d), b.smallLocal(bb, d), b.panelLocal(c, d)
	}
	goDevices(nd, func(d int) {
		b.subs[d].Gemm(false, transB, alpha, aLocal[d], bLocal[d], beta, cLocal[d])
	})
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
	// Concurrent over devices, and staged through the reusable per-device buffers.
	//
	// Bit-for-bit free: worker d READS only the [bound[d], bound[d+1]) row band of every column
	// of full (disjoint bands, full is never written), writes only stage[d], and accumulates
	// only into part[d] on its own sub-backend. There is no shared accumulator anywhere, so each
	// Axpy sums exactly the same values in exactly the same order however the workers interleave.
	//
	// Both halves earn their place on this path — the scatter half of the -mgpu satellite
	// fallback, run once per apply. Serially it was ndev sequential blocking H2D uploads of a
	// full panel band, each preceded by a fresh host make of rd·cols float64 (~17 GB per device
	// at production DIP scale). That make is exactly the allocation churn BufferedDownloader exists
	// to remove: Go returns large freed spans to the OS only lazily, and the accumulated RSS is
	// what the cgroup OOM-killed at 733 GB (jobs 14040959 / 14075367). Reusing `stage` costs the
	// same peak and none of the churn — and it is safe to hand a reused buffer to Upload because
	// every backend's Upload copies it (Gonum.Upload makes and copies; the GPU backends
	// cudaMemcpy H2D), so no sub-backend retains a reference past the call.
	b.stageMu.Lock()
	defer b.stageMu.Unlock()
	goDevices(b.ndev(), func(d int) {
		if dv.part[d] == nil {
			return
		}
		rd := b.rowsOn(d)
		if len(b.stage[d]) < rd*cols {
			b.stage[d] = make([]float64, rd*cols)
		}
		band := b.stage[d][:rd*cols]
		for c := range cols {
			copy(band[c*rd:(c+1)*rd], full[c*b.n+b.bound[d]:c*b.n+b.bound[d]+rd])
		}
		up := b.subs[d].Upload(band)
		// The panel is column-major with leading dimension rowsOn(d), and may have more columns
		// allocated than this apply uses (the solver sizes panels to the max block width), so add
		// only into the first cols columns — their rd·cols storage is contiguous at the front.
		b.subs[d].Axpy(1, up, dv.part[d].Slice(0, rd*cols))
		b.subs[d].Free(up)
	})
}

// --- per-device apply capability (PartitionedDevices) ------------------------

// The distributed backend's sub-backends, partition boundaries and peer capability are
// unexported, so an operator block that wants to run per-device (each device recomputing only
// its own output band on-device, rather than AddPanel's gather-to-host) has no way to reach
// them. PartitionedDevices is that handle; see backend.go for the contract.

func (b *distBackend) NumParts() int { return b.ndev() }

// Bounds returns a COPY: callers derive row bands from it, and must not be able to mutate the
// partitioning the resident panels were allocated against.
func (b *distBackend) Bounds() []int { return append([]int(nil), b.bound...) }

func (b *distBackend) PartBackend(d int) Backend { return b.subs[d] }

func (b *distBackend) PartKernels(d int) (DeviceKernels, bool) {
	dk, ok := b.subs[d].(DeviceKernels)
	return dk, ok
}

// PartVector returns device d's storage for a row-partitioned panel. It rejects a replicated
// small buffer and a located row band (a RowRange view): both are shapes the per-device apply
// must never be handed, and silently accepting one would apply the operator to the wrong rows.
func (b *distBackend) PartVector(v Vector, d int) Vector {
	dv, ok := v.(distVec)
	if !ok {
		panic("distributed PartVector: not a distributed vector")
	}
	if dv.repl {
		panic("distributed PartVector: expected a row-partitioned panel, got a replicated buffer")
	}
	if dv.loc != nil {
		panic("distributed PartVector: expected a full panel, got a located row band")
	}
	return dv.part[d]
}

// AllPeered reports whether every ordered pair of distinct sub-backends can peer-copy. A single
// partition is trivially peered (no cross-device traffic at all); any non-PeerCopier sub-backend
// (Gonum) makes it false, leaving the host-staging fallback in place.
func (b *distBackend) AllPeered() bool {
	for i, s := range b.subs {
		pc, ok := s.(PeerCopier)
		if !ok {
			return false
		}
		for j, o := range b.subs {
			if i != j && !pc.PeerAvailable(o) {
				return false
			}
		}
	}
	return true
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
func (b *distBackend) gemmMatOne(transA bool, alpha float64, a DeviceMat, bb, c BlockView, beta float64, rc *remoteCtx) {
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
	reused := false
	if pc, ok := b.subs[do].(PeerCopier); ok && pc.PeerAvailable(b.subs[di]) {
		// Drain the source device: a peer read does not synchronize the source stream, and the
		// band may still be mid-write from an async panel kernel (unlike the host path below,
		// whose Download drains it implicitly). subs[di] is a *gpuBackend here (PeerAvailable
		// proved it), so it implements PeerCopier.Sync. remoteCtx collapses this to once per
		// source device per batched call where that is provably sufficient — see needSync.
		if rc.needSync(di) {
			b.subs[di].(PeerCopier).Sync()
		}
		b.bandMu.Lock()
		band = b.ensureBand(do, rows*cols)
		pc.PeerCopy2D(band, bdv.part[di], b.subs[di], rows, cols, rows, ld) // compact dst
		reused = true
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
	if reused {
		// The GEMM has consumed the band by the time GemmMat returns (every sub-backend call is
		// a blocking round-trip through that device's owning goroutine), so the scratch is free
		// to be overwritten by the next remote block. It is NOT freed: that cudaFree was one of
		// the two whole-device drains this path used to pay per block.
		b.bandMu.Unlock()
	} else {
		b.subs[do].Free(band)
	}
}

func (b *distBackend) GemmMat(transA bool, alpha float64, a DeviceMat, bb BlockView, beta float64, c BlockView) {
	b.gemmMatOne(transA, alpha, a, bb, c, beta, nil)
}

// GemmMatBatched groups the batch by the device owning each output band and issues one real
// batched GEMM per device, instead of one cuBLAS call per block.
//
// Why regrouping is numerically free: gpuBackend.GemmMatBatched's contract (gpu_device.go)
// already requires batch members to have pairwise non-overlapping outputs and uniform shapes —
// they execute concurrently and cannot interact. Partitioning an independent set therefore
// changes no arithmetic, only the launch count. The previous form unpacked the batch into
// len(a) sequential gemmMatOne calls, reintroducing exactly the dispatch tax batching exists to
// remove (measured there at 181 s of a 379 s formic-acid sector).
//
// Blocks whose INPUT band is remote still go one at a time through gemmMatOne: each needs its
// own gathered, compacted band (Ld = rows), which would break the uniform-Ld requirement if
// mixed into a batch of local bands. Batching those too means sub-bucketing by leading
// dimension and holding every gathered band alive across the call — worthwhile only if a
// profile shows remote-input blocks dominating, which row-partitioning is chosen to avoid.
func (b *distBackend) GemmMatBatched(transA bool, alpha float64, a []DeviceMat, bb []BlockView, beta float64, c []BlockView) {
	if len(a) == 0 {
		return
	}
	nd := b.ndev()
	// Per-device local-input members, in their original batch order.
	la := make([][]DeviceMat, nd)
	lb := make([][]BlockView, nd)
	lc := make([][]BlockView, nd)

	// Which devices receive output here. A remote block's SOURCE device only needs re-draining
	// between reads if something can dirty it, and inside this call the only writes are the
	// GEMMs, which land on output bands — so a source that is not also an output device is
	// drained once instead of once per block. See remoteCtx.needSync.
	rc := &remoteCtx{synced: make([]bool, nd), outDev: make([]bool, nd)}
	for i := range c {
		if cdv, ok := c[i].V.(distVec); ok && cdv.loc != nil {
			rc.outDev[cdv.loc.dev] = true
		}
	}

	for i := range a {
		cdv := c[i].V.(distVec)
		if cdv.loc == nil {
			panic("distributed GemmMatBatched: output is not a located row band")
		}
		bdv := bb[i].V.(distVec)
		if bdv.loc == nil {
			panic("distributed GemmMatBatched: input is not a located row band")
		}
		do := cdv.loc.dev
		if bdv.loc.dev != do {
			b.gemmMatOne(transA, alpha, a[i], bb[i], c[i], beta, rc) // remote input: see above
			continue
		}
		la[do] = append(la[do], a[i].(*distMat).on(do))
		lb[do] = append(lb[do], BlockView{V: bdv.part[do], Rows: bdv.loc.rows, Cols: bb[i].Cols, Ld: bdv.loc.ld})
		lc[do] = append(lc[do], BlockView{V: cdv.part[do], Rows: cdv.loc.rows, Cols: c[i].Cols, Ld: cdv.loc.ld})
	}

	// Issue the per-device batches CONCURRENTLY. This is the hottest fan-out in the file: one
	// call per batch per mat-vec (dip/matvec.go applyBatches), thousands of batches for a
	// production DIP sector, and every sub-call is a blocking round-trip through that device's
	// owning goroutine — so a serial loop ran one H200 at a time and idled the other seven for
	// the whole dense main/coupling apply. The production DIP trace (job 14561251) put block 0's
	// total apply at 2h09m07s with only 16m53s of it in the satellite phase; the remaining
	// ~1h52m is the dense path that runs through exactly this loop.
	//
	// Bit-for-bit free, for the same reason the regrouping above is: GemmMatBatched's contract
	// requires a batch's members to have pairwise non-overlapping outputs (PlanBatches
	// establishes it — batches are formed per shape, taking at most one block per distinct write
	// offset), and the bucketing above only ever sends a member to the device that OWNS its
	// output band. So device d reads input bands it holds and accumulates into rows no other
	// device touches: there is no shared accumulator, and no output element's summation order
	// changes. Only the issue order does.
	goDevices(nd, func(d int) {
		if len(la[d]) == 0 {
			return
		}
		b.subs[d].GemmMatBatched(transA, alpha, la[d], lb[d], beta, lc[d])
	})
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

var (
	_ Backend            = (*distBackend)(nil)
	_ PanelScatterAdd    = (*distBackend)(nil)
	_ PartitionedDevices = (*distBackend)(nil)
)
