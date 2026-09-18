package sip

import (
	"math"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// matfree22.go — matrix-free application of the two ADC(2,2) blocks that cannot be
// stored: the 2h1p×3h2p coupling (A15-A17, elements22.go c23_1) and the 3h2p×3h2p
// block (A18 plus A19-A22, c33_01).
//
// At ADC(2,2) it is these, not the 2h1p block, that set the memory ceiling, and by
// a wide margin: the 3h2p class is ~n_occ³·n_virt²·5 configurations before
// symmetry. Water in a reduced basis already reaches n3 = 4209 in one irrep (a
// 142 MB dense block); n3 grows as the fifth power of the system size and the block
// as the tenth, so anything past a small molecule has to recompute. ADC(2,2)_m is
// exempt — its 3h2p/3h2p block is diagonal and rides as a diagPart — but its
// coupling block is not.
//
// Both appliers are host-only. Unlike the order-3 satellite and the order-4 WERT2
// coupling there is no CUDA kernel to dispatch to, because the element is not a
// fixed set of ERI lookups: it is a contraction of hamNO over two determinant
// expansions (slater.go, spinadapt.go), which is a port in its own right rather
// than a transcription. A device backend therefore falls back to the dense block —
// the same position matFreeC22 is in, for the same reason.
//
// PRUNING IS NOT AN OPTIMIZATION HERE. hamNO vanishes unless the two determinants
// differ by at most two spin orbitals, so both blocks are extremely sparse, and an
// applier that swept every column for every row would do n3² element evaluations
// per mat-vec — each a ≤5×5 contraction over determinant pairs. That is hours of
// legitimate work per Lanczos block with nothing logged inside it, i.e.
// indistinguishable from a hang; it is exactly the failure the order-3 satellite
// applier hit before buildC22Buckets was added. The selection rules below are
// derived from the excitation-difference count, so they are supersets of the true
// nonzero pattern: a false positive costs one evaluation that returns 0, and the
// appliers already skip g == 0, so walking a superset in ascending index order
// replays the dense block's accumulation sequence unchanged.

// ---------------------------------------------------------------------------
// Bucket index and ascending merge.
// ---------------------------------------------------------------------------

// bucketSet maps an integer key to an ascending list of configuration indices,
// with every list packed into one flat backing array.
//
// The flat array is the point: a production 3h2p space has ~10⁶ distinct keys, and
// one append-grown slice per key would cost more in slice headers and heap
// fragmentation than the index saves. Keys are int64 because the encodings below
// are mixed-radix products of n_occ and n_virt, which overflow int32 well before
// the configuration space does.
type bucketSet struct {
	span  map[int64]int64 // key -> start<<32 | end, both offsets into items
	items []int32
}

// newBucketSet indexes n configurations by the keys keysOf emits for each of them.
// Two passes — count, then fill — so items is allocated exactly once at its final
// size. Because the fill walks i ascending, every list comes out sorted, which is
// what lets the appliers merge several of them into one ascending candidate sweep.
//
// The offset assignment iterates a map, so which range a key gets is not
// reproducible run to run. That is harmless: it changes neither the contents of
// any list nor their order, and the appliers only ever consume merged, ascending
// candidate indices.
func newBucketSet(n int, keysOf func(i int, emit func(int64))) *bucketSet {
	cnt := make(map[int64]int32)
	total := 0
	for i := 0; i < n; i++ {
		keysOf(i, func(k int64) {
			cnt[k]++
			total++
		})
	}
	b := &bucketSet{span: make(map[int64]int64, len(cnt)), items: make([]int32, total)}
	var off int64
	for k, c := range cnt {
		b.span[k] = off<<32 | off // start, with the fill cursor initialized to it
		off += int64(c)
	}
	for i := 0; i < n; i++ {
		keysOf(i, func(k int64) {
			s := b.span[k]
			cur := int64(uint32(s))
			b.items[cur] = int32(i)
			b.span[k] = s&^0xffffffff | (cur + 1)
		})
	}
	return b
}

// get returns the ascending list of configurations registered under key, or nil.
func (b *bucketSet) get(key int64) []int32 {
	s, ok := b.span[key]
	if !ok {
		return nil
	}
	return b.items[int64(uint32(s>>32)):int64(uint32(s))]
}

// ascMerge merges a handful of ascending int32 lists into one ascending,
// duplicate-free sweep. It carries its own scratch so a per-row or per-column
// candidate list costs no allocation after the first call.
type ascMerge struct {
	lists [][]int32
	pos   []int
	out   []int32
}

// reset starts a new candidate set.
func (m *ascMerge) reset() { m.lists = m.lists[:0] }

// add contributes one ascending list. Empty lists and repeats of the same list are
// both fine — merge dedups by value.
func (m *ascMerge) add(l []int32) {
	if len(l) > 0 {
		m.lists = append(m.lists, l)
	}
}

// merge returns the union of the added lists, ascending and deduped. The returned
// slice is valid until the next merge on this ascMerge.
func (m *ascMerge) merge() []int32 {
	m.out = m.out[:0]
	if cap(m.pos) < len(m.lists) {
		m.pos = make([]int, len(m.lists))
	} else {
		m.pos = m.pos[:len(m.lists)]
		clear(m.pos)
	}
	last := int32(-1)
	for {
		var v int32
		found := false
		for i, l := range m.lists {
			if p := m.pos[i]; p < len(l) && (!found || l[p] < v) {
				v, found = l[p], true
			}
		}
		if !found {
			return m.out
		}
		for i, l := range m.lists {
			for m.pos[i] < len(l) && l[m.pos[i]] == v {
				m.pos[i]++
			}
		}
		if v != last {
			m.out = append(m.out, v)
			last = v
		}
	}
}

// sort3 orders three small ints ascending, without touching the heap.
func sort3(a, b, c int) (int, int, int) {
	if a > b {
		a, b = b, a
	}
	if b > c {
		b, c = c, b
	}
	if a > b {
		a, b = b, a
	}
	return a, b, c
}

// ---------------------------------------------------------------------------
// 2h1p × 3h2p (A15-A17).
// ---------------------------------------------------------------------------

// c23Index prunes the candidate 2h1p rows of a 3h2p column.
//
// Selection rule. Both determinants hold the same number of electrons, so the
// element vanishes unless the count of spin orbitals occupied in one and not the
// other is at most 2. For a 2h1p row (holes k, l; particle a) against a 3h2p
// column (holes K, L, M; particles I, J) that count is
//
//	|{K,L,M} \ {k,l}| + [a not in {I,J}],
//
// and three holes can never all be matched by two, so the first term is at least 1.
// Hence the element can be nonzero only when
//
//	both of the row's holes are among the column's, or
//	exactly one is and the row's particle is one of the column's.
//
// Read over spatial orbitals that gives two families of candidates: rows whose hole
// pair is one of the column's three hole pairs, and rows carrying one of the
// column's particles together with one of its holes. Per column that is
// ~6·n_virt + 12·n_occ rows out of n2 ≈ n_occ²·n_virt, so the visited fraction goes
// as 6/n_occ² + 12/(n_occ·n_virt) — ~300x at the production sector's
// n_occ = 58, n_virt = 154, and only ~4x for water in a small basis.
type c23Index struct {
	nocc      int
	byPair    *bucketSet // sorted hole pair (k ≤ l)  -> 2h1p rows
	byVirHole *bucketSet // (particle, hole)           -> 2h1p rows
}

func buildC23Index(rows []Config, nocc int) *c23Index {
	ix := &c23Index{nocc: nocc}
	no := int64(nocc)
	ix.byPair = newBucketSet(len(rows), func(i int, emit func(int64)) {
		k, l := rows[i].Occ[0], rows[i].Occ[1]
		if k > l {
			k, l = l, k
		}
		emit(int64(k)*no + int64(l))
	})
	ix.byVirHole = newBucketSet(len(rows), func(i int, emit func(int64)) {
		c := rows[i]
		emit(int64(c.Vir)*no + int64(c.Occ[0]))
		if c.Occ[1] != c.Occ[0] {
			emit(int64(c.Vir)*no + int64(c.Occ[1]))
		}
	})
	return ix
}

// candidates gathers, ascending, the 2h1p rows that can couple to 3h2p column col.
func (ix *c23Index) candidates(m *ascMerge, col Config3) []int32 {
	h0, h1, h2 := sort3(col.Core, col.L, col.M)
	no := int64(ix.nocc)
	m.reset()
	for _, pr := range [3][2]int{{h0, h1}, {h0, h2}, {h1, h2}} {
		m.add(ix.byPair.get(int64(pr[0])*no + int64(pr[1])))
	}
	for _, v := range [2]int{col.I, col.J} {
		base := int64(v) * no
		m.add(ix.byVirHole.get(base + int64(h0)))
		if h1 != h0 {
			m.add(ix.byVirHole.get(base + int64(h1)))
		}
		if h2 != h1 && h2 != h0 {
			m.add(ix.byVirHole.get(base + int64(h2)))
		}
		if col.J == col.I {
			break
		}
	}
	return m.merge()
}

// newC23MatFree builds the matrix-free applier for the 2h1p×3h2p coupling — the
// on-the-fly equivalent of coupling23 placed at (BeginSat, Begin3h2p). The apply is
// fused over both directions (forward into the 2h1p band, transpose into the 3h2p
// band) and parallelized over the 3h2p columns: that is the large, contention-free
// side, and each worker owns a disjoint span of the 3h2p output.
func (mx *Matrix) newC23MatFree() matFreePart {
	sp := mx.sp
	main, off3 := sp.BeginSat, sp.Begin3h2p
	rows := sp.Configs[main:off3]
	cols := sp.Sat3
	n2 := len(rows)
	el := mx.el
	hd := mx.be.(backend.HostData)
	ix := buildC23Index(rows, sp.Nocc)

	// W and the per-worker forward accumulators are latched HERE, not derived per
	// apply, for the reasons newWert2MatFree spells out: Go re-reads GOMAXPROCS from
	// the cgroup limit while the process runs, and a W that moved mid-solve would
	// regroup this reduction and silently change the operator's rounding underneath
	// Lanczos' short recurrence. Sizing partials once also keeps a GB-scale buffer
	// out of the allocator on every mat-vec.
	W := parallel.ChunkWorkers(len(cols))
	var partials []float64

	apply := func(in, out backend.BlockView) {
		n3 := len(cols)
		if n2 == 0 || n3 == 0 {
			return
		}
		xin := hd.HostSlice(in.V)
		yout := hd.HostSlice(out.V)
		b := in.Cols
		ldi, ldo := in.Ld, out.Ld

		// b is the Krylov block width and may shrink on the last block, so size to
		// the request and keep the larger buffer. ApplyBlock drives one Matrix
		// serially, so this needs no locking.
		if need := W * n2 * b; cap(partials) < need {
			partials = make([]float64, need)
		} else {
			partials = partials[:need]
			clear(partials)
		}

		parallel.Chunks(n3, W, func(w, c0, c1 int) {
			y2 := partials[w*n2*b : (w+1)*n2*b]
			var mrg ascMerge
			for c := c0; c < c1; c++ {
				col := cols[c]
				colBase := off3 + c
				for _, r32 := range ix.candidates(&mrg, col) {
					r := int(r32)
					g := el.c23_1(rows[r], col)
					if g == 0 {
						continue
					}
					rowBase := main + r
					for j := 0; j < b; j++ {
						y2[r+j*n2] += g * xin[colBase+j*ldi]          // forward:   y2 += G·x3
						yout[colBase+j*ldo] += g * xin[rowBase+j*ldi] // transpose: y3 += Gᵀ·x2
					}
				}
			}
		})

		// Reduce the per-worker forward partials into the 2h1p band, fixed order.
		for w := 0; w < W; w++ {
			y2 := partials[w*n2*b : (w+1)*n2*b]
			for j := 0; j < b; j++ {
				base := main + j*ldo
				col := j * n2
				for r := 0; r < n2; r++ {
					yout[base+r] += y2[col+r]
				}
			}
		}
	}
	return matFreePart{apply: apply, release: func() {}}
}

// ---------------------------------------------------------------------------
// 3h2p × 3h2p (A18 plus A19-A22).
// ---------------------------------------------------------------------------

// sat3Index prunes the candidate columns of a 3h2p row.
//
// Selection rule. Both determinants carry three holes and two particles, so with h
// holes and p particles shared (as spatial-orbital multisets) the
// excitation-difference count is (3−h) + (2−p), and the element vanishes unless
//
//	h + p >= 3.
//
// Since h ≤ 3 and p ≤ 2, every admissible case is covered by exactly three
// families, which are the three bucket indices below:
//
//	h = 3            -> equal hole triples                      (byTriple)
//	h >= 2, p >= 1   -> a shared hole pair and a shared particle (byPairVir)
//	p = 2, h >= 1    -> equal particle pairs and a shared hole    (byVirPairHole)
//
// (h,p) = (3,0..2) is the first, (2,1) and (2,2) the second, (1,2) the third; every
// other combination sums below 3. The relations are symmetric in row and column, so
// the swept pattern is symmetric — which is what lets the symmetric diagonal block
// be applied in a single row sweep with no transpose leg.
//
// How much this prunes. The three families hold ~n_virt²/2, ~6·n_occ·n_virt and
// ~1.5·n_occ² columns respectively, against n3 ≈ n_occ³n_virt²/12, so the visited
// fraction of the block goes as
//
//	6/n_occ³ + 72/(n_occ²·n_virt) + 18/(n_occ·n_virt²),
//
// which TestSat3PruneScalesWithNocc confirms to within ~30% over a grid of synthetic
// spaces. Note which term dominates: it is the MIDDLE one, so the prune is
// essentially n_occ²·n_virt and not the n_occ³ a glance at the hole triples
// suggests. At the production sector's n_occ = 58, n_virt = 154 that is ~1.3e-4, a
// ~8000x prune; for water in a small basis it is only ~5x, which is why the
// scaling has its own test rather than being read off the fixture.
type sat3Index struct {
	nocc, nvir    int
	byTriple      *bucketSet // sorted hole triple                  -> 3h2p configs
	byPairVir     *bucketSet // (sorted hole pair, particle)        -> 3h2p configs
	byVirPairHole *bucketSet // (sorted particle pair, hole)        -> 3h2p configs
}

// holesOf and partsOf give a 3h2p configuration's holes and particles in ascending
// order. Neither is sorted as enumerated: addSat3h2p22 orders indices only within
// one irrep block, so K ≤ L ≤ M holds per irrep and not globally.
func holesOf(c Config3) (int, int, int) { return sort3(c.Core, c.L, c.M) }

func partsOf(c Config3) (int, int) {
	if c.I > c.J {
		return c.J, c.I
	}
	return c.I, c.J
}

func buildSat3Index(cfgs []Config3, nocc, nvir int) *sat3Index {
	ix := &sat3Index{nocc: nocc, nvir: nvir}
	no, nv := int64(nocc), int64(nvir)

	ix.byTriple = newBucketSet(len(cfgs), func(i int, emit func(int64)) {
		h0, h1, h2 := holesOf(cfgs[i])
		emit((int64(h0)*no+int64(h1))*no + int64(h2))
	})
	// A coincident hole pair or a doubly occupied particle orbital makes several of
	// the key expressions collide, so both of the multi-key indices dedup locally.
	// A duplicate would only waste an items slot (the merge dedups by value), but
	// the slot count is what sizes the index.
	ix.byPairVir = newBucketSet(len(cfgs), func(i int, emit func(int64)) {
		h0, h1, h2 := holesOf(cfgs[i])
		p0, p1 := partsOf(cfgs[i])
		var seen [6]int64
		n := 0
		for _, pr := range [3][2]int{{h0, h1}, {h0, h2}, {h1, h2}} {
			base := (int64(pr[0])*no + int64(pr[1])) * nv
			for _, v := range [2]int{p0, p1} {
				k := base + int64(v)
				dup := false
				for s := 0; s < n; s++ {
					if seen[s] == k {
						dup = true
						break
					}
				}
				if dup {
					continue
				}
				seen[n] = k
				n++
				emit(k)
			}
		}
	})
	ix.byVirPairHole = newBucketSet(len(cfgs), func(i int, emit func(int64)) {
		h0, h1, h2 := holesOf(cfgs[i])
		p0, p1 := partsOf(cfgs[i])
		base := (int64(p0)*nv + int64(p1)) * no
		emit(base + int64(h0))
		if h1 != h0 {
			emit(base + int64(h1))
		}
		if h2 != h1 && h2 != h0 {
			emit(base + int64(h2))
		}
	})
	return ix
}

// candidates gathers, ascending, the 3h2p columns that can couple to cfg.
func (ix *sat3Index) candidates(m *ascMerge, cfg Config3) []int32 {
	h0, h1, h2 := holesOf(cfg)
	p0, p1 := partsOf(cfg)
	no, nv := int64(ix.nocc), int64(ix.nvir)
	m.reset()
	m.add(ix.byTriple.get((int64(h0)*no+int64(h1))*no + int64(h2)))
	for _, pr := range [3][2]int{{h0, h1}, {h0, h2}, {h1, h2}} {
		base := (int64(pr[0])*no + int64(pr[1])) * nv
		m.add(ix.byPairVir.get(base + int64(p0)))
		if p1 != p0 {
			m.add(ix.byPairVir.get(base + int64(p1)))
		}
	}
	base := (int64(p0)*nv + int64(p1)) * no
	m.add(ix.byVirPairHole.get(base + int64(h0)))
	if h1 != h0 {
		m.add(ix.byVirPairHole.get(base + int64(h1)))
	}
	if h2 != h1 && h2 != h0 {
		m.add(ix.byVirPairHole.get(base + int64(h2)))
	}
	return m.merge()
}

// newSat3MatFree builds the matrix-free applier for the 3h2p×3h2p block — the
// on-the-fly equivalent of sat3Block22 at (Begin3h2p, Begin3h2p). The block is
// symmetric and sits on the operator diagonal, so it is applied in one pass with no
// transpose leg, parallelized over rows: each row owns its output cell, so there is
// no reduction and nothing to latch.
//
// The element is evaluated as c33_01(row, col) with the sweep's row first, matching
// the argument order the dense block fills with.
func (mx *Matrix) newSat3MatFree() matFreePart {
	sp := mx.sp
	off3 := sp.Begin3h2p
	cfgs := sp.Sat3
	n3 := len(cfgs)
	el := mx.el
	hd := mx.be.(backend.HostData)
	ix := buildSat3Index(cfgs, sp.Nocc, sp.Nvir)

	apply := func(in, out backend.BlockView) {
		if n3 == 0 {
			return
		}
		xin := hd.HostSlice(in.V)
		yout := hd.HostSlice(out.V)
		b := in.Cols
		ldi, ldo := in.Ld, out.Ld
		// Chunks over an explicit worker count, not Rows: Rows falls back to serial
		// below 2*GOMAXPROCS rows, and a small sector's n3 lands in exactly that band.
		parallel.Chunks(n3, parallel.ChunkWorkers(n3), func(_, lo, hi int) {
			var mrg ascMerge
			for r := lo; r < hi; r++ {
				row := cfgs[r]
				rowBase := off3 + r
				for _, c32 := range ix.candidates(&mrg, row) {
					g := el.c33_01(row, cfgs[c32])
					if g == 0 {
						continue
					}
					colBase := off3 + int(c32)
					for j := 0; j < b; j++ {
						yout[rowBase+j*ldo] += g * xin[colBase+j*ldi]
					}
				}
			}
		})
	}
	return matFreePart{apply: apply, release: func() {}}
}

// ---------------------------------------------------------------------------
// Budget decisions.
// ---------------------------------------------------------------------------

// matFreeC23 decides the ADC(2,2) 2h1p×3h2p coupling and matFreeSat3 the 3h2p×3h2p
// block. Both are host-only: there is no device kernel for c23_1/c33_01, so a CUDA
// backend must fall back to the dense block.
//
// The DeviceKernels test MUST come first, for the reason matFreeC22 records — a GPU
// backend embeds Gonum and so satisfies HostData through the promoted HostSlice,
// even though its vectors are device memory, so testing HostData alone would select
// the host applier and panic on the first HostSlice.
func (mx *Matrix) matFreeC23(denseBytes int64) bool { return mx.matFreeHostOnly(denseBytes) }

func (mx *Matrix) matFreeSat3(denseBytes int64) bool { return mx.matFreeHostOnly(denseBytes) }

func (mx *Matrix) matFreeHostOnly(denseBytes int64) bool {
	if _, dev := mx.be.(backend.DeviceKernels); dev {
		return false
	}
	_, host := mx.be.(backend.HostData)
	return mx.matFreeDecision(denseBytes, host)
}

// blockBytes is the dense byte size of an r×c float64 block, saturating rather than
// wrapping: a 3h2p block can overflow int64 at sizes the space itself still admits,
// and a wrapped negative would read as "fits comfortably".
func blockBytes(r, c int) int64 {
	const w = 8
	if r <= 0 || c <= 0 {
		return 0
	}
	if int64(r) > math.MaxInt64/(int64(c)*w) {
		return math.MaxInt64
	}
	return int64(r) * int64(c) * w
}
