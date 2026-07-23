package dip

import (
	"unsafe"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// matfree_batched.go — the jiiLKK half of the matrix-free satellite apply, routed through
// batched GEMM instead of the hand-written gemvForward/gemvTranspose loops.
//
// WHY. The satellite apply is ~96.6% of DIP solver wall time and runs at 0.58% of fp64 peak
// (job 14015067). The reason is visible in the code, not just the profile: gemvForward
// (matfree.go) is a hand-rolled triple loop, even though `Backend.GemmMat`/`GemmMatBatched`
// already exist and route to OpenBLAS/cuBLAS. Measured on this machine
// (BenchmarkBlockApplyCrossover), BLAS beats that loop at EVERY block size — 1.10× at dim=8,
// 3.84× at dim=154 (melanin's sizeVirGroup), 8.12× at 256 — and BenchmarkBlockBuildVsApply shows
// the apply is 79% of matrix-free cost at b=64, rising to ~99% at melanin's b=1653. So this
// targets the dominant half and Amdahl does not cap it. See docs/sigma_build_contractions.md.
//
// THE STRUCTURE THAT MAKES IT POSSIBLE. Which blocks exist, and their shapes, depend only on the
// Space and the virtual-symmetry groups — never on the input panel (blocks.go jiiLKKGate is a
// pure function of two Configs). Only the VALUES need recomputing per mat-vec, which is the whole
// point of matrix-free: storing them is what does not fit (docs/dip_operator_memory.md). So the
// plan is built once and reused, and each apply only refills values into it.
//
// This reuses backend.PlanBatches unchanged — the same machinery the dense path uses
// (matvec.go newAssembledOp/applyBatches). PlanBatches guarantees that within one batch no two
// blocks share a write offset, which is what makes a concurrent batched GEMM safe. It also
// subsumes the manual forward/transpose split: a jiiLKK block with gr==gc is on the block
// diagonal (Diag, applied once), gr>gc is off-diagonal (applied as A and Aᵀ), exactly as
// backend.Block already models.
//
// NOT WIRED IN YET. newSatelliteMatFreePart still selects the loop-based applier; this path is
// exercised by tests only until its parity and speedup are confirmed. That staging is deliberate
// (plan step 2 before step 3).

// jiiShapeMat is a shape-only DeviceMat stand-in for planning. PlanBatches reads nothing but
// Dims(), so the plan can be built before any block values exist — which is exactly the
// separation this design depends on.
type jiiShapeMat struct{ rows, cols int }

func (m jiiShapeMat) Dims() (int, int) { return m.rows, m.cols }

// satKind names which satellite block a slot holds. The three have different row/col families
// and different diagonal behaviour, but identical batching structure — their dims already fold in
// the spin-parts factor (blocks.go ijkMLLShape/ijkLMNShape), so a block is just a dense Mat.
type satKind int8

const (
	kindJII    satKind = iota // jiiLKK: JII×JII, diagonal when gr==gc
	kindIJKMLL                // ijkMLL: IJK×JII, never diagonal (different config families)
	kindIJKLMN                // ijkLMN: IJK×IJK, diagonal when gr==gc
)

// jiiSlot is one planned satellite block: the config pair whose values must be recomputed each
// apply, and where the block lands in the operator.
type jiiSlot struct {
	kind           satKind
	rowCfg, colCfg Config
	rowOff, colOff int
	rows, cols     int
	diag           bool
}

// build recomputes this slot's block values. The builders are unchanged — they already fold in
// the dSym&&ra==sb diagonal term and the spin-part structure, so there is no split to handle here.
func (mx *Matrix) buildSlot(s jiiSlot) (backend.Mat, bool) {
	switch s.kind {
	case kindJII:
		return mx.blk.jiiLKK(s.rowCfg, s.colCfg)
	case kindIJKMLL:
		return mx.blk.ijkMLL(s.rowCfg, s.colCfg)
	default:
		return mx.blk.ijkLMN(s.rowCfg, s.colCfg)
	}
}

// jiiBatchPlan is the reusable plan plus the per-apply scratch the batched calls need.
type jiiBatchPlan struct {
	slots   []jiiSlot
	batches []backend.Batch

	// Scratch, sized to the widest batch and reused across applies (allocating per apply would
	// trade the GEMM win for allocator churn on the hot path).
	sa []backend.DeviceMat
	sb []backend.BlockView
	sc []backend.BlockView
}

// buildJIIBatchPlan enumerates every nonzero jiiLKK block via the cheap gate (no integrals, no
// Mat allocation) and groups them with PlanBatches. Walk order matches satelliteResidentBytes'
// (gr descending-inclusive over gc <= gr), so the two agree on which blocks exist by
// construction — TestJIIBatchPlanMatchesGateWalk pins that.
func (mx *Matrix) buildJIIBatchPlan() *jiiBatchPlan {
	sp := mx.sp
	var slots []jiiSlot
	var blocks []backend.Block

	add := func(k satKind, rc, cc Config, r0, c0, rows, cols int, diag bool) {
		slots = append(slots, jiiSlot{
			kind: k, rowCfg: rc, colCfg: cc,
			rowOff: r0, colOff: c0, rows: rows, cols: cols, diag: diag,
		})
		blocks = append(blocks, backend.Block{
			A: jiiShapeMat{rows, cols}, RowOff: r0, ColOff: c0, Diag: diag,
		})
	}

	// jiiLKK: JII rows × JII cols, lower triangle inclusive (mirrors pass 1a/2a).
	for gr := range sp.JII {
		r0 := sp.JII[gr]
		rc := sp.Configs[r0]
		for gc := 0; gc <= gr; gc++ {
			c0 := sp.JII[gc]
			if rows, cols, ok := mx.blk.jiiLKKGate(rc, sp.Configs[c0]); ok {
				add(kindJII, rc, sp.Configs[c0], r0, c0, rows, cols, gr == gc)
			}
		}
	}

	// ijkMLL: IJK rows × ALL JII cols — never on the block diagonal, since the row and column
	// families differ, so it is always applied in both directions (mirrors pass 1b/2a).
	for gr := range sp.IJK {
		r0 := sp.IJK[gr]
		rc := sp.Configs[r0]
		for _, c0 := range sp.JII {
			if rows, cols, ok := mx.blk.ijkMLLGate(rc, sp.Configs[c0]); ok {
				add(kindIJKMLL, rc, sp.Configs[c0], r0, c0, rows, cols, false)
			}
		}
	}

	// ijkLMN: IJK rows × IJK cols, lower triangle inclusive (mirrors pass 1b/2b).
	for gr := range sp.IJK {
		r0 := sp.IJK[gr]
		rc := sp.Configs[r0]
		for gc := 0; gc <= gr; gc++ {
			c0 := sp.IJK[gc]
			if rows, cols, ok := mx.blk.ijkLMNGate(rc, sp.Configs[c0]); ok {
				add(kindIJKLMN, rc, sp.Configs[c0], r0, c0, rows, cols, gr == gc)
			}
		}
	}

	p := &jiiBatchPlan{slots: slots, batches: backend.PlanBatches(blocks)}
	widest := 0
	for _, bt := range p.batches {
		if len(bt.Blocks) > widest {
			widest = len(bt.Blocks)
		}
	}
	p.sa = make([]backend.DeviceMat, widest)
	p.sb = make([]backend.BlockView, widest)
	p.sc = make([]backend.BlockView, widest)
	return p
}

// newJIIMatFreeBatched builds the batched jiiLKK applier. Values are recomputed per apply
// (mx.blk.jiiLKK, unchanged — it already folds in the dSym&&ra==sb diagonal term, so there is no
// diagonal/off-diagonal split to handle here) and issued through GemmMatBatched.
//
// It accumulates (beta=1) and only ever addresses row bands inside the 3h1p satellite region, so
// main-space output rows are never written — required, because TestApplyBlockSatelliteMatFree
// asserts they are LITERALLY zero, not merely small.
func (mx *Matrix) newJIIMatFreeBatched() matFreePart {
	p := mx.buildJIIBatchPlan()

	mats := make([]backend.DeviceMat, len(p.slots))

	apply := func(in, out backend.BlockView) {
		// Fill every slot ONCE per apply. An off-diagonal block appears in two batches (A and
		// Aᵀ); rebuilding it per batch would double the block-build cost, which
		// BenchmarkBlockBuildVsApply shows is ~20% of the work at b=64.
		for i, s := range p.slots {
			blk, ok := mx.buildSlot(s)
			if !ok {
				// The gate said this block is nonzero, so the value builder must agree.
				// Disagreement would silently drop operator contributions.
				panic("dip: satellite gate/value disagreement in batched plan")
			}
			mats[i] = mx.be.UploadMat(blk)
		}
		p.runBatches(mx.be, mats, in, out)
	}
	return matFreePart{apply: apply, release: func() {}}
}

// jiiDeviceBufs is the uploaded, apply-invariant half of the device jiiLKK path: the per-slot
// block geometry plus the shared integral tensors. Only the block VALUES change per mat-vec, so
// everything here is uploaded once.
type jiiDeviceBufs struct {
	dk   backend.DeviceKernels
	args backend.DipFillJIIArgs
	ptrs []unsafe.Pointer // everything to free
}

func (b *jiiDeviceBufs) free() {
	for _, p := range b.ptrs {
		b.dk.FreeDev(p)
	}
}

// buildJIIDeviceBufs marshals the plan into the flat int32 SoA the fill kernel indexes. The
// virtual-orbital lists of each group are concatenated into one flat array addressed by
// (offset, count) per slot — the same ragged-list encoding matfree_device.go already uses for the
// 3h2p/2h1p group virtuals.
func (mx *Matrix) buildJIIDeviceBufs(dk backend.DeviceKernels, p *jiiBatchPlan, s *satDeviceSoA) *jiiDeviceBufs {
	n := len(p.slots)
	kind := make([]int32, n)
	rowO0, rowO1, rowO2 := make([]int32, n), make([]int32, n), make([]int32, n)
	colO0, colO1, colO2 := make([]int32, n), make([]int32, n), make([]int32, n)
	rowVOff, rowNv := make([]int32, n), make([]int32, n)
	colVOff, colNv := make([]int32, n), make([]int32, n)
	bufOff := make([]int32, n)
	rows, cols := make([]int, n), make([]int, n)

	var virs []int32
	// Group virtual lists are shared between slots; cache by virtual-symmetry so the flat array
	// does not blow up with duplicates.
	cache := map[int]int32{}
	putVirs := func(cfg Config) (off, cnt int32) {
		vs := mx.blk.virSym(cfg)
		orbs := mx.blk.virOrbs(vs)
		if o, ok := cache[vs]; ok {
			return o, int32(len(orbs))
		}
		o := int32(len(virs))
		for _, orb := range orbs {
			virs = append(virs, int32(orb))
		}
		cache[vs] = o
		return o, int32(len(orbs))
	}

	// Occ[2] is meaningful only for the 3-occupied IJK family; -1 marks "unused" for JII so a
	// stray read in the kernel would index out of range rather than silently alias orbital 0.
	occ2 := func(c Config, three bool) int32 {
		if !three {
			return -1
		}
		return int32(c.Occ[2])
	}

	total, maxElems := 0, 0
	for i, sl := range p.slots {
		rowThree := sl.kind != kindJII    // ijkMLL/ijkLMN have 3-occupied rows
		colThree := sl.kind == kindIJKLMN // only ijkLMN has a 3-occupied column
		kind[i] = int32(sl.kind)
		rowO0[i], rowO1[i] = int32(sl.rowCfg.Occ[0]), int32(sl.rowCfg.Occ[1])
		colO0[i], colO1[i] = int32(sl.colCfg.Occ[0]), int32(sl.colCfg.Occ[1])
		rowO2[i] = occ2(sl.rowCfg, rowThree)
		colO2[i] = occ2(sl.colCfg, colThree)
		// Virtual-group offset/size WITHOUT the parts factor — the kernel re-applies it when
		// decoding (r,c), so these stay the plain per-group virtual counts.
		rowVOff[i], rowNv[i] = putVirs(sl.rowCfg)
		colVOff[i], colNv[i] = putVirs(sl.colCfg)
		bufOff[i] = int32(total)
		rows[i], cols[i] = sl.rows, sl.cols
		e := sl.rows * sl.cols
		total += e
		maxElems = max(maxElems, e)
	}

	up := func(x []int32) unsafe.Pointer { return dk.UploadInts(x) }
	a := backend.DipFillJIIArgs{
		NSlot: n, Spin: s.spin, Norb: s.norb, Parts: mx.sp.Mult,
		TotalElems: total, MaxElems: maxElems,
		Rows: rows, Cols: cols,
		Kind:  up(kind),
		RowO0: up(rowO0), RowO1: up(rowO1), RowO2: up(rowO2),
		ColO0: up(colO0), ColO1: up(colO1), ColO2: up(colO2),
		RowVOff: up(rowVOff), RowNv: up(rowNv), ColVOff: up(colVOff), ColNv: up(colNv),
		BufOff: up(bufOff), Virs: up(virs),
	}
	return &jiiDeviceBufs{dk: dk, args: a, ptrs: []unsafe.Pointer{
		a.Kind, a.RowO0, a.RowO1, a.RowO2, a.ColO0, a.ColO1, a.ColO2,
		a.RowVOff, a.RowNv, a.ColVOff, a.ColNv, a.BufOff, a.Virs,
	}}
}

// newJIIMatFreeBatchedDevice is the CUDA twin of newJIIMatFreeBatched: per mat-vec it launches
// the fill kernel to materialize every jiiLKK block into device scratch, then issues the same
// planned batched GEMMs against those handles.
//
// The plan, the SoA and the ERI/eps/orbsym tensors are all uploaded once; only the fill runs per
// apply. It shares runBatches with the host path, so both realize the identical plan — only the
// producer of `mats` differs, which is what makes the host tests meaningful evidence for the
// device path's batching logic (the element arithmetic is separately pinned, since the kernel
// calls the same d_jii_s/d_jii_t as dip_sat_apply).
// newSatBatchedDevice is the production entry point for the single-GPU contraction path: it
// uploads the integral tensors and the plan SoA, then returns an applier that per mat-vec fills
// every satellite block on-device and issues the planned batched GEMMs.
//
// It replaces the per-scalar DipSatApply on a DeviceKernels backend. Both are correct; this one
// exists because the per-scalar kernel broadcasts each recomputed element across the b panel
// columns with a scalar loop (measured 0.58% of fp64 peak), whereas materializing the block lets
// cuBLAS read it with a tiled GEMM b times over.
func (mx *Matrix) newSatBatchedDevice(dk backend.DeviceKernels) matFreePart {
	s := mx.buildSatDeviceSoA(mx.buildSatScalarPlan())
	eri, eps, osym := dk.DeviceERI(s.eri), dk.UploadFloats(s.eps), dk.UploadInts(s.osym)
	part, _ := mx.newJIIMatFreeBatchedDevice(dk, s, eri, eps, osym)
	inner := part.release
	part.release = func() {
		inner()
		dk.FreeDev(eri)
		dk.FreeDev(eps)
		dk.FreeDev(osym)
	}
	return part
}

func (mx *Matrix) newJIIMatFreeBatchedDevice(dk backend.DeviceKernels, s *satDeviceSoA, eri, eps, osym unsafe.Pointer) (matFreePart, *jiiBatchPlan) {
	p := mx.buildJIIBatchPlan()
	bufs := mx.buildJIIDeviceBufs(dk, p, s)
	bufs.args.ERI, bufs.args.Eps, bufs.args.OrbSym = eri, eps, osym

	apply := func(in, out backend.BlockView) {
		mats := dk.DipSatFillJII(bufs.args)
		if len(mats) == 0 {
			return
		}
		p.runBatches(mx.be, mats, in, out)
	}
	return matFreePart{apply: apply, release: bufs.free}, p
}

// runBatchesOwned is runBatches restricted to the slots this device owns, with output offsets
// rebased into its local partition.
//
// in is a FULL-HEIGHT slab (global row indexing, Ld = n) because a block's column band can live on
// any partition; out is this device's local band, so its global write offset has outRowOff
// subtracted. owned[bi] lists which slots of batch bi belong here — precomputed once, since the
// plan and the partition bounds are both apply-invariant.
func (p *jiiBatchPlan) runBatchesOwned(be backend.Backend, mats []backend.DeviceMat, in, out backend.BlockView, owned [][]int, outRowOff int) {
	for bi, bt := range p.batches {
		n := 0
		for _, si := range owned[bi] {
			s := p.slots[si]
			p.sa[n] = mats[si]
			if bt.Trans {
				p.sb[n] = in.RowRange(s.rowOff, s.rows)            // global: full-height slab
				p.sc[n] = out.RowRange(s.colOff-outRowOff, s.cols) // local: this partition
			} else {
				p.sb[n] = in.RowRange(s.colOff, s.cols)
				p.sc[n] = out.RowRange(s.rowOff-outRowOff, s.rows)
			}
			n++
		}
		if n > 0 {
			be.GemmMatBatched(bt.Trans, 1, p.sa[:n], p.sb[:n], 1, p.sc[:n])
		}
	}
}

// runBatches issues the planned batched GEMMs against already-filled block handles. Shared by the
// host and device appliers so both realize the plan identically — only how `mats` gets filled
// differs between them.
func (p *jiiBatchPlan) runBatches(be backend.Backend, mats []backend.DeviceMat, in, out backend.BlockView) {
	for _, bt := range p.batches {
		n := 0
		for _, si := range bt.Blocks {
			s := p.slots[si]
			p.sa[n] = mats[si]
			if bt.Trans {
				p.sb[n] = in.RowRange(s.rowOff, s.rows)
				p.sc[n] = out.RowRange(s.colOff, s.cols)
			} else {
				p.sb[n] = in.RowRange(s.colOff, s.cols)
				p.sc[n] = out.RowRange(s.rowOff, s.rows)
			}
			n++
		}
		if n > 0 {
			be.GemmMatBatched(bt.Trans, 1, p.sa[:n], p.sb[:n], 1, p.sc[:n])
		}
	}
}
