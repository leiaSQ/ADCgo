package dip

import (
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// Matrix is the DIP-ADC(2) secular matrix for one (symmetry, spin) sector. It is
// never stored densely in production — the Lanczos driver calls ApplyFull — but
// BuildMatrix materializes it for the dense validation path and tests.
type Matrix struct {
	sp   *Space
	blk  blocks
	be   backend.Backend
	ints *integrals.Store // kept for the matrix-free device path's ERI flatten
	eps  []float64        // orbital energies (absolute index), for the device path
	op   *assembledOp     // built lazily on the first ApplyFull, reused thereafter

	matFree       matfree.Mode // dense (default) vs matrix-free 3h1p↔3h1p satellite region
	matFreeBudget int64        // Auto threshold: satellite dense bytes above which to go matrix-free
}

// New builds the matrix engine for space sp over the given integrals and orbital
// energies (absolute index).
func New(sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) *Matrix {
	b := base{sp: sp, ints: ints, eps: eps}
	var blk blocks
	if sp.Spin == Triplet {
		blk = &triplet{b}
	} else {
		blk = &singlet{b}
	}
	return &Matrix{sp: sp, blk: blk, be: be, ints: ints, eps: eps}
}

// placement is one block of the operator: a backend-resident matrix a applied at
// row offset rowOff, column offset colOff. A diagonal (on the block diagonal)
// block is applied once (GemvN); an off-diagonal block is applied both ways
// (GemvN into its rows, GemvT into its columns) to realize the symmetric M.
type placement = backend.Block

// assembledOp is the block-structured operator uploaded once and reused every
// ApplyFull: the 2h/2h main block (a dense symmetric square, applied as one
// GemvN), the 2h↔3h1p coupling vectors, and the 3h1p/3h1p satellite blocks. It
// stores M's nonzero blocks (block-sparse M) rather than the dense matrix; for
// very large satellite spaces the future path is recompute-blocks-on-device.
type assembledOp struct {
	parts []placement
	// batches groups same-shaped parts into batched-GEMM calls with disjoint output
	// offsets, computed once here because it depends only on the operator's structure.
	// For formic acid's largest sector this turns 55,097 GEMM calls per apply into
	// 1,002. Scratch slices are reused across applies to keep the hot path allocation
	// free.
	batches []backend.Batch
	// satParts / satBatches are the satellite↔satellite subset of the operator (both
	// offsets in the 3h1p space), with its own batching. ApplyBlockSatellite applies only
	// these — the sub-operator with the 2h main block and the 2h↔3h1p couplings removed —
	// which is what the limited-memory Lanczos driver's Tarantelli subspace-iteration gate
	// needs (lanczos.SolveLowMem, Mode B). Indices in satBatches point into satParts.
	satParts   []placement
	satBatches []backend.Batch
	sa         []backend.DeviceMat
	sb, sc     []backend.BlockView
	// mf holds the 3h1p↔3h1p satellite region when it is applied matrix-free (recomputed
	// per mat-vec instead of stored). When set, no satellite blocks appear in parts/satParts;
	// the mf appliers run after the dense batches in ApplyFull/ApplyBlock/ApplyBlockSatellite.
	mf []matFreePart
}

// OperatorNNZ reports the number of stored matrix elements in the assembled
// block-sparse operator (assembling it if needed), together with the block count.
// Every apply streams all of them, so this is the memory-traffic unit of a
// mat-vec and the sizing input for a device-residency check.
func (mx *Matrix) OperatorNNZ() (nnz, nblocks int) {
	if mx.op == nil {
		mx.op = mx.assemble()
	}
	for _, p := range mx.op.parts {
		r, c := p.A.Dims()
		nnz += r * c
	}
	return nnz, len(mx.op.parts)
}

// blockEmit receives one nonzero block of the operator (dense block, row offset,
// column offset, on-block-diagonal flag).
type blockEmit = func(m backend.Mat, r0, c0 int, diag bool)

// mainCouplingTasks enumerates the operator's non-satellite blocks — the 2h/2h main
// block and the 2h↔3h1p couplings — as independent tasks, one per row-group (plus the
// main block), each calling emit for its nonzero blocks in the order BuildMatrix
// produces them. These blocks are always assembled densely (they are small); only the
// 3h1p↔3h1p satellite region is a matrix-free candidate. Each task reads only immutable
// space/integral/energy data and writes to task-local storage, so a worker pool can
// evaluate them concurrently (this runs in the assemble phase, before any GEMM solve,
// so it does not oversubscribe the BLAS backend — cf. internal/adc/parallel).
func (mx *Matrix) mainCouplingTasks() []func(emit blockEmit) {
	sp := mx.sp
	var tasks []func(emit blockEmit)

	// 2h/2h main block: a dense symmetric square (both triangles filled), applied
	// as a single GemvN over the main sub-range. One task; evaluated serially by its
	// worker (main is small next to the satellite region, which carries the tasks
	// that actually fill the pool).
	if main := sp.BeginJII; main > 0 {
		tasks = append(tasks, func(emit blockEmit) {
			M := backend.NewMat(main, main)
			for row := range main {
				for col := 0; col <= row; col++ {
					el, ok := mx.twoHoleElement(row, col)
					if !ok {
						continue
					}
					M.Set(row, col, el)
					if row != col {
						M.Set(col, row, el)
					}
				}
			}
			emit(M, 0, 0, true)
		})
	}

	// 2h ↔ 3h1p coupling (each an nvR×1 column block at (r0, col)): one task per
	// 3h1p row-group, JII (type I) then IJK (type II).
	place := func(groups []int, typeII bool) {
		for _, r0 := range groups {
			tasks = append(tasks, func(emit blockEmit) {
				rc := sp.Configs[r0]
				for col := range sp.BeginJII {
					blk, ok := mx.couplingBlock(rc, sp.Configs[col], typeII, col >= sp.BeginIJ)
					if !ok {
						continue
					}
					emit(blk, r0, col, false)
				}
			})
		}
	}
	place(sp.JII, false)
	place(sp.IJK, true)
	return tasks
}

// satelliteTasks enumerates the 3h1p↔3h1p satellite blocks (the operator's dominant
// resident term) as independent tasks, one per row-group, in section order (JII×JII lower
// triangle, IJK×JII, IJK×IJK lower triangle). These are what the matrix-free path applies
// on the fly instead of storing (see matfree.go); when matrix-free is off they are
// assembled densely like everything else.
func (mx *Matrix) satelliteTasks() []func(emit blockEmit) {
	sp := mx.sp
	var tasks []func(emit blockEmit)
	for gr, r0 := range sp.JII {
		tasks = append(tasks, func(emit blockEmit) {
			for gc := 0; gc <= gr; gc++ {
				c0 := sp.JII[gc]
				if blk, ok := mx.blk.jiiLKK(sp.Configs[r0], sp.Configs[c0]); ok {
					emit(blk, r0, c0, gr == gc)
				}
			}
		})
	}
	for _, r0 := range sp.IJK {
		tasks = append(tasks, func(emit blockEmit) {
			for _, c0 := range sp.JII {
				if blk, ok := mx.blk.ijkMLL(sp.Configs[r0], sp.Configs[c0]); ok {
					emit(blk, r0, c0, false)
				}
			}
		})
	}
	for gr, r0 := range sp.IJK {
		tasks = append(tasks, func(emit blockEmit) {
			for gc := 0; gc <= gr; gc++ {
				c0 := sp.IJK[gc]
				if blk, ok := mx.blk.ijkLMN(sp.Configs[r0], sp.Configs[c0]); ok {
					emit(blk, r0, c0, gr == gc)
				}
			}
		})
	}
	return tasks
}

// runTaskParts evaluates block tasks across a worker pool and returns the uploaded
// placements. Each task writes only its own results slot; the slots are concatenated in
// task order, so parts is byte-identical to a serial walk. Blocks upload as they are
// produced, so host memory holds only the blocks in flight, not the whole operator.
func (mx *Matrix) runTaskParts(tasks []func(emit blockEmit)) []placement {
	results := make([][]placement, len(tasks))
	parallel.Rows(len(tasks), func(t int) {
		tasks[t](func(m backend.Mat, r0, c0 int, diag bool) {
			results[t] = append(results[t], placement{A: mx.be.UploadMat(m), RowOff: r0, ColOff: c0, Diag: diag})
		})
	})
	var parts []placement
	for _, r := range results {
		parts = append(parts, r...)
	}
	return parts
}

// sumTaskBytes evaluates block tasks across a worker pool and returns their total resident
// size (Σ rows·cols·8) without uploading.
func (mx *Matrix) sumTaskBytes(tasks []func(emit blockEmit)) uint64 {
	sums := make([]uint64, len(tasks))
	parallel.Rows(len(tasks), func(t int) {
		var elems uint64
		tasks[t](func(m backend.Mat, r0, c0 int, diag bool) {
			elems += uint64(m.Rows) * uint64(m.Cols)
		})
		sums[t] = elems
	})
	var total uint64
	for _, s := range sums {
		total += s
	}
	return total * 8 // sizeof(float64)
}

// assemble builds the block-sparse operator on the backend. The main block and 2h↔3h1p
// couplings are always uploaded densely; the 3h1p↔3h1p satellite region is either uploaded
// densely or, when matFreeSatellite() elects it, applied on the fly by op.mf and never
// stored (the memory fix — the satellite region is the bulk of the resident footprint).
func (mx *Matrix) assemble() *assembledOp {
	matFree := mx.matFreeSatellite()
	tasks := mx.mainCouplingTasks()
	if !matFree {
		tasks = append(tasks, mx.satelliteTasks()...)
	}
	op := newAssembledOp(mx.runTaskParts(tasks), mx.sp.BeginJII)
	if matFree {
		op.mf = []matFreePart{mx.newSatelliteMatFreePart()}
	}
	return op
}

// OperatorResidentBytes reports the device memory the assembled operator would occupy, in
// bytes, without uploading anything, so the backend chooser can decide up front whether a
// sector fits a GPU. It sums the dense main+coupling blocks (worker-pool block eval) plus
// the satellite region — but when the satellite region is applied matrix-free it contributes
// zero resident bytes (the whole point), and otherwise it is sized by the cheap gate walk
// (satelliteResidentBytes, no integral evaluation). This is the exact resident size, not a
// dense-n² upper bound.
func (mx *Matrix) OperatorResidentBytes() uint64 {
	bytes := mx.sumTaskBytes(mx.mainCouplingTasks())
	if !mx.matFreeSatellite() {
		bytes += mx.satelliteResidentBytes()
	}
	return bytes
}

// newAssembledOp plans the batched applies and sizes the scratch slices. main is the 2h
// main-block size (sp.BeginJII): the satellite↔satellite subset (both offsets ≥ main) is
// planned separately for the gated apply.
func newAssembledOp(parts []placement, main int) *assembledOp {
	batches := backend.PlanBatches(parts)
	var satParts []placement
	for _, p := range parts {
		if p.RowOff >= main && p.ColOff >= main {
			satParts = append(satParts, p)
		}
	}
	satBatches := backend.PlanBatches(satParts)
	widest := 0
	for _, bs := range [][]backend.Batch{batches, satBatches} {
		for _, b := range bs {
			if len(b.Blocks) > widest {
				widest = len(b.Blocks)
			}
		}
	}
	return &assembledOp{
		parts:      parts,
		batches:    batches,
		satParts:   satParts,
		satBatches: satBatches,
		sa:         make([]backend.DeviceMat, widest),
		sb:         make([]backend.BlockView, widest),
		sc:         make([]backend.BlockView, widest),
	}
}

// Release frees the backend-resident operator blocks. On a host backend this is a
// no-op; on a device it is the difference between one sector's operator (0.25–0.48 GB
// for formic acid) living until the process exits and being reclaimed for the next
// sector. Safe to call more than once; the operator is rebuilt on the next apply.
func (mx *Matrix) Release() {
	if mx.op == nil {
		return
	}
	for _, p := range mx.op.parts {
		mx.be.FreeMat(p.A)
	}
	for _, p := range mx.op.mf {
		p.release()
	}
	mx.op = nil
}

// Size is the matrix dimension.
func (mx *Matrix) Size() int { return mx.sp.Size() }

// MainBlockSize is the dimension of the 2h main space (pole strengths are the
// squared weight of the eigenvector on these first rows).
func (mx *Matrix) MainBlockSize() int { return mx.sp.MainBlockSize() }

// Space returns the underlying configuration space.
func (mx *Matrix) Space() *Space { return mx.sp }

// Diagonal returns the resident diagonal of the secular matrix, assembled directly from
// the block element functions rather than from BuildMatrix. Only the block-diagonal
// blocks contribute — the 2h/2h main block and each 3h1p group's self-block (jiiLKK /
// ijkLMN); the 2h↔3h1p and cross-group 3h1p blocks are off-diagonal. It backs the
// Davidson (θ−D)⁻¹ preconditioner (lanczos.PreconOperator).
func (mx *Matrix) Diagonal(be backend.Backend) backend.Vector {
	sp := mx.sp
	d := make([]float64, sp.Size())
	// 2h/2h main block.
	for row := range sp.BeginJII {
		if el, ok := mx.twoHoleElement(row, row); ok {
			d[row] = el
		}
	}
	// 3h1p/3h1p diagonal self-blocks (type I via jiiLKK, type II via ijkLMN).
	for _, r0 := range sp.JII {
		if blk, ok := mx.blk.jiiLKK(sp.Configs[r0], sp.Configs[r0]); ok {
			for a := range blk.Rows {
				d[r0+a] = blk.At(a, a)
			}
		}
	}
	for _, r0 := range sp.IJK {
		if blk, ok := mx.blk.ijkLMN(sp.Configs[r0], sp.Configs[r0]); ok {
			for a := range blk.Rows {
				d[r0+a] = blk.At(a, a)
			}
		}
	}
	return be.Upload(d)
}

// twoHoleElement returns the 2h/2h element (row,col) with row>=col, dispatching
// on which 2h family each config belongs to.
func (mx *Matrix) twoHoleElement(row, col int) (float64, bool) {
	sp := mx.sp
	rc, cc := sp.Configs[row], sp.Configs[col]
	switch {
	case row < sp.BeginIJ: // both |ii> closed-shell
		return mx.blk.iiJJ(rc, cc)
	case col < sp.BeginIJ: // row |ij> open, col |ii> closed
		return mx.blk.ijKK(rc, cc)
	default: // both |ij> open-shell
		return mx.blk.ijKL(rc, cc)
	}
}

// couplingBlock returns the 3h1p(row-group)↔2h(col) coupling block.
func (mx *Matrix) couplingBlock(rc, cc Config, typeII, colOpen bool) (backend.Mat, bool) {
	switch {
	case !typeII && !colOpen:
		return mx.blk.lkkII(rc, cc)
	case !typeII && colOpen:
		return mx.blk.lkkIJ(rc, cc)
	case typeII && !colOpen:
		return mx.blk.klmII(rc, cc)
	default:
		return mx.blk.klmIJ(rc, cc)
	}
}

// BuildMatrix materializes the full symmetric secular matrix (both triangles).
func (mx *Matrix) BuildMatrix() backend.Mat {
	sp := mx.sp
	M := backend.NewMat(sp.Size(), sp.Size())

	// 2h / 2h main block.
	for row := range sp.BeginJII {
		for col := 0; col <= row; col++ {
			el, ok := mx.twoHoleElement(row, col)
			if !ok {
				continue
			}
			M.Set(row, col, el)
			if row != col {
				M.Set(col, row, el)
			}
		}
	}

	// 2h ↔ 3h1p coupling (place column block + transpose).
	place := func(groups []int, typeII bool) {
		for _, r0 := range groups {
			rc := sp.Configs[r0]
			for col := range sp.BeginJII {
				blk, ok := mx.couplingBlock(rc, sp.Configs[col], typeII, col >= sp.BeginIJ)
				if !ok {
					continue
				}
				for a := range blk.Rows {
					v := blk.At(a, 0)
					M.Set(r0+a, col, v)
					M.Set(col, r0+a, v)
				}
			}
		}
	}
	place(sp.JII, false)
	place(sp.IJK, true)

	// 3h1p / 3h1p.
	placeMat := func(r0, c0 int, blk backend.Mat, diag bool) {
		for a := range blk.Rows {
			for b := range blk.Cols {
				v := blk.At(a, b)
				M.Set(r0+a, c0+b, v)
				if !diag {
					M.Set(c0+b, r0+a, v)
				}
			}
		}
	}
	// type I × type I.
	for gr, r0 := range sp.JII {
		for gc := 0; gc <= gr; gc++ {
			c0 := sp.JII[gc]
			if blk, ok := mx.blk.jiiLKK(sp.Configs[r0], sp.Configs[c0]); ok {
				placeMat(r0, c0, blk, gr == gc)
			}
		}
	}
	// type II × type I (always off-diagonal region).
	for _, r0 := range sp.IJK {
		for _, c0 := range sp.JII {
			if blk, ok := mx.blk.ijkMLL(sp.Configs[r0], sp.Configs[c0]); ok {
				placeMat(r0, c0, blk, false)
			}
		}
	}
	// type II × type II.
	for gr, r0 := range sp.IJK {
		for gc := 0; gc <= gr; gc++ {
			c0 := sp.IJK[gc]
			if blk, ok := mx.blk.ijkLMN(sp.Configs[r0], sp.Configs[c0]); ok {
				placeMat(r0, c0, blk, gr == gc)
			}
		}
	}
	return M
}

// ApplyFull computes out = M·in matrix-free on backend-resident vectors, via the
// block-sparse operator assembled once by assemble(). Every block — the 2h/2h
// main square, the 2h↔3h1p coupling vectors, and the 3h1p/3h1p satellite blocks
// — is a resident GemvN (plus a GemvT for off-diagonal blocks, realizing the
// symmetric M). No host↔device transfer occurs per apply.
func (mx *Matrix) ApplyFull(out, in backend.Vector) {
	if mx.op == nil {
		mx.op = mx.assemble()
	}
	mx.be.Zero(out)
	for _, p := range mx.op.parts {
		rows, cols := p.A.Dims()
		mx.be.GemvN(1, p.A, in.Slice(p.ColOff, cols), out.Slice(p.RowOff, rows))
		if !p.Diag {
			mx.be.GemvT(1, p.A, in.Slice(p.RowOff, rows), out.Slice(p.ColOff, cols))
		}
	}
	if len(mx.op.mf) > 0 {
		n := mx.sp.Size()
		inV := backend.BlockView{V: in, Rows: n, Cols: 1, Ld: n}
		outV := backend.BlockView{V: out, Rows: n, Cols: 1, Ld: n}
		for _, p := range mx.op.mf {
			p.apply(inV, outV)
		}
	}
}

// ApplyBlock computes out = M·in for every column of in at once.
//
// This is ApplyFull's level-3 twin, and the reason it exists is memory traffic, not
// arithmetic. ApplyFull streams the entire assembled operator once per vector; over a
// Lanczos run that is `dim` passes over ~0.25 GB of blocks (formic acid), plus one
// GEMV call per block per vector — 1.8e8 calls, where per-call overhead, not
// bandwidth, sets the pace. Applying to a whole b-column block streams the operator
// once per block and cuts the call count by b, turning a bandwidth/overhead-bound
// GEMV into a compute-bound GEMM.
//
// out must be a dedicated buffer with Ld == Rows; it is zeroed first, then every
// block accumulates into it (beta = 1), mirroring ApplyFull's additive GEMVs.
func (mx *Matrix) ApplyBlock(out, in backend.BlockView) {
	if mx.op == nil {
		mx.op = mx.assemble()
	}
	mx.applyBatches(mx.op.parts, mx.op.batches, out, in)
	for _, p := range mx.op.mf {
		p.apply(in, out)
	}
}

// ApplyBlockSatellite computes out = M_sat·in, where M_sat is M with the 2h main block and
// the 2h↔3h1p couplings removed — only the 3h1p↔3h1p satellite blocks act. It is
// lanczos.SolveLowMem's Tarantelli subspace-iteration gate (Mode B): applied after the
// first two blocks, it forces every later Lanczos vector to zero main-space weight, which is
// what lets the banded eigensolver read pole strengths from the projected eigenvectors'
// top rows without keeping the basis. out must be a dedicated buffer with Ld == Rows.
func (mx *Matrix) ApplyBlockSatellite(out, in backend.BlockView) {
	if mx.op == nil {
		mx.op = mx.assemble()
	}
	mx.applyBatches(mx.op.satParts, mx.op.satBatches, out, in)
	// Under matrix-free the satellite region lives in op.mf, not satParts; apply it here so
	// ApplyBlockSatellite realizes the same 3h1p↔3h1p sub-operator the dense path does.
	for _, p := range mx.op.mf {
		p.apply(in, out)
	}
}

// applyBatches zeroes out and accumulates the given batched parts into it (beta = 1),
// shared by ApplyBlock (all parts) and ApplyBlockSatellite (the satellite subset).
func (mx *Matrix) applyBatches(parts []placement, batches []backend.Batch, out, in backend.BlockView) {
	mx.be.Zero(out.V)
	op := mx.op
	for _, bt := range batches {
		n := len(bt.Blocks)
		for i, pi := range bt.Blocks {
			p := parts[pi]
			rows, cols := p.A.Dims()
			op.sa[i] = p.A
			if bt.Trans {
				op.sb[i] = in.RowRange(p.RowOff, rows)
				op.sc[i] = out.RowRange(p.ColOff, cols)
			} else {
				op.sb[i] = in.RowRange(p.ColOff, cols)
				op.sc[i] = out.RowRange(p.RowOff, rows)
			}
		}
		mx.be.GemmMatBatched(bt.Trans, 1, op.sa[:n], op.sb[:n], 1, op.sc[:n])
	}
}
