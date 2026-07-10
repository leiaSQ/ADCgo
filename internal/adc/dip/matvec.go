package dip

import (
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
)

// Matrix is the DIP-ADC(2) secular matrix for one (symmetry, spin) sector. It is
// never stored densely in production — the Lanczos driver calls ApplyFull — but
// BuildMatrix materializes it for the dense validation path and tests.
type Matrix struct {
	sp  *Space
	blk blocks
	be  backend.Backend
	op  *assembledOp // built lazily on the first ApplyFull, reused thereafter
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
	return &Matrix{sp: sp, blk: blk, be: be}
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
	sa      []backend.DeviceMat
	sb, sc  []backend.BlockView
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

// assemble builds the block-sparse operator on the backend. It mirrors the block
// enumeration of BuildMatrix, but uploads each block via be.UploadMat instead of
// scattering into a dense matrix, and folds the 2h/2h scalar loop into a single
// dense main-block GemvN so the whole apply is expressible as resident GEMVs.
func (mx *Matrix) assemble() *assembledOp {
	sp := mx.sp
	var parts []placement
	add := func(m backend.Mat, r0, c0 int, diag bool) {
		parts = append(parts, placement{A: mx.be.UploadMat(m), RowOff: r0, ColOff: c0, Diag: diag})
	}

	// 2h/2h main block: a dense symmetric square (both triangles filled), applied
	// as a single GemvN over the main sub-range.
	if main := sp.BeginJII; main > 0 {
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
		add(M, 0, 0, true)
	}

	// 2h ↔ 3h1p coupling (each an nvR×1 column block at (r0, col)).
	place := func(groups []int, typeII bool) {
		for _, r0 := range groups {
			rc := sp.Configs[r0]
			for col := range sp.BeginJII {
				blk, ok := mx.couplingBlock(rc, sp.Configs[col], typeII, col >= sp.BeginIJ)
				if !ok {
					continue
				}
				add(blk, r0, col, false)
			}
		}
	}
	place(sp.JII, false)
	place(sp.IJK, true)

	// 3h1p / 3h1p.
	for gr, r0 := range sp.JII {
		for gc := 0; gc <= gr; gc++ {
			c0 := sp.JII[gc]
			if blk, ok := mx.blk.jiiLKK(sp.Configs[r0], sp.Configs[c0]); ok {
				add(blk, r0, c0, gr == gc)
			}
		}
	}
	for _, r0 := range sp.IJK {
		for _, c0 := range sp.JII {
			if blk, ok := mx.blk.ijkMLL(sp.Configs[r0], sp.Configs[c0]); ok {
				add(blk, r0, c0, false)
			}
		}
	}
	for gr, r0 := range sp.IJK {
		for gc := 0; gc <= gr; gc++ {
			c0 := sp.IJK[gc]
			if blk, ok := mx.blk.ijkLMN(sp.Configs[r0], sp.Configs[c0]); ok {
				add(blk, r0, c0, gr == gc)
			}
		}
	}
	return newAssembledOp(parts)
}

// newAssembledOp plans the batched applies and sizes the scratch slices.
func newAssembledOp(parts []placement) *assembledOp {
	batches := backend.PlanBatches(parts)
	widest := 0
	for _, b := range batches {
		if len(b.Blocks) > widest {
			widest = len(b.Blocks)
		}
	}
	return &assembledOp{
		parts:   parts,
		batches: batches,
		sa:      make([]backend.DeviceMat, widest),
		sb:      make([]backend.BlockView, widest),
		sc:      make([]backend.BlockView, widest),
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
	mx.op = nil
}

// Size is the matrix dimension.
func (mx *Matrix) Size() int { return mx.sp.Size() }

// MainBlockSize is the dimension of the 2h main space (pole strengths are the
// squared weight of the eigenvector on these first rows).
func (mx *Matrix) MainBlockSize() int { return mx.sp.MainBlockSize() }

// Space returns the underlying configuration space.
func (mx *Matrix) Space() *Space { return mx.sp }

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
	mx.be.Zero(out.V)
	op := mx.op
	for _, bt := range op.batches {
		n := len(bt.Blocks)
		for i, pi := range bt.Blocks {
			p := op.parts[pi]
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
