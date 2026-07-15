package sip

import (
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
)

// Matrix is the IP-ADC(n) secular matrix for one target-symmetry sector. It is
// never stored densely in production — the Lanczos driver calls ApplyFull — but
// BuildMatrix materializes it for the dense validation path and tests.
type Matrix struct {
	sp    *Space
	el    *elements
	be    backend.Backend
	op    *assembledOp // built lazily on the first ApplyFull, reused thereafter
	sigma func(i, j int) float64

	matFree       MatFreeMode // dense (default) vs matrix-free for large ADC(4) blocks
	matFreeBudget int64       // Auto threshold: dense block bytes above which to go matrix-free
	wert3         bool        // include the WERT3 5th-order 3h2p-diagonal correction (opt-in)
}

// SetWert3 enables the WERT3 5th-order 3h2p-CI diagonal correction (the full EIGAB
// effective diagonal). Off by default; opt-in until the theADCcode EIGAB value-gate is
// unblocked. See sat3Diag / elements4.go wert3elem.
func (mx *Matrix) SetWert3(on bool) { mx.wert3 = on }

// SetStaticSelfEnergy supplies the static self-energy Σ_ij (a.u.) for the 1h/1h block,
// indexed by absolute occupied-orbital indices. It is subtracted from the main block in
// both paths: CVS-ADC(4) (−ε_i δ_ij − Σ_ij, mainBlock4) and ADC(2)/(3) (c11 − Σ,
// mainBlock). This mirrors theADCcode, which in both propagators reads Σ from a separate
// self-energy module (&self-energy) rather than from the ADC matrix code — for ndadc3ip
// the omission is worth ~0.2 eV on the main lines. Pass nil (the default) for no Σ.
// See docs/adc4_sip_spec.md, TestADC4StaticSigmaGate and TestSIPMatchedReference.
func (mx *Matrix) SetStaticSelfEnergy(sigma func(i, j int) float64) { mx.sigma = sigma }

// New builds the IP-ADC(order) matrix engine for space sp. order is 2 or 3.
func New(sp *Space, ints *integrals.Store, eps []float64, order int, be backend.Backend) *Matrix {
	return &Matrix{sp: sp, el: newElements(sp, ints, eps, order), be: be}
}

// placement is one block of the operator: a backend-resident matrix a applied at
// row offset rowOff, column offset colOff. A block on the block diagonal (diag)
// is applied once (GemvN); an off-diagonal block is applied both ways (GemvN into
// its rows, GemvT into its columns) to realize the symmetric M.
type placement = backend.Block

// assembledOp is the block-structured operator uploaded once and reused every
// ApplyFull: the 1h/1h main block (dense symmetric square), the 1h↔2h1p coupling
// (a dense main×sat block), and the 2h1p/2h1p satellite block (a dense symmetric
// square). For very large satellite spaces the future path is recompute-on-device
// / matrix-free c22; here the dense assembly is exact and backend-accelerated.
type assembledOp struct {
	parts   []placement
	batches []backend.Batch
	sa      []backend.DeviceMat
	sb, sc  []backend.BlockView
	diags   []diagPart    // purely diagonal blocks, applied elementwise (not via GEMM)
	mf      []matFreePart // blocks applied by on-the-fly element recompute (not stored)
}

// diagPart is a diagonal operator block on the main diagonal at offset off: the resident
// vector d holds its diagonal entries, applied as out[off:] += d ⊙ in[off:]. It exists so
// the CVS-ADC(4) 3h2p/3h2p block (diagonal until WERT3 is added) never materializes as a
// dense n×n matrix — the difference between MB and TB for a large satellite space.
type diagPart struct {
	off int
	d   backend.Vector
}

// mainBlock builds the dense symmetric 1h/1h main block: c11 (= k1 + c11_2 [+ c11_3])
// minus the external static self-energy Σ when one is supplied via SetStaticSelfEnergy.
// theADCcode assembles exactly this — nd_adc3_matrix.cpp build_main_block() calls
// calc_k1 + calc_c11_2 (+ calc_c11_3) and then does main_block->daxpy(-1., *sigma_) —
// with Σ coming from its separate self-energy module (&self-energy), not from the ADC
// matrix code. With Σ nil (the default) this is the bare c11 block.
func (mx *Matrix) mainBlock() backend.Mat {
	sp := mx.sp
	M := backend.NewMat(sp.BeginSat, sp.BeginSat)
	for r := range sp.BeginSat {
		i := sp.Configs[r].Occ[0]
		for c := 0; c <= r; c++ {
			j := sp.Configs[c].Occ[0]
			el := mx.el.c11(i, j)
			if mx.sigma != nil {
				el -= mx.sigma(i, j)
			}
			M.Set(r, c, el)
			if r != c {
				M.Set(c, r, el)
			}
		}
	}
	return M
}

// coupling builds the dense 1h×2h1p coupling block (c12).
func (mx *Matrix) coupling() backend.Mat {
	sp := mx.sp
	nSat := sp.Size() - sp.BeginSat
	C := backend.NewMat(sp.BeginSat, nSat)
	for r := range sp.BeginSat {
		j := sp.Configs[r].Occ[0]
		for cIdx := range nSat {
			C.Set(r, cIdx, mx.el.c12(j, sp.Configs[sp.BeginSat+cIdx]))
		}
	}
	return C
}

// satBlock builds the dense symmetric 2h1p/2h1p satellite block (k2 + c22_1).
func (mx *Matrix) satBlock() backend.Mat {
	sp := mx.sp
	nSat := sp.Size() - sp.BeginSat
	S := backend.NewMat(nSat, nSat)
	for r := range nSat {
		S.Set(r, r, mx.el.c22diag(sp.Configs[sp.BeginSat+r]))
		for c := r + 1; c < nSat; c++ {
			// Reference fills column = higher index (the FOR_ALL outer config).
			el := mx.el.c22off(sp.Configs[sp.BeginSat+r], sp.Configs[sp.BeginSat+c])
			S.Set(r, c, el)
			S.Set(c, r, el)
		}
	}
	return S
}

// assemble uploads the blocks once for the resident matrix-vector product. Order 4
// (CVS ADC(4), with a 3h2p space) uses the assemble4 path (matvec4.go).
func (mx *Matrix) assemble() *assembledOp {
	if mx.isADC4() {
		return mx.assemble4()
	}
	sp := mx.sp
	main := sp.BeginSat
	var parts []placement
	add := func(m backend.Mat, r0, c0 int, diag bool) {
		parts = append(parts, placement{A: mx.be.UploadMat(m), RowOff: r0, ColOff: c0, Diag: diag})
	}
	if main > 0 {
		add(mx.mainBlock(), 0, 0, true)
		if sp.Size() > main {
			add(mx.coupling(), 0, main, false)
		}
	}
	if sp.Size() > main {
		add(mx.satBlock(), main, main, true)
	}
	batches := backend.PlanBatches(parts)
	widest := 0
	for _, b := range batches {
		if len(b.Blocks) > widest {
			widest = len(b.Blocks)
		}
	}
	return &assembledOp{
		parts: parts, batches: batches,
		sa: make([]backend.DeviceMat, widest),
		sb: make([]backend.BlockView, widest),
		sc: make([]backend.BlockView, widest),
	}
}

// Release frees the backend-resident operator blocks (a no-op on host backends).
// See the DIP twin.
func (mx *Matrix) Release() {
	if mx.op == nil {
		return
	}
	for _, p := range mx.op.parts {
		mx.be.FreeMat(p.A)
	}
	for _, dg := range mx.op.diags {
		mx.be.Free(dg.d)
	}
	for _, p := range mx.op.mf {
		if p.release != nil {
			p.release()
		}
	}
	mx.op = nil
}

// Size is the matrix dimension.
func (mx *Matrix) Size() int { return mx.sp.Size() }

// MainBlockSize is the dimension of the 1h main space (spectroscopic factors are
// the squared weight of the eigenvector on these first rows).
func (mx *Matrix) MainBlockSize() int { return mx.sp.MainBlockSize() }

// Space returns the underlying configuration space.
func (mx *Matrix) Space() *Space { return mx.sp }

// Diagonal returns the resident diagonal of the secular matrix, assembled directly from
// the per-block element functions — never from BuildMatrix, which is terabytes for a
// large matrix-free order-4 sector. Only the block-diagonal blocks (1h, 2h1p, 3h2p)
// contribute; the off-diagonal coupling blocks have zero diagonal. It backs the Davidson
// (θ−D)⁻¹ preconditioner (lanczos.PreconOperator).
func (mx *Matrix) Diagonal(be backend.Backend) backend.Vector {
	sp := mx.sp
	d := make([]float64, sp.Size())
	main := sp.BeginSat
	if mx.isADC4() {
		// 1h: −ε_P − Σ_PP (mainBlock4).
		for r := range main {
			p := sp.Configs[r].Occ[0]
			v := -mx.el.eps[p]
			if mx.sigma != nil {
				v -= mx.sigma(p, p)
			}
			d[r] = v
		}
		// 2h1p: c22elem4 on the diagonal (satBlock2_4).
		for r := main; r < sp.Begin3h2p; r++ {
			cfg := sp.Configs[r]
			d[r] = mx.el.c22elem4(cfg, cfg)
		}
		// 3h2p: the EIGAB effective diagonal (already a resident vector in the matfree path).
		for r, v := range mx.sat3Diag() {
			d[sp.Begin3h2p+r] = v
		}
	} else {
		// 1h: c11 − Σ (mainBlock).
		for r := range main {
			i := sp.Configs[r].Occ[0]
			v := mx.el.c11(i, i)
			if mx.sigma != nil {
				v -= mx.sigma(i, i)
			}
			d[r] = v
		}
		// 2h1p: k2 + c22_1 on the diagonal (satBlock).
		for r := main; r < sp.Size(); r++ {
			d[r] = mx.el.c22diag(sp.Configs[r])
		}
	}
	return be.Upload(d)
}

// BuildMatrix materializes the full symmetric secular matrix (both triangles).
// Order 4 uses the buildMatrix4 path (matvec4.go).
func (mx *Matrix) BuildMatrix() backend.Mat {
	if mx.isADC4() {
		return mx.buildMatrix4()
	}
	sp := mx.sp
	main := sp.BeginSat
	M := backend.NewMat(sp.Size(), sp.Size())

	mb := mx.mainBlock()
	for r := range main {
		for c := range main {
			M.Set(r, c, mb.At(r, c))
		}
	}
	if sp.Size() > main {
		c12 := mx.coupling()
		for r := range c12.Rows {
			for c := range c12.Cols {
				v := c12.At(r, c)
				M.Set(r, main+c, v)
				M.Set(main+c, r, v)
			}
		}
		sb := mx.satBlock()
		for r := range sb.Rows {
			for c := range sb.Cols {
				M.Set(main+r, main+c, sb.At(r, c))
			}
		}
	}
	return M
}

// ApplyFull computes out = M·in matrix-free on backend-resident vectors, via the
// block-sparse operator assembled once. Each block is a resident GemvN (plus a
// GemvT for the off-diagonal coupling, realizing the symmetric M).
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
	for _, dg := range mx.op.diags {
		n := dg.d.Len()
		mx.be.AxpyDiag(dg.d, in.Slice(dg.off, n), out.Slice(dg.off, n))
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

// ApplyBlock computes out = M·in for every column of in at once, streaming the
// assembled operator once per block instead of once per vector. See the DIP twin
// (dip.Matrix.ApplyBlock) for why that matters. out must have Ld == Rows.
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
	// Diagonal blocks: apply d ⊙ (each column) over the block's row band. Columns are
	// independent, so this is a per-column AxpyDiag on the sub-panel.
	for _, dg := range op.diags {
		n := dg.d.Len()
		inSub := in.RowRange(dg.off, n)
		outSub := out.RowRange(dg.off, n)
		for j := range inSub.Cols {
			mx.be.AxpyDiag(dg.d, inSub.Col(j), outSub.Col(j))
		}
	}
	for _, p := range op.mf {
		p.apply(in, out)
	}
}
