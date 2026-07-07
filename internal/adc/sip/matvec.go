package sip

import (
	"adcgo/internal/adc/backend"
	"adcgo/internal/adc/integrals"
)

// Matrix is the IP-ADC(n) secular matrix for one target-symmetry sector. It is
// never stored densely in production — the Lanczos driver calls ApplyFull — but
// BuildMatrix materializes it for the dense validation path and tests.
type Matrix struct {
	sp *Space
	el *elements
	be backend.Backend
	op *assembledOp // built lazily on the first ApplyFull, reused thereafter
}

// New builds the IP-ADC(order) matrix engine for space sp. order is 2 or 3.
func New(sp *Space, ints *integrals.Store, eps []float64, order int, be backend.Backend) *Matrix {
	return &Matrix{sp: sp, el: newElements(sp, ints, eps, order), be: be}
}

// placement is one block of the operator: a backend-resident matrix a applied at
// row offset rowOff, column offset colOff. A block on the block diagonal (diag)
// is applied once (GemvN); an off-diagonal block is applied both ways (GemvN into
// its rows, GemvT into its columns) to realize the symmetric M.
type placement struct {
	a      backend.DeviceMat
	rowOff int
	colOff int
	diag   bool
}

// assembledOp is the block-structured operator uploaded once and reused every
// ApplyFull: the 1h/1h main block (dense symmetric square), the 1h↔2h1p coupling
// (a dense main×sat block), and the 2h1p/2h1p satellite block (a dense symmetric
// square). For very large satellite spaces the future path is recompute-on-device
// / matrix-free c22; here the dense assembly is exact and backend-accelerated.
type assembledOp struct {
	parts []placement
}

// mainBlock builds the dense symmetric 1h/1h main block.
func (mx *Matrix) mainBlock() backend.Mat {
	sp := mx.sp
	M := backend.NewMat(sp.BeginSat, sp.BeginSat)
	for r := range sp.BeginSat {
		i := sp.Configs[r].Occ[0]
		for c := 0; c <= r; c++ {
			j := sp.Configs[c].Occ[0]
			el := mx.el.c11(i, j)
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

// assemble uploads the three blocks once for the resident matrix-vector product.
func (mx *Matrix) assemble() *assembledOp {
	sp := mx.sp
	main := sp.BeginSat
	var parts []placement
	add := func(m backend.Mat, r0, c0 int, diag bool) {
		parts = append(parts, placement{a: mx.be.UploadMat(m), rowOff: r0, colOff: c0, diag: diag})
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
	return &assembledOp{parts: parts}
}

// Size is the matrix dimension.
func (mx *Matrix) Size() int { return mx.sp.Size() }

// MainBlockSize is the dimension of the 1h main space (spectroscopic factors are
// the squared weight of the eigenvector on these first rows).
func (mx *Matrix) MainBlockSize() int { return mx.sp.MainBlockSize() }

// Space returns the underlying configuration space.
func (mx *Matrix) Space() *Space { return mx.sp }

// BuildMatrix materializes the full symmetric secular matrix (both triangles).
func (mx *Matrix) BuildMatrix() backend.Mat {
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
		rows, cols := p.a.Dims()
		mx.be.GemvN(1, p.a, in.Slice(p.colOff, cols), out.Slice(p.rowOff, rows))
		if !p.diag {
			mx.be.GemvT(1, p.a, in.Slice(p.rowOff, rows), out.Slice(p.colOff, cols))
		}
	}
}
