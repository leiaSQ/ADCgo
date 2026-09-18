package sip

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// matvec22.go — assembly of the ISR-ADC(2,2) secular matrix (1h | 2h1p | 3h2p),
// dispatched from Matrix.assemble()/BuildMatrix() when the space was built by
// NewSpace22. Blocks, per Table I of the paper:
//
//	[1h  , 1h  ]  c11 at order 2 (k1 + c11_2) − Σ        0,2
//	[1h  , 2h1p]  c12_22 (c12_1, plus c12_2 for variant f)  1 / 1,2
//	[2h1p,2h1p ]  c22diag/c22off + c22_2                  0,1,2
//	[2h1p,3h2p ]  c23_1                                       1
//	[3h2p,3h2p ]  c33_0 (diagonal) or c33_01              0 / 0,1
//
// The sign convention follows the rest of the package: off-diagonal blocks are
// placed once and realized both ways by the operator apply (GemvT), so the
// assembled M is symmetric.
//
// The 3h2p/3h2p block is n3² and the 2h1p/3h2p coupling n2·n3, which for anything
// past a small molecule cannot be stored; both go matrix-free (matfree22.go) under
// -matfree, and OperatorResidentBytes reports what a sector costs either way. The
// block builders below are the dense path they are gated against.

// isADC22 reports whether this Matrix is an ADC(2,2) matrix.
func (mx *Matrix) isADC22() bool { return mx.el.order == Order22 && mx.sp.adc22 }

// SetVariant selects the ADC(2,2) scheme (Table I). Has no effect on other orders.
func (mx *Matrix) SetVariant(v Variant) { mx.el.variant = v }

// Variant reports the ADC(2,2) scheme in force.
func (mx *Matrix) Variant() Variant { return mx.el.variant }

// mainBlock22 is the 1h/1h block: c11 at order 2, minus the external static
// self-energy when one is supplied. Identical in structure to mainBlock; the
// order-2 restriction is enforced inside c11 by isADC22.
func (mx *Matrix) mainBlock22() backend.Mat {
	sp := mx.sp
	n := sp.BeginSat
	M := backend.NewMat(n, n)
	parallel.Rows(n, func(r int) {
		i := sp.Configs[r].Occ[0]
		for c := range n {
			j := sp.Configs[c].Occ[0]
			v := mx.el.c11(i, j)
			if mx.sigma != nil {
				v -= mx.sigma(i, j)
			}
			M.Set(r, c, v)
		}
	})
	return M
}

// coupling22 is the 1h × 2h1p block. Parallel over columns for the same reason the
// order-3 coupling is: the row count is the tiny main space, so row-parallelism
// would leave almost every core idle.
func (mx *Matrix) coupling22() backend.Mat {
	sp := mx.sp
	main, n2 := sp.BeginSat, sp.Begin3h2p-sp.BeginSat
	C := backend.NewMat(main, n2)
	parallel.HeavyRows(n2, func(c int) {
		cfg := sp.Configs[sp.BeginSat+c]
		for r := range main {
			C.Set(r, c, mx.el.c12_22(sp.Configs[r].Occ[0], cfg))
		}
	})
	return C
}

// satBlock22 is the 2h1p × 2h1p block: the zeroth/first-order element reused from
// the non-Dyson ADC(3) path plus the second-order A9-A14 term.
func (mx *Matrix) satBlock22() backend.Mat {
	sp := mx.sp
	n2 := sp.Begin3h2p - sp.BeginSat
	M := backend.NewMat(n2, n2)
	parallel.Rows(n2, func(r int) {
		row := sp.Configs[sp.BeginSat+r]
		for c := range n2 {
			col := sp.Configs[sp.BeginSat+c]
			v := mx.el.c22_2(row, col)
			if r == c {
				v += mx.el.c22diag(row)
			} else {
				v += mx.el.c22off(row, col)
			}
			M.Set(r, c, v)
		}
	})
	return M
}

// coupling23 is the 2h1p × 3h2p block (A15-A17) — the block that couples the final
// states of a first-order decay to those of a second-order one, and hence the
// block that makes double Auger and double ICD describable at all.
func (mx *Matrix) coupling23() backend.Mat {
	sp := mx.sp
	n2, n3 := sp.Begin3h2p-sp.BeginSat, len(sp.Sat3)
	C := backend.NewMat(n2, n3)
	parallel.Rows(n2, func(r int) {
		row := sp.Configs[sp.BeginSat+r]
		for c := range n3 {
			C.Set(r, c, mx.el.c23_1(row, sp.Sat3[c]))
		}
	})
	return C
}

// sat3Diag22 is the zeroth-order 3h2p diagonal (A18), the whole 3h2p/3h2p block of
// ADC(2,2)_m.
func (mx *Matrix) sat3Diag22() []float64 {
	sp := mx.sp
	d := make([]float64, len(sp.Sat3))
	parallel.Rows(len(d), func(r int) { d[r] = mx.el.c33_0(sp.Sat3[r]) })
	return d
}

// sat3Block22 is the full zeroth-plus-first-order 3h2p × 3h2p block (A18 plus
// A19-A22), used by ADC(2,2)_x and _f. This is the block that sets the method's
// n_occ³·n_virt⁴ cost.
func (mx *Matrix) sat3Block22() backend.Mat {
	sp := mx.sp
	n3 := len(sp.Sat3)
	M := backend.NewMat(n3, n3)
	parallel.Rows(n3, func(r int) {
		row := sp.Sat3[r]
		for c := range n3 {
			M.Set(r, c, mx.el.c33_01(row, sp.Sat3[c]))
		}
	})
	return M
}

// assemble22 builds the operator for an ADC(2,2) sector.
func (mx *Matrix) assemble22() *assembledOp {
	sp := mx.sp
	main := sp.BeginSat
	n2 := sp.Begin3h2p - main
	n3 := len(sp.Sat3)
	defer mx.assembleDone(mx.assembleStart("ADC(2,2)"+mx.el.variant.String(), main, n2, n3))

	var parts []placement
	var diags []diagPart
	var mfree []matFreePart
	add := func(m backend.Mat, r0, c0 int, diag bool) {
		parts = append(parts, placement{A: mx.be.UploadMat(m), RowOff: r0, ColOff: c0, Diag: diag})
	}
	if main > 0 {
		mx.assembleStep(fmt.Sprintf("1h/1h main block (%d×%d)", main, main), func() {
			add(mx.mainBlockCached22(), 0, 0, true)
		})
		if n2 > 0 {
			mx.assembleStep(fmt.Sprintf("1h/2h1p coupling (%d×%d)", main, n2), func() {
				add(mx.coupling22(), 0, main, false)
			})
		}
	}
	if n2 > 0 {
		mx.assembleStep(fmt.Sprintf("2h1p/2h1p block (%d×%d)", n2, n2), func() {
			add(mx.satBlock22(), main, main, true)
		})
		if n3 > 0 {
			// The 2h1p/3h2p coupling is n2·n3·8 bytes dense. Recompute it instead
			// when requested or when that exceeds the budget.
			if mx.matFreeC23(blockBytes(n2, n3)) {
				mx.assembleStep(fmt.Sprintf("2h1p/3h2p coupling, matrix-free (%d×%d)", n2, n3), func() {
					mfree = append(mfree, mx.newC23MatFree())
				})
			} else {
				mx.assembleStep(fmt.Sprintf("2h1p/3h2p coupling, dense (%d×%d)", n2, n3), func() {
					add(mx.coupling23(), main, sp.Begin3h2p, false)
				})
			}
		}
	}
	if n3 > 0 {
		switch {
		case mx.el.variant == VariantM:
			// ADC(2,2)_m's 3h2p/3h2p block is the zeroth-order diagonal (A18), so it
			// rides as a resident vector — the difference between MB and TB.
			mx.assembleStep(fmt.Sprintf("3h2p diagonal (%d)", n3), func() {
				diags = append(diags, diagPart{off: sp.Begin3h2p, d: mx.be.Upload(mx.sat3Diag22())})
			})
		case mx.matFreeSat3(blockBytes(n3, n3)):
			mx.assembleStep(fmt.Sprintf("3h2p/3h2p block, matrix-free (%d×%d)", n3, n3), func() {
				mfree = append(mfree, mx.newSat3MatFree())
			})
		default:
			mx.assembleStep(fmt.Sprintf("3h2p/3h2p block, dense (%d×%d)", n3, n3), func() {
				add(mx.sat3Block22(), sp.Begin3h2p, sp.Begin3h2p, true)
			})
		}
	}
	return finalizeOp(parts, diags, mfree)
}

// mainBlockCached22 is mainBlock22() behind the same persistent cache the other
// orders use.
func (mx *Matrix) mainBlockCached22() backend.Mat {
	if mx.loadMain != nil {
		if m, ok := mx.loadMain(); ok {
			return m
		}
	}
	m := mx.mainBlock22()
	if mx.saveMain != nil {
		mx.saveMain(m)
	}
	return m
}

// buildMatrix22 materializes the full symmetric ADC(2,2) secular matrix. Dense and
// exact — the validation oracle, not a production path.
func (mx *Matrix) buildMatrix22() backend.Mat {
	sp := mx.sp
	main := sp.BeginSat
	n2 := sp.Begin3h2p - main
	n3 := len(sp.Sat3)
	M := backend.NewMat(sp.Size(), sp.Size())

	mb := mx.mainBlockCached22()
	for r := range main {
		for c := range main {
			M.Set(r, c, mb.At(r, c))
		}
	}
	if n2 > 0 && main > 0 {
		c12 := mx.coupling22()
		for r := range main {
			for c := range n2 {
				v := c12.At(r, c)
				M.Set(r, main+c, v)
				M.Set(main+c, r, v)
			}
		}
	}
	if n2 > 0 {
		s2 := mx.satBlock22()
		for r := range n2 {
			for c := range n2 {
				M.Set(main+r, main+c, s2.At(r, c))
			}
		}
	}
	if n3 > 0 && n2 > 0 {
		c23 := mx.coupling23()
		for r := range n2 {
			for c := range n3 {
				v := c23.At(r, c)
				M.Set(main+r, sp.Begin3h2p+c, v)
				M.Set(sp.Begin3h2p+c, main+r, v)
			}
		}
	}
	if n3 > 0 {
		if mx.el.variant == VariantM {
			for r, v := range mx.sat3Diag22() {
				M.Set(sp.Begin3h2p+r, sp.Begin3h2p+r, v)
			}
		} else {
			s3 := mx.sat3Block22()
			for r := range n3 {
				for c := range n3 {
					M.Set(sp.Begin3h2p+r, sp.Begin3h2p+c, s3.At(r, c))
				}
			}
		}
	}
	return M
}

// diagonal22 fills the resident diagonal for the Davidson preconditioner.
func (mx *Matrix) diagonal22(d []float64) {
	sp := mx.sp
	for r := range sp.BeginSat {
		i := sp.Configs[r].Occ[0]
		v := mx.el.c11(i, i)
		if mx.sigma != nil {
			v -= mx.sigma(i, i)
		}
		d[r] = v
	}
	for r := sp.BeginSat; r < sp.Begin3h2p; r++ {
		cfg := sp.Configs[r]
		d[r] = mx.el.c22diag(cfg) + mx.el.c22_2(cfg, cfg)
	}
	for r, cfg := range sp.Sat3 {
		if mx.el.variant == VariantM {
			d[sp.Begin3h2p+r] = mx.el.c33_0(cfg)
		} else {
			d[sp.Begin3h2p+r] = mx.el.c33_01(cfg, cfg)
		}
	}
}
