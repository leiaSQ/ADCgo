package selfenergy

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// linear.go — the ph linear equation that promotes Σ⁽⁴⁾ to Σ(4+), and which Σ(∞) reuses with a
// different inhomogeneity. Ported from ../ADC/self_energy/original/original_self_energy.cpp
// linear_eq_selfenergy; equations (B.5) and (B.6) of von Niessen, Schirmer & Cederbaum,
// Comput. Phys. Rep. 1, 57 (1984). The FORTRAN module does the identical thing in sigma.f
// (its A11/A21 + MATIN2 Gauss-Jordan inversion), which is why Σ(4+) and Σ(∞) differ only in
// the inhomogeneity b.
//
// Σ_ph is implicit because the ph density itself contains Σ_ph:
//
//	(1 − A11)·Σ_ph = b_ph          then      Σ_hh/pp = b_hh/pp + A21·Σ_ph
//
// solvePH overwrites sig in place: it arrives holding the inhomogeneity b (the contracted
// density) and leaves holding Σ.

// phPair is one particle-hole index pair (a virtual, an occupied) in the reference's ordering:
// irrep-major, then virtual, then occupied.
type phPair struct{ a, i int }

// pairSpace enumerates the ph pairs and the hh/pp pairs, in the reference's order.
func (e *engine) pairSpace() (ph []phPair, hh, pp [][2]int) {
	for sym := range e.nsym {
		for _, a := range e.virs[sym] {
			for _, i := range e.occs[sym] {
				ph = append(ph, phPair{a, i})
			}
		}
	}
	// hh/pp pairs are stored as {p, q} with p the inner (≤) index, matching the reference's
	// A21 rows, whose expression takes the inner index first: A21[(p,q),·] with p ≤ q.
	for sym := range e.nsym {
		occ := e.occs[sym]
		for qi := range occ {
			for pi := 0; pi <= qi; pi++ {
				hh = append(hh, [2]int{occ[pi], occ[qi]})
			}
		}
	}
	for sym := range e.nsym {
		vir := e.virs[sym]
		for qi := range vir {
			for pi := 0; pi <= qi; pi++ {
				pp = append(pp, [2]int{vir[pi], vir[qi]})
			}
		}
	}
	return
}

// aElem is the coupling that fills both A11 and A21 (they are literally the same expression,
// only the row index set differs):
//
//	A[(p,q),(k,l)] = [2(<pk|ql> + <pl|qk>) − (<pk|lq> + <pl|kq>)] / (ε_k − ε_l)
//
// with (k,l) the ph column pair (l virtual, k occupied).
func (e *engine) aElem(p, q, k, l int) float64 {
	return (2*(e.v(p, k, q, l)+e.v(p, l, q, k)) -
		(e.v(p, k, l, q) + e.v(p, l, k, q))) / (e.eps[k] - e.eps[l])
}

func (e *engine) solvePH(sig *Sigma) error {
	ph, hh, pp := e.pairSpace()
	n := len(ph)
	if n == 0 {
		return nil
	}

	// (B.5): A11 over ph×ph, then M = A11 − 1 (the reference's add_diag(-1.)).
	m := mat.NewDense(n, n, nil)
	// The A11 fill is nph² aElem calls — 8932² ≈ 8.0e7 at production scale (nph = nvir·nocc = 154·58),
	// each four ERI lookups and two divides — and it runs on EVERY FourPlus/Infinite call, in
	// front of the solve below. Row r is the work item: it writes m's row r (a disjoint span of
	// the Dense backing slice) and reads nothing another row writes, including the diagonal
	// fix-up m(r,r), so the fill is bit-identical whatever order the rows land in.
	//
	// HeavyRows, not Rows: nph collapses with point-group symmetry, and Rows would give a
	// small-nph case back to a single core.
	parallel.HeavyRows(n, func(r int) {
		row := ph[r]
		for c, col := range ph {
			// row pair is (q=virtual, p=occupied); column pair is (l=virtual, k=occupied).
			m.Set(r, c, e.aElem(row.i, row.a, col.i, col.a))
		}
		m.Set(r, r, m.At(r, r)-1)
	})

	b := mat.NewVecDense(n, nil)
	for r, row := range ph {
		b.SetVec(r, sig.At(row.a, row.i))
	}

	// (B.6): Σ_ph = −M⁻¹·b. Solve rather than invert — same answer, better conditioned.
	//
	// This is the one piece of solvePH that is NOT parallelized here, deliberately: it is an
	// opaque gonum dense LU on an nph×nph matrix — 8932×8932 at production scale, O(n³) ≈ 7.1e11 flops —
	// and whether it runs on one core or all of them is decided entirely by the BLAS the binary
	// was linked against, not by anything in this package. Built plain it goes through gonum's
	// pure-Go single-threaded BLAS; built with `-tags openblas` it goes through a threaded
	// dgemm/dtrsm. If Σ(∞) is still slow once the loops around it are parallel, this is the
	// remaining serial block, and the fix is the build tag rather than a work pool here.
	var x mat.VecDense
	if err := x.SolveVec(m, b); err != nil {
		return fmt.Errorf("selfenergy: ph linear system is singular: %w", err)
	}
	x.ScaleVec(-1, &x)

	// Σ_hh/pp = b_hh/pp + A21·Σ_ph, with A21 the same coupling over the hh and pp row pairs.
	//
	// (nhh+npp)·nph aElem calls — (1711+11935)·8932 ≈ 1.2e8 at production scale — again on every
	// FourPlus/Infinite call. The row pair is the work item: hh is the occupied triangle and pp
	// the virtual one, the two sets are disjoint, so each (p,q) with p ≤ q is enumerated exactly
	// once and owns the cells (p,q) and (q,p). acc is private and its Σ_c order is untouched;
	// the only sig cells read are the pair's own b_hh/pp, which no other item writes (the ph
	// cells are written afterwards, still serially).
	rows := append(append([][2]int{}, hh...), pp...)
	parallel.HeavyRows(len(rows), func(ri int) {
		p, q := rows[ri][0], rows[ri][1] // p ≤ q, as aElem's first two arguments expect
		acc := sig.At(p, q)
		for c, col := range ph {
			acc += e.aElem(p, q, col.i, col.a) * x.AtVec(c)
		}
		sig.set(p, q, acc)
		sig.set(q, p, acc)
	})
	// Write the ph solution back last, so the hh/pp loop above still saw the original b_ph
	// (it does not read ph elements, but keep the ordering honest).
	for c, col := range ph {
		sig.set(col.a, col.i, x.AtVec(c))
		sig.set(col.i, col.a, x.AtVec(c))
	}
	return nil
}
