package selfenergy

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
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
	for r, row := range ph {
		for c, col := range ph {
			// row pair is (q=virtual, p=occupied); column pair is (l=virtual, k=occupied).
			m.Set(r, c, e.aElem(row.i, row.a, col.i, col.a))
		}
		m.Set(r, r, m.At(r, r)-1)
	}

	b := mat.NewVecDense(n, nil)
	for r, row := range ph {
		b.SetVec(r, sig.At(row.a, row.i))
	}

	// (B.6): Σ_ph = −M⁻¹·b. Solve rather than invert — same answer, better conditioned.
	var x mat.VecDense
	if err := x.SolveVec(m, b); err != nil {
		return fmt.Errorf("selfenergy: ph linear system is singular: %w", err)
	}
	x.ScaleVec(-1, &x)

	// Σ_hh/pp = b_hh/pp + A21·Σ_ph, with A21 the same coupling over the hh and pp row pairs.
	for _, r := range append(append([][2]int{}, hh...), pp...) {
		p, q := r[0], r[1] // p ≤ q, as aElem's first two arguments expect
		acc := sig.At(p, q)
		for c, col := range ph {
			acc += e.aElem(p, q, col.i, col.a) * x.AtVec(c)
		}
		sig.set(p, q, acc)
		sig.set(q, p, acc)
	}
	// Write the ph solution back last, so the hh/pp loop above still saw the original b_ph
	// (it does not read ph elements, but keep the ordering honest).
	for c, col := range ph {
		sig.set(col.a, col.i, x.AtVec(c))
		sig.set(col.i, col.a, x.AtVec(c))
	}
	return nil
}
