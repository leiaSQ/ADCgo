package sip

import (
	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// The spectroscopic (transition) amplitudes: the effective one-hole overlap of a
// final cationic state is a = F·Y, where Y is the state's 1h (main-block)
// eigenvector part and F is the ND-ADC F-matrix (calc_k1/calc_c11_2/calc_c11_3
// fill it: F = 1 + F⁽²⁾ + F⁽³⁾, symmetric). The spectroscopic factor (pole
// strength) is ‖a‖²; the per-orbital decomposition is a itself. This renormalizes
// the raw main-block weight ‖Y‖² by the ISR effective transition moments.

// f2 is the 2nd-order F-matrix contribution F⁽²⁾_ij (calc_c11_2.c, f_ij term),
// symmetric in i,j (absolute occupied indices of the target irrep).
func (e *elements) f2(i, j int) float64 {
	var fij float64
	for a := e.nocc; a < e.norb; a++ {
		for b := e.nocc; b < e.norb; b++ {
			for l := range e.nocc {
				if e.so(a)^e.so(b)^e.so(l) != e.so(i) {
					continue
				}
				ea, eb, el := e.eps[a], e.eps[b], e.eps[l]
				vabil := e.v(a, b, i, l)
				vabli := e.v(a, b, l, i)
				vabjl := e.v(a, b, j, l)
				vablj := e.v(a, b, l, j)
				vv := vabil*(2*vabjl-vablj) + vabli*(2*vablj-vabjl)
				fij += -0.25 * vv / ((ea + eb - el - e.eps[i]) * (ea + eb - el - e.eps[j]))
			}
		}
	}
	return fij
}

// FMatrix builds the symmetric dim_1h × dim_1h F-matrix (transition amplitudes)
// for this sector: F = 1 + F⁽²⁾ (+ F⁽³⁾ at order 3, the hermitian (f_ij+f_ji)/2
// contribution from calc_c11_3). For order 4 the ≤3rd-order F is reused; the F⁽⁴⁾
// spectroscopic-amplitude term is not yet ported (it does not affect the secular
// matrix, only pole strengths).
//
// This is the SECOND sweep over the lower triangle of the 1h/1h block, and at order 3 it is
// as expensive as mainBlock(): c11_3sums is the same five-deep a,b,l,c,d sum, O(nvir⁴·nocc)
// ≈ 3.3e10 innermost iterations per element, over the same 1711 elements for the production system
// (nocc=58, nvir=154). Job 14551670 measured 8 h 16 m for mainBlock's traversal of exactly
// this shape; left serial, FMatrix silently pays that a second time — and it runs AFTER the
// solver, i.e. after the whole eigenproblem has already been paid for, where nothing is
// checkpointed and a walltime kill throws away the entire run's spectrum. C1 symmetry prunes
// none of it (every so(a)^so(b)^so(l) != so(i) test is 0 != 0), and the main-block cache
// (SetMainBlockCache) does not cover F.
//
// Flattened lower triangle + HeavyRows, exactly as mainBlock does and for the same two
// reasons: row r holds r+1 elements of equal cost, so a row split strands the longest row
// (58 elements) on the critical path, and a row count of 58 sits below parallel.Rows'
// 2*GOMAXPROCS serial fallback (128 on a 64-core node) — the case that fallback gets wrong.
//
// Bit-identical to the serial fill: each (r,c) writes only its own cell and its mirror, and
// distinct lower-triangle pairs have distinct mirrors, so no two workers touch a cell. Every
// element is still computed once by the same expression; only the order in which independent
// cells are filled changes. `elements` is immutable after construction and f2/c11_3sums are
// pure reads of v/so/eps.
func (mx *Matrix) FMatrix() backend.Mat {
	sp := mx.sp
	n := sp.BeginSat
	F := backend.NewMat(n, n)

	type cell struct{ r, c int }
	cells := make([]cell, 0, n*(n+1)/2)
	for r := range n {
		for c := 0; c <= r; c++ {
			cells = append(cells, cell{r, c})
		}
	}

	parallel.HeavyRows(len(cells), func(k int) {
		r, c := cells[k].r, cells[k].c
		i := sp.Configs[r].Occ[0]
		j := sp.Configs[c].Occ[0]
		val := mx.el.f2(i, j)
		if i == j {
			val += 1 // k1: identity
		}
		if mx.el.order >= 3 {
			_, fij, fji := mx.el.c11_3sums(i, j)
			val += (fij + fji) / 2
		}
		F.Set(r, c, val)
		if r != c {
			F.Set(c, r, val)
		}
	})
	return F
}
