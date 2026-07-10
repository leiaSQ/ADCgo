package sip

import "github.com/leiaSQ/ADCgo/internal/adc/backend"

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
func (mx *Matrix) FMatrix() backend.Mat {
	sp := mx.sp
	n := sp.BeginSat
	F := backend.NewMat(n, n)
	for r := range n {
		i := sp.Configs[r].Occ[0]
		for c := 0; c <= r; c++ {
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
		}
	}
	return F
}
