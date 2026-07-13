package selfenergy

// dynamic3.go — the third-order dynamic self-energy M⁽³⁾_ak, evaluated at the HF poles.
// Ported from ../ADC/self_energy/original/original_self_energy.cpp dynamic_self_energy_3.
// Equations (C.12)-(C.23) of von Niessen, Schirmer & Cederbaum, Comput. Phys. Rep. 1, 57 (1984):
// M⁽³⁾⁺ is (C.12-C.14, C.18-C.20) (the C-terms) and M⁽³⁾⁻ is (C.15-C.17, C.21-C.23) (the D-terms).
//
// It feeds only ρ⁽³⁾'s hole/particle block (eq. A24). Returned as a flat nvir×nocc slice
// indexed [(p−nocc)*nocc + q].

func (e *engine) dynamicSelfEnergy3() []float64 {
	ep := e.eps
	mak := make([]float64, (e.norb-e.nocc)*e.nocc)

	for sym := range e.nsym {
		for _, p := range e.virs[sym] {
			for _, q := range e.occs[sym] {
				eP, eQ := ep[p], ep[q]
				var c1, c2, c3, c4, c5, c6 float64
				var d1, d2, d3, d4, d5, d6 float64

				// (C.12),(C.13)
				for i := range e.nocc {
					for a := e.nocc; a < e.norb; a++ {
						symB := e.so(p) ^ e.so(i) ^ e.so(a)
						for _, b := range e.virs[symB] {
							vPiab := (2*e.v(p, i, a, b) - e.v(p, i, b, a)) /
								(eQ + ep[i] - ep[a] - ep[b])
							for c := e.nocc; c < e.norb; c++ {
								symD := e.so(a) ^ symB ^ e.so(c)
								for _, d := range e.virs[symD] {
									c1 += vPiab * e.v(a, b, c, d) * e.v(q, i, c, d) /
										(eQ + ep[i] - ep[c] - ep[d])
								}
							}
							for j := range e.nocc {
								symK := e.so(a) ^ symB ^ e.so(j)
								for _, k := range e.occs[symK] {
									c2 += vPiab * e.v(a, b, j, k) * e.v(q, i, j, k) /
										(ep[j] + ep[k] - ep[a] - ep[b])
								}
							}
						}
					}
				}
				// (C.14)
				for i := range e.nocc {
					for a := e.nocc; a < e.norb; a++ {
						symB := e.so(p) ^ e.so(i) ^ e.so(a)
						for _, b := range e.virs[symB] {
							vQiab := e.v(q, i, a, b) / (eQ + ep[i] - ep[a] - ep[b])
							for k := range e.nocc {
								symJ := e.so(a) ^ symB ^ e.so(k)
								for _, j := range e.occs[symJ] {
									c3 += (2*e.v(p, i, j, k) - e.v(p, i, k, j)) *
										e.v(a, b, j, k) * vQiab /
										(ep[j] + ep[k] - ep[a] - ep[b])
								}
							}
						}
					}
				}
				// (C.15),(C.17)
				for i := range e.nocc {
					for j := range e.nocc {
						symA := e.so(p) ^ e.so(i) ^ e.so(j)
						for _, a := range e.virs[symA] {
							vPaij := (2*e.v(p, a, i, j) - e.v(p, a, j, i)) /
								(eP + ep[a] - ep[i] - ep[j])
							for b := e.nocc; b < e.norb; b++ {
								symC := e.so(q) ^ symA ^ e.so(b)
								for _, c := range e.virs[symC] {
									c4 += vPaij * e.v(i, j, b, c) * e.v(q, a, b, c) /
										(ep[i] + ep[j] - ep[b] - ep[c])
								}
							}
							for k := range e.nocc {
								symL := e.so(q) ^ symA ^ e.so(k)
								for _, l := range e.occs[symL] {
									c6 += vPaij * e.v(i, j, k, l) * e.v(q, a, k, l) /
										(eP + ep[a] - ep[k] - ep[l])
								}
							}
						}
					}
				}
				// (C.16)
				for i := range e.nocc {
					for j := range e.nocc {
						symA := e.so(q) ^ e.so(i) ^ e.so(j)
						for _, a := range e.virs[symA] {
							vQaij := e.v(q, a, i, j) / (eP + ep[a] - ep[i] - ep[j])
							for c := e.nocc; c < e.norb; c++ {
								symB := e.so(p) ^ symA ^ e.so(c)
								for _, b := range e.virs[symB] {
									c5 += (2*e.v(p, a, b, c) - e.v(p, a, c, b)) *
										e.v(i, j, b, c) * vQaij /
										(ep[i] + ep[j] - ep[b] - ep[c])
								}
							}
						}
					}
				}
				// (C.18),(C.20)
				for j := range e.nocc {
					for b := e.nocc; b < e.norb; b++ {
						symC := e.so(q) ^ e.so(j) ^ e.so(b)
						for _, c := range e.virs[symC] {
							pole := 1. / (eQ + ep[j] - ep[b] - ep[c])
							vQjcb := e.v(q, j, c, b) * pole
							vQjbc := e.v(q, j, b, c) * pole
							exp1 := vQjcb - 2*vQjbc
							exp2 := vQjbc - 2*vQjcb
							exp3 := -2 * exp1
							for i := range e.nocc {
								symA := e.so(j) ^ e.so(i) ^ symC
								for _, a := range e.virs[symA] {
									vAjic := e.v(a, j, i, c)
									vAjci := e.v(a, j, c, i)
									d1 += (e.v(p, i, a, b)*(vAjic*exp1+vAjci*exp2) +
										e.v(p, i, b, a)*(vAjic*exp3+vAjci*exp1)) /
										(eQ + ep[i] - ep[a] - ep[b])
									vIjac := e.v(i, j, a, c)
									vIjca := e.v(i, j, c, a)
									d3 += (e.v(p, a, i, b)*(vIjac*exp1+vIjca*exp2) +
										e.v(p, a, b, i)*(vIjac*exp3+vIjca*exp1)) /
										(ep[i] + ep[j] - ep[c] - ep[a])
								}
							}
						}
					}
				}
				// (C.19)
				for i := range e.nocc {
					for c := e.nocc; c < e.norb; c++ {
						symA := e.so(p) ^ e.so(i) ^ e.so(c)
						for _, a := range e.virs[symA] {
							pole := 1. / (eQ + ep[i] - ep[a] - ep[c])
							vPica := e.v(p, i, c, a) * pole
							vPiac := e.v(p, i, a, c) * pole
							for j := range e.nocc {
								symB := symA ^ e.so(i) ^ e.so(j)
								for _, b := range e.virs[symB] {
									vAbij := e.v(a, b, i, j)
									vAbji := e.v(a, b, j, i)
									vQbjc := e.v(q, b, j, c)
									vQbcj := e.v(q, b, c, j)
									x1 := vQbjc - 2*vQbcj
									x2 := vQbcj - 2*vQbjc
									x3 := -2 * x1
									d2 += (vPica*(vAbij*x3+vAbji*x1) +
										vPiac*(vAbij*x1+vAbji*x2)) /
										(ep[i] + ep[j] - ep[a] - ep[b])
								}
							}
						}
					}
				}
				// (C.21),(C.22),(C.23)
				for j := range e.nocc {
					for k := range e.nocc {
						symA := e.so(q) ^ e.so(j) ^ e.so(k)
						for _, a := range e.virs[symA] {
							pole := 1. / (eP + ep[a] - ep[j] - ep[k])
							vQajk := e.v(q, a, j, k) * pole
							vQakj := e.v(q, a, k, j) * pole
							exp1 := vQajk - 2*vQakj
							exp2 := vQakj - 2*vQajk
							exp3 := -2 * exp1
							vPakj := e.v(p, a, k, j) * pole
							vPajk := e.v(p, a, j, k) * pole
							for i := range e.nocc {
								symB := e.so(j) ^ e.so(i) ^ symA
								for _, b := range e.virs[symB] {
									vJiab := e.v(j, i, a, b)
									vJiba := e.v(j, i, b, a)
									d5 += (e.v(p, i, b, k)*(vJiab*exp1+vJiba*exp2) +
										e.v(p, i, k, b)*(vJiab*exp3+vJiba*exp1)) /
										(ep[i] + ep[j] - ep[a] - ep[b])
									vIabj := e.v(i, a, b, j)
									vIajb := e.v(i, a, j, b)
									d6 += (e.v(p, b, k, i)*(vIabj*exp3+vIajb*exp1) +
										e.v(p, b, i, k)*(vIabj*exp1+vIajb*exp2)) /
										(eP + ep[b] - ep[i] - ep[k])
									vQibk := e.v(q, i, b, k)
									vQikb := e.v(q, i, k, b)
									x4 := vQibk - 2*vQikb
									x5 := vQikb - 2*vQibk
									x6 := -2 * x4
									d4 += (vPakj*(vJiab*x6+vJiba*x4) +
										vPajk*(vJiab*x4+vJiba*x5)) /
										(ep[i] + ep[j] - ep[a] - ep[b])
								}
							}
						}
					}
				}

				mak[(p-e.nocc)*e.nocc+q] = (c1 + c2) + (c3 + c4) + (c5 - c6) +
					(d1 + d2) + (d3 + d4) + (d5 - d6)
			}
		}
	}
	return mak
}
