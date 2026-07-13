package selfenergy

// density3.go — the third-order correlation density ρ⁽³⁾ and the third-order *dynamic*
// self-energy M⁽³⁾ its hole/particle block needs. Ported from
// ../ADC/self_energy/original/original_self_energy.cpp (rho_hole_part_3 /
// rho_holeparticle_3 / rho_particle_part_3 / dynamic_self_energy_3). Equation numbers are
// Schirmer, Trofimov & Stelter, JCP 109, 4734 (1998) App. A for the density and
// von Niessen, Schirmer & Cederbaum, Comput. Phys. Rep. 1, 57 (1984) App. C for M⁽³⁾.

// rho3 accumulates ρ⁽³⁾ into rho. mak is the ph intermediate: for Σ⁽⁴⁾ the caller passes
// M⁽³⁾ + Σ⁽³⁾_ak (static+dynamic); for Σ(4+) it passes the dynamic M⁽³⁾ alone, because there
// the static ph part is resummed by the linear equation instead (see static_selfenergy_4 vs
// static_selfenergy_4plus in the reference).
func (e *engine) rho3(rho *Sigma, mak []float64) {
	e.rhoHole3(rho)
	e.rhoHoleParticle3(rho, mak)
	e.rhoParticle3(rho)
}

// rhoHole3 is the hole/hole block, eqs. (A19)-(A23) and (A33). Note the reference runs the
// full (k,k1) square — not a triangle — and adds the result to both (k,k1) and (k1,k), so the
// block is symmetrised by construction and the diagonal picks up 2·f(k,k).
func (e *engine) rhoHole3(rho *Sigma) {
	ep := e.eps
	for sym := range e.nsym {
		occ := e.occs[sym]
		for _, k := range occ {
			for _, k1 := range occ {
				var fA, fB, fC, fD float64
				for a := e.nocc; a < e.norb; a++ {
					for m := range e.nocc {
						symC := e.so(a) ^ e.so(m) ^ e.so(k)
						for _, c := range e.virs[symC] {
							pole := 1. / ((ep[a] + ep[c] - ep[k1] - ep[m]) *
								(ep[a] + ep[c] - ep[k] - ep[m]))
							vAcK1m := e.v(a, c, k1, m) * pole
							vAcMk1 := e.v(a, c, m, k1) * pole
							exp1 := vAcMk1 - 2*vAcK1m
							exp2 := vAcK1m - 2*vAcMk1
							exp3 := -2 * exp1
							exp4 := -2 * exp2

							// (A20)
							for b := e.nocc; b < e.norb; b++ {
								symD := e.so(b) ^ e.so(k) ^ e.so(m)
								for _, d := range e.virs[symD] {
									fA += e.v(d, b, k, m) *
										(e.v(a, c, d, b)*exp1 + e.v(a, c, b, d)*exp2) /
										(ep[k] + ep[m] - ep[d] - ep[b])
								}
							}
							// (A21)
							for b := e.nocc; b < e.norb; b++ {
								symL := e.so(a) ^ e.so(b) ^ e.so(k)
								for _, l := range e.occs[symL] {
									vLcBm := e.v(l, c, b, m)
									vLcMb := e.v(l, c, m, b)
									fB += (e.v(a, b, k, l)*(vLcBm*exp3+vLcMb*exp1) +
										e.v(a, b, l, k)*(vLcBm*exp1+vLcMb*exp2)) /
										(ep[a] + ep[b] - ep[k] - ep[l])
								}
							}
							// (A22)
							for j := range e.nocc {
								symL := e.so(a) ^ e.so(c) ^ e.so(j)
								for _, l := range e.occs[symL] {
									fC += e.v(a, c, j, l) *
										(e.v(j, l, m, k)*exp2 + e.v(j, l, k, m)*exp1) /
										(ep[j] + ep[l] - ep[a] - ep[c])
								}
							}
							// (A23)
							for b := e.nocc; b < e.norb; b++ {
								symL := e.so(b) ^ e.so(a) ^ e.so(m)
								for _, l := range e.occs[symL] {
									vLcKb := e.v(l, c, k, b)
									vLcBk := e.v(l, c, b, k)
									fD += (e.v(b, a, l, m)*(vLcKb*exp2+vLcBk*exp4) +
										e.v(b, a, m, l)*(vLcKb*exp1+vLcBk*exp2)) /
										(ep[b] + ep[a] - ep[l] - ep[m])
								}
							}
						}
					}
				}
				// (A19) then (A33).
				fkk := 0.5*(fA+fC) + (fB + fD)
				rho.set(k, k1, rho.At(k, k1)+fkk)
				rho.set(k1, k, rho.At(k1, k)+fkk)
			}
		}
	}
}

// rhoHoleParticle3 is the hole/particle block, eq. (A24): ρ_ka += M_ak/(ε_k − ε_a).
// mak is indexed [(a−nocc)*nocc + k].
func (e *engine) rhoHoleParticle3(rho *Sigma, mak []float64) {
	ep := e.eps
	for sym := range e.nsym {
		for _, k := range e.occs[sym] {
			for _, a := range e.virs[sym] {
				fka := mak[(a-e.nocc)*e.nocc+k] / (ep[k] - ep[a])
				rho.set(k, a, rho.At(k, a)+fka)
				rho.set(a, k, rho.At(a, k)+fka)
			}
		}
	}
}

// rhoParticle3 is the particle/particle block, eqs. (A27)-(A32) and (A36):
// ρ_ba += (f1ᵀf2 + f2ᵀf1)_ba.
func (e *engine) rhoParticle3(rho *Sigma) {
	ep := e.eps
	for sym := range e.nsym {
		vir, sat := e.virs[sym], e.sats[sym]
		nv, ns := len(vir), len(sat)
		if nv == 0 || ns == 0 {
			continue
		}
		f1 := make([]float64, ns*nv)
		f2 := make([]float64, ns*nv)

		for si, c := range sat {
			a, k, l, typ := c.a, c.k, c.l, c.typ
			for bi, b := range vir {
				// (A27) — note the k==l and singlet branches use the (ε_k+ε_l−ε_a−ε_b)
				// denominator while the triplet branch uses its negative.
				dOcc := ep[k] + ep[l] - ep[a] - ep[b]
				dVir := ep[a] + ep[b] - ep[k] - ep[l]
				switch {
				case k == l:
					f1[si*nv+bi] = e.v(a, b, l, k) / dOcc
				case typ == 0:
					f1[si*nv+bi] = sqrt1_2 * (e.v(a, b, l, k) + e.v(a, b, k, l)) / dOcc
				default:
					f1[si*nv+bi] = sqrt3_2 * (e.v(a, b, l, k) - e.v(a, b, k, l)) / dVir
				}

				var gA, gB, gC, gD, gC1, gD1 float64
				switch {
				case k == l:
					for i := range e.nocc { // (A29)
						symJ := e.so(a) ^ e.so(b) ^ e.so(i)
						for _, j := range e.occs[symJ] {
							gA += e.v(a, b, i, j) * e.v(i, j, l, k) /
								(ep[a] + ep[b] - ep[i] - ep[j])
						}
					}
					for cc := e.nocc; cc < e.norb; cc++ { // (A30)
						symD := e.so(a) ^ e.so(b) ^ e.so(cc)
						for _, d := range e.virs[symD] {
							gB += e.v(cc, d, l, k) * e.v(a, b, cc, d) /
								(ep[cc] + ep[d] - ep[k] - ep[l])
						}
					}
					for j := range e.nocc { // (A31,A32)
						symC := e.so(b) ^ e.so(j) ^ e.so(l)
						for _, cc := range e.virs[symC] {
							gC += e.v(b, cc, j, l) * e.v(a, j, cc, k) /
								(ep[j] + ep[l] - ep[b] - ep[cc])
							gD += e.v(a, cc, j, k) * e.v(b, j, cc, l) /
								(ep[j] + ep[k] - ep[a] - ep[cc])
						}
					}
					for j := range e.nocc { // (A31,A32)
						symC := e.so(a) ^ e.so(j) ^ e.so(l)
						for _, cc := range e.virs[symC] {
							vBcKj := e.v(b, cc, k, j)
							gC1 += ((2*vBcKj-e.v(b, cc, j, k))*e.v(a, j, l, cc) -
								vBcKj*e.v(a, j, cc, l)) /
								(ep[b] + ep[cc] - ep[j] - ep[k])
							vBjKc := e.v(b, j, k, cc)
							gD1 += (e.v(a, cc, l, j)*(2*vBjKc-e.v(b, j, cc, k)) -
								e.v(a, cc, j, l)*vBjKc) /
								(ep[a] + ep[cc] - ep[j] - ep[l])
						}
					}

				case typ == 0:
					for i := range e.nocc { // (A29)
						symJ := e.so(a) ^ e.so(b) ^ e.so(i)
						for _, j := range e.occs[symJ] {
							gA += e.v(a, b, i, j) * (e.v(i, j, l, k) + e.v(i, j, k, l)) /
								(ep[a] + ep[b] - ep[i] - ep[j])
						}
					}
					for cc := e.nocc; cc < e.norb; cc++ { // (A30)
						symD := e.so(a) ^ e.so(b) ^ e.so(cc)
						for _, d := range e.virs[symD] {
							gB += e.v(a, b, cc, d) * (e.v(cc, d, l, k) + e.v(cc, d, k, l)) /
								(ep[cc] + ep[d] - ep[k] - ep[l])
						}
					}
					for j := range e.nocc { // (A31,A32)
						symC := e.so(b) ^ e.so(j) ^ e.so(l)
						for _, cc := range e.virs[symC] {
							vAjKc := e.v(a, j, k, cc)
							vAjCk := e.v(a, j, cc, k)
							gC += (e.v(b, cc, l, j)*(2*vAjKc-vAjCk) -
								e.v(b, cc, j, l)*(vAjCk+vAjKc)) /
								(ep[b] + ep[cc] - ep[j] - ep[l])
							vAcKj := e.v(a, cc, k, j)
							vAcJk := e.v(a, cc, j, k)
							gD += ((2*vAcKj-vAcJk)*e.v(b, j, l, cc) -
								(vAcJk+vAcKj)*e.v(b, j, cc, l)) /
								(ep[a] + ep[cc] - ep[j] - ep[k])
						}
					}
					for j := range e.nocc { // (A31,A32)
						symC := e.so(a) ^ e.so(j) ^ e.so(l)
						for _, cc := range e.virs[symC] {
							vBcKj := e.v(b, cc, k, j)
							vBcJk := e.v(b, cc, j, k)
							gC1 += ((2*vBcKj-vBcJk)*e.v(a, j, l, cc) -
								(vBcKj+vBcJk)*e.v(a, j, cc, l)) /
								(ep[b] + ep[cc] - ep[j] - ep[k])
							vAcLj := e.v(a, cc, l, j)
							vAcJl := e.v(a, cc, j, l)
							gD1 += ((2*vAcLj-vAcJl)*e.v(b, j, k, cc) -
								(vAcJl+vAcLj)*e.v(b, j, cc, k)) /
								(ep[a] + ep[cc] - ep[j] - ep[l])
						}
					}
					gA *= sqrt1_2
					gB *= sqrt1_2
					gC *= sqrt1_2
					gD *= sqrt1_2
					gC1 *= sqrt1_2
					gD1 *= sqrt1_2

				default: // triplet coupling
					for i := range e.nocc { // (A29)
						symJ := e.so(a) ^ e.so(b) ^ e.so(i)
						for _, j := range e.occs[symJ] {
							gA += e.v(a, b, i, j) * (e.v(i, j, k, l) - e.v(i, j, l, k)) /
								(ep[a] + ep[b] - ep[i] - ep[j])
						}
					}
					for cc := e.nocc; cc < e.norb; cc++ { // (A30)
						symD := e.so(a) ^ e.so(b) ^ e.so(cc)
						for _, d := range e.virs[symD] {
							gB += (e.v(cc, d, k, l) - e.v(cc, d, l, k)) * e.v(a, b, cc, d) /
								(ep[cc] + ep[d] - ep[k] - ep[l])
						}
					}
					for j := range e.nocc { // (A31,A32)
						symC := e.so(b) ^ e.so(j) ^ e.so(l)
						for _, cc := range e.virs[symC] {
							vAjCk := e.v(a, j, cc, k)
							vAjKc := e.v(a, j, k, cc)
							gC += (e.v(b, cc, j, l)*(vAjCk-vAjKc) +
								e.v(b, cc, l, j)*(2*vAjKc-vAjCk)) /
								(ep[b] + ep[cc] - ep[j] - ep[l])
							vAcJk := e.v(a, cc, j, k)
							vAcKj := e.v(a, cc, k, j)
							gD += ((vAcJk-vAcKj)*e.v(b, j, cc, l) +
								(2*vAcKj-vAcJk)*e.v(b, j, l, cc)) /
								(ep[a] + ep[cc] - ep[j] - ep[k])
						}
					}
					for j := range e.nocc { // (A31,A32)
						symC := e.so(a) ^ e.so(j) ^ e.so(l)
						for _, cc := range e.virs[symC] {
							vAjCl := e.v(a, j, cc, l)
							vAjLc := e.v(a, j, l, cc)
							gC1 += (e.v(b, cc, j, k)*(vAjCl-vAjLc) +
								(2*vAjLc-vAjCl)*e.v(b, cc, k, j)) /
								(ep[j] + ep[k] - ep[b] - ep[cc])
							vAcJl := e.v(a, cc, j, l)
							vAcLj := e.v(a, cc, l, j)
							gD1 += ((vAcJl-vAcLj)*e.v(b, j, cc, k) +
								(2*vAcLj-vAcJl)*e.v(b, j, k, cc)) /
								(ep[j] + ep[l] - ep[a] - ep[cc])
						}
					}
					gA *= sqrt3_2
					gB *= sqrt3_2
					gC *= sqrt3_2
					gD *= sqrt3_2
					gC1 *= sqrt3_2
					gD1 *= sqrt3_2
				}

				f2[si*nv+bi] = (gA + gC + gB + gD + gC1 + gD1) / dVir
			}
		}

		// (A36): ρ_ba += (f1ᵀf2 + f2ᵀf1)_ba.
		for bi, b := range vir {
			for ai, a := range vir {
				var s float64
				for si := range sat {
					s += f1[si*nv+bi]*f2[si*nv+ai] + f2[si*nv+bi]*f1[si*nv+ai]
				}
				rho.set(b, a, rho.At(b, a)+s)
			}
		}
	}
}
