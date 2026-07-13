package selfenergy

import "math"

// density2.go — the second-order correlation density ρ⁽²⁾ and the density→self-energy
// contraction. Ported from ../ADC/self_energy/original/original_self_energy.cpp
// (rho_hole_part_2 / rho_holeparticle_2 / rho_particle_part_2 / rho2sigma), equation numbers
// refer to Schirmer, Trofimov & Stelter, J. Chem. Phys. 109, 4734 (1998), Appendix A.
//
// ρ here is the *correction* only — the zeroth-order δ_ij n_i is never added, because the
// contraction below wants the correlation part alone.

var (
	sqrt1_2 = math.Sqrt(0.5)
	sqrt1_6 = math.Sqrt(1.0 / 6.0)
	sqrt3_2 = math.Sqrt(1.5)
)

// rho2 accumulates ρ⁽²⁾ (hole/hole, hole/particle, particle/particle) into rho.
func (e *engine) rho2(rho *Sigma) {
	e.rhoHole2(rho)
	e.rhoHoleParticle2(rho)
	e.rhoParticle2(rho)
}

// rhoHole2 is the hole/hole block, eqs. (A15) and (A33).
func (e *engine) rhoHole2(rho *Sigma) {
	ep := e.eps
	for sym := range e.nsym {
		occ := e.occs[sym]
		for ki := range occ {
			for k1i := 0; k1i <= ki; k1i++ {
				k, k1 := occ[ki], occ[k1i]
				var fkk float64
				for l := range e.nocc {
					for a := e.nocc; a < e.norb; a++ {
						symB := e.so(a) ^ e.so(l) ^ e.so(k)
						for _, b := range e.virs[symB] {
							num := 2*e.v(a, b, k, l)*e.v(a, b, k1, l) -
								e.v(a, b, l, k)*e.v(a, b, k1, l)
							d1 := ep[a] + ep[b] - ep[k] - ep[l]
							d2 := ep[a] + ep[b] - ep[k1] - ep[l]
							fkk += num / (d1 * d2)
						}
					}
				}
				rho.set(k, k1, rho.At(k, k1)-fkk)
				if k != k1 {
					rho.set(k1, k, rho.At(k1, k)-fkk)
				}
			}
		}
	}
}

// rhoHoleParticle2 is the hole/particle block, eqs. (A16)-(A18) and (A34).
func (e *engine) rhoHoleParticle2(rho *Sigma) {
	ep := e.eps
	for sym := range e.nsym {
		for _, k := range e.occs[sym] {
			for _, a := range e.virs[sym] {
				// (A17): the 2p1h ("+") branch.
				var mp float64
				for j := range e.nocc {
					for b := e.nocc; b < e.norb; b++ {
						symC := e.so(a) ^ e.so(j) ^ e.so(b)
						for _, c := range e.virs[symC] {
							num := 2*e.v(a, j, b, c)*e.v(b, c, k, j) -
								e.v(a, j, c, b)*e.v(b, c, k, j)
							mp += num / (ep[b] + ep[c] - ep[k] - ep[j])
						}
					}
				}
				// (A18): the 2h1p ("−") branch.
				var mm float64
				for i := range e.nocc {
					for j := range e.nocc {
						symB := e.so(i) ^ e.so(j) ^ e.so(k)
						for _, b := range e.virs[symB] {
							num := 2*e.v(i, j, k, b)*e.v(a, b, i, j) -
								e.v(i, j, b, k)*e.v(a, b, i, j)
							mm += num / (ep[a] + ep[b] - ep[i] - ep[j])
						}
					}
				}
				// (A16) then (A34).
				fka := (mm - mp) / (ep[k] - ep[a])
				rho.set(k, a, rho.At(k, a)+fka)
				rho.set(a, k, rho.At(a, k)+fka)
			}
		}
	}
}

// rhoParticle2 is the particle/particle block, eqs. (A27) and (A36): build the 2h1p×virtual
// amplitude matrix f, then ρ_ab += (fᵀf)_ab. The spin adaptation of the |a k l> doublet
// eigenfunctions is what produces the 1/√2 and √(1/6) coefficients.
func (e *engine) rhoParticle2(rho *Sigma) {
	ep := e.eps
	for sym := range e.nsym {
		vir, sat := e.virs[sym], e.sats[sym]
		if len(vir) == 0 || len(sat) == 0 {
			continue
		}
		f := make([]float64, len(sat)*len(vir)) // row = satellite, col = virtual
		for si, c := range sat {
			for bi, b := range vir {
				d := ep[c.a] + ep[b] - ep[c.k] - ep[c.l]
				var val float64
				switch {
				case c.k == c.l:
					val = -e.v(c.a, b, c.l, c.k) / d
				case c.typ == 0:
					val = -sqrt1_2 * (e.v(c.a, b, c.l, c.k) + e.v(c.a, b, c.k, c.l)) / d
				default:
					val = sqrt1_6 * (3*e.v(c.a, b, c.l, c.k) - 3*e.v(c.a, b, c.k, c.l)) / d
				}
				f[si*len(vir)+bi] = val
			}
		}
		// (A36): ρ_ba += Σ_s f(s,b) f(s,a).
		for bi, b := range vir {
			for ai, a := range vir {
				var s float64
				for si := range sat {
					s += f[si*len(vir)+bi] * f[si*len(vir)+ai]
				}
				rho.set(b, a, rho.At(b, a)+s)
			}
		}
	}
}

// rhoToSigma is eq. (A25): Σ_pq = Σ_rs (2<pr|qs> − <pr|sq>) ρ_rs, with p,q in one irrep and
// r,s in one irrep (ρ is irrep-diagonal). Takes the density *correction* and returns Σ.
func (e *engine) rhoToSigma(rho *Sigma) *Sigma {
	sig := newSigma(e.norb)
	// All orbitals of an irrep, occupied then virtual, as theADCcode's orbs[] does.
	orbs := make([][]int, e.nsym)
	for s := range e.nsym {
		orbs[s] = append(append([]int{}, e.occs[s]...), e.virs[s]...)
	}
	for symPQ := range e.nsym {
		for _, p := range orbs[symPQ] {
			for _, q := range orbs[symPQ] {
				var sum float64
				for symRS := range e.nsym {
					for _, r := range orbs[symRS] {
						for _, s := range orbs[symRS] {
							sum += (2*e.v(p, r, q, s) - e.v(p, r, s, q)) * rho.At(r, s)
						}
					}
				}
				sig.set(p, q, sum)
			}
		}
	}
	return sig
}
