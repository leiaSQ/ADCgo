package selfenergy

import (
	"math"

	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// density2.go — the second-order correlation density ρ⁽²⁾ and the density→self-energy
// contraction. Ported from ../ADC/self_energy/original/original_self_energy.cpp
// (rho_hole_part_2 / rho_holeparticle_2 / rho_particle_part_2 / rho2sigma), equation numbers
// refer to Schirmer, Trofimov & Stelter, J. Chem. Phys. 109, 4734 (1998), Appendix A.
//
// ρ here is the *correction* only — the zeroth-order δ_ij n_i is never added, because the
// contraction below wants the correlation part alone.
//
// Every block here is parallel over its OUTER index pairs only. Each pair owns the ρ (or Σ)
// cells it writes, and its own inner Σ keeps the reference's summation order untouched, so the
// result is bit-identical to the serial walk — which matters: Σ enters the SIP main block
// directly and a perturbed value shifts every ionization line by ~0.2–0.35 eV without failing.
// All four use parallel.HeavyRows rather than parallel.Rows: the outer counts collapse with
// point-group symmetry, and Rows' "fewer than 2·GOMAXPROCS rows runs serially" fallback would
// then hand a multi-hour loop back to a single core.

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
	// The (k,k1) triangle is the work item: 1711 pairs at the production system's nocc=58, each summing
	// l·a·b = 58·154·154 ≈ 1.4e6 terms, so ~2.4e9 integral pairs that ran on one core.
	// The triangle visits every unordered pair once and irreps hold disjoint orbitals, so a
	// pair owns cells (k,k1) and (k1,k) outright; fkk's own accumulation order is unchanged.
	pairs := e.occPairs(true)
	parallel.HeavyRows(len(pairs), func(pi int) {
		k, k1 := pairs[pi][0], pairs[pi][1]
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
	})
}

// occPairs lists the occupied index pairs of every irrep, in the reference's enumeration order.
// tri keeps only k1 ≤ k (the (A15) triangle); otherwise the full square (A19) is returned.
func (e *engine) occPairs(tri bool) [][2]int {
	var pairs [][2]int
	for sym := range e.nsym {
		occ := e.occs[sym]
		for ki := range occ {
			n := len(occ)
			if tri {
				n = ki + 1
			}
			for k1i := 0; k1i < n; k1i++ {
				pairs = append(pairs, [2]int{occ[ki], occ[k1i]})
			}
		}
	}
	return pairs
}

// occVirPairs lists the (occupied, virtual) index pairs of every irrep, in the reference's
// enumeration order. Shared by the hole/particle blocks and by M⁽³⁾.
func (e *engine) occVirPairs() [][2]int {
	var pairs [][2]int
	for sym := range e.nsym {
		for _, k := range e.occs[sym] {
			for _, a := range e.virs[sym] {
				pairs = append(pairs, [2]int{k, a})
			}
		}
	}
	return pairs
}

// rhoHoleParticle2 is the hole/particle block, eqs. (A16)-(A18) and (A34).
func (e *engine) rhoHoleParticle2(rho *Sigma) {
	ep := e.eps
	// The (k,a) pairs are the work item: nocc·nvir = 58·154 = 8932 of them at production scale, each
	// running j·b·c + i·j·b ≈ 1.9e6 terms — ~1.7e10 in total, and this block runs under EVERY
	// scheme, Density() included. An (occupied, virtual) pair is enumerated once, so it owns
	// both (k,a) and (a,k); mp and mm keep their own summation order.
	pairs := e.occVirPairs()
	parallel.HeavyRows(len(pairs), func(pi int) {
		k, a := pairs[pi][0], pairs[pi][1]
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
	})
}

// rhoParticle2 is the particle/particle block, eqs. (A27) and (A36): build the 2h1p×virtual
// amplitude matrix f, then ρ_ab += (fᵀf)_ab. The spin adaptation of the |a k l> doublet
// eigenfunctions is what produces the 1/√2 and √(1/6) coefficients.
func (e *engine) rhoParticle2(rho *Sigma) {
	ep := e.eps
	for sym := range e.nsym {
		vir, sat := e.virs[sym], e.sats[sym]
		nv, ns := len(vir), len(sat)
		if nv == 0 || ns == 0 {
			continue
		}
		f := make([]float64, ns*nv) // row = satellite, col = virtual
		// One satellite row per work item: the production system's 2h1p list is ~5.1e5 configurations against
		// 154 virtuals, so ~8e7 amplitudes. Row si owns f[si*nv:(si+1)*nv] and nothing else.
		parallel.HeavyRows(ns, func(si int) {
			c := sat[si]
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
				f[si*nv+bi] = val
			}
		})
		// (A36): ρ_ba += Σ_s f(s,b) f(s,a).
		//
		// b is the work item, and it is the bigger half of this block: 154² = 23716 cells each
		// reducing over ~5.1e5 satellites is ~1.2e10 multiply-adds at production scale. Row b owns the
		// cells (b,·); the Σ_s inside each cell is left exactly as it was. nv = 154 is precisely
		// the case parallel.Rows would have run serially on a 96-core node.
		parallel.HeavyRows(nv, func(bi int) {
			b := vir[bi]
			for ai, a := range vir {
				var s float64
				for si := range sat {
					s += f[si*nv+bi] * f[si*nv+ai]
				}
				rho.set(b, a, rho.At(b, a)+s)
			}
		})
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
	// The (p,q) cells are the work item. In C1 this is a full norb⁴ contraction — 212⁴ = 2.0e9
	// inner evaluations at production scale, each two ERI lookups and a ρ read — and it sits on the
	// PRODUCTION Σ(∞) path (Static calls it on the all-order density) as well as on Four's.
	// Every (p,q) is enumerated once and writes its own sig cell with a private accumulator,
	// so the r,s reduction order is untouched.
	var cells [][2]int
	for symPQ := range e.nsym {
		for _, p := range orbs[symPQ] {
			for _, q := range orbs[symPQ] {
				cells = append(cells, [2]int{p, q})
			}
		}
	}
	parallel.HeavyRows(len(cells), func(ci int) {
		p, q := cells[ci][0], cells[ci][1]
		var sum float64
		for symRS := range e.nsym {
			for _, r := range orbs[symRS] {
				for _, s := range orbs[symRS] {
					sum += (2*e.v(p, r, q, s) - e.v(p, r, s, q)) * rho.At(r, s)
				}
			}
		}
		sig.set(p, q, sum)
	})
	return sig
}
