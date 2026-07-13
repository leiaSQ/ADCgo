package selfenergy

import "math"

// coupling.go — the coupling amplitudes U_I(p) between orbital p and the satellite space,
// through third order. Ported from ../ADC/self_energy/constanti/common/aufbau1.f:
// KOPP1 (2nd order, lines 1-90) and KOPP2 (3rd order, lines 91-377).
//
// U is built for EVERY orbital p of the target irrep — occupied and virtual alike. The
// occupied/virtual distinction only enters later, in the resolvent: on the 2h1p block only the
// virtual p columns are iterated (the occupied ones keep their raw U, which INVQKL then consumes),
// and vice versa on 2p1h.
//
// Both KOPP routines are essentially IAB-blind: KOPP1 takes no IAB at all, and KOPP2's only
// dependence is a global sign ASIG(IAB) = +1 (2h1p) / −1 (2p1h). Everything else follows from
// which orbital set the configuration holds.

var (
	// SPIN(·,MS) — the intermediate-spin adaptation (aufbau1.f:33-34). Column MS=1 is S=0,
	// MS=2 is S=1.
	spinA = [2][2]float64{ // spinA[row][MS]
		{math.Sqrt(0.5), math.Sqrt(1.5)},
		{math.Sqrt(0.5), -math.Sqrt(1.5)},
	}
	// SPIN2(·,MS) (aufbau1.f:138-143).
	spinB = [4][2]float64{ // spinB[row][MS]
		{math.Sqrt(0.5), math.Sqrt(1.5)},
		{math.Sqrt(0.5), -math.Sqrt(1.5)},
		{-math.Sqrt(2.0), -2 * math.Sqrt(1.5)},
		{math.Sqrt(0.5), math.Sqrt(1.5)},
	}
)

// faktor is FAKTOR(MAXS): the configuration norm, 1/√2 when the two paired orbitals coincide.
func faktor(maxS int) float64 {
	if maxS == 1 {
		return math.Sqrt(0.5)
	}
	return 1
}

// asig is ASIG(IAB): +1 on the 2h1p block, −1 on 2p1h. It flips the sign of the entire
// third-order coupling.
func asig(b iab) float64 {
	if b == iab2h1p {
		return 1
	}
	return -1
}

// orbsOfSym lists the orbitals of one irrep, occupied first then virtual — the reference's
// NRP(PSYM,·) ordering, which is what U's column index refers to.
func (e *engine) orbsOfSym(sym int) []int {
	return append(append([]int{}, e.occs[sym]...), e.virs[sym]...)
}

// coupling returns U as a flat [dim × len(orbsOfSym(sym))] slice, row = spin-resolved
// configuration, column = position of p within the irrep.
func (e *engine) coupling(sp *satSpace) []float64 {
	orbs := e.orbsOfSym(sp.sym)
	u := make([]float64, sp.dim*len(orbs))
	e.kopp1(sp, orbs, u)
	e.kopp2(sp, orbs, u)
	return u
}

// kopp1 is the second-order coupling (aufbau1.f:38-87):
//
//	S=0: FKL·(⟨pj|kl⟩ + ⟨pj|lk⟩)/√2      S=1: FKL·√(3/2)·(⟨pj|kl⟩ − ⟨pj|lk⟩)
func (e *engine) kopp1(sp *satSpace, orbs []int, u []float64) {
	nc := len(orbs)
	for _, c := range sp.confs {
		fkl := faktor(c.maxS)
		for np, p := range orbs {
			add1 := e.v(p, c.j, c.k, c.l) * fkl
			add2 := e.v(p, c.j, c.l, c.k) * fkl
			for ms := range c.maxS {
				u[(c.off+ms)*nc+np] += add1*spinA[0][ms] + add2*spinA[1][ms]
			}
		}
	}
}

// kopp2 is the third-order correction (aufbau1.f:176-370), accumulated on top of KOPP1. Three
// contributions: a ladder over the complementary space, and two particle-hole terms (the second
// being the k↔l exchange partner, which vanishes when k == l).
//
// The reference carries its block offset in a loop variable (`NABMTX = IADD`, aufbau1.f:373) that
// sits outside the symmetry guards and can go stale if no guard fires for a block; we take the
// offset straight from the configuration (c.off), which is what it is supposed to be.
func (e *engine) kopp2(sp *satSpace, orbs []int, u []float64) {
	ep, nc := e.eps, len(orbs)
	sgn := asig(sp.blk)
	pair := e.pairSet(sp.blk)           // same space as (k,l)
	single := e.pairSet(sp.blk.other()) // same space as j

	for _, c := range sp.confs {
		j, k, l := c.j, c.k, c.l
		fkl := faktor(c.maxS)
		sumkl := 1.0 // SUMFAK: 2 when k == l
		if c.maxS == 1 {
			sumkl = 2
		}

		// --- Contribution 1: ladder over pairs (kk,ll) of the COMPLEMENTARY space.
		// Guard: sym(kk)⊗sym(ll) == sym(p)⊗sym(j).
		want := sp.sym ^ e.so(j)
		for llSym := range e.nsym {
			for kkSym := 0; kkSym <= llSym; kkSym++ {
				if kkSym^llSym != want {
					continue
				}
				for lli, ll := range single[llSym] {
					kkMax := len(single[kkSym])
					if kkSym == llSym {
						kkMax = lli + 1
					}
					for kki := 0; kki < kkMax; kki++ {
						kk := single[kkSym][kki]
						maxSS := 2
						if kk == ll {
							maxSS = 1
						}
						fkkll := faktor(maxSS) * faktor(maxSS) // ½ when kk == ll
						e1 := ep[kk] + ep[ll] - ep[k] - ep[l]
						a1 := e.v(k, l, kk, ll)
						a2 := e.v(k, l, ll, kk)
						upro := fkl * fkkll * sgn
						for np, p := range orbs {
							a3 := e.v(p, j, kk, ll)
							a4 := e.v(p, j, ll, kk)
							add1 := a1 * a3 * upro
							add2 := a1 * a4 * upro
							add3 := a2 * a3 * upro
							add4 := a2 * a4 * upro
							for ms := range c.maxS {
								if maxSS < ms+1 {
									continue
								}
								s := -add1*spinA[0][ms] - add2*spinA[1][ms] -
									add3*spinA[1][ms] - add4*spinA[0][ms]
								u[(c.off+ms)*nc+np] += s / e1
							}
						}
					}
				}
			}
		}

		// --- Contributions 2a and 2b: jj from the PAIR space, ll from the COMPLEMENTARY space.
		for jjSym := range e.nsym {
			for llSym := range e.nsym {
				prod := jjSym ^ llSym
				do2a := prod == sp.sym^e.so(k)
				do2b := prod == sp.sym^e.so(l) && c.maxS != 1
				if !do2a && !do2b {
					continue
				}
				for _, jj := range pair[jjSym] {
					for _, ll := range single[llSym] {
						if do2a {
							e1 := ep[j] + ep[ll] - ep[jj] - ep[l]
							a1 := e.v(j, ll, jj, l)
							a2 := e.v(j, ll, l, jj)
							opro := fkl * sgn
							for np, p := range orbs {
								a3 := e.v(p, jj, k, ll)
								a4 := e.v(p, jj, ll, k)
								add1 := a1 * a3 * opro
								add2 := a1 * a4 * opro
								add3 := a2 * a3 * opro
								add4 := a2 * a4 * opro
								for ms := range c.maxS {
									s := add1*spinB[0][ms] + add2*spinB[1][ms] +
										add3*spinB[2][ms] + add4*spinB[3][ms]
									u[(c.off+ms)*nc+np] += sumkl * s / e1
								}
							}
						}
						if do2b {
							// the k↔l exchange partner: k and l swap roles throughout
							e1 := ep[j] + ep[ll] - ep[jj] - ep[k]
							a1 := e.v(j, ll, jj, k)
							a2 := e.v(j, ll, k, jj)
							opri := fkl * sgn
							for np, p := range orbs {
								a3 := e.v(p, jj, l, ll)
								a4 := e.v(p, jj, ll, l)
								add1 := a1 * a3 * opri
								add2 := a1 * a4 * opri
								add3 := a2 * a3 * opri
								add4 := a2 * a4 * opri
								for ms := range c.maxS {
									sigsp := 1.0 // (−1)^(MS−1)
									if ms == 1 {
										sigsp = -1
									}
									s := sigsp * (add1*spinB[0][ms] + add2*spinB[1][ms] +
										add3*spinB[2][ms] + add4*spinB[3][ms])
									u[(c.off+ms)*nc+np] += s / e1
								}
							}
						}
					}
				}
			}
		}
	}
}
