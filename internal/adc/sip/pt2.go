package sip

import "github.com/leiaSQ/ADCgo/internal/adc/parallel"

// pt2.go — the second-order perturbation-theory expressions of Appendix A4-A14,
// evaluated in the spin-orbital basis.
//
// These are the blocks Slater-Condon cannot supply: at second order the
// intermediate states differ from the HF configurations, so the elements are
// genuine PT expressions rather than Hamiltonian matrix elements.
//
// A4 (1h/1h) and A6 (1h/2h1p) are implemented here even though elements.go
// already has them as spin-free formulas (c11_2, c12_2, both bit-exact against
// theADCcode). They are the GATE: they share every ingredient with the new
// second-order 2h1p/2h1p block A9-A14 — the v amplitudes of A1, the antisymmetrized
// "1212" integrals, the 1/2 prefactors, the epsilon combinations, and A6 even
// carries the same [k <-> l] permutation operator as A10 and A13. Reproducing
// both from this file establishes that the transcription conventions are right
// before they are relied on where no reference exists.

// epsOf is the HF orbital energy of a spin orbital.
func (e *elements) epsOf(s sorb) float64 { return e.eps[s.Orb] }

// vAmp is the amplitude v_abij of Eq. (A1): the antisymmetrized "1212" integral
// over the zeroth-order energy denominator,
//
//	v_abij = <ab||ij> / (eps_a + eps_b - eps_i - eps_j).
func (e *elements) vAmp(a, b, i, j sorb) float64 {
	num := e.eriSO(a, b, i, j)
	if num == 0 {
		return 0
	}
	return num / (e.epsOf(a) + e.epsOf(b) - e.epsOf(i) - e.epsOf(j))
}

// occSO and virSO iterate the occupied and virtual spin orbitals. Kept as
// closures rather than materialized slices because they sit in the innermost
// loops of the second-order sums.
func (e *elements) occSO(f func(sorb)) {
	for o := range e.nocc {
		f(sorb{Orb: o})
		f(sorb{Orb: o, Dn: true})
	}
}

func (e *elements) virSO(f func(sorb)) {
	for o := e.nocc; o < e.norb; o++ {
		f(sorb{Orb: o})
		f(sorb{Orb: o, Dn: true})
	}
}

// m2_1h is Eq. (A4), the second-order 1h/1h element:
//
//	M^(2)_kk' = 1/2 sum_{a,b,j} v_abkj v*_abk'j ( eps_a + eps_b - eps_j
//	                                              - 1/2 (eps_k + eps_k') )
func (e *elements) m2_1h(k, kp sorb) float64 {
	half := 0.5 * (e.epsOf(k) + e.epsOf(kp))
	var sum float64
	e.virSO(func(a sorb) {
		e.virSO(func(b sorb) {
			e.occSO(func(j sorb) {
				v1 := e.vAmp(a, b, k, j)
				if v1 == 0 {
					return
				}
				v2 := e.vAmp(a, b, kp, j)
				if v2 == 0 {
					return
				}
				sum += v1 * v2 * (e.epsOf(a) + e.epsOf(b) - e.epsOf(j) - half)
			})
		})
	})
	return 0.5 * sum
}

// m2_1h2h1p is Eq. (A6), the second-order 1h/2h1p element for one PRIMITIVE
// spin-orbital 2h1p configuration (particle a, holes k < l) and 1h spin orbital i:
//
//	M^(2)_{i,akl} = 1/2 sum_{b,c} v*_bckl V_bc[ai]
//	                - [ sum_{b,j} v*_ablj V_kb[ij] ] + [k <-> l]
//
// Reading of the [k <-> l] operator. It ANTISYMMETRIZES the bracketed group,
// -B(k,l) + B(l,k); the leading term is left alone, being already antisymmetric in
// k and l (v_bclk = -v_bckl), so permuting it would either cancel or double it.
// That reading is not a guess: TestA8Notation settles it on Eq. (A8), whose value
// is independently known from Slater-Condon, by trying every candidate reading and
// finding this the only one that reproduces it — including in the cross-pairing
// cases (l == k') that are the whole reason the permutation terms exist.
// TestPT2GateC12_2 then confirms it here against c12_2.
//
// A10 and A13 need no such inference: they print all four signs of the double
// antisymmetrizer (1 - P_kl)(1 - P_k'l') explicitly.
func (e *elements) m2_1h2h1p(i sorb, a, k, l sorb) float64 {
	var s1 float64
	e.virSO(func(b sorb) {
		e.virSO(func(c sorb) {
			v := e.vAmp(b, c, k, l)
			if v == 0 {
				return
			}
			s1 += v * e.eriSO(b, c, a, i)
		})
	})
	// B(k,l) = sum_{b,j} v*_ablj V_kb[ij], antisymmetrized over k <-> l.
	bterm := func(k, l sorb) float64 {
		var s float64
		e.virSO(func(b sorb) {
			e.occSO(func(j sorb) {
				v := e.vAmp(a, b, l, j)
				if v == 0 {
					return
				}
				s += v * e.eriSO(k, b, i, j)
			})
		})
		return s
	}
	return 0.5*s1 - bterm(k, l) + bterm(l, k)
}

// ---------------------------------------------------------------------------
// A9-A14: the second-order 2h1p/2h1p block.
// ---------------------------------------------------------------------------
//
// This is the one block of ADC(2,2) that is neither a Slater-Condon element nor
// already present in non-Dyson ADC(3) — the paper singles it out as exactly that.
// It is the sum of five terms (A9),
//
//	M^(2) = M^(A) + M^(B) + M^(C) + M^(D) + M^(E),
//
// of which M^(A) (A10) and M^(D) (A13) carry the double antisymmetrizer
// (1 - P_kl)(1 - P_k'l'); the paper prints all four of its signs, so unlike A6
// and A8 there is nothing to infer. M^(B), M^(C) and M^(E) carry no permutation:
// per the note following A14, the k < l and k' < l' restriction on permissible
// intermediate states removes them.
//
// Every free index here is a spin orbital: a, a' virtual; k, l, k', l' occupied
// with k < l and k' < l'.

// antisym4 applies (1 - P_kl)(1 - P_k'l') to a function of the four hole indices.
func antisym4(k, l, kp, lp sorb, f func(k, l, kp, lp sorb) float64) float64 {
	return f(k, l, kp, lp) - f(l, k, kp, lp) - f(k, l, lp, kp) + f(l, k, lp, kp)
}

// m2A is Eq. (A10):
//
//	[ 1/2 d_aa' d_kk' sum_{b,c,j} v_bclj v*_bcl'j
//	      ( eps_b + eps_c - eps_j - 1/2 (eps_l + eps_l') ) ]
//	+ [k<->l, k'<->l'] - [k<->l] - [k'<->l']
//
// Once its two deltas are imposed the inner sum depends on nothing but the hole
// pair (l,l'), so it is a table (tblA, built by buildPT2Tables) and not a
// per-element sum. That is the difference between O(n_virt²n_occ) and O(1) per
// matrix element, over a block with n2² of them.
func (e *elements) m2A(a, k, l, ap, kp, lp sorb) float64 {
	if a != ap {
		return 0
	}
	e.pt2.Do(e.buildPT2Tables)
	no := 2 * e.nocc
	return antisym4(k, l, kp, lp, func(k, l, kp, lp sorb) float64 {
		if k != kp {
			return 0
		}
		return e.tblA[l.Idx()*no+lp.Idx()]
	})
}

// m2AInner is the literal inner sum of Eq. (A10) for one hole pair, as written. It
// is the oracle tblA is gated against (TestPT2TablesMatchDirectSums) — the table is
// accumulated in a different loop order, so this is what pins that the
// restructuring is algebra and not a change of formula.
func (e *elements) m2AInner(l, lp sorb) float64 {
	half := 0.5 * (e.epsOf(l) + e.epsOf(lp))
	var sum float64
	e.virSO(func(b sorb) {
		e.virSO(func(c sorb) {
			e.occSO(func(j sorb) {
				v1 := e.vAmp(b, c, l, j)
				if v1 == 0 {
					return
				}
				v2 := e.vAmp(b, c, lp, j)
				if v2 == 0 {
					return
				}
				sum += v1 * v2 * (e.epsOf(b) + e.epsOf(c) - e.epsOf(j) - half)
			})
		})
	})
	return 0.5 * sum
}

// m2B is Eq. (A11):
//
//	1/2 d_kk' d_ll' sum_{c,i,j} v_acij v*_a'cij
//	    ( eps_c - eps_i - eps_j + 1/2 (eps_a + eps_a') )
//
// Its sum depends only on the particle pair (a,a'), so it too is a table (tblB).
func (e *elements) m2B(a, k, l, ap, kp, lp sorb) float64 {
	if k != kp || l != lp {
		return 0
	}
	e.pt2.Do(e.buildPT2Tables)
	nv := 2 * (e.norb - e.nocc)
	off := 2 * e.nocc
	return e.tblB[(a.Idx()-off)*nv+(ap.Idx()-off)]
}

// m2BInner is the literal sum of Eq. (A11) for one particle pair — the oracle for
// tblB, as m2AInner is for tblA.
func (e *elements) m2BInner(a, ap sorb) float64 {
	half := 0.5 * (e.epsOf(a) + e.epsOf(ap))
	var sum float64
	e.virSO(func(c sorb) {
		e.occSO(func(i sorb) {
			e.occSO(func(j sorb) {
				v1 := e.vAmp(a, c, i, j)
				if v1 == 0 {
					return
				}
				v2 := e.vAmp(ap, c, i, j)
				if v2 == 0 {
					return
				}
				sum += v1 * v2 * (e.epsOf(c) - e.epsOf(i) - e.epsOf(j) + half)
			})
		})
	})
	return 0.5 * sum
}

// ---------------------------------------------------------------------------
// The A10/A11 tables.
// ---------------------------------------------------------------------------

// soLists returns the occupied and virtual spin orbitals in Idx() order, so that
// occSO[x].Idx() == x and virSO[x].Idx() == 2*nocc + x. That is the indexing the
// tables use.
func (e *elements) soLists() (occ, vir []sorb) {
	occ = make([]sorb, 2*e.nocc)
	for o := range e.nocc {
		occ[2*o] = sorb{Orb: o}
		occ[2*o+1] = sorb{Orb: o, Dn: true}
	}
	nv := e.norb - e.nocc
	vir = make([]sorb, 2*nv)
	for x := range vir {
		vir[x] = sorb{Orb: e.nocc + x/2, Dn: x%2 == 1}
	}
	return occ, vir
}

// buildPT2Tables fills tblA and tblB, once per elements (guarded by e.pt2).
//
// Why a table at all. Evaluated literally, A10 costs O(n_virt²n_occ) and A11
// O(n_virtn_occ²) integral-weighted terms PER matrix element, and the second-order
// 2h1p/2h1p block has n2² elements — for the production sector that is ~5e7 terms
// per element over ~2.7e11 elements, which is the whole assembly budget spent
// twice over on two sums that do not depend on the element. Tabulated, both become
// one load; A10's table is (2n_occ)² and A11's (2n_virt)², under a megabyte each.
//
// Structure. Both sums are outer products once the shared index triple is fixed:
//
//	A10:  S(l,l') = sum_{b,c,j} u_l u_l'      with u_l  = v_bclj
//	A11:  T(a,a') = sum_{c,i,j} u_a u_a'      with u_a  = v_acij
//
// so each is accumulated by gathering u once per (b,c,j) resp. (c,i,j) and adding
// its outer product. Gathering first is what makes this affordable: it costs one
// vAmp per table index instead of one per table ENTRY, i.e. 2n_occ fewer integral
// lookups per triple, and those lookups are scattered reads of the n_orb⁴ tensor.
//
// The epsilon weights that depend on the summation indices are carried in a second
// accumulator; the ones that depend only on the table keys (the 1/2(eps_l+eps_l')
// and 1/2(eps_a+eps_a') halves) are applied after the reduction, where they cost
// nothing.
//
// Determinism. Parallelized over the outermost summation index with private
// per-worker accumulators reduced in fixed worker order, so the tables are
// bit-reproducible for a given worker count. The transient cost is
// W·(2n_virt)²·2·8 bytes for A11 — ~100 MB at production width, freed on return —
// which buys a reduction that needs no atomics and no locks.
func (e *elements) buildPT2Tables() {
	occ, vir := e.soLists()
	no, nv := len(occ), len(vir)

	// --- A10: keyed by the hole pair (l,l'). ---
	a0 := make([]float64, no*no) // sum of u_l u_l'
	a1 := make([]float64, no*no) // the same, weighted by (eps_b + eps_c - eps_j)
	if nv > 0 {
		W := parallel.ChunkWorkers(nv)
		part := make([]float64, 2*W*no*no)
		parallel.Chunks(nv, W, func(w, b0, b1 int) {
			s0 := part[2*w*no*no : (2*w+1)*no*no]
			s1 := part[(2*w+1)*no*no : (2*w+2)*no*no]
			u := make([]float64, no)
			for bi := b0; bi < b1; bi++ {
				b := vir[bi]
				for _, c := range vir {
					ebc := e.epsOf(b) + e.epsOf(c)
					for _, j := range occ {
						wt := ebc - e.epsOf(j)
						any := false
						for li, l := range occ {
							u[li] = e.vAmp(b, c, l, j)
							if u[li] != 0 {
								any = true
							}
						}
						if !any {
							continue
						}
						for li := range no {
							ul := u[li]
							if ul == 0 {
								continue
							}
							r0, r1 := s0[li*no:li*no+no], s1[li*no:li*no+no]
							for lp := range no {
								p := ul * u[lp]
								r0[lp] += p
								r1[lp] += p * wt
							}
						}
					}
				}
			}
		})
		for w := 0; w < W; w++ {
			s0 := part[2*w*no*no : (2*w+1)*no*no]
			s1 := part[(2*w+1)*no*no : (2*w+2)*no*no]
			for x := range a0 {
				a0[x] += s0[x]
				a1[x] += s1[x]
			}
		}
	}
	e.tblA = make([]float64, no*no)
	for li, l := range occ {
		for lp, lpp := range occ {
			half := 0.5 * (e.epsOf(l) + e.epsOf(lpp))
			x := li*no + lp
			e.tblA[x] = 0.5 * (a1[x] - half*a0[x])
		}
	}

	// --- A11: keyed by the particle pair (a,a'). ---
	b0acc := make([]float64, nv*nv) // sum of u_a u_a'
	b1acc := make([]float64, nv*nv) // the same, weighted by (eps_c - eps_i - eps_j)
	if nv > 0 && no > 0 {
		W := parallel.ChunkWorkers(nv)
		part := make([]float64, 2*W*nv*nv)
		parallel.Chunks(nv, W, func(w, c0, c1 int) {
			t0 := part[2*w*nv*nv : (2*w+1)*nv*nv]
			t1 := part[(2*w+1)*nv*nv : (2*w+2)*nv*nv]
			u := make([]float64, nv)
			for ci := c0; ci < c1; ci++ {
				c := vir[ci]
				for _, i := range occ {
					for _, j := range occ {
						wt := e.epsOf(c) - e.epsOf(i) - e.epsOf(j)
						any := false
						for ai, a := range vir {
							u[ai] = e.vAmp(a, c, i, j)
							if u[ai] != 0 {
								any = true
							}
						}
						if !any {
							continue
						}
						for ai := range nv {
							ua := u[ai]
							if ua == 0 {
								continue
							}
							r0, r1 := t0[ai*nv:ai*nv+nv], t1[ai*nv:ai*nv+nv]
							for apx := range nv {
								p := ua * u[apx]
								r0[apx] += p
								r1[apx] += p * wt
							}
						}
					}
				}
			}
		})
		for w := 0; w < W; w++ {
			t0 := part[2*w*nv*nv : (2*w+1)*nv*nv]
			t1 := part[(2*w+1)*nv*nv : (2*w+2)*nv*nv]
			for x := range b0acc {
				b0acc[x] += t0[x]
				b1acc[x] += t1[x]
			}
		}
	}
	e.tblB = make([]float64, nv*nv)
	for ai, a := range vir {
		for apx, ap := range vir {
			half := 0.5 * (e.epsOf(a) + e.epsOf(ap))
			x := ai*nv + apx
			e.tblB[x] = 0.5 * (b1acc[x] + half*b0acc[x])
		}
	}
}

// m2C is Eq. (A12):
//
//	-1/2 d_aa' sum_{b,c} v_bckl v*_bck'l'
//	     ( eps_b + eps_c - 1/2 (eps_k + eps_k' + eps_l + eps_l') )
func (e *elements) m2C(a, k, l, ap, kp, lp sorb) float64 {
	if a != ap {
		return 0
	}
	half := 0.5 * (e.epsOf(k) + e.epsOf(kp) + e.epsOf(l) + e.epsOf(lp))
	var sum float64
	e.virSO(func(b sorb) {
		e.virSO(func(c sorb) {
			v1 := e.vAmp(b, c, k, l)
			if v1 == 0 {
				return
			}
			v2 := e.vAmp(b, c, kp, lp)
			if v2 == 0 {
				return
			}
			sum += v1 * v2 * (e.epsOf(b) + e.epsOf(c) - half)
		})
	})
	return -0.5 * sum
}

// m2D is Eq. (A13):
//
//	[ -d_kk' sum_{c,j} v_aclj v*_a'cl'j
//	      ( eps_c - eps_j + 1/2 (eps_a + eps_a' - eps_l - eps_l') ) ]
//	+ [k<->l, k'<->l'] - [k<->l] - [k'<->l']
func (e *elements) m2D(a, k, l, ap, kp, lp sorb) float64 {
	return antisym4(k, l, kp, lp, func(k, l, kp, lp sorb) float64 {
		if k != kp {
			return 0
		}
		half := 0.5 * (e.epsOf(a) + e.epsOf(ap) - e.epsOf(l) - e.epsOf(lp))
		var sum float64
		e.virSO(func(c sorb) {
			e.occSO(func(j sorb) {
				v1 := e.vAmp(a, c, l, j)
				if v1 == 0 {
					return
				}
				v2 := e.vAmp(ap, c, lp, j)
				if v2 == 0 {
					return
				}
				sum += v1 * v2 * (e.epsOf(c) - e.epsOf(j) + half)
			})
		})
		return -sum
	})
}

// m2E is Eq. (A14):
//
//	sum_c v_ackl v*_a'ck'l'
//	      ( eps_c + 1/2 (eps_a + eps_a' - eps_k - eps_k' - eps_l - eps_l') )
func (e *elements) m2E(a, k, l, ap, kp, lp sorb) float64 {
	half := 0.5 * (e.epsOf(a) + e.epsOf(ap) -
		e.epsOf(k) - e.epsOf(kp) - e.epsOf(l) - e.epsOf(lp))
	var sum float64
	e.virSO(func(c sorb) {
		v1 := e.vAmp(a, c, k, l)
		if v1 == 0 {
			return
		}
		v2 := e.vAmp(ap, c, kp, lp)
		if v2 == 0 {
			return
		}
		sum += v1 * v2 * (e.epsOf(c) + half)
	})
	return sum
}

// m2_2h1p is Eq. (A9), the full second-order 2h1p/2h1p element for one pair of
// primitive spin-orbital configurations.
func (e *elements) m2_2h1p(a, k, l, ap, kp, lp sorb) float64 {
	return e.m2A(a, k, l, ap, kp, lp) +
		e.m2B(a, k, l, ap, kp, lp) +
		e.m2C(a, k, l, ap, kp, lp) +
		e.m2D(a, k, l, ap, kp, lp) +
		e.m2E(a, k, l, ap, kp, lp)
}

// c22_2 is the spin-adapted second-order 2h1p/2h1p element: A9 contracted over the
// two configurations' expansions into primitives.
func (e *elements) c22_2(row, col Config) float64 {
	rs, cs := e.expand2(row), e.expand2(col)
	var sum float64
	for _, r := range rs {
		for _, c := range cs {
			if r.C == 0 || c.C == 0 {
				continue
			}
			sum += r.C * c.C * e.m2_2h1p(r.P.A, r.P.K, r.P.L, c.P.A, c.P.K, c.P.L)
		}
	}
	return sum
}
