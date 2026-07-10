package sip

import "slices"

// config4.go — the CVS-restricted IP-ADC(4) configuration space (Track A, A2).
//
// Unlike the non-Dyson ADC(3) space (config.go), the ADC(4) reference
// (../ADC/adc4core/adc4_constr/state.F STATE_core, "CORE-NAEHERUNG") is a
// *core-ionization* spectrum: every configuration carries exactly one core hole.
// The three families, in global index order (1h | 2h1p | 3h2p):
//
//	|K>            1h main   : one core hole K
//	|K L a,S>      2h1p      : core hole K, valence hole L, particle a; spin S
//	|K L M a b,S>  3h2p      : core hole K, valence holes L,M, particles a,b; spin S
//
// Holes are absolute occupied orbital indices; particles are 0-based positions in
// the virtual block (absolute = Nocc+Vir), matching Config.Vir. The core set is
// supplied by the caller (absolute occupied indices). See docs/adc4_sip_spec.md §3.

// Config3 is one 3h2p configuration in the reference's role convention (state.F
// ZSTATE packing): Core (K) is the core hole; L,M are the valence holes; I,J are
// the particles (0-based virtual positions, absolute = Nocc+I). Roles are assigned
// by the enumeration (M from the outer hole-symmetry loop, L the inner with
// LSYM<=MSYM; J the outer particle-symmetry loop, I the inner with ISYM<=JSYM), NOT
// canonicalized by absolute index — the KOPP4/AB5 spin-coupling coefficients are
// tied to this ordering. Spin is the 1-based spin-function index (1..MAXS).
type Config3 struct {
	Core int // K, core hole (absolute occ index)
	L, M int // valence holes (absolute occ indices)
	I, J int // particles (0-based virtual positions)
	Spin int
}

// maxS3 is the 3h2p spin multiplicity (state.F:190-218) given the coincidences of
// the valence holes (L==M) and particles (I==J); K (core) is always distinct.
func maxS3(lEqM, iEqJ bool) int {
	switch {
	case lEqM && iEqJ:
		return 1
	case lEqM || iEqJ:
		return 2
	default:
		return 5
	}
}

// isCore reports whether absolute occupied orbital o is in the core set.
func (s *Space) isCore(o int) bool { return slices.Contains(s.core, o) }

// coreBySym returns the core occupied orbitals of 0-based irrep g, ascending.
func (s *Space) coreBySym(g int) []int {
	var out []int
	for _, o := range s.core {
		if s.irrep(o) == g {
			out = append(out, o)
		}
	}
	return out
}

// valOccBySym returns the non-core occupied orbitals of irrep g, ascending.
func (s *Space) valOccBySym(g int) []int {
	var out []int
	for i := range s.Nocc {
		if s.irrep(i) == g && !s.isCore(i) {
			out = append(out, i)
		}
	}
	return out
}

// NewSpace4 enumerates the CVS-restricted IP-ADC(4) space for the target-symmetry
// sector: 1h (core) | 2h1p (one core hole) | 3h2p (one core hole). core is the set
// of absolute occupied core-orbital indices. orbSym is 1-based GAMESS-UK labels or
// nil (symmetry off); sym is the target cation irrep (0-based). Mirrors
// STATE_core's ordering (symmetry-block nesting, then orbital loops, spin functions
// innermost).
func NewSpace4(nocc, norb int, orbSym []int, sym int, core []int) *Space {
	s := &Space{
		Sym:    sym,
		Nocc:   nocc,
		Norb:   norb,
		Nvir:   norb - nocc,
		orbSym: orbSym,
		nSym:   numIrreps(orbSym, norb),
		core:   append([]int(nil), core...),
		adc4:   true,
	}
	s.addMain4()
	s.addSat4()
	s.addSat3h2p()
	return s
}

// addMain4: |K> 1h, core orbitals of the target irrep in orbital order.
func (s *Space) addMain4() {
	for _, k := range s.core {
		if s.irrep(k) == s.Sym {
			s.Configs = append(s.Configs, Config{Occ: [2]int{k, k}})
		}
	}
	s.BeginSat = len(s.Configs)
}

// addSat4: 2h1p satellites with one core hole K, one valence hole L, particle a.
// Symmetry: irrep(K) ^ irrep(L) ^ irrep(a) == Sym. Spin: MAXS = 2 (K != L, always
// true since K is core and L is valence) — two spin functions (type I/II). Group
// boundaries recorded in Group.
func (s *Space) addSat4() {
	s.Group = append(s.Group, s.BeginSat)
	push := func(k, l, a, typ int) {
		s.Configs = append(s.Configs, Config{Occ: [2]int{k, l}, Vir: a, Typ: typ})
	}
	group := func() {
		if n := len(s.Configs); n > s.Group[len(s.Group)-1] {
			s.Group = append(s.Group, n)
		}
	}
	for kSym := range s.nSym {
		for lSym := range s.nSym {
			aSym := symProduct(s.Sym, kSym, lSym)
			for _, k := range s.coreBySym(kSym) {
				for _, l := range s.valOccBySym(lSym) {
					for _, a := range s.virBySym(aSym) {
						push(k, l, a, 0)
						push(k, l, a, 1)
					}
					group()
				}
			}
		}
	}
	s.Group = s.Group[:len(s.Group)-1]
	s.Begin3h2p = len(s.Configs)
}

// addSat3h2p: 3h2p configs with core hole K, valence holes L,M, particles a,b.
// Mirrors STATE_core (state.F:136-250): five nested symmetry loops — mSym, lSym<=
// mSym, kSym, jSym(particle), iSym<=jSym — with the XOR product constrained to Sym,
// so each unordered valence-hole-symmetry pair and particle-symmetry pair is visited
// once. Within a block, orbital pairs in the same irrep are ordered (L<=M, a<=b).
// Spin multiplicity per state.F:190-218: with one core hole K (always distinct from
// the valence holes L,M, so IFKL=IFLM handle only the L==M coincidence):
//
//	L == M, a == b : 1     L == M, a != b : 2
//	L != M, a == b : 2     L != M, a != b : 5
//
// Holes[0]=K (core), {Holes[1],Holes[2]}={L,M} with L<=M; Virs={a,b} with a<=b.
func (s *Space) addSat3h2p() {
	s.Sat3 = nil
	s.Group3 = []int{s.Begin3h2p}
	// Roles stored as the enumeration assigns them (see Config3): K core, L the
	// inner valence hole, M the outer, I the inner particle, J the outer. No
	// canonicalization — the spin coefficients depend on this ordering.
	push := func(k, l, m, a, b, spin int) {
		s.Sat3 = append(s.Sat3, Config3{Core: k, L: l, M: m, I: a, J: b, Spin: spin})
	}
	group := func() {
		if n := s.Begin3h2p + len(s.Sat3); n > s.Group3[len(s.Group3)-1] {
			s.Group3 = append(s.Group3, n)
		}
	}
	// valence-hole symmetry pair (mSym >= lSym), core-hole symmetry, particle
	// symmetry pair (jSym >= iSym); iSym fixed by the target XOR product.
	for mSym := range s.nSym {
		for lSym := 0; lSym <= mSym; lSym++ {
			for kSym := range s.nSym {
				for jSym := range s.nSym {
					// iSym ^ jSym ^ kSym ^ lSym ^ mSym == Sym
					iSym := symProduct(s.Sym, jSym, kSym, lSym, mSym)
					if iSym > jSym {
						continue // enforce iSym <= jSym (each particle-sym pair once)
					}
					// Orbital nesting mirrors state.F:163-189 (M outer, L inner with
					// L<=M; K core; J outer, I inner with I<=J) so the flat 3h2p order
					// matches the reference tape column order exactly.
					for mi, m := range s.valOccBySym(mSym) {
						lList := s.valOccBySym(lSym)
						lEnd := len(lList)
						if lSym == mSym {
							lEnd = mi + 1 // L index <= M index within the same irrep
						}
						for _, l := range lList[:lEnd] {
							for _, k := range s.coreBySym(kSym) {
								for ji, j := range s.virBySym(jSym) {
									iList := s.virBySym(iSym)
									iEnd := len(iList)
									if iSym == jSym {
										iEnd = ji + 1 // I index <= J index
									}
									for _, i := range iList[:iEnd] {
										ms := maxS3(l == m, i == j)
										for spin := 1; spin <= ms; spin++ {
											push(k, l, m, i, j, spin)
										}
									}
								}
							}
						}
					}
					group()
				}
			}
		}
	}
	if len(s.Group3) > 1 {
		s.Group3 = s.Group3[:len(s.Group3)-1]
	}
}
