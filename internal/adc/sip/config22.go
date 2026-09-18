package sip

// config22.go — the ISR-ADC(2,2) configuration space: 1h | 2h1p | 3h2p, with no
// core-valence restriction.
//
// ADC(2,2) (Kolorenč/Averbukh, J. Chem. Phys. 152, 214107 (2020), Table I) treats
// the 1h and 2h1p states consistently through second order, which forces the 3h2p
// class into the explicit space. The 1h and 2h1p families are exactly those of the
// non-Dyson ADC(2)/(3) space (config.go), so NewSpace22 reuses addMain/addSat
// verbatim and only adds the 3h2p family.
//
// The 3h2p enumeration mirrors addSat3h2p (config4.go) in shape — the same nested
// symmetry loops, the same Group3 boundaries driving the block-strided operator —
// but drops the CVS restriction: all three holes run over the full occupied set,
// so the hole-symmetry loops must themselves be ordered (kSym <= lSym <= mSym) to
// visit each unordered triple once. In the CVS space that ordering was
// unnecessary because the core hole was distinguishable from the valence pair.
//
// Config3's field names are inherited from the CVS space, where Core meant the
// core hole. Here Core is simply the FIRST hole of the ordered triple K <= L <= M
// and carries no core character; nothing outside this file depends on that name.

// openShells3 is the number of singly occupied spatial orbitals of a 3h2p
// determinant, which is what fixes its doublet spin multiplicity.
//
// Three holes: all distinct leaves three singly occupied orbitals; one coincident
// pair empties that orbital and leaves one. (All three equal is impossible — an
// orbital holds two electrons — and the enumeration skips it.) Two particles:
// distinct gives two singly occupied orbitals, equal gives one doubly occupied
// orbital and none.
func openShells3(k, l, m, i, j int) int {
	n := 3
	if k == l || l == m || k == m {
		n = 1
	}
	if i != j {
		n += 2
	}
	return n
}

// nSpin3 is the number of doublet spin functions of a 3h2p configuration. It
// agrees with maxS3 on the cases the CVS space can produce (where the core hole is
// always distinct, so only l == m can coincide): 1, 2, 2, 5. TestSpin3MatchesMaxS3
// pins that.
func nSpin3(k, l, m, i, j int) int {
	return doubletCSF(openShells3(k, l, m, i, j)).NumFuncs()
}

// NewSpace22 enumerates the ADC(2,2) space for one target-symmetry sector:
// 1h | 2h1p | 3h2p, non-CVS. orbSym is 1-based GAMESS-UK labels or nil (symmetry
// off); sym is the target cation irrep (0-based).
func NewSpace22(nocc, norb int, orbSym []int, sym int) *Space {
	s := &Space{
		Sym:    sym,
		Nocc:   nocc,
		Norb:   norb,
		Nvir:   norb - nocc,
		orbSym: orbSym,
		nSym:   numIrreps(orbSym, norb),
		adc22:  true,
	}
	s.addMain()
	s.addSat()
	s.Begin3h2p = len(s.Configs)
	s.addSat3h2p22()
	return s
}

// addSat3h2p22 appends the general 3h2p configurations.
//
// Symmetry nesting: the hole-irrep triple is ordered kSym <= lSym <= mSym and the
// particle-irrep pair iSym <= jSym, with iSym fixed by the target XOR product, so
// each unordered combination of symmetry blocks is visited exactly once. Within a
// block, orbitals of the same irrep are ordered by index for the same reason.
// Group boundaries are recorded per symmetry block, matching Group3's contract.
func (s *Space) addSat3h2p22() {
	s.Sat3 = nil
	s.Group3 = []int{s.Begin3h2p}
	push := func(k, l, m, i, j, spin int) {
		s.Sat3 = append(s.Sat3, Config3{Core: k, L: l, M: m, I: i, J: j, Spin: spin})
	}
	group := func() {
		if n := s.Begin3h2p + len(s.Sat3); n > s.Group3[len(s.Group3)-1] {
			s.Group3 = append(s.Group3, n)
		}
	}
	for mSym := range s.nSym {
		for lSym := 0; lSym <= mSym; lSym++ {
			for kSym := 0; kSym <= lSym; kSym++ {
				for jSym := range s.nSym {
					iSym := symProduct(s.Sym, jSym, kSym, lSym, mSym)
					if iSym > jSym {
						continue // each particle-irrep pair once
					}
					s.fill3h2pBlock(kSym, lSym, mSym, iSym, jSym, push)
					group()
				}
			}
		}
	}
	if len(s.Group3) > 1 {
		s.Group3 = s.Group3[:len(s.Group3)-1]
	}
}

// fill3h2pBlock emits every 3h2p configuration of one symmetry block, holes
// outermost (M, then L <= M, then K <= L within coincident irreps) and particles
// inside (J, then I <= J), with the spin functions innermost.
func (s *Space) fill3h2pBlock(kSym, lSym, mSym, iSym, jSym int, push func(k, l, m, i, j, spin int)) {
	mList := s.occBySym(mSym)
	lList := s.occBySym(lSym)
	kList := s.occBySym(kSym)
	jList := s.virBySym(jSym)
	iList := s.virBySym(iSym)

	for mi, m := range mList {
		lEnd := len(lList)
		if lSym == mSym {
			lEnd = mi + 1 // L index <= M index within one irrep
		}
		for li, l := range lList[:lEnd] {
			kEnd := len(kList)
			if kSym == lSym {
				kEnd = li + 1 // K index <= L index within one irrep
			}
			for _, k := range kList[:kEnd] {
				if k == l && l == m {
					continue // three electrons out of one orbital
				}
				for ji, j := range jList {
					iEnd := len(iList)
					if iSym == jSym {
						iEnd = ji + 1 // I index <= J index
					}
					for _, i := range iList[:iEnd] {
						for spin := 1; spin <= nSpin3(k, l, m, i, j); spin++ {
							push(k, l, m, i, j, spin)
						}
					}
				}
			}
		}
	}
}
