// Package sip implements the single-ionization non-Dyson IP-ADC(3) secular
// problem (1h main / 2h1p satellite), ported from theADCcode's ndadc3_ip
// (../ADC/ndadc3_ip, J. Breidbach, J. Chem. Phys. 109 (1998) 4734). It reuses
// the DIP engine's backend / integrals / lanczos infrastructure; only the
// configuration space and the block matrix elements are new.
//
// The cation is a doublet, so — unlike DIP — there is a single spin channel (the
// reference's spin() == 1). Where the paper and the code disagree, the code
// (the calc_*.c files) is authoritative.
package sip

// Config maps one row/column of the ADC matrix to an electronic configuration,
// mirroring ndadc3_ip's add_configs()/FOR_ALL_2H1P_AKL ordering. Two families,
// distinguished by region (index < Space.BeginSat is main):
//
//	|i>       1h main     : Occ[0] = hole, Occ[1] unused, Vir unused
//	<k,l,a,T| 2h1p type I/II (k!=l): Occ = {k,l} (macro roles), Vir = particle,
//	                                 Typ = 0 (I) or 1 (II)
//	<k,k,a|   2h1p single  (k==l): Occ = {k,k}, Vir = particle, Typ = 0
//
// Occupied indices are absolute (0-based). Vir is the 0-based position within the
// virtual block (absolute orbital = Nocc+Vir). k/l are macro roles (k from the
// outer symmetry loop, l from the inner), not ordered by magnitude in the
// cross-symmetry case; in the same-symmetry case l < k by orbital index.
type Config struct {
	Occ [2]int
	Vir int
	Typ int
}

// Space is the configuration space for one target-symmetry sector: the flat
// index→config map, the main/satellite boundary, and the per-(k,l,a) satellite
// group boundaries that drive the block-strided matrix-vector product.
type Space struct {
	Configs []Config

	BeginSat int // start of the 2h1p satellite space == MainBlockSize()

	// Sat group boundaries: Group[g]..Group[g+1] is one (k,l) satellite group
	// spanning all allowed particles a (and the 1 or 2 spin functions). The last
	// group runs to len(Configs). Used by the assembled satellite operator.
	Group []int

	// ADC(4) 3h2p satellite space (config4.go; empty for order 2/3). Sat3 holds the
	// 3h2p configs; Begin3h2p is their global start index (== len(Configs)); Group3
	// records 3h2p group boundaries in the global index. Group boundaries are the
	// per-symmetry-block starts used by the assembled 3h2p operator.
	Sat3      []Config3
	Begin3h2p int
	Group3    []int
	core      []int // absolute occupied core-orbital indices (CVS); nil for order 2/3
	adc4      bool  // true when built by NewSpace4 (CVS ADC(4) 1h|2h1p|3h2p space)

	Sym  int // target cation irrep (0-based)
	nSym int // number of point-group irreps (power of two spanning the labels)

	Nocc, Nvir, Norb int
	orbSym           []int // 1-based GAMESS-UK ORBSYM per orbital, or nil (symmetry off)
}

// MainBlockSize is the dimension of the 1h main space; everything at or above it
// is the 2h1p satellite space. Spectroscopic factors are the squared 1h weight.
func (s *Space) MainBlockSize() int { return s.BeginSat }

// Size is the full matrix dimension (1h + 2h1p, plus 3h2p for order 4).
func (s *Space) Size() int { return len(s.Configs) + len(s.Sat3) }

// irrep returns the 0-based point-group irrep of an orbital.
func (s *Space) irrep(orb int) int {
	if s.orbSym == nil {
		return 0
	}
	return s.orbSym[orb] - 1
}

// symProduct combines irreps by XOR on 0-based labels — the direct product for
// the boolean point groups FCIDUMP uses (C1..D2h). With symmetry off every
// product is 0. This is theADCcode's Multab_ for these groups.
func symProduct(irreps ...int) int {
	p := 0
	for _, r := range irreps {
		p ^= r
	}
	return p
}

// numIrreps is the group order implied by the ORBSYM labels (the smallest power
// of two spanning them), matching integrals.Store's grouping and the C code's
// symtab->nSym.
func numIrreps(orbSym []int, norb int) int {
	if orbSym == nil {
		return 1
	}
	max0 := 0
	for o := range norb {
		if lab := orbSym[o] - 1; lab > max0 {
			max0 = lab
		}
	}
	n := 1
	for n < max0+1 {
		n <<= 1
	}
	return n
}

// NewSpace enumerates the configuration space for the target-symmetry sector,
// reproducing ndadc3_ip's add_configs() ordering (main 1h first, then the 2h1p
// satellites in FOR_ALL_2H1P_AKL order). orbSym is 1-based GAMESS-UK labels or nil
// (symmetry off). sym is the target cation irrep (0-based).
func NewSpace(nocc, norb int, orbSym []int, sym int) *Space {
	s := &Space{
		Sym:    sym,
		Nocc:   nocc,
		Norb:   norb,
		Nvir:   norb - nocc,
		orbSym: orbSym,
		nSym:   numIrreps(orbSym, norb),
	}
	s.addMain()
	s.addSat()
	return s
}

// occBySym / virBySym return the occupied / virtual orbitals of one 0-based irrep
// in ascending orbital order (matching symtab->occ[]/vir[]). Virtuals are 0-based
// positions within the virtual block.
func (s *Space) occBySym(g int) []int {
	var out []int
	for i := range s.Nocc {
		if s.irrep(i) == g {
			out = append(out, i)
		}
	}
	return out
}

func (s *Space) virBySym(g int) []int {
	var out []int
	for rp := range s.Nvir {
		if s.irrep(s.Nocc+rp) == g {
			out = append(out, rp)
		}
	}
	return out
}

// addMain: |i> 1h, occupied orbitals of the target irrep in orbital order.
func (s *Space) addMain() {
	for i := range s.Nocc {
		if s.irrep(i) == s.Sym {
			s.Configs = append(s.Configs, Config{Occ: [2]int{i, i}})
		}
	}
	s.BeginSat = len(s.Configs)
}

// addSat: 2h1p satellites in FOR_ALL_2H1P_AKL order (adc_macros.h). Two holes
// k (outer symmetry) and l (inner symmetry, or same symmetry with l < k), one
// particle a of symmetry sym⊗sym(k)⊗sym(l). Type I/II spin functions when k!=l,
// a single function when k==l. Each (k,l,a) group's start is recorded in Group.
func (s *Space) addSat() {
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
		kISym := symProduct(s.Sym, kSym)
		// l_sym < k_sym  =>  k != l
		for lSym := range kSym {
			aSym := symProduct(kISym, lSym)
			for _, k := range s.occBySym(kSym) {
				for _, l := range s.occBySym(lSym) {
					for _, a := range s.virBySym(aSym) {
						push(k, l, a, 0)
						push(k, l, a, 1)
					}
					group()
				}
			}
		}
		// l_sym == k_sym, a in the target symmetry
		occ := s.occBySym(kSym)
		vir := s.virBySym(s.Sym)
		for ki, k := range occ {
			for _, l := range occ[:ki] { // l < k by orbital index
				for _, a := range vir {
					push(k, l, a, 0)
					push(k, l, a, 1)
				}
				group()
			}
			for _, a := range vir { // k == l single spin function
				push(k, k, a, 0)
			}
			group()
		}
	}
	s.Group = s.Group[:len(s.Group)-1] // drop trailing boundary → Group[g] = group g start
}
