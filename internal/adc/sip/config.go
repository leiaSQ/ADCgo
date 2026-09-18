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
	adc22     bool  // true when built by NewSpace22 (ADC(2,2) 1h|2h1p|3h2p space, non-CVS)

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

// ---------------------------------------------------------------------------
// Sub-spaces (Fano/Feshbach partitioning).
// ---------------------------------------------------------------------------

// Holes appends the occupied orbitals of row's configuration to dst and returns the
// extended slice: one orbital for a 1h row, two for 2h1p, three for 3h2p. dst may be
// nil; passing a reused buffer with dst[:0] keeps this allocation-free.
//
// It exists so a Fano Q/P selector can be written against the configuration's holes
// without importing this package's config types — the paper's scheme A predicate
// ("all holes localized on subunit A", "retains the initial vacancy") is a function
// of exactly this list. Rows are indexed globally, as Size() counts them: below
// BeginSat they are 1h, below Begin3h2p 2h1p, above it 3h2p.
func (s *Space) Holes(row int, dst []int) []int {
	switch {
	case row < s.BeginSat:
		// A 1h config stores its single hole duplicated (addMain).
		return append(dst, s.Configs[row].Occ[0])
	case row < len(s.Configs):
		c := s.Configs[row]
		return append(dst, c.Occ[0], c.Occ[1])
	default:
		c := s.Sat3[row-len(s.Configs)]
		return append(dst, c.Core, c.L, c.M)
	}
}

// begin3h2p is the global row index at which the 3h2p class starts. It is
// len(Configs) by the field's own contract, and NOT Begin3h2p: that field is left
// zero for an order-2/3 space, which has no 3h2p class at all, so reading it would
// send every 2h1p row of such a space down the 3h2p branch.
func (s *Space) begin3h2p() int { return len(s.Configs) }

// Restrict returns the sub-space spanned by the given global rows, which must be
// ascending and distinct. The result is a Space in its own right: New over it
// assembles exactly the corresponding sub-block of the parent's secular matrix, so
// the Fano QMQ and PMP operators are ordinary ADC matrices and every solver, the
// matrix-free paths and the GPU backends all work on them unchanged. That is the
// whole reason this exists rather than a projected operator — and it is what the
// paper means by scheme A leaving the cost unchanged.
//
// The parent's orbital data (Nocc, Norb, symmetry labels, the CVS core set) and its
// scheme flags carry over untouched, because the matrix elements depend on the
// orbitals and not on which configurations were kept: an element function sums over
// all orbitals either way. Only the row set shrinks.
//
// Filtering preserves the parent's order, so the 1h | 2h1p | 3h2p class bands stay
// contiguous and the class boundaries are just recounted. The group boundaries are
// recomputed from the retained configurations rather than inherited, since a
// restricted space's groups are not the parent's.
//
// The caller keeps rows as the map back to the parent, which is what Fano needs to
// embed a Q eigenvector into the parent index space and read the coupling off.
func (s *Space) Restrict(rows []int) *Space {
	out := &Space{
		Sym: s.Sym, nSym: s.nSym,
		Nocc: s.Nocc, Nvir: s.Nvir, Norb: s.Norb,
		orbSym: s.orbSym,
		core:   s.core,
		adc4:   s.adc4, adc22: s.adc22,
	}
	split := s.begin3h2p()
	for _, r := range rows {
		switch {
		case r < s.BeginSat:
			out.Configs = append(out.Configs, s.Configs[r])
			out.BeginSat = len(out.Configs)
		case r < split:
			out.Configs = append(out.Configs, s.Configs[r])
		default:
			out.Sat3 = append(out.Sat3, s.Sat3[r-split])
		}
	}
	// Begin3h2p stays zero for a space with no 3h2p class, as the constructors leave
	// it, so a restricted order-2/3 space is indistinguishable from a freshly built one.
	if len(s.Sat3) > 0 {
		out.Begin3h2p = len(out.Configs)
	}
	out.regroup()
	return out
}

// regroup rebuilds Group and Group3 for a space whose configurations were filtered.
//
// Group's contract is one entry per (k,l) satellite group; Group3's is one entry per
// 3h2p symmetry block. Both are recovered by scanning for a change in the defining
// key — the hole pair for Group, the irrep tuple for Group3 — which reproduces the
// constructors' boundaries exactly when nothing was filtered out, because the
// enumeration emits each group contiguously.
func (s *Space) regroup() {
	s.Group = nil
	s.Group3 = nil
	prev := [2]int{-1, -1}
	for i := s.BeginSat; i < len(s.Configs); i++ {
		if key := s.Configs[i].Occ; key != prev {
			s.Group = append(s.Group, i)
			prev = key
		}
	}
	prevSym := [5]int{-1, -1, -1, -1, -1}
	for i, c := range s.Sat3 {
		key := [5]int{
			s.irrep(c.Core), s.irrep(c.L), s.irrep(c.M),
			s.irrep(s.Nocc + c.I), s.irrep(s.Nocc + c.J),
		}
		if key != prevSym {
			s.Group3 = append(s.Group3, len(s.Configs)+i)
			prevSym = key
		}
	}
}
