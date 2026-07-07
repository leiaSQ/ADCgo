// Package dip implements the DIP-ADC(2) double-ionization secular problem,
// ported from theADCcode's adc2_dip (../ADC). The physics traces to
// F. Tarantelli, Chemical Physics 329 (2005) 11-21; where the paper and the
// reference code disagree, the code (singlet.cpp/triplet.cpp) is authoritative.
package dip

// Spin selects the spin adaptation of the 3h1p satellite space. The values match
// the reference's spin() convention (0 = singlet, 2 = triplet).
type Spin int

const (
	Singlet Spin = 0
	Triplet Spin = 2
)

// Config maps one row/column of the ADC matrix to an electronic configuration,
// mirroring ../ADC/adc2_dip/config.hpp. The four config families are
// distinguished by which region of the flat config slice they fall in:
//
//	|ii>     2h closed-shell : Occ[0]==Occ[1], Vir unused
//	|ij>     2h open-shell   : Occ[0] > Occ[1], Vir unused
//	|jiir>   3h1p type I     : Occ = {j,i,i}, Vir = virtual position
//	|ijkr,T> 3h1p type II    : Occ = {i,j,k} (i>j>k), Vir, Typ = spin function
//
// Vir is the 0-based position within the virtual block (absolute orbital
// = Nocc+Vir). Occupied indices are absolute (0-based).
type Config struct {
	Occ [3]int
	Vir int
	Typ int
}

// Space is the configuration space for one (symmetry, spin) sector: the flat
// index→config map plus the region boundaries and 3h1p group-start indices that
// drive the block-strided matrix-vector product.
type Space struct {
	Configs []Config

	BeginIJ  int // start of |ij> 2h open-shell
	BeginJII int // start of |jiir> 3h1p type I  == MainBlockSize()
	BeginIJK int // start of |ijkr> 3h1p type II

	// Group-start indices: JII[m]..JII[m+1] is one (j,i) group spanning all
	// allowed virtuals; similarly IJK for (i,j,k) groups (each spanning all
	// spin types × virtuals). The final group runs to the region end.
	JII []int
	IJK []int

	Mult int // number of |ijkr> spin functions: 2 (singlet) or 3 (triplet)
	Sym  int // target dication irrep (0-based)
	Spin Spin

	Nocc, Nvir, Norb int
	orbSym           []int // 0-based irrep per orbital
}

// MainBlockSize is the dimension of the 2h "main" space (everything below it is
// the 3h1p satellite space). Pole strengths are the squared 2h weight.
func (s *Space) MainBlockSize() int { return s.BeginJII }

// Size is the full matrix dimension.
func (s *Space) Size() int { return len(s.Configs) }

// irrep returns the 0-based point-group irrep of an orbital.
func (s *Space) irrep(orb int) int {
	if s.orbSym == nil {
		return 0
	}
	return s.orbSym[orb] - 1
}

// symProduct combines irreps. For the boolean point groups FCIDUMP uses (C1..D2h
// in the standard ordering) the direct product is XOR on 0-based labels; with
// symmetry off every product is 0.
func symProduct(irreps ...int) int {
	p := 0
	for _, r := range irreps {
		p ^= r
	}
	return p
}

// NewSpace enumerates the configuration space for the given sector, faithfully
// reproducing the ordering of ../ADC/adc2_dip/adc2_matrix.cpp (add_*_configs).
// orbSym may be nil (symmetry off → single totally-symmetric irrep). sym is the
// target dication irrep (0-based).
func NewSpace(nocc, norb int, orbSym []int, sym int, spin Spin) *Space {
	s := &Space{
		Sym:    sym,
		Spin:   spin,
		Nocc:   nocc,
		Norb:   norb,
		Nvir:   norb - nocc,
		orbSym: orbSym,
	}
	if spin == Triplet {
		s.Mult = 3
	} else {
		s.Mult = 2
	}
	s.addII()
	s.addIJ()
	s.addJIIR()
	s.addIJKR()
	return s
}

// addII: |ii> 2h closed-shell (singlet, totally-symmetric only).
func (s *Space) addII() {
	if s.Mult == 3 || s.Sym != 0 {
		s.BeginIJ = len(s.Configs)
		return
	}
	for i := range s.Nocc {
		s.Configs = append(s.Configs, Config{Occ: [3]int{i, i}})
	}
	s.BeginIJ = len(s.Configs)
}

// addIJ: |ij> 2h open-shell, i>j, irrep(i)⊗irrep(j) == sym.
func (s *Space) addIJ() {
	for i := range s.Nocc {
		for j := range i {
			if s.Sym != symProduct(s.irrep(i), s.irrep(j)) {
				continue
			}
			s.Configs = append(s.Configs, Config{Occ: [3]int{i, j}})
		}
	}
	s.BeginJII = len(s.Configs)
}

// addJIIR: |jiir> 3h1p type I. Group boundaries recorded in JII.
func (s *Space) addJIIR() {
	s.JII = append(s.JII, s.BeginJII)
	for j := range s.Nocc {
		for i := range s.Nocc {
			if i == j {
				continue
			}
			exists := false
			for rp := range s.Nvir {
				r := s.Nocc + rp
				if s.Sym != symProduct(s.irrep(j), s.irrep(r)) {
					continue
				}
				s.Configs = append(s.Configs, Config{Occ: [3]int{j, i, i}, Vir: rp})
				exists = true
			}
			if exists {
				s.JII = append(s.JII, len(s.Configs))
			}
		}
	}
	s.JII = s.JII[:len(s.JII)-1] // drop trailing boundary → JII[m] = group m start
	s.BeginIJK = len(s.Configs)
}

// addIJKR: |ijkr,T> 3h1p type II, i>j>k, type outer / r inner. Groups in IJK.
func (s *Space) addIJKR() {
	s.IJK = append(s.IJK, s.BeginIJK)
	for i := range s.Nocc {
		for j := range i {
			for k := range j {
				exists := false
				for typ := range s.Mult {
					for rp := range s.Nvir {
						r := s.Nocc + rp
						if s.Sym != symProduct(s.irrep(i), s.irrep(j), s.irrep(k), s.irrep(r)) {
							continue
						}
						exists = true
						s.Configs = append(s.Configs, Config{Occ: [3]int{i, j, k}, Vir: rp, Typ: typ})
					}
				}
				if exists {
					s.IJK = append(s.IJK, len(s.Configs))
				}
			}
		}
	}
	s.IJK = s.IJK[:len(s.IJK)-1]
}
