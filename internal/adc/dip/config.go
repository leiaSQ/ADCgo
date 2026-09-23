// Package dip implements the DIP-ADC(2) double-ionization secular problem,
// ported from theADCcode's adc2_dip (../ADC). The physics traces to
// F. Tarantelli, Chemical Physics 329 (2005) 11-21; where the paper and the
// reference code disagree, the code (singlet.cpp/triplet.cpp) is authoritative.
package dip

import (
	"fmt"
	"slices"

	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

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

// PartitionBounds returns row-partition boundaries that split the sector across up to g
// devices for the distributed (multi-GPU) Mode B backend. Splits fall only on group
// boundaries — the 2h main block and every 3h1p group (JII/IJK) stay intact — so every
// operator block's row band lands on a single device (the precondition
// backend.NewDistributed relies on). The result is ascending, starts at 0, ends at
// Size(), and has between 2 and g+1 entries (fewer than g+1 when the sector has too few
// groups to fill g partitions). Balances by cumulative size, snapping each target to the
// nearest group boundary.
func (s *Space) PartitionBounds(g int) []int {
	n := s.Size()
	if g < 1 {
		g = 1
	}
	// Candidate split points: every group start (0, main end, each JII/IJK group), plus n.
	cand := []int{0, s.BeginJII, s.BeginIJK, n}
	cand = append(cand, s.JII...)
	cand = append(cand, s.IJK...)
	seen := map[int]bool{}
	uniq := cand[:0]
	for _, c := range cand {
		if c >= 0 && c <= n && !seen[c] {
			seen[c] = true
			uniq = append(uniq, c)
		}
	}
	cand = uniq
	slices.Sort(cand)

	bounds := []int{0}
	for k := 1; k < g; k++ {
		target := k * n / g
		last := bounds[len(bounds)-1]
		best := -1
		for _, c := range cand {
			if c <= last || c >= n {
				continue
			}
			if best < 0 || abs(c-target) < abs(best-target) {
				best = c
			}
		}
		if best < 0 {
			break // out of interior group boundaries → fewer partitions than g
		}
		bounds = append(bounds, best)
	}
	return append(bounds, n)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

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

// walkJIIRow visits every |jiir> config whose outer occupied index is j, in the exact order the
// serial enumeration emitted them, calling emit once per config. `first` marks the opening config
// of a nonempty (j,i) group, i.e. exactly the points the group-start list JII records.
//
// One function drives both the counting and the filling pass below, so the two cannot drift —
// the same discipline buildJIIBatchPlan uses one file over. The enumeration ORDER is load-bearing
// (Configs and JII index the operator and the reference comparison), so it lives here once.
func (s *Space) walkJIIRow(j int, emit func(i, rp int, first bool)) {
	for i := range s.Nocc {
		if i == j {
			continue
		}
		first := true
		for rp := range s.Nvir {
			r := s.Nocc + rp
			if s.Sym != symProduct(s.irrep(j), s.irrep(r)) {
				continue
			}
			emit(i, rp, first)
			first = false
		}
	}
}

// addJIIR: |jiir> 3h1p type I. Group boundaries recorded in JII.
//
// Count / prefix-sum / parallel-fill over the outer index j. The serial version was
// nocc²·nvir ≈ 5.2e5 iterations for the production system (nocc=58, nvir=154) on one core, growing Configs by
// unsized append — and Configs reaches 10.0 M entries × 40 B = 400 MB on the production singlet, so
// amortized doubling left a ~1.5× peak plus repeated copies of a slice that size. Counting first
// gives the exact allocation.
//
// Bit-reproducible: this is pure enumeration, no floating-point reduction. Every worker owns one
// j and writes only into Configs[cfgOf[j]:cfgOf[j+1]] and JII[grpOf[j]:grpOf[j+1]] — offsets
// computed by the serial prefix sum, so each config lands at exactly the index the serial append
// would have given it, whatever order the workers run in. HeavyRows, not parallel.Rows: nocc=58
// is below Rows' 2*GOMAXPROCS cutoff, which would have run the whole thing serially anyway.
func (s *Space) addJIIR() {
	nocc := s.Nocc
	cfgOf := make([]int, nocc+1) // configs emitted by rows < j
	grpOf := make([]int, nocc+1) // JII groups opened by rows < j
	parallel.HeavyRows(nocc, func(j int) {
		nc, ng := 0, 0
		s.walkJIIRow(j, func(_, _ int, first bool) {
			nc++
			if first {
				ng++
			}
		})
		cfgOf[j+1], grpOf[j+1] = nc, ng
	})
	for j := range nocc {
		cfgOf[j+1] += cfgOf[j]
		grpOf[j+1] += grpOf[j]
	}

	// JII[m] is group m's absolute start. The serial version built it as the running end of each
	// nonempty group with the leading BeginJII prepended and the trailing boundary dropped; the
	// groups are contiguous, so end(m) == start(m+1) and the two lists are identical.
	base := len(s.Configs) // == s.BeginJII
	s.Configs = slices.Grow(s.Configs, cfgOf[nocc])[:base+cfgOf[nocc]]
	s.JII = make([]int, grpOf[nocc])
	parallel.HeavyRows(nocc, func(j int) {
		ci, gi := base+cfgOf[j], grpOf[j]
		s.walkJIIRow(j, func(i, rp int, first bool) {
			if first {
				s.JII[gi] = ci
				gi++
			}
			s.Configs[ci] = Config{Occ: [3]int{j, i, i}, Vir: rp}
			ci++
		})
	})
	s.BeginIJK = len(s.Configs)
}

// walkIJKRow visits every |ijkr,T> config whose leading occupied index is i, in the exact order
// the serial enumeration emitted them (j > k inner, then type outer / r inner). `first` marks the
// opening config of a nonempty (i,j,k) group — the points IJK records. Shared by the counting and
// filling passes of addIJKR for the same no-drift reason as walkJIIRow.
func (s *Space) walkIJKRow(i int, emit func(j, k, typ, rp int, first bool)) {
	for j := range i {
		for k := range j {
			first := true
			for typ := range s.Mult {
				for rp := range s.Nvir {
					r := s.Nocc + rp
					if s.Sym != symProduct(s.irrep(i), s.irrep(j), s.irrep(k), s.irrep(r)) {
						continue
					}
					emit(j, k, typ, rp, first)
					first = false
				}
			}
		}
	}
}

// addIJKR: |ijkr,T> 3h1p type II, i>j>k, type outer / r inner. Groups in IJK.
//
// Same count / prefix-sum / parallel-fill structure as addJIIR, and the bigger of the two by far:
// C(58,3)=30,856 groups × mult × nvir is ~9.5e6 iterations on the production singlet and ~1.43e7 on
// the triplet, all of it serial on one core with an unsized append behind it. Rows are wildly
// uneven (row i holds C(i,2) triples, so the last row alone is ~5% of the work), which is exactly
// what HeavyRows' work-stealing schedule absorbs and a static split would not.
//
// Bit-reproducible for the same reason as addJIIR: enumeration only, disjoint precomputed output
// ranges per row, no floating-point reduction and no concurrent appends.
func (s *Space) addIJKR() {
	nocc := s.Nocc
	cfgOf := make([]int, nocc+1)
	grpOf := make([]int, nocc+1)
	parallel.HeavyRows(nocc, func(i int) {
		nc, ng := 0, 0
		s.walkIJKRow(i, func(_, _, _, _ int, first bool) {
			nc++
			if first {
				ng++
			}
		})
		cfgOf[i+1], grpOf[i+1] = nc, ng
	})
	for i := range nocc {
		cfgOf[i+1] += cfgOf[i]
		grpOf[i+1] += grpOf[i]
	}

	base := len(s.Configs) // == s.BeginIJK
	s.Configs = slices.Grow(s.Configs, cfgOf[nocc])[:base+cfgOf[nocc]]
	s.IJK = make([]int, grpOf[nocc])
	parallel.HeavyRows(nocc, func(i int) {
		ci, gi := base+cfgOf[i], grpOf[i]
		s.walkIJKRow(i, func(j, k, typ, rp int, first bool) {
			if first {
				s.IJK[gi] = ci
				gi++
			}
			s.Configs[ci] = Config{Occ: [3]int{i, j, k}, Vir: rp, Typ: typ}
			ci++
		})
	})
}

// Holes appends the occupied orbitals of row's configuration to dst, a doubly emptied
// orbital twice (the fano.Space contract): |ii> gives i,i; |ij> gives i,j; |jiir> gives
// j,i,i; |ijkr,T> gives i,j,k.
func (s *Space) Holes(row int, dst []int) []int {
	c := s.Configs[row]
	if row < s.BeginJII {
		return append(dst, c.Occ[0], c.Occ[1])
	}
	return append(dst, c.Occ[0], c.Occ[1], c.Occ[2])
}

// Particles appends the virtual orbital (0-based position in the virtual block) of a
// 3h1p row to dst; a 2h row has none.
func (s *Space) Particles(row int, dst []int) []int {
	if row < s.BeginJII {
		return dst
	}
	return append(dst, s.Configs[row].Vir)
}

// Restrict returns the sub-space spanned by the given rows, which must be ascending and
// distinct, and must keep every 3h1p group (JII / IJK) either whole or not at all.
//
// The assembled operator builds each 3h1p group as one panel over all of its virtuals
// and spin functions (singlet.go, triplet.go, the matrix-free and batched satellite
// paths), so a group is the smallest unit a sub-space can hold; within that rule New
// over the result assembles exactly the corresponding sub-block of the parent's matrix.
// A partition by holes (fano scheme A) always satisfies it, since every row of a group
// carries the same holes. 2h rows are independent and may be kept singly.
func (s *Space) Restrict(rows []int) (*Space, error) {
	n := s.Size()
	keep := make([]bool, n)
	for i, r := range rows {
		if r < 0 || r >= n {
			return nil, fmt.Errorf("dip: Restrict row %d outside 0..%d", r, n-1)
		}
		if i > 0 && r <= rows[i-1] {
			return nil, fmt.Errorf("dip: Restrict rows not strictly ascending at %d", i)
		}
		keep[r] = true
	}
	// group boundaries of the parent, region by region
	type grp struct{ lo, hi int }
	groups := func(starts []int, end int) []grp {
		out := make([]grp, len(starts))
		for m, lo := range starts {
			hi := end
			if m+1 < len(starts) {
				hi = starts[m+1]
			}
			out[m] = grp{lo, hi}
		}
		return out
	}
	jii := groups(s.JII, s.BeginIJK)
	ijk := groups(s.IJK, n)
	whole := func(g grp) (bool, error) {
		k := 0
		for r := g.lo; r < g.hi; r++ {
			if keep[r] {
				k++
			}
		}
		if k != 0 && k != g.hi-g.lo {
			return false, fmt.Errorf("dip: Restrict keeps %d of the %d rows of the 3h1p group "+
				"at row %d (holes %v); a group must be kept whole", k, g.hi-g.lo, g.lo,
				s.Holes(g.lo, nil))
		}
		return k != 0, nil
	}
	out := &Space{Sym: s.Sym, Spin: s.Spin, Mult: s.Mult, Nocc: s.Nocc, Nvir: s.Nvir,
		Norb: s.Norb, orbSym: s.orbSym}
	out.Configs = make([]Config, 0, len(rows))
	for _, r := range rows {
		if r >= s.BeginJII {
			break
		}
		out.Configs = append(out.Configs, s.Configs[r])
		if r < s.BeginIJ {
			out.BeginIJ = len(out.Configs)
		}
	}
	out.BeginJII = len(out.Configs)
	for _, g := range jii {
		ok, err := whole(g)
		if err != nil {
			return nil, err
		}
		if ok {
			out.JII = append(out.JII, len(out.Configs))
			out.Configs = append(out.Configs, s.Configs[g.lo:g.hi]...)
		}
	}
	out.BeginIJK = len(out.Configs)
	for _, g := range ijk {
		ok, err := whole(g)
		if err != nil {
			return nil, err
		}
		if ok {
			out.IJK = append(out.IJK, len(out.Configs))
			out.Configs = append(out.Configs, s.Configs[g.lo:g.hi]...)
		}
	}
	if len(out.Configs) != len(rows) {
		return nil, fmt.Errorf("dip: Restrict kept %d rows, %d requested", len(out.Configs), len(rows))
	}
	return out, nil
}
