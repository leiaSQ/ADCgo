// Package integrals provides the occ/vir-blocked two-electron-integral
// accessors the DIP-ADC(2) matrix elements are written in terms of. It mirrors
// the V/A/B block proxies of theADCcode's integral_table (../ADC): each block is
// restricted to one virtual-orbital symmetry group, so a (symmetry, spin) sector
// of the ADC matrix only ever touches the virtuals its configurations span. For
// M1/M2 the blocks are materialized on demand from the dense FCIDUMP store rather
// than the symmetry-blocked, permutation-packed tables (that packing is an M3
// memory optimization).
//
// Index conventions match the reference exactly (chemist notation (pq|rs), all
// 0-based). r, s denote virtual orbitals; i, j, k denote occupied orbitals:
//
//	V(i,j,k, sym)[r]  = (rk|ij)   r over the sym virtual group
//	A(i,j, sym)[r,s]  = (ri|sj)   s over the sym virtual group, r over its own
//	B(i,j, sym)[r,s]  = (rs|ij)   s over the sym virtual group, r over its own
//
// The A/B row index r runs over the virtual group whose symmetry is fixed by the
// integral's total symmetry, sym_product(irrep(i), irrep(j), sym); V's single
// virtual index r runs over the passed sym group directly. Virtuals within a
// group are ordered by ascending orbital index, matching the configuration
// enumeration in package dip. With point-group symmetry off (ORBSYM all equal),
// there is a single group holding every virtual and the accessors reduce to full
// nvir blocks.
package integrals

import (
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
)

// Store wraps the FCIDUMP integrals with occ/vir + symmetry metadata.
type Store struct {
	d    *fcidump.Data
	nocc int
	nvir int

	orbSym   []int   // 0-based irrep per absolute orbital (all 0 when symmetry off)
	nsym     int     // number of symmetry groups (power of two ≥ max label + 1)
	virBySym [][]int // virBySym[σ] = virtual positions (0-based, ascending) of irrep σ
}

// New builds a Store for a closed-shell reference with nocc occupied orbitals.
// orbSym gives the point-group label per orbital (1-based GAMESS-UK labels, as in
// FCIDUMP ORBSYM); pass nil to disable symmetry (a single virtual group). It must
// match the orbSym handed to dip.NewSpace so the configuration enumeration and
// the integral blocks agree on the virtual grouping.
func New(d *fcidump.Data, nocc int, orbSym []int) *Store {
	s := &Store{d: d, nocc: nocc, nvir: d.NORB - nocc}

	// 0-based irrep per orbital. FCIDUMP ORBSYM is 1-based (nil ⇒ symmetry off).
	s.orbSym = make([]int, d.NORB)
	maxLabel := 0
	for o := range d.NORB {
		lab := 0
		if orbSym != nil {
			lab = orbSym[o] - 1
		}
		s.orbSym[o] = lab
		if lab > maxLabel {
			maxLabel = lab
		}
	}
	// Number of groups: smallest power of two that can index every XOR product of
	// the present labels (the boolean point groups FCIDUMP uses are closed under
	// XOR only up to a power-of-two label count).
	s.nsym = 1
	for s.nsym < maxLabel+1 {
		s.nsym <<= 1
	}
	s.virBySym = make([][]int, s.nsym)
	for rp := range s.nvir {
		σ := s.orbSym[nocc+rp]
		s.virBySym[σ] = append(s.virBySym[σ], rp)
	}
	return s
}

// NOcc returns the number of occupied orbitals.
func (s *Store) NOcc() int { return s.nocc }

// NVir returns the number of virtual orbitals.
func (s *Store) NVir() int { return s.nvir }

// NSym returns the number of virtual-symmetry groups.
func (s *Store) NSym() int { return s.nsym }

// OrbIrrep returns the 0-based irrep of absolute orbital orb.
func (s *Store) OrbIrrep(orb int) int { return s.orbSym[orb] }

// SizeVirGroup returns the number of virtual orbitals in symmetry group sym.
func (s *Store) SizeVirGroup(sym int) int {
	if sym < 0 || sym >= s.nsym {
		return 0
	}
	return len(s.virBySym[sym])
}

// symProduct is the Abelian direct product on 0-based irrep labels (XOR), the
// same convention package dip uses.
func symProduct(a, b, c int) int { return a ^ b ^ c }

// Eri returns the two-electron integral (pq|rs) in chemist notation (0-based,
// absolute orbital indices) — the reference's V1122(p,q,r,s).
func (s *Store) Eri(p, q, r, t int) float64 { return s.d.TwoE(p, q, r, t) }

// EriMinus returns (pq|rs) − (ps|rq): the reference's V1122_MINUS.
func (s *Store) EriMinus(p, q, r, t int) float64 {
	return s.d.TwoE(p, q, r, t) - s.d.TwoE(p, t, r, q)
}

// EriPlus returns (pq|rs) + (ps|rq): the reference's V1122_PLUS.
func (s *Store) EriPlus(p, q, r, t int) float64 {
	return s.d.TwoE(p, q, r, t) + s.d.TwoE(p, t, r, q)
}

// V returns the vector V[r] = (rk|ij) with r running over the virtual group of
// symmetry sym (length SizeVirGroup(sym)).
func (s *Store) V(i, j, k, sym int) []float64 {
	rs := s.virGroup(sym)
	v := make([]float64, len(rs))
	for idx, rp := range rs {
		v[idx] = s.d.TwoE(s.nocc+rp, k, i, j)
	}
	return v
}

// A returns the block A[r,s] = (ri|sj); s runs over the virtual group of symmetry
// sym, r over the group fixed by the integral symmetry (irrep(i)⊗irrep(j)⊗sym).
func (s *Store) A(i, j, sym int) backend.Mat {
	rows := s.virGroup(symProduct(s.orbSym[i], s.orbSym[j], sym))
	cols := s.virGroup(sym)
	m := backend.NewMat(len(rows), len(cols))
	for ri, rp := range rows {
		for ci, sp := range cols {
			m.Set(ri, ci, s.d.TwoE(s.nocc+rp, i, s.nocc+sp, j))
		}
	}
	return m
}

// B returns the block B[r,s] = (rs|ij); s runs over the virtual group of symmetry
// sym, r over the group fixed by the integral symmetry (irrep(i)⊗irrep(j)⊗sym).
func (s *Store) B(i, j, sym int) backend.Mat {
	rows := s.virGroup(symProduct(s.orbSym[i], s.orbSym[j], sym))
	cols := s.virGroup(sym)
	m := backend.NewMat(len(rows), len(cols))
	for ri, rp := range rows {
		for ci, sp := range cols {
			m.Set(ri, ci, s.d.TwoE(s.nocc+rp, s.nocc+sp, i, j))
		}
	}
	return m
}

// VirGroup returns the ordered virtual positions (0-based, ascending) of symmetry
// group sym — the ordering the A/B/V blocks and diagEnergies index by (absolute
// orbital = Nocc+position). It backs the matrix-free satellite path, which needs a
// group position's absolute orbital to recompute a block entry from the ERIs.
func (s *Store) VirGroup(sym int) []int { return s.virGroup(sym) }

// virGroup returns the ordered virtual positions of symmetry group sym (an empty
// slice for an out-of-range group).
func (s *Store) virGroup(sym int) []int {
	if sym < 0 || sym >= s.nsym {
		return nil
	}
	return s.virBySym[sym]
}
