package dip

import (
	"math"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
)

// Spin-coupling coefficients from the reference (singlet.cpp:4-12).
var (
	sqrt2       = math.Sqrt2
	sqrt3       = math.Sqrt(3)
	sqrt1_2     = math.Sqrt(0.5)
	sqrt3_2     = math.Sqrt(1.5)
	sqrt3_4     = math.Sqrt(0.75)
	threehalves = 1.5
)

// blocks is the ADC2-DIP building-block interface, mirroring ADC2_DIP_blocks
// (../ADC/adc2_dip/adc2_dip_blocks.hpp). Scalar 2h/2h blocks return an element;
// coupling and satellite blocks return a dense sub-block (a column vector for
// the 2h↔3h1p couplings, a matrix for 3h1p/3h1p). The bool reports whether the
// block is nonzero (the reference's return value).
//
// Row/col Configs are group representatives: the block spans all virtuals (and
// spin parts) of the group, so only the occupied indices and the virtual
// symmetry are read.
type blocks interface {
	iiJJ(row, col Config) (float64, bool)
	ijKK(row, col Config) (float64, bool)
	ijKL(row, col Config) (float64, bool)
	lkkII(row, col Config) (backend.Mat, bool)
	lkkIJ(row, col Config) (backend.Mat, bool)
	klmII(row, col Config) (backend.Mat, bool)
	klmIJ(row, col Config) (backend.Mat, bool)
	jiiLKK(row, col Config) (backend.Mat, bool)
	ijkMLL(row, col Config) (backend.Mat, bool)
	ijkLMN(row, col Config) (backend.Mat, bool)
}

// base holds the SCF/integral data shared by the singlet and triplet block
// implementations and the shared u/w perturbation sums.
type base struct {
	sp   *Space
	ints *integrals.Store
	eps  []float64 // orbital energies (absolute index)
}

func (b *base) energy(o int) float64 { return b.eps[o] }
func (b *base) nocc() int            { return b.sp.Nocc }
func (b *base) norb() int            { return b.sp.Norb }
func (b *base) nvir() int            { return b.sp.Nvir }

// symOrb is the 0-based irrep of an absolute orbital index.
func (b *base) symOrb(o int) int { return b.sp.irrep(o) }

// V1122 and its (anti)symmetrized combinations (adc2_dip_blocks.hpp:39-46).
func (b *base) v(p, q, r, s int) float64      { return b.ints.Eri(p, q, r, s) }
func (b *base) vplus(p, q, r, s int) float64  { return b.ints.EriPlus(p, q, r, s) }
func (b *base) vminus(p, q, r, s int) float64 { return b.ints.EriMinus(p, q, r, s) }

// A, B, V integral blocks (adc2_dip_blocks.hpp:69-76). sym is the virtual
// symmetry group: for A/B the column (s) group with the row (r) group fixed by
// the integral symmetry; for V the single virtual index's group directly.
func (b *base) A(i, j, sym int) backend.Mat  { return b.ints.A(i, j, sym) }
func (b *base) B(i, j, sym int) backend.Mat  { return b.ints.B(i, j, sym) }
func (b *base) V(i, j, k, sym int) []float64 { return b.ints.V(i, j, k, sym) }

// sizeVirGroup is the number of virtual orbitals in symmetry group sym; it sets
// the row/column dimensions of the satellite building blocks.
func (b *base) sizeVirGroup(sym int) int { return b.ints.SizeVirGroup(sym) }

// virSym is the symmetry group of a 3h1p configuration's virtual orbital (its
// group representative determines the block's virtual dimension).
func (b *base) virSym(c Config) int { return b.symOrb(b.nocc() + c.Vir) }

// diagEnergies is the vector of virtual-orbital energies for symmetry group sym,
// ordered to match that group's building-block rows (adc2_dip_blocks.cpp:36-42).
func (b *base) diagEnergies(sym int) []float64 {
	var d []float64
	for rp := range b.nvir() {
		if b.symOrb(b.nocc()+rp) == sym {
			d = append(d, b.eps[b.nocc()+rp])
		}
	}
	return d
}

// wTerm is the 2nd-order W double-sum (singlet.cpp:15-43; identical for
// triplet, triplet.cpp:9-37).
func (b *base) wTerm(i, k, s, r int) float64 {
	var result float64
	for m := range b.nocc() {
		if b.symOrb(m) != symProduct(b.symOrb(i), b.symOrb(s), b.symOrb(r)) {
			continue
		}
		ers := b.energy(r) + b.energy(s)
		term := ers - b.energy(m) - 0.5*(b.energy(i)+b.energy(k))
		term /= (ers - (b.energy(i) + b.energy(m))) * (ers - (b.energy(k) + b.energy(m)))
		term *= b.v(r, i, s, m)*b.v(r, k, s, m) +
			b.v(r, m, s, i)*b.v(r, m, s, k) +
			b.vminus(r, i, s, m)*b.vminus(r, k, s, m)
		if r == s {
			result += 0.5 * term
		} else {
			result += term
		}
	}
	return result
}

// uTerm is the 2nd-order U double-sum (singlet.cpp:46-63 / triplet.cpp:40-59).
// plus selects V1122_PLUS (singlet) vs V1122_MINUS (triplet).
func (b *base) uTerm(i, j, k, l, s, r int, plus bool) float64 {
	ers := b.energy(r) + b.energy(s)
	term := ers - 0.5*(b.energy(i)+b.energy(j)+b.energy(k)+b.energy(l))
	term /= (ers - (b.energy(i) + b.energy(j))) * (ers - (b.energy(k) + b.energy(l)))
	if plus {
		term *= b.vplus(r, i, s, j) * b.vplus(r, k, s, l)
	} else {
		term *= b.vminus(r, i, s, j) * b.vminus(r, k, s, l)
	}
	if r == s {
		return 0.5 * term
	}
	return term
}
