package dip

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
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

	// Gate variants report the same nonzero decision and the block's dimensions
	// (rows, cols) without evaluating any integrals — the cheap primitive the
	// matrix-free satellite path uses to size and to prune candidate blocks. The
	// value methods above call these, so the nonzero test has one source of truth.
	jiiLKKGate(row, col Config) (rows, cols int, ok bool)
	ijkMLLGate(row, col Config) (rows, cols int, ok bool)
	ijkLMNGate(row, col Config) (rows, cols int, ok bool)

	// Elem variants return one scalar entry of a satellite block: the (row-part pr,
	// row-virtual ra),(col-part pc, col-virtual sb) element, with ra/sb absolute
	// virtual orbital indices. They are the per-entry form of the value blocks (a GPU
	// thread owns one output scalar and recomputes just its entry from the ERIs), and
	// the single source of truth the CUDA kernel transcribes (satelem.go). jiiLKK has
	// one spin part per side; ijkMLL's column side has one (pc≡0).
	jiiLKKElem(row, col Config, ra, sb int) float64
	ijkMLLElem(row, col Config, pr, ra, sb int) float64
	ijkLMNElem(row, col Config, pr, ra, pc, sb int) float64

	// virSym is a config's virtual symmetry group; virOrbs is that group's absolute
	// virtual orbitals in block row/column order. The matrix-free applier and the Elem
	// callers use them to map a 3h1p config's virtual position to its orbital.
	virSym(c Config) int
	virOrbs(sym int) []int

	// ensureSecondOrder materializes the W/U tables the 2h/2h elements read. Callers hoist it
	// out of a worker pool before evaluating those elements, so the table build (itself
	// parallel) gets the whole pool rather than nesting inside one worker.
	ensureSecondOrder()
}

// base holds the SCF/integral data shared by the singlet and triplet block
// implementations and the shared u/w perturbation sums.
type base struct {
	sp   *Space
	ints *integrals.Store
	eps  []float64 // orbital energies (absolute index)

	// sec is a POINTER so that copying a base (matvec.go New does `&singlet{b}`) copies the
	// handle, not the sync.Once inside it.
	sec *secondOrder
}

// secondOrder holds the precomputed second-order W and U tables the 2h/2h elements are built
// from, plus the per-symmetry virtual-group lookups the block builders would otherwise rebuild
// on every call.
//
// WHY. Both W and U were evaluated inside every one of the main²/2 element builds, each an
// nvir²/2 sweep over virtual pairs. At the production system's main=1711, nvir=154, nocc=58 that is ~1.15e12
// strided ERI loads on ONE core — hours of single-threaded work on the critical path of the
// first apply, in a block that matvec.go's comment waves off as "small next to the satellite
// region". It is small in DIMENSION, not in cost.
//
// W(i,k) depends only on an occupied PAIR — 3,364 distinct values for the production system — so it is
// tabulated once instead of recomputed per element (~60× fewer evaluations, and the sweep
// parallelizes over the table). U genuinely depends on all four occupied indices, but it
// factorizes exactly into a GEMM; see buildSecondOrder.
type secondOrder struct {
	once sync.Once

	wTab         []float64 // nocc·nocc, indexed i*nocc+k
	uTab         []float64 // npair·npair over canonical occupied pairs (pairIndex)
	pairI, pairJ []int32
	npair        int

	virOrbs  [][]int     // per symmetry group, absolute virtual orbitals in block order
	diagEner [][]float64 // per symmetry group, that group's virtual orbital energies
}

// newSecondOrder builds the cheap per-symmetry tables eagerly; W and U are built on first use
// (they cost real work and the satellite-only paths never touch them).
func newSecondOrder(sp *Space, eps []float64) *secondOrder {
	// Group count exactly as integrals.Store derives it: the largest 0-based irrep label
	// present, rounded up to a power of two. Space.orbSym holds the 1-based FCIDUMP labels, so
	// go through irrep(); with symmetry off it returns 0 for every orbital and nsym is 1.
	maxLabel := 0
	for o := range sp.Norb {
		if lab := sp.irrep(o); lab > maxLabel {
			maxLabel = lab
		}
	}
	nsym := 1
	for nsym < maxLabel+1 {
		nsym <<= 1
	}
	sec := &secondOrder{
		virOrbs:  make([][]int, nsym),
		diagEner: make([][]float64, nsym),
	}
	for σ := range nsym {
		for rp := range sp.Nvir {
			orb := sp.Nocc + rp
			if sp.irrep(orb) != σ {
				continue
			}
			sec.virOrbs[σ] = append(sec.virOrbs[σ], orb)
			sec.diagEner[σ] = append(sec.diagEner[σ], eps[orb])
		}
	}
	return sec
}

// pairIndex maps a canonical occupied pair (a ≥ b) to its slot in uTab's square. Every 2h config
// stores its pair that way — |ii⟩ as Occ={i,i}, |ij⟩ as Occ={i,j} with i>j (config.go) — so no
// caller ever needs to reorder, which matters because the triplet's vminus kernel is
// ANTISYMMETRIC under the swap.
func pairIndex(a, b int) int { return a*(a+1)/2 + b }

// ensureSecondOrder builds the W/U tables once, whichever goroutine asks first.
func (b *base) ensureSecondOrder() { b.sec.once.Do(b.buildSecondOrder) }

// wAt is Σ_{r≥s} wTerm(i,k,s,r) — the second-order W double sum for one occupied pair.
func (b *base) wAt(i, k int) float64 { return b.sec.wTab[i*b.nocc()+k] }

// uAt is Σ_{r≥s} uTerm(i,j,k,l,s,r) over the sector's symmetry channel.
func (b *base) uAt(i, j, k, l int) float64 {
	return b.sec.uTab[pairIndex(i, j)*b.sec.npair+pairIndex(k, l)]
}

// buildSecondOrder materializes wTab and uTab.
//
// W is a straight memoization: the (r,s) sweep per pair is byte-for-byte the loop the element
// builders ran, so each table entry is bit-identical to what they computed — only the number of
// times it is computed changes.
//
// U IS A GEMM. Write x = ε_r+ε_s, e_p = ε_i+ε_j for the row pair and e_q = ε_k+ε_l for the
// column pair. uTerm's energy factor is
//
//	[x − ½(e_p+e_q)] / [(x−e_p)(x−e_q)]
//
// and since x − ½(e_p+e_q) = ½[(x−e_p) + (x−e_q)], that is exactly ½[1/(x−e_q) + 1/(x−e_p)] —
// a SUM OF A ROW-ONLY AND A COLUMN-ONLY FACTOR, which is what makes the double sum separable.
// The integral factor is already separable: vplus(r,i,s,j)·vplus(r,k,s,l) is P_p[t]·P_q[t] over
// the flattened (r,s) list t. So with
//
//	X[p,t] = P_p[t]                        (P = vplus for singlet, vminus for triplet)
//	Y[p,t] = w_t · P_p[t] / (x_t − e_p)    (w_t = ½ on the r==s diagonal, else 1)
//
// the whole table is U = ½(Y·Xᵀ + X·Yᵀ) = ½(M + Mᵀ) with M = Y·Xᵀ — ONE dgemm. At production scale
// that is 2·1711²·11,935 ≈ 7.0e10 flops in place of ~7.0e10 scattered ERI loads with two
// divisions each, i.e. ~1,700× fewer integral reads.
//
// The (r,s) list is the sector's symmetry channel σ_rs == Sym, which is what all three element
// builders' guards reduce to: ijKL gates on σ_rs == σ_ij and addIJ admits a config only when
// σ_ij == Sym, while iiJJ/ijKK gate on σ_rs == 0 and addII admits their family only when
// Sym == 0.
//
// This reassociates the sum, so U is no longer bit-identical to the per-element form — it is
// equal to ~1 ulp per term. TestSecondOrderTablesMatchDirectSums pins it against the direct
// double sums, and the theADCcode-matched reference spectra bound the end-to-end effect.
func (b *base) buildSecondOrder() {
	nocc, norb := b.nocc(), b.norb()
	sec := b.sec

	// --- W(i,k): pure memoization, same sweep order as the element builders. ---
	sec.wTab = make([]float64, nocc*nocc)
	parallel.HeavyRows(nocc*nocc, func(t int) {
		i, k := t/nocc, t%nocc
		var acc float64
		for r := nocc; r < norb; r++ {
			for s := nocc; s <= r; s++ {
				acc += b.wTerm(i, k, s, r)
			}
		}
		sec.wTab[t] = acc
	})

	// --- U(p,q): the GEMM. ---
	npair := nocc * (nocc + 1) / 2
	sec.npair = npair
	sec.uTab = make([]float64, npair*npair)
	sec.pairI, sec.pairJ = make([]int32, npair), make([]int32, npair)
	for a := range nocc {
		for c := 0; c <= a; c++ {
			sec.pairI[pairIndex(a, c)], sec.pairJ[pairIndex(a, c)] = int32(a), int32(c)
		}
	}

	var rsR, rsS []int32
	var rsW, rsX []float64
	for r := nocc; r < norb; r++ {
		for s := nocc; s <= r; s++ {
			if symProduct(b.symOrb(r), b.symOrb(s)) != b.sp.Sym {
				continue
			}
			w := 1.0
			if r == s {
				w = 0.5
			}
			rsR, rsS = append(rsR, int32(r)), append(rsS, int32(s))
			rsW = append(rsW, w)
			rsX = append(rsX, b.energy(r)+b.energy(s))
		}
	}
	nt := len(rsR)
	if nt == 0 {
		return // no (r,s) pair in this sector's channel: every U term was gated out
	}

	x := make([]float64, npair*nt)
	y := make([]float64, npair*nt)
	minus := b.sp.Spin == Triplet
	parallel.HeavyRows(npair, func(p int) {
		i, j := int(sec.pairI[p]), int(sec.pairJ[p])
		ep := b.energy(i) + b.energy(j)
		xr, yr := x[p*nt:(p+1)*nt], y[p*nt:(p+1)*nt]
		for t := range nt {
			r, s := int(rsR[t]), int(rsS[t])
			v := b.vplus(r, i, s, j)
			if minus {
				v = b.vminus(r, i, s, j)
			}
			xr[t] = v
			yr[t] = rsW[t] * v / (rsX[t] - ep)
		}
	})

	m := make([]float64, npair*npair)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1,
		blas64.General{Rows: npair, Cols: nt, Stride: nt, Data: y},
		blas64.General{Rows: npair, Cols: nt, Stride: nt, Data: x},
		0, blas64.General{Rows: npair, Cols: npair, Stride: npair, Data: m})

	parallel.HeavyRows(npair, func(p int) {
		for q := range npair {
			sec.uTab[p*npair+q] = 0.5 * (m[p*npair+q] + m[q*npair+p])
		}
	})
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

// virOrbs returns the absolute virtual orbital indices of symmetry group sym, in the
// block row/column order (position a ↔ virOrbs(sym)[a]). The matrix-free satellite path
// maps a 3h1p config's virtual position to its absolute orbital to recompute a block
// entry from the ERIs (satelem.go); it is the ordering diagEnergies also uses.
// The returned slice is SHARED and must not be mutated: it was allocated per call and rebuilt
// from scratch on every one, including once per block application in the device SoA marshal.
func (b *base) virOrbs(sym int) []int {
	if sym < 0 || sym >= len(b.sec.virOrbs) {
		return nil // matches integrals.Store.virGroup's out-of-range behaviour
	}
	return b.sec.virOrbs[sym]
}

// diagEnergies is the vector of virtual-orbital energies for symmetry group sym,
// ordered to match that group's building-block rows (adc2_dip_blocks.cpp:36-42).
// The returned slice is SHARED and must not be mutated. It used to be append-grown from a full
// nvir scan on every call, i.e. inside every diagonal jiiLKK/ijkLMN build — which under the
// matrix-free path is once per block PER MAT-VEC.
func (b *base) diagEnergies(sym int) []float64 {
	if sym < 0 || sym >= len(b.sec.diagEner) {
		return nil
	}
	return b.sec.diagEner[sym]
}

// Satellite gate/shape helpers. The nonzero (Kronecker-δ) guard conditions are
// identical for singlet and triplet — only the number of spin parts differs, which
// scales the row (and, for ijkLMN, the column) dimension. So the guards live here on
// the shared base, parametrized by parts (2 for singlet, 3 for triplet), and the
// per-spin gate methods just supply parts. Each returns (rows, cols, nonzero) using
// only occupied-index equality and virtual-symmetry group sizes — no integrals.

// jiiLKKShape gates the 3h1p-I × 3h1p-I block (Table A.3): nvR × nvC (spin-part
// independent).
func (b *base) jiiLKKShape(row, col Config) (int, int, bool) {
	j, i := row.Occ[0], row.Occ[1]
	l, k := col.Occ[0], col.Occ[1]
	deltaIL, deltaJL := i == l, j == l
	deltaIK, deltaJK := i == k, j == k
	rowSym, colSym := b.virSym(row), b.virSym(col)
	deltaSym := rowSym == colSym
	if !(deltaIK || (deltaIK && deltaJL) || (deltaIL && deltaJK) ||
		(deltaSym && (deltaIK || deltaIL || deltaJK || deltaJL))) {
		return 0, 0, false
	}
	return b.sizeVirGroup(rowSym), b.sizeVirGroup(colSym), true
}

// ijkMLLShape gates the 3h1p-II × 3h1p-I block (Tables A.4/A.5): parts·nvR × nvC.
func (b *base) ijkMLLShape(row, col Config, parts int) (int, int, bool) {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	m, l := col.Occ[0], col.Occ[1]
	deltaIM, deltaJM, deltaKM := i == m, j == m, k == m
	deltaIL, deltaJL, deltaKL := i == l, j == l, k == l
	rowSym, colSym := b.virSym(row), b.virSym(col)
	deltaSym := rowSym == colSym
	if !((deltaIM && deltaJL) || (deltaIM && deltaKL) ||
		(deltaJM && deltaIL) || (deltaJM && deltaKL) ||
		(deltaKM && deltaIL) || (deltaKM && deltaJL) ||
		(deltaSym && (deltaIM || deltaIL || deltaJM || deltaJL || deltaKM || deltaKL))) {
		return 0, 0, false
	}
	return parts * b.sizeVirGroup(rowSym), b.sizeVirGroup(colSym), true
}

// ijkLMNShape gates the 3h1p-II × 3h1p-II block (Table A.6): parts·nvR × parts·nvC.
func (b *base) ijkLMNShape(row, col Config, parts int) (int, int, bool) {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	l, m, n := col.Occ[0], col.Occ[1], col.Occ[2]
	deltaIL, deltaJL, deltaKL := i == l, j == l, k == l
	deltaJM, deltaKM := j == m, k == m
	deltaJN, deltaKN := j == n, k == n
	rowSym, colSym := b.virSym(row), b.virSym(col)
	deltaSym := rowSym == colSym
	if !((deltaIL && deltaJM) || (deltaIL && deltaKM) || (deltaIL && deltaKN) ||
		(deltaJL && deltaKM) || (deltaJL && deltaKN) || (deltaJM && deltaKN) ||
		(deltaSym && (deltaIL || deltaJL || deltaJM || deltaJN || deltaKL || deltaKM || deltaKN))) {
		return 0, 0, false
	}
	return parts * b.sizeVirGroup(rowSym), parts * b.sizeVirGroup(colSym), true
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
