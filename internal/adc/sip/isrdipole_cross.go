package sip

import (
	"fmt"
	"math"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// Transition dipoles between eigenstates of two *different* SIP configuration spaces
// (docs/adc4_rassi_plan.md, Chunk 5). Two regimes, one mechanism:
//
//   - cross-sector: a plain Space of one target irrep against a plain Space of another.
//     A core or inner-valence hole and an outer-valence hole generally carry different
//     irreps (O 1s is a₁ in H2O, 1b₁ is b₁), so even the plain X-ray-emission element is
//     out of reach of the square same-space D — inside one sector only the totally
//     symmetric dipole component survives.
//   - cross-space: a CVS ADC(4) Space4 (core hole) against a plain valence Space.
//
// Two things make this more than a shape change.
//
// Spin-function conventions. The 2h1p spin functions are the same in both spaces up to
// two sign conventions that must be undone before the block formulas of isrdipole.go
// apply: type II is antisymmetric under k↔l, so a space's (k,l) role order matters; and
// elements4.go's ADC(4) elements use the reference's Dyson convention, in which type II
// is negated relative to elements.go (compare kopp1's +√(3/2)(A1−A2) with c12_1's
// −√(3/2)(A1−A2), and c22elem4's +√(3/4)(a4−a6) with deltaV's −√(3/4)·v2). canonical
// folds both into one sign. TestADC4SpinFunctionsReproduceKopp1 pins the ADC(4) factor
// against a bit-exact secular element rather than assuming it.
//
// Non-orthogonality. Bra and ket are eigenvectors of different, non-commuting truncated
// Hamiltonians, so their overlap S = X_i† S_config X_m need not vanish, and the nuclear
// term of the dipole operator no longer drops out of the transition moment:
//
//	μ = μ_nuc·S − ⟨Ψ_i|D̂|Ψ_m⟩
//
// A cation carries charge +1, so shifting the gauge origin by δ moves μ by −δ·S: the
// moment is origin-dependent whenever S ≠ 0. For the emission this chunk exists to
// describe — different irreps — no configuration is shared, S vanishes identically, and μ
// is well defined. When the two sectors *do* share an irrep (O 1s → 2a₁), S ≠ 0 is an
// artifact of the CVS projection; CrossEmission reports it so the number can be judged.
//
// The 3h2p truncation. A one-particle operator connects 3h2p to 2h1p (c†_a c_K creates a
// particle and a core hole), so ⟨3h2p_CVS|D̂|2h1p_val⟩ is a real block. It is dropped.
// Counting orders in the fluctuation potential: a core main state has X_1h ~ O(1),
// X_2h1p ~ O(V), X_3h2p ~ O(V²); a valence main state has X_2h1p ~ O(V). The retained
// blocks contribute at O(1), O(V) and O(V²), and since ⟨3h2p|D̂|1h⟩ vanishes by
// particle-hole rank, the dropped term is the sole O(V³) contribution. TestNeglected3h2pBound
// turns that argument into a number via Cauchy–Schwarz. It is subordinate anyway to the
// deferred first-order ISR density corrections, which touch the *dominant* 1h×1h element
// at O(V).

// canonical returns c with its holes in ascending order, together with the sign of sp's
// spin function relative to that canonical one. Type I (and the closed-hole single, and
// every 1h config) is invariant; type II picks up a minus for each of an unsorted role
// order and an ADC(4) Dyson-convention space.
func canonical(sp *Space, c Config) (Config, float64) {
	sign := 1.0
	if c.Typ == 1 {
		if c.Occ[0] > c.Occ[1] {
			sign = -sign // |akl;II⟩ = −|alk;II⟩
		}
		if sp.adc4 {
			sign = -sign // elements4.go negates type II relative to elements.go
		}
	}
	if c.Occ[0] > c.Occ[1] {
		c.Occ[0], c.Occ[1] = c.Occ[1], c.Occ[0]
	}
	return c, sign
}

// compatible reports whether two spaces describe the same orbitals, so that a
// configuration shared between them means the same determinant expansion.
func compatible(bra, ket *Space) error {
	if bra.Nocc != ket.Nocc || bra.Norb != ket.Norb {
		return fmt.Errorf("sip: spaces disagree on the orbital set (%d/%d vs %d/%d occ/orb)",
			bra.Nocc, bra.Norb, ket.Nocc, ket.Norb)
	}
	for o := range bra.Norb {
		if bra.irrep(o) != ket.irrep(o) {
			return fmt.Errorf("sip: spaces disagree on the irrep of orbital %d", o)
		}
	}
	return nil
}

// NewISRDipoleCross builds the rectangular D between the configurations of bra and ket
// for the MO-basis operator dmo. The two spaces must span the same orbitals; the ket may
// not carry a 3h2p space (its columns are not implemented), the bra may — those rows come
// back zero, per the truncation documented above.
func NewISRDipoleCross(bra, ket *Space, dmo backend.Mat) (*ISRDipole, error) {
	if err := checkDipole(bra, dmo); err != nil {
		return nil, err
	}
	if err := compatible(bra, ket); err != nil {
		return nil, err
	}
	if len(ket.Sat3) != 0 {
		return nil, fmt.Errorf("sip: the ket space's 3h2p columns are not implemented; " +
			"put the CVS ADC(4) space on the bra side")
	}
	o := &ISRDipole{bra: bra, ket: ket, d: dmo}
	for i := range ket.Nocc {
		o.d0 += 2 * dmo.At(i, i)
	}
	o.dnull = o.d0 // no correlation: (13a) is the reference moment
	o.rows = o.sparsify()
	return o, nil
}

// NewISRDipolesCross builds all three Cartesian components at once.
func NewISRDipolesCross(bra, ket *Space, dmo [3]backend.Mat) ([3]*ISRDipole, error) {
	var ops [3]*ISRDipole
	for x := range 3 {
		op, err := NewISRDipoleCross(bra, ket, dmo[x])
		if err != nil {
			return ops, fmt.Errorf("component %d: %w", x, err)
		}
		ops[x] = op
	}
	return ops, nil
}

// ISROverlap is ⟨Φ_I^bra|Φ_J^ket⟩: the two configuration bases are built from the same
// orthonormal determinants, so it is ±1 on shared configurations and 0 everywhere else.
type ISROverlap struct {
	bra, ket *Space
	rows     [][]dipEntry
}

// ConfigOverlap builds the bra×ket configuration overlap. A bra 3h2p configuration
// overlaps nothing in a plain ket space, so those rows are empty.
func ConfigOverlap(bra, ket *Space) (*ISROverlap, error) {
	if err := compatible(bra, ket); err != nil {
		return nil, err
	}
	type key struct{ lo, hi, vir, typ int }
	index := make(map[key]int)
	for j := ket.BeginSat; j < len(ket.Configs); j++ {
		c, _ := canonical(ket, ket.Configs[j])
		index[key{c.Occ[0], c.Occ[1], c.Vir, c.Typ}] = j
	}
	mainIndex := make(map[int]int)
	for j := range ket.BeginSat {
		mainIndex[ket.Configs[j].Occ[0]] = j
	}

	o := &ISROverlap{bra: bra, ket: ket, rows: make([][]dipEntry, bra.Size())}
	for i := range bra.BeginSat {
		if j, ok := mainIndex[bra.Configs[i].Occ[0]]; ok {
			o.rows[i] = []dipEntry{{j, 1}}
		}
	}
	for i := bra.BeginSat; i < len(bra.Configs); i++ {
		r, sr := canonical(bra, bra.Configs[i])
		j, ok := index[key{r.Occ[0], r.Occ[1], r.Vir, r.Typ}]
		if !ok {
			continue
		}
		_, sc := canonical(ket, ket.Configs[j])
		o.rows[i] = []dipEntry{{j, sr * sc}}
	}
	return o, nil
}

// Braket returns braᵀ·S·ket, the overlap of two states from the two spaces.
func (o *ISROverlap) Braket(bra, ket []float64) float64 {
	var acc float64
	for i, row := range o.rows {
		for _, e := range row {
			acc += bra[i] * e.val * ket[e.col]
		}
	}
	return acc
}

// Nonzero counts the shared configurations. Zero means the two spaces are orthogonal by
// construction — the case in which a cross-space transition dipole is origin-independent.
func (o *ISROverlap) Nonzero() int {
	var n int
	for _, row := range o.rows {
		n += len(row)
	}
	return n
}

// CrossEmission is one radiative transition between eigenstates of two different spaces:
// an initial (core- or inner-hole) state of the bra space decaying to a middle
// (outer-valence-hole) state of the ket space.
type CrossEmission struct {
	Init, Mid int
	Omega     float64 // E_init − E_mid, hartree; positive for emission

	// Overlap is ⟨Ψ_init|Ψ_mid⟩. It vanishes identically when the two spaces share no
	// configuration (different irreps). When it does not, Mu depends on the gauge origin
	// — it shifts by −δ·Overlap — and the transition is not cleanly defined.
	Overlap float64

	Mu   [3]float64 // μ_nuc·Overlap − ⟨Ψ_init|D̂|Ψ_mid⟩, a.u.
	Osc  float64    // oscillator strength f
	Rate float64    // Einstein A coefficient, a.u. of inverse time
}

// CrossEmissions evaluates every (init, mid) pair. vi and vm are the full Ritz vectors of
// the bra and ket spaces (lanczos.Result.FullVecs); ei and em their eigenvalues; muNuc is
// mo.Data.NuclearDipole() about the same gauge origin the dipole integrals use.
func CrossEmissions(ops [3]*ISRDipole, ov *ISROverlap, muNuc [3]float64,
	ei, em []float64, vi, vm backend.Mat, inits, mids []int) ([]CrossEmission, error) {

	if vi.Rows != ops[0].Size() {
		return nil, fmt.Errorf("sip: initial-state eigenvectors have %d rows, want %d "+
			"(bra Space.Size); lanczos.Options.WantFull retains the satellite rows",
			vi.Rows, ops[0].Size())
	}
	if vm.Rows != ops[0].Cols() {
		return nil, fmt.Errorf("sip: middle-state eigenvectors have %d rows, want %d "+
			"(ket Space.Size); lanczos.Options.WantFull retains the satellite rows",
			vm.Rows, ops[0].Cols())
	}
	var out []CrossEmission
	for _, i := range inits {
		xi := matColumn(vi, i)
		for _, m := range mids {
			xm := matColumn(vm, m)
			e := CrossEmission{Init: i, Mid: m, Omega: ei[i] - em[m]}
			e.Overlap = ov.Braket(xi, xm)
			for c := range 3 {
				e.Mu[c] = muNuc[c]*e.Overlap - ops[c].Braket(xi, xm)
			}
			e.Osc = OscillatorStrength(e.Omega, e.Mu)
			e.Rate = EinsteinA(e.Omega, e.Osc)
			out = append(out, e)
		}
	}
	return out, nil
}

// Sat3Weight is ‖X_3h2p‖, the norm of a bra eigenvector's 3h2p amplitudes — the first
// factor of the Cauchy–Schwarz bound on the block CrossEmissions drops. It is zero for a
// plain (order ≤3) space.
func Sat3Weight(sp *Space, x []float64) float64 {
	if len(sp.Sat3) == 0 {
		return 0 // Begin3h2p is only set by NewSpace4
	}
	var acc float64
	for i := sp.Begin3h2p; i < sp.Size(); i++ {
		acc += x[i] * x[i]
	}
	return math.Sqrt(acc)
}
