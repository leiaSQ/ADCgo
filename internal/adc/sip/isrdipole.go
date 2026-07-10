package sip

import (
	"fmt"
	"math"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// The intermediate-state representation of a one-electron operator over the SIP
// configuration space, at leading order — the matrix of
//
//	D̂ = Σ_pq d_pq Σ_σ c†_pσ c_qσ
//
// between the bare (Hartree-Fock) configurations |Φ_I⟩ that the ADC secular matrix
// is built on. This is the ion↔ion half of the RASSI-like transition-moment chain
// (docs/adc4_rassi_plan.md, element 1): both bra and ket are (N−1)-electron states,
// so no Dyson orbital is involved and the moment is a plain μ = X_i† D X_m.
//
// Ported from theADCcode's ndadc3_prop (my_calc_d11.c/my_calc_d12.c/my_calc_d22_*.c,
// reached through ND_ADC3_CAP_matrix), with the correlation corrections dropped: the
// legacy 1h/1h block additionally carries the second-order self-energy terms (12a–c)
// and a ground-state-density term that needs ρ from an external self-energy module,
// and its 1h/2h1p block carries the first-order terms (8a–c). Those are deferred with
// the rest of the order-consistent ISR property expansion (see the "new physics"
// section of the plan); everything below is exact at zeroth order in the fluctuation
// potential and complete — no term of that order is missing.
//
// Spin functions. The eigenvectors X come from the secular matrix, so D must be built
// over exactly the configurations elements.go uses. Those are, for the M_S = +1/2
// doublet (|0⟩ the closed-shell reference, k,l occupied, a virtual):
//
//	|i⟩         = −c_iβ |0⟩                                                (see below)
//	|akk⟩       = c†_aα c_kα c_kβ |0⟩
//	|akl; I⟩    = (1/√2)  (c†_aα c_kα c_lβ − c†_aα c_kβ c_lα) |0⟩          (holes → singlet)
//	|akl; II⟩   = √(2/3) c†_aβ c_kβ c_lβ |0⟩
//	              + (1/√6)(c†_aα c_kα c_lβ + c†_aα c_kβ c_lα) |0⟩          (holes → triplet)
//
// TestSpinFunctionsReproduceSecularBlocks pins this down: it evaluates ⟨Φ_I|Ĥ|Φ_J⟩ by
// Slater–Condon over those determinant expansions and reproduces c12_1, c22diag and
// c22off. The leading minus on |i⟩ is fixed by c12_1's sign, not chosen. Type I is
// symmetric and type II antisymmetric under k↔l, which is why the (k,l) role order a
// Space assigns to a hole pair is load-bearing — see canonical in isrdipole_cross.go,
// which is where that order (and the ADC(4) space's opposite type-II phase) is undone
// before the block formulas below are applied.

// SpeedOfLight is c in atomic units (the inverse fine-structure constant, CODATA 2022).
const SpeedOfLight = 137.035999177

// ISRDipole is the leading-order ISR matrix D_IJ of one Cartesian component of the
// electronic one-particle operator, between a bra and a ket SIP configuration space. It
// is extremely sparse — a 2h1p row couples only to configurations that share a hole pair
// or differ in exactly one hole — so it is never materialized on the hotpath: At gives an
// element, Apply a mat-vec, BuildMatrix a dense copy for tests.
//
// When bra == ket (NewISRDipole) it is square and symmetric, and both indices run over
// one target-symmetry sector. The rectangular case (NewISRDipoleCross, isrdipole_cross.go)
// couples two sectors, or a CVS ADC(4) space to a valence one.
//
// The operator here is the *electronic* one, without the electron's charge. The dipole
// moment of a state is μ_nuc − ⟨D̂⟩; TransitionDipole applies that sign for you.
type ISRDipole struct {
	bra, ket *Space
	d        backend.Mat // MO-basis d_pq, Norb × Norb, symmetric
	d0       float64     // 2·Σ_{i<Nocc} d_ii — the closed-shell expectation ⟨0|D̂|0⟩

	rows [][]dipEntry // sparse D by bra row, built once
}

type dipEntry struct {
	col int
	val float64
}

// NewISRDipole builds D over sp for the MO-basis operator dmo (Norb × Norb, symmetric).
//
// dmo is one Cartesian component of mo.Data.DipMO. It need not be totally symmetric:
// a component that transforms as a non-trivial irrep simply has no matrix elements
// within a single target-symmetry sector, and D comes out identically zero. That is
// the physics, not a limitation — for H2O in C2v only μ_z connects a sector to itself.
// Use NewISRDipoleCross to reach the other components.
func NewISRDipole(sp *Space, dmo backend.Mat) (*ISRDipole, error) {
	if len(sp.Sat3) != 0 {
		return nil, fmt.Errorf("sip: ISR dipole over a 3h2p (ADC(4)) space is not implemented")
	}
	return NewISRDipoleCross(sp, sp, dmo)
}

// checkDipole validates one Cartesian component of the MO-basis operator against a space.
func checkDipole(sp *Space, dmo backend.Mat) error {
	if dmo.Rows != sp.Norb || dmo.Cols != sp.Norb {
		return fmt.Errorf("sip: dipole matrix is %d×%d, want %d×%d (Space.Norb)",
			dmo.Rows, dmo.Cols, sp.Norb, sp.Norb)
	}
	for p := range sp.Norb {
		for q := range p {
			if math.Abs(dmo.At(p, q)-dmo.At(q, p)) > 1e-10 {
				return fmt.Errorf("sip: dipole matrix asymmetric at (%d,%d): %g vs %g",
					p, q, dmo.At(p, q), dmo.At(q, p))
			}
		}
	}
	return nil
}

// D0 is ⟨0|D̂|0⟩ = 2·Σ_{i occupied} d_ii, the reference expectation value. It sits on
// every diagonal element of D and therefore cancels out of every transition moment; it
// only shifts state expectation values. A frozen-core FCIDUMP has no core orbitals to
// trace over, so its D0 — and only its D0 — is short by the core contribution.
func (o *ISRDipole) D0() float64 { return o.d0 }

// Size is the number of bra rows, == bra Space.Size(). For the square (same-space) case
// it is the matrix dimension.
func (o *ISRDipole) Size() int { return o.bra.Size() }

// Cols is the number of ket columns, == ket Space.Size().
func (o *ISRDipole) Cols() int { return o.ket.Size() }

// vir maps a Config's virtual position to an absolute orbital index.
func (o *ISRDipole) vir(c Config) int { return o.ket.Nocc + c.Vir }

// At returns D_IJ, with I a bra configuration and J a ket one.
//
// Every element is evaluated on *canonicalized* configurations — holes in ascending
// order, spin functions in the plain (non-Dyson) convention — with the two phases put
// back afterwards. The block formulas below are only self-consistent when both sides use
// the same role order, which two different spaces need not.
func (o *ISRDipole) At(i, j int) float64 {
	if i >= len(o.bra.Configs) || j >= len(o.ket.Configs) {
		return 0 // 3h2p rows: truncated, see isrdipole_cross.go
	}
	ri, cj := o.bra.Configs[i], o.ket.Configs[j]
	bi, bj := i < o.bra.BeginSat, j < o.ket.BeginSat
	switch {
	case bi && bj:
		p, q := ri.Occ[0], cj.Occ[0]
		v := -o.d.At(p, q)
		if p == q {
			v += o.d0 // the same configuration, however each space indexes it
		}
		return v
	case bi:
		c, s := canonical(o.ket, cj)
		return s * o.mainSat(ri.Occ[0], c)
	case bj:
		r, s := canonical(o.bra, ri)
		return s * o.mainSat(cj.Occ[0], r)
	default:
		r, sr := canonical(o.bra, ri)
		c, sc := canonical(o.ket, cj)
		return sr * sc * o.satsat(r, c)
	}
}

// mainSat is ⟨i|D̂|c⟩ for a 1h main config i and a 2h1p config c. A one-particle
// operator reaches the 1h space from 2h1p only by annihilating the particle into one
// of the two holes, so only the occupied–virtual block of d contributes and only when
// i is itself one of the holes.
//
// Every sign here is the opposite of theADCcode's cap_calc_d12 (7a)/(7b), and that is
// deliberate. The relative phase of the 1h and 2h1p configurations is not free: it is
// the sign of the secular coupling block, which calc_c12_1.c fixes. Evaluating ⟨i|Ĥ|c⟩
// over the determinant expansions of those very configurations reproduces calc_c12_1
// only with the 1h phase this block assumes, and then ⟨i|D̂|c⟩ comes out negated
// relative to the legacy kernel. ndadc3_prop knows: nd_adc3_cap_matrix.cpp:335 carries a
// commented-out `(*d12) *= -1.;` under the note "a sign error in Joerg's code". The
// error is real, it is here, and it does not cancel — μ = X†DX mixes the main and
// satellite amplitudes of both states. TestSpinFunctionsReproduceSecularBlocks and
// TestISRDipoleMatchesDeterminants are what pin it down.
func (o *ISRDipole) mainSat(i int, c Config) float64 {
	k, l, a := c.Occ[0], c.Occ[1], o.vir(c)
	if k == l {
		if i != k {
			return 0
		}
		return -o.d.At(k, a)
	}
	switch {
	case i == k:
		if c.Typ == 0 {
			return -sqrt1_2 * o.d.At(l, a)
		}
		return sqrt3_2 * o.d.At(l, a)
	case i == l:
		if c.Typ == 0 {
			return -sqrt1_2 * o.d.At(k, a)
		}
		return -sqrt3_2 * o.d.At(k, a)
	}
	return 0
}

// satsat is ⟨r|D̂|c⟩ between two 2h1p configs, including the diagonal. Both must be
// canonicalized (holes ascending): the four hole-moving cases distinguish "the surviving
// hole of the bra is the first hole of the ket" from "…the second", so a bra and a ket
// that order the same physical hole pair differently would take the wrong branch. Within
// one space every hole pair carries one role order, which is why the same-space code got
// away without this for so long.
//
// Within the 2h1p
// space, a one-particle operator either moves the particle (leaving the holes alone) or
// moves one hole (leaving the particle alone); it can never do both, which is what
// makes the block so sparse.
//
// The four hole-moving cases below are theADCcode's (1c)–(1f). Two of them — (1d) and
// (1f), where the surviving hole of the bra is the *second* hole of the ket — differ in
// sign between the two spin functions, because |II⟩ changes sign under k↔l while |I⟩
// does not. The spin-off-diagonal elements ⟨I|D̂|II⟩ vanish identically in every case,
// which is why only x00 and x11 are accumulated.
func (o *ISRDipole) satsat(r, c Config) float64 {
	k, l, a := r.Occ[0], r.Occ[1], o.vir(r)
	m, n, b := c.Occ[0], c.Occ[1], o.vir(c)
	d := o.d
	rowPair, colPair := k != l, m != n

	switch {
	case rowPair && colPair:
		if r.Typ != c.Typ {
			return 0
		}
		var x float64
		if k == m && l == n {
			x += d.At(a, b) // (1b) particle transition
		}
		if a == b {
			if k == m {
				x -= d.At(l, n) // (1c)
			}
			if l == n {
				x -= d.At(k, m) // (1e)
			}
			sgn := 1.0 // (1d), (1f): the exchanged-hole terms, odd under k↔l
			if r.Typ == 1 {
				sgn = -1
			}
			if k == n {
				x -= sgn * d.At(m, l)
			}
			if l == m {
				x -= sgn * d.At(k, n)
			}
		}
		if a == b && k == m && l == n {
			x += o.d0 // (1a), diagonal only
		}
		return x

	case rowPair: // c is a (b,m,m) closed-hole single
		if a != b || r.Typ != 0 || (m != k && m != l) {
			return 0
		}
		return -math.Sqrt2 * d.At(k, l)

	case colPair: // r is a single; symmetric image of the case above
		if a != b || c.Typ != 0 || (k != m && k != n) {
			return 0
		}
		return -math.Sqrt2 * d.At(m, n)

	default: // both singles
		if k != m {
			return 0
		}
		if a == b {
			return o.d0 + d.At(a, a) - 2*d.At(k, k)
		}
		return d.At(a, b)
	}
}

// sparsify enumerates the nonzero ket columns of every bra row once. Rather than scanning
// all Size()×Cols() pairs it walks the short list of ket configurations a row can possibly
// reach — same hole pair with any particle, or one hole moved — and asks At for the value.
// Bra rows in the 3h2p space stay empty.
func (o *ISRDipole) sparsify() [][]dipEntry {
	bra, ket := o.bra, o.ket

	// byHoles indexes the ket satellites by (sorted hole pair, virtual position).
	type key struct{ lo, hi, vir int }
	byHoles := make(map[key][]int)
	holesOf := make([][]int, ket.Nocc) // ket satellites having a given orbital as a hole
	for idx := ket.BeginSat; idx < len(ket.Configs); idx++ {
		c := ket.Configs[idx]
		lo, hi := c.Occ[0], c.Occ[1]
		if lo > hi {
			lo, hi = hi, lo
		}
		byHoles[key{lo, hi, c.Vir}] = append(byHoles[key{lo, hi, c.Vir}], idx)
		holesOf[c.Occ[0]] = append(holesOf[c.Occ[0]], idx)
		if c.Occ[1] != c.Occ[0] {
			holesOf[c.Occ[1]] = append(holesOf[c.Occ[1]], idx)
		}
	}
	pair := func(p, q, v int) []int {
		if p > q {
			p, q = q, p
		}
		return byHoles[key{p, q, v}]
	}

	rows := make([][]dipEntry, bra.Size())
	push := func(i, j int) {
		if v := o.At(i, j); v != 0 {
			rows[i] = append(rows[i], dipEntry{j, v})
		}
	}
	for i := range bra.BeginSat {
		hole := bra.Configs[i].Occ[0]
		for j := range ket.BeginSat {
			push(i, j)
		}
		for _, j := range holesOf[hole] {
			push(i, j)
		}
	}
	for i := bra.BeginSat; i < len(bra.Configs); i++ {
		r := bra.Configs[i]
		k, l := r.Occ[0], r.Occ[1]
		for j := range ket.BeginSat {
			push(i, j)
		}
		for v := range ket.Nvir { // same holes, any particle (includes the diagonal)
			for _, j := range pair(k, l, v) {
				push(i, j)
			}
		}
		// One hole moved, particle fixed. h == k in the first sweep (h == l in the
		// second) reaches the closed-hole single; the h that would reproduce {k,l} is
		// skipped, since the sweep above already emitted it.
		for h := range ket.Nocc {
			if h != l {
				for _, j := range pair(k, h, r.Vir) {
					push(i, j)
				}
			}
			if l != k && h != k {
				for _, j := range pair(l, h, r.Vir) {
					push(i, j)
				}
			}
		}
	}
	return rows
}

// Apply computes out = D·in on host vectors: len(in) == Cols(), len(out) == Size(); out
// is overwritten.
func (o *ISRDipole) Apply(out, in []float64) {
	for i, row := range o.rows {
		var acc float64
		for _, e := range row {
			acc += e.val * in[e.col]
		}
		out[i] = acc
	}
}

// Braket returns braᵀ·D·ket, with len(bra) == Size() and len(ket) == Cols().
func (o *ISRDipole) Braket(bra, ket []float64) float64 {
	var acc float64
	for i, row := range o.rows {
		if bra[i] == 0 {
			continue
		}
		var s float64
		for _, e := range row {
			s += e.val * ket[e.col]
		}
		acc += bra[i] * s
	}
	return acc
}

// BuildMatrix materializes D densely, Size() × Cols() — for tests and small sectors only.
func (o *ISRDipole) BuildMatrix() backend.Mat {
	m := backend.NewMat(o.Size(), o.Cols())
	for i := range o.Size() {
		for j := range o.Cols() {
			m.Set(i, j, o.At(i, j))
		}
	}
	return m
}

// NewISRDipoles builds all three Cartesian components at once.
func NewISRDipoles(sp *Space, dmo [3]backend.Mat) ([3]*ISRDipole, error) {
	var ops [3]*ISRDipole
	for x := range 3 {
		op, err := NewISRDipole(sp, dmo[x])
		if err != nil {
			return ops, fmt.Errorf("component %d: %w", x, err)
		}
		ops[x] = op
	}
	return ops, nil
}

// ElectronicDipole is ⟨X|D̂|X⟩ per Cartesian component: the electronic part of the
// state's dipole expectation value, in a.u. and without the electron charge. The full
// state dipole is mo.Data.NuclearDipole() minus this.
func ElectronicDipole(ops [3]*ISRDipole, x []float64) [3]float64 {
	var mu [3]float64
	for c := range 3 {
		mu[c] = ops[c].Braket(x, x)
	}
	return mu
}

// TransitionDipole is ⟨X_bra|μ̂|X_ket⟩ in a.u. for two *distinct* eigenvectors. The
// nuclear term of μ̂ is a multiple of the identity and drops out between orthogonal
// states, so only the electronic part survives — with the electron's charge, hence the
// minus sign (the same flip RASSI applies to its MLTPL integrals).
func TransitionDipole(ops [3]*ISRDipole, bra, ket []float64) [3]float64 {
	var mu [3]float64
	for c := range 3 {
		mu[c] = -ops[c].Braket(bra, ket)
	}
	return mu
}

// OscillatorStrength is the dimensionless f = (2/3)·ω·|μ|², with ω in hartree and μ in
// a.u. (RASSI's convention).
func OscillatorStrength(omega float64, mu [3]float64) float64 {
	return 2.0 / 3.0 * omega * (mu[0]*mu[0] + mu[1]*mu[1] + mu[2]*mu[2])
}

// EinsteinA is the spontaneous-emission rate A = (2/c³)·ω²·f, in a.u. of inverse time
// (equivalently (4/3)·ω³|μ|²/c³). Divide by 4.1341373e16 to get s⁻¹.
func EinsteinA(omega, f float64) float64 {
	return 2 * omega * omega * f / (SpeedOfLight * SpeedOfLight * SpeedOfLight)
}

// Emission is one ion→ion radiative transition: the initial (inner-valence hole) state
// decaying to the middle (outer-valence hole) state of the same SIP sector.
type Emission struct {
	Init, Mid int        // state indices into the eigenpair list
	Omega     float64    // E_init − E_mid, hartree; positive for emission
	Mu        [3]float64 // transition dipole moment, a.u.
	Osc       float64    // oscillator strength f
	Rate      float64    // Einstein A coefficient, a.u. of inverse time
}

// Emissions evaluates every (init, mid) pair. vecs holds the full Ritz vectors, one per
// column, as returned by lanczos.Result.FullVecs (Options.WantFull) or by SolveDense;
// vals are the matching ionization energies. Pairs with init == mid are skipped, since
// μ there is a state expectation value and not a transition moment — use
// ElectronicDipole for those.
func Emissions(ops [3]*ISRDipole, vals []float64, vecs backend.Mat, inits, mids []int) ([]Emission, error) {
	if vecs.Rows != ops[0].Size() {
		return nil, fmt.Errorf("sip: eigenvectors have %d rows, want %d (Space.Size); "+
			"lanczos.Options.WantFull retains the satellite rows", vecs.Rows, ops[0].Size())
	}
	col := func(k int) []float64 {
		v := make([]float64, vecs.Rows)
		for r := range vecs.Rows {
			v[r] = vecs.At(r, k)
		}
		return v
	}
	var out []Emission
	for _, i := range inits {
		xi := col(i)
		for _, m := range mids {
			if i == m {
				continue
			}
			e := Emission{Init: i, Mid: m, Omega: vals[i] - vals[m]}
			e.Mu = TransitionDipole(ops, xi, col(m))
			e.Osc = OscillatorStrength(e.Omega, e.Mu)
			e.Rate = EinsteinA(e.Omega, e.Osc)
			out = append(out, e)
		}
	}
	return out, nil
}
