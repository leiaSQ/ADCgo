package sip

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// crossSpaces builds a symmetry-on CVS ADC(4) core-hole space (bra) and a plain valence
// space of irrep ketSym (ket), on the H2O fixture, sharing one integral store.
func crossSpaces(t *testing.T, ketSym int) (*Space, *Space, *Matrix, *Matrix, *fcidump.Data) {
	t.Helper()
	fd := h2oData(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	ints := integrals.New(fd, nocc, fd.OrbSym)
	be := backend.Gonum{}
	bra := NewSpace4(nocc, fd.NORB, fd.OrbSym, 0, []int{0}) // A1 core hole (O 1s)
	ket := NewSpace(nocc, fd.NORB, fd.OrbSym, ketSym)
	return bra, ket, New(bra, ints, eps, 4, be), New(ket, ints, eps, 3, be), fd
}

// TestADC4SpinFunctionsReproduceKopp1 is the convention gate for the cross-space D, the
// exact counterpart of TestSpinFunctionsReproduceSecularBlocks for the plain space.
//
// The ADC(4) eigenvectors are expressed in whatever spin functions elements4.go assumed.
// kopp1 is a pure first-order element — bare integrals, no denominators — so
// ⟨Φ_K|Ĥ|Φ_{KLa,S}⟩ evaluated over determinant expansions must reproduce it exactly, and
// only one choice of the type-II phase does. (c22elem4 cannot serve here: with a single
// core orbital its 4th-order sum1_4 piece is always active, so it is not a Slater–Condon
// quantity.)
func TestADC4SpinFunctionsReproduceKopp1(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping determinant-level ADC(4) convention check in -short mode")
	}
	fd := h2oData(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	ints := integrals.New(fd, nocc, fd.OrbSym)

	ref := sdet{a: uint64(1)<<nocc - 1, b: uint64(1)<<nocc - 1}
	eHF := hamDet(ref, ref, fd)
	me := func(x, y sdet) float64 { return hamDet(x, y, fd) }

	checked := 0
	for _, sym := range []int{0, 1, 2, 3} {
		sp := NewSpace4(nocc, fd.NORB, fd.OrbSym, sym, []int{0})
		if sp.MainBlockSize() == 0 {
			continue // no core hole of this irrep
		}
		mx := New(sp, ints, eps, 4, backend.Gonum{})
		n := len(sp.Configs)
		dets := make([][]dterm, n)
		for i := range n {
			dets[i] = configDets(sp, i)
		}
		for i := range sp.BeginSat {
			k := sp.Configs[i].Occ[0]
			if got := configME(dets[i], dets[i], me) - eHF; math.Abs(got+eps[k]) > 1e-9 {
				t.Errorf("irrep %d: 1h diagonal for core %d is %g, want −ε = %g",
					sym, k, got, -eps[k])
			}
			for j := sp.BeginSat; j < n; j++ {
				want := mx.el.kopp1(k, sp.Configs[j])
				got := configME(dets[i], dets[j], me)
				if math.Abs(got-want) > 1e-9 {
					t.Fatalf("irrep %d: <%d|H|%v> = %g, kopp1 gives %g (ratio %g); "+
						"the ADC(4) spin functions are not the ones elements4.go uses",
						sym, k, sp.Configs[j], got, want, got/want)
				}
				checked++
			}
		}
		mx.Release()
	}
	if checked < 40 {
		t.Fatalf("only %d coupling elements checked: the test proved nothing", checked)
	}
}

// TestCrossDipoleMatchesDeterminants: every element of the rectangular D, against the two
// spaces' determinant expansions. Covers the cross-sector plain×plain case (which the
// square D of Chunk 3 cannot express at all — inside one irrep only the totally symmetric
// dipole component survives) and the CVS ADC(4) × valence case, where the bra and the ket
// disagree both on hole role order and on the type-II phase.
func TestCrossDipoleMatchesDeterminants(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping determinant-level cross-space dipole check in -short mode")
	}
	fd := h2oData(t)
	nocc := mp.NOcc(fd)
	d := randomSymmetric(fd.NORB)

	type pair struct {
		name     string
		bra, ket *Space
	}
	var cases []pair
	for _, g := range []int{1, 2, 3} {
		cases = append(cases, pair{"plain A1 × plain", NewSpace(nocc, fd.NORB, fd.OrbSym, 0),
			NewSpace(nocc, fd.NORB, fd.OrbSym, g)})
	}
	for _, g := range []int{0, 2, 3} {
		cases = append(cases, pair{"CVS ADC(4) × plain", NewSpace4(nocc, fd.NORB, fd.OrbSym, 0, []int{0}),
			NewSpace(nocc, fd.NORB, fd.OrbSym, g)})
	}

	for _, c := range cases {
		if c.bra.MainBlockSize() == 0 || c.ket.MainBlockSize() == 0 {
			continue
		}
		o, err := NewISRDipoleCross(c.bra, c.ket, d)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		nb, nk := len(c.bra.Configs), len(c.ket.Configs)
		braDets := make([][]dterm, nb)
		for i := range nb {
			braDets[i] = configDets(c.bra, i)
		}
		ketDets := make([][]dterm, nk)
		for j := range nk {
			ketDets[j] = configDets(c.ket, j)
		}
		me := func(x, y sdet) float64 { return oneBodyDet(x, y, d) }

		var maxErr, maxVal float64
		for i := range nb {
			for j := range nk {
				want := configME(braDets[i], ketDets[j], me)
				if v := math.Abs(want); v > maxVal {
					maxVal = v
				}
				if e := math.Abs(o.At(i, j) - want); e > maxErr {
					maxErr = e
					if e > 1e-10 {
						t.Fatalf("%s: D[%d,%d] = %g, determinants give %g (configs %v, %v)",
							c.name, i, j, o.At(i, j), want, c.bra.Configs[i], c.ket.Configs[j])
					}
				}
			}
		}
		if maxVal < 1e-2 {
			t.Errorf("%s: largest determinant element is %g — the test proved nothing", c.name, maxVal)
		}
		// The bra's 3h2p rows are the documented truncation, not an accident.
		for i := nb; i < o.Size(); i++ {
			for j := range o.Cols() {
				if o.At(i, j) != 0 {
					t.Fatalf("%s: 3h2p row %d is not zero", c.name, i)
				}
			}
		}
		if maxErr > 1e-10 {
			t.Errorf("%s: max |D − determinant D| = %g", c.name, maxErr)
		}
	}
}

// TestCrossApplyMatchesDense: the sparse row structure reaches every nonzero of the
// rectangular D too, not just the square one.
func TestCrossApplyMatchesDense(t *testing.T) {
	bra, ket, _, _, fd := crossSpaces(t, 2)
	o, err := NewISRDipoleCross(bra, ket, randomSymmetric(fd.NORB))
	if err != nil {
		t.Fatal(err)
	}
	m := o.BuildMatrix()
	in := make([]float64, o.Cols())
	out := make([]float64, o.Size())
	var maxErr float64
	for j := range o.Cols() {
		in[j] = 1
		o.Apply(out, in)
		in[j] = 0
		for i := range o.Size() {
			if e := math.Abs(out[i] - m.At(i, j)); e > maxErr {
				maxErr = e
			}
		}
	}
	if maxErr > 1e-14 {
		t.Errorf("Apply vs BuildMatrix max diff %g: sparsify missed a nonzero column", maxErr)
	}
}

// TestCrossOverlapMatchesDeterminants: the configuration overlap is ±1 on shared
// configurations, and the signs are the ones the determinant expansions imply. Against a
// same-irrep valence space the CVS core-hole space shares configurations, so S ≠ 0 and the
// cross-space transition dipole is origin-dependent; against a different irrep it shares
// none, and S vanishes identically.
func TestCrossOverlapMatchesDeterminants(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping determinant-level overlap check in -short mode")
	}
	fd := h2oData(t)
	nocc := mp.NOcc(fd)
	bra := NewSpace4(nocc, fd.NORB, fd.OrbSym, 0, []int{0})

	same := NewSpace(nocc, fd.NORB, fd.OrbSym, 0)
	ov, err := ConfigOverlap(bra, same)
	if err != nil {
		t.Fatal(err)
	}
	if ov.Nonzero() == 0 {
		t.Fatal("the CVS space and the valence A1 space share no configuration")
	}
	nb, nk := len(bra.Configs), len(same.Configs)
	braDets := make([][]dterm, nb)
	for i := range nb {
		braDets[i] = configDets(bra, i)
	}
	ketDets := make([][]dterm, nk)
	for j := range nk {
		ketDets[j] = configDets(same, j)
	}
	delta := func(x, y sdet) float64 {
		if x == y {
			return 1
		}
		return 0
	}
	dense := make([][]float64, bra.Size())
	for i := range dense {
		dense[i] = make([]float64, same.Size())
	}
	for i, row := range ov.rows {
		for _, e := range row {
			dense[i][e.col] = e.val
		}
	}
	for i := range nb {
		for j := range nk {
			want := configME(braDets[i], ketDets[j], delta)
			if math.Abs(dense[i][j]-want) > 1e-12 {
				t.Fatalf("S[%d,%d] = %g, determinants give %g (configs %v, %v)",
					i, j, dense[i][j], want, bra.Configs[i], same.Configs[j])
			}
		}
	}
	for i := nb; i < bra.Size(); i++ {
		for j := range same.Size() {
			if dense[i][j] != 0 {
				t.Fatalf("a 3h2p configuration overlaps a 2h1p one at (%d,%d)", i, j)
			}
		}
	}

	for _, g := range []int{1, 2, 3} {
		other := NewSpace(nocc, fd.NORB, fd.OrbSym, g)
		if other.MainBlockSize() == 0 {
			continue
		}
		ov, err := ConfigOverlap(bra, other)
		if err != nil {
			t.Fatal(err)
		}
		if n := ov.Nonzero(); n != 0 {
			t.Errorf("irrep %d: %d configurations shared with the A1 core-hole space, want 0", g, n)
		}
	}
}

// TestCrossOriginShift is the non-orthogonality guard. Moving the gauge origin by δ
// replaces d_pq with d_pq − δ·δ_pq and μ_nuc with μ_nuc − δ·Z_tot, so a transition moment
// must move by exactly −δ·⟨Ψ_i|Ψ_m⟩ — the cation's charge is +1. When the two states share
// no configuration the moment must not move at all.
//
// The whole point is that μ is only a physical number when the overlap vanishes; a bug
// that dropped the μ_nuc·S term would sail past every other test in this file.
func TestCrossOriginShift(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping origin-shift check in -short mode")
	}
	fd := h2oData(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	ints := integrals.New(fd, nocc, fd.OrbSym)
	be := backend.Gonum{}
	const delta = 0.37 // an arbitrary shift of the gauge origin along x
	zTot := float64(fd.NELEC)

	bra := NewSpace4(nocc, fd.NORB, fd.OrbSym, 0, []int{0})
	mxBra := New(bra, ints, eps, 4, be)
	defer mxBra.Release()
	resBra := lanczos.SolveDense(mxBra, be)

	d := randomSymmetric(fd.NORB) // not symmetry-adapted: every block is exercised
	shifted := backend.NewMat(fd.NORB, fd.NORB)
	for p := range fd.NORB {
		for q := range fd.NORB {
			v := d.At(p, q)
			if p == q {
				v -= delta
			}
			shifted.Set(p, q, v)
		}
	}
	var zero backend.Mat = backend.NewMat(fd.NORB, fd.NORB)

	for _, g := range []int{0, 2} { // 0: shares configurations (S≠0); 2: disjoint (S=0)
		ket := NewSpace(nocc, fd.NORB, fd.OrbSym, g)
		if ket.MainBlockSize() == 0 {
			continue
		}
		mxKet := New(ket, ints, eps, 3, be)
		resKet := lanczos.SolveDense(mxKet, be)
		ov, err := ConfigOverlap(bra, ket)
		if err != nil {
			t.Fatal(err)
		}

		emit := func(dip backend.Mat, muNucX float64) CrossEmission {
			ops, err := NewISRDipolesCross(bra, ket, [3]backend.Mat{dip, zero, zero})
			if err != nil {
				t.Fatal(err)
			}
			ems, err := CrossEmissions(ops, ov, [3]float64{muNucX, 0, 0},
				resBra.Values, resKet.Values, resBra.FullVecs, resKet.FullVecs, []int{0}, []int{0})
			if err != nil {
				t.Fatal(err)
			}
			return ems[0]
		}
		const muNuc = 1.25
		before := emit(d, muNuc)
		after := emit(shifted, muNuc-delta*zTot)

		want := before.Mu[0] - delta*before.Overlap
		if math.Abs(after.Mu[0]-want) > 1e-9 {
			t.Errorf("irrep %d: μ_x moved to %g under a %g origin shift, want %g "+
				"(= μ − δ·S with S = %g)", g, after.Mu[0], delta, want, before.Overlap)
		}
		if g == 0 && math.Abs(before.Overlap) < 1e-6 {
			t.Errorf("the CVS and valence A1 states are orthogonal (S = %g): "+
				"the origin-dependence branch was never exercised", before.Overlap)
		}
		if g != 0 {
			if math.Abs(before.Overlap) > 1e-14 {
				t.Errorf("irrep %d: overlap %g, want 0 by symmetry", g, before.Overlap)
			}
			if math.Abs(after.Mu[0]-before.Mu[0]) > 1e-9 {
				t.Errorf("irrep %d: μ_x is origin-dependent (%g → %g) with zero overlap",
					g, before.Mu[0], after.Mu[0])
			}
		}
	}
}
