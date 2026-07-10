package sip

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

func h2oSidecar(t *testing.T) *mo.Data {
	t.Helper()
	md, err := mo.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.mo.json"))
	if err != nil {
		t.Fatalf("read sidecar: %v", err)
	}
	if !md.HasDipole {
		t.Fatal("h2o.mo.json has no dipole keys")
	}
	return md
}

// componentIrrep infers the point-group irrep each Cartesian dipole component carries,
// from the orbital-irrep product of its own nonzero MO elements. That every nonzero
// element of one component shares a single product is itself the assertion: a dipole
// component transforms as one irrep, and if D^MO says otherwise the sidecar's orbital
// symmetry labels and its integrals disagree.
func componentIrrep(t *testing.T, d backend.Mat, irrep func(int) int, norb int) int {
	t.Helper()
	g, found := 0, false
	for p := range norb {
		for q := range norb {
			if math.Abs(d.At(p, q)) < 1e-10 {
				continue
			}
			pq := irrep(p) ^ irrep(q)
			if !found {
				g, found = pq, true
			} else if pq != g {
				t.Fatalf("dipole component spans irreps %d and %d", g, pq)
			}
		}
	}
	if !found {
		t.Fatal("dipole component is identically zero")
	}
	return g
}

// TestPhotoionizationSymmetry: the Dyson orbital of a sector carries that sector's
// irrep, so ⟨φ_a|r|d⟩ can only survive when irrep(a) = irrep(component) ⊗ Sym. Within
// the sector's own irrep that leaves the totally symmetric component alone — for H2O in
// C2v, μ_z — while μ_x and μ_y reach the continuum only through virtuals of a different
// irrep. Nothing in tdmoment.go knows about irreps; this falls out of D^MO and d.
func TestPhotoionizationSymmetry(t *testing.T) {
	fd := h2oData(t)
	md := h2oSidecar(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	be := backend.Gonum{}

	sawNonzero := false
	for sym := range 4 {
		sp := NewSpace(nocc, fd.NORB, fd.OrbSym, sym)
		if sp.MainBlockSize() == 0 {
			continue
		}
		mx := New(sp, integrals.New(fd, nocc, fd.OrbSym), eps, 3, be)
		res := lanczos.SolveDense(mx, be)
		dy, err := mx.DysonOrbitals(res.FullVecs, []int{0})
		if err != nil {
			t.Fatal(err)
		}
		d := matColumn(dy, 0)

		var comp [3]int
		for c := range 3 {
			comp[c] = componentIrrep(t, md.DipMO[c], sp.irrep, fd.NORB)
		}
		for _, cm := range PhotoionizationMoments(md.DipMO, sp, eps, d, res.Values[0], 0) {
			a := sp.Nocc + cm.Vir
			for c := range 3 {
				allowed := sp.irrep(a) == comp[c]^sym
				if !allowed && math.Abs(cm.Mu[c]) > 1e-12 {
					t.Errorf("irrep %d, virtual %d, component %d: symmetry-forbidden μ = %g",
						sym, a, c, cm.Mu[c])
				}
				if allowed && math.Abs(cm.Mu[c]) > 1e-6 {
					sawNonzero = true
				}
			}
		}
	}
	if !sawNonzero {
		t.Error("every allowed μ vanished: the test proved nothing")
	}
}

// TestPhotoionizationMomentsContraction: μ is the dipole matrix element between the
// virtual and the Dyson orbital, and f is (2/3)ω|μ|² at ω = E_n + ε_a. Feeding a
// Kronecker-delta "Dyson orbital" must reproduce a column of D^MO exactly.
func TestPhotoionizationMomentsContraction(t *testing.T) {
	fd := h2oData(t)
	md := h2oSidecar(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	sp := NewSpace(nocc, fd.NORB, nil, 0)

	for _, p := range []int{0, 3, nocc, nocc + 5} {
		d := make([]float64, sp.Norb)
		d[p] = 1
		for _, cm := range PhotoionizationMoments(md.DipMO, sp, eps, d, 0.5, 7) {
			a := sp.Nocc + cm.Vir
			if cm.State != 7 {
				t.Fatalf("state label %d, want 7", cm.State)
			}
			if cm.Eps != eps[a] || cm.Omega != 0.5+eps[a] {
				t.Errorf("virtual %d: eps %g omega %g, want %g / %g",
					a, cm.Eps, cm.Omega, eps[a], 0.5+eps[a])
			}
			for c := range 3 {
				if e := math.Abs(cm.Mu[c] - md.DipMO[c].At(a, p)); e > 1e-14 {
					t.Errorf("virtual %d component %d: μ = %g, D^MO = %g",
						a, c, cm.Mu[c], md.DipMO[c].At(a, p))
				}
			}
			want := 2.0 / 3.0 * cm.Omega * (cm.Mu[0]*cm.Mu[0] + cm.Mu[1]*cm.Mu[1] + cm.Mu[2]*cm.Mu[2])
			if math.Abs(cm.Osc-want) > 1e-15*math.Abs(want)+1e-20 {
				t.Errorf("oscillator strength %g, want %g", cm.Osc, want)
			}
		}
	}
}

// TestDysonADC4: the CVS ADC(4) space takes the same path with kopp1 in place of
// c12_1 (its 2h1p spin functions are the reference's Dyson-convention ones), its 3h2p
// rows are ignored, and the occupied block is still F·Y. Nothing here touches the
// secular matrix.
func TestDysonADC4(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping ADC(4) Dyson smoke in -short mode")
	}
	fd := h2oData(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	be := backend.Gonum{}
	sp := NewSpace4(nocc, fd.NORB, fd.OrbSym, 0, []int{0}) // A1 core hole (O 1s)
	if len(sp.Sat3) == 0 {
		t.Fatal("the ADC(4) space has no 3h2p configurations")
	}
	mx := New(sp, integrals.New(fd, nocc, fd.OrbSym), eps, 4, be)
	defer mx.Release()
	res := lanczos.SolveDense(mx, be)

	dy, err := mx.DysonOrbitals(res.FullVecs, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	d := matColumn(dy, 0)

	f := mx.FMatrix()
	y := make([]float64, sp.BeginSat)
	for c := range sp.BeginSat {
		y[c] = res.FullVecs.At(c, 0)
	}
	a := f.MulVec(y)
	for c := range sp.BeginSat {
		orb := sp.Configs[c].Occ[0]
		if e := math.Abs(d[orb] - a[c]); e > 1e-14 {
			t.Errorf("core orbital %d: Dyson %g, F·Y %g", orb, d[orb], a[c])
		}
	}

	// The 3h2p rows are skipped, not silently folded in: zeroing them must not move d.
	vecs := backend.NewMat(sp.Size(), 1)
	for r := range len(sp.Configs) {
		vecs.Set(r, 0, res.FullVecs.At(r, 0))
	}
	dy2, err := mx.DysonOrbitals(vecs, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	for p := range sp.Norb {
		if dy2.At(p, 0) != d[p] {
			t.Fatalf("orbital %d: the 3h2p rows changed the Dyson orbital (%g vs %g)",
				p, dy2.At(p, 0), d[p])
		}
	}

	var virt float64
	for p := sp.Nocc; p < sp.Norb; p++ {
		virt += d[p] * d[p]
	}
	if virt < 1e-8 {
		t.Errorf("the core-hole Dyson orbital has no virtual weight (%g)", virt)
	}
}
