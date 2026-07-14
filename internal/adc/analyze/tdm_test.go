package analyze

import (
	"math"
	"path/filepath"
	"sort"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// tdmFixture loads the H2O FCIDUMP and its dipole-carrying MO sidecar.
func tdmFixture(t *testing.T) (*fcidump.Data, int, []float64, *mo.Data) {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.fcidump"))
	if err != nil {
		t.Fatal(err)
	}
	md, err := mo.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.mo.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !md.HasDipole {
		t.Skip("fixture sidecar has no dipole integrals")
	}
	nocc := mp.NOcc(d)
	return d, nocc, mp.OrbitalEnergies(d, nocc), md
}

// TestBuildSIPEmissions checks the analyze layer's bookkeeping around sip.Emissions:
// the state indices it reports are the 1-based positions of the surviving states, the
// energies come back consistent, only the emission direction survives, and the
// oscillator strength / Einstein-A unit conversions hold.
func TestBuildSIPEmissions(t *testing.T) {
	d, nocc, eps, md := tdmFixture(t)
	opts := Options{PSThresh: 1, CoeffThresh: 0.1}

	sp := sip.NewSpace(nocc, d.NORB, d.OrbSym, 0) // A1 sector
	mx := sip.New(sp, integrals.New(d, nocc, d.OrbSym), eps, 3, backend.Gonum{})
	res := lanczos.SolveDense(mx, backend.Gonum{})

	sec := BuildSIPSector(sp, res, mx.FMatrix(), opts)
	ems, err := BuildSIPEmissions(sp, res, mx.FMatrix(), md, opts, SIPTDMOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(ems) == 0 {
		t.Fatal("no emissions")
	}
	for _, e := range ems {
		if e.OmegaEV <= 0 {
			t.Errorf("emission has non-positive ω=%.3f", e.OmegaEV)
		}
		if e.Init < 1 || e.Init > len(sec.States) || e.Mid < 1 || e.Mid > len(sec.States) {
			t.Fatalf("state index out of range: %d -> %d (have %d)", e.Init, e.Mid, len(sec.States))
		}
		// The reported energies must be the sector's own state energies, and ω their gap.
		if got := sec.States[e.Init-1].EnergyEV; math.Abs(got-e.InitEV) > 1e-9 {
			t.Errorf("init energy %.6f != sector state %.6f", e.InitEV, got)
		}
		if got := sec.States[e.Mid-1].EnergyEV; math.Abs(got-e.MidEV) > 1e-9 {
			t.Errorf("mid energy %.6f != sector state %.6f", e.MidEV, got)
		}
		if math.Abs((e.InitEV-e.MidEV)-e.OmegaEV) > 1e-9 {
			t.Errorf("ω=%.6f != ΔE=%.6f", e.OmegaEV, e.InitEV-e.MidEV)
		}
		// f = (2/3)·ω·|μ|² and A[s⁻¹] = (2/c³)·ω²·f · auToPerSec, with ω in hartree.
		w := e.OmegaEV / au2eV
		mu2 := e.Mu[0]*e.Mu[0] + e.Mu[1]*e.Mu[1] + e.Mu[2]*e.Mu[2]
		if wantF := 2.0 / 3.0 * w * mu2; math.Abs(wantF-e.Osc) > 1e-12 {
			t.Errorf("f=%.3e != (2/3)ω|μ|²=%.3e", e.Osc, wantF)
		}
		wantA := 2 * w * w * e.Osc / math.Pow(sip.SpeedOfLight, 3) * auToPerSec
		if e.Osc > 1e-6 && math.Abs(wantA-e.RatePerSec)/wantA > 1e-9 {
			t.Errorf("A=%.3e != %.3e s⁻¹", e.RatePerSec, wantA)
		}
	}
}

// TestBuildSIPPhotoionization checks the Dyson pseudo-spectrum bookkeeping: the pole
// strength matches the sector's spectroscopic factor, every emitted channel is a
// continuum proxy (ε > 0) above the oscillator-strength cutoff, and ω = E + ε.
func TestBuildSIPPhotoionization(t *testing.T) {
	d, nocc, eps, md := tdmFixture(t)
	opts := Options{PSThresh: 1, CoeffThresh: 0.1}

	sp := sip.NewSpace(nocc, d.NORB, d.OrbSym, 0)
	mx := sip.New(sp, integrals.New(d, nocc, d.OrbSym), eps, 3, backend.Gonum{})
	res := lanczos.SolveDense(mx, backend.Gonum{})

	sec := BuildSIPSector(sp, res, mx.FMatrix(), opts)
	const cut = 1e-8
	ph, err := BuildSIPPhotoionization(sp, mx, res, mx.FMatrix(), md, eps, opts, SIPTDMOptions{OscThresh: cut})
	if err != nil {
		t.Fatal(err)
	}
	if len(ph) != len(sec.States) {
		t.Fatalf("%d photoionization records for %d states", len(ph), len(sec.States))
	}
	for i, p := range ph {
		// The Dyson orbital's occupied block is the spectroscopic amplitude a = F·Y,
		// so Σ_occ d² must equal the sector's reported pole strength.
		if want := sec.States[i].PSPercent / 100; math.Abs(p.SpecFactor-want) > 1e-9 {
			t.Errorf("state %d spec_factor=%.6f != ps/100=%.6f", p.State, p.SpecFactor, want)
		}
		for _, c := range p.Channels {
			if c.EpsEV <= 0 {
				t.Errorf("channel ε=%.3f eV is not a continuum proxy", c.EpsEV)
			}
			if c.Osc < cut {
				t.Errorf("channel below cutoff leaked: f=%.3e", c.Osc)
			}
			if math.Abs((p.EnergyEV+c.EpsEV)-c.OmegaEV) > 1e-9 {
				t.Errorf("ω=%.4f != E+ε=%.4f", c.OmegaEV, p.EnergyEV+c.EpsEV)
			}
		}
	}
}

// TestBuildSIPCrossEmissions checks the CVS core → valence X-ray-emission path: across
// different irreps the state overlap vanishes identically (so the moment is
// gauge-independent), and the O 1s core hole reaches the valence sector with a nonzero
// dipole.
func TestBuildSIPCrossEmissions(t *testing.T) {
	d, nocc, eps, md := tdmFixture(t)
	opts := Options{PSThresh: 1, CoeffThresh: 0.1}
	ints := integrals.New(d, nocc, d.OrbSym)

	bra := sip.NewSpace4(nocc, d.NORB, d.OrbSym, 0, []int{0}) // O 1s, A1
	mxB := sip.New(bra, ints, eps, 4, backend.Gonum{})
	resB := lanczos.SolveDense(mxB, backend.Gonum{})

	ket := sip.NewSpace(nocc, d.NORB, d.OrbSym, 3) // a valence irrep, different from A1
	mxK := sip.New(ket, ints, eps, 3, backend.Gonum{})
	resK := lanczos.SolveDense(mxK, backend.Gonum{})

	ems, err := BuildSIPCrossEmissions(bra, resB, mxB.FMatrix(), ket, resK, mxK.FMatrix(), md, opts, SIPTDMOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(ems) == 0 {
		t.Fatal("no cross emissions")
	}
	var maxMu float64
	for _, e := range ems {
		if !e.Cross {
			t.Error("cross emission not flagged Cross")
		}
		if e.Overlap != 0 {
			t.Errorf("overlap %.3e across different irreps, want 0", e.Overlap)
		}
		mu := math.Sqrt(e.Mu[0]*e.Mu[0] + e.Mu[1]*e.Mu[1] + e.Mu[2]*e.Mu[2])
		if mu > maxMu {
			maxMu = mu
		}
	}
	if maxMu < 1e-3 {
		t.Errorf("largest cross dipole %.3e is implausibly small", maxMu)
	}
}

// emLine is one emission reduced to the quantities the identity below compares.
type emLine struct {
	init, mid float64 // the two state energies, eV — the identity's matching keys
	mu        [3]float64
	osc       float64
}

// collectEmissions solves every sector of one symmetry mode and returns the complete
// emission line list — within-sector plus, when the sectors are symmetry-blocked, across —
// together with the whole spectrum, sorted, in eV. orbSym == nil turns symmetry off, which
// puts every configuration into a single space.
func collectEmissions(t *testing.T, d *fcidump.Data, nocc int, eps []float64, md *mo.Data, orbSym []int, nsym int) ([]emLine, []float64) {
	t.Helper()
	opts := Options{PSThresh: 0, CoeffThresh: 0} // keep every root: the identity is over the full spectrum

	type sector struct {
		sp  *sip.Space
		res lanczos.Result
		f   backend.Mat
	}
	ints := integrals.New(d, nocc, orbSym)
	var secs []sector
	var levels []float64
	for g := range nsym {
		sp := sip.NewSpace(nocc, d.NORB, orbSym, g)
		if sp.MainBlockSize() == 0 {
			continue
		}
		mx := sip.New(sp, ints, eps, 3, backend.Gonum{})
		defer mx.Release()
		res := lanczos.SolveDense(mx, backend.Gonum{})
		for _, v := range res.Values {
			levels = append(levels, v*au2eV)
		}
		secs = append(secs, sector{sp, res, mx.FMatrix()})
	}

	var out []emLine
	add := func(ems []SIPTransition) {
		for _, e := range ems {
			out = append(out, emLine{e.InitEV, e.MidEV, e.Mu, e.Osc})
		}
	}
	for _, s := range secs {
		ems, err := BuildSIPEmissions(s.sp, s.res, s.f, md, opts, SIPTDMOptions{})
		if err != nil {
			t.Fatal(err)
		}
		add(ems)
	}
	for _, bra := range secs {
		for _, ket := range secs {
			if bra.sp.Sym == ket.sp.Sym {
				continue
			}
			ems, err := BuildSIPCrossEmissions(bra.sp, bra.res, bra.f, ket.sp, ket.res, ket.f, md, opts, SIPTDMOptions{})
			if err != nil {
				t.Fatal(err)
			}
			add(ems)
		}
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].init != out[b].init {
			return out[a].init < out[b].init
		}
		return out[a].mid < out[b].mid
	})
	sort.Float64s(levels)
	return out, levels
}

// TestSymmetryBlockedEmissionsMatchUnsymmetrized is the identity that symmetry blocking
// must satisfy, and the gate that catches a missing cross-sector path.
//
// Turning symmetry off puts every configuration into one space; the per-irrep spaces of a
// symmetry-on run partition exactly that same configuration set, and the secular matrix is
// block-diagonal in the irrep. So the union of the sector spectra *is* the unsymmetrized
// spectrum, and the two runs must produce the identical list of emission lines — energies,
// dipole components and oscillator strengths alike.
//
// The symmetry-on list only closes if the cross-sector emissions are included: inside one
// sector the square ISR dipole sees only the totally symmetric component, so without them
// every x/y-polarized line is missing and the counts do not even match.
func TestSymmetryBlockedEmissionsMatchUnsymmetrized(t *testing.T) {
	d, nocc, eps, md := tdmFixture(t)

	nsym := 1
	for o := range d.NORB {
		for nsym < d.OrbSym[o] {
			nsym <<= 1
		}
	}

	off, levels := collectEmissions(t, d, nocc, eps, md, nil, 1)
	on, _ := collectEmissions(t, d, nocc, eps, md, d.OrbSym, nsym)

	if len(off) != len(on) {
		t.Fatalf("line count differs: symmetry off %d, symmetry on %d — a whole class of "+
			"transitions is missing from one of them", len(off), len(on))
	}

	// Two states that happen to be degenerate across irreps come back from the unsymmetrized
	// dense solver as an arbitrary mixture of the two, and the dipole of a mixture is not
	// the dipole of either. That is a property of degenerate eigenvectors, not a defect, so
	// skip any line built on such a level. levels is the full spectrum, sorted.
	const degTol = 1e-6 // eV
	isDegenerate := func(e float64) bool {
		for i := 1; i < len(levels); i++ {
			if levels[i]-levels[i-1] < degTol &&
				(math.Abs(levels[i]-e) < degTol || math.Abs(levels[i-1]-e) < degTol) {
				return true
			}
		}
		return false
	}

	var compared, skipped int
	for i := range off {
		a, b := off[i], on[i]
		if math.Abs(a.init-b.init) > 1e-6 || math.Abs(a.mid-b.mid) > 1e-6 {
			t.Fatalf("line %d: energies differ: off (%.6f -> %.6f), on (%.6f -> %.6f)",
				i, a.init, a.mid, b.init, b.mid)
		}
		if isDegenerate(a.init) || isDegenerate(a.mid) {
			skipped++
			continue
		}
		compared++
		for c := range 3 {
			// The overall sign of an eigenvector is arbitrary, so compare |μ| per component.
			if math.Abs(math.Abs(a.mu[c])-math.Abs(b.mu[c])) > 1e-8 {
				t.Errorf("line %d (%.4f -> %.4f eV): |μ_%d| off %.10f, on %.10f",
					i, a.init, a.mid, c, a.mu[c], b.mu[c])
			}
		}
		if math.Abs(a.osc-b.osc) > 1e-10 {
			t.Errorf("line %d (%.4f -> %.4f eV): f off %.12f, on %.12f", i, a.init, a.mid, a.osc, b.osc)
		}
	}
	if compared == 0 {
		t.Fatal("every line was skipped as degenerate; the test compared nothing")
	}
	t.Logf("%d lines compared, %d skipped as degenerate", compared, skipped)
}
