package analyze

import (
	"math"
	"path/filepath"
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
	ems, err := BuildSIPEmissions(sp, res, mx.FMatrix(), md, opts)
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

	ems, err := BuildSIPCrossEmissions(bra, resB, mxB.FMatrix(), ket, resK, mxK.FMatrix(), md, opts)
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
