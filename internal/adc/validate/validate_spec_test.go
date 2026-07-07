// Decay-channel spectrum regeneration guard: re-solves the DIP-ADC(2) problem on
// the matched DZP integrals in-process, classifies it into the Auger/ICD/ETMD
// stick spectrum exactly as `adcgo -dip -mo … -spectrum -init-atom O` does, and
// asserts the committed testdata/h2o_dzp.spec.json fixture reproduces it — so the
// spectrum JSON (the contract with ADCanalysis's plotspec) cannot silently drift.
package validate

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"adcgo/internal/adc/analyze"
	"adcgo/internal/adc/backend"
	"adcgo/internal/adc/dip"
	"adcgo/internal/adc/fcidump"
	"adcgo/internal/adc/integrals"
	"adcgo/internal/adc/lanczos"
	"adcgo/internal/adc/mo"
	"adcgo/internal/adc/mp"
	"adcgo/internal/adc/spectrum"
)

const specTolFixture = 1e-8

// numIrreps mirrors cmd/adcgo's helper: the number of symmetry groups implied by
// the ORBSYM labels (the smallest power of two spanning them).
func numIrreps(orbSym []int, norb int) int {
	if orbSym == nil {
		return 1
	}
	max0 := 0
	for o := range norb {
		if lab := orbSym[o] - 1; lab > max0 {
			max0 = lab
		}
	}
	n := 1
	for n < max0+1 {
		n <<= 1
	}
	return n
}

// solveDIPSpectrum reproduces runDIP + buildDIPSpectrum for the DZP fixture:
// singlet then triplet, per irrep, dense solve with atom-resolved populations,
// classified against O with each atom its own site.
func solveDIPSpectrum(t *testing.T) *spectrum.Spectrum {
	t.Helper()
	d, err := fcidump.ReadFile(testdata("h2o_dzp.fcidump"))
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	moData, err := mo.ReadFile(testdata("h2o_dzp.mo.json"))
	if err != nil {
		t.Fatalf("read mo: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}
	ints := integrals.New(d, nocc, d.OrbSym)
	opts := analyze.Options{PSThresh: 1.0, CoeffThresh: 0.1} // cmd/adcgo defaults

	nsym := numIrreps(d.OrbSym, d.NORB)
	var secs []analyze.Sector
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		for targetSym := range nsym {
			sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, targetSym, spin)
			if sp.Size() == 0 {
				continue
			}
			mx := dip.New(sp, ints, eps, be)
			res := lanczos.SolveDense(mx, be)
			pe := analyze.NewPopEngine(sp, moData)
			secs = append(secs, analyze.BuildSector(sp, res, opts, pe))
		}
	}

	sites := make([]spectrum.Site, len(moData.AtomNames))
	for i, name := range moData.AtomNames {
		sites[i] = spectrum.Site{Name: name, Members: []string{name}}
	}
	spec, _, err := spectrum.BuildDIP(secs, sites, spectrum.DIPOptions{
		InitialAtom:         "O",
		Classify:            spectrum.DefaultOptions(),
		SingletTripletRatio: 3.0,
	})
	if err != nil {
		t.Fatalf("build spectrum: %v", err)
	}
	return spec
}

func TestSpectrumFixtureMatchesSolver(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in-process DIP spectrum re-solve in -short mode")
	}
	b, err := os.ReadFile(testdata("h2o_dzp.spec.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var want spectrum.Spectrum
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	got := solveDIPSpectrum(t)

	if len(got.Channels) != len(want.Channels) {
		t.Fatalf("channel count: solver %d, fixture %d", len(got.Channels), len(want.Channels))
	}
	for i := range got.Channels {
		if got.Channels[i] != want.Channels[i] {
			t.Errorf("channel %d: solver %q vs fixture %q", i, got.Channels[i], want.Channels[i])
		}
	}
	if got.Meta.Kind != want.Meta.Kind || got.Meta.InitialIonization.Atom != want.Meta.InitialIonization.Atom {
		t.Errorf("meta: solver %+v vs fixture %+v", got.Meta, want.Meta)
	}

	if len(got.Lines) != len(want.Lines) {
		t.Fatalf("line count: solver %d, fixture %d", len(got.Lines), len(want.Lines))
	}
	for i := range got.Lines {
		g, w := got.Lines[i], want.Lines[i]
		if g.Channel != w.Channel || g.StateRef != w.StateRef || g.Spin != w.Spin || g.Irrep != w.Irrep {
			t.Errorf("line %d labels: solver %+v vs fixture %+v", i, g, w)
			continue
		}
		if de := math.Abs(g.Energy - w.Energy); de > specTolFixture {
			t.Errorf("line %d (%s) energy: solver %.10f vs fixture %.10f (Δ=%.1e)", i, g.Channel, g.Energy, w.Energy, de)
		}
		if di := math.Abs(g.Intensity - w.Intensity); di > specTolFixture {
			t.Errorf("line %d (%s) intensity: solver %.10f vs fixture %.10f (Δ=%.1e)", i, g.Channel, g.Intensity, w.Intensity, di)
		}
		if dp := math.Abs(g.PSPercent - w.PSPercent); dp > specTolFixture {
			t.Errorf("line %d (%s) ps: solver %.10f vs fixture %.10f", i, g.Channel, g.PSPercent, w.PSPercent)
		}
	}
}
