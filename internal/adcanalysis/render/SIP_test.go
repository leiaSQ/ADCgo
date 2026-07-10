package render

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adcanalysis/model"
)

// sipFile builds a small two-state single-ionization file across two symmetries.
func sipFile() *model.SIPOutFile {
	return &model.SIPOutFile{
		MOTable: []model.MO{
			{Index: 1, Sym: 1, EnergyAU: -1.359},
			{Index: 3, Sym: 1, EnergyAU: -0.580},
			{Index: 4, Sym: 3, EnergyAU: -0.509},
		},
		States: []model.SIPState{
			{Irrep: 1, Spin: 2, Index: 1, EnergyEV: 14.81, PSPercent: 94.13,
				Main: []model.OrbWeight{{Orbital: 3, Coeff: -0.9698}}},
			{Irrep: 3, Spin: 2, Index: 1, EnergyEV: 18.5, PSPercent: 90.0,
				Main: []model.OrbWeight{{Orbital: 4, Coeff: 0.94}, {Orbital: 1, Coeff: 0.1}}},
		},
	}
}

func TestBuildSIPPerOrbitalLines(t *testing.T) {
	spec, err := BuildSIP(sipFile(), SIPOptions{Molecule: "H2O", SourceFiles: []string{"ADC.out"}})
	if err != nil {
		t.Fatal(err)
	}

	if spec.Meta.Kind != "sip" {
		t.Errorf("meta.kind = %q, want sip", spec.Meta.Kind)
	}

	// One line per (state, orbital): 1 + 2 = 3.
	if len(spec.Lines) != 3 {
		t.Fatalf("got %d lines, want 3: %+v", len(spec.Lines), spec.Lines)
	}

	// Intensity is the squared one-hole amplitude.
	if got, want := spec.Lines[0].Intensity, 0.9698*0.9698; math.Abs(got-want) > 1e-12 {
		t.Errorf("line[0] intensity = %v, want %v", got, want)
	}
	if spec.Lines[0].Spin != 2 || spec.Lines[0].Channel != "MO 3 (sym 1)" {
		t.Errorf("line[0] = %+v, want spin 2 / MO 3 (sym 1)", spec.Lines[0])
	}
	if spec.Lines[0].StateRef != "irrep1/s2/#1" {
		t.Errorf("line[0] state_ref = %q", spec.Lines[0].StateRef)
	}

	// Channels are ordered by MO index: 1, 3, 4.
	wantCh := []string{"MO 1 (sym 1)", "MO 3 (sym 1)", "MO 4 (sym 3)"}
	if len(spec.Channels) != len(wantCh) {
		t.Fatalf("channels = %v, want %v", spec.Channels, wantCh)
	}
	for i, c := range wantCh {
		if spec.Channels[i] != c {
			t.Errorf("channel[%d] = %q, want %q", i, spec.Channels[i], c)
		}
	}

	// Per-state summed intensity recovers Σ coeff² (~ ps/100 for a real run).
	sum := 0.0
	for _, l := range spec.Lines {
		if l.StateRef == "irrep3/s2/#1" {
			sum += l.Intensity
		}
	}
	if want := 0.94*0.94 + 0.1*0.1; math.Abs(sum-want) > 1e-12 {
		t.Errorf("state 2 summed intensity = %v, want %v", sum, want)
	}
}
