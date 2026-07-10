package spectrum

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
)

func groups() []Site {
	return []Site{{Name: "O"}, {Name: "H1"}, {Name: "H2"}}
}

// oneSector builds a minimal DIP sector with a single state carrying the
// 39.66 eV population row from the h2o DIP reference.
func oneSector() analyze.Sector {
	pop := &analyze.Pop{
		OneSite: map[string]float64{"O": 0.8065, "H1": 0.0004, "H2": 0.0004},
		TwoSite: map[string]float64{"O/H1": 0.0131, "O/H2": 0.0131, "H1/H2": 0.0002},
	}
	return analyze.Sector{
		Irrep: 1, Spin: 1,
		States: []analyze.State{{
			Index: 1, EnergyEV: 39.660357, PSPercent: 83.39, Pop: pop,
		}},
	}
}

func TestBuildDIPFlattensStateIntoLines(t *testing.T) {
	spec, skipped, err := BuildDIP([]analyze.Sector{oneSector()}, groups(), DIPOptions{
		InitialAtom:         "O",
		Classify:            DefaultOptions(),
		SingletTripletRatio: 3.0,
		SourceFiles:         []string{"h2o.fcidump"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if skipped != 0 {
		t.Errorf("skipped = %d, want 0", skipped)
	}

	// One line per surviving channel: Auger@O, two ICDs, ETMD(2), ETMD(3).
	if len(spec.Lines) != 5 {
		t.Fatalf("got %d lines, want 5: %+v", len(spec.Lines), spec.Lines)
	}
	if got := spec.Channels; len(got) != 5 || got[0] != "Auger@O" || got[4] != "ETMD(3)" {
		t.Errorf("canonical channels = %v", got)
	}
	for _, l := range spec.Lines {
		if l.Energy != 39.660357 || l.Spin != 1 || l.Irrep != 1 || l.StateRef != "irrep1/s1/#1" {
			t.Errorf("line carries wrong state fields: %+v", l)
		}
	}
	if spec.Meta.Kind != "dip" || spec.Meta.EnergyUnit != "eV" || spec.Meta.SingletTripletRatio != 3.0 {
		t.Errorf("meta = %+v", spec.Meta)
	}
	if len(spec.Meta.Irreps) != 1 || spec.Meta.Irreps[0] != "1" {
		t.Errorf("irreps = %v, want [1]", spec.Meta.Irreps)
	}
}

// Intensities must sum to the row total: spectrum building only flattens, it
// does not pre-weight by the singlet:triplet ratio.
func TestBuildDIPConservesIntensity(t *testing.T) {
	spec, _, err := BuildDIP([]analyze.Sector{oneSector()}, groups(), DIPOptions{
		InitialAtom: "O",
		Classify:    Options{IncludeZero: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, l := range spec.Lines {
		sum += l.Intensity
	}
	if d := math.Abs(sum - 0.8337); d > 1e-9 {
		t.Errorf("intensity sum = %.6f, want 0.8337 (|Δ|=%.2e)", sum, d)
	}
}

func TestBuildDIPSkipsStatesWithoutPop(t *testing.T) {
	sec := oneSector()
	sec.States = append(sec.States, analyze.State{Index: 2, EnergyEV: 50, Pop: nil})
	spec, skipped, err := BuildDIP([]analyze.Sector{sec}, groups(), DIPOptions{
		InitialAtom: "O", Classify: DefaultOptions(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if skipped != 1 {
		t.Errorf("skipped = %d, want 1", skipped)
	}
	for _, l := range spec.Lines {
		if l.StateRef == "irrep1/s1/#2" {
			t.Errorf("pop-less state leaked a line: %+v", l)
		}
	}
}

func TestBuildDIPRejectsUnknownAtom(t *testing.T) {
	if _, _, err := BuildDIP([]analyze.Sector{oneSector()}, groups(), DIPOptions{InitialAtom: "N"}); err == nil {
		t.Error("expected error for unknown initial atom")
	}
}

// BuildSIP flattens per orbital; intensities are Coeff² and recover ps/100.
func TestBuildSIPPerOrbital(t *testing.T) {
	secs := []analyze.SIPSector{{
		Irrep: 1, Spin: 2,
		States: []analyze.SIPState{{
			Index: 1, EnergyEV: 12.5, PSPercent: 91.0,
			Main: []analyze.OrbWeight{{Orbital: 3, Coeff: 0.9}, {Orbital: 5, Coeff: -0.3}},
		}},
	}}
	orbSym := []int{0, 0, 1, 0, 3} // MO 3 -> sym 1, MO 5 -> sym 3
	spec, err := BuildSIP(secs, orbSym, SIPOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if spec.Meta.Kind != "sip" {
		t.Errorf("kind = %q, want sip", spec.Meta.Kind)
	}
	if len(spec.Lines) != 2 {
		t.Fatalf("got %d lines, want 2: %+v", len(spec.Lines), spec.Lines)
	}
	byChan := map[string]float64{}
	for _, l := range spec.Lines {
		byChan[l.Channel] = l.Intensity
		if l.Energy != 12.5 || l.Spin != 2 || l.StateRef != "irrep1/s2/#1" {
			t.Errorf("line fields wrong: %+v", l)
		}
	}
	if d := math.Abs(byChan["MO 3 (sym 1)"] - 0.81); d > 1e-9 {
		t.Errorf("MO 3 intensity = %.6f, want 0.81", byChan["MO 3 (sym 1)"])
	}
	if d := math.Abs(byChan["MO 5 (sym 3)"] - 0.09); d > 1e-9 {
		t.Errorf("MO 5 intensity = %.6f, want 0.09", byChan["MO 5 (sym 3)"])
	}
	if got := spec.Channels; len(got) != 2 || got[0] != "MO 3 (sym 1)" || got[1] != "MO 5 (sym 3)" {
		t.Errorf("channels = %v", got)
	}
}

func TestBuildSIPRejectsEmpty(t *testing.T) {
	if _, err := BuildSIP(nil, nil, SIPOptions{}); err == nil {
		t.Error("expected error for no states")
	}
}
