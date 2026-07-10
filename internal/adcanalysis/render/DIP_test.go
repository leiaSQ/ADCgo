package render

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adcanalysis/classify"
	"github.com/leiaSQ/ADCgo/internal/adcanalysis/model"
)

func groups() []model.Site {
	return []model.Site{{Name: "O"}, {Name: "H1"}, {Name: "H2"}}
}

// oneFile builds a minimal OutFile with a single state carrying the 39.66 eV
// popana row from adcdip1.out.
func oneFile() *model.OutFile {
	pop := &model.PopRow{
		EnergyEV: 39.6604,
		OneSite:  map[string]float64{"O": 0.8065, "H1": 0.0004, "H2": 0.0004},
		TwoSite:  map[string]float64{"O/H1": 0.0131, "O/H2": 0.0131, "H1/H2": 0.0002},
	}
	return &model.OutFile{
		Symmetry: 1,
		Groups:   model.PopGroups{OneSite: []string{"O", "H1", "H2"}},
		States: []model.State{{
			Irrep: 1, Spin: 1, Index: 1,
			EnergyEV: 39.660357, PSPercent: 83.39, Pop: pop,
		}},
	}
}

func TestBuildFlattensStateIntoLines(t *testing.T) {
	spec, skipped, err := BuildDIP([]*model.OutFile{oneFile()}, groups(), DIPOptions{
		InitialAtom:         "O",
		Classify:            classify.DefaultOptions(),
		SingletTripletRatio: 3.0,
		SourceFiles:         []string{"adcdip1.out"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if skipped != 0 {
		t.Errorf("skipped = %d, want 0", skipped)
	}

	// One line per surviving channel: Auger@O, two ICDs, ETMD(2) (0.0008) and
	// ETMD(3) (0.0002), all above the default zero floor.
	if len(spec.Lines) != 5 {
		t.Fatalf("got %d lines, want 5: %+v", len(spec.Lines), spec.Lines)
	}
	if got := spec.Channels; len(got) != 5 || got[0] != "Auger@O" || got[4] != "ETMD(3)" {
		t.Errorf("canonical channels = %v", got)
	}
	// Every line inherits the state's energy/spin/irrep/ref.
	for _, l := range spec.Lines {
		if l.Energy != 39.660357 || l.Spin != 1 || l.Irrep != 1 || l.StateRef != "irrep1/s1/#1" {
			t.Errorf("line carries wrong state fields: %+v", l)
		}
	}
	if spec.Meta.EnergyUnit != "eV" || spec.Meta.SingletTripletRatio != 3.0 {
		t.Errorf("meta = %+v", spec.Meta)
	}
	if len(spec.Meta.Irreps) != 1 || spec.Meta.Irreps[0] != "1" {
		t.Errorf("irreps = %v, want [1]", spec.Meta.Irreps)
	}
}

// Intensities must sum to the row total: spectrum building only flattens, it
// does not pre-weight by the singlet:triplet ratio.
func TestBuildConservesIntensity(t *testing.T) {
	spec, _, err := BuildDIP([]*model.OutFile{oneFile()}, groups(), DIPOptions{
		InitialAtom: "O",
		Classify:    classify.Options{IncludeZero: true},
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

func TestBuildSkipsStatesWithoutPop(t *testing.T) {
	f := oneFile()
	f.States = append(f.States, model.State{Irrep: 1, Spin: 3, Index: 2, EnergyEV: 50, Pop: nil})
	spec, skipped, err := BuildDIP([]*model.OutFile{f}, groups(), DIPOptions{
		InitialAtom: "O", Classify: classify.DefaultOptions(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if skipped != 1 {
		t.Errorf("skipped = %d, want 1", skipped)
	}
	for _, l := range spec.Lines {
		if l.StateRef == "irrep1/s3/#2" {
			t.Errorf("pop-less state leaked a line: %+v", l)
		}
	}
}

func TestBuildRejectsUnknownAtom(t *testing.T) {
	if _, _, err := BuildDIP([]*model.OutFile{oneFile()}, groups(), DIPOptions{InitialAtom: "N"}); err == nil {
		t.Error("expected error for unknown initial atom")
	}
}
