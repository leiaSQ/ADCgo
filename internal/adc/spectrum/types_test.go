package spectrum

import (
	"encoding/json"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
)

// TestSchemaKeys pins the JSON key contract with ADCanalysis's plotspec (which
// reads meta.kind, meta.singlet_triplet_ratio, meta.initial_ionization.atom,
// channels, and lines[].{energy,intensity,channel,spin,irrep,state_ref,
// ps_percent}). It marshals a built spectrum and checks those keys survive, so a
// stray json-tag rename fails here rather than silently breaking the renderer.
func TestSchemaKeys(t *testing.T) {
	spec, _, err := BuildDIP([]analyze.Sector{oneSector()}, groups(), DIPOptions{
		InitialAtom: "O", Classify: DefaultOptions(), SingletTripletRatio: 3.0,
	})
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Meta *struct {
			Kind                *string  `json:"kind"`
			SingletTripletRatio *float64 `json:"singlet_triplet_ratio"`
			EnergyUnit          *string  `json:"energy_unit"`
			InitialIonization   *struct {
				Atom *string `json:"atom"`
			} `json:"initial_ionization"`
			Irreps []string `json:"irreps"`
		} `json:"meta"`
		Channels []string `json:"channels"`
		Lines    []struct {
			Energy    *float64 `json:"energy"`
			Intensity *float64 `json:"intensity"`
			Channel   *string  `json:"channel"`
			Spin      *int     `json:"spin"`
			Irrep     *int     `json:"irrep"`
			StateRef  *string  `json:"state_ref"`
			PSPercent *float64 `json:"ps_percent"`
		} `json:"lines"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if doc.Meta == nil || doc.Meta.Kind == nil || doc.Meta.SingletTripletRatio == nil ||
		doc.Meta.EnergyUnit == nil || doc.Meta.InitialIonization == nil || doc.Meta.InitialIonization.Atom == nil {
		t.Fatalf("missing meta keys: %s", b)
	}
	if *doc.Meta.Kind != "dip" || *doc.Meta.InitialIonization.Atom != "O" {
		t.Errorf("meta values: kind=%q atom=%q", *doc.Meta.Kind, *doc.Meta.InitialIonization.Atom)
	}
	if len(doc.Channels) == 0 {
		t.Error("channels missing")
	}
	if len(doc.Lines) == 0 {
		t.Fatal("lines missing")
	}
	for i, l := range doc.Lines {
		if l.Energy == nil || l.Intensity == nil || l.Channel == nil || l.Spin == nil ||
			l.Irrep == nil || l.StateRef == nil || l.PSPercent == nil {
			t.Fatalf("line %d missing a required key: %s", i, b)
		}
	}
}
