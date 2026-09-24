package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestReadTDM checks that a -tdm document is flattened into the (energy,
// intensity, channel) spectrum the renderer expects: omega_ev → energy,
// osc → intensity, the three transition families → channels in canonical order,
// and non-positive oscillator strengths dropped.
func TestReadTDM(t *testing.T) {
	const doc = `{
	  "norb": 10, "order": 3,
	  "emissions": [
	    {"omega_ev": 12.5, "osc": 0.30},
	    {"omega_ev": 9.0,  "osc": 0.0}
	  ],
	  "photoionization": [
	    {"channels": [
	      {"omega_ev": 540.1, "osc": 0.80},
	      {"omega_ev": 545.6, "osc": 0.05},
	      {"omega_ev": 550.0, "osc": -1e-9}
	    ]}
	  ]
	}`
	path := filepath.Join(t.TempDir(), "tdm.json")
	if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}

	s, err := readTDM(path)
	if err != nil {
		t.Fatalf("readTDM: %v", err)
	}
	if s.Meta.Kind != "tdm" || s.Meta.EnergyUnit != "eV" {
		t.Errorf("meta = %+v, want kind=tdm unit=eV", s.Meta)
	}

	// Two positive emission/photoionization lines survive; the osc=0 emission and
	// the negative photoionization channel are dropped.
	if len(s.Lines) != 3 {
		t.Fatalf("got %d lines, want 3: %+v", len(s.Lines), s.Lines)
	}
	got := map[string]int{}
	for _, l := range s.Lines {
		got[l.Channel]++
		if l.Intensity <= 0 {
			t.Errorf("line with non-positive intensity leaked: %+v", l)
		}
	}
	if got["emission"] != 1 || got["photoionization"] != 2 {
		t.Errorf("channel counts = %v, want emission:1 photoionization:2", got)
	}

	// Channels are only the present families, in canonical order.
	want := []string{"emission", "photoionization"}
	if len(s.Channels) != len(want) {
		t.Fatalf("channels = %v, want %v", s.Channels, want)
	}
	for i := range want {
		if s.Channels[i] != want[i] {
			t.Errorf("channels[%d] = %q, want %q", i, s.Channels[i], want[i])
		}
	}

	// The first emission maps omega_ev→energy and osc→intensity verbatim.
	var em *specLine
	for i := range s.Lines {
		if s.Lines[i].Channel == "emission" {
			em = &s.Lines[i]
			break
		}
	}
	if em == nil || em.Energy != 12.5 || em.Intensity != 0.30 {
		t.Errorf("emission line = %+v, want energy=12.5 intensity=0.30", em)
	}
}
