package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestStackBaselines checks that baselines track the true relative energies but
// are nudged apart when basins are isoenergetic, which is what keeps duplicate
// minima from drawing exactly on top of one another.
func TestStackBaselines(t *testing.T) {
	traces := []stackTrace{{offset: 0}, {offset: 0}, {offset: 2}, {offset: 3}}
	stackBaselines(traces, 0.5) // mean spacing 1 -> minimum gap 0.5

	want := []float64{0, 0.5, 2, 3}
	for i, w := range want {
		if math.Abs(traces[i].base-w) > 1e-12 {
			t.Errorf("trace %d: base = %g, want %g", i, traces[i].base, w)
		}
	}
	for i := 1; i < len(traces); i++ {
		if traces[i].base < traces[i-1].base {
			t.Errorf("trace %d: baselines not monotonic", i)
		}
		if traces[i].base < traces[i].offset-1e-12 {
			t.Errorf("trace %d: baseline %g below its energy %g", i, traces[i].base, traces[i].offset)
		}
	}
}

// TestStackBaselinesDegenerate covers the all-isoenergetic case, where there is
// no energy span to key the spacing off and the traces must still separate.
func TestStackBaselinesDegenerate(t *testing.T) {
	traces := []stackTrace{{offset: 1}, {offset: 1}, {offset: 1}}
	stackBaselines(traces, 0.25)
	for i := 1; i < len(traces); i++ {
		if got := traces[i].base - traces[i-1].base; math.Abs(got-0.25) > 1e-12 {
			t.Errorf("gap %d = %g, want 0.25", i, got)
		}
	}
}

// TestStackDedup checks that near-identical curves fold into the first (most
// stable) member as aliases, and that a distinct curve survives.
func TestStackDedup(t *testing.T) {
	traces := []stackTrace{
		{label: "a", curve: []float64{0, 1, 0}},
		{label: "b", curve: []float64{0, 1.005, 0}}, // same basin, re-found
		{label: "c", curve: []float64{0, 0.5, 0}},   // genuinely different
	}
	kept, dropped := stackDedup(traces, 0.01)
	if dropped != 1 {
		t.Fatalf("dropped = %d, want 1", dropped)
	}
	if len(kept) != 2 {
		t.Fatalf("kept %d traces, want 2", len(kept))
	}
	if kept[0].label != "a" || len(kept[0].aliases) != 1 || kept[0].aliases[0] != "b" {
		t.Errorf("representative = %q aliases %v, want a [b]", kept[0].label, kept[0].aliases)
	}
	if kept[1].label != "c" || len(kept[1].aliases) != 0 {
		t.Errorf("second trace = %q aliases %v, want c []", kept[1].label, kept[1].aliases)
	}
}

// TestStackDedupOff checks that the zero tolerance the flag defaults to keeps
// every basin, so no data disappears unless it is asked for.
func TestStackDedupOff(t *testing.T) {
	traces := []stackTrace{
		{label: "a", curve: []float64{0, 1, 0}},
		{label: "b", curve: []float64{0, 1, 0}},
	}
	if _, dropped := stackDedup(traces, 0); dropped != 0 {
		t.Errorf("dropped = %d with tol 0, want 0", dropped)
	}
}

func TestStackTrim(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   []string
		want []string
	}{
		{"shared prefix", []string{"UW1_O2-N1H", "UW1_O4-N3H"}, []string{"O2-N1H", "O4-N3H"}},
		{"no underscore", []string{"alpha", "alfred"}, []string{"alpha", "alfred"}},
		{"nothing shared", []string{"a_x", "b_y"}, []string{"a_x", "b_y"}},
		{"trim would empty", []string{"UW1_", "UW1_O2"}, []string{"UW1_", "UW1_O2"}},
		{"single trace", []string{"UW1_O2"}, []string{"UW1_O2"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			traces := make([]stackTrace, len(tc.in))
			for i, l := range tc.in {
				traces[i].label = l
			}
			stackTrim(traces)
			for i, w := range tc.want {
				if traces[i].label != w {
					t.Errorf("label %d = %q, want %q", i, traces[i].label, w)
				}
			}
		})
	}
}

// TestStackColor checks the ramp runs dark (global minimum) to light (least
// stable) and stays inside its end points.
func TestStackColor(t *testing.T) {
	dark, light := stackColor(0), stackColor(1)
	if dark != stackRamp[len(stackRamp)-1] {
		t.Errorf("stackColor(0) = %v, want the dark end %v", dark, stackRamp[len(stackRamp)-1])
	}
	if light != stackRamp[0] {
		t.Errorf("stackColor(1) = %v, want the light end %v", light, stackRamp[0])
	}
	if mid := stackColor(0.5); mid.R <= dark.R || mid.R >= light.R {
		t.Errorf("stackColor(0.5).R = %d, want between %d and %d", mid.R, dark.R, light.R)
	}
	// Out-of-range input is clamped, not wrapped.
	if stackColor(-1) != dark || stackColor(2) != light {
		t.Error("stackColor does not clamp outside [0,1]")
	}
}

// TestStackFromManifest checks the manifest path end to end: group selection,
// the hartree->kcal conversion, re-referencing to the most stable run that
// actually has a spectrum, and skipping runs whose calculation has not landed.
func TestStackFromManifest(t *testing.T) {
	dir := t.TempDir()
	manifest := map[string]map[string]any{
		"A": {"n_waters": 1, "relaxed_energy": -100.0},
		"B": {"n_waters": 1, "relaxed_energy": -100.0 + 1/627.5094740631}, // +1 kcal/mol
		"C": {"n_waters": 1, "relaxed_energy": -200.0},                    // pending: no spectrum
		"D": {"n_waters": 2, "relaxed_energy": -300.0},                    // other group
	}
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	mPath := filepath.Join(dir, "m.json")
	if err := os.WriteFile(mPath, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"A", "B", "D"} {
		if err := os.WriteFile(filepath.Join(dir, n+".sip.json"), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	p := stackParams{
		manifest: mPath, specDir: dir, specSuffix: ".sip.json",
		energyKey: "relaxed_energy", groupKey: "n_waters", group: "1",
		manifestUnit: "hartree", offsetUnit: "kcal",
	}
	traces, err := stackFromManifest(p, stackUnits["kcal"].perHartree)
	if err != nil {
		t.Fatal(err)
	}
	if len(traces) != 2 {
		t.Fatalf("got %d traces, want 2 (C pending, D other group)", len(traces))
	}
	got := map[string]float64{}
	for _, tr := range traces {
		got[tr.label] = tr.offset
	}
	// C is more stable but has no spectrum, so A — not C — is the zero.
	if math.Abs(got["A"]) > 1e-9 {
		t.Errorf("A offset = %g, want 0", got["A"])
	}
	if math.Abs(got["B"]-1) > 1e-6 {
		t.Errorf("B offset = %g kcal/mol, want 1", got["B"])
	}
}

// TestStackLoadGeometry checks the geometry lookup beside each spectrum: the
// spectrum suffix is swapped for the geometry suffix, and a trace whose
// geometry is absent or unreadable is left without a picture rather than
// failing the whole figure.
func TestStackLoadGeometry(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"has", "broken"} {
		if err := os.WriteFile(filepath.Join(dir, n+".sip.json"), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "has.zmat"), []byte(uracilW1), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "broken.zmat"), []byte("zmat angstroms\nN\nC 1 r\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	traces := []stackTrace{
		{label: "has", path: filepath.Join(dir, "has.sip.json")},
		{label: "broken", path: filepath.Join(dir, "broken.sip.json")},
		{label: "missing", path: filepath.Join(dir, "missing.sip.json")},
	}
	stackLoadGeometry(traces, ".sip.json", ".zmat")

	if traces[0].depict == nil {
		t.Error("has: expected a depiction")
	} else if len(traces[0].depict.pts) != 15 {
		t.Errorf("has: drew %d atoms, want 15", len(traces[0].depict.pts))
	}
	// A symbolic Z-matrix is rejected, but only that trace loses its picture.
	if traces[1].depict != nil {
		t.Error("broken: expected no depiction from a symbolic Z-matrix")
	}
	if traces[2].depict != nil {
		t.Error("missing: expected no depiction")
	}
}
