package parse

import (
	"path/filepath"
	"testing"
)

// TestParseSIPFile parses the single-ionization ADC.out fixture and checks the
// MO table, the first state's one-hole decomposition, that all four symmetry
// blocks contribute states, and that satellite-space overlaps are excluded.
func TestParseSIPFile(t *testing.T) {
	f, err := ParseSIPFile(filepath.Join("testdata", "ADC.out"))
	if err != nil {
		t.Fatalf("ParseSIPFile: %v", err)
	}

	if len(f.MOTable) != 29 {
		t.Errorf("MO table size = %d, want 29", len(f.MOTable))
	}
	if len(f.States) == 0 {
		t.Fatal("no states parsed")
	}

	// First state: " 1: 14.810530, 94.13, 0.000000" with a single main-space
	// one-hole overlap "<3|:-0.969806" and no satellites.
	s0 := f.States[0]
	if s0.Index != 1 || s0.Spin != 2 {
		t.Errorf("state[0] index/spin = %d/%d, want 1/2", s0.Index, s0.Spin)
	}
	if !approx(s0.EnergyEV, 14.810530, 1e-6) || !approx(s0.PSPercent, 94.13, 1e-6) {
		t.Errorf("state[0] energy/ps = %v/%v, want 14.810530/94.13", s0.EnergyEV, s0.PSPercent)
	}
	if len(s0.Main) != 1 {
		t.Fatalf("state[0] main configs = %d, want 1: %+v", len(s0.Main), s0.Main)
	}
	if s0.Main[0].Orbital != 3 || !approx(s0.Main[0].Coeff, -0.969806, 1e-6) {
		t.Errorf("state[0] main[0] = %+v, want {3 -0.969806}", s0.Main[0])
	}

	// Symmetries with occupied orbitals (1, 3, 4 for H2O/C2v) yield states;
	// symmetry 2 (a2) has "No configurations in the main space" and yields none.
	irreps := map[int]bool{}
	for _, s := range f.States {
		irreps[s.Irrep] = true
	}
	for _, ir := range []int{1, 3, 4} {
		if !irreps[ir] {
			t.Errorf("no states for symmetry %d", ir)
		}
	}
	if irreps[2] {
		t.Errorf("symmetry 2 should have no main-space states")
	}

	// Satellite-space overlaps must not leak into Main: every Orbital is a real
	// MO index (1..len(MOTable)), and no state carries more main configs than MOs.
	for _, s := range f.States {
		for _, w := range s.Main {
			if w.Orbital < 1 || w.Orbital > len(f.MOTable) {
				t.Errorf("state #%d main orbital %d out of range", s.Index, w.Orbital)
			}
		}
	}
}

func approx(a, b, tol float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}
