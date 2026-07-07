package refout

import (
	"math"
	"path/filepath"
	"testing"
)

func refPath(name string) string {
	return filepath.Join("..", "..", "..", "testdata", "reference", name)
}

// TestParseADCDip1 checks the parser recovers the MO table, the spin-1 states
// (energy/ps/leading), and the population table of the reference symmetry-1 file.
func TestParseADCDip1(t *testing.T) {
	f, err := ParseFile(refPath("adcdip1.out"))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if f.Symmetry != 1 {
		t.Errorf("symmetry = %d, want 1", f.Symmetry)
	}
	if len(f.MOs) != 29 {
		t.Fatalf("parsed %d MOs, want 29", len(f.MOs))
	}
	// MO table spot checks (m.o. 1: sym 1, -1.35904; m.o. 2: sym 4, -0.731183).
	if f.MOs[0].Sym != 1 || math.Abs(f.MOs[0].EnergyAU+1.35904) > 1e-5 {
		t.Errorf("MO1 = %+v, want sym 1 energy -1.35904", f.MOs[0])
	}
	if f.MOs[1].Sym != 4 {
		t.Errorf("MO2 sym = %d, want 4", f.MOs[1].Sym)
	}

	// Population group names.
	if got := len(f.OneSiteGroups); got != 3 {
		t.Errorf("one-site groups = %v, want 3 (O,H1,H2)", f.OneSiteGroups)
	}

	// The lowest spin-1 state: 39.660357 eV, ps 83.39, leading <4,4|:-0.889428.
	var s1 *State
	for i := range f.States {
		if f.States[i].Spin == 1 {
			s1 = &f.States[i]
			break
		}
	}
	if s1 == nil {
		t.Fatal("no spin-1 state parsed")
	}
	if math.Abs(s1.EnergyEV-39.660357) > 1e-6 || math.Abs(s1.PSPercent-83.39) > 1e-6 {
		t.Errorf("state 1 = %.6f eV ps %.2f, want 39.660357 / 83.39", s1.EnergyEV, s1.PSPercent)
	}
	if len(s1.Leading) == 0 || s1.Leading[0].I != 4 || s1.Leading[0].J != 4 ||
		math.Abs(s1.Leading[0].Coeff+0.889428) > 1e-6 {
		t.Errorf("state 1 leading = %+v, want <4,4|:-0.889428", s1.Leading)
	}
	if s1.Pop == nil {
		t.Fatal("state 1 has no joined population row")
	}
	if o := s1.Pop.OneSite["O"]; math.Abs(o-0.8065) > 1e-4 {
		t.Errorf("state 1 O one-site = %.4f, want 0.8065", o)
	}
	// PopSum ≈ ps/100 (same invariant ADCanalysis's dipfile_test enforces).
	if d := math.Abs(s1.Pop.Sum() - s1.PSPercent/100); d > 1e-3 {
		t.Errorf("state 1 PopSum %.4f vs ps/100 %.4f (Δ=%.1e)", s1.Pop.Sum(), s1.PSPercent/100, d)
	}
}

// TestParseAllFiles confirms every reference file parses into both spin blocks
// with sane state fields, and that the NaN residues some (degenerate) blocks
// print are handled without error.
func TestParseAllFiles(t *testing.T) {
	for _, name := range []string{"adcdip1.out", "adcdip2.out", "adcdip3.out", "adcdip4.out"} {
		f, err := ParseFile(refPath(name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(f.States) == 0 {
			t.Fatalf("%s: no states parsed", name)
		}
		var haveSinglet, haveTriplet bool
		for _, s := range f.States {
			switch s.Spin {
			case 1:
				haveSinglet = true
			case 3:
				haveTriplet = true
			default:
				t.Errorf("%s: state %d has spin %d (want 1 or 3)", name, s.Index, s.Spin)
			}
			if s.PSPercent < 0 || s.PSPercent > 100.001 {
				t.Errorf("%s: state %d ps %.2f out of range", name, s.Index, s.PSPercent)
			}
		}
		if !haveSinglet || !haveTriplet {
			t.Errorf("%s: missing a spin block (singlet=%v triplet=%v)", name, haveSinglet, haveTriplet)
		}
	}
}
