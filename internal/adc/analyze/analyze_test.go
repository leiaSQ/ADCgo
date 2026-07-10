package analyze

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

func h2oSinglet(t *testing.T) *dip.Matrix {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := dip.NewSpace(nocc, d.NORB, nil, 0, dip.Singlet)
	return dip.New(sp, integrals.New(d, nocc, nil), eps, backend.Gonum{})
}

func TestBuildSector(t *testing.T) {
	mx := h2oSinglet(t)
	res := lanczos.SolveDense(mx, backend.Gonum{})
	sec := BuildSector(mx.Space(), res, Options{PSThresh: 5, CoeffThresh: 0.1}, nil)

	if sec.Irrep != 1 || sec.Spin != 1 {
		t.Fatalf("sector irrep/spin = %d/%d, want 1/1", sec.Irrep, sec.Spin)
	}
	if len(sec.States) == 0 {
		t.Fatal("no states")
	}

	// Ground dication state: ~39.17 eV, ps ~83%, dominated by |5,5> (HOMO²).
	g := sec.States[0]
	if math.Abs(g.EnergyEV-39.172) > 0.01 {
		t.Errorf("ground DIP energy = %.3f eV, want ~39.17", g.EnergyEV)
	}
	if math.Abs(g.PSPercent-83.35) > 0.5 {
		t.Errorf("ground pole strength = %.2f%%, want ~83.3", g.PSPercent)
	}
	if len(g.Leading) == 0 || g.Leading[0].I != 5 || g.Leading[0].J != 5 {
		t.Errorf("ground leading config = %+v, want first {5,5}", g.Leading)
	}

	// States must be energy-ordered with sequential 1-based indices, and every
	// leading list sorted by descending |coeff|.
	for i, s := range sec.States {
		if s.Index != i+1 {
			t.Errorf("state %d has index %d", i, s.Index)
		}
		if i > 0 && s.EnergyEV < sec.States[i-1].EnergyEV {
			t.Errorf("states not energy-ordered at %d", i)
		}
		for j := 1; j < len(s.Leading); j++ {
			if math.Abs(s.Leading[j].Coeff) > math.Abs(s.Leading[j-1].Coeff)+1e-12 {
				t.Errorf("state %d leading not sorted by |coeff|", s.Index)
			}
		}
	}
}
