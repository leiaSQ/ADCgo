package mp

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
)

type reference struct {
	Norb     int       `json:"norb"`
	Nelec    int       `json:"nelec"`
	ENuc     float64   `json:"e_nuc"`
	EScf     float64   `json:"e_scf"`
	EMP2Corr float64   `json:"e_mp2_corr"`
	MOEnergy []float64 `json:"mo_energy"`
}

func loadRef(t *testing.T) reference {
	t.Helper()
	b, err := os.ReadFile("../../../testdata/h2o.ref.json")
	if err != nil {
		t.Fatalf("read reference: %v", err)
	}
	var r reference
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("parse reference: %v", err)
	}
	return r
}

func loadData(t *testing.T) *fcidump.Data {
	t.Helper()
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	return d
}

// TestMP2Corr is the M0 acceptance gate: ADCgo's RHF-MP2 correlation energy,
// computed purely from the FCIDUMP, must match pyscf to ~1e-8 Ha.
func TestMP2Corr(t *testing.T) {
	r := loadRef(t)
	d := loadData(t)
	nocc := NOcc(d)
	eps := OrbitalEnergies(d, nocc)

	got := MP2Corr(d, nocc, eps)
	if diff := math.Abs(got - r.EMP2Corr); diff > 1e-8 {
		t.Fatalf("MP2 corr = %.12f Ha, want %.12f Ha (|Δ|=%.2e)", got, r.EMP2Corr, diff)
	}
}

// TestOrbitalEnergies checks that the canonical HF orbital energies
// reconstructed from the Fock diagonal match pyscf's mo_energy.
func TestOrbitalEnergies(t *testing.T) {
	r := loadRef(t)
	d := loadData(t)
	nocc := NOcc(d)
	eps := OrbitalEnergies(d, nocc)

	if len(eps) != len(r.MOEnergy) {
		t.Fatalf("got %d orbital energies, want %d", len(eps), len(r.MOEnergy))
	}
	for p := range eps {
		if diff := math.Abs(eps[p] - r.MOEnergy[p]); diff > 1e-7 {
			t.Errorf("ε[%d] = %.10f, want %.10f (|Δ|=%.2e)", p, eps[p], r.MOEnergy[p], diff)
		}
	}
}
