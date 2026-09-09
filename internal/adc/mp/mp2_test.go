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

// mp2Serial is the original single-threaded reduction, in the exact i,j,a,b order
// MP2Corr used before it was chunked. It is the reference for how far the
// reassociation at the chunk boundaries is allowed to move the answer.
func mp2Serial(d *fcidump.Data, nocc int, eps []float64) float64 {
	n := d.NORB
	var e2 float64
	for i := 0; i < nocc; i++ {
		for j := 0; j < nocc; j++ {
			for a := nocc; a < n; a++ {
				for b := nocc; b < n; b++ {
					iajb := d.TwoE(i, a, j, b)
					ibja := d.TwoE(i, b, j, a)
					denom := eps[i] + eps[j] - eps[a] - eps[b]
					e2 += iajb * (2*iajb - ibja) / denom
				}
			}
		}
	}
	return e2
}

// TestMP2CorrChunkedReduction gates the one deliberate floating-point change in this
// package: MP2Corr sums per-chunk partials instead of one running total, so it may
// differ from the serial order in the last few ulp. Two things must hold — the drift
// stays orders below the 1e-8 Ha physics tolerance, and the value is reproducible,
// since the chunk count is a constant and the partials are combined in fixed order.
func TestMP2CorrChunkedReduction(t *testing.T) {
	d := loadData(t)
	nocc := NOcc(d)
	eps := OrbitalEnergies(d, nocc)

	want := mp2Serial(d, nocc, eps)
	got := MP2Corr(d, nocc, eps)
	if diff := math.Abs(got - want); diff > 1e-12 {
		t.Errorf("chunked MP2 = %.17g, serial-order %.17g (|Δ|=%.2e), more drift "+
			"than reassociation alone can explain", got, want, diff)
	}
	for range 4 {
		if again := MP2Corr(d, nocc, eps); again != got {
			t.Fatalf("MP2Corr not reproducible: %.17g then %.17g", got, again)
		}
	}
}
