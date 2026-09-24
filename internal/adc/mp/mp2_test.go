package mp

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
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

// TestRequireCanonical: every canonical FCIDUMP in testdata passes, and the
// localized-orbital dumps (rotated on purpose) are refused.
func TestRequireCanonical(t *testing.T) {
	root := filepath.Join("..", "..", "..", "testdata")
	var paths []string
	filepath.WalkDir(root, func(p string, e os.DirEntry, err error) error {
		if err == nil && !e.IsDir() && strings.HasSuffix(p, ".fcidump") {
			paths = append(paths, p)
		}
		return nil
	})
	if len(paths) == 0 {
		t.Fatal("no FCIDUMPs found")
	}
	sawLocalized := false
	for _, p := range paths {
		d, err := fcidump.ReadFile(p)
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		m := MaxFockOffDiagonal(d, NOcc(d))
		if strings.Contains(p, "matched") {
			// theADCcode's matched-integral tape is semi-canonical by construction (see
			// RequireCanonical); it is not a canonical-HF dump and not a localized one
			continue
		}
		localized := strings.Contains(p, "localized") || strings.Contains(p, "he3_ghost")
		sawLocalized = sawLocalized || localized
		err = RequireCanonical(d, NOcc(d))
		t.Logf("%s: max off-diagonal Fock %.2e", filepath.Base(p), m)
		if localized && err == nil {
			t.Errorf("%s: localized dump accepted as canonical (max off-diagonal %.2e)", p, m)
		}
		if !localized && err != nil {
			t.Errorf("%s: %v", p, err)
		}
	}
	if !sawLocalized {
		t.Error("no localized dump in testdata: the refusal path is untested")
	}
}

// TestSyntheticIsCanonical: the synthetic system's Fock matrix is diag(eps), and its
// integrals have the 8-fold symmetry.
func TestSyntheticIsCanonical(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	eps := []float64{-1.2, -0.9, -0.7, 0.3, 0.5, 0.8, 1.1}
	d, err := fcidump.Synthetic(len(eps), 6, eps, 0.2, rng.Float64)
	if err != nil {
		t.Fatal(err)
	}
	for p, e := range OrbitalEnergies(d, 3) {
		if math.Abs(e-eps[p]) > 1e-13 {
			t.Fatalf("eps[%d] = %g, want %g", p, e, eps[p])
		}
	}
	if off := MaxFockOffDiagonal(d, 3); off > 1e-13 {
		t.Fatalf("off-diagonal Fock %g", off)
	}
	n := d.NORB
	for p := range n {
		for q := range n {
			for r := range n {
				for s := range n {
					v := d.TwoE(p, q, r, s)
					if v != d.TwoE(q, p, r, s) || v != d.TwoE(r, s, p, q) || v != d.TwoE(p, q, s, r) {
						t.Fatalf("(%d%d|%d%d) breaks the 8-fold symmetry", p, q, r, s)
					}
				}
			}
		}
	}
	if _, err := fcidump.Synthetic(4, 3, eps[:4], 1, rng.Float64); err == nil {
		t.Fatal("odd electron count accepted")
	}
}
