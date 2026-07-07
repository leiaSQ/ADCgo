package fcidump

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func loadData(t *testing.T) *Data {
	t.Helper()
	d, err := ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	return d
}

func TestHeader(t *testing.T) {
	d := loadData(t)
	if d.NORB != 24 {
		t.Errorf("NORB = %d, want 24", d.NORB)
	}
	if d.NELEC != 10 {
		t.Errorf("NELEC = %d, want 10", d.NELEC)
	}
	if d.MS2 != 0 {
		t.Errorf("MS2 = %d, want 0", d.MS2)
	}
	if len(d.OrbSym) != 24 {
		t.Errorf("len(OrbSym) = %d, want 24", len(d.OrbSym))
	}
}

// TestEcore: with no frozen core the FCIDUMP core energy is the nuclear
// repulsion, which the reference records as e_nuc.
func TestEcore(t *testing.T) {
	d := loadData(t)
	b, err := os.ReadFile("../../../testdata/h2o.ref.json")
	if err != nil {
		t.Fatalf("read reference: %v", err)
	}
	var r struct {
		ENuc float64 `json:"e_nuc"`
	}
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	if diff := math.Abs(d.Ecore - r.ENuc); diff > 1e-9 {
		t.Errorf("Ecore = %.12f, want e_nuc %.12f (|Δ|=%.2e)", d.Ecore, r.ENuc, diff)
	}
}

func TestPermutationSymmetry(t *testing.T) {
	d := loadData(t)
	// One-electron integrals are symmetric.
	for p := 0; p < d.NORB; p++ {
		for q := 0; q < d.NORB; q++ {
			if d.OneE(p, q) != d.OneE(q, p) {
				t.Fatalf("h not symmetric at (%d,%d)", p, q)
			}
		}
	}
	// Two-electron 8-fold symmetry on a representative index tuple.
	p, q, r, s := 0, 3, 5, 2
	v := d.TwoE(p, q, r, s)
	perms := [][4]int{
		{q, p, r, s}, {p, q, s, r}, {q, p, s, r},
		{r, s, p, q}, {s, r, p, q}, {r, s, q, p}, {s, r, q, p},
	}
	for _, pm := range perms {
		if got := d.TwoE(pm[0], pm[1], pm[2], pm[3]); got != v {
			t.Errorf("(%d%d|%d%d)=%.10f but permutation %v=%.10f", p, q, r, s, v, pm, got)
		}
	}
}
