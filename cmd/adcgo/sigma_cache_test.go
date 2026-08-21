package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/selfenergy"
)

// sampleSigma builds an n×n Σ with distinguishable entries.
func sampleSigma(t *testing.T, n int) *selfenergy.Sigma {
	t.Helper()
	d := make([]float64, n*n)
	for i := range d {
		d[i] = 0.125 * float64(i+1)
	}
	s, err := selfenergy.FromMatrix(n, d)
	if err != nil {
		t.Fatalf("FromMatrix: %v", err)
	}
	return s
}

func testKey(fcidump string) sigmaCacheKey {
	return sigmaCacheKeyFor("infinite", 6, 3, 30, 1e-9, []float64{-1.5, -1.25, -0.75, 0.5, 1.25, 2}, fcidump)
}

// TestSigmaCacheRoundTrip: Σ written and read back under a matching key must be identical. This is
// the whole point — the production system's Σ(∞) costs 78 h to rebuild and is 351 KB to store.
func TestSigmaCacheRoundTrip(t *testing.T) {
	dir := t.TempDir()
	fc := filepath.Join(dir, "fake.fcidump")
	if err := os.WriteFile(fc, []byte("not really an fcidump"), 0o644); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, "sigma.cache")
	want := sampleSigma(t, 6)
	k := testKey(fc)

	if err := writeSigmaCache(p, k, want); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := readSigmaCache(p, k)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got == nil {
		t.Fatal("cache miss on the key it was just written with")
	}
	if got.N() != want.N() {
		t.Fatalf("n = %d, want %d", got.N(), want.N())
	}
	for i := range want.Data() {
		if got.Data()[i] != want.Data()[i] {
			t.Fatalf("element %d = %g, want %g", i, got.Data()[i], want.Data()[i])
		}
	}
	// And the adapter the solver actually consumes must agree.
	for i := range 6 {
		for j := range 6 {
			if got.Func()(i, j) != want.Func()(i, j) {
				t.Fatalf("Func mismatch at (%d,%d)", i, j)
			}
		}
	}
}

// TestSigmaCacheRejectsStale pins every way the cache must MISS rather than hand back a Σ built for
// a different problem. A wrong Σ does not crash — it silently shifts every main line by
// ~0.2-0.35 eV — so each of these is a correctness guard, not a convenience.
func TestSigmaCacheRejectsStale(t *testing.T) {
	dir := t.TempDir()
	fc := filepath.Join(dir, "fake.fcidump")
	if err := os.WriteFile(fc, []byte("not really an fcidump"), 0o644); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, "sigma.cache")
	if err := writeSigmaCache(p, testKey(fc), sampleSigma(t, 6)); err != nil {
		t.Fatalf("write: %v", err)
	}

	eps := []float64{-1.5, -1.25, -0.75, 0.5, 1.25, 2}
	cases := []struct {
		name string
		key  sigmaCacheKey
	}{
		{"different scheme", sigmaCacheKeyFor("fplus", 6, 3, 30, 1e-9, eps, fc)},
		{"different norb", sigmaCacheKeyFor("infinite", 7, 3, 30, 1e-9, eps, fc)},
		{"different nocc", sigmaCacheKeyFor("infinite", 6, 4, 30, 1e-9, eps, fc)},
		{"different maxit", sigmaCacheKeyFor("infinite", 6, 3, 200, 1e-9, eps, fc)},
		{"different akrit", sigmaCacheKeyFor("infinite", 6, 3, 30, 1e-16, eps, fc)},
		{"different orbital energies", sigmaCacheKeyFor("infinite", 6, 3, 30, 1e-9,
			[]float64{-1.5, -1.25, -0.75, 0.5, 1.25, 2.5}, fc)},
	}
	for _, c := range cases {
		got, err := readSigmaCache(p, c.key)
		if err != nil {
			t.Errorf("%s: unexpected error %v", c.name, err)
			continue
		}
		if got != nil {
			t.Errorf("%s: cache HIT on a key that must miss — this would silently use a Σ built "+
				"for a different problem", c.name)
		}
	}

	// A changed FCIDUMP must also invalidate, even at identical dimensions and energies.
	if err := os.WriteFile(fc, []byte("a different fcidump entirely"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := readSigmaCache(p, testKey(fc)); err != nil || got != nil {
		t.Errorf("cache survived an FCIDUMP change (got=%v err=%v)", got != nil, err)
	}
}

// TestSigmaCacheMissingAndCorrupt: absence is a plain miss (the common first-run case); a damaged
// file is an error the caller reports and then rebuilds from — never a silent wrong answer.
func TestSigmaCacheMissingAndCorrupt(t *testing.T) {
	dir := t.TempDir()
	k := testKey(filepath.Join(dir, "nope.fcidump"))

	got, err := readSigmaCache(filepath.Join(dir, "absent.cache"), k)
	if got != nil || err != nil {
		t.Errorf("absent cache: got=%v err=%v, want (nil, nil)", got != nil, err)
	}

	bad := filepath.Join(dir, "bad.cache")
	if err := os.WriteFile(bad, []byte("XXXXXXXXnot a sigma cache at all"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readSigmaCache(bad, k); err == nil {
		t.Error("corrupt cache accepted silently; want an error so the caller rebuilds")
	}
}
