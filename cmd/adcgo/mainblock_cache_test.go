package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
)

func mbKey() mainBlockCacheKey {
	eps := []float64{-1.5, -0.9, 0.3, 1.1}
	sig := func(i, j int) float64 { return 0.01 * float64(i*3+j) }
	return mainBlockCacheKeyFor(3, 0, 4, 2, 2, 17, false, nil, eps, sig, "/nonexistent/fcidump")
}

func mbMat() backend.Mat {
	m := backend.NewMat(2, 2)
	m.Set(0, 0, -1.25)
	m.Set(0, 1, 0.5)
	m.Set(1, 0, 0.5)
	m.Set(1, 1, -0.75)
	return m
}

// TestMainBlockCacheRoundTrip: a block written and read back under a matching key must come back
// bit-identical. The block is subtracted straight into the ionization spectrum, so anything less
// than exact equality is a wrong answer that still looks plausible.
func TestMainBlockCacheRoundTrip(t *testing.T) {
	p := filepath.Join(t.TempDir(), "mb.cache")
	want := mbMat()
	if err := writeMainBlockCache(p, mbKey(), want); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, ok, err := readMainBlockCache(p, mbKey())
	if err != nil || !ok {
		t.Fatalf("read: ok=%v err=%v", ok, err)
	}
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("dims %dx%d, want %dx%d", got.Rows, got.Cols, want.Rows, want.Cols)
	}
	for i := range want.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("entry %d: got %v want %v (must be bit-identical)", i, got.Data[i], want.Data[i])
		}
	}
}

// TestMainBlockCacheRejectsStale pins every way the cache must MISS rather than hand back a block
// built for a different problem. A false hit does not crash — it silently shifts every ionization
// line, which is the failure mode this guard exists to prevent, so each field is exercised.
func TestMainBlockCacheRejectsStale(t *testing.T) {
	eps := []float64{-1.5, -0.9, 0.3, 1.1}
	sig := func(i, j int) float64 { return 0.01 * float64(i*3+j) }
	const fc = "/nonexistent/fcidump"

	cases := []struct {
		name string
		key  mainBlockCacheKey
	}{
		{"order", mainBlockCacheKeyFor(4, 0, 4, 2, 2, 17, false, nil, eps, sig, fc)},
		{"irrep", mainBlockCacheKeyFor(3, 1, 4, 2, 2, 17, false, nil, eps, sig, fc)},
		{"norb", mainBlockCacheKeyFor(3, 0, 5, 2, 2, 17, false, nil, eps, sig, fc)},
		{"nocc", mainBlockCacheKeyFor(3, 0, 4, 3, 2, 17, false, nil, eps, sig, fc)},
		{"main", mainBlockCacheKeyFor(3, 0, 4, 2, 3, 17, false, nil, eps, sig, fc)},
		{"sector dim", mainBlockCacheKeyFor(3, 0, 4, 2, 2, 18, false, nil, eps, sig, fc)},
		{"wert3", mainBlockCacheKeyFor(3, 0, 4, 2, 2, 17, true, nil, eps, sig, fc)},
		{"cvs core", mainBlockCacheKeyFor(3, 0, 4, 2, 2, 17, false, []int{0}, eps, sig, fc)},
		{"orbital energies", mainBlockCacheKeyFor(3, 0, 4, 2, 2, 17, false, nil,
			[]float64{-1.5, -0.9, 0.3, 1.2}, sig, fc)},
		{"sigma values", mainBlockCacheKeyFor(3, 0, 4, 2, 2, 17, false, nil, eps,
			func(i, j int) float64 { return 0.02 * float64(i*3+j) }, fc)},
		{"sigma off", mainBlockCacheKeyFor(3, 0, 4, 2, 2, 17, false, nil, eps, nil, fc)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := filepath.Join(t.TempDir(), "mb.cache")
			if err := writeMainBlockCache(p, mbKey(), mbMat()); err != nil {
				t.Fatalf("write: %v", err)
			}
			_, ok, err := readMainBlockCache(p, tc.key)
			if err != nil {
				t.Fatalf("read: %v", err)
			}
			if ok {
				t.Fatalf("%s differs but the cache reported a HIT — a stale main block would "+
					"silently shift every ionization line", tc.name)
			}
		})
	}
}

// TestMainBlockCacheMissesAreNotErrors: absent and corrupt files must both read as "no usable
// cache" or a plain error, never as a hit, so the caller simply rebuilds.
func TestMainBlockCacheMissesAreNotErrors(t *testing.T) {
	dir := t.TempDir()

	if _, ok, err := readMainBlockCache(filepath.Join(dir, "absent"), mbKey()); ok || err != nil {
		t.Fatalf("absent file: ok=%v err=%v, want (false, nil)", ok, err)
	}

	bad := filepath.Join(dir, "corrupt")
	if err := os.WriteFile(bad, []byte("not an adcgo cache at all"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := readMainBlockCache(bad, mbKey()); ok {
		t.Fatal("corrupt file reported a hit")
	}

	// A truncated but well-formed header must not hand back a partial block.
	full := filepath.Join(dir, "trunc")
	if err := writeMainBlockCache(full, mbKey(), mbMat()); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(full)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, b[:len(b)-8], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := readMainBlockCache(full, mbKey()); ok {
		t.Fatal("truncated file reported a hit")
	}
}

// TestMainBlockCacheWriteIsAtomic: a completed write leaves no .tmp behind, so a crash mid-write
// can never be mistaken for a finished cache.
func TestMainBlockCacheWriteIsAtomic(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "mb.cache")
	if err := writeMainBlockCache(p, mbKey(), mbMat()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("temp file survived the write: %v", err)
	}
}
