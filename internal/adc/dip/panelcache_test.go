package dip

import (
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
)

// TestPanelCacheNotMutated guards the invariant integrals.Store's A/B/V caches rest on: the
// panels they hand out are SHARED, so a consumer that wrote through one would corrupt every
// later block built from it — silently, and only for runs long enough to reach the second use.
//
// It snapshots every panel the sector can ask for, drives the full operator build (BuildMatrix
// exercises the 2h/2h, both 2h↔3h1p coupling families and all three satellite families), and
// re-reads. Every consumer today accumulates FROM the panel via Mat.AddSubMat / Mat.AddSubVec;
// this fails the moment one stops doing so.
func TestPanelCacheNotMutated(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		nocc, nsym := sp.Nocc, ints.NSym()

		snapA := map[[3]int][]float64{}
		snapB := map[[3]int][]float64{}
		snapV := map[[4]int][]float64{}
		copyOf := func(x []float64) []float64 { return append([]float64(nil), x...) }
		for i := range nocc {
			for j := range nocc {
				for σ := range nsym {
					snapA[[3]int{i, j, σ}] = copyOf(ints.A(i, j, σ).Data)
					snapB[[3]int{i, j, σ}] = copyOf(ints.B(i, j, σ).Data)
					for k := range nocc {
						snapV[[4]int{i, j, k, σ}] = copyOf(ints.V(i, j, k, σ))
					}
				}
			}
		}

		mx := New(sp, ints, eps, be)
		defer mx.Release()
		mx.BuildMatrix()

		same := func(what string, key any, got, want []float64) {
			t.Helper()
			if len(got) != len(want) {
				t.Fatalf("spin=%v sym=%d: %s%v length %d, was %d", spin, sym, what, key, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("spin=%v sym=%d: cached %s%v entry %d changed: %v -> %v — a consumer "+
						"wrote through a shared panel", spin, sym, what, key, i, want[i], got[i])
				}
			}
		}
		for k, want := range snapA {
			same("A", k, ints.A(k[0], k[1], k[2]).Data, want)
		}
		for k, want := range snapB {
			same("B", k, ints.B(k[0], k[1], k[2]).Data, want)
		}
		for k, want := range snapV {
			same("V", k, ints.V(k[0], k[1], k[2], k[3]), want)
		}
	})
}
