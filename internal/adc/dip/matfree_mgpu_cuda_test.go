//go:build cuda

package dip

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
)

// TestSatelliteMatFreePerDeviceParity validates the per-device on-device satellite apply — each
// GPU recomputing only its own output row band, reading a gathered full-height slab over NVLink —
// against BOTH references that must agree with it:
//
//   - the dense single-node operator (the physics), and
//   - the host gather-apply-scatter path it replaces (the thing being optimized away).
//
// Bit-exactness is required, not just closeness: every output row is summed by one owner in a
// fixed candidate order regardless of how the rows are partitioned (satscalar.go elem() resolves
// orientation per scalar pair), so partitioning must not perturb the arithmetic at all. A drift
// here would mean rows are being double-counted or dropped, not that floating point moved.
//
// It wants >= 4 devices: at 2 the partition-resolution and row-band clamping are trivial and a
// boundary bug can hide. Skips otherwise.
func TestSatelliteMatFreePerDeviceParity(t *testing.T) {
	const minDev = 4
	if c := backend.DeviceCount("cuda"); c < minDev {
		t.Skipf("need >= %d cuda devices for a meaningful partitioning, have %d", minDev, c)
	}
	subs, err := backend.NewAll("cuda", minDev)
	if err != nil {
		t.Skipf("no cuda devices: %v", err)
	}
	npartWant := len(subs)

	rng := rand.New(rand.NewSource(31))
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		n, main := sp.Size(), sp.MainBlockSize()
		bounds := sp.PartitionBounds(npartWant)
		npart := len(bounds) - 1
		if npart < 2 || n <= 2*main*main {
			return // sector too small to partition; the shape invariant would reject it
		}
		dist, err := backend.NewDistributed(subs[:npart], n, main, bounds)
		if err != nil {
			t.Fatalf("spin=%v sym=%d: NewDistributed: %v", spin, sym, err)
		}
		pd, ok := dist.(backend.PartitionedDevices)
		if !ok {
			t.Fatal("distributed backend does not expose PartitionedDevices")
		}
		if !perDeviceSatelliteOK(pd) {
			t.Skipf("cuda sub-backends are not fully peered; per-device path unavailable")
		}

		// Report the band layout: a run where some partition owns no satellite rows, or where
		// one straddles the main boundary, is exactly the interesting case — surface it so a
		// green run on a trivial layout is not mistaken for coverage.
		lo, hi := satRowBands(bounds, main, n)
		empty, straddle := 0, 0
		for d := range npart {
			if hi[d] <= lo[d] {
				empty++
			}
			if bounds[d] < main && bounds[d+1] > main {
				straddle++
			}
		}
		t.Logf("spin=%v sym=%d: n=%d main=%d parts=%d (empty bands=%d, straddling=%d)",
			spin, sym, n, main, npart, empty, straddle)

		dense := New(sp, ints, eps, be)
		perDev := New(sp, ints, eps, dist)
		perDev.SetMatFree(matfree.On, 0)
		if !perDev.matFreeSatellite() {
			t.Fatalf("spin=%v sym=%d: matFreeSatellite() false on distributed cuda backend", spin, sym)
		}

		// Deflation is real for this backend (panels are sized to the max block width but an
		// apply commonly uses fewer columns), and the chunk loop must handle a final short
		// chunk, so exercise a width that is not a multiple of satChunkCols.
		for _, b := range []int{3, satChunkCols + 1} {
			panel := make([]float64, n*b)
			for i := range panel {
				panel[i] = rng.NormFloat64()
			}

			want := backend.BlockView{V: be.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			dense.ApplyBlockSatellite(want, backend.BlockView{V: be.Upload(panel), Rows: n, Cols: b, Ld: n})
			wantH := be.Download(want.V)

			got := backend.BlockView{V: dist.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			perDev.ApplyBlockSatellite(got, backend.BlockView{V: dist.Upload(panel), Rows: n, Cols: b, Ld: n})
			gotH := dist.Download(got.V)

			var maxErr float64
			for i := range wantH {
				if d := math.Abs(gotH[i] - wantH[i]); d > maxErr {
					maxErr = d
				}
			}
			if maxErr > 1e-10 {
				t.Errorf("spin=%v sym=%d b=%d: per-device vs dense max diff %g exceeds 1e-10",
					spin, sym, b, maxErr)
			}

			// Same panel through the host gather-apply-scatter fallback: the two -mgpu paths
			// must agree with each other, not merely both be near the dense reference.
			hostSubs := make([]backend.Backend, npart)
			for i := range hostSubs {
				hostSubs[i] = backend.Gonum{}
			}
			hdist, err := backend.NewDistributed(hostSubs, n, main, bounds)
			if err != nil {
				t.Fatalf("NewDistributed(host): %v", err)
			}
			hmx := New(sp, ints, eps, hdist)
			hmx.SetMatFree(matfree.On, 0)
			hgot := backend.BlockView{V: hdist.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			hmx.ApplyBlockSatellite(hgot, backend.BlockView{V: hdist.Upload(panel), Rows: n, Cols: b, Ld: n})
			hostH := hdist.Download(hgot.V)

			var maxPath float64
			for i := range hostH {
				if d := math.Abs(gotH[i] - hostH[i]); d > maxPath {
					maxPath = d
				}
			}
			if maxPath > 1e-10 {
				t.Errorf("spin=%v sym=%d b=%d: per-device vs gather-apply-scatter max diff %g exceeds 1e-10",
					spin, sym, b, maxPath)
			}
		}
	})
}
