package main

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// The distributed (multi-GPU) backend row-partitions one sector across G sub-backends.
// These tests validate it against a single Gonum backend on the host, so the exact same
// code path is exercised before it ever runs on a GPU: the operator apply must match
// column-for-column, and Mode B SolveLowMem must return the same spectrum.

// distFixture reads h2o_dzp and returns the shared integrals/eps for building sectors.
func distFixture(t *testing.T) (*fcidump.Data, int, []float64, *integrals.Store) {
	t.Helper()
	d, err := fcidump.ReadFile("../../testdata/h2o_dzp.fcidump")
	if err != nil {
		t.Skipf("fixture not available: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	return d, nocc, eps, integrals.New(d, nocc, d.OrbSym)
}

// newDist builds a distributed backend over len(bounds)-1 Gonum partitions for a sector.
func newDist(t *testing.T, sp *dip.Space, g int) (backend.Backend, int) {
	t.Helper()
	bounds := sp.PartitionBounds(g)
	subs := make([]backend.Backend, len(bounds)-1)
	for i := range subs {
		subs[i] = backend.Gonum{}
	}
	be, err := backend.NewDistributed(subs, sp.Size(), sp.MainBlockSize(), bounds)
	if err != nil {
		t.Fatalf("NewDistributed(g=%d, bounds=%v): %v", g, bounds, err)
	}
	return be, len(subs)
}

// eligible reports whether a sector satisfies the distributed backend's shape invariant
// (n > 2·main²) and is non-trivial.
func eligible(sp *dip.Space) bool {
	n, main := sp.Size(), sp.MainBlockSize()
	return main > 0 && n > 2*main*main
}

// TestDistributedApplyMatchesSingle pins the row-partitioned operator apply (including the
// cross-partition input gather) to the single-backend ApplyBlock / ApplyBlockSatellite.
func TestDistributedApplyMatchesSingle(t *testing.T) {
	d, nocc, eps, ints := distFixture(t)
	rng := rand.New(rand.NewSource(7))
	const blk = 4
	tested := 0
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		for sym := range 4 {
			sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			if sp.Size() == 0 || !eligible(sp) {
				continue
			}
			n := sp.Size()
			single := backend.Gonum{}
			mxS := dip.New(sp, ints, eps, single)
			for _, g := range []int{2, 3, 4} {
				dbe, ndev := newDist(t, sp, g)
				mxD := dip.New(sp, ints, eps, dbe)

				in := make([]float64, n*blk)
				for i := range in {
					in[i] = rng.NormFloat64()
				}
				for _, sat := range []bool{false, true} {
					want := applyOnce(single, mxS, in, n, blk, sat)
					got := applyOnce(dbe, mxD, in, n, blk, sat)
					if rel := maxRelDiff(want, got); rel > 1e-10 {
						t.Errorf("spin=%v sym=%d n=%d g=%d(dev=%d) sat=%v: apply diff %.3e",
							spin, sym, n, g, ndev, sat, rel)
					}
				}
			}
			tested++
		}
	}
	if tested == 0 {
		t.Skip("no h2o_dzp sector satisfies n > 2·main²")
	}
	t.Logf("%d sectors matched (apply, block + satellite)", tested)
}

// applyOnce uploads in, applies the (full or satellite) block operator, and returns the host
// result.
func applyOnce(be backend.Backend, mx *dip.Matrix, in []float64, n, blk int, sat bool) []float64 {
	inV := backend.BlockView{V: be.Upload(in), Rows: n, Cols: blk, Ld: n}
	outV := backend.BlockView{V: be.Alloc(n * blk), Rows: n, Cols: blk, Ld: n}
	if sat {
		mx.ApplyBlockSatellite(outV, inV)
	} else {
		mx.ApplyBlock(outV, inV)
	}
	out := be.Download(outV.V)
	be.Free(inV.V)
	be.Free(outV.V)
	return out
}

// TestDistributedSolveLowMemMatchesSingle runs Mode B (lowmem-block 0) on the distributed
// backend and requires the same spectrum and pole strengths as the single backend.
func TestDistributedSolveLowMemMatchesSingle(t *testing.T) {
	d, nocc, eps, ints := distFixture(t)
	opts := lanczos.Options{MaxBlocks: 12, LowMemBlock: 0}
	tested := 0
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		for sym := range 4 {
			sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			if sp.Size() == 0 || !eligible(sp) {
				continue
			}
			single := backend.Gonum{}
			resS := lanczos.SolveLowMem(dip.New(sp, ints, eps, single), single, opts)

			for _, g := range []int{2, 3, 4} {
				dbe, ndev := newDist(t, sp, g)
				resD := lanczos.SolveLowMem(dip.New(sp, ints, eps, dbe), dbe, opts)

				m := min(len(resS.Values), len(resD.Values))
				if m == 0 {
					t.Errorf("spin=%v sym=%d g=%d: no eigenvalues (S=%d D=%d)",
						spin, sym, g, len(resS.Values), len(resD.Values))
					continue
				}
				k := min(m, 6)
				for i := range k {
					if diff := math.Abs(resS.Values[i] - resD.Values[i]); diff > 1e-6 {
						t.Errorf("spin=%v sym=%d n=%d g=%d(dev=%d): eig[%d] single=%.10f dist=%.10f diff=%.2e",
							spin, sym, sp.Size(), g, ndev, i, resS.Values[i], resD.Values[i], diff)
					}
					if ps := math.Abs(resS.PS[i] - resD.PS[i]); ps > 1e-5 {
						t.Errorf("spin=%v sym=%d g=%d: PS[%d] single=%.6f dist=%.6f diff=%.2e",
							spin, sym, g, i, resS.PS[i], resD.PS[i], ps)
					}
				}
			}
			tested++
		}
	}
	if tested == 0 {
		t.Skip("no h2o_dzp sector satisfies n > 2·main²")
	}
	t.Logf("%d sectors matched (SolveLowMem Mode B)", tested)
}

func maxRelDiff(want, got []float64) float64 {
	var maxDiff, scale float64
	for i := range want {
		scale = math.Max(scale, math.Abs(want[i]))
		maxDiff = math.Max(maxDiff, math.Abs(want[i]-got[i]))
	}
	return maxDiff / math.Max(scale, 1e-300)
}
