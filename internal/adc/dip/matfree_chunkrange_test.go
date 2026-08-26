package dip

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
)

// scanRange is the O(members) formulation runBatches* used before: walk every member and keep the
// ones inside [lo,hi). It is the oracle sliceRange must reproduce exactly.
func scanRange(mem []int, lo, hi int) []int {
	var out []int
	for _, si := range mem {
		if si >= lo && si < hi {
			out = append(out, si)
		}
	}
	return out
}

// TestSliceRangeMatchesScan pins the binary-search member selection against the linear scan it
// replaced, across every boundary case that matters: empty lists, ranges before/after/spanning the
// whole list, single-element ranges, duplicated-free ascending lists with gaps, and ranges whose
// endpoints fall between members.
//
// This is the guard for the production apply fix. If sliceRange ever disagrees with the scan, the
// batched path silently DROPS operator blocks — the solve still runs and still looks converged, it
// is just wrong, which is exactly the failure mode that is hardest to notice.
func TestSliceRangeMatchesScan(t *testing.T) {
	rng := rand.New(rand.NewSource(11))

	lists := [][]int{
		{},
		{0},
		{5},
		{0, 1, 2, 3, 4},
		{2, 7, 11, 40, 41, 90},
	}
	// Plus randomized ascending lists with gaps.
	for range 20 {
		n := rng.Intn(30)
		v, cur := make([]int, 0, n), 0
		for range n {
			cur += 1 + rng.Intn(5)
			v = append(v, cur)
		}
		lists = append(lists, v)
	}

	for li, mem := range lists {
		for lo := -2; lo <= 100; lo++ {
			for _, hi := range []int{lo - 1, lo, lo + 1, lo + 7, 100, 101} {
				want := scanRange(mem, lo, hi)
				a, b := sliceRange(mem, lo, hi)
				if a > b {
					// An empty range must still yield a usable (possibly empty) slice.
					t.Fatalf("list %d [%d,%d): sliceRange returned inverted bounds %d>%d", li, lo, hi, a, b)
				}
				got := append([]int(nil), mem[a:b]...)
				if len(got) != len(want) {
					t.Fatalf("list %d %v [%d,%d): got %v, want %v", li, mem, lo, hi, got, want)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("list %d %v [%d,%d): got %v, want %v", li, mem, lo, hi, got, want)
					}
				}
			}
		}
	}
}

// TestBatchBlocksSorted pins the invariant sliceRange depends on: PlanBatches emits every batch's
// member list in ascending block order.
//
// It is asserted on the REAL satellite plans, not a synthetic input, because the ordering hazard is
// specific to them: PlanBatches collects members in ascending WRITE-OFFSET order, and for the
// transposed half of a symmetric operator the write offset is the block's COLUMN offset, which has
// no relation to block index. Before the sort was added, transposed batches came out unsorted, and
// binary-searching them would have dropped blocks.
func TestBatchBlocksSorted(t *testing.T) {
	sawTrans := false
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		mx := New(sp, ints, eps, be)
		defer mx.Release()
		p := mx.buildJIIBatchPlan()
		for bi, bt := range p.batches {
			if bt.Trans {
				sawTrans = true
			}
			for i := 1; i < len(bt.Blocks); i++ {
				if bt.Blocks[i-1] >= bt.Blocks[i] {
					t.Fatalf("spin=%v sym=%d: batch %d (trans=%v) not ascending at %d: %d >= %d",
						spin, sym, bi, bt.Trans, i, bt.Blocks[i-1], bt.Blocks[i])
				}
			}
		}
	})
	if !sawTrans {
		t.Error("no transposed batch in any plan — the ordering hazard went untested")
	}
}

// TestRunBatchesChunkedEqualsWhole drives the batched applier over the SAME plan twice — once in
// one range, once split into many small chunks — and requires agreement to a few ulp.
//
// Chunking is what the device path does (JIIFillBudgetElems bounds the fill scratch), and it is the
// only place the member-range selection is exercised. The test exists to catch a dropped or
// double-counted member: either moves a result by O(1), because a satellite block is a dense
// 154×154-scale contribution, so a 1e-12 relative bound separates them from rounding by four orders
// of magnitude.
//
// Why NOT bit-identical, which is what this test originally asserted. Within one batch the members
// have pairwise-disjoint outputs, so their order is irrelevant — that much the code comments claim
// and it holds. But blocks that share an output offset are deliberately placed in DIFFERENT batches
// (that is what PlanBatches' "depth" is), and chunking interleaves the batches differently from a
// whole-range run: whole issues all of batch 0, then all of batch 1; chunked issues batch 0's and
// batch 1's members for chunk 0, then both again for chunk 1. Contributions into a shared output row
// therefore accumulate in a different order, and fp addition is not associative. Measured here:
// exactly 1 ulp. This is structural and pre-existing, not something the range selection introduced,
// and it is consistent with the batched path's GPU parity tests, which have always used a tolerance
// (max|Δ| ≈ 5.7e-14) rather than bit-equality — unlike the per-scalar path, which is bit-exact.
func TestRunBatchesChunkedEqualsWhole(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		mx := New(sp, ints, eps, be)
		defer mx.Release()
		p := mx.buildJIIBatchPlan()
		if len(p.slots) == 0 {
			return
		}

		// Values are apply-invariant here: fill every slot once and reuse for both runs.
		mats := make([]backend.DeviceMat, len(p.slots))
		for i, s := range p.slots {
			blk, ok := mx.buildSlot(s)
			if !ok {
				t.Fatalf("spin=%v sym=%d: gate/value disagreement at slot %d", spin, sym, i)
			}
			mats[i] = be.UploadMat(blk)
		}

		n := mx.sp.Size()
		const cols = 3
		rng := rand.New(rand.NewSource(int64(len(p.slots))))
		hostIn := make([]float64, n*cols)
		for i := range hostIn {
			hostIn[i] = rng.NormFloat64()
		}
		in := backend.BlockView{V: be.Upload(hostIn), Rows: n, Cols: cols, Ld: n}

		run := func(chunks []jiiChunk) []float64 {
			outBuf := be.Alloc(n * cols)
			be.Zero(outBuf)
			out := backend.BlockView{V: outBuf, Rows: n, Cols: cols, Ld: n}
			for _, ch := range chunks {
				// mats is chunk-local in the device path (mats[si-lo]); mirror that exactly.
				p.runBatchesRange(be, mats[ch.lo:ch.hi], in, out, ch.lo, ch.hi)
			}
			return be.Download(outBuf)
		}

		whole := run([]jiiChunk{{0, len(p.slots), 0}})

		// Many small chunks, including size-1, to cross every batch boundary.
		var tiny []jiiChunk
		for lo := 0; lo < len(p.slots); lo += 3 {
			tiny = append(tiny, jiiChunk{lo, min(lo+3, len(p.slots)), 0})
		}
		split := run(tiny)

		if len(whole) != len(split) {
			t.Fatalf("spin=%v sym=%d: length mismatch %d vs %d", spin, sym, len(whole), len(split))
		}
		var maxRel float64
		for i := range whole {
			d := math.Abs(split[i] - whole[i])
			scale := math.Max(math.Abs(whole[i]), 1)
			if rel := d / scale; rel > maxRel {
				maxRel = rel
			}
		}
		if maxRel > 1e-12 {
			t.Errorf("spin=%v sym=%d: chunked apply differs from whole by %.3e relative — "+
				"far above rounding, so a block was dropped or double-counted", spin, sym, maxRel)
		}
	})
}
