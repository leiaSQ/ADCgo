package dip

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestJIIBatchPlanMatchesGateWalk is the mitigation for the batched path's headline risk: a
// planning-pass bug whose block set silently disagrees with the enumeration the loop-based
// applier walks. That would drop or double-count operator contributions and show up only as small
// numeric drift — the failure mode the loosened tolerances elsewhere make easy to miss.
//
// It cross-checks three things against an independent walk: the same (rowOff, colOff) block set,
// the same shapes, and — via PlanBatches — that every block is applied the right NUMBER of times
// (once if on the block diagonal, twice otherwise, as the symmetric operator requires).
func TestJIIBatchPlanMatchesGateWalk(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		mx := New(sp, ints, eps, be)
		p := mx.buildJIIBatchPlan()

		// Independent walk over ALL THREE satellite blocks, enumerated separately from
		// buildJIIBatchPlan and mirroring the loop applier's passes exactly.
		type key struct{ rowOff, colOff int }
		want := map[key][2]int{} // -> (rows, cols)
		wantDiag := map[key]bool{}
		for gr := range sp.JII { // jiiLKK: JII×JII, gc<=gr
			r0 := sp.JII[gr]
			rc := sp.Configs[r0]
			for gc := 0; gc <= gr; gc++ {
				c0 := sp.JII[gc]
				if rows, cols, ok := mx.blk.jiiLKKGate(rc, sp.Configs[c0]); ok {
					want[key{r0, c0}] = [2]int{rows, cols}
					wantDiag[key{r0, c0}] = gr == gc
				}
			}
		}
		for gr := range sp.IJK { // ijkMLL: IJK×JII, never diagonal
			r0 := sp.IJK[gr]
			rc := sp.Configs[r0]
			for _, c0 := range sp.JII {
				if rows, cols, ok := mx.blk.ijkMLLGate(rc, sp.Configs[c0]); ok {
					want[key{r0, c0}] = [2]int{rows, cols}
					wantDiag[key{r0, c0}] = false
				}
			}
		}
		for gr := range sp.IJK { // ijkLMN: IJK×IJK, gc<=gr
			r0 := sp.IJK[gr]
			rc := sp.Configs[r0]
			for gc := 0; gc <= gr; gc++ {
				c0 := sp.IJK[gc]
				if rows, cols, ok := mx.blk.ijkLMNGate(rc, sp.Configs[c0]); ok {
					want[key{r0, c0}] = [2]int{rows, cols}
					wantDiag[key{r0, c0}] = gr == gc
				}
			}
		}

		if len(p.slots) != len(want) {
			t.Fatalf("spin=%v sym=%d: plan has %d slots, gate walk found %d blocks",
				spin, sym, len(p.slots), len(want))
		}
		for _, s := range p.slots {
			k := key{s.rowOff, s.colOff}
			w, ok := want[k]
			if !ok {
				t.Fatalf("spin=%v sym=%d: plan contains block (%d,%d) the gate walk does not",
					spin, sym, s.rowOff, s.colOff)
			}
			if s.rows != w[0] || s.cols != w[1] {
				t.Errorf("spin=%v sym=%d: block (%d,%d) shape %dx%d, gate says %dx%d",
					spin, sym, s.rowOff, s.colOff, s.rows, s.cols, w[0], w[1])
			}
			if s.diag != wantDiag[k] {
				t.Errorf("spin=%v sym=%d: block (%d,%d) diag=%v, want %v",
					spin, sym, s.rowOff, s.colOff, s.diag, wantDiag[k])
			}
		}

		// Application count: a diagonal block must be issued exactly once, an off-diagonal one
		// exactly twice (A and Aᵀ). Getting this wrong is the double-count/drop bug.
		applied := map[int]int{}
		for _, bt := range p.batches {
			for _, si := range bt.Blocks {
				applied[si]++
			}
		}
		for i, s := range p.slots {
			wantN := 2
			if s.diag {
				wantN = 1
			}
			if applied[i] != wantN {
				t.Errorf("spin=%v sym=%d: block (%d,%d) diag=%v applied %d times, want %d",
					spin, sym, s.rowOff, s.colOff, s.diag, applied[i], wantN)
			}
		}

		// Disjointness within a batch — the invariant that makes a concurrent batched GEMM safe.
		for bi, bt := range p.batches {
			seen := map[int]bool{}
			for _, si := range bt.Blocks {
				s := p.slots[si]
				off := s.rowOff
				if bt.Trans {
					off = s.colOff
				}
				if seen[off] {
					t.Errorf("spin=%v sym=%d: batch %d writes offset %d twice (races)",
						spin, sym, bi, off)
				}
				seen[off] = true
			}
		}
	})
}

// TestSatelliteMatFreeBatchedEqualsLoop is the whole-region gate now that the batched plan covers
// all three satellite blocks (jiiLKK, ijkMLL, ijkLMN): the batched applier must reproduce the
// complete loop applier, not just its jiiLKK half.
//
// The loop applier (newSatelliteMatFreeExcept(false)) is deliberately kept for exactly this — it
// is the reference the batched path is measured against, so it is not dead code even though the
// host production path no longer calls it.
func TestSatelliteMatFreeBatchedEqualsLoop(t *testing.T) {
	rng := rand.New(rand.NewSource(404))
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		n := sp.Size()
		const b = 4
		panel := make([]float64, n*b)
		for i := range panel {
			panel[i] = rng.NormFloat64()
		}

		run := func(part matFreePart) []float64 {
			outV := be.Alloc(n * b)
			be.Zero(outV)
			part.apply(
				backend.BlockView{V: be.Upload(panel), Rows: n, Cols: b, Ld: n},
				backend.BlockView{V: outV, Rows: n, Cols: b, Ld: n},
			)
			return be.Download(outV)
		}

		want := run(New(sp, ints, eps, be).newSatelliteMatFreeExcept(false))
		got := run(New(sp, ints, eps, be).newJIIMatFreeBatched())

		var maxErr, scale float64
		for i := range want {
			if d := math.Abs(got[i] - want[i]); d > maxErr {
				maxErr = d
			}
			if a := math.Abs(want[i]); a > scale {
				scale = a
			}
		}
		if maxErr > 1e-10*(1+scale) {
			t.Errorf("spin=%v sym=%d: batched whole-satellite vs loop: max |Δ| = %g (scale %g)",
				spin, sym, maxErr, scale)
		}

		// Main-space rows must remain literally zero.
		for j := range b {
			for i := range sp.MainBlockSize() {
				if got[i+j*n] != 0 {
					t.Fatalf("spin=%v sym=%d: wrote main-space row %d col %d (%g)",
						spin, sym, i, j, got[i+j*n])
				}
			}
		}
	})
}

// TestJIIBatchedSymmetryOff covers the degenerate case the h2oSectors sweep never reaches: with
// symmetry OFF every virtual lands in ONE group, so blocks are large and uniform instead of small
// and ragged — which is the regime production actually runs in (the production system is C1: one irrep,
// nvir=154, blocks 154×154). It is also the case where PlanBatches emits few, wide batches, so it
// exercises batch DEPTH where the symmetric sectors exercise shape bucketing.
func TestJIIBatchedSymmetryOff(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil) // nil orbSym = symmetry off
	be := backend.Gonum{}
	rng := rand.New(rand.NewSource(202))

	for _, spin := range []Spin{Singlet, Triplet} {
		sp := NewSpace(nocc, d.NORB, nil, 0, spin)
		if sp.Size() == 0 || sp.Size() == sp.MainBlockSize() {
			continue
		}
		n := sp.Size()
		const b = 3
		panel := make([]float64, n*b)
		for i := range panel {
			panel[i] = rng.NormFloat64()
		}

		run := func(part matFreePart) []float64 {
			outV := be.Alloc(n * b)
			be.Zero(outV)
			part.apply(
				backend.BlockView{V: be.Upload(panel), Rows: n, Cols: b, Ld: n},
				backend.BlockView{V: outV, Rows: n, Cols: b, Ld: n},
			)
			return be.Download(outV)
		}
		refOut := run(New(sp, ints, eps, be).newSatelliteMatFreeExcept(false))
		mx := New(sp, ints, eps, be)
		p := mx.buildJIIBatchPlan()
		got := run(mx.newJIIMatFreeBatched())

		var maxErr, scale float64
		for i := range refOut {
			if dd := math.Abs(got[i] - refOut[i]); dd > maxErr {
				maxErr = dd
			}
			if a := math.Abs(refOut[i]); a > scale {
				scale = a
			}
		}
		if maxErr > 1e-10*(1+scale) {
			t.Errorf("spin=%v symmetry-off: batched vs loop max |Δ| = %g", spin, maxErr)
		}
		t.Logf("spin=%v symmetry-off: n=%d blocks=%d batches=%d max|Δ|=%g",
			spin, n, len(p.slots), len(p.batches), maxErr)
	}
}

// TestAppChunks pins the application-order chunking that both bounds the device fill (the 424 GB
// OOM, job 14026481) and keeps the batched GEMM actually batched.
//
// The invariants: chunks tile the application range contiguously with no gap or overlap; NO CHUNK
// CROSSES A BATCH BOUNDARY (this is what makes a chunk issue as exactly one GemmMatBatched call —
// cutting the SLOT order instead put one member of each of ~685 batches in every chunk and drove
// members/call to 2.2 at production scale, job 14391094); each chunk's element total matches the
// blocks it covers; every chunk except a lone oversized block stays within budget; and the
// applications are exactly the batch members, in batch order. These are the properties the
// chunk-local BufOff, the pointer-offset fill, and runChunk's single-batch assumption depend on.
func TestAppChunks(t *testing.T) {
	slot := func(rows, cols int) jiiSlot { return jiiSlot{rows: rows, cols: cols} }
	batch := func(trans bool, blocks ...int) backend.Batch {
		return backend.Batch{Trans: trans, Blocks: blocks}
	}

	cases := []struct {
		name    string
		slots   []jiiSlot
		batches []backend.Batch
		owned   [][]int
		budget  int
	}{
		{name: "empty", budget: 10},
		{
			name:    "single",
			slots:   []jiiSlot{slot(3, 3)},
			batches: []backend.Batch{batch(false, 0)},
			budget:  100,
		},
		{
			name:    "one-oversized-block", // 400 > budget: its own chunk
			slots:   []jiiSlot{slot(20, 20)},
			batches: []backend.Batch{batch(false, 0)},
			budget:  10,
		},
		{
			name:    "exact-fit",
			slots:   []jiiSlot{slot(2, 2), slot(2, 2), slot(2, 2)},
			batches: []backend.Batch{batch(false, 0, 1, 2)},
			budget:  4,
		},
		{
			name:    "splits-within-one-batch", // 9 each, budget 20 -> 2 per chunk
			slots:   []jiiSlot{slot(3, 3), slot(3, 3), slot(3, 3), slot(3, 3)},
			batches: []backend.Batch{batch(false, 0, 1, 2, 3)},
			budget:  20,
		},
		{
			name:    "oversized-midstream",
			slots:   []jiiSlot{slot(2, 2), slot(10, 10), slot(2, 2)},
			batches: []backend.Batch{batch(false, 0, 1, 2)},
			budget:  8,
		},
		{
			// Two batches that would comfortably share a chunk by budget alone: the cut must
			// still fall on the boundary, because runChunk issues one Trans per chunk.
			name:    "batch-boundary-forces-a-cut",
			slots:   []jiiSlot{slot(1, 1), slot(1, 1)},
			batches: []backend.Batch{batch(false, 0, 1), batch(true, 0, 1)},
			budget:  1000,
		},
		{
			// The -mgpu case: a device sees only the members it owns, and an entirely unowned
			// batch contributes no chunk at all.
			name:    "owned-subset-skips-empty-batches",
			slots:   []jiiSlot{slot(2, 2), slot(2, 2), slot(2, 2)},
			batches: []backend.Batch{batch(false, 0, 1, 2), batch(true, 0, 1, 2)},
			owned:   [][]int{{1}, {}},
			budget:  1000,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := &jiiBatchPlan{slots: tc.slots, batches: tc.batches}
			apps, chunks := p.appChunks(tc.owned, tc.budget)

			// The applications must be exactly the (owned) batch members, batch by batch, in each
			// batch's own order — the order the device fill SoA is built in.
			var wantApps []int32
			for bi := range tc.batches {
				mem := tc.batches[bi].Blocks
				if tc.owned != nil {
					mem = tc.owned[bi]
				}
				for _, si := range mem {
					wantApps = append(wantApps, int32(si))
				}
			}
			if len(apps) != len(wantApps) {
				t.Fatalf("got %d applications, want %d", len(apps), len(wantApps))
			}
			for i := range apps {
				if apps[i] != wantApps[i] {
					t.Fatalf("application %d: got slot %d, want %d", i, apps[i], wantApps[i])
				}
			}

			if len(wantApps) == 0 {
				if apps != nil || chunks != nil {
					t.Fatalf("no applications: got %d apps / %d chunks, want none", len(apps), len(chunks))
				}
				return
			}

			// Contiguous tiling: first starts at 0, each continues the previous, last ends at len.
			if chunks[0].lo != 0 {
				t.Fatalf("first chunk starts at %d, want 0", chunks[0].lo)
			}
			for i := 1; i < len(chunks); i++ {
				if chunks[i].lo != chunks[i-1].hi {
					t.Fatalf("gap/overlap: chunk %d starts at %d, previous ended at %d",
						i, chunks[i].lo, chunks[i-1].hi)
				}
			}
			if last := chunks[len(chunks)-1].hi; last != len(apps) {
				t.Fatalf("last chunk ends at %d, want %d", last, len(apps))
			}

			// Every application's batch, recovered independently from the member lists.
			batchOf := make([]int, len(apps))
			at := 0
			for bi := range tc.batches {
				mem := tc.batches[bi].Blocks
				if tc.owned != nil {
					mem = tc.owned[bi]
				}
				for range mem {
					batchOf[at] = bi
					at++
				}
			}

			for i, ch := range chunks {
				if ch.hi <= ch.lo {
					t.Fatalf("chunk %d is empty: [%d,%d)", i, ch.lo, ch.hi)
				}
				// THE invariant: one batch per chunk.
				for e := ch.lo; e < ch.hi; e++ {
					if batchOf[e] != ch.batch {
						t.Fatalf("chunk %d claims batch %d but application %d belongs to batch %d",
							i, ch.batch, e, batchOf[e])
					}
				}
				sum := 0
				for e := ch.lo; e < ch.hi; e++ {
					sl := tc.slots[apps[e]]
					sum += sl.rows * sl.cols
				}
				if sum != ch.elems {
					t.Errorf("chunk %d elems %d, recomputed %d", i, ch.elems, sum)
				}
				// Within budget unless the chunk is a single block that alone exceeds it.
				if ch.elems > tc.budget && ch.hi-ch.lo != 1 {
					t.Errorf("chunk %d has %d elems over budget %d with %d blocks (only a lone "+
						"oversized block may exceed budget)", i, ch.elems, tc.budget, ch.hi-ch.lo)
				}
			}
		})
	}
}

// TestAppChunksCoverEveryApplicationOnRealSectors is the whole-plan counterpart: on the real h2o
// sectors, with a budget small enough to force many chunks, every planned block must be applied
// exactly as many times as PlanBatches says (once on the block diagonal, twice otherwise), each
// chunk must stay inside one batch, and the per-device split must partition — not duplicate or
// drop — the applications across partitions.
func TestAppChunksCoverEveryApplicationOnRealSectors(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		mx := New(sp, ints, eps, be)
		p := mx.buildJIIBatchPlan()
		if len(p.slots) == 0 {
			return
		}

		// Force many chunks: a budget of one element cuts after every block.
		apps, chunks := p.appChunks(nil, 1)

		want := map[int]int{} // slot -> number of applications PlanBatches asks for
		for _, bt := range p.batches {
			for _, si := range bt.Blocks {
				want[si]++
			}
		}
		got := map[int]int{}
		for _, si := range apps {
			got[int(si)]++
		}
		for si, n := range want {
			if got[si] != n {
				t.Fatalf("spin=%v sym=%d: slot %d applied %d times, plan asks for %d",
					spin, sym, si, got[si], n)
			}
		}
		if len(got) != len(want) {
			t.Fatalf("spin=%v sym=%d: %d slots applied, plan covers %d", spin, sym, len(got), len(want))
		}

		covered := 0
		for _, ch := range chunks {
			if ch.batch < 0 || ch.batch >= len(p.batches) {
				t.Fatalf("chunk names batch %d, plan has %d", ch.batch, len(p.batches))
			}
			covered += ch.hi - ch.lo
		}
		if covered != len(apps) {
			t.Fatalf("spin=%v sym=%d: chunks cover %d applications, want %d", spin, sym, covered, len(apps))
		}

		// Two partitions split at a group boundary: the per-device application lists must be a
		// partition of the whole-plan list.
		bounds := sp.PartitionBounds(2)
		perDev := map[int]int{}
		for d := range 2 {
			members := make([][]int, len(p.batches))
			for bi, bt := range p.batches {
				for _, si := range bt.Blocks {
					off := p.slots[si].rowOff
					if bt.Trans {
						off = p.slots[si].colOff
					}
					if ownerOf(bounds, off) == d {
						members[bi] = append(members[bi], si)
					}
				}
			}
			dApps, dChunks := p.appChunks(members, JIIFillBudgetElems)
			for _, si := range dApps {
				perDev[int(si)]++
			}
			for _, ch := range dChunks {
				for e := ch.lo; e < ch.hi; e++ {
					si := int(dApps[e])
					off := p.slots[si].rowOff
					if p.batches[ch.batch].Trans {
						off = p.slots[si].colOff
					}
					if ownerOf(bounds, off) != d {
						t.Fatalf("spin=%v sym=%d: device %d fills slot %d owned by %d",
							spin, sym, d, si, ownerOf(bounds, off))
					}
				}
			}
		}
		for si, n := range want {
			if perDev[si] != n {
				t.Fatalf("spin=%v sym=%d: slot %d applied %d times across partitions, want %d",
					spin, sym, si, perDev[si], n)
			}
		}
	})
}

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

		whole := run([]jiiChunk{{lo: 0, hi: len(p.slots)}})

		// Many small chunks, including size-1, to cross every batch boundary.
		var tiny []jiiChunk
		for lo := 0; lo < len(p.slots); lo += 3 {
			tiny = append(tiny, jiiChunk{lo: lo, hi: min(lo+3, len(p.slots))})
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

// TestRunChunkEqualsWholeBatches is the application-order counterpart: the device path issues one
// GemmMatBatched per fill chunk (runChunk), and a batch may be split across several chunks. This
// pins that the split is exact — every member issued once, into the same output — against a run
// that issues each batch whole.
//
// It also asserts the batching health directly: with the production budget every chunk carries its
// whole batch, so calls equals the number of non-empty batches. That is the property whose absence
// (members/call = 2.2 against ~32,567-member batches) cost the production system 452 s of dispatch per device
// per column chunk, job 14391094.
func TestRunChunkEqualsWholeBatches(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		mx := New(sp, ints, eps, be)
		defer mx.Release()
		p := mx.buildJIIBatchPlan()
		if len(p.slots) == 0 {
			return
		}

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
		rng := rand.New(rand.NewSource(int64(len(p.slots)) + 1))
		hostIn := make([]float64, n*cols)
		for i := range hostIn {
			hostIn[i] = rng.NormFloat64()
		}
		in := backend.BlockView{V: be.Upload(hostIn), Rows: n, Cols: cols, Ld: n}

		// run issues the given application order and chunking, handing runChunk the chunk-local
		// handle slice the device fill would have returned.
		run := func(apps []int32, chunks []jiiChunk) ([]float64, satStats) {
			q := p.cloneWith(apps, chunks)
			outBuf := be.Alloc(n * cols)
			be.Zero(outBuf)
			out := backend.BlockView{V: outBuf, Rows: n, Cols: cols, Ld: n}
			for _, ch := range chunks {
				local := make([]backend.DeviceMat, ch.hi-ch.lo)
				for e := ch.lo; e < ch.hi; e++ {
					local[e-ch.lo] = mats[apps[e]]
				}
				q.runChunk(be, local, in, out, ch, 0)
			}
			return be.Download(outBuf), q.stats
		}

		wholeApps, wholeChunks := p.appChunks(nil, JIIFillBudgetElems)
		whole, st := run(wholeApps, wholeChunks)

		nonEmpty := 0
		for _, bt := range p.batches {
			if len(bt.Blocks) > 0 {
				nonEmpty++
			}
		}
		if st.calls != int64(nonEmpty) {
			t.Errorf("spin=%v sym=%d: %d batched calls for %d non-empty batches — the production "+
				"budget must leave every batch in one chunk", spin, sym, st.calls, nonEmpty)
		}
		if st.members != int64(len(wholeApps)) {
			t.Errorf("spin=%v sym=%d: issued %d members, plan has %d applications",
				spin, sym, st.members, len(wholeApps))
		}

		// A budget of one element cuts after every block, so every batch is split maximally.
		tinyApps, tinyChunks := p.appChunks(nil, 1)
		split, _ := run(tinyApps, tinyChunks)

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
			t.Errorf("spin=%v sym=%d: split-batch apply differs from whole-batch by %.3e relative — "+
				"far above rounding, so a member was dropped or double-counted", spin, sym, maxRel)
		}
	})
}
