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

// TestComputeChunks pins the bounded-chunking partition that keeps the device fill from
// materializing the whole satellite operator at once (the 424 GB OOM, job 14026481). The
// invariants it guards: chunks tile the slot range contiguously with no gap or overlap, each
// chunk's element total matches the blocks it covers, every chunk except a lone oversized block
// stays within budget, and a block larger than the budget forms its own single-block chunk
// (a block cannot be split). These are exactly the properties the chunk-local BufOff and the
// pointer-offset fill depend on.
func TestComputeChunks(t *testing.T) {
	slot := func(rows, cols int) jiiSlot { return jiiSlot{rows: rows, cols: cols} }

	cases := []struct {
		name   string
		slots  []jiiSlot
		budget int
	}{
		{"empty", nil, 10},
		{"single", []jiiSlot{slot(3, 3)}, 100},
		{"one-oversized-block", []jiiSlot{slot(20, 20)}, 10}, // 400 > budget: its own chunk
		{"exact-fit", []jiiSlot{slot(2, 2), slot(2, 2), slot(2, 2)}, 4},
		{"splits", []jiiSlot{slot(3, 3), slot(3, 3), slot(3, 3), slot(3, 3)}, 20}, // 9 each
		{"oversized-midstream", []jiiSlot{slot(2, 2), slot(10, 10), slot(2, 2)}, 8},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			chunks := computeChunks(tc.slots, tc.budget)

			if len(tc.slots) == 0 {
				if chunks != nil {
					t.Fatalf("empty slots: got %d chunks, want none", len(chunks))
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
			if last := chunks[len(chunks)-1].hi; last != len(tc.slots) {
				t.Fatalf("last chunk ends at %d, want %d", last, len(tc.slots))
			}

			for i, ch := range chunks {
				if ch.hi <= ch.lo {
					t.Fatalf("chunk %d is empty: [%d,%d)", i, ch.lo, ch.hi)
				}
				sum := 0
				for _, s := range tc.slots[ch.lo:ch.hi] {
					sum += s.rows * s.cols
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
