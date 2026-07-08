package backend

import "testing"

// fakeMat is a DeviceMat with the given dims and no storage.
type fakeMat struct{ r, c int }

func (f fakeMat) Dims() (int, int) { return f.r, f.c }

// TestPlanBatchesDisjointWrites is the safety property the batched GEMM depends on:
// within a batch, no two blocks write to the same output offset. A batched GEMM runs
// its members concurrently and they accumulate, so a shared output would race and
// silently drop contributions.
func TestPlanBatchesDisjointWrites(t *testing.T) {
	// A block-structured operator with heavy write-offset collisions: several column
	// groups per row group, mixed shapes, and a diagonal block.
	var blocks []Block
	for gr := range 5 {
		for gc := 0; gc <= gr; gc++ {
			shape := fakeMat{4, 4}
			if gr%2 == 1 {
				shape = fakeMat{6, 4}
			}
			blocks = append(blocks, Block{A: shape, RowOff: gr * 10, ColOff: gc * 10, Diag: gr == gc})
		}
	}
	// Coupling blocks: many share a row offset.
	for gr := range 5 {
		for col := range 3 {
			blocks = append(blocks, Block{A: fakeMat{4, 1}, RowOff: gr * 10, ColOff: col, Diag: false})
		}
	}

	batches := PlanBatches(blocks)
	if len(batches) == 0 {
		t.Fatal("no batches produced")
	}

	seenN := map[int]bool{} // block index -> covered by an untransposed batch
	seenT := map[int]bool{}
	for bi, b := range batches {
		writes := map[int]int{}
		var r0, c0 int
		for i, idx := range b.Blocks {
			blk := blocks[idx]
			r, c := blk.A.Dims()
			if i == 0 {
				r0, c0 = r, c
			} else if r != r0 || c != c0 {
				t.Fatalf("batch %d mixes shapes: %dx%d and %dx%d", bi, r0, c0, r, c)
			}
			off := blk.RowOff
			if b.Trans {
				off = blk.ColOff
				if blk.Diag {
					t.Fatalf("batch %d: diagonal block %d appears in a transposed batch", bi, idx)
				}
				seenT[idx] = true
			} else {
				seenN[idx] = true
			}
			if prev, dup := writes[off]; dup {
				t.Errorf("batch %d (trans=%v): blocks %d and %d both write offset %d",
					bi, b.Trans, prev, idx, off)
			}
			writes[off] = idx
		}
	}

	// Every block must be applied exactly once untransposed, and off-diagonal blocks
	// exactly once transposed — the same coverage as the unbatched path.
	for i, blk := range blocks {
		if !seenN[i] {
			t.Errorf("block %d never applied untransposed", i)
		}
		if blk.Diag && seenT[i] {
			t.Errorf("diagonal block %d applied transposed", i)
		}
		if !blk.Diag && !seenT[i] {
			t.Errorf("off-diagonal block %d never applied transposed", i)
		}
	}
	t.Logf("%d blocks -> %d batches (unbatched would be %d calls)",
		len(blocks), len(batches), len(seenN)+len(seenT))
}

// TestPlanBatchesDeterministic: Go map iteration is randomized, so the plan must be
// sorted into a stable order or runs will differ in floating-point summation order.
func TestPlanBatchesDeterministic(t *testing.T) {
	var blocks []Block
	for gr := range 6 {
		for gc := 0; gc <= gr; gc++ {
			blocks = append(blocks, Block{A: fakeMat{3, 3}, RowOff: gr * 5, ColOff: gc * 5, Diag: gr == gc})
		}
	}
	first := PlanBatches(blocks)
	for range 20 {
		got := PlanBatches(blocks)
		if len(got) != len(first) {
			t.Fatalf("batch count varies: %d vs %d", len(got), len(first))
		}
		for i := range got {
			if got[i].Trans != first[i].Trans || len(got[i].Blocks) != len(first[i].Blocks) {
				t.Fatalf("batch %d differs across runs", i)
			}
			for j := range got[i].Blocks {
				if got[i].Blocks[j] != first[i].Blocks[j] {
					t.Fatalf("batch %d member %d differs across runs", i, j)
				}
			}
		}
	}
}
