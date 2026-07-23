package backend

import "testing"

// TestGemmMatBatchedBucketingMatchesLoop pins distBackend.GemmMatBatched's device bucketing
// against the one-call-per-block form it replaced.
//
// The optimization groups a batch by the device owning each output row band and issues one
// batched call per device. That is numerically free — gpuBackend.GemmMatBatched's contract
// already requires batch members to have pairwise non-overlapping outputs and uniform shapes,
// so members cannot interact and may be regrouped freely. What bucketing CAN get wrong is
// routing: dropping a block, applying one twice, sending it to the wrong sub-backend, or
// mis-slicing the local band. None of those show up as a crash — they show up as a quietly
// wrong operator apply, which is why this compares against the reference elementwise.
//
// The layout deliberately mixes both paths: three blocks whose input band is local to the
// output's device (batched) and one whose input lives on the other partition (still routed
// one at a time through gemmMatOne, by design — a gathered band is compacted to Ld=rows and
// would violate the batch's uniform-Ld requirement).
func TestGemmMatBatchedBucketingMatchesLoop(t *testing.T) {
	const (
		n    = 12 // n > 2·main² = 8, so distBackend's shape invariant holds
		main = 2
		bw   = 3 // block is bw×bw, and each output band is bw rows
		cols = 2 // panel width
		ndev = 2
		nblk = 4
	)
	bounds := []int{0, 6, n} // partition 0 owns rows [0,6), partition 1 owns [6,12)

	newBE := func() Backend {
		be, err := NewDistributed([]Backend{Gonum{}, Gonum{}}, n, main, bounds)
		if err != nil {
			t.Fatalf("NewDistributed: %v", err)
		}
		return be
	}

	// Operator blocks: distinct values per block so a swap or a drop cannot cancel out.
	blocks := make([]Mat, nblk)
	for i := range nblk {
		m := NewMat(bw, bw)
		for r := range bw {
			for c := range bw {
				m.Set(r, c, float64(10*(i+1)+3*r+c)*0.25)
			}
		}
		blocks[i] = m
	}

	// (output row offset, input row offset). Outputs are pairwise disjoint and every band lies
	// wholly inside one partition (bounds are group-aligned), as RowRange requires.
	//   0,1 -> device 0, local input      2 -> device 1, local input
	//   3   -> device 1 output, device 0 input: the remote-input path
	offs := [nblk][2]int{{0, 0}, {3, 3}, {6, 6}, {9, 0}}

	input := make([]float64, n*cols)
	for i := range input {
		input[i] = float64(i%7) - 3.5 // mixed signs; nothing cancels by symmetry
	}

	// run applies the four blocks via f, returning the resulting output panel.
	run := func(be Backend, f func(be Backend, a []DeviceMat, in, out []BlockView)) []float64 {
		inV := be.Upload(input)
		outV := be.Alloc(n * cols)
		be.Zero(outV)
		inBlk := BlockView{V: inV, Rows: n, Cols: cols, Ld: n}
		outBlk := BlockView{V: outV, Rows: n, Cols: cols, Ld: n}

		a := make([]DeviceMat, nblk)
		ib := make([]BlockView, nblk)
		ob := make([]BlockView, nblk)
		for i := range nblk {
			a[i] = be.UploadMat(blocks[i])
			ib[i] = inBlk.RowRange(offs[i][1], bw)
			ob[i] = outBlk.RowRange(offs[i][0], bw)
		}
		f(be, a, ib, ob)
		return be.Download(outV)
	}

	got := run(newBE(), func(be Backend, a []DeviceMat, in, out []BlockView) {
		be.GemmMatBatched(false, 1, a, in, 1, out)
	})
	want := run(newBE(), func(be Backend, a []DeviceMat, in, out []BlockView) {
		for i := range a { // the pre-bucketing form: one call per block
			be.GemmMat(false, 1, a[i], in[i], 1, out[i])
		}
	})

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	// Bit-exact: the same blocks, the same operands, only the grouping differs.
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d (row %d, col %d): batched %g != per-block %g",
				i, i%n, i/n, got[i], want[i])
		}
	}

	// Guard against the degenerate pass where both paths compute nothing.
	nz := 0
	for _, v := range want {
		if v != 0 {
			nz++
		}
	}
	if nz == 0 {
		t.Fatal("reference output is entirely zero — the test would pass vacuously")
	}
}
