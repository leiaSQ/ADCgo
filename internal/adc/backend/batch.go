package backend

// Batched application of a block-sparse symmetric operator.
//
// The DIP/SIP secular matrices are stored as thousands of small dense blocks (27,725
// for formic acid's largest sector). Applying them one block at a time issues one BLAS
// call per block per Lanczos block-iteration — 11.1 M cuBLAS calls for one formic-acid
// sector, at ~16 µs of launch overhead each. That is the same failure mode as running
// GEMM-shaped work as level-1 calls, one level down: the arithmetic is trivial next to
// the dispatch.
//
// The blocks fall into few distinct shapes (17 for that sector), so most of them can be
// issued as one batched GEMM. The only hazard is the accumulation: blocks sharing a row
// band write the same rows of the output, and a batched GEMM runs its members
// concurrently. PlanBatches therefore guarantees that within a batch no two blocks
// write to the same offset — batches are formed per shape, taking at most one block per
// distinct write offset. For formic acid this collapses 55,097 calls into 1,002.

// Block is one resident operator block placed with its top-left corner at
// (RowOff, ColOff). A Diag block lies on the block diagonal and is applied once; an
// off-diagonal block is applied twice (once as A, once as Aᵀ) to realize the symmetric
// operator, exactly as the unbatched path does.
type Block struct {
	A              DeviceMat
	RowOff, ColOff int
	Diag           bool
}

// Batch is a set of same-shaped blocks that may be applied in one batched GEMM.
// Trans selects Aᵀ (the transposed half of the symmetric operator). Every block in the
// batch writes to a distinct output offset, so their accumulations cannot race.
type Batch struct {
	Trans  bool
	Blocks []int // indices into the Block slice PlanBatches was given
}

// PlanBatches groups blocks into batched-GEMM calls. The plan depends only on the
// operator's structure, so it is computed once when the operator is assembled.
//
// Output offset is RowOff for the untransposed application and ColOff for the
// transposed one. Blocks are bucketed by (shape, trans); within a bucket they are
// grouped by write offset, and batch j takes the j-th block from every offset group.
// The number of batches per bucket is therefore the largest number of blocks sharing
// any single write offset — which is what makes the batches as wide as possible while
// keeping every write disjoint.
func PlanBatches(blocks []Block) []Batch {
	type key struct {
		rows, cols int
		trans      bool
	}
	// bucket → write offset → block indices sharing that offset
	buckets := map[key]map[int][]int{}
	order := []key{} // deterministic emission order

	add := func(k key, off, idx int) {
		byOff, ok := buckets[k]
		if !ok {
			byOff = map[int][]int{}
			buckets[k] = byOff
			order = append(order, k)
		}
		byOff[off] = append(byOff[off], idx)
	}

	for i, b := range blocks {
		rows, cols := b.A.Dims()
		add(key{rows, cols, false}, b.RowOff, i)
		if !b.Diag {
			add(key{rows, cols, true}, b.ColOff, i)
		}
	}

	var out []Batch
	for _, k := range order {
		byOff := buckets[k]
		// Widest offset group decides how many batches this bucket needs.
		depth := 0
		for _, idxs := range byOff {
			if len(idxs) > depth {
				depth = len(idxs)
			}
		}
		// Collect offsets in ascending order so the plan is reproducible across runs
		// (Go map iteration is randomized).
		offs := make([]int, 0, len(byOff))
		for off := range byOff {
			offs = append(offs, off)
		}
		sortInts(offs)

		for j := range depth {
			var members []int
			for _, off := range offs {
				if idxs := byOff[off]; j < len(idxs) {
					members = append(members, idxs[j])
				}
			}
			out = append(out, Batch{Trans: k.trans, Blocks: members})
		}
	}
	return out
}

// sortInts is a small insertion sort; the offset lists are short (hundreds) and this
// avoids pulling in sort for one call.
func sortInts(a []int) {
	for i := 1; i < len(a); i++ {
		v := a[i]
		j := i - 1
		for j >= 0 && a[j] > v {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = v
	}
}
