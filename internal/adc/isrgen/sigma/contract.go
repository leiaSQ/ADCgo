package sigma

import (
	"fmt"
	"slices"
)

// colLabel is the label of the panel-column mode the engine appends to every tensor that
// depends on the amplitude vector. Generated programs never use it.
const colLabel uint8 = 255

func has(ls []uint8, l uint8) bool { return slices.Contains(ls, l) }

// alloc returns a zero tensor over labels with the given extents.
func (e *engine) alloc(labels []uint8, dims []int) tens {
	n := max(size(dims), 1)
	return tens{v: e.be.Alloc(n), labels: labels, dims: dims}
}

func (e *engine) free(t tens) { e.be.Free(t.v) }

// dimsOf reads the extents of labels off the operands carrying them.
func dimsOf(labels []uint8, ops ...tens) ([]int, error) {
	out := make([]int, len(labels))
	for m, l := range labels {
		found := false
		for _, t := range ops {
			for k, tl := range t.labels {
				if tl == l {
					out[m] = t.dims[k]
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("sigma: label %d carried by no operand", l)
		}
	}
	return out, nil
}

// reduceTo returns t restricted to the distinct labels in keep (first-occurrence order):
// diagonals taken for repeated labels, labels outside keep summed. owned reports whether
// the result is a new tensor the caller must free.
func (e *engine) reduceTo(t tens, keep func(uint8) bool) (tens, bool, error) {
	var ls []uint8
	for _, l := range t.labels {
		if keep(l) && !has(ls, l) {
			ls = append(ls, l)
		}
	}
	if len(ls) == len(t.labels) {
		return t, false, nil
	}
	dims, err := dimsOf(ls, t)
	if err != nil {
		return t, false, err
	}
	r := e.alloc(ls, dims)
	if err := e.k.einsum(1, r, t, nil); err != nil {
		e.free(r)
		return t, false, err
	}
	return r, true, nil
}

// permuted returns t with its (distinct) labels in the order want, copying only when the
// order differs.
func (e *engine) permuted(t tens, want []uint8) (tens, bool, error) {
	if slices.Equal(t.labels, want) {
		return t, false, nil
	}
	dims, err := dimsOf(want, t)
	if err != nil {
		return t, false, err
	}
	r := e.alloc(want, dims)
	if err := e.k.einsum(1, r, t, nil); err != nil {
		e.free(r)
		return t, false, err
	}
	return r, true, nil
}

func prod(dims []int, labels, pick []uint8) int {
	n := 1
	for m, l := range labels {
		if has(pick, l) {
			n *= dims[m]
		}
	}
	return n
}

// contract returns a·b over the label set out (a new tensor): labels shared by a and b
// and absent from out are contracted by GEMM (one per batch label assignment), labels in
// one operand only and absent from out are summed first. The result's mode order is the
// GEMM's natural [free-a free-b batch] (the labels say which is which), so no step pays
// a permutation of its output; the order within out does not matter to any consumer.
func (e *engine) contract(a, b tens, out []uint8) (tens, error) {
	if len(slices.Compact(slices.Sorted(slices.Values(out)))) != len(out) {
		return tens{}, fmt.Errorf("sigma: step output labels %v repeat", out)
	}
	ra, ownA, err := e.reduceTo(a, func(l uint8) bool { return has(b.labels, l) || has(out, l) })
	if err != nil {
		return tens{}, err
	}
	if ownA {
		defer e.free(ra)
	}
	rb, ownB, err := e.reduceTo(b, func(l uint8) bool { return has(ra.labels, l) || has(out, l) })
	if err != nil {
		return tens{}, err
	}
	if ownB {
		defer e.free(rb)
	}
	var batch, contr, freeA, freeB []uint8
	for _, l := range ra.labels {
		switch {
		case has(rb.labels, l) && has(out, l):
			batch = append(batch, l)
		case has(rb.labels, l):
			contr = append(contr, l)
		default:
			freeA = append(freeA, l)
		}
	}
	for _, l := range rb.labels {
		if !has(ra.labels, l) {
			freeB = append(freeB, l)
		}
	}
	for _, l := range out {
		if !has(ra.labels, l) && !has(rb.labels, l) {
			return tens{}, fmt.Errorf("sigma: output label %d in neither operand", l)
		}
	}
	if len(contr) == 0 {
		dims, err := dimsOf(out, ra, rb)
		if err != nil {
			return tens{}, err
		}
		res := e.alloc(out, dims)
		// a product with no summation (Hadamard, outer or batched-outer): the label
		// kernel is the whole operation
		if err := e.k.einsum(1, res, ra, &rb); err != nil {
			e.free(res)
			return tens{}, err
		}
		return res, nil
	}
	m := prod(ra.dims, ra.labels, freeA)
	k := prod(ra.dims, ra.labels, contr)
	n := prod(rb.dims, rb.labels, freeB)
	nb := prod(ra.dims, ra.labels, batch)

	// A as [freeA contr batch] (M×K per batch), or [contr freeA batch] read transposed
	transA := false
	lA := slices.Concat(freeA, contr, batch)
	if lt := slices.Concat(contr, freeA, batch); slices.Equal(ra.labels, lt) && len(freeA) > 0 {
		lA, transA = lt, true
	}
	pa, ownPA, err := e.permuted(ra, lA)
	if err != nil {
		return tens{}, err
	}
	if ownPA {
		defer e.free(pa)
	}
	transB := false
	lB := slices.Concat(contr, freeB, batch)
	if lt := slices.Concat(freeB, contr, batch); slices.Equal(rb.labels, lt) && len(freeB) > 0 {
		lB, transB = lt, true
	}
	pb, ownPB, err := e.permuted(rb, lB)
	if err != nil {
		return tens{}, err
	}
	if ownPB {
		defer e.free(pb)
	}
	lC := slices.Concat(freeA, freeB, batch)
	cd, err := dimsOf(lC, ra, rb)
	if err != nil {
		return tens{}, err
	}
	c := e.alloc(lC, cd)
	lda, ldb := m, k
	if transA {
		lda = k
	}
	if transB {
		ldb = n
	}
	e.k.gemmBatched(transA, transB, m, n, k, nb, pa.v, lda, pb.v, ldb, c.v)
	return c, nil
}
