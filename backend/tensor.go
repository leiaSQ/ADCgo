package backend

import (
	"cmp"
	"fmt"
	"slices"
)

// tensor.go — the device capability behind the generated σ-build (internal/adc/isrgen/sigma):
// dense tensors addressed by mode labels, contracted by batched GEMM between label-driven
// elementwise kernels, and moved in and out of the packed configuration vector by signed
// index maps. The host implementation lives in the sigma package; a device backend that
// implements TensorKernels runs the same program on its own memory.

// TensorOperand is a dense tensor on a backend, column-major (the first mode is fastest).
// Labels name its modes; a label repeated within one operand is a diagonal.
type TensorOperand struct {
	V      Vector
	Labels []uint8
	Dims   []int
}

// TensorMap is a signed index map uploaded to a backend (UploadTensorMap).
type TensorMap interface {
	Len() int
}

// TensorKernels is the optional device capability of the σ engine.
type TensorKernels interface {
	// TensorEinsum accumulates out += alpha · Σ a·b over every label assignment: a label
	// absent from out is summed, a label absent from an input is broadcast over. b may
	// be nil (a reduction, diagonal extraction, permutation or broadcast of a).
	TensorEinsum(alpha float64, out, a TensorOperand, b *TensorOperand) error
	// UploadTensorMap makes a map resident: entry i links element elem[i] of a tensor to
	// row row[i] of a packed vector with sign sign[i].
	UploadTensorMap(elem []int64, row []int32, sign []int8) TensorMap
	FreeTensorMap(m TensorMap)
	// TensorUnpack sets t[elem[i] + c·tsize] = sign[i]·y[row[i] + c·ld] for c < cols.
	TensorUnpack(t Vector, tsize int, y Vector, ld, cols int, m TensorMap)
	// TensorPack accumulates out[row[i] + c·ld] += sign[i]·t[elem[i] + c·tsize]. The rows of
	// the map must be distinct.
	TensorPack(out Vector, ld, cols int, t Vector, tsize int, m TensorMap)
	// GemmStridedBatched sets c_t = op(a_t)·op(b_t) for t < nb over the consecutive
	// m×k, k×n and m×n column-major matrices of a, b and c (lda, ldb as stored, ldc = m).
	GemmStridedBatched(transA, transB bool, m, n, k, nb int, a Vector, lda int, b Vector, ldb int, c Vector)
}

// TensorMaxLabels bounds the distinct labels of one TensorEinsum (the device kernel's
// fixed-size plan).
const TensorMaxLabels = 16

// TensorPlan is the loop nest of one TensorEinsum: the distinct labels, their extents and
// every operand's stride per label (the sum over its modes carrying the label, which is
// what makes a repeated label a diagonal). Operand 0 is out, 1 is a, 2 is b.
type TensorPlan struct {
	Dims    []int
	Strides [][]int // [operand][label]
}

// PlanTensor builds the plan of out += Σ a·b (b optional) and checks the operands'
// extents and storage.
func PlanTensor(ops ...TensorOperand) (TensorPlan, error) {
	idx := map[uint8]int{}
	var p TensorPlan
	for _, v := range ops {
		if len(v.Labels) != len(v.Dims) {
			return p, fmt.Errorf("backend: tensor with %d labels for %d modes", len(v.Labels), len(v.Dims))
		}
		for m, l := range v.Labels {
			k, ok := idx[l]
			if !ok {
				k = len(p.Dims)
				idx[l] = k
				p.Dims = append(p.Dims, v.Dims[m])
			} else if p.Dims[k] != v.Dims[m] {
				return p, fmt.Errorf("backend: tensor label %d has extents %d and %d", l, p.Dims[k], v.Dims[m])
			}
		}
		n := 1
		for _, d := range v.Dims {
			n *= d
		}
		if v.V != nil && v.V.Len() < n {
			return p, fmt.Errorf("backend: tensor storage holds %d values for dims %v", v.V.Len(), v.Dims)
		}
	}
	if len(p.Dims) > TensorMaxLabels {
		return p, fmt.Errorf("backend: %d distinct labels, at most %d", len(p.Dims), TensorMaxLabels)
	}
	p.Strides = make([][]int, len(ops))
	for o, v := range ops {
		st := make([]int, len(p.Dims))
		s := 1
		for m, l := range v.Labels {
			st[idx[l]] += s
			s *= v.Dims[m]
		}
		p.Strides[o] = st
	}
	return p, nil
}

// LoopOrder is the label order a kernel should nest its loops in, innermost first: the
// label that walks out (then a) most contiguously first, labels out does not carry last.
func (p TensorPlan) LoopOrder() []int {
	order := make([]int, len(p.Dims))
	for i := range order {
		order[i] = i
	}
	key := func(l int) (int, int) {
		so := p.Strides[0][l]
		if so == 0 {
			so = int(^uint(0) >> 1)
		}
		return so, p.Strides[1][l]
	}
	slices.SortStableFunc(order, func(x, y int) int {
		ox, ax := key(x)
		oy, ay := key(y)
		if ox != oy {
			return cmp.Compare(ox, oy)
		}
		return cmp.Compare(ax, ay)
	})
	return order
}
