package sigma

import (
	"fmt"
	"slices"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// view is a dense host tensor laid out column-major: the first mode is fastest, mode m
// has stride Π_{k<m} dims[k]. labels name the modes (a label repeated in one view is a
// diagonal).
type view struct {
	data   []float64
	labels []uint8
	dims   []int
}

func size(dims []int) int {
	n := 1
	for _, d := range dims {
		n *= d
	}
	return n
}

// einsumHost accumulates out += alpha · Σ a·b over every label assignment, where a label
// absent from out is summed and a label absent from an input is broadcast over it. b may
// be nil (a single-operand reduction, diagonal extraction, permutation or broadcast).
// Workers split the out label with the largest extent; distinct assignments of the out
// labels address distinct out elements, so no two workers write the same element.
func einsumHost(alpha float64, out view, a view, b *view) error {
	ops := []view{out, a}
	if b != nil {
		ops = append(ops, *b)
	}
	tops := make([]backend.TensorOperand, len(ops))
	for o, v := range ops {
		if len(v.data) < size(v.dims) {
			return fmt.Errorf("sigma: operand %d holds %d values for dims %v", o, len(v.data), v.dims)
		}
		tops[o] = backend.TensorOperand{Labels: v.labels, Dims: v.dims}
	}
	p, err := backend.PlanTensor(tops...)
	if err != nil {
		return err
	}
	if slices.Contains(p.Dims, 0) {
		return nil
	}
	order := p.LoopOrder()
	// the parallel label: the out label with the largest extent
	par := -1
	for l := range p.Dims {
		if p.Strides[0][l] != 0 && (par < 0 || p.Dims[l] > p.Dims[par]) {
			par = l
		}
	}
	if par < 0 { // a scalar out: one worker owns it
		loopLabels(alpha, ops, p, order, -1, 0)
		return nil
	}
	parallel.HeavyRows(p.Dims[par], func(x int) { loopLabels(alpha, ops, p, order, par, x) })
	return nil
}

// loopLabels runs the odometer over every label except fix, which is held at x (fix < 0:
// no label is fixed), nesting the loops in order (innermost first).
func loopLabels(alpha float64, ops []view, p backend.TensorPlan, order []int, fix, x int) {
	free := make([]int, 0, len(order))
	for _, l := range order {
		if l != fix {
			free = append(free, l)
		}
	}
	off := make([]int, len(ops))
	if fix >= 0 {
		for o := range ops {
			off[o] = x * p.Strides[o][fix]
		}
	}
	outD, aD := ops[0].data, ops[1].data
	two := len(ops) == 3
	var bD []float64
	if two {
		bD = ops[2].data
	}
	if len(free) == 0 {
		v := aD[off[1]]
		if two {
			v *= bD[off[2]]
		}
		outD[off[0]] += alpha * v
		return
	}
	cnt := make([]int, len(free))
	in := free[0]
	nIn := p.Dims[in]
	so, sa := p.Strides[0][in], p.Strides[1][in]
	var sb int
	if two {
		sb = p.Strides[2][in]
	}
	for {
		o, ao := off[0], off[1]
		if two {
			bo := off[2]
			for range nIn {
				outD[o] += alpha * aD[ao] * bD[bo]
				o += so
				ao += sa
				bo += sb
			}
		} else {
			for range nIn {
				outD[o] += alpha * aD[ao]
				o += so
				ao += sa
			}
		}
		// advance the outer labels
		k := 1
		for ; k < len(free); k++ {
			l := free[k]
			cnt[k]++
			for op := range ops {
				off[op] += p.Strides[op][l]
			}
			if cnt[k] < p.Dims[l] {
				break
			}
			for op := range ops {
				off[op] -= cnt[k] * p.Strides[op][l]
			}
			cnt[k] = 0
		}
		if k == len(free) {
			return
		}
	}
}
