package sigma

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
)

// partitioned.go — the σ-build on a row-partitioned multi-device backend (-mgpu).
//
// The contractions need whole spin-block tensors, so the work is split by panel COLUMN,
// not by row: partition d owns columns [lo_d, hi_d) of every apply. It gathers those
// columns full height from every partition's row band, applies its own engine (every
// partition holds the integrals and intermediates), and scatters the result's row bands
// back to their owners. Each apply moves 2·n·b values between devices against o⁴v²-type
// arithmetic per column, and the partitions never write the same element.

type partitioned struct {
	pd     backend.PartitionedDevices
	ops    []*Operator // one per partition, on its sub-backend
	bounds []int
	n      int
}

func newPartitioned(prog *Program, sp *khci.Space, ints *integrals.Store, eps []float64, nocc int,
	maxOrder [6]int, be backend.Backend, pd backend.PartitionedDevices) (*Operator, error) {

	nd := pd.NumParts()
	ops := make([]*Operator, nd)
	errs := make([]error, nd)
	backend.GoDevices(nd, func(d int) {
		ops[d], errs[d] = newOn(prog, sp, ints, eps, nocc, maxOrder, pd.PartBackend(d))
	})
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	bounds := pd.Bounds()
	if bounds[len(bounds)-1] != sp.Size() {
		return nil, fmt.Errorf("sigma: the partition bounds end at %d, the space has %d rows",
			bounds[len(bounds)-1], sp.Size())
	}
	return &Operator{sp: sp, main: sp.MainBlockSize(), MaxPanelBytes: ops[0].MaxPanelBytes,
		split: &partitioned{pd: pd, ops: ops, bounds: bounds, n: sp.Size()}}, nil
}

// columns is partition d's share [lo, hi) of cols panel columns.
func (p *partitioned) columns(d, cols int) (lo, hi int) {
	nd := len(p.ops)
	return d * cols / nd, (d + 1) * cols / nd
}

func (p *partitioned) apply(out, in backend.BlockView, satOnly bool) error {
	nd, n := len(p.ops), p.n
	errs := make([]error, nd)
	backend.SyncParts(p.pd) // the producers of in may still be writing
	backend.GoDevices(nd, func(d int) {
		lo, hi := p.columns(d, in.Cols)
		cw := hi - lo
		if cw == 0 {
			return
		}
		sub := p.pd.PartBackend(d)
		slab, res := sub.Alloc(n*cw), sub.Alloc(n*cw)
		defer sub.Free(slab)
		defer sub.Free(res)
		for src := range nd {
			rows := p.bounds[src+1] - p.bounds[src]
			backend.CopyBand(sub, slab.Slice(p.bounds[src], n*cw-p.bounds[src]), n,
				p.pd.PartBackend(src), p.pd.PartVector(in.V, src).Slice(lo*rows, cw*rows), rows, rows, cw)
		}
		if pc, ok := sub.(backend.PeerCopier); ok {
			pc.Sync()
		}
		errs[d] = p.ops[d].apply(backend.BlockView{V: res, Rows: n, Cols: cw, Ld: n},
			backend.BlockView{V: slab, Rows: n, Cols: cw, Ld: n}, satOnly)
		if errs[d] != nil {
			return
		}
		if pc, ok := sub.(backend.PeerCopier); ok {
			pc.Sync()
		}
		for dst := range nd {
			rows := p.bounds[dst+1] - p.bounds[dst]
			to := p.pd.PartBackend(dst)
			backend.CopyBand(to, p.pd.PartVector(out.V, dst).Slice(lo*rows, cw*rows), rows,
				sub, res.Slice(p.bounds[dst], n*cw-p.bounds[dst]), n, rows, cw)
		}
	})
	backend.SyncParts(p.pd) // fence the scattered bands before the caller reads them
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}
