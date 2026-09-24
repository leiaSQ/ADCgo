package sigma

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// tens is a resident tensor: backend storage plus the view metadata.
type tens struct {
	v      backend.Vector
	labels []uint8
	dims   []int
}

// kernels is what the engine needs beyond the backend's BLAS: the label-driven
// elementwise kernel, batched GEMM, and the signed gather/scatter between the packed
// khci vector and spin-block tensors. hostKernels runs them on host memory,
// deviceKernels on a backend implementing backend.TensorKernels.
type kernels interface {
	// einsum accumulates out += alpha · Σ a·b over the labels (einsumHost's contract).
	einsum(alpha float64, out, a tens, b *tens) error
	// unpack sets t[m.elem[i] + c·tsize] = m.sign[i] · y[m.row[i] + c·ld] for every
	// entry and column c < cols. t must be zero on entry wherever no entry writes.
	unpack(t backend.Vector, tsize int, y backend.Vector, ld, cols int, m *packMap)
	// pack accumulates out[m.row[i] + c·ld] += m.sign[i] · t[m.elem[i] + c·tsize].
	pack(out backend.Vector, ld, cols int, t backend.Vector, tsize int, m *packMap)
	// gemmBatched sets c_t = op(a_t)·op(b_t) for t < nb, where a_t, b_t, c_t are the
	// consecutive m×k, k×n and m×n column-major matrices of a, b and c (lda, ldb their
	// leading dimensions as stored, m for c).
	gemmBatched(transA, transB bool, m, n, k, nb int, a backend.Vector, lda int,
		b backend.Vector, ldb int, c backend.Vector)
	// release frees what the kernels made resident for a map.
	release(m *packMap)
}

// hostKernels runs the kernels on a host backend.
type hostKernels struct {
	h  backend.HostData
	be backend.Backend
}

func (k hostKernels) view(t tens) view {
	return view{data: k.h.HostSlice(t.v), labels: t.labels, dims: t.dims}
}

func (k hostKernels) einsum(alpha float64, out, a tens, b *tens) error {
	var bv *view
	if b != nil {
		v := k.view(*b)
		bv = &v
	}
	return einsumHost(alpha, k.view(out), k.view(a), bv)
}

// packChunk is the number of map entries one worker handles per task.
const packChunk = 1 << 14

func (k hostKernels) unpack(t backend.Vector, tsize int, y backend.Vector, ld, cols int, m *packMap) {
	td, yd := k.h.HostSlice(t), k.h.HostSlice(y)
	nch := (len(m.elem) + packChunk - 1) / packChunk
	// distinct entries write distinct tensor elements (the map is injective per column)
	parallel.HeavyRows(nch*cols, func(task int) {
		c, ch := task/nch, task%nch
		to, yo := c*tsize, c*ld
		lo, hi := ch*packChunk, min((ch+1)*packChunk, len(m.elem))
		for i := lo; i < hi; i++ {
			td[to+int(m.elem[i])] = float64(m.sign[i]) * yd[yo+int(m.row[i])]
		}
	})
}

func (k hostKernels) pack(out backend.Vector, ld, cols int, t backend.Vector, tsize int, m *packMap) {
	od, td := k.h.HostSlice(out), k.h.HostSlice(t)
	nch := (len(m.elem) + packChunk - 1) / packChunk
	// a σ map holds each row once, so chunks write disjoint rows
	parallel.HeavyRows(nch*cols, func(task int) {
		c, ch := task/nch, task%nch
		oo, to := c*ld, c*tsize
		lo, hi := ch*packChunk, min((ch+1)*packChunk, len(m.elem))
		for i := lo; i < hi; i++ {
			od[oo+int(m.row[i])] += float64(m.sign[i]) * td[to+int(m.elem[i])]
		}
	})
}

func (k hostKernels) gemmBatched(transA, transB bool, m, n, kk, nb int, a backend.Vector, lda int,
	b backend.Vector, ldb int, c backend.Vector) {
	be := k.be
	one := func(t int) {
		av := backend.BlockView{V: a.Slice(t*m*kk, m*kk), Rows: m, Cols: kk, Ld: lda}
		if transA {
			av = backend.BlockView{V: a.Slice(t*m*kk, m*kk), Rows: kk, Cols: m, Ld: lda}
		}
		bv := backend.BlockView{V: b.Slice(t*kk*n, kk*n), Rows: kk, Cols: n, Ld: ldb}
		if transB {
			bv = backend.BlockView{V: b.Slice(t*kk*n, kk*n), Rows: n, Cols: kk, Ld: ldb}
		}
		cv := backend.BlockView{V: c.Slice(t*m*n, m*n), Rows: m, Cols: n, Ld: m}
		be.Gemm(transA, transB, 1, av, bv, 0, cv)
	}
	if nb == 1 {
		one(0)
		return
	}
	// batch members write disjoint c_t
	parallel.HeavyRows(nb, one)
}

func (hostKernels) release(*packMap) {}

// deviceKernels runs the kernels through a backend's TensorKernels.
type deviceKernels struct{ tk backend.TensorKernels }

func operand(t tens) backend.TensorOperand {
	return backend.TensorOperand{V: t.v, Labels: t.labels, Dims: t.dims}
}

func (k deviceKernels) einsum(alpha float64, out, a tens, b *tens) error {
	var bo *backend.TensorOperand
	if b != nil {
		x := operand(*b)
		bo = &x
	}
	return k.tk.TensorEinsum(alpha, operand(out), operand(a), bo)
}

// resident uploads a map on first use (one engine drives one backend).
func (k deviceKernels) resident(m *packMap) backend.TensorMap {
	if m.dev == nil {
		m.dev = k.tk.UploadTensorMap(m.elem, m.row, m.sign)
	}
	return m.dev
}

func (k deviceKernels) unpack(t backend.Vector, tsize int, y backend.Vector, ld, cols int, m *packMap) {
	if len(m.elem) == 0 {
		return
	}
	k.tk.TensorUnpack(t, tsize, y, ld, cols, k.resident(m))
}

func (k deviceKernels) pack(out backend.Vector, ld, cols int, t backend.Vector, tsize int, m *packMap) {
	if len(m.elem) == 0 {
		return
	}
	k.tk.TensorPack(out, ld, cols, t, tsize, k.resident(m))
}

func (k deviceKernels) gemmBatched(transA, transB bool, m, n, kk, nb int, a backend.Vector, lda int,
	b backend.Vector, ldb int, c backend.Vector) {
	k.tk.GemmStridedBatched(transA, transB, m, n, kk, nb, a, lda, b, ldb, c)
}

func (k deviceKernels) release(m *packMap) {
	if m.dev != nil {
		k.tk.FreeTensorMap(m.dev)
		m.dev = nil
	}
}

// hostResident reports whether be's vectors live in host memory: a device backend can
// satisfy backend.HostData through an embedded host backend, which only a probe tells.
func hostResident(be backend.Backend) (h backend.HostData, ok bool) {
	h, ok = be.(backend.HostData)
	if !ok {
		return nil, false
	}
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	v := be.Alloc(1)
	defer be.Free(v)
	_ = h.HostSlice(v)
	return h, true
}

// newKernels picks the kernels for be: its TensorKernels, else host memory. A backend
// with neither returns staged = true — the operator then runs on the host and moves each
// panel across (Operator.stage).
func newKernels(be backend.Backend) (k kernels, staged bool, err error) {
	if _, ok := be.(backend.PartitionedDevices); ok {
		return nil, false, fmt.Errorf("sigma: row-partitioned multi-device backends are not supported")
	}
	if _, ok := be.(backend.PanelScatterAdd); ok {
		return nil, false, fmt.Errorf("sigma: row-partitioned multi-device backends are not supported")
	}
	if tk, ok := be.(backend.TensorKernels); ok {
		return deviceKernels{tk}, false, nil
	}
	if h, ok := hostResident(be); ok {
		return hostKernels{h: h, be: be}, false, nil
	}
	return nil, true, nil
}
