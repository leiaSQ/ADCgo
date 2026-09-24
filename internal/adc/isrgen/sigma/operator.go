package sigma

import (
	"fmt"
	"math"
	"math/bits"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// engine holds the backend, the tensors built once (Coulomb slices, intermediates,
// energy factors, amplitude-free partial contractions) and the pack maps.
type engine struct {
	be   backend.Backend
	k    kernels
	prog *Program
	ints *integrals.Store
	eps  []float64 // spatial orbital energies
	nocc int       // occupied spatial orbitals
	nvir int       // virtual spatial orbitals

	eri   map[string]tens // by the four label spaces, e.g. "ovov"
	itmd  map[string]tens // by name + "|" + spin
	inBld map[string]bool // intermediates under construction (cycle guard)
}

// cterm is a Term bound to its resident operands.
type cterm struct {
	t      *Term
	leaves []tens // resolved non-amplitude leaves (zero value for Amp leaves)
	amp    []int  // amp[i] is the amplitude block index of leaf i, or -1
	cache  []tens // results of amplitude-free steps, computed once
	static []bool // per operand (leaves, then steps): free of the amplitude
}

// block is a target spin block: its pack map, extents and compiled terms.
type block struct {
	tensor *Tensor
	dims   []int
	labels []uint8 // 0..n-1: the target modes
	pm     *packMap
	terms  []*cterm
	sat    []*cterm // the terms of blocks free of class 0 (ApplyBlockSatellite)
}

type ampKey struct {
	class int
	spin  string
}

// amp is an amplitude spin block read from the vector.
type amp struct {
	key  ampKey
	dims []int
	pm   *packMap
}

// Operator applies a generated secular matrix, truncated to one scheme, on a khci.Space.
// It satisfies lanczos.Operator, lanczos.SatelliteOperator and lanczos.PreconOperator.
type Operator struct {
	e      *engine
	sp     *khci.Space
	main   int
	blocks []*block
	amps   []*amp
	diag   []*block

	// MaxPanelBytes bounds the amplitude and σ spin-block tensors of one column chunk
	// (default 4 GiB); ApplyBlock splits its columns to stay under it.
	MaxPanelBytes int64

	proj *khci.SpinProjector // non-nil: every apply is P·M·P (SetSpin)

	maxOrder [6]int // the scheme's per-block maximum orders

	// stage is the caller's backend when it offers neither backend.TensorKernels nor
	// host memory: the engine then runs on the host and each apply downloads its input
	// panel and uploads its result.
	stage backend.Backend

	// split drives one operator per partition of a row-partitioned multi-device backend
	// (partitioned.go); the fields above are then unused.
	split *partitioned
}

// Staged reports whether applies move their panels to the host and back (a backend
// without tensor kernels).
func (op *Operator) Staged() bool {
	if op.split != nil {
		return op.split.ops[0].Staged()
	}
	return op.stage != nil
}

// Release frees every resident tensor and map. The operator must not be used afterwards.
func (op *Operator) Release() {
	if op.split != nil {
		for _, sub := range op.split.ops {
			sub.Release()
		}
		return
	}
	e := op.e
	seen := map[*packMap]bool{}
	for _, b := range append(append([]*block(nil), op.blocks...), op.diag...) {
		for _, ct := range b.terms {
			e.release(ct)
		}
		if !seen[b.pm] {
			e.k.release(b.pm)
			seen[b.pm] = true
		}
	}
	for _, a := range op.amps {
		e.k.release(a.pm)
	}
	for _, t := range e.eri {
		e.be.Free(t.v)
	}
	for _, t := range e.itmd {
		e.be.Free(t.v)
	}
	for _, b := range op.blocks {
		for _, ct := range b.terms {
			for i, l := range ct.t.Leaves {
				if l.Kind == Energy {
					e.be.Free(ct.leaves[i].v)
				}
			}
		}
	}
	op.blocks, op.diag, op.amps = nil, nil, nil
	e.eri, e.itmd = map[string]tens{}, map[string]tens{}
}

// SetSpin restricts the operator to total spin twoS/2: every apply becomes P·M·P with P
// the Löwdin projector over the space. M commutes with S², so on the range of P this is
// M itself; what P removes is the roundoff that a Krylov method would otherwise grow into
// states of other multiplicities (a spin-summed Lanczos run from pure-spin start vectors
// finds them within a few hundred vectors). twoS < 0 lifts the restriction.
func (op *Operator) SetSpin(twoS int) error {
	if op.split != nil {
		for _, sub := range op.split.ops {
			if err := sub.SetSpin(twoS); err != nil {
				return err
			}
		}
		return nil
	}
	if twoS < 0 {
		op.proj = nil
		return nil
	}
	p, err := op.sp.SpinProjector(twoS)
	if err != nil {
		return err
	}
	op.proj = p
	return nil
}

// New binds prog to the space sp (whose K must match), with ints and the spatial
// orbital energies eps of a canonical reference with nocc doubly occupied orbitals.
// maxOrder is the scheme's per-block maximum order (B00 B01 B11 B02 B12 B22, -1 absent).
func New(prog *Program, sp *khci.Space, ints *integrals.Store, eps []float64, nocc int,
	maxOrder [6]int, be backend.Backend) (*Operator, error) {

	if pd, ok := be.(backend.PartitionedDevices); ok {
		return newPartitioned(prog, sp, ints, eps, nocc, maxOrder, be, pd)
	}
	return newOn(prog, sp, ints, eps, nocc, maxOrder, be)
}

// newOn binds the program to one backend.
func newOn(prog *Program, sp *khci.Space, ints *integrals.Store, eps []float64, nocc int,
	maxOrder [6]int, be backend.Backend) (*Operator, error) {

	if sp.K() != prog.K {
		return nil, fmt.Errorf("sigma: %s program has K=%d, space has K=%d", prog.Variant, prog.K, sp.K())
	}
	opt := sp.Options()
	if opt.NOcc != nocc || opt.NVir != ints.NVir() {
		return nil, fmt.Errorf("sigma: space has %d occupied and %d virtual orbitals, integrals %d and %d",
			opt.NOcc, opt.NVir, nocc, ints.NVir())
	}
	if len(eps) < nocc+opt.NVir {
		return nil, fmt.Errorf("sigma: %d orbital energies for %d orbitals", len(eps), nocc+opt.NVir)
	}
	k, staged, err := newKernels(be)
	if err != nil {
		return nil, err
	}
	ebe := be
	var stage backend.Backend
	if staged {
		// no tensor kernels and no host memory: run on the host, move panels across
		ebe, stage = backend.Gonum{}, be
		k = hostKernels{h: backend.Gonum{}, be: backend.Gonum{}}
	}
	e := &engine{be: ebe, k: k, prog: prog, ints: ints, eps: eps, nocc: nocc, nvir: opt.NVir,
		eri: map[string]tens{}, itmd: map[string]tens{}, inBld: map[string]bool{}}
	op := &Operator{e: e, sp: sp, main: sp.MainBlockSize(), MaxPanelBytes: 4 << 30, stage: stage,
		maxOrder: maxOrder}
	maxC := opt.MaxClass - sp.K()
	ampIdx := map[ampKey]int{}

	active := func(tn *Tensor) bool {
		return tn.Class <= maxC && (opt.AllMs || patternMs(tn.Spin, tn.Class) == opt.TwoMs)
	}
	keep := func(t *Term, withAmp bool) bool {
		if t.Block < 0 || t.Block > 5 || maxOrder[t.Block] < t.Order {
			return false
		}
		found := false
		for _, l := range t.Leaves {
			if l.Kind != Amp {
				continue
			}
			found = true
			if l.Class > maxC || (!opt.AllMs && patternMs(l.Spin, l.Class) != opt.TwoMs) {
				return false
			}
		}
		return found == withAmp
	}
	for i := range prog.Sigma {
		tn := &prog.Sigma[i]
		if !active(tn) {
			continue
		}
		b, err := e.newBlock(sp, tn, false)
		if err != nil {
			return nil, err
		}
		for j := range tn.Terms {
			t := &tn.Terms[j]
			if !keep(t, true) {
				continue
			}
			ct, err := e.compile(t, func(l Leaf) (int, error) {
				key := ampKey{l.Class, l.Spin}
				if ix, ok := ampIdx[key]; ok {
					return ix, nil
				}
				pm, err := buildPackMap(sp, l.Class, l.Spin, e.nocc, e.nvir, true)
				if err != nil {
					return 0, err
				}
				ampIdx[key] = len(op.amps)
				op.amps = append(op.amps, &amp{key: key, dims: e.modeDims(l.Class, len(l.Spin)), pm: pm})
				return ampIdx[key], nil
			})
			if err != nil {
				return nil, fmt.Errorf("sigma: %s block class %d spin %s: %w", prog.Variant, tn.Class, tn.Spin, err)
			}
			b.terms = append(b.terms, ct)
			if t.Block == 2 || t.Block >= 4 {
				b.sat = append(b.sat, ct)
			}
		}
		op.blocks = append(op.blocks, b)
	}
	for i := range prog.Diag {
		tn := &prog.Diag[i]
		if !active(tn) {
			continue
		}
		b, err := e.newBlock(sp, tn, true)
		if err != nil {
			return nil, err
		}
		for j := range tn.Terms {
			t := &tn.Terms[j]
			if !keep(t, false) {
				continue
			}
			ct, err := e.compile(t, nil)
			if err != nil {
				return nil, fmt.Errorf("sigma: %s diagonal class %d spin %s: %w", prog.Variant, tn.Class, tn.Spin, err)
			}
			b.terms = append(b.terms, ct)
		}
		op.diag = append(op.diag, b)
	}
	return op, nil
}

// modeDims are the extents of a class tensor: particles (virtual) then holes (occupied).
func (e *engine) modeDims(class, n int) []int {
	d := make([]int, n)
	for m := range d {
		d[m] = e.nocc
		if m < class {
			d[m] = e.nvir
		}
	}
	return d
}

func (e *engine) newBlock(sp *khci.Space, tn *Tensor, diag bool) (*block, error) {
	n := len(tn.Spin)
	pm, err := buildPackMap(sp, tn.Class, tn.Spin, e.nocc, e.nvir, false)
	if err != nil {
		return nil, err
	}
	if diag {
		for i := range pm.sign {
			pm.sign[i] = 1 // M(PI,PI) = M(I,I): the diagonal carries no permutation sign
		}
	}
	labels := make([]uint8, n)
	for m := range labels {
		labels[m] = uint8(m)
	}
	return &block{tensor: tn, dims: e.modeDims(tn.Class, n), labels: labels, pm: pm}, nil
}

// labelDim is the extent of a generated label of term t.
func (e *engine) labelDim(t *Term, l uint8) (int, error) {
	if int(l) >= len(t.Spaces) {
		return 0, fmt.Errorf("label %d beyond the term's %d label spaces", l, len(t.Spaces))
	}
	switch t.Spaces[l] {
	case 'o':
		return e.nocc, nil
	case 'v':
		return e.nvir, nil
	}
	return 0, fmt.Errorf("label %d in unknown space %q", l, t.Spaces[l])
}

func (e *engine) leafDims(t *Term, labels []uint8) ([]int, error) {
	d := make([]int, len(labels))
	for m, l := range labels {
		x, err := e.labelDim(t, l)
		if err != nil {
			return nil, err
		}
		d[m] = x
	}
	return d, nil
}

// compile resolves a term's leaves and precomputes its amplitude-free steps. ampOf
// registers an amplitude leaf (nil: the term must have none).
func (e *engine) compile(t *Term, ampOf func(Leaf) (int, error)) (*cterm, error) {
	ct := &cterm{t: t, leaves: make([]tens, len(t.Leaves)), amp: make([]int, len(t.Leaves))}
	for i, l := range t.Leaves {
		ct.amp[i] = -1
		switch l.Kind {
		case Amp:
			if ampOf == nil {
				return nil, fmt.Errorf("amplitude leaf in an amplitude-free term")
			}
			ix, err := ampOf(l)
			if err != nil {
				return nil, err
			}
			ct.amp[i] = ix
		case ERI:
			tn, err := e.eriSlice(t, l)
			if err != nil {
				return nil, err
			}
			ct.leaves[i] = tn
		case Itmd:
			tn, err := e.intermediate(l.Name, l.Spin)
			if err != nil {
				return nil, err
			}
			if len(tn.dims) != len(l.Labels) {
				return nil, fmt.Errorf("intermediate %s has %d modes, leaf %d labels", l.Name, len(tn.dims), len(l.Labels))
			}
			ct.leaves[i] = tens{v: tn.v, labels: l.Labels, dims: tn.dims}
		case Energy:
			tn, err := e.energy(t, l)
			if err != nil {
				return nil, err
			}
			ct.leaves[i] = tn
		default:
			return nil, fmt.Errorf("leaf kind %d", l.Kind)
		}
	}
	nops := len(t.Leaves) + len(t.Steps)
	ct.static = make([]bool, nops)
	ct.cache = make([]tens, nops)
	for i := range t.Leaves {
		ct.static[i] = ct.amp[i] < 0
		ct.cache[i] = ct.leaves[i]
	}
	for s, st := range t.Steps {
		o := len(t.Leaves) + s
		if st.A >= o || st.B >= o || st.A < 0 || st.B < 0 {
			return nil, fmt.Errorf("step %d reads operand %d/%d, only %d exist", s, st.A, st.B, o)
		}
		if !ct.static[st.A] || !ct.static[st.B] {
			continue
		}
		r, err := e.contract(ct.cache[st.A], ct.cache[st.B], st.Out)
		if err != nil {
			return nil, fmt.Errorf("step %d: %w", s, err)
		}
		ct.static[o] = true
		ct.cache[o] = r
	}
	return ct, nil
}

// eriSlice is the Coulomb tensor (l0 l1|l2 l3) over the leaf's label spaces.
func (e *engine) eriSlice(t *Term, l Leaf) (tens, error) {
	if len(l.Labels) != 4 {
		return tens{}, fmt.Errorf("Coulomb leaf with %d labels", len(l.Labels))
	}
	key := make([]byte, 4)
	for m, lb := range l.Labels {
		if int(lb) >= len(t.Spaces) {
			return tens{}, fmt.Errorf("label %d has no space", lb)
		}
		key[m] = t.Spaces[lb]
	}
	dims, err := e.leafDims(t, l.Labels)
	if err != nil {
		return tens{}, err
	}
	if c, ok := e.eri[string(key)]; ok {
		return tens{v: c.v, labels: l.Labels, dims: dims}, nil
	}
	off := [4]int{}
	for m, s := range key {
		if s == 'v' {
			off[m] = e.nocc
		}
	}
	host := make([]float64, size(dims))
	d0, d1, d2 := dims[0], dims[1], dims[2]
	for s := range dims[3] {
		for r := range d2 {
			for q := range d1 {
				base := ((s*d2+r)*d1 + q) * d0
				for p := range d0 {
					host[base+p] = e.ints.Eri(p+off[0], q+off[1], r+off[2], s+off[3])
				}
			}
		}
	}
	c := tens{v: e.be.Upload(host), dims: dims}
	e.eri[string(key)] = c
	return tens{v: c.v, labels: l.Labels, dims: dims}, nil
}

// energy evaluates an Energy leaf's expression at every assignment of its labels.
func (e *engine) energy(t *Term, l Leaf) (tens, error) {
	dims, err := e.leafDims(t, l.Labels)
	if err != nil {
		return tens{}, err
	}
	for _, in := range l.Expr {
		if in.Op == Eps && !has(l.Labels, in.Label) {
			return tens{}, fmt.Errorf("energy expression reads label %d outside its labels %v", in.Label, l.Labels)
		}
	}
	n := size(dims)
	host := make([]float64, n)
	val := make([]int, 256) // the value of every label at the current element
	stack := make([]float64, 0, 16)
	for x := range n {
		r := x
		for m, lb := range l.Labels {
			val[lb] = r % dims[m]
			r /= dims[m]
		}
		stack = stack[:0]
		for _, in := range l.Expr {
			switch in.Op {
			case Const:
				stack = append(stack, in.Val)
			case Eps:
				o := val[in.Label]
				if t.Spaces[in.Label] == 'v' {
					o += e.nocc
				}
				stack = append(stack, e.eps[o])
			case Add, Mul:
				k := int(in.N)
				if k < 1 || k > len(stack) {
					return tens{}, fmt.Errorf("energy instruction pops %d of %d", k, len(stack))
				}
				acc := stack[len(stack)-k]
				for _, v := range stack[len(stack)-k+1:] {
					if in.Op == Add {
						acc += v
					} else {
						acc *= v
					}
				}
				stack = append(stack[:len(stack)-k], acc)
			case Pow:
				if len(stack) == 0 || in.N == 0 {
					return tens{}, fmt.Errorf("energy power %d on %d values", in.N, len(stack))
				}
				stack[len(stack)-1] = math.Pow(stack[len(stack)-1], float64(in.N))
			}
		}
		if len(stack) != 1 {
			return tens{}, fmt.Errorf("energy expression leaves %d values", len(stack))
		}
		host[x] = stack[0]
	}
	return tens{v: e.be.Upload(host), labels: l.Labels, dims: dims}, nil
}

// intermediate builds (once) the spin block spin of intermediate name from its program.
func (e *engine) intermediate(name, spin string) (tens, error) {
	key := name + "|" + spin
	if t, ok := e.itmd[key]; ok {
		return t, nil
	}
	if e.inBld[key] {
		return tens{}, fmt.Errorf("intermediate %s defined through itself", key)
	}
	var def *Tensor
	for i := range e.prog.Itmds {
		if e.prog.Itmds[i].Name == name && e.prog.Itmds[i].Spin == spin {
			def = &e.prog.Itmds[i]
			break
		}
	}
	if def == nil {
		return tens{}, fmt.Errorf("no program for intermediate %s spin block %s", name, spin)
	}
	e.inBld[key] = true
	defer delete(e.inBld, key)
	dims := make([]int, len(def.Spaces))
	labels := make([]uint8, len(def.Spaces))
	for m, s := range def.Spaces {
		dims[m] = e.nocc
		if s == 'v' {
			dims[m] = e.nvir
		}
		labels[m] = uint8(m)
	}
	out := e.alloc(labels, dims)
	for j := range def.Terms {
		t := &def.Terms[j]
		ct, err := e.compile(t, nil)
		if err != nil {
			return tens{}, fmt.Errorf("intermediate %s: %w", key, err)
		}
		if err := e.accumulate(out, ct, ct.cache, t.Target); err != nil {
			return tens{}, fmt.Errorf("intermediate %s: %w", key, err)
		}
		e.release(ct)
	}
	out.labels = nil
	e.itmd[key] = out
	return out, nil
}

// accumulate adds Coef times the term's final operand into out, whose modes the target
// labels name (plus the column mode when the result carries one).
func (e *engine) accumulate(out tens, ct *cterm, ops []tens, target []uint8) error {
	t := ct.t
	last := len(ops) - 1
	if len(t.Steps) == 0 && len(t.Leaves) != 1 {
		return fmt.Errorf("term with %d leaves and no steps", len(t.Leaves))
	}
	r := ops[last]
	tl := target
	if has(r.labels, colLabel) {
		tl = append(append([]uint8(nil), target...), colLabel)
	}
	if len(tl) != len(out.dims) {
		return fmt.Errorf("target has %d labels for %d modes", len(tl), len(out.dims))
	}
	return e.k.einsum(t.Coef, tens{v: out.v, labels: tl, dims: out.dims}, r, nil)
}

// release frees a compiled term's step caches (not its shared leaves).
func (e *engine) release(ct *cterm) {
	for i := len(ct.t.Leaves); i < len(ct.cache); i++ {
		if ct.static[i] && ct.cache[i].v != nil {
			e.free(ct.cache[i])
		}
	}
}

// evaluate runs a term's amplitude-dependent steps with the amplitude blocks ampT (each
// with the column mode last) and returns every operand; owned marks the ones to free.
func (e *engine) evaluate(ct *cterm, ampT []tens) ([]tens, []bool, error) {
	t := ct.t
	ops := make([]tens, len(ct.cache))
	owned := make([]bool, len(ct.cache))
	for i, l := range t.Leaves {
		if ct.amp[i] >= 0 {
			a := ampT[ct.amp[i]]
			ops[i] = tens{v: a.v, labels: append(append([]uint8(nil), l.Labels...), colLabel), dims: a.dims}
		} else {
			ops[i] = ct.cache[i]
		}
	}
	for s, st := range t.Steps {
		o := len(t.Leaves) + s
		if ct.static[o] {
			ops[o] = ct.cache[o]
			continue
		}
		out := st.Out
		if has(ops[st.A].labels, colLabel) || has(ops[st.B].labels, colLabel) {
			out = append(append([]uint8(nil), out...), colLabel)
		}
		r, err := e.contract(ops[st.A], ops[st.B], out)
		if err != nil {
			return nil, nil, fmt.Errorf("step %d: %w", s, err)
		}
		ops[o] = r
		owned[o] = true
	}
	return ops, owned, nil
}

// Size is the dimension of the space.
func (op *Operator) Size() int { return op.sp.Size() }

// MainBlockSize is the number of main-class rows (they come first).
func (op *Operator) MainBlockSize() int { return op.main }

// Space is the configuration space the operator acts on.
func (op *Operator) Space() *khci.Space { return op.sp }

// ApplyFull computes out = M·in.
func (op *Operator) ApplyFull(out, in backend.Vector) {
	n := op.sp.Size()
	op.ApplyBlock(backend.BlockView{V: out, Rows: n, Cols: 1, Ld: n},
		backend.BlockView{V: in, Rows: n, Cols: 1, Ld: n})
}

// ApplyBlock computes out = M·in column by column block.
func (op *Operator) ApplyBlock(out, in backend.BlockView) {
	if err := op.apply(out, in, false); err != nil {
		panic(err)
	}
}

// ApplyBlockSatellite applies only the blocks free of the main class (B11 B12 B22).
func (op *Operator) ApplyBlockSatellite(out, in backend.BlockView) {
	if err := op.apply(out, in, true); err != nil {
		panic(err)
	}
}

// chunkCols is how many panel columns fit MaxPanelBytes of amplitude and σ tensors.
func (op *Operator) chunkCols(cols int) int {
	var per int64
	for _, a := range op.amps {
		per += int64(size(a.dims)) * 8
	}
	for _, b := range op.blocks {
		per += int64(size(b.dims)) * 8
	}
	if per == 0 {
		return cols
	}
	return int(max(1, min(int64(cols), op.MaxPanelBytes/per)))
}

// panelLen is the storage a column-major panel spans.
func panelLen(v backend.BlockView) int {
	if v.Cols == 0 {
		return 0
	}
	return (v.Cols-1)*v.Ld + v.Rows
}

// project applies the spin projector to a panel in place, on the host: directly for host
// memory, through a download and upload otherwise.
func (op *Operator) project(be backend.Backend, v backend.BlockView) {
	if h, ok := hostResident(be); ok {
		op.proj.Apply(h.HostSlice(v.V), v.Cols, v.Ld)
		return
	}
	sub := v.V.Slice(0, panelLen(v))
	host := be.Download(sub)
	op.proj.Apply(host, v.Cols, v.Ld)
	up := be.Upload(host)
	be.Copy(sub, up)
	be.Free(up)
}

func (op *Operator) apply(out, in backend.BlockView, satOnly bool) error {
	n := op.sp.Size()
	if in.Rows != n || out.Rows != n || in.Cols != out.Cols {
		return fmt.Errorf("sigma: apply %dx%d -> %dx%d on a space of %d", in.Rows, in.Cols, out.Rows, out.Cols, n)
	}
	if op.split != nil {
		return op.split.apply(out, in, satOnly)
	}
	if op.stage == nil {
		return op.applyOn(op.e.be, out, in, satOnly)
	}
	// a backend without tensor kernels: the panels cross to the host and back
	sb, g := op.stage, backend.Gonum{}
	hin := g.Upload(sb.Download(in.V.Slice(0, panelLen(in))))
	hout := g.Alloc(panelLen(out))
	err := op.applyOn(g, backend.BlockView{V: hout, Rows: n, Cols: out.Cols, Ld: out.Ld},
		backend.BlockView{V: hin, Rows: n, Cols: in.Cols, Ld: in.Ld}, satOnly)
	if err != nil {
		return err
	}
	// the rows between n and ld of out are not the operator's: keep what the caller had
	dst := sb.Download(out.V.Slice(0, panelLen(out)))
	hd := g.HostSlice(hout)
	for c := range out.Cols {
		copy(dst[c*out.Ld:c*out.Ld+n], hd[c*out.Ld:c*out.Ld+n])
	}
	up := sb.Upload(dst)
	sb.Copy(out.V.Slice(0, panelLen(out)), up)
	sb.Free(up)
	return nil
}

// applyOn runs the apply on be, the backend the engine's tensors live on.
func (op *Operator) applyOn(be backend.Backend, out, in backend.BlockView, satOnly bool) error {
	e := op.e
	n := op.sp.Size()
	for j := range out.Cols {
		be.Zero(out.Col(j))
	}
	if op.proj != nil {
		pin := be.Alloc(n * in.Cols)
		defer be.Free(pin)
		for c := range in.Cols {
			be.Copy(pin.Slice(c*n, n), in.Col(c))
		}
		in = backend.BlockView{V: pin, Rows: n, Cols: in.Cols, Ld: n}
		op.project(be, in)
		defer op.project(be, out)
	}
	step := op.chunkCols(in.Cols)
	for c0 := 0; c0 < in.Cols; c0 += step {
		nc := min(step, in.Cols-c0)
		inC, outC := in.ColRange(c0, c0+nc), out.ColRange(c0, c0+nc)
		ampT := make([]tens, len(op.amps))
		for i, a := range op.amps {
			d := append(append([]int(nil), a.dims...), nc)
			ampT[i] = e.alloc(nil, d)
			e.k.unpack(ampT[i].v, size(a.dims), inC.V, inC.Ld, nc, a.pm)
		}
		err := op.applyChunk(outC, ampT, nc, satOnly)
		for _, a := range ampT {
			e.free(a)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (op *Operator) applyChunk(out backend.BlockView, ampT []tens, nc int, satOnly bool) error {
	e := op.e
	for _, b := range op.blocks {
		terms := b.terms
		if satOnly {
			terms = b.sat
		}
		if len(terms) == 0 {
			continue
		}
		d := append(append([]int(nil), b.dims...), nc)
		tgt := e.alloc(nil, d)
		for _, ct := range terms {
			ops, owned, err := e.evaluate(ct, ampT)
			if err == nil {
				err = e.accumulate(tgt, ct, ops, ct.t.Target)
			}
			for i, o := range owned {
				if o {
					e.free(ops[i])
				}
			}
			if err != nil {
				e.free(tgt)
				return fmt.Errorf("sigma: class %d spin %s block %d order %d: %w",
					b.tensor.Class, b.tensor.Spin, ct.t.Block, ct.t.Order, err)
			}
		}
		e.k.pack(out.V, out.Ld, nc, tgt.v, size(b.dims), b.pm)
		e.free(tgt)
	}
	return nil
}

// Diagonal is diag(M) as a resident vector (the Davidson preconditioner's D).
func (op *Operator) Diagonal(be backend.Backend) backend.Vector {
	if op.split != nil {
		sub := op.split.ops[0]
		d := sub.Diagonal(backend.Gonum{})
		return be.Upload(backend.Gonum{}.Download(d))
	}
	e := op.e
	n := op.sp.Size()
	d := make([]float64, n)
	op.ciDiagonal(d)
	if len(op.diag) > 0 {
		dv := e.be.Alloc(n)
		for _, b := range op.diag {
			if len(b.terms) == 0 {
				continue
			}
			tgt := e.alloc(b.labels, b.dims)
			for _, ct := range b.terms {
				if err := e.accumulate(tgt, ct, ct.cache, ct.t.Target); err != nil {
					panic(fmt.Errorf("sigma: diagonal class %d spin %s: %w", b.tensor.Class, b.tensor.Spin, err))
				}
			}
			e.k.pack(dv, n, 1, tgt.v, size(b.dims), b.pm)
			e.free(tgt)
		}
		for r, v := range e.be.Download(dv) {
			d[r] += v
		}
		e.be.Free(dv)
	}
	return be.Upload(d)
}

// diagBlock is the diagonal block of each class: B00, B11, B22.
var diagBlock = [3]int{0, 2, 5}

// ciDiagonal adds the orders <= 1 of diag(M) the scheme keeps, row by row. Through first
// order a diagonal block of the ISR is CI (generate_adc.py asserts ISR(<=1) = CI on every
// block but B02), and the CI diagonal of a configuration relative to the reference is
//
//	order 0:  Σ_a ε_a − Σ_i ε_i
//	order 1:  Σ_{i<j} <ij||ij> + Σ_{a<b} <ab||ab> − Σ_{a,i} <ai||ai>
//
// over its particles a, b and holes i, j (spin orbitals, <pq||pq> = (pp|qq) − δ_σ (pq|qp)).
// The generated diagonal programs carry the orders from 2 on.
func (op *Operator) ciDiagonal(d []float64) {
	e, sp := op.e, op.sp
	K := sp.K()
	parallel.Rows(len(d), func(r int) {
		c := sp.Class(r) - K
		mo := op.maxOrder[diagBlock[c]]
		if mo < 0 {
			return
		}
		type so struct{ orb, spin int }
		var holes, parts []so
		for q := sp.HoleMask(r); q != 0; q &= q - 1 {
			p := bits.TrailingZeros64(q)
			holes = append(holes, so{p >> 1, p & 1})
		}
		for _, a := range sp.PartSO(r, make([]int, 0, 2)) {
			parts = append(parts, so{e.nocc + a>>1, a & 1})
		}
		var v float64
		for _, a := range parts {
			v += e.eps[a.orb]
		}
		for _, i := range holes {
			v -= e.eps[i.orb]
		}
		if mo >= 1 {
			asym := func(p, q so) float64 {
				x := e.ints.Eri(p.orb, p.orb, q.orb, q.orb)
				if p.spin == q.spin {
					x -= e.ints.Eri(p.orb, q.orb, q.orb, p.orb)
				}
				return x
			}
			for x := range holes {
				for y := x + 1; y < len(holes); y++ {
					v += asym(holes[x], holes[y])
				}
			}
			for x := range parts {
				for y := x + 1; y < len(parts); y++ {
					v += asym(parts[x], parts[y])
				}
			}
			for _, a := range parts {
				for _, i := range holes {
					v -= asym(a, i)
				}
			}
		}
		d[r] += v
	})
}
