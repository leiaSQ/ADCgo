package backend

import (
	"reflect"
	"strings"
	"testing"
)

// TestDistBackendAddPanel guards the matrix-free -mgpu scatter-add: AddPanel must add a full
// host panel into only the first cols columns of a wider allocated output panel (the solver
// sizes panels to the max block width, so an apply commonly uses fewer columns than allocated).
// The regression it locks: adding across the whole partition storage instead of the first
// cols·rowsOn(d) contiguous columns, which mismatched lengths and corrupted the strided layout.
func TestDistBackendAddPanel(t *testing.T) {
	const n, main = 12, 2 // n > 2·main² = 8, so the shape invariant holds
	subs := []Backend{Gonum{}, Gonum{}}
	bounds := []int{0, 6, n}
	be, err := NewDistributed(subs, n, main, bounds)
	if err != nil {
		t.Fatalf("NewDistributed: %v", err)
	}

	const cols = 4    // allocated panel width
	const addCols = 2 // apply width (fewer than allocated)
	out := be.Alloc(n * cols)
	be.Zero(out)

	full := make([]float64, n*addCols)
	for i := range full {
		full[i] = float64(i + 1)
	}
	be.(PanelScatterAdd).AddPanel(out, full)

	got := be.Download(out) // n × cols, column-major
	for c := range cols {
		for r := range n {
			want := 0.0
			if c < addCols {
				want = full[c*n+r]
			}
			if g := got[c*n+r]; g != want {
				t.Errorf("col %d row %d: got %g, want %g (columns >= addCols must stay untouched)", c, r, g, want)
			}
		}
	}
}

// TestDistBackendPartitionedDevices covers the PartitionedDevices capability that backs the
// per-device satellite apply: the partition metadata a caller needs (count, bounds, sub-backend,
// device kernels, peer capability) and the per-device panel storage. Uneven bands are used
// deliberately — equal splits would hide an off-by-one in the row-band arithmetic. Gonum
// sub-backends give full coverage of everything except the kernel launch itself.
func TestDistBackendPartitionedDevices(t *testing.T) {
	const n, main = 12, 2
	subs := []Backend{Gonum{}, Gonum{}, Gonum{}}
	bounds := []int{0, 3, 7, n} // uneven: 3, 4, 5 rows
	be, err := NewDistributed(subs, n, main, bounds)
	if err != nil {
		t.Fatalf("NewDistributed: %v", err)
	}
	pd, ok := be.(PartitionedDevices)
	if !ok {
		t.Fatal("distributed backend does not implement PartitionedDevices")
	}

	if got := pd.NumParts(); got != len(subs) {
		t.Errorf("NumParts = %d, want %d", got, len(subs))
	}

	// Bounds must be a defensive copy — a caller mutating it must not repartition the backend.
	got := pd.Bounds()
	for i, w := range bounds {
		if got[i] != w {
			t.Fatalf("Bounds()[%d] = %d, want %d", i, got[i], w)
		}
	}
	got[1] = 999
	if again := pd.Bounds(); again[1] != bounds[1] {
		t.Errorf("Bounds() is not a copy: caller mutation leaked into the backend (%d)", again[1])
	}

	for d := range subs {
		if pd.PartBackend(d) != subs[d] {
			t.Errorf("PartBackend(%d) returned the wrong sub-backend", d)
		}
		if _, ok := pd.PartKernels(d); ok {
			t.Errorf("PartKernels(%d) reported device kernels for a Gonum sub-backend", d)
		}
	}
	if pd.AllPeered() {
		t.Error("AllPeered = true over Gonum sub-backends; must be false so the host fallback stays")
	}

	// PartVector must hand back exactly device d's row band, column-major with ld = rowsOn(d).
	const cols = 2
	host := make([]float64, n*cols)
	for i := range host {
		host[i] = float64(i + 1)
	}
	v := be.Upload(host)
	for d := range subs {
		lo, hi := bounds[d], bounds[d+1]
		rd := hi - lo
		local := subs[d].Download(pd.PartVector(v, d))
		if len(local) != rd*cols {
			t.Fatalf("PartVector(%d) length = %d, want %d", d, len(local), rd*cols)
		}
		for c := range cols {
			for r := range rd {
				want := host[c*n+lo+r]
				if g := local[c*rd+r]; g != want {
					t.Errorf("part %d col %d row %d: got %g, want %g", d, c, r, g, want)
				}
			}
		}
	}

	// Shapes the per-device apply must never be handed: a replicated small buffer, and a
	// located row band. Accepting either silently would apply the operator to the wrong rows.
	// The message is asserted too: without it a panic raised somewhere else (e.g. inside Slice)
	// would pass the test while PartVector's guard was never reached.
	mustPanic := func(name, want string, fn func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Errorf("PartVector accepted %s; want panic", name)
				return
			}
			if msg, _ := r.(string); !strings.Contains(msg, want) {
				t.Errorf("PartVector(%s) panicked with %q; want a message containing %q", name, r, want)
			}
		}()
		fn()
	}
	mustPanic("a replicated buffer", "replicated", func() { pd.PartVector(be.Alloc(main*main), 0) })
	mustPanic("a located row band", "located row band", func() {
		pd.PartVector(v.Slice(bounds[1], cols*n-(n-1)), 1)
	})
}

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

// TestDistBackendDownloadInto guards a trap that cost a production run 40 h of compute.
//
// distBackend EMBEDS Gonum, to inherit the host SymEig and anything else it does not override.
// That embedding also makes distBackend satisfy BufferedDownloader whether or not it implements
// DownloadInto itself — and the inherited Gonum.DownloadInto does host(v), i.e. v.(hostVec), which
// panics on a distVec. So a caller doing the ordinary
//
//	bd, ok := be.(BufferedDownloader)
//
// gets ok == true and a method that cannot work, instead of falling back to the allocating
// Download path it intended. That is precisely how the production DIP probe (job 14158038) died at
// its first checkpoint with "interface conversion: backend.Vector is backend.distVec, not
// backend.hostVec", losing a 40 h block.
//
// The test therefore asserts both halves: DownloadInto must agree with Download element-for-element
// on a distributed panel, and it must do so WITHOUT panicking — which only holds while distBackend
// overrides the embedded method. Deleting the override reintroduces the outage.
func TestDistBackendDownloadInto(t *testing.T) {
	const n, main = 12, 2 // n > 2·main² = 8
	// Deliberately uneven bands, so a per-device offset error cannot cancel out.
	subs := []Backend{Gonum{}, Gonum{}, Gonum{}}
	bounds := []int{0, 3, 7, n}
	be, err := NewDistributed(subs, n, main, bounds)
	if err != nil {
		t.Fatalf("NewDistributed: %v", err)
	}

	bd, ok := be.(BufferedDownloader)
	if !ok {
		t.Fatal("distBackend does not satisfy BufferedDownloader")
	}

	for _, cols := range []int{1, 2, 5} {
		host := make([]float64, n*cols)
		for i := range host {
			host[i] = float64(i) * 0.5
		}
		v := be.Upload(host)

		want := be.Download(v)
		dst := make([]float64, n*cols)
		bd.DownloadInto(dst, v)
		if !reflect.DeepEqual(dst, want) {
			t.Errorf("cols=%d: DownloadInto disagrees with Download\n got %v\nwant %v", cols, dst, want)
		}
		if !reflect.DeepEqual(dst, host) {
			t.Errorf("cols=%d: round trip lost data", cols)
		}

		// A column sub-range must work too: that is what the checkpoint writer actually streams.
		if cols >= 2 {
			panel := BlockView{V: v, Rows: n, Cols: cols, Ld: n}
			sub := panel.ColRange(1, cols).V
			subWant := be.Download(sub)
			subDst := make([]float64, len(subWant))
			bd.DownloadInto(subDst, sub)
			if !reflect.DeepEqual(subDst, subWant) {
				t.Errorf("cols=%d: DownloadInto on a ColRange disagrees with Download", cols)
			}
		}
		be.Free(v)
	}

	// Reusing the staging buffers across calls must not leak state between them.
	a := be.Upload(make([]float64, n*2))
	big := make([]float64, n*2)
	for i := range big {
		big[i] = 7
	}
	bcv := be.Upload(big)
	dst := make([]float64, n*2)
	bd.DownloadInto(dst, bcv)
	bd.DownloadInto(dst, a)
	for i, v := range dst {
		if v != 0 {
			t.Fatalf("stale staging data at %d: got %g, want 0", i, v)
		}
	}
}

// TestDistBackendDownloadIntoRejectsSmallDst pins the same contract gpuBackend states: a dst that
// cannot hold the panel is a programming error, reported as such rather than silently truncating.
func TestDistBackendDownloadIntoRejectsSmallDst(t *testing.T) {
	const n, main = 12, 2
	be, err := NewDistributed([]Backend{Gonum{}, Gonum{}}, n, main, []int{0, 6, n})
	if err != nil {
		t.Fatalf("NewDistributed: %v", err)
	}
	v := be.Upload(make([]float64, n*2))
	defer be.Free(v)

	defer func() {
		if recover() == nil {
			t.Error("DownloadInto accepted an undersized dst")
		}
	}()
	be.(BufferedDownloader).DownloadInto(make([]float64, n), v)
}
