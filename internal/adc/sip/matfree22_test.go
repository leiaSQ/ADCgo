package sip

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
)

// applyDense multiplies the densely materialized matrix by v, as the reference for
// every operator gate below.
func applyDense(M backend.Mat, v []float64) []float64 {
	out := make([]float64, M.Rows)
	for r := range M.Rows {
		var s float64
		for c := range len(v) {
			s += M.At(r, c) * v[c]
		}
		out[r] = s
	}
	return out
}

func probeVec(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = math.Sin(float64(i)*0.7) + 0.3
	}
	return v
}

// TestADC22MatFreeMatchesDense is the gate on the matrix-free path: with -matfree on,
// the 2h1p/3h2p coupling and the 3h2p/3h2p block are recomputed element by element
// over pruned candidate lists instead of being stored, and the resulting operator
// must reproduce the dense one. It runs all three variants, so it covers the
// coupling alone (ADC(2,2)_m, whose 3h2p block is a diagonal vector) and the
// coupling plus the first-order 3h2p block (x and f).
//
// Exact agreement is not expected and not required: the dense path sums each row
// inside a BLAS GEMV while the matrix-free path accumulates over its candidate
// sweep, so the two reassociate the same terms. Agreement at the 1e-12 level is
// what says they are the same operator.
func TestADC22MatFreeMatchesDense(t *testing.T) {
	for _, v := range []Variant{VariantM, VariantX, VariantF} {
		dense := build22(t, 8, v)
		M := dense.BuildMatrix()
		n := dense.Size()
		in := probeVec(n)
		want := applyDense(M, in)

		mf := build22(t, 8, v)
		mf.SetMatFree(MatFreeOn, 0)
		xv, yv := mf.be.Upload(in), mf.be.Upload(make([]float64, n))
		mf.ApplyFull(yv, xv)
		got := mf.be.Download(yv)

		if len(mf.op.mf) == 0 {
			t.Fatalf("variant %s: MatFreeOn assembled no matrix-free part", v)
		}
		wantParts := 1 // the 2h1p/3h2p coupling
		if v != VariantM {
			wantParts = 2 // plus the 3h2p/3h2p block
		}
		if len(mf.op.mf) != wantParts {
			t.Errorf("variant %s: %d matrix-free parts, want %d", v, len(mf.op.mf), wantParts)
		}

		var maxDiff, maxAbs float64
		for i := range got {
			maxAbs = math.Max(maxAbs, math.Abs(want[i]))
			maxDiff = math.Max(maxDiff, math.Abs(got[i]-want[i]))
		}
		t.Logf("variant %s: matrix-free vs dense ApplyFull over n=%d (3h2p=%d), "+
			"max |component| = %.6g, max deviation = %.3g",
			v, n, len(mf.sp.Sat3), maxAbs, maxDiff)
		if maxDiff > 1e-12*math.Max(1, maxAbs) {
			t.Errorf("variant %s: matrix-free deviates from dense by %.3g", v, maxDiff)
		}
		mf.Release()
	}
}

// TestADC22MatFreeBlockApply checks the block (multi-column) apply path, which is
// the one the Lanczos solver actually drives: it exercises the leading-dimension
// arithmetic and the per-worker forward accumulators at a Krylov width above 1,
// where a partials buffer sized for b == 1 would silently truncate.
func TestADC22MatFreeMatchesDenseBlock(t *testing.T) {
	const b = 3
	for _, v := range []Variant{VariantM, VariantF} {
		dense := build22(t, 8, v)
		M := dense.BuildMatrix()
		n := dense.Size()

		cols := make([][]float64, b)
		flat := make([]float64, n*b)
		for j := range b {
			cols[j] = make([]float64, n)
			for i := range n {
				cols[j][i] = math.Cos(float64(i*(j+1))*0.31) - 0.2
				flat[i+j*n] = cols[j][i]
			}
		}

		mf := build22(t, 8, v)
		mf.SetMatFree(MatFreeOn, 0)
		be := mf.be
		inV := backend.BlockView{V: be.Upload(flat), Rows: n, Cols: b, Ld: n}
		outV := backend.BlockView{V: be.Upload(make([]float64, n*b)), Rows: n, Cols: b, Ld: n}
		mf.ApplyBlock(outV, inV)
		got := be.Download(outV.V)

		var maxDiff, maxAbs float64
		for j := range b {
			want := applyDense(M, cols[j])
			for i := range n {
				maxAbs = math.Max(maxAbs, math.Abs(want[i]))
				maxDiff = math.Max(maxDiff, math.Abs(got[i+j*n]-want[i]))
			}
		}
		t.Logf("variant %s: matrix-free ApplyBlock(b=%d) vs dense, max deviation %.3g", v, b, maxDiff)
		if maxDiff > 1e-12*math.Max(1, maxAbs) {
			t.Errorf("variant %s: matrix-free block apply deviates by %.3g", v, maxDiff)
		}
		mf.Release()
	}
}

// TestADC22MatFreeDeterministic requires repeated applies of one assembled operator
// to agree BIT FOR BIT. The 2h1p/3h2p applier reduces per-worker partials, and a
// buffer that was not fully re-zeroed, or a worker count derived per call, would
// show up here as a moving operator — which Lanczos cannot survive, since its short
// recurrence assumes a fixed M.
func TestADC22MatFreeDeterministic(t *testing.T) {
	mx := build22(t, 8, VariantF)
	mx.SetMatFree(MatFreeOn, 0)
	defer mx.Release()
	n := mx.Size()
	in := probeVec(n)
	xv := mx.be.Upload(in)

	var first []float64
	for it := range 4 {
		yv := mx.be.Upload(make([]float64, n))
		mx.ApplyFull(yv, xv)
		got := mx.be.Download(yv)
		if it == 0 {
			first = got
			continue
		}
		for i := range got {
			if got[i] != first[i] {
				t.Fatalf("apply %d differs from apply 0 at row %d: %.17g vs %.17g",
					it, i, got[i], first[i])
			}
		}
	}
	t.Logf("4 applies of the matrix-free ADC(2,2)f operator agree bit for bit over n=%d", n)
}

// TestC23SelectionRuleComplete is the gate on the 2h1p/3h2p prune. The applier only
// ever visits candidate rows, so a selection rule that missed a nonzero element
// would drop it from the operator silently — no error, just a wrong spectrum. This
// walks the whole dense block and requires every nonzero to be in the candidate set.
func TestC23SelectionRuleComplete(t *testing.T) {
	mx := build22(t, 8, VariantF)
	sp := mx.sp
	rows := sp.Configs[sp.BeginSat:sp.Begin3h2p]
	ix := buildC23Index(rows, sp.Nocc)

	var mrg ascMerge
	visited := 0
	nonzero := 0
	for _, col := range sp.Sat3 {
		cand := ix.candidates(&mrg, col)
		in := make(map[int32]bool, len(cand))
		for _, r := range cand {
			in[r] = true
		}
		visited += len(cand)
		for r := range rows {
			g := mx.el.c23_1(rows[r], col)
			if g == 0 {
				continue
			}
			nonzero++
			if !in[int32(r)] {
				t.Fatalf("selection rule misses a nonzero: 2h1p row %+v x 3h2p col %+v = %.6g",
					rows[r], col, g)
			}
		}
	}
	total := len(rows) * len(sp.Sat3)
	t.Logf("2h1p/3h2p: %d nonzero of %d elements (%.2f%%); the prune visits %d (%.2f%%), "+
		"covering every nonzero",
		nonzero, total, 100*float64(nonzero)/float64(total),
		visited, 100*float64(visited)/float64(total))
	if visited >= total {
		t.Errorf("the prune visits %d of %d candidates — it is not pruning", visited, total)
	}
}

// TestSat3SelectionRuleComplete is the same gate on the 3h2p/3h2p prune, where it
// matters most: that block is n3² and n3 grows as the fifth power of the system, so
// the applier can never afford to sweep it.
func TestSat3SelectionRuleComplete(t *testing.T) {
	mx := build22(t, 8, VariantF)
	sp := mx.sp
	cfgs := sp.Sat3
	ix := buildSat3Index(cfgs, sp.Nocc, sp.Nvir)

	var mrg ascMerge
	visited, nonzero := 0, 0
	for r, row := range cfgs {
		cand := ix.candidates(&mrg, row)
		in := make(map[int32]bool, len(cand))
		asc := int32(-1)
		for _, c := range cand {
			if c <= asc {
				t.Fatalf("candidate list for row %d is not strictly ascending at %d", r, c)
			}
			asc = c
			in[c] = true
		}
		visited += len(cand)
		for c := range cfgs {
			g := mx.el.c33_01(row, cfgs[c])
			if g == 0 {
				continue
			}
			nonzero++
			if !in[int32(c)] {
				t.Fatalf("selection rule misses a nonzero: 3h2p %+v x %+v = %.6g",
					row, cfgs[c], g)
			}
		}
	}
	total := len(cfgs) * len(cfgs)
	t.Logf("3h2p/3h2p: %d nonzero of %d elements (%.2f%%); the prune visits %d (%.2f%%), "+
		"covering every nonzero",
		nonzero, total, 100*float64(nonzero)/float64(total),
		visited, 100*float64(visited)/float64(total))
	if visited >= total {
		t.Errorf("the prune visits %d of %d candidates — it is not pruning", visited, total)
	}
}

// TestADC22MatFreeBudget pins the -matfree / -maxmem policy: Off keeps both blocks
// dense, On recomputes both, and Auto follows the byte budget. OperatorResidentBytes
// must agree with the decision, since that is what a caller sizes a sector against
// before committing a job to it.
func TestADC22MatFreeBudget(t *testing.T) {
	sizeOf := func(mode MatFreeMode, budget int64) (uint64, int) {
		mx := build22(t, 8, VariantF)
		mx.SetMatFree(mode, budget)
		b := mx.OperatorResidentBytes()
		mx.ApplyFull(mx.be.Upload(make([]float64, mx.Size())), mx.be.Upload(probeVec(mx.Size())))
		n := len(mx.op.mf)
		mx.Release()
		return b, n
	}

	offBytes, offParts := sizeOf(MatFreeOff, 0)
	onBytes, onParts := sizeOf(MatFreeOn, 0)
	if offParts != 0 {
		t.Errorf("MatFreeOff assembled %d matrix-free parts, want 0", offParts)
	}
	if onParts != 2 {
		t.Errorf("MatFreeOn assembled %d matrix-free parts, want 2", onParts)
	}
	if onBytes >= offBytes {
		t.Errorf("matrix-free residency %d is not below dense %d", onBytes, offBytes)
	}

	// Auto with a budget above the largest block keeps everything dense; below it,
	// both oversized blocks recompute.
	if b, n := sizeOf(MatFreeAuto, 1<<40); n != 0 || b != offBytes {
		t.Errorf("MatFreeAuto with a 1 TiB budget went matrix-free: %d parts, %d bytes", n, b)
	}
	if b, n := sizeOf(MatFreeAuto, 1<<10); n != 2 || b != onBytes {
		t.Errorf("MatFreeAuto with a 1 KiB budget stayed dense: %d parts, %d bytes", n, b)
	}
	t.Logf("ADC(2,2)f operator residency: dense %d bytes, matrix-free %d bytes (%.1fx smaller)",
		offBytes, onBytes, float64(offBytes)/float64(onBytes))
}

// TestPT2TablesMatchDirectSums gates the A10/A11 memoization against the literal
// sums it replaces. The tables are accumulated as outer products over a different
// loop nesting than Eqs. (A10)/(A11) are written in, so this is what pins that the
// restructuring is algebra: the same terms, regrouped, not a different formula.
func TestPT2TablesMatchDirectSums(t *testing.T) {
	mx := build22(t, 8, VariantF)
	e := mx.el
	e.pt2.Do(e.buildPT2Tables)
	occ, vir := e.soLists()
	no, nv := len(occ), len(vir)

	var maxA, absA float64
	for li, l := range occ {
		for lp, lpp := range occ {
			want := e.m2AInner(l, lpp)
			got := e.tblA[li*no+lp]
			absA = math.Max(absA, math.Abs(want))
			maxA = math.Max(maxA, math.Abs(got-want))
		}
	}
	var maxB, absB float64
	for ai, a := range vir {
		for api, ap := range vir {
			want := e.m2BInner(a, ap)
			got := e.tblB[ai*nv+api]
			absB = math.Max(absB, math.Abs(want))
			maxB = math.Max(maxB, math.Abs(got-want))
		}
	}
	t.Logf("A10 table vs Eq. (A10) over %d pairs: max |value| %.6g, max deviation %.3g",
		no*no, absA, maxA)
	t.Logf("A11 table vs Eq. (A11) over %d pairs: max |value| %.6g, max deviation %.3g",
		nv*nv, absB, maxB)
	if maxA > 1e-12*math.Max(1, absA) {
		t.Errorf("A10 table deviates from the literal sum by %.3g", maxA)
	}
	if maxB > 1e-12*math.Max(1, absB) {
		t.Errorf("A11 table deviates from the literal sum by %.3g", maxB)
	}
}

// TestBucketSetRoundTrip pins the flat-backed bucket index: every configuration
// lands in exactly the buckets its keys name, each list ascending and the lists
// jointly accounting for every emitted key.
func TestBucketSetRoundTrip(t *testing.T) {
	keys := [][]int64{{7}, {7, 9}, {9}, {}, {7, 9, 11}, {11}}
	b := newBucketSet(len(keys), func(i int, emit func(int64)) {
		for _, k := range keys[i] {
			emit(k)
		}
	})
	want := map[int64][]int32{7: {0, 1, 4}, 9: {1, 2, 4}, 11: {4, 5}}
	for k, w := range want {
		got := b.get(k)
		if len(got) != len(w) {
			t.Fatalf("key %d: got %v, want %v", k, got, w)
		}
		for i := range w {
			if got[i] != w[i] {
				t.Fatalf("key %d: got %v, want %v", k, got, w)
			}
		}
	}
	if got := b.get(13); got != nil {
		t.Errorf("absent key returned %v, want nil", got)
	}
	if n := len(b.items); n != 8 {
		t.Errorf("backing array holds %d entries, want 8", n)
	}
}

// TestAscMergeDedupes checks the candidate merge: ascending, deduplicated, and
// reusable without leaking state between calls.
func TestAscMergeDedupes(t *testing.T) {
	var m ascMerge
	m.reset()
	m.add([]int32{1, 4, 9})
	m.add([]int32{4, 5})
	m.add(nil)
	m.add([]int32{0, 9, 12})
	got := m.merge()
	want := []int32{0, 1, 4, 5, 9, 12}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	m.reset()
	m.add([]int32{3})
	if got := m.merge(); len(got) != 1 || got[0] != 3 {
		t.Errorf("reuse leaked state: got %v, want [3]", got)
	}
}

// TestSort3 covers every ordering of three values, including coincidences — a
// coincident hole pair is the common case in the 3h2p space, and an unsorted key
// would silently split one bucket in two.
func TestSort3(t *testing.T) {
	for a := range 3 {
		for b := range 3 {
			for c := range 3 {
				x, y, z := sort3(a, b, c)
				if x > y || y > z {
					t.Fatalf("sort3(%d,%d,%d) = %d,%d,%d", a, b, c, x, y, z)
				}
				if x+y+z != a+b+c {
					t.Fatalf("sort3(%d,%d,%d) = %d,%d,%d loses a value", a, b, c, x, y, z)
				}
			}
		}
	}
}

// TestSat3PruneScalesWithNocc measures the quantity the whole matrix-free design
// rests on, and checks it against the scaling law matfree22.go claims: the visited
// fraction of the 3h2p block goes as
//
//	6/n_occ³ + 72/(n_occ²·n_virt) + 18/(n_occ·n_virt²)
//
// for the three candidate families in order. It has to be measured on synthetic
// spaces, because the only fixture in testdata is water, whose n_occ = 5 is the
// worst case there is — the tests above see a mere 2-9x, and a reader would
// reasonably conclude from those numbers that the index is not worth its memory. At
// the production sector's n_occ = 58, n_virt = 154 the same formula gives ~8000x.
// The index depends on nothing but the configuration lists, so counting candidates
// needs no integrals.
//
// The asserted bound is one-sided and loose (the law must be an upper bound, and
// within an order of magnitude) because the law drops the spin multiplicities, which
// differ between the families. What it pins is the shape: the prune must tighten in
// BOTH n_occ and n_virt, and must not be a constant factor.
func TestSat3PruneScalesWithNocc(t *testing.T) {
	if testing.Short() {
		t.Skip("counts candidates over synthetic 3h2p spaces up to ~10^5 configs; -short skips it")
	}
	law := func(nocc, nvir int) float64 {
		o, v := float64(nocc), float64(nvir)
		return 6/(o*o*o) + 72/(o*o*v) + 18/(o*v*v)
	}
	measure := func(nocc, nvir int) float64 {
		sp := NewSpace22(nocc, nocc+nvir, nil, 0)
		n3 := len(sp.Sat3)
		ix := buildSat3Index(sp.Sat3, sp.Nocc, sp.Nvir)
		var mrg ascMerge
		visited := 0
		for _, row := range sp.Sat3 {
			visited += len(ix.candidates(&mrg, row))
		}
		frac := float64(visited) / (float64(n3) * float64(n3))
		pred := law(nocc, nvir)
		t.Logf("nocc=%2d nvir=%2d: n3=%6d, %7.4f%% of the block visited (%5.0fx prune); "+
			"law predicts %7.4f%%", nocc, nvir, n3, 100*frac, 1/frac, 100*pred)
		if frac > pred {
			t.Errorf("nocc=%d nvir=%d: visited fraction %.5f exceeds the predicted bound %.5f",
				nocc, nvir, frac, pred)
		}
		if frac < 0.1*pred {
			t.Errorf("nocc=%d nvir=%d: visited fraction %.5f is more than 10x below the law %.5f; "+
				"the law has drifted from the code", nocc, nvir, frac, pred)
		}
		return frac
	}

	// Tightening in n_occ at fixed n_virt.
	prev := math.Inf(1)
	for _, nocc := range []int{4, 6, 8, 10, 12} {
		frac := measure(nocc, 6)
		if frac >= prev {
			t.Errorf("nocc=%d: visited fraction %.5f did not fall below the previous %.5f", nocc, frac, prev)
		}
		prev = frac
	}
	// and in n_virt at fixed n_occ.
	prev = math.Inf(1)
	for _, nvir := range []int{4, 8, 14, 20} {
		frac := measure(8, nvir)
		if frac >= prev {
			t.Errorf("nvir=%d: visited fraction %.5f did not fall below the previous %.5f", nvir, frac, prev)
		}
		prev = frac
	}
}
