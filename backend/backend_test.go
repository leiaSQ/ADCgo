package backend

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestGemvNT(t *testing.T) {
	b := Gonum{}
	// a = [[1 2 3],[4 5 6]] (2x3), row-major.
	a := b.UploadMat(Mat{Rows: 2, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6}})

	// y += a*x, x length 3.
	x := b.Upload(Vec{1, 1, 1})
	y := b.Alloc(2)
	b.GemvN(1, a, x, y)
	if gy := b.Download(y); gy[0] != 6 || gy[1] != 15 {
		t.Fatalf("GemvN = %v, want [6 15]", gy)
	}

	// y += aᵀ*x, x length 2 (accumulate semantics: start non-zero).
	xt := b.Upload(Vec{1, 1})
	yt := b.Upload(Vec{10, 20, 30})
	b.GemvT(1, a, xt, yt)
	if gyt := b.Download(yt); gyt[0] != 15 || gyt[1] != 27 || gyt[2] != 39 {
		t.Fatalf("GemvT = %v, want [15 27 39]", gyt)
	}
}

// TestAxpyDiag: y += d ⊙ x with accumulate semantics, and through a Slice view (the
// mechanism the CVS-ADC(4) 3h2p diagonal block uses to apply over its row band).
func TestAxpyDiag(t *testing.T) {
	b := Gonum{}
	d := b.Upload(Vec{2, 3, 4})
	x := b.Upload(Vec{5, 6, 7})
	y := b.Upload(Vec{1, 1, 1}) // non-zero start: must accumulate
	b.AxpyDiag(d, x, y)
	if gy := b.Download(y); gy[0] != 11 || gy[1] != 19 || gy[2] != 29 {
		t.Fatalf("AxpyDiag = %v, want [11 19 29]", gy)
	}

	// Applied to rows [1,4) of a length-5 vector, leaving the rest untouched.
	out := b.Upload(Vec{9, 0, 0, 0, 9})
	xin := b.Upload(Vec{9, 5, 6, 7, 9})
	b.AxpyDiag(d, xin.Slice(1, 3), out.Slice(1, 3))
	if got := b.Download(out); got[0] != 9 || got[1] != 10 || got[2] != 18 || got[3] != 28 || got[4] != 9 {
		t.Fatalf("AxpyDiag slice = %v, want [9 10 18 28 9]", got)
	}
}

// TestSliceView: a GEMV into a Slice view must write through to the parent
// vector's sub-range (the mechanism the DIP mat-vec uses for block offsets).
func TestSliceView(t *testing.T) {
	b := Gonum{}
	a := b.UploadMat(Mat{Rows: 2, Cols: 2, Data: []float64{1, 0, 0, 1}}) // identity
	out := b.Alloc(5)
	x := b.Upload(Vec{7, 9})
	// Write identity*x into rows [2,4) of out.
	b.GemvN(1, a, x, out.Slice(2, 2))
	got := b.Download(out)
	want := []float64{0, 0, 7, 9, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("slice-view GemvN = %v, want %v", got, want)
		}
	}
}

func TestSymEig(t *testing.T) {
	b := Gonum{}
	// [[2 1],[1 2]] → eigenvalues 1, 3; vectors (1,-1)/√2, (1,1)/√2.
	a := Mat{Rows: 2, Cols: 2, Data: []float64{2, 1, 1, 2}}
	evals, evecs := b.SymEig(a)
	if len(evals) != 2 {
		t.Fatalf("got %d eigenvalues", len(evals))
	}
	if math.Abs(evals[0]-1) > 1e-12 || math.Abs(evals[1]-3) > 1e-12 {
		t.Fatalf("eigenvalues = %v, want [1 3]", evals)
	}
	// Reconstruct A from V diag(λ) Vᵀ and check.
	n := 2
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var v float64
			for k := 0; k < n; k++ {
				v += evecs.At(i, k) * evals[k] * evecs.At(j, k)
			}
			if math.Abs(v-a.At(i, j)) > 1e-12 {
				t.Errorf("reconstructed A[%d,%d]=%g want %g", i, j, v, a.At(i, j))
			}
		}
	}
}

func TestAxpyDotNrm2(t *testing.T) {
	b := Gonum{}
	x := b.Upload(Vec{1, 2, 2})
	y := b.Upload(Vec{1, 0, 0})
	b.Axpy(2, x, y) // y = [3 4 4]
	if gy := b.Download(y); gy[0] != 3 || gy[1] != 4 || gy[2] != 4 {
		t.Fatalf("Axpy = %v, want [3 4 4]", gy)
	}
	e0 := b.Upload(Vec{1, 0, 0})
	if d := b.Dot(y, e0); d != 3 {
		t.Fatalf("Dot = %v, want 3", d)
	}
	if n := b.Nrm2(b.Upload(Vec{3, 4})); math.Abs(n-5) > 1e-12 {
		t.Fatalf("Nrm2 = %v, want 5", n)
	}
}

// colMajor builds a BlockView over freshly-allocated storage and fills it from a
// row-major reference, so tests can state matrices in the natural reading order.
// ldPad exercises a leading dimension strictly greater than Rows, which is what a
// column panel of a larger basis buffer looks like.
func colMajor(be Backend, rows, cols, ldPad int, rowMajor []float64) BlockView {
	ld := rows + ldPad
	buf := make([]float64, ld*cols)
	for i := range rows {
		for j := range cols {
			buf[j*ld+i] = rowMajor[i*cols+j]
		}
	}
	return BlockView{V: be.Upload(buf), Rows: rows, Cols: cols, Ld: ld}
}

// readBack returns c as a row-major slice.
func readBack(be Backend, c BlockView) []float64 {
	buf := be.Download(c.V)
	out := make([]float64, c.Rows*c.Cols)
	for i := range c.Rows {
		for j := range c.Cols {
			out[i*c.Cols+j] = buf[j*c.Ld+i]
		}
	}
	return out
}

// refGemm is the obvious triple loop over row-major operands, the oracle.
func refGemm(transA, transB bool, alpha float64, a []float64, ar, ac int,
	b []float64, bc int, beta float64, c []float64, cr, cc int) {
	at := func(i, j int) float64 {
		if transA {
			return a[j*ac+i]
		}
		return a[i*ac+j]
	}
	bt := func(i, j int) float64 {
		if transB {
			return b[j*bc+i]
		}
		return b[i*bc+j]
	}
	k := ac
	if transA {
		k = ar
	}
	for i := range cr {
		for j := range cc {
			var acc float64
			for p := range k {
				acc += at(i, p) * bt(p, j)
			}
			c[i*cc+j] = alpha*acc + beta*c[i*cc+j]
		}
	}
}

// TestGemmAgainstReference sweeps shapes, transpose flags, alpha/beta, and a padded
// leading dimension. Non-square, transposed shapes are the point: a Gemm that
// silently transposes an operand still passes on square symmetric inputs.
func TestGemmAgainstReference(t *testing.T) {
	be := Gonum{}
	rng := rand.New(rand.NewSource(7))
	fill := func(n int) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		return v
	}

	shapes := []struct{ m, n, k int }{
		{1, 1, 1}, {3, 4, 5}, {5, 3, 4}, {8, 8, 8}, {17, 5, 11}, {2, 9, 3},
	}
	for _, s := range shapes {
		for _, transA := range []bool{false, true} {
			for _, transB := range []bool{false, true} {
				for _, pad := range []int{0, 3} {
					// op(A) is m×k, op(B) is k×n, C is m×n.
					ar, ac := s.m, s.k
					if transA {
						ar, ac = s.k, s.m
					}
					br, bc := s.k, s.n
					if transB {
						br, bc = s.n, s.k
					}
					aRM, bRM, cRM := fill(ar*ac), fill(br*bc), fill(s.m*s.n)
					want := append([]float64(nil), cRM...)
					const alpha, beta = 0.75, -1.25
					refGemm(transA, transB, alpha, aRM, ar, ac, bRM, bc, beta, want, s.m, s.n)

					A := colMajor(be, ar, ac, pad, aRM)
					B := colMajor(be, br, bc, pad, bRM)
					C := colMajor(be, s.m, s.n, pad, cRM)
					be.Gemm(transA, transB, alpha, A, B, beta, C)
					got := readBack(be, C)

					var maxDiff float64
					for i := range want {
						maxDiff = math.Max(maxDiff, math.Abs(got[i]-want[i]))
					}
					if maxDiff > 1e-12 {
						t.Errorf("m=%d n=%d k=%d tA=%v tB=%v pad=%d: max diff %.3e\n got  %v\n want %v",
							s.m, s.n, s.k, transA, transB, pad, maxDiff, got, want)
					}
				}
			}
		}
	}
}

// TestGemmGramMatrix covers the exact shape block-Lanczos uses: P = Bᵀ·V, where B is
// a tall n×dim basis and V an n×b block. Bᵀ·B must come out symmetric positive
// semi-definite with the right trace.
func TestGemmGramMatrix(t *testing.T) {
	be := Gonum{}
	const n, b = 40, 6
	rng := rand.New(rand.NewSource(11))
	rm := make([]float64, n*b)
	for i := range rm {
		rm[i] = rng.NormFloat64()
	}
	V := colMajor(be, n, b, 0, rm)
	G := BlockView{V: be.Alloc(b * b), Rows: b, Cols: b, Ld: b}
	be.Gemm(true, false, 1, V, V, 0, G) // G = Vᵀ V

	g := readBack(be, G)
	var trace float64
	for i := range b {
		trace += g[i*b+i]
		for j := range b {
			if d := math.Abs(g[i*b+j] - g[j*b+i]); d > 1e-12 {
				t.Fatalf("Gram not symmetric at (%d,%d): %.3e", i, j, d)
			}
		}
	}
	// trace(VᵀV) == ‖V‖_F², computable straight from the source data.
	var frob float64
	for _, x := range rm {
		frob += x * x
	}
	if math.Abs(trace-frob) > 1e-10 {
		t.Errorf("trace(VᵀV)=%.6f, want ‖V‖_F²=%.6f", trace, frob)
	}
}

// TestGonumDownload2DGathersStridedBlock pins StridedDownloader's contract on the reference
// backend: a compact rows×cols column-major copy of the sub-block whose columns sit ld apart.
//
// The Ritz back-transform (lanczos.go) reads the leading `main` rows of every basis column
// through this, so an off-by-one in the stride arithmetic would silently transpose or shift the
// Ritz vectors rather than fail loudly. ld > rows deliberately — equal values would hide exactly
// the indexing bug this guards.
func TestGonumDownload2DGathersStridedBlock(t *testing.T) {
	const (
		ld   = 5 // panel leading dimension (full column height)
		rows = 3 // sub-block height: the leading rows of each column
		cols = 4
	)
	be := Gonum{}

	// Column-major panel: element (r, c) = 100*c + r, so a wrong stride is unmistakable.
	host := make([]float64, ld*cols)
	for c := range cols {
		for r := range ld {
			host[c*ld+r] = float64(100*c + r)
		}
	}
	v := be.Upload(host)

	got := be.Download2D(v, rows, cols, ld)
	if len(got) != rows*cols {
		t.Fatalf("length: got %d, want %d", len(got), rows*cols)
	}
	for c := range cols {
		for r := range rows {
			want := float64(100*c + r)
			if g := got[c*rows+r]; g != want {
				t.Errorf("(row %d, col %d): got %g, want %g", r, c, g, want)
			}
		}
	}

	// The rows beyond the sub-block (r >= rows) must not appear anywhere in the result — the
	// failure mode where a contiguous copy silently pulls in the tail of each column.
	for _, v := range got {
		if r := int(v) % 100; r >= rows {
			t.Errorf("value %g comes from row %d, outside the requested %d rows", v, r, rows)
		}
	}
}

// randSym builds a reproducible dense symmetric n×n matrix.
func randSym(n int, seed int64) Mat {
	rng := rand.New(rand.NewSource(seed))
	m := NewMat(n, n)
	for i := range n {
		for j := i; j < n; j++ {
			v := rng.NormFloat64()
			m.Set(i, j, v)
			m.Set(j, i, v)
		}
	}
	return m
}

// checkEigen asserts the defining property A·v_k = λ_k·v_k and orthonormality of
// the eigenvectors. This validates any implementation on its own terms, without
// assuming a particular sign or ordering convention beyond ascending eigenvalues.
func checkEigen(t *testing.T, name string, a Mat, evals []float64, evecs Mat, tol float64) {
	t.Helper()
	n := a.Rows
	if len(evals) != n {
		t.Fatalf("%s: got %d eigenvalues, want %d", name, len(evals), n)
	}
	for k := 1; k < n; k++ {
		if evals[k] < evals[k-1] {
			t.Fatalf("%s: eigenvalues not ascending at k=%d: %g < %g", name, k, evals[k], evals[k-1])
		}
	}
	// Residual ‖A v_k − λ_k v_k‖_∞ and ‖VᵀV − I‖_∞.
	var maxRes, maxOrth float64
	for k := range n {
		for i := range n {
			var av float64
			for j := range n {
				av += a.At(i, j) * evecs.At(j, k)
			}
			maxRes = math.Max(maxRes, math.Abs(av-evals[k]*evecs.At(i, k)))
		}
		for l := range n {
			var dot float64
			for i := range n {
				dot += evecs.At(i, k) * evecs.At(i, l)
			}
			want := 0.0
			if k == l {
				want = 1
			}
			maxOrth = math.Max(maxOrth, math.Abs(dot-want))
		}
	}
	if maxRes > tol {
		t.Errorf("%s: max |A v - lambda v| = %.3e > %.3e", name, maxRes, tol)
	}
	if maxOrth > tol {
		t.Errorf("%s: max |V^T V - I| = %.3e > %.3e", name, maxOrth, tol)
	}
	t.Logf("%s: n=%d residual=%.3e orthogonality=%.3e", name, n, maxRes, maxOrth)
}

// TestSymEigActive validates whichever implementation this build selected.
func TestSymEigActive(t *testing.T) {
	for _, n := range []int{1, 2, 17, 64} {
		a := randSym(n, int64(n))
		evals, evecs := Gonum{}.SymEig(a)
		checkEigen(t, "active", a, evals, evecs, 1e-10)
	}
}

// TestSymEigMatchesGonum pins any accelerated symEig (e.g. LAPACKE_dsyevd under the
// openblas tag) to the pure-Go reference. Under the default build symEig IS
// symEigGonum and this degenerates to a self-check, which is harmless and keeps the
// assertion in place for whichever build runs it.
//
// Eigenvectors are compared only up to sign, and only where the eigenvalue is
// simple: for a degenerate eigenvalue any orthonormal basis of the eigenspace is a
// valid answer, so the individual vectors need not agree.
func TestSymEigMatchesGonum(t *testing.T) {
	const tol = 1e-11
	for _, n := range []int{2, 17, 64, 128} {
		a := randSym(n, int64(1000+n))
		wantVal, wantVec := symEigGonum(a)
		gotVal, gotVec := symEig(a)

		var maxVal float64
		for k := range n {
			maxVal = math.Max(maxVal, math.Abs(gotVal[k]-wantVal[k]))
		}
		if maxVal > tol {
			t.Errorf("n=%d: max |dlambda| = %.3e > %.3e", n, maxVal, tol)
		}

		var maxVec float64
		for k := range n {
			// Skip near-degenerate eigenvalues: the eigenvector is not unique.
			gap := math.Inf(1)
			if k > 0 {
				gap = math.Min(gap, wantVal[k]-wantVal[k-1])
			}
			if k < n-1 {
				gap = math.Min(gap, wantVal[k+1]-wantVal[k])
			}
			if gap < 1e-6 {
				continue
			}
			// Fix the sign by the largest-magnitude component of the reference.
			pivot, best := 0, 0.0
			for i := range n {
				if v := math.Abs(wantVec.At(i, k)); v > best {
					pivot, best = i, v
				}
			}
			sign := 1.0
			if gotVec.At(pivot, k)*wantVec.At(pivot, k) < 0 {
				sign = -1
			}
			for i := range n {
				maxVec = math.Max(maxVec, math.Abs(sign*gotVec.At(i, k)-wantVec.At(i, k)))
			}
		}
		if maxVec > tol {
			t.Errorf("n=%d: max |dv| (sign-fixed, non-degenerate) = %.3e > %.3e", n, maxVec, tol)
		}
		t.Logf("n=%d: max|dlambda|=%.3e max|dv|=%.3e", n, maxVal, maxVec)
	}
}

// Sizes bracket the real Lanczos working set: sector dimensions run from ~10^3
// (H2O, CH2O) to ~1.8·10^4 (formic acid, cc-pVDZ).
var benchN = []int{1024, 4096, 16384}

// benchBackends returns every backend compiled into this build. Under the default
// build that is just "gonum"; under -tags cuda/hip the accelerated one appears too,
// so the same benchmark body calibrates both and feeds the dispatch cost model.
func benchBackends(tb testing.TB) map[string]Backend {
	out := map[string]Backend{}
	for _, name := range Available() {
		be, err := New(name)
		if err != nil {
			tb.Fatalf("New(%q): %v", name, err)
		}
		out[name] = be
	}
	return out
}

func randSlice(n int) []float64 {
	rng := rand.New(rand.NewSource(int64(n)))
	v := make([]float64, n)
	for i := range v {
		v[i] = rng.NormFloat64()
	}
	return v
}

// BenchmarkDot measures the per-call cost of a BLAS-1 reduction. On a GPU backend
// this is dominated by kernel launch plus the device→host synchronization forced by
// cuBLAS's default host pointer mode — not by the O(n) arithmetic. The Lanczos
// reorthogonalization issues O(dim²) of these, which is the whole problem.
func BenchmarkDot(b *testing.B) {
	for name, be := range benchBackends(b) {
		for _, n := range benchN {
			b.Run(fmt.Sprintf("%s/n=%d", name, n), func(b *testing.B) {
				x, y := be.Upload(randSlice(n)), be.Upload(randSlice(n))
				defer be.Free(x)
				defer be.Free(y)
				b.SetBytes(int64(2 * n * 8))
				b.ResetTimer()
				var acc float64
				for range b.N {
					acc += be.Dot(x, y)
				}
				_ = acc
			})
		}
	}
}

// BenchmarkAxpy is the other half of the modified-Gram-Schmidt inner loop.
func BenchmarkAxpy(b *testing.B) {
	for name, be := range benchBackends(b) {
		for _, n := range benchN {
			b.Run(fmt.Sprintf("%s/n=%d", name, n), func(b *testing.B) {
				x, y := be.Upload(randSlice(n)), be.Upload(randSlice(n))
				defer be.Free(x)
				defer be.Free(y)
				b.SetBytes(int64(3 * n * 8))
				b.ResetTimer()
				for range b.N {
					be.Axpy(1e-12, x, y)
				}
			})
		}
	}
}

// BenchmarkGemv measures the mat-vec kernel behind ApplyFull. It is bandwidth-bound
// (arithmetic intensity ≈ 0.25 flop/byte), so its throughput in B/s is the
// gemv_bytes/s rate the dispatch cost model needs.
func BenchmarkGemv(b *testing.B) {
	// Square blocks sized like the assembled operator's satellite blocks.
	for name, be := range benchBackends(b) {
		for _, n := range []int{256, 1024, 4096} {
			b.Run(fmt.Sprintf("%s/n=%d", name, n), func(b *testing.B) {
				m := Mat{Rows: n, Cols: n, Data: randSlice(n * n)}
				a := be.UploadMat(m)
				x, y := be.Upload(randSlice(n)), be.Alloc(n)
				defer be.Free(x)
				defer be.Free(y)
				b.SetBytes(int64(n * n * 8))
				b.ResetTimer()
				for range b.N {
					be.GemvN(1, a, x, y)
				}
			})
		}
	}
}

// BenchmarkGemm measures the level-3 kernel the rewritten block-Lanczos leans on.
// Shapes mirror the two real calls: the tall-skinny projection P = Bᵀ·V (n×dim by
// n×b) and its back-substitution V -= B·P. Throughput in FLOP/s is the gemm_rate
// the dispatch cost model needs; compare it against BenchmarkDot's per-call latency
// to see why O(dim²) BLAS-1 calls lose to O(dim/b) BLAS-3 calls of the same flops.
func BenchmarkGemm(b *testing.B) {
	cases := []struct{ n, dim, blk int }{
		{4096, 512, 20},
		{16384, 2048, 46},
	}
	for name, be := range benchBackends(b) {
		for _, c := range cases {
			B := BlockView{V: be.Upload(randSlice(c.n * c.dim)), Rows: c.n, Cols: c.dim, Ld: c.n}
			V := BlockView{V: be.Upload(randSlice(c.n * c.blk)), Rows: c.n, Cols: c.blk, Ld: c.n}
			P := BlockView{V: be.Alloc(c.dim * c.blk), Rows: c.dim, Cols: c.blk, Ld: c.dim}

			// cublasDgemm is asynchronous: it enqueues and returns. Timing the loop
			// alone measures launch overhead, not execution, and reports throughput
			// above the card's FP64 peak. Nrm2 uses host pointer mode, so it forces a
			// full device sync; call it inside the timed region to drain the queue.
			// On host backends it is a negligible O(n) pass.
			sync := func(v Vector) { _ = be.Nrm2(v) }

			b.Run(fmt.Sprintf("%s/proj/n=%d,dim=%d,b=%d", name, c.n, c.dim, c.blk), func(b *testing.B) {
				b.SetBytes(int64(2 * c.n * c.dim * c.blk)) // flops, reported as "bytes"
				b.ResetTimer()
				for range b.N {
					be.Gemm(true, false, 1, B, V, 0, P) // P = Bᵀ V
				}
				sync(P.V)
			})
			b.Run(fmt.Sprintf("%s/back/n=%d,dim=%d,b=%d", name, c.n, c.dim, c.blk), func(b *testing.B) {
				b.SetBytes(int64(2 * c.n * c.dim * c.blk))
				b.ResetTimer()
				for range b.N {
					be.Gemm(false, false, -1, B, P, 1, V) // V -= B P
				}
				sync(V.V)
			})
			be.Free(B.V)
			be.Free(V.V)
			be.Free(P.V)
		}
	}
}

// BenchmarkSymEig sizes the projected-matrix diagonalization. dim reaches ~11600 for
// formic acid at -blocks 200, where this is O(dim³) and becomes the bottleneck once
// the BLAS-1 phases are fixed. Kept small by default; run with -benchtime=1x.
func BenchmarkSymEig(b *testing.B) {
	for name, be := range benchBackends(b) {
		for _, n := range []int{256, 1024} {
			b.Run(fmt.Sprintf("%s/n=%d", name, n), func(b *testing.B) {
				m := NewMat(n, n)
				src := randSlice(n * n)
				for i := range n { // symmetrize
					for j := i; j < n; j++ {
						v := src[i*n+j]
						m.Set(i, j, v)
						m.Set(j, i, v)
					}
				}
				b.ResetTimer()
				for range b.N {
					be.SymEig(m)
				}
			})
		}
	}
}
