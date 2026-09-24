//go:build hip || cuda

package backend

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"testing"
)

// gpuUnderTest returns the accelerated backend compiled into this build (the one
// registered besides "gonum").
func gpuUnderTest(t *testing.T) (Backend, string) {
	t.Helper()
	for _, name := range Available() {
		if name == "gonum" {
			continue
		}
		be, err := New(name)
		if err != nil {
			t.Fatalf("New(%q): %v", name, err)
		}
		return be, name
	}
	t.Skip("no accelerated backend registered in this build")
	return nil, ""
}

func randVec(n int) Vec {
	v := make([]float64, n)
	for i := range v {
		v[i] = rand.NormFloat64()
	}
	return v
}

const gpuTol = 1e-11

func maxAbsDiff(a, b Vec) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

// TestGPUAgreesWithGonum is the M3 op-level gate: every BLAS-1/2 kernel of the
// accelerated backend must reproduce the pure-Go Gonum reference to ~1e-11.
func TestGPUAgreesWithGonum(t *testing.T) {
	gpu, name := gpuUnderTest(t)
	ref := Gonum{}
	const n = 37

	x, y := randVec(n), randVec(n)

	// Axpy.
	gx, gy := gpu.Upload(x), gpu.Upload(y)
	rx, ry := ref.Upload(x), ref.Upload(y)
	gpu.Axpy(2.5, gx, gy)
	ref.Axpy(2.5, rx, ry)
	if d := maxAbsDiff(gpu.Download(gy), ref.Download(ry)); d > gpuTol {
		t.Errorf("%s Axpy differs by %g", name, d)
	}

	// Dot / Nrm2 (fresh operands).
	gx, gy = gpu.Upload(x), gpu.Upload(y)
	if d := math.Abs(gpu.Dot(gx, gy) - ref.Dot(ref.Upload(x), ref.Upload(y))); d > gpuTol {
		t.Errorf("%s Dot differs by %g", name, d)
	}
	if d := math.Abs(gpu.Nrm2(gx) - ref.Nrm2(ref.Upload(x))); d > gpuTol {
		t.Errorf("%s Nrm2 differs by %g", name, d)
	}

	// Scal.
	gx = gpu.Upload(x)
	rx = ref.Upload(x)
	gpu.Scal(-1.75, gx)
	ref.Scal(-1.75, rx)
	if d := maxAbsDiff(gpu.Download(gx), ref.Download(rx)); d > gpuTol {
		t.Errorf("%s Scal differs by %g", name, d)
	}

	// GemvN and GemvT on a non-square block (rows != cols).
	const rows, cols = 5, 8
	A := Mat{Rows: rows, Cols: cols, Data: randVec(rows * cols)}
	ga, ra := gpu.UploadMat(A), ref.UploadMat(A)

	xc := randVec(cols)
	gy2, ry2 := gpu.Alloc(rows), ref.Alloc(rows)
	gpu.GemvN(1, ga, gpu.Upload(xc), gy2)
	ref.GemvN(1, ra, ref.Upload(xc), ry2)
	if d := maxAbsDiff(gpu.Download(gy2), ref.Download(ry2)); d > gpuTol {
		t.Errorf("%s GemvN differs by %g", name, d)
	}

	xr := randVec(rows)
	gy3, ry3 := gpu.Alloc(cols), ref.Alloc(cols)
	gpu.GemvT(1, ga, gpu.Upload(xr), gy3)
	ref.GemvT(1, ra, ref.Upload(xr), ry3)
	if d := maxAbsDiff(gpu.Download(gy3), ref.Download(ry3)); d > gpuTol {
		t.Errorf("%s GemvT differs by %g", name, d)
	}
}

// TestGPUSliceView checks that a GemvN into a Slice view writes the correct
// sub-range of a resident vector (the block-offset mechanism of the mat-vec).
func TestGPUSliceView(t *testing.T) {
	gpu, name := gpuUnderTest(t)
	const rows, cols = 3, 4
	A := Mat{Rows: rows, Cols: cols, Data: randVec(rows * cols)}
	ga := gpu.UploadMat(A)

	out := gpu.Alloc(10)
	xc := randVec(cols)
	gpu.GemvN(1, ga, gpu.Upload(xc), out.Slice(4, rows)) // write rows [4,7)

	got := gpu.Download(out)
	want := Mat{Rows: rows, Cols: cols, Data: A.Data}.MulVec(xc)
	for i := range 10 {
		exp := 0.0
		if i >= 4 && i < 4+rows {
			exp = want[i-4]
		}
		if math.Abs(got[i]-exp) > gpuTol {
			t.Fatalf("%s slice-view GemvN[%d] = %g, want %g", name, i, got[i], exp)
		}
	}
}

// TestGPUGemmAgreesWithGonum pins the device GEMM to the pure-Go reference across
// non-square shapes, both transpose flags, and a padded leading dimension.
//
// This exists because of a specific trap. GemvN/GemvT (gpu_device.go) invert the
// transpose flag to compensate for operator blocks that were uploaded row-major and
// are read column-major by the vendor BLAS. A BlockView, by contrast, is already
// column-major, so Gemm must NOT invert. A Gemm that wrongly inverted would still
// pass on square symmetric operands with alpha=1, beta=0 — hence the shape sweep,
// the asymmetric alpha/beta, and the ld padding.
func TestGPUGemmAgreesWithGonum(t *testing.T) {
	gpu, name := gpuUnderTest(t)
	cpu := Gonum{}
	rng := rand.New(rand.NewSource(23))

	fill := func(n int) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		return v
	}
	// upload builds the same column-major panel on a given backend.
	upload := func(be Backend, rows, cols, ld int, colMajorData []float64) BlockView {
		return BlockView{V: be.Upload(colMajorData), Rows: rows, Cols: cols, Ld: ld}
	}

	shapes := []struct{ m, n, k int }{
		{1, 1, 1}, {3, 4, 5}, {5, 3, 4}, {17, 5, 11}, {64, 8, 33}, {2, 9, 3},
	}
	const alpha, beta = 0.75, -1.25

	for _, s := range shapes {
		for _, transA := range []bool{false, true} {
			for _, transB := range []bool{false, true} {
				for _, pad := range []int{0, 2} {
					ar, ac := s.m, s.k
					if transA {
						ar, ac = s.k, s.m
					}
					br, bc := s.k, s.n
					if transB {
						br, bc = s.n, s.k
					}
					lda, ldb, ldc := ar+pad, br+pad, s.m+pad
					aData, bData := fill(lda*ac), fill(ldb*bc)
					cData := fill(ldc * s.n)

					cA := upload(cpu, ar, ac, lda, aData)
					cB := upload(cpu, br, bc, ldb, bData)
					cC := upload(cpu, s.m, s.n, ldc, cData)
					cpu.Gemm(transA, transB, alpha, cA, cB, beta, cC)

					gA := upload(gpu, ar, ac, lda, aData)
					gB := upload(gpu, br, bc, ldb, bData)
					gC := upload(gpu, s.m, s.n, ldc, cData)
					gpu.Gemm(transA, transB, alpha, gA, gB, beta, gC)

					want, got := cpu.Download(cC.V), gpu.Download(gC.V)
					// Compare only the live entries; the ld padding is scratch.
					var maxDiff float64
					for j := range s.n {
						for i := range s.m {
							d := math.Abs(got[j*ldc+i] - want[j*ldc+i])
							maxDiff = math.Max(maxDiff, d)
						}
					}
					if maxDiff > gpuTol {
						t.Errorf("%s Gemm m=%d n=%d k=%d tA=%v tB=%v pad=%d: max diff %.3e",
							name, s.m, s.n, s.k, transA, transB, pad, maxDiff)
					}
					gpu.Free(gA.V)
					gpu.Free(gB.V)
					gpu.Free(gC.V)
				}
			}
		}
	}
}

// TestGPUFreeMat exercises the UploadMat/FreeMat pair. Before FreeMat existed every
// uploaded operator block leaked for the process lifetime; a sector loop that
// re-assembles per sector would exhaust an 8 GB card. Allocating and freeing far
// more than device memory would hold proves the release path works.
func TestGPUFreeMat(t *testing.T) {
	gpu, name := gpuUnderTest(t)
	const side = 512 // 2 MiB per block
	m := Mat{Rows: side, Cols: side, Data: make([]float64, side*side)}
	for i := 0; i < 8000; i++ { // 16 GB cumulative, on an 8 GB card
		dm := gpu.UploadMat(m)
		gpu.FreeMat(dm)
	}
	t.Logf("%s: 8000 x 2 MiB upload/free cycles (16 GB cumulative) completed", name)
}

// TestGemmLeadingDimensionCrossesInt32 probes the one scale-dependent hazard the mgpu parity tests
// structurally cannot reach: a GEMM whose OUTPUT panel spans more than 2^31 elements because its
// leading dimension is a device row band, not because any single dimension is large.
//
// Why this shape and not a big square: distBackend.gemmMatOne applies the operator one block at a
// time, so m (the block's rows) and k stay small — but C is a view into the device's whole row band,
// so ldc is the partition height and the far column sits at (cols-1)*ldc. For the production DIP sector
// that killed job 14040960 (n=10,014,483 over 8 partitions, panel width 1711) that offset is
//
//	1710 * 1,251,810 + m  ~=  2.1418e9   vs   INT32_MAX = 2.1475e9
//
// i.e. 0.26% of headroom. The mgpu tests run at n ~= 200, four orders of magnitude below any of
// this, so they cannot say whether the vendor BLAS (or our wrapper) computes that offset in 32 or
// 64 bits. Job 14040960 died with cudaErrorLaunchFailure (719) reported at a later
// cudaDeviceSynchronize, which is exactly how a wrapped-index out-of-bounds store would surface.
//
// The parameters below deliberately STRADDLE the boundary: with ld = 1,300,000, column j starts at
// j*ld, which crosses 2^31 between j=1651 and j=1652. The check therefore reads a column below the
// boundary, the first column above it, and the last column. A 32-bit offset anywhere wraps negative
// and either faults or writes somewhere else, so both outcomes fail this test rather than passing
// quietly.
//
// Cost: ld*cols*8 = 17.8 GB of device memory, allocated (not uploaded) and never copied to the
// host in full — only three m-element columns come back. Skips when the device cannot spare it.
func TestGemmLeadingDimensionCrossesInt32(t *testing.T) {
	gpu, name := gpuUnderTest(t)

	const (
		ld   = 1_300_000 // device row band (production n/8 rounded up); the leading dimension of C
		cols = 1711      // the production system's panel width
		m    = 64        // one operator block's rows: small, as in gemmMatOne
		k    = 1
	)
	const total = uint64(ld) * uint64(cols) // 2.2243e9 elements
	if total <= 1<<31 {
		t.Fatalf("test misconfigured: %d elements does not cross 2^31", total)
	}
	need := total*elemSize + (1 << 30) // C plus slack for A, B and the context

	if dev, ok := gpu.(*gpuBackend); ok {
		var free uint64
		dev.do(func() { free, _ = devMemInfo() })
		if free < need {
			t.Skipf("%s: needs %.1f GB free, have %.1f GB", name, float64(need)/(1<<30), float64(free)/(1<<30))
		}
	}

	// C = A*B with A = ones(m x 1) and B = [1, 2, ... cols], so column j must hold j+1 in every
	// one of its m rows. A wrong-column write is therefore detected by VALUE, not just by a fault.
	aData := make([]float64, m*k)
	for i := range aData {
		aData[i] = 1
	}
	bData := make([]float64, k*cols)
	for j := range bData {
		bData[j] = float64(j + 1)
	}

	av := gpu.Upload(aData)
	defer gpu.Free(av)
	bv := gpu.Upload(bData)
	defer gpu.Free(bv)

	cv := gpu.Alloc(int(total)) // Alloc zeroes, so an un-written column reads back 0
	defer gpu.Free(cv)

	a := BlockView{V: av, Rows: m, Cols: k, Ld: m}
	b := BlockView{V: bv, Rows: k, Cols: cols, Ld: k}
	c := BlockView{V: cv, Rows: m, Cols: cols, Ld: ld}

	gpu.Gemm(false, false, 1, a, b, 0, c)

	// Below the boundary, the first column past it, and the last column.
	for _, j := range []int{0, 1651, 1652, cols - 1} {
		off := uint64(j) * uint64(ld)
		got := gpu.Download(cv.Slice(int(off), m))
		want := float64(j + 1)
		for i, v := range got {
			if v != want {
				t.Fatalf("%s: C[%d,%d] = %g, want %g (column offset %d, %s 2^31) — "+
					"the GEMM output addressing is not 64-bit clean at this leading dimension",
					name, i, j, v, want, off, map[bool]string{true: "above", false: "below"}[off >= 1<<31])
			}
		}
	}
}

// TestGPUHandleThreadAffinity is a regression test for a silent-corruption bug.
//
// A cuBLAS handle belongs to the OS thread (and device context) that created it.
// Goroutines migrate between OS threads, so a backend that called cuBLAS directly
// from whatever goroutine happened to invoke it would intermittently get
// CUBLAS_STATUS_INTERNAL_ERROR (14) with cudaError_t 0 — and the original shim
// discarded every status, so the GEMM silently did nothing. It surfaced only as a
// benchmark reporting 828 GFLOP/s on a card whose FP64 peak is 253.
//
// gpuBackend now funnels all device work through one dedicated locked thread. This
// test drives it from many goroutines at once, each pinned to its own OS thread, and
// checks both that nothing errors and that the arithmetic is right.
func TestGPUHandleThreadAffinity(t *testing.T) {
	gpu, name := gpuUnderTest(t)
	const n, dim, blk = 512, 64, 8
	const workers = 8

	var wg sync.WaitGroup
	errs := make(chan any, workers)
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			defer func() {
				if r := recover(); r != nil {
					errs <- r
				}
			}()
			// Each worker owns its own buffers; only the handle is shared.
			bData := make([]float64, n*dim)
			for i := range bData {
				bData[i] = float64((i + w) % 7)
			}
			vData := make([]float64, n*blk)
			for i := range vData {
				vData[i] = float64((i * 3) % 5)
			}
			B := BlockView{V: gpu.Upload(bData), Rows: n, Cols: dim, Ld: n}
			V := BlockView{V: gpu.Upload(vData), Rows: n, Cols: blk, Ld: n}
			P := BlockView{V: gpu.Alloc(dim * blk), Rows: dim, Cols: blk, Ld: dim}
			defer gpu.Free(B.V)
			defer gpu.Free(V.V)
			defer gpu.Free(P.V)

			for range 20 {
				gpu.Gemm(true, false, 1, B, V, 0, P) // P = Bᵀ V
				_ = gpu.Nrm2(P.V)
				gpu.Gemm(false, false, -1, B, P, 1, V) // V -= B P
			}

			// Cross-check the last projection against the CPU backend.
			cpu := Gonum{}
			cB := BlockView{V: cpu.Upload(gpu.Download(B.V)), Rows: n, Cols: dim, Ld: n}
			cV := BlockView{V: cpu.Upload(gpu.Download(V.V)), Rows: n, Cols: blk, Ld: n}
			cP := BlockView{V: cpu.Alloc(dim * blk), Rows: dim, Cols: blk, Ld: dim}
			cpu.Gemm(true, false, 1, cB, cV, 0, cP)
			gpu.Gemm(true, false, 1, B, V, 0, P)

			want, got := cpu.Download(cP.V), gpu.Download(P.V)
			var scale, diff float64
			for i := range want {
				scale = math.Max(scale, math.Abs(want[i]))
				diff = math.Max(diff, math.Abs(want[i]-got[i]))
			}
			if rel := diff / math.Max(scale, 1); rel > 1e-12 {
				panic(fmt.Sprintf("gpu/cpu Gemm disagree: relative %.3e", rel))
			}
		}()
	}
	wg.Wait()
	close(errs)
	for r := range errs {
		t.Fatalf("%s: worker failed: %v", name, r)
	}
	t.Logf("%s: %d pinned goroutines x 20 Gemm/Nrm2 rounds, results match CPU", name, workers)
}

// TestGPUGemmRepeated mimics the benchmark's call sequence on one goroutine:
// N projections, a sync, N back-substitutions, a sync.
func TestGPUGemmRepeated(t *testing.T) {
	gpu, _ := gpuUnderTest(t)
	const n, dim, blk = 4096, 512, 20
	B := BlockView{V: gpu.Upload(make([]float64, n*dim)), Rows: n, Cols: dim, Ld: n}
	V := BlockView{V: gpu.Upload(make([]float64, n*blk)), Rows: n, Cols: blk, Ld: n}
	P := BlockView{V: gpu.Alloc(dim * blk), Rows: dim, Cols: blk, Ld: dim}
	defer gpu.Free(B.V)
	defer gpu.Free(V.V)
	defer gpu.Free(P.V)

	for range 3 {
		for range 5 {
			gpu.Gemm(true, false, 1, B, V, 0, P)
		}
		_ = gpu.Nrm2(P.V)
		for range 5 {
			gpu.Gemm(false, false, -1, B, P, 1, V)
		}
		_ = gpu.Nrm2(V.V)
	}
}

// TestGPUSymEigMatchesHost drives the device eigensolver above gpuSymEigMin and checks
// it against the host reference.
//
// The device leaves eigenvectors as columns of a column-major matrix; read back into
// row-major storage that is the transpose. If the in-place transpose were dropped, the
// eigenvalues would still be right and only the eigenvectors wrong — which the pole
// strengths depend on, and which no eigenvalue check would catch. So this asserts
// A·v_k = λ_k·v_k directly.
func TestGPUSymEigMatchesHost(t *testing.T) {
	gpu, name := gpuUnderTest(t)
	const n = gpuSymEigMin + 64 // just over the threshold, so the device path is taken

	rng := rand.New(rand.NewSource(5))
	a := NewMat(n, n)
	for i := range n {
		for j := i; j < n; j++ {
			v := rng.NormFloat64()
			a.Set(i, j, v)
			a.Set(j, i, v)
		}
	}

	gotVal, gotVec := gpu.SymEig(a)
	if len(gotVal) != n {
		t.Fatalf("%s: got %d eigenvalues, want %d", name, len(gotVal), n)
	}
	for k := 1; k < n; k++ {
		if gotVal[k] < gotVal[k-1] {
			t.Fatalf("%s: eigenvalues not ascending at %d", name, k)
		}
	}

	// Residual ‖A v_k − λ_k v_k‖_inf on a sample of eigenpairs (a full check is O(n³)).
	var maxRes, maxOrth float64
	for _, k := range []int{0, 1, n / 3, n / 2, n - 2, n - 1} {
		for i := range n {
			var av float64
			for j := range n {
				av += a.At(i, j) * gotVec.At(j, k)
			}
			maxRes = math.Max(maxRes, math.Abs(av-gotVal[k]*gotVec.At(i, k)))
		}
		var nrm float64
		for i := range n {
			nrm += gotVec.At(i, k) * gotVec.At(i, k)
		}
		maxOrth = math.Max(maxOrth, math.Abs(nrm-1))
	}
	// Scale: ‖A‖ grows like sqrt(n) for a random symmetric matrix.
	tol := 1e-9 * math.Sqrt(float64(n))
	if maxRes > tol {
		t.Errorf("%s: max |A v - lambda v| = %.3e > %.3e", name, maxRes, tol)
	}
	if maxOrth > 1e-10 {
		t.Errorf("%s: eigenvector norm deviates by %.3e", name, maxOrth)
	}

	// Eigenvalues must match the host solver.
	wantVal, _ := Gonum{}.SymEig(a)
	var maxVal float64
	for k := range n {
		maxVal = math.Max(maxVal, math.Abs(gotVal[k]-wantVal[k]))
	}
	if maxVal > 1e-9 {
		t.Errorf("%s: max |dlambda| vs host = %.3e", name, maxVal)
	}
	t.Logf("%s: n=%d residual=%.3e orth=%.3e max|dlambda|=%.3e", name, n, maxRes, maxOrth, maxVal)
}

// TestGPUSymEigFallsBackBelowThreshold: small matrices must not touch the device, and
// must still be correct.
func TestGPUSymEigFallsBackBelowThreshold(t *testing.T) {
	gpu, name := gpuUnderTest(t)
	const n = 64
	a := randSym(n, 9)
	gotVal, gotVec := gpu.SymEig(a)
	wantVal, _ := Gonum{}.SymEig(a)
	for k := range n {
		if math.Abs(gotVal[k]-wantVal[k]) > 1e-12 {
			t.Fatalf("%s: small-matrix path diverged at k=%d", name, k)
		}
	}
	checkEigen(t, name+" (host fallback)", a, gotVal, gotVec, 1e-10)
}
