package backend_test

import (
	"fmt"
	"math/rand"
	"testing"

	"adcgo/internal/adc/backend"
)

// Sizes bracket the real Lanczos working set: sector dimensions run from ~10^3
// (H2O, CH2O) to ~1.8·10^4 (formic acid, cc-pVDZ).
var benchN = []int{1024, 4096, 16384}

// benchBackends returns every backend compiled into this build. Under the default
// build that is just "gonum"; under -tags cuda/hip the accelerated one appears too,
// so the same benchmark body calibrates both and feeds the dispatch cost model.
func benchBackends(tb testing.TB) map[string]backend.Backend {
	out := map[string]backend.Backend{}
	for _, name := range backend.Available() {
		be, err := backend.New(name)
		if err != nil {
			tb.Fatalf("backend.New(%q): %v", name, err)
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
				m := backend.Mat{Rows: n, Cols: n, Data: randSlice(n * n)}
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
			B := backend.BlockView{V: be.Upload(randSlice(c.n * c.dim)), Rows: c.n, Cols: c.dim, Ld: c.n}
			V := backend.BlockView{V: be.Upload(randSlice(c.n * c.blk)), Rows: c.n, Cols: c.blk, Ld: c.n}
			P := backend.BlockView{V: be.Alloc(c.dim * c.blk), Rows: c.dim, Cols: c.blk, Ld: c.dim}

			// cublasDgemm is asynchronous: it enqueues and returns. Timing the loop
			// alone measures launch overhead, not execution, and reports throughput
			// above the card's FP64 peak. Nrm2 uses host pointer mode, so it forces a
			// full device sync; call it inside the timed region to drain the queue.
			// On host backends it is a negligible O(n) pass.
			sync := func(v backend.Vector) { _ = be.Nrm2(v) }

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
				m := backend.NewMat(n, n)
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
