//go:build cuda || hip

package backend

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
)

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
