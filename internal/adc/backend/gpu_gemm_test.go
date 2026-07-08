//go:build hip || cuda

package backend

import (
	"math"
	"math/rand"
	"testing"
)

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
