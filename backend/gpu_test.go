//go:build hip || cuda

package backend

import (
	"math"
	"math/rand"
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
