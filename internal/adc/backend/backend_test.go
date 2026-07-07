package backend

import (
	"math"
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
