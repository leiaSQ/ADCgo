//go:build cuda || hip

package backend

import (
	"math"
	"math/rand"
	"testing"
)

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
