package backend

import (
	"math"
	"math/rand"
	"testing"
)

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
