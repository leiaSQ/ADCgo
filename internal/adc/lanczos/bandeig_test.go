package lanczos

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// buildBanded returns a random real-symmetric matrix of dimension dim with half-bandwidth
// band (entries beyond the band are zero), both as a dense backend.Mat (for the dense
// oracle) and as bandStorage (for bandSymDiagFast). The bandStorage entry (col=j, i) is the
// coupling between indices j and j+i, i.e. the lower-triangle element M[j+i][j].
func buildBanded(dim, band int, seed int64) (backend.Mat, bandStorage) {
	rng := rand.New(rand.NewSource(seed))
	M := backend.NewMat(dim, dim)
	bs := newBandStorage(dim, band)
	for j := 0; j < dim; j++ {
		for i := 0; i <= band && j+i < dim; i++ {
			v := rng.NormFloat64()
			bs.set(j, i, v)
			M.Set(j+i, j, v)
			M.Set(j, j+i, v)
		}
	}
	return M, bs
}

// TestBandSymDiagFastEigenvalues checks the ported bnd2td/tddiag eigenvalues against the
// dense SymEig oracle across several bandwidths, including the degenerate band==0 and
// band==1 control-flow branches.
func TestBandSymDiagFastEigenvalues(t *testing.T) {
	be := backend.Gonum{}
	for _, tc := range []struct{ dim, band int }{
		{1, 0}, {8, 0}, {8, 1}, {12, 2}, {30, 5}, {40, 7}, {50, 12},
	} {
		M, bs := buildBanded(tc.dim, tc.band, int64(1000+tc.dim*100+tc.band))
		want, _ := be.SymEig(M) // ascending
		got, _ := bandSymDiagFast(bs)
		if len(got) != tc.dim {
			t.Fatalf("dim=%d band=%d: got %d evals, want %d", tc.dim, tc.band, len(got), tc.dim)
		}
		for k := range want {
			if math.Abs(got[k]-want[k]) > 1e-9 {
				t.Errorf("dim=%d band=%d: eval[%d]=%.12f, want %.12f (Δ%.2e)",
					tc.dim, tc.band, k, got[k], want[k], got[k]-want[k])
			}
		}
	}
}

// TestBandSymDiagFastPartialVectors checks that the 2*band-row partial eigenvectors match
// the top and bottom band rows of the dense eigenvectors (up to a per-eigenvector sign),
// which is the property Mode B relies on to read pole strengths without the basis.
func TestBandSymDiagFastPartialVectors(t *testing.T) {
	be := backend.Gonum{}
	const dim, band = 36, 6
	M, bs := buildBanded(dim, band, 424242)
	evalsD, evecsD := be.SymEig(M)
	evals, z := bandSymDiagFast(bs)
	nm := 2 * band

	for k := 0; k < dim; k++ {
		// Skip near-degenerate eigenvalues: their eigenvectors are only defined up to a
		// rotation within the degenerate subspace, so a row-by-row match is not meaningful.
		degenerate := false
		for j := 0; j < dim; j++ {
			if j != k && math.Abs(evalsD[j]-evalsD[k]) < 1e-6 {
				degenerate = true
				break
			}
		}
		if degenerate {
			continue
		}
		// Reference top/bottom slices from the dense eigenvector (column k).
		ref := make([]float64, nm)
		for r := 0; r < band; r++ {
			ref[r] = evecsD.At(r, k)
			ref[band+r] = evecsD.At(dim-band+r, k)
		}
		got := z[k*nm : (k+1)*nm]
		// Fix the global sign by the largest-magnitude reference component.
		pivot := 0
		for r := 1; r < nm; r++ {
			if math.Abs(ref[r]) > math.Abs(ref[pivot]) {
				pivot = r
			}
		}
		sign := 1.0
		if ref[pivot]*got[pivot] < 0 {
			sign = -1.0
		}
		for r := 0; r < nm; r++ {
			if math.Abs(sign*got[r]-ref[r]) > 1e-7 {
				t.Errorf("eval %d (%.6f): partial vec row %d = %.9f, want %.9f",
					k, evals[k], r, sign*got[r], ref[r])
			}
		}
	}
}
