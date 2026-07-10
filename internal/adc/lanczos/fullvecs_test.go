package lanczos

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
)

// col extracts Ritz vector k from a row-major (rows × states) matrix.
func col(m backend.Mat, k int) []float64 {
	v := make([]float64, m.Rows)
	for r := range m.Rows {
		v[r] = m.At(r, k)
	}
	return v
}

// residualNorm is ‖M·v − θ·v‖, evaluated through the operator itself. It is the only
// check that reads the satellite rows of v: the main-block rows alone cannot make it
// vanish.
func residualNorm(mx *dip.Matrix, be backend.Backend, v []float64, theta float64) float64 {
	in := be.Upload(v)
	out := be.Alloc(len(v))
	defer be.Free(in)
	defer be.Free(out)
	mx.ApplyFull(out, in)
	o := be.Download(out)
	var acc float64
	for i := range o {
		d := o[i] - theta*v[i]
		acc += d * d
	}
	return math.Sqrt(acc)
}

// sampleStates picks up to n state indices spread across [0, states).
func sampleStates(states, n int) []int {
	if states <= n {
		out := make([]int, states)
		for i := range states {
			out[i] = i
		}
		return out
	}
	out := make([]int, n)
	for i := range n {
		out[i] = i * (states - 1) / (n - 1)
	}
	return out
}

// TestSolveDenseFullVecsAreEigenvectors: the dense path hands back the eigenvectors it
// already computed, so M·y = θ·y to machine precision on every row.
func TestSolveDenseFullVecsAreEigenvectors(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping dense eigenvector test in -short mode")
	}
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Triplet)
	res := SolveDense(mx, be)
	if !res.HasFull() {
		t.Fatal("SolveDense must always retain FullVecs")
	}
	if res.FullVecs.Rows != mx.Size() || res.FullVecs.Cols != len(res.Values) {
		t.Fatalf("FullVecs is %d×%d, want %d×%d",
			res.FullVecs.Rows, res.FullVecs.Cols, mx.Size(), len(res.Values))
	}
	for _, k := range sampleStates(len(res.Values), 12) {
		if r := residualNorm(mx, be, col(res.FullVecs, k), res.Values[k]); r > 1e-10 {
			t.Errorf("state %d: ‖M y − θ y‖ = %g", k, r)
		}
	}
}

// TestFullVecsSatisfyRitzResidual is the load-bearing check on the back-transform: the
// true residual computed from the reconstructed vector must reproduce the Ritz residual
// Solve derives from R_{j+1}·s_k, which it obtains without ever forming the vector. A
// transposed s, a wrong chunk offset, or a mis-strided download all break this.
func TestFullVecsSatisfyRitzResidual(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full-vector residual test in -short mode")
	}
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Triplet)
	res := Solve(mx, be, Options{MaxBlocks: 8, WantFull: true})
	if !res.HasFull() {
		t.Fatal("WantFull did not retain FullVecs")
	}
	if res.FullVecs.Rows != mx.Size() || res.FullVecs.Cols != len(res.Values) {
		t.Fatalf("FullVecs is %d×%d, want %d×%d",
			res.FullVecs.Rows, res.FullVecs.Cols, mx.Size(), len(res.Values))
	}
	for _, k := range sampleStates(len(res.Values), 12) {
		got := residualNorm(mx, be, col(res.FullVecs, k), res.Values[k])
		want := res.Residual[k]
		if math.Abs(got-want) > 1e-7+1e-5*want {
			t.Errorf("state %d: true residual %g, Ritz residual %g", k, got, want)
		}
	}
}

// TestFullVecsLeadingRowsMatchMainVecs: the two back-transforms are the same product
// B·S evaluated in different places (device GEMM vs host MatMul), so they agree to
// rounding, not to the bit.
func TestFullVecsLeadingRowsMatchMainVecs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Triplet)
	res := Solve(mx, be, Options{MaxBlocks: 8, WantFull: true})
	main := mx.MainBlockSize()
	var maxErr float64
	for k := range res.Values {
		for c := range main {
			if d := math.Abs(res.FullVecs.At(c, k) - res.MainVecs.At(c, k)); d > maxErr {
				maxErr = d
			}
		}
	}
	if maxErr > 1e-12 {
		t.Errorf("FullVecs main rows differ from MainVecs by %g", maxErr)
	}
}

// TestFullVecsOrthonormal: B is orthonormal and S orthogonal, so B·S is orthonormal.
func TestFullVecsOrthonormal(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Triplet)
	res := Solve(mx, be, Options{MaxBlocks: 6, WantFull: true})
	fv := res.FullVecs
	var maxErr float64
	for i := range fv.Cols {
		for j := i; j < fv.Cols; j++ {
			var acc float64
			for r := range fv.Rows {
				acc += fv.At(r, i) * fv.At(r, j)
			}
			want := 0.0
			if i == j {
				want = 1.0
			}
			if d := math.Abs(acc - want); d > maxErr {
				maxErr = d
			}
		}
	}
	if maxErr > 1e-10 {
		t.Errorf("‖FullVecsᵀ·FullVecs − I‖_max = %g", maxErr)
	}
}

// TestWantFullDoesNotPerturbResults: WantFull is additive. Everything the spectrum is
// built from must come back bit-for-bit identical, or every validated DIP/SIP number
// silently depends on whether transition moments were requested.
func TestWantFullDoesNotPerturbResults(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Triplet)
	opts := Options{MaxBlocks: 6}
	plain := Solve(mx, be, opts)
	opts.WantFull = true
	full := Solve(mx, be, opts)

	if plain.HasFull() {
		t.Error("FullVecs retained without WantFull")
	}
	same := func(name string, a, b []float64) {
		if len(a) != len(b) {
			t.Fatalf("%s: length %d vs %d", name, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%s[%d]: %v vs %v (not bit-identical)", name, i, a[i], b[i])
			}
		}
	}
	same("Values", plain.Values, full.Values)
	same("PS", plain.PS, full.PS)
	same("Residual", plain.Residual, full.Residual)
	same("MainVecs", plain.MainVecs.Data, full.MainVecs.Data)
}
