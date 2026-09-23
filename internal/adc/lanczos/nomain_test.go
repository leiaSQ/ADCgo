package lanczos

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
)

// nomain_test.go — a subspace with NO main-class configuration.
//
// This is not a degenerate corner to be defended against; it is the paper's Mg+(2s^-1)
// continuum subspace. Its Q/P rule makes every 1h configuration bound (2s^-1, 3s^-1 and
// 2p^-1 are all bound states there), so P holds only 2h1p and 3h2p rows and
// MainBlockSize() is 0. Solve used to return an empty Result in that case, and the run
// then failed downstream in fano.Widths with "the PMP solve did not retain full Ritz
// vectors" — a message that names the symptom and not the cause.

// tridiagOp is a symmetric operator with a prescribed spectrum and a settable main block,
// so the same matrix can be solved with and without a main class.
type tridiagOp struct {
	n, main int
	d, e    []float64 // diagonal, off-diagonal
}

func newTridiagOp(n, main int) *tridiagOp {
	o := &tridiagOp{n: n, main: main, d: make([]float64, n), e: make([]float64, n)}
	for i := range n {
		o.d[i] = 1 + float64(i)           // ascending, well separated
		o.e[i] = 0.25 + 0.01*float64(i%7) // couples every row to the next
	}
	o.e[n-1] = 0
	return o
}

func (o *tridiagOp) Size() int          { return o.n }
func (o *tridiagOp) MainBlockSize() int { return o.main }

func (o *tridiagOp) apply(out, in []float64) {
	for i := range o.n {
		v := o.d[i] * in[i]
		if i > 0 {
			v += o.e[i-1] * in[i-1]
		}
		if i < o.n-1 {
			v += o.e[i] * in[i+1]
		}
		out[i] = v
	}
}

func (o *tridiagOp) ApplyFull(out, in backend.Vector) {
	be := backend.Gonum{}
	x := be.Download(in)
	y := make([]float64, o.n)
	o.apply(y, x)
	be.Copy(out, be.Upload(y))
}

func (o *tridiagOp) ApplyBlock(out, in backend.BlockView) {
	be := backend.Gonum{}
	src := be.Download(in.V)
	dst := make([]float64, in.Cols*in.Ld)
	for c := range in.Cols {
		o.apply(dst[c*in.Ld:c*in.Ld+o.n], src[c*in.Ld:c*in.Ld+o.n])
	}
	be.Copy(out.V, be.Upload(dst))
}

func (o *tridiagOp) dense() backend.Mat {
	m := backend.NewMat(o.n, o.n)
	for i := range o.n {
		m.Set(i, i, o.d[i])
		if i < o.n-1 {
			m.Set(i, i+1, o.e[i])
			m.Set(i+1, i, o.e[i])
		}
	}
	return m
}

// TestSolveWithNoMainBlock is the regression: main == 0 must still produce the spectrum
// and, under WantFull, the Ritz vectors — because gamma_i is an overlap over every P row
// and there is nothing else to compute it from.
func TestSolveWithNoMainBlock(t *testing.T) {
	be := backend.Gonum{}
	const n, b = 281, 64 // Mg's actual P size and the -fano-block the sweep used
	op := newTridiagOp(n, 0)

	res := Solve(op, be, Options{Block: b, MaxBlocks: 120, WantFull: true})
	if !res.HasFull() {
		t.Fatal("main == 0 produced no full Ritz vectors; fano.Widths cannot form gamma_i")
	}
	if res.FullVecs.Rows != n {
		t.Fatalf("FullVecs has %d rows, want %d", res.FullVecs.Rows, n)
	}
	if len(res.Values) == 0 {
		t.Fatal("no eigenvalues returned")
	}
	if res.MainVecs.Rows != 0 {
		t.Fatalf("MainVecs has %d rows, want 0 for a subspace with no main class", res.MainVecs.Rows)
	}

	// The lowest roots are what a block Lanczos converges first and they must be exact.
	// The top of this spectrum is not converged at this subspace size, which is ordinary
	// Lanczos behaviour and not what this test is about.
	want, _ := be.SymEig(op.dense())
	for i := range 10 {
		if math.Abs(res.Values[i]-want[i]) > 1e-6*math.Max(1, math.Abs(want[i])) {
			t.Errorf("eigenvalue %d = %.12g, want %.12g", i, res.Values[i], want[i])
		}
	}

	// Every Ritz vector must satisfy ‖Mv − θv‖ ≈ 0, which only the satellite rows can
	// deliver when there are no main rows at all.
	for _, k := range []int{0, 1, 2} {
		v := col(res.FullVecs, k)
		y := make([]float64, n)
		op.apply(y, v)
		var acc float64
		for i := range y {
			d := y[i] - res.Values[k]*v[i]
			acc += d * d
		}
		if r := math.Sqrt(acc); r > 1e-6 {
			t.Errorf("root %d: residual %.3e", k, r)
		}
	}
	// Normalization must hold for EVERY Ritz vector, converged or not: gamma_i =
	// 2 pi <g|chi_i>^2 is a width only if the chi_i are unit vectors.
	for k := range len(res.Values) {
		v := col(res.FullVecs, k)
		var nrm float64
		for _, x := range v {
			nrm += x * x
		}
		if math.Abs(math.Sqrt(nrm)-1) > 1e-9 {
			t.Fatalf("root %d: norm %.12g, want 1", k, math.Sqrt(nrm))
		}
	}
	// Bessel: the Ritz vectors are orthonormal, so sum_i <g|chi_i>^2 can never exceed
	// ‖g‖^2. The shortfall IS the sum-rule residual that fano.Pseudo reports — this
	// subspace deflates to 183 of 281 dimensions — so equality is asserted separately
	// below on a Krylov space that is actually complete.
	g := make([]float64, n)
	for i := range g {
		g[i] = math.Sin(float64(i)*0.7) + 0.3
	}
	var gg, tot float64
	for _, x := range g {
		gg += x * x
	}
	for k := range len(res.Values) {
		v := col(res.FullVecs, k)
		var ov float64
		for i := range v {
			ov += g[i] * v[i]
		}
		tot += ov * ov
	}
	if tot > gg*(1+1e-12) {
		t.Errorf("sum <g|chi>^2 = %.12g exceeds ‖g‖^2 = %.12g; the Ritz vectors are not orthonormal",
			tot, gg)
	}
}

// TestFullVecsChunkNarrowerThanMainBlock covers the second half of the same bug: the
// full-vector panel used to be built in column chunks of `main` while its output buffer
// is allocated for the KRYLOV width b, so a Block smaller than the operator's main block
// wrote past the end of that buffer.
func TestFullVecsChunkNarrowerThanMainBlock(t *testing.T) {
	be := backend.Gonum{}
	const n = 200
	op := newTridiagOp(n, 40) // main = 40
	res := Solve(op, be, Options{Block: 8, MaxBlocks: 20, WantFull: true})
	if !res.HasFull() {
		t.Fatal("no full Ritz vectors")
	}
	for _, k := range []int{0, 1} {
		v := col(res.FullVecs, k)
		y := make([]float64, n)
		op.apply(y, v)
		var acc float64
		for i := range y {
			d := y[i] - res.Values[k]*v[i]
			acc += d * d
		}
		if r := math.Sqrt(acc); r > 1e-6 {
			t.Errorf("root %d: residual %.3e (block 8 < main 40)", k, r)
		}
	}
}

// TestNoMainBlockCompleteness is the sum rule itself, on a main-less subspace whose
// Krylov space spans everything: sum_i <g|chi_i>^2 = ‖g‖^2 exactly. That identity is what
// fano.Widths' gate checks, and it can only be reached through the full Ritz vectors — the
// very thing a main-less P space used not to produce.
func TestNoMainBlockCompleteness(t *testing.T) {
	be := backend.Gonum{}
	const n = 100
	op := newTridiagOp(n, 0)
	res := Solve(op, be, Options{Block: n, MaxBlocks: 2, WantFull: true})
	if !res.HasFull() || len(res.Values) != n {
		t.Fatalf("got %d roots with HasFull=%v, want %d complete", len(res.Values), res.HasFull(), n)
	}
	g := make([]float64, n)
	for i := range g {
		g[i] = math.Cos(float64(i)*0.31) - 0.2
	}
	var gg, tot float64
	for _, x := range g {
		gg += x * x
	}
	for k := range n {
		v := col(res.FullVecs, k)
		var ov float64
		for i := range v {
			ov += g[i] * v[i]
		}
		tot += ov * ov
	}
	if rel := math.Abs(tot-gg) / gg; rel > 1e-12 {
		t.Errorf("sum <g|chi>^2 = %.14g, ‖g‖^2 = %.14g (rel %.2e)", tot, gg, rel)
	}
}
