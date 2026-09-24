package lanczos

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

func buildH2O(t *testing.T, spin dip.Spin) *dip.Matrix {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := dip.NewSpace(nocc, d.NORB, nil, 0, spin)
	return dip.New(sp, integrals.New(d, nocc, nil), eps, backend.Gonum{})
}

// denseState is a reference (energy, ps) main-line from the dense path.
type denseState struct{ e, ps float64 }

func denseMainStates(mx *dip.Matrix, psMin float64) []denseState {
	be := backend.Gonum{}
	M := mx.BuildMatrix()
	evals, evecs := be.SymEig(M)
	main := mx.MainBlockSize()
	var out []denseState
	for k := range evals {
		var ps float64
		for c := range main {
			ps += evecs.At(c, k) * evecs.At(c, k)
		}
		ps *= 100
		if ps >= psMin {
			out = append(out, denseState{evals[k], ps})
		}
	}
	return out
}

// TestLanczosMatchesDense is Gate 2: the block-Lanczos spectrum must reproduce
// the dense eigenvalues and pole strengths of every main-space-weighted state.
func TestLanczosMatchesDense(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping heavy block-Lanczos convergence test in -short mode")
	}
	be := backend.Gonum{}
	// Tolerances reflect finite Lanczos convergence (not algorithm error — the
	// full-subspace test below shows exactness): energies to ~5e-3 Ha (~0.14 eV)
	// and pole strengths to 5% for the clear main lines.
	const tolE, tolPS = 5e-3, 5.0
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		mx := buildH2O(t, spin)
		ref := denseMainStates(mx, 65) // clear main lines
		res := Solve(mx, be, Options{MaxBlocks: 24})

		for _, d := range ref {
			// nearest non-spurious Ritz value.
			best, bestPS, bestErr := 0.0, 0.0, math.Inf(1)
			for k := range res.Values {
				if res.Spurious(k, 1e-9) {
					continue
				}
				if e := math.Abs(res.Values[k] - d.e); e < bestErr {
					bestErr, best, bestPS = e, res.Values[k], res.PS[k]
				}
			}
			if bestErr > tolE {
				t.Errorf("spin %d: dense state %.4f Ha (ps %.1f%%) unmatched; nearest Ritz %.4f (Δ=%.2e)",
					spin, d.e, d.ps, best, bestErr)
				continue
			}
			if math.Abs(bestPS-d.ps) > tolPS {
				t.Errorf("spin %d: state %.4f Ha ps mismatch dense=%.2f lanczos=%.2f",
					spin, d.e, d.ps, bestPS)
			}
		}
	}
}

// TestLanczosInvariance: full subspace reproduces the dense spectrum exactly.
func TestLanczosFullExact(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full-subspace exactness test in -short mode")
	}
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Triplet) // smaller main block → cheaper full build
	M := mx.BuildMatrix()
	dense, _ := be.SymEig(M)

	res := Solve(mx, be, Options{}) // no caps → grows to invariant subspace
	// Every dense eigenvalue must appear among the Ritz values.
	var maxErr float64
	for _, e := range dense {
		best := math.Inf(1)
		for _, r := range res.Values {
			if d := math.Abs(r - e); d < best {
				best = d
			}
		}
		if best > maxErr {
			maxErr = best
		}
	}
	if maxErr > 1e-8 {
		t.Errorf("full-subspace Lanczos vs dense max eigenvalue error %g", maxErr)
	}
}

// The block count is the one solver parameter that has to mean the same thing here as
// in theADCcode, because it is what a user sets to reproduce a reference run. There,
// `iter N` diagonalizes a Krylov space of N blocks of `main_block_size()` columns:
// Iterate() is called N+1 times (the first only registers the start block), reaching
// dim = (N+1)·block, and Diagonalize() sets dimd = dim − block = N·block, which is the
// "Size of Lanczos space" it prints (../ADC/libLanczos/lanczos.h:226-238, :257;
// ../ADC/analysis/adc_analyzer.cpp:229-247). The trailing block only supplies the
// coupling for the residuals — exactly the discarded orthogonalization in Solve.
//
// So MaxBlocks == iter, and the subspace is MaxBlocks·main. These tests pin that; they
// fail against the older (MaxBlocks+1)·main convention.

func TestSubspaceDimCountsBlocks(t *testing.T) {
	cases := []struct {
		n, main, blocks, want int
	}{
		{139, 5, 100, 139}, // reference sym1 singlet: 100·5 = 500 overshoots n, we cap
		{151, 1, 100, 100}, // reference sym1 triplet: "Size of Lanczos space: 100"
		{123, 2, 100, 123}, // reference sym3 singlet: 100·2 = 200 overshoots n
		{10000, 7, 30, 210},
		{10000, 7, 1, 7}, // one block == just the start block
		{500, 5, 0, 500}, // 0 → unbounded, capped at n
		{0, 5, 30, 0},
		{500, 0, 30, 0},
	}
	for _, c := range cases {
		if got := SubspaceDim(c.n, c.main, Options{MaxBlocks: c.blocks}); got != c.want {
			t.Errorf("SubspaceDim(n=%d, main=%d, blocks=%d) = %d, want %d",
				c.n, c.main, c.blocks, got, c.want)
		}
	}
}

// TestSolveBuildsMaxBlocksBlocks checks that Solve actually stops at MaxBlocks blocks,
// not MaxBlocks+1: the Ritz count is the subspace dimension it built.
func TestSolveBuildsMaxBlocksBlocks(t *testing.T) {
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Singlet)
	n, main := mx.Size(), mx.MainBlockSize()

	for _, blocks := range []int{1, 2, 5} {
		// Spelled out rather than taken from SubspaceDim: this must fail if both drift
		// together.
		want := blocks * main
		if want >= n {
			t.Fatalf("blocks=%d saturates the %d-dim sector; test needs a truncated run", blocks, n)
		}
		res := Solve(mx, be, Options{MaxBlocks: blocks})
		if got := len(res.Values); got != want {
			t.Errorf("blocks=%d: Solve returned %d Ritz values, want %d (= %d blocks × %d main)",
				blocks, got, want, blocks, main)
		}
		if got := SubspaceDim(n, main, Options{MaxBlocks: blocks}); got != want {
			t.Errorf("blocks=%d: SubspaceDim = %d, want %d — Solve and SubspaceDim disagree",
				blocks, got, want)
		}
	}
}

// refSector is one (irrep, spin) block of theADCcode's h2o/DZP DIP run, read off
// testdata/reference/adcdip{irrep}.out. `main` is its "block size" line, `size` its
// "Number of ISR configurations" line, and lancSpace its "Size of Lanczos space" —
// all four files ran `iter 100`.
type refSector struct {
	irrep     int
	spin      dip.Spin
	main      int
	size      int
	lancSpace int
}

var refSectors = []refSector{
	{0, dip.Singlet, 5, 139, 500},
	{0, dip.Triplet, 1, 151, 100},
	{1, dip.Singlet, 1, 117, 100},
	{1, dip.Triplet, 1, 151, 100},
	{2, dip.Singlet, 2, 123, 200},
	{2, dip.Triplet, 2, 152, 200},
	{3, dip.Singlet, 2, 131, 200},
	{3, dip.Triplet, 2, 152, 200},
}

// TestReferenceBlockAndSpaceSizes ties the two quantities theADCcode prints per sector —
// the Lanczos block size and the Lanczos space — to ADCgo's MainBlockSize() and
// SubspaceDim(). The matched FCIDUMP carries theADCcode's own ORBSYM, so ADCgo sector
// index N is reference file adcdip{N+1}.out.
func TestReferenceBlockAndSpaceSizes(t *testing.T) {
	const refIter = 100 // the `iter` value used in every adcdip*.out run

	path := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skip("matched FCIDUMP not present (regenerate with ../ADC/fcidump_export)")
		}
		t.Fatalf("read matched fcidump: %v", err)
	}
	nocc := mp.NOcc(d)

	for _, r := range refSectors {
		sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, r.irrep, r.spin)
		if got := sp.MainBlockSize(); got != r.main {
			t.Errorf("adcdip%d spin %d: MainBlockSize = %d, reference 'block size' = %d",
				r.irrep+1, spinCode(r.spin), got, r.main)
		}
		if got := sp.Size(); got != r.size {
			t.Errorf("adcdip%d spin %d: Size = %d, reference 'Number of ISR configurations' = %d",
				r.irrep+1, spinCode(r.spin), got, r.size)
		}
		// MaxBlocks == iter: the unclamped subspace is exactly what the reference prints.
		if got := refIter * r.main; got != r.lancSpace {
			t.Errorf("adcdip%d spin %d: %d blocks × %d main = %d, reference 'Size of Lanczos space' = %d",
				r.irrep+1, spinCode(r.spin), refIter, r.main, got, r.lancSpace)
		}
		// ADCgo additionally clamps at the sector dimension rather than generating the
		// ghost roots the reference then drops (adcdip1: 361 of 500 spurious).
		want := min(r.size, r.lancSpace)
		if got := SubspaceDim(sp.Size(), sp.MainBlockSize(), Options{MaxBlocks: refIter}); got != want {
			t.Errorf("adcdip%d spin %d: SubspaceDim = %d, want %d",
				r.irrep+1, spinCode(r.spin), got, want)
		}
	}
}

func spinCode(s dip.Spin) int {
	if s == dip.Triplet {
		return 3
	}
	return 1
}

// TestSolveReportsProgressPerBlock pins the contract the production logs depend on: Solve
// calls Options.Progress exactly once per block, with ascending 0-based block indices, so
// counting "progress" lines in a job's stderr counts the blocks it actually built. Job
// 14717237 had to have its 200 blocks reconstructed from Slurm's recorded checkpoint write
// volume because Solve reported nothing at all; this is what replaces that arithmetic.
func TestSolveReportsProgressPerBlock(t *testing.T) {
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Singlet)
	n, main := mx.Size(), mx.MainBlockSize()

	type call struct{ iter, dim, size int }
	for _, blocks := range []int{1, 2, 5} {
		if blocks*main >= n {
			t.Fatalf("blocks=%d saturates the %d-dim sector; test needs a truncated run", blocks, n)
		}
		var got []call
		res := Solve(mx, be, Options{
			MaxBlocks: blocks,
			Progress: func(iter, dim, size int, _ Timing) {
				got = append(got, call{iter, dim, size})
			},
		})
		if len(got) != blocks {
			t.Fatalf("blocks=%d: Progress fired %d times, want %d (once per block)",
				blocks, len(got), blocks)
		}
		for i, c := range got {
			if c.iter != i {
				t.Errorf("blocks=%d: call %d reported block=%d, want %d", blocks, i, c.iter, i)
			}
			if c.size != main {
				t.Errorf("blocks=%d: call %d reported size=%d, want the undeflated block %d",
					blocks, i, c.size, main)
			}
		}
		// The last line's dim is the subspace the run ends with, which is what makes a
		// truncated log readable without waiting for the result.
		if last := got[len(got)-1].dim; last != len(res.Values) {
			t.Errorf("blocks=%d: final reported dim=%d, but Solve returned %d Ritz values",
				blocks, last, len(res.Values))
		}
		// The trailing iteration only builds the discarded R_{j+1}, so it repeats the
		// previous dim rather than growing the basis.
		if blocks > 1 {
			if a, b := got[len(got)-2].dim, got[len(got)-1].dim; a != b {
				t.Errorf("blocks=%d: truncating block moved dim %d -> %d, want it unchanged",
					blocks, a, b)
			}
		}
	}
}

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

// lanczos_test.go — a subspace with NO main-class configuration.
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

// orthHarness builds an orthonormal basis B (n×d) and scratch buffers, then runs
// orthBlock on a caller-supplied candidate block and reports the two quantities the
// conditional second pass is supposed to protect: within-block orthonormality
// ‖qᵀq − I‖_max, and orthogonality to the basis ‖Bᵀq‖_max.
func orthHarness(t *testing.T, n, d int, vcols [][]float64) (rank int, orthErr, basisErr float64) {
	t.Helper()
	be := backend.Gonum{}
	rng := rand.New(rand.NewSource(17))

	// Orthonormal basis via Gram-Schmidt on random columns.
	bcols := make([][]float64, d)
	for j := range d {
		c := make([]float64, n)
		for i := range n {
			c[i] = rng.NormFloat64()
		}
		for k := range j {
			var dot float64
			for i := range n {
				dot += bcols[k][i] * c[i]
			}
			for i := range n {
				c[i] -= dot * bcols[k][i]
			}
		}
		var nrm float64
		for i := range n {
			nrm += c[i] * c[i]
		}
		nrm = math.Sqrt(nrm)
		for i := range n {
			c[i] /= nrm
		}
		bcols[j] = c
	}
	bdata := make([]float64, n*d)
	for j := range d {
		copy(bdata[j*n:(j+1)*n], bcols[j])
	}
	basis := backend.BlockView{V: be.Upload(bdata), Rows: n, Cols: d, Ld: n}

	nc := len(vcols)
	vdata := make([]float64, n*nc)
	for j, c := range vcols {
		copy(vdata[j*n:(j+1)*n], c)
	}
	v := backend.BlockView{V: be.Upload(vdata), Rows: n, Cols: nc, Ld: n}

	pbuf := be.Alloc(d * nc)
	gbuf := be.Alloc(nc * nc)
	rank, _ = orthBlock(be, basis, v, pbuf, d, gbuf, 1e-8)
	if rank == 0 {
		return 0, 0, 0
	}

	q := be.Download(v.V)
	for a := range rank {
		for b := range rank {
			var dot float64
			for i := range n {
				dot += q[a*n+i] * q[b*n+i]
			}
			want := 0.0
			if a == b {
				want = 1
			}
			orthErr = math.Max(orthErr, math.Abs(dot-want))
		}
		for j := range d {
			var dot float64
			for i := range n {
				dot += bcols[j][i] * q[a*n+i]
			}
			basisErr = math.Max(basisErr, math.Abs(dot))
		}
	}
	return rank, orthErr, basisErr
}

// TestOrthBlockWellConditioned: a healthy block keeps full rank and, after the single
// pass the criterion permits, is orthonormal and orthogonal to the basis to ~eps.
func TestOrthBlockWellConditioned(t *testing.T) {
	const n, d, b = 300, 40, 6
	rng := rand.New(rand.NewSource(23))
	vcols := make([][]float64, b)
	for j := range b {
		c := make([]float64, n)
		for i := range n {
			c[i] = rng.NormFloat64()
		}
		vcols[j] = c
	}
	rank, orthErr, basisErr := orthHarness(t, n, d, vcols)
	if rank != b {
		t.Fatalf("rank %d, want %d (nothing should deflate)", rank, b)
	}
	if orthErr > 1e-13 || basisErr > 1e-13 {
		t.Errorf("well-conditioned: ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", orthErr, basisErr)
	}
	t.Logf("well-conditioned: rank=%d ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", rank, orthErr, basisErr)
}

// illConditionedBlock builds a full-rank block with cond(v) ≈ 1/scale. The perturbation
// must stay well above the relative deflation floor (relCond2 = 1e-14 on λ, i.e. 1e-7 on
// singular values), or the offending direction is simply *deflated* and the survivors
// come out well conditioned — in which case the `rank < cols` branch, not the cond²
// branch, is what triggers the second pass.
func illConditionedBlock(n, b int, scale float64, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	cols := make([][]float64, b)
	for j := range b {
		c := make([]float64, n)
		for i := range n {
			c[i] = rng.NormFloat64()
		}
		cols[j] = c
	}
	for i := range n {
		cols[1][i] = cols[0][i] + scale*cols[1][i]
	}
	return cols
}

// TestOrthBlockIllConditionedStaysFullRank exercises the cond² branch: the block is
// ill conditioned (cond ≈ 1e5, so cond² ≈ 1e10 ≫ maxGramCond2) but nothing deflates.
// A single Gram-QR pass would lose orthogonality at ~cond²·eps ≈ 1e-6; the criterion
// must catch it on cond² alone and reorthogonalize.
func TestOrthBlockIllConditionedStaysFullRank(t *testing.T) {
	const n, d, b = 300, 40, 5
	vcols := illConditionedBlock(n, b, 1e-5, 29)

	rank, orthErr, basisErr := orthHarness(t, n, d, vcols)
	if rank != b {
		t.Fatalf("rank %d, want %d: this block should NOT deflate, so only cond² can trigger the second pass", rank, b)
	}
	if orthErr > 1e-12 {
		t.Errorf("‖qᵀq−I‖=%.3e — the second pass did not recover orthogonality", orthErr)
	}
	if basisErr > 1e-12 {
		t.Errorf("‖Bᵀq‖=%.3e — q is not orthogonal to the basis", basisErr)
	}
	t.Logf("cond²-triggered: rank=%d ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", rank, orthErr, basisErr)
}

// TestOrthBlockDeflates exercises the rank branch: a duplicated column and a column
// perturbed below the deflation floor must both be dropped, and the survivors must come
// out orthonormal.
func TestOrthBlockDeflates(t *testing.T) {
	const n, d, b = 300, 40, 6
	vcols := illConditionedBlock(n, b, 1e-9, 41) // 1e-9 is below the 1e-7 singular-value floor
	copy(vcols[3], vcols[2])                     // exact duplicate

	rank, orthErr, basisErr := orthHarness(t, n, d, vcols)
	if rank == 0 {
		t.Fatal("everything deflated")
	}
	if rank >= b {
		t.Errorf("rank %d of %d: the duplicate and the sub-floor direction should have deflated", rank, b)
	}
	if orthErr > 1e-12 || basisErr > 1e-12 {
		t.Errorf("after deflation: ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", orthErr, basisErr)
	}
	t.Logf("rank-triggered: rank=%d (of %d) ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", rank, b, orthErr, basisErr)
}

// TestSinglePassLosesOrthogonality documents *why* the criterion exists. On a full-rank
// block with cond² ≫ maxGramCond2, one Gram-QR pass leaves an orthogonality error many
// orders of magnitude above the solver's 1e-10 gate. This is the failure the conditional
// must never skip: if it ever stops holding, maxGramCond2 can be revisited.
func TestSinglePassLosesOrthogonality(t *testing.T) {
	be := backend.Gonum{}
	const n, b = 300, 3
	vcols := illConditionedBlock(n, b, 1e-6, 31) // cond ≈ 1e6 → cond² ≈ 1e12
	vdata := make([]float64, n*b)
	for j, c := range vcols {
		copy(vdata[j*n:(j+1)*n], c)
	}
	v := backend.BlockView{V: be.Upload(vdata), Rows: n, Cols: b, Ld: n}
	gbuf := be.Alloc(b * b)

	rank, _, cond2 := blockOrth(be, v, gbuf, 1e-8) // ONE pass, deliberately
	if rank != b {
		t.Fatalf("rank %d: block deflated, so this does not test the cond² path", rank)
	}
	if cond2 <= maxGramCond2 {
		t.Fatalf("cond²=%.3e <= maxGramCond2=%.3e: the criterion would skip the second pass here", cond2, maxGramCond2)
	}

	q := be.Download(v.V)
	var orthErr float64
	for a := range rank {
		for c := range rank {
			var dot float64
			for i := range n {
				dot += q[a*n+i] * q[c*n+i]
			}
			want := 0.0
			if a == c {
				want = 1
			}
			orthErr = math.Max(orthErr, math.Abs(dot-want))
		}
	}
	if orthErr < 1e-11 {
		t.Errorf("single pass at cond²=%.3e gave ‖qᵀq−I‖=%.3e — unexpectedly accurate; "+
			"the second pass may be unnecessary and maxGramCond2 should be re-derived", cond2, orthErr)
	}
	t.Logf("single pass at cond²=%.3e: ‖qᵀq−I‖=%.3e (%.0fx the 1e-12 target) — second pass required",
		cond2, orthErr, orthErr/1e-12)
}

// TestStartVecsMatchesDefaultSeed: StartVecs holding the Cartesian units e_0..e_{b-1},
// scaled and mixed by an invertible transformation, spans the default start block, so
// the Krylov space and its Ritz values are the default run's.
func TestStartVecsMatchesDefaultSeed(t *testing.T) {
	mx := buildH2O(t, dip.Singlet)
	be := backend.Gonum{}
	n, b := mx.Size(), mx.MainBlockSize()
	opts := Options{MaxBlocks: 6}
	ref := Solve(mx, be, opts)
	vecs := make([]float64, n*b)
	for c := range b {
		vecs[c*n+c] = 2
		if c > 0 {
			vecs[c*n+c-1] = 0.5 // mixes column c-1 in: still full rank
		}
	}
	opts.StartVecs = vecs
	got := Solve(mx, be, opts)
	if len(got.Values) != len(ref.Values) {
		t.Fatalf("%d Ritz values with StartVecs, %d by default", len(got.Values), len(ref.Values))
	}
	for k := range ref.Values {
		if d := math.Abs(got.Values[k] - ref.Values[k]); d > 1e-10 {
			t.Fatalf("Ritz value %d differs by %.2e", k, d)
		}
		if d := math.Abs(got.PS[k] - ref.PS[k]); d > 1e-8 {
			t.Fatalf("pole strength %d differs by %.2e", k, d)
		}
	}
}

func TestStartVecsRejectsBadPanels(t *testing.T) {
	expectPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: no panic", name)
			}
		}()
		f()
	}
	n := 4
	dup := []float64{1, 0, 0, 0, 2, 0, 0, 0} // column 1 = 2·column 0
	expectPanic("rank deficient", func() { startBlock(n, 2, nil, dup) })
	expectPanic("short panel", func() { startBlock(n, 2, nil, dup[:6]) })
	expectPanic("with StartRows", func() { startBlock(n, 2, []int{0, 1}, dup) })
	nan := []float64{math.NaN(), 0, 0, 0}
	expectPanic("not finite", func() { startBlock(n, 1, nil, nan) })
	ok := startBlock(n, 2, nil, []float64{3, 0, 0, 0, 1, 1, 0, 0})
	want := []float64{1, 0, 0, 0, 0, 1, 0, 0}
	for i := range want {
		if math.Abs(ok[i]-want[i]) > 1e-15 {
			t.Fatalf("orthonormalized panel %v, want %v", ok, want)
		}
	}
}

// TestRitzResidualBoundsEigenvalueError checks that Result.Residual is a genuine
// convergence measure and not decoration.
//
// For a real symmetric M and any unit y with θ = yᵀMy, the residual r = ‖My − θy‖
// bounds the distance from θ to the spectrum: min_i |θ − λ_i| ≤ r. So every reported
// Ritz value must sit within its own residual of some exact eigenvalue. A truncated
// run (small -blocks) produces both converged and unconverged roots, which is exactly
// what makes this test discriminating: a residual that were always ~0, or one
// unrelated to the error, would fail.
func TestRitzResidualBoundsEigenvalueError(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	be := backend.Gonum{}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)

	// A sector large enough that -blocks 6 leaves real convergence spread.
	sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, 0, dip.Singlet)
	if sp.Size() == 0 {
		t.Skip("empty sector")
	}
	mx := dip.New(sp, ints, eps, be)
	exact := SolveDense(mx, be).Values

	res := Solve(mx, be, Options{MaxBlocks: 6})
	if len(res.Residual) != len(res.Values) {
		t.Fatalf("Residual has %d entries, want %d", len(res.Residual), len(res.Values))
	}

	var maxSlack, maxResid float64
	nonTrivial := 0
	for k, theta := range res.Values {
		best := math.Inf(1)
		for _, lam := range exact {
			best = math.Min(best, math.Abs(theta-lam))
		}
		r := res.Residual[k]
		maxResid = math.Max(maxResid, r)
		if r > 1e-9 {
			nonTrivial++
		}
		// |θ − λ| ≤ r, with a little slack for round-off.
		if best > r+1e-8 {
			t.Errorf("Ritz %d: θ=%.10f is %.3e from the spectrum but residual is only %.3e",
				k, theta, best, r)
		}
		maxSlack = math.Max(maxSlack, best-r)
	}
	if nonTrivial == 0 {
		t.Fatal("every residual was ~0; the truncated run should leave unconverged roots")
	}
	t.Logf("dim=%d roots=%d, %d with residual>1e-9, max residual %.3e, max (err - residual) %.3e",
		sp.Size(), len(res.Values), nonTrivial, maxResid, maxSlack)
}

// TestRitzResidualVanishesAtFullSubspace: once the Krylov space is the whole space,
// every Ritz pair is exact and the residual must collapse to round-off.
func TestRitzResidualVanishesAtFullSubspace(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	be := backend.Gonum{}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)

	sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, 1, dip.Triplet)
	if sp.Size() == 0 {
		t.Skip("empty sector")
	}
	mx := dip.New(sp, ints, eps, be)
	res := Solve(mx, be, Options{}) // run to full subspace

	var maxResid float64
	for _, r := range res.Residual {
		maxResid = math.Max(maxResid, r)
	}
	if maxResid > 1e-8 {
		t.Errorf("full-subspace max residual %.3e, want ~0", maxResid)
	}
	t.Logf("dim=%d: max residual at full subspace %.3e", sp.Size(), maxResid)
}
