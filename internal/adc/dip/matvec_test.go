package dip

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

func buildH2O(t *testing.T, spin Spin) *Matrix {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace(nocc, d.NORB, nil, 0, spin)
	return New(sp, integrals.New(d, nocc, nil), eps, backend.Gonum{})
}

// TestOperatorResidentBytes: the pre-flight footprint estimate must exactly equal the
// assembled operator's real size (nnz·8), since the backend chooser refuses a GPU based
// on it. blockTasks is the shared enumeration behind both, so they cannot drift.
func TestOperatorResidentBytes(t *testing.T) {
	for _, spin := range []Spin{Singlet, Triplet} {
		mx := buildH2O(t, spin)
		nnz, _ := mx.OperatorNNZ() // assembles (uploads to the host backend) and sums r*c
		want := uint64(nnz) * 8
		if got := mx.OperatorResidentBytes(); got != want {
			t.Errorf("spin %d: OperatorResidentBytes=%d, want nnz*8=%d", spin, got, want)
		}
	}
}

// TestMatrixSymmetric: the assembled DIP matrix must be symmetric (each block +
// its transpose consistently placed, and diagonal blocks symmetric).
func TestMatrixSymmetric(t *testing.T) {
	for _, spin := range []Spin{Singlet, Triplet} {
		mx := buildH2O(t, spin)
		M := mx.BuildMatrix()
		n := M.Rows
		var maxAsym float64
		for i := range n {
			for j := range i {
				d := math.Abs(M.At(i, j) - M.At(j, i))
				if d > maxAsym {
					maxAsym = d
				}
			}
		}
		if maxAsym > 1e-10 {
			t.Errorf("spin %d: matrix asymmetry %g exceeds 1e-10", spin, maxAsym)
		}
	}
}

// TestApplyEqualsBuild is the primary correctness gate: the matrix-free ApplyFull
// must reproduce every column of the densely-built matrix.
func TestApplyEqualsBuild(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full column-by-column Apply==Build check in -short mode")
	}
	for _, spin := range []Spin{Singlet, Triplet} {
		mx := buildH2O(t, spin)
		M := mx.BuildMatrix()
		n := mx.Size()

		be := mx.be
		e := make([]float64, n)
		out := be.Alloc(n)
		var maxErr float64
		for j := range n {
			e[j] = 1
			in := be.Upload(e)
			e[j] = 0
			mx.ApplyFull(out, in)
			col := be.Download(out)
			for i := range n {
				d := math.Abs(col[i] - M.At(i, j))
				if d > maxErr {
					maxErr = d
				}
			}
		}
		if maxErr > 1e-10 {
			t.Errorf("spin %d: ApplyFull vs BuildMatrix max diff %g exceeds 1e-10", spin, maxErr)
		}
	}
}

// TestDenseSpectrumSane checks the dense diagonalization yields real, ordered
// double-ionization energies and that pole strengths are bounded in (0,100].
func TestDenseSpectrumSane(t *testing.T) {
	mx := buildH2O(t, Singlet)
	M := mx.BuildMatrix()
	evals, evecs := mx.be.SymEig(M)
	main := mx.Space().MainBlockSize()

	// The lowest double-ionization energy should be positive and physical
	// (H2O double ionization is tens of eV; in Hartree, > 1).
	if evals[0] < 1.0 {
		t.Errorf("lowest DIP eigenvalue %g Ha unphysically small", evals[0])
	}
	// Pole strength of each state = 100 * sum over 2h block of c^2, in (0,100].
	var maxPS float64
	for st := range evals {
		var ps float64
		for c := range main {
			ps += evecs.At(c, st) * evecs.At(c, st)
		}
		ps *= 100
		if ps > maxPS {
			maxPS = ps
		}
		if ps > 100.0001 {
			t.Fatalf("state %d pole strength %g exceeds 100%%", st, ps)
		}
	}
	// At least one state should be dominated by the 2h space (a main line).
	if maxPS < 50 {
		t.Errorf("no strong main-space state found (max ps %g%%)", maxPS)
	}
}

// TestApplyBlockMatchesApplyFull pins the level-3 mat-vec to the level-2 one it
// replaces: for every (spin, irrep) sector, M applied to a block of random vectors
// must equal M applied to each vector individually.
//
// This is the correctness gate for the whole apply-side optimization. It also
// exercises BlockView.RowRange and the row-major/column-major flag inversion in
// GemmMat, which is where a silent transpose would hide.
func TestApplyBlockMatchesApplyFull(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)
	rng := rand.New(rand.NewSource(3))

	const blk = 5
	// Every backend compiled into this build: under -tags cuda/hip this also covers
	// GemmMat's row-major -> column-major flag inversion on the device.
	for _, name := range backend.Available() {
		be, err := backend.New(name)
		if err != nil {
			t.Fatalf("backend.New(%q): %v", name, err)
		}
		t.Run(name, func(t *testing.T) { applyBlockVsFull(t, be, d, nocc, eps, ints, rng, blk) })
	}
}

func applyBlockVsFull(t *testing.T, be backend.Backend, d *fcidump.Data, nocc int,
	eps []float64, ints *integrals.Store, rng *rand.Rand, blk int) {
	tested := 0
	for _, spin := range []Spin{Singlet, Triplet} {
		for sym := range 4 {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			n := sp.Size()
			if n == 0 {
				continue
			}
			mx := New(sp, ints, eps, be)

			// Random n×blk input panel, column-major.
			inData := make([]float64, n*blk)
			for i := range inData {
				inData[i] = rng.NormFloat64()
			}
			in := backend.BlockView{V: be.Upload(inData), Rows: n, Cols: blk, Ld: n}
			out := backend.BlockView{V: be.Alloc(n * blk), Rows: n, Cols: blk, Ld: n}
			mx.ApplyBlock(out, in)
			got := be.Download(out.V)

			// Column-by-column reference through ApplyFull.
			var maxDiff, scale float64
			for j := range blk {
				col := be.Upload(inData[j*n : (j+1)*n])
				ref := be.Alloc(n)
				mx.ApplyFull(ref, col)
				want := be.Download(ref)
				for i := range n {
					scale = math.Max(scale, math.Abs(want[i]))
					maxDiff = math.Max(maxDiff, math.Abs(want[i]-got[j*n+i]))
				}
			}
			rel := maxDiff / math.Max(scale, 1e-300)
			if rel > 1e-12 {
				t.Errorf("spin=%v sym=%d n=%d: ApplyBlock vs ApplyFull relative diff %.3e", spin, sym, n, rel)
			}
			tested++
		}
	}
	if tested == 0 {
		t.Fatal("no sectors exercised")
	}
	t.Logf("%d sectors: ApplyBlock == ApplyFull", tested)
}

// TestPanelCacheNotMutated guards the invariant integrals.Store's A/B/V caches rest on: the
// panels they hand out are SHARED, so a consumer that wrote through one would corrupt every
// later block built from it — silently, and only for runs long enough to reach the second use.
//
// It snapshots every panel the sector can ask for, drives the full operator build (BuildMatrix
// exercises the 2h/2h, both 2h↔3h1p coupling families and all three satellite families), and
// re-reads. Every consumer today accumulates FROM the panel via Mat.AddSubMat / Mat.AddSubVec;
// this fails the moment one stops doing so.
func TestPanelCacheNotMutated(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		nocc, nsym := sp.Nocc, ints.NSym()

		snapA := map[[3]int][]float64{}
		snapB := map[[3]int][]float64{}
		snapV := map[[4]int][]float64{}
		copyOf := func(x []float64) []float64 { return append([]float64(nil), x...) }
		for i := range nocc {
			for j := range nocc {
				for σ := range nsym {
					snapA[[3]int{i, j, σ}] = copyOf(ints.A(i, j, σ).Data)
					snapB[[3]int{i, j, σ}] = copyOf(ints.B(i, j, σ).Data)
					for k := range nocc {
						snapV[[4]int{i, j, k, σ}] = copyOf(ints.V(i, j, k, σ))
					}
				}
			}
		}

		mx := New(sp, ints, eps, be)
		defer mx.Release()
		mx.BuildMatrix()

		same := func(what string, key any, got, want []float64) {
			t.Helper()
			if len(got) != len(want) {
				t.Fatalf("spin=%v sym=%d: %s%v length %d, was %d", spin, sym, what, key, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("spin=%v sym=%d: cached %s%v entry %d changed: %v -> %v — a consumer "+
						"wrote through a shared panel", spin, sym, what, key, i, want[i], got[i])
				}
			}
		}
		for k, want := range snapA {
			same("A", k, ints.A(k[0], k[1], k[2]).Data, want)
		}
		for k, want := range snapB {
			same("B", k, ints.B(k[0], k[1], k[2]).Data, want)
		}
		for k, want := range snapV {
			same("V", k, ints.V(k[0], k[1], k[2], k[3]), want)
		}
	})
}
