package dip

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
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
