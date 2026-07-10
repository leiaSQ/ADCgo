package sip

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// buildH2O4 builds the CVS ADC(4) matrix for one target irrep on H2O. sym<0 uses
// the symmetry-off space; core is the core-orbital set.
func buildH2O4(t *testing.T, sym int, core []int) *Matrix {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	var orbSym []int
	target := 0
	if sym >= 0 {
		orbSym = d.OrbSym
		target = sym
	}
	sp := NewSpace4(nocc, d.NORB, orbSym, target, core)
	return New(sp, integrals.New(d, nocc, orbSym), eps, 4, backend.Gonum{})
}

// TestMatrixSymmetric4 checks the assembled order-4 matrix is symmetric — the
// structural oracle guarding block placement and the −DYSON coupling sign.
func TestMatrixSymmetric4(t *testing.T) {
	// Use a symmetry irrep to keep the 3h2p space modest.
	mx := buildH2O4(t, 2, []int{0}) // B1 irrep (0-based 2)
	M := mx.BuildMatrix()
	var maxAsym float64
	for i := range M.Rows {
		for j := range i {
			if d := math.Abs(M.At(i, j) - M.At(j, i)); d > maxAsym {
				maxAsym = d
			}
		}
	}
	if maxAsym > 1e-10 {
		t.Errorf("order 4: matrix asymmetry %g exceeds 1e-10", maxAsym)
	}
}

// TestApplyEqualsBuild4 is the primary structural gate for the order-4 operator:
// the matrix-free ApplyFull reproduces BuildMatrix. Columns are sampled across the
// 1h / 2h1p / 3h2p regions (the full n×n sweep is too slow for the large space).
func TestApplyEqualsBuild4(t *testing.T) {
	mx := buildH2O4(t, 2, []int{0})
	M := mx.BuildMatrix()
	n := mx.Size()
	sp := mx.Space()
	be := mx.be

	// Sample columns: all of 1h, a stride through 2h1p, a stride through 3h2p.
	var cols []int
	for j := range sp.BeginSat {
		cols = append(cols, j)
	}
	stride := func(lo, hi, k int) {
		if hi <= lo {
			return
		}
		step := max((hi-lo)/k, 1)
		for j := lo; j < hi; j += step {
			cols = append(cols, j)
		}
	}
	stride(sp.BeginSat, sp.Begin3h2p, 40)
	stride(sp.Begin3h2p, n, 40)

	e := make([]float64, n)
	out := be.Alloc(n)
	var maxErr float64
	for _, j := range cols {
		e[j] = 1
		in := be.Upload(e)
		e[j] = 0
		mx.ApplyFull(out, in)
		col := be.Download(out)
		for i := range n {
			if d := math.Abs(col[i] - M.At(i, j)); d > maxErr {
				maxErr = d
			}
		}
	}
	if maxErr > 1e-10 {
		t.Errorf("order 4: ApplyFull vs BuildMatrix max diff %g exceeds 1e-10", maxErr)
	}
}
