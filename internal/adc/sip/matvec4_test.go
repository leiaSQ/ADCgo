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
	checkApplyEqualsBuild4(t, buildH2O4(t, 2, []int{0}))
}

// TestApplyEqualsBuild4MatFree is the same oracle with MatFreeOn, which applies BOTH the
// 2h1p×3h2p coupling (wert2elem4) and the 2h1p×2h1p block (c22elem4) matrix-free
// (recomputed on the fly) instead of assembled densely. The matrix-free paths evaluate the
// same elements, so ApplyFull must still reproduce the dense BuildMatrix (always dense).
func TestApplyEqualsBuild4MatFree(t *testing.T) {
	mx := buildH2O4(t, 2, []int{0})
	mx.SetMatFree(MatFreeOn, 0)
	checkApplyEqualsBuild4(t, mx)
}

func checkApplyEqualsBuild4(t *testing.T, mx *Matrix) {
	t.Helper()
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

// TestWert2GateExhaustive guards the matrix-free correctness invariant: the candidate
// buckets used by newWert2MatFree (a 2h1p row is a candidate for a 3h2p column when its
// particle matches a column particle or its valence hole matches a column hole) must be
// a superset of every nonzero element of the 2h1p×3h2p coupling. If any nonzero element
// fell outside the candidate set, the matrix-free apply would silently skip it. We
// assert the contrapositive over ALL (row,col) pairs: a non-candidate ⇒ wert2elem4 == 0.
func TestWert2GateExhaustive(t *testing.T) {
	mx := buildH2O4(t, 0, []int{0}) // A1 sector: has 1h + 2h1p + 3h2p
	sp := mx.Space()
	rows := sp.Configs[sp.BeginSat:sp.Begin3h2p]
	if len(rows) == 0 || len(sp.Sat3) == 0 {
		t.Fatalf("need a non-empty 2h1p and 3h2p space (got %d, %d)", len(rows), len(sp.Sat3))
	}
	candidate := func(r Config, c Config3) bool {
		return r.Vir == c.I || r.Vir == c.J || r.Occ[1] == c.L || r.Occ[1] == c.M
	}
	var checked, nonzero int
	for _, col := range sp.Sat3 {
		for _, row := range rows {
			if candidate(row, col) {
				continue
			}
			checked++
			if v := mx.el.wert2elem4(row, col); v != 0 {
				t.Fatalf("non-candidate (row Vir=%d Occ=%v, col I=%d J=%d L=%d M=%d) has nonzero wert2elem4=%g",
					row.Vir, row.Occ, col.I, col.J, col.L, col.M, v)
			}
		}
	}
	// Sanity that the coupling isn't trivially all-zero (the test would be vacuous).
	for _, col := range sp.Sat3 {
		for _, row := range rows {
			if candidate(row, col) && mx.el.wert2elem4(row, col) != 0 {
				nonzero++
			}
		}
	}
	if nonzero == 0 {
		t.Fatal("coupling block is entirely zero — gate test is vacuous")
	}
	t.Logf("checked %d non-candidate pairs (all zero); %d candidate pairs nonzero", checked, nonzero)
}
