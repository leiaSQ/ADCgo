package sip

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

func buildH2O(t *testing.T, order int) *Matrix {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace(nocc, d.NORB, nil, 0)
	return New(sp, integrals.New(d, nocc, nil), eps, order, backend.Gonum{})
}

// TestMatrixSymmetric: the assembled IP matrix must be symmetric.
func TestMatrixSymmetric(t *testing.T) {
	for _, order := range []int{2, 3} {
		mx := buildH2O(t, order)
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
			t.Errorf("order %d: matrix asymmetry %g exceeds 1e-10", order, maxAsym)
		}
	}
}

// TestApplyEqualsBuild is the primary correctness gate: the matrix-free ApplyFull
// reproduces every column of the densely-built matrix.
func TestApplyEqualsBuild(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping column-by-column Apply==Build check in -short mode")
	}
	for _, order := range []int{2, 3} {
		mx := buildH2O(t, order)
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
				if d := math.Abs(col[i] - M.At(i, j)); d > maxErr {
					maxErr = d
				}
			}
		}
		if maxErr > 1e-10 {
			t.Errorf("order %d: ApplyFull vs BuildMatrix max diff %g exceeds 1e-10", order, maxErr)
		}
	}
}

// TestDenseSpectrumSane: the lowest ionization energy is a physical, positive
// Koopmans-like value with a strong (near-unity) spectroscopic factor.
func TestDenseSpectrumSane(t *testing.T) {
	for _, order := range []int{2, 3} {
		mx := buildH2O(t, order)
		M := mx.BuildMatrix()
		evals, evecs := mx.be.SymEig(M)
		main := mx.Space().MainBlockSize()

		// H2O first IP ≈ 0.4–0.5 Ha (≈ 12 eV).
		if evals[0] < 0.2 || evals[0] > 1.0 {
			t.Errorf("order %d: lowest IP %g Ha outside physical window", order, evals[0])
		}
		var maxSF float64
		for st := range evals {
			var sf float64
			for c := range main {
				sf += evecs.At(c, st) * evecs.At(c, st)
			}
			if sf > maxSF {
				maxSF = sf
			}
			if sf > 1.0001 {
				t.Fatalf("order %d state %d spectroscopic factor %g exceeds 1", order, st, sf)
			}
		}
		if maxSF < 0.8 {
			t.Errorf("order %d: no strong main line (max SF %g)", order, maxSF)
		}
	}
}

// BenchmarkC11_3sums times the order-3 main-block element, whose C^(A) term is the
// O(nvir^4 * nocc) five-deep loop that dominates sip.mainBlock(). H2O/DZP is far smaller than
// a production system (nvir 24 vs the production system's 154) and its ERI fits in L3, so this understates
// the gain from moving the l-invariant reads into compact tables — it is a lower bound.
func BenchmarkC11_3sums(b *testing.B) {
	fc := filepath.Join("..", "..", "..", "testdata", "h2o_dzp.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		b.Skipf("fcidump unavailable: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace(nocc, d.NORB, d.OrbSym, 0)
	el := newElements(sp, integrals.New(d, nocc, d.OrbSym), eps, 3)

	b.ReportAllocs()
	for b.Loop() {
		var acc float64
		for i := range nocc {
			for j := 0; j <= i; j++ {
				c, f1, f2 := el.c11_3sums(i, j)
				acc += c + f1 + f2
			}
		}
		_ = acc
	}
}

// TestDavidsonMatchesDenseSIP checks the block-Davidson driver against the dense path on
// an IP-ADC(2) SIP sector, covering the sip.Matrix.Diagonal() preconditioner surface (the
// c11/c22diag block-diagonal assembly). The lowest NRoots eigenvalues and pole strengths
// must agree to a tight tolerance when converged to a small residual.
func TestDavidsonMatchesDenseSIP(t *testing.T) {
	be := backend.Gonum{}
	const nr = 6
	const tolE, tolPS = 1e-7, 1.0

	_, mxRef, _ := dipoleSpace(t)
	ref := lanczos.SolveDense(mxRef, be)
	if len(ref.Values) < nr {
		t.Fatalf("dense SIP spectrum has only %d states, need %d", len(ref.Values), nr)
	}

	_, mx, _ := dipoleSpace(t)
	res := lanczos.SolveDavidson(mx, be, lanczos.Options{NRoots: nr, ConvThr: 1e-9, MaxDim: 2 * nr, MaxIters: 500})
	if len(res.Values) != nr {
		t.Fatalf("davidson returned %d roots, want %d", len(res.Values), nr)
	}
	for k := range nr {
		if e := math.Abs(res.Values[k] - ref.Values[k]); e > tolE {
			t.Errorf("root %d: davidson %.10f dense %.10f Δ=%.2e", k, res.Values[k], ref.Values[k], e)
		}
		if p := math.Abs(res.PS[k] - ref.PS[k]); p > tolPS {
			t.Errorf("root %d PS: davidson %.4f dense %.4f", k, res.PS[k], ref.PS[k])
		}
	}
}
