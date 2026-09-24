package qip

import (
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// sigma_generated_test.go (hand-written; survives regeneration): the generated σ-build
// against the element evaluators on the blocks they carry (B00, B01, B11: adc2x) and, for
// the ci scheme over every class, against khci's Slater–Condon CI with the B02 block
// removed (the ISR starts that block at second order). Its B12 and B22 are plain CI by
// construction (generate_adc.py --ci-blocks), so this pins them independently. khci builds
// the general Fock matrix while the σ-build assumes a canonical one: the fixture's 2.3e-11
// off-diagonal Fock is what the 1e-11 relative tolerance absorbs (1.2e-10 at scale 18).

func sigmaCase(t *testing.T) (*fcidump.Data, *integrals.Store, []float64, int) {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join("..", "..", "..", "..", "testdata", "khci", "he3chain_631g_canonical.fcidump"))
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	return d, integrals.New(d, nocc, nil), mp.OrbitalEnergies(d, nocc), nocc
}

func panelApply(op interface {
	ApplyBlock(out, in backend.BlockView)
}, x []float64, n, b int) []float64 {
	be := backend.Gonum{}
	out := be.Alloc(n * b)
	op.ApplyBlock(backend.BlockView{V: out, Rows: n, Cols: b, Ld: n},
		backend.BlockView{V: be.Upload(x), Rows: n, Cols: b, Ld: n})
	return be.Download(out)
}

func denseTimes(m backend.Mat, x []float64, n, b int) []float64 {
	y := make([]float64, n*b)
	for c := range b {
		for r := range n {
			var s float64
			for k := range n {
				s += m.Data[r*n+k] * x[c*n+k]
			}
			y[c*n+r] = s
		}
	}
	return y
}

func checkSigma(t *testing.T, what string, got, want []float64, dg []float64, m backend.Mat) {
	t.Helper()
	var dev, scale float64
	for i := range want {
		dev = math.Max(dev, math.Abs(got[i]-want[i]))
		scale = math.Max(scale, math.Abs(want[i]))
	}
	n := m.Rows
	var ddev float64
	for r := range n {
		ddev = math.Max(ddev, math.Abs(dg[r]-m.Data[r*n+r]))
	}
	t.Logf("%s: n=%d |σ - M·X| %.2e (scale %.2e), |diag| %.2e", what, n, dev, scale, ddev)
	if dev > 1e-11*math.Max(scale, 1) || ddev > 1e-11 {
		t.Errorf("%s: σ or its diagonal differs from the reference", what)
	}
}

func TestSigmaMatchesElements(t *testing.T) {
	_, ints, eps, nocc := sigmaCase(t)
	sp, err := khci.NewSpace(khci.Options{K: K, NOcc: nocc, NVir: ints.NVir(), TwoMs: K % 2, MaxClass: K + 1})
	if err != nil {
		t.Fatal(err)
	}
	el, err := New(ints, eps, nocc, "adc2x")
	if err != nil {
		t.Fatal(err)
	}
	op, err := NewSigma(sp, ints, eps, nocc, "adc2x", backend.Gonum{})
	if err != nil {
		t.Fatal(err)
	}
	n, b := sp.Size(), 3
	rng := rand.New(rand.NewPCG(51, 52))
	x := make([]float64, n*b)
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	m := el.BuildDense(sp)
	checkSigma(t, "adc2x", panelApply(op, x, n, b), denseTimes(m, x, n, b),
		backend.Gonum{}.Download(op.Diagonal(backend.Gonum{})), m)
}

func TestSigmaCIMatchesSlaterCondon(t *testing.T) {
	d, ints, eps, nocc := sigmaCase(t)
	for _, twoMs := range []int{K % 2, K%2 + 2} {
		sp, err := khci.NewSpace(khci.Options{K: K, NOcc: nocc, NVir: ints.NVir(), TwoMs: twoMs, MaxClass: K + 2})
		if err != nil {
			t.Fatal(err)
		}
		op, err := NewSigma(sp, ints, eps, nocc, "ci", backend.Gonum{})
		if err != nil {
			t.Fatal(err)
		}
		ci, err := khci.NewCI(sp, d, backend.Gonum{})
		if err != nil {
			t.Fatal(err)
		}
		n := sp.Size()
		m := backend.NewMat(n, n)
		for r := range n {
			for c := range n {
				cr, cc := sp.Class(r)-K, sp.Class(c)-K
				if (cr == 0 && cc == 2) || (cr == 2 && cc == 0) {
					continue // B02: zero through first order in the ISR, nonzero in CI
				}
				m.Data[r*n+c] = ci.Element(r, c)
			}
		}
		b := 2
		rng := rand.New(rand.NewPCG(53, uint64(twoMs)))
		x := make([]float64, n*b)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		checkSigma(t, fmt.Sprintf("ci 2Ms=%d", twoMs), panelApply(op, x, n, b), denseTimes(m, x, n, b),
			backend.Gonum{}.Download(op.Diagonal(backend.Gonum{})), m)
	}
}
