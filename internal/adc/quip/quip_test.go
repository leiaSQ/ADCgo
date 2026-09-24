package quip

import (
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/isrgen/qip"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"gonum.org/v1/gonum/mat"
)

// The public gate system: a He3 chain (1.30 A) in 6-31G, canonical and localized dumps of
// one SCF (testdata/khci/he3chain_631g_*.in). With three occupied orbitals it carries the
// whole K = 4 class structure 4h | 5h1p | 6h2p.
var khciDir = filepath.Join("..", "..", "..", "testdata", "khci")

// gateSystem is a canonical/localized pair of dumps and sidecars from one SCF.
type gateSystem struct {
	can, loc   *fcidump.Data
	mcan, mloc *mo.Data
	nocc, nvir int
	escf       float64
}

// loadPair reads <dir>/<prefix>_{canonical,localized}.{fcidump,mo.json}.
func loadPair(t *testing.T, dir, prefix string) gateSystem {
	t.Helper()
	var g gateSystem
	var err error
	path := func(kind, ext string) string { return filepath.Join(dir, prefix+"_"+kind+ext) }
	if g.can, err = fcidump.ReadFile(path("canonical", ".fcidump")); err != nil {
		t.Fatal(err)
	}
	if g.loc, err = fcidump.ReadFile(path("localized", ".fcidump")); err != nil {
		t.Fatal(err)
	}
	if g.mcan, err = mo.ReadFile(path("canonical", ".mo.json")); err != nil {
		t.Fatal(err)
	}
	if g.mloc, err = mo.ReadFile(path("localized", ".mo.json")); err != nil {
		t.Fatal(err)
	}
	g.nocc = g.can.NELEC / 2
	g.nvir = g.can.NORB - g.nocc
	ci, err := khci.NewCI(space(t, g, 4), g.can, backend.Gonum{})
	if err != nil {
		t.Fatal(err)
	}
	g.escf = ci.EHF() + g.can.Ecore
	return g
}

func space(t *testing.T, g gateSystem, maxClass int) *khci.Space {
	t.Helper()
	sp, err := khci.NewSpace(khci.Options{K: 4, NOcc: g.nocc, NVir: g.nvir, MaxClass: maxClass})
	if err != nil {
		t.Fatal(err)
	}
	return sp
}

func ciMatrix(t *testing.T, sp *khci.Space, d *fcidump.Data) backend.Mat {
	t.Helper()
	ci, err := khci.NewCI(sp, d, backend.Gonum{})
	if err != nil {
		t.Fatal(err)
	}
	return ci.BuildMatrix()
}

// TestRotationMapsCanonicalCIToLocalized: the rotated representation is exact.
// R built from the two sidecars is orthogonal, and R^T M_CI(canonical) R equals the CI
// matrix built directly in the localized orbitals, element by element, over the full
// 4h | 5h1p | 6h2p space — which checks the hole minors, the particle-pair minors and the
// khci phase convention together.
func TestRotationMapsCanonicalCIToLocalized(t *testing.T) {
	g := loadPair(t, khciDir, "he3chain_631g")
	U, W, err := Rotations(g.mcan, g.mloc, g.nocc, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	sp := space(t, g, 6)
	R, err := Rotation(sp, U, W)
	if err != nil {
		t.Fatal(err)
	}
	n := sp.Size()
	r := mat.NewDense(n, n, R.Data)
	var rtr mat.Dense
	rtr.Mul(r.T(), r)
	var orth float64
	for i := range n {
		for j := range n {
			want := 0.0
			if i == j {
				want = 1
			}
			orth = math.Max(orth, math.Abs(rtr.At(i, j)-want))
		}
	}
	rot := Rotate(ciMatrix(t, sp, g.can), R)
	loc := ciMatrix(t, sp, g.loc)
	var worst, scale float64
	for i := range rot.Data {
		worst = math.Max(worst, math.Abs(rot.Data[i]-loc.Data[i]))
		scale = math.Max(scale, math.Abs(loc.Data[i]))
	}
	t.Logf("%d rows: ||R^T R - 1||max %.1e; max |R^T M_can R - M_loc| %.1e (max |M| %.2f)", n, orth, worst, scale)
	if orth > 1e-10 || worst > 1e-10*scale {
		t.Errorf("rotated representation not exact: orthogonality %.2e, elements %.2e", orth, worst)
	}
}

// TestAssemblyBlocks: every block of Op comes from its source — the generated qip
// evaluator on 4h and 5h1p rows, zero on 4h/6h2p, khci.CI on 5h1p/6h2p and (x, f) on
// 6h2p/6h2p, orbital-energy differences on 6h2p/6h2p for m — and the matrix is symmetric.
// Runs for every scheme the committed qip package carries.
func TestAssemblyBlocks(t *testing.T) {
	g := loadPair(t, khciDir, "he3chain_631g")
	ran := 0
	for scheme, maxClass := range Schemes() {
		if _, ok := qip.Schemes[scheme]; !ok {
			t.Logf("%s: not in the committed isrgen/qip (needs the second-order 5h1p/5h1p block)", scheme)
			continue
		}
		ran++
		sp := space(t, g, maxClass)
		op, err := New(sp, g.can, scheme)
		if err != nil {
			t.Fatal(err)
		}
		ci, err := khci.NewCI(sp, g.can, backend.Gonum{})
		if err != nil {
			t.Fatal(err)
		}
		M := op.BuildMatrix()
		var asym float64
		counts := map[string]int{}
		for r := range sp.Size() {
			for c := range sp.Size() {
				asym = math.Max(asym, math.Abs(M.At(r, c)-M.At(c, r)))
				cr, cc := sp.Class(r)-4, sp.Class(c)-4
				if cr > cc {
					continue
				}
				v := M.At(r, c)
				switch {
				case cr == 0 && cc == 2:
					if v != 0 {
						t.Fatalf("%s: 4h/6h2p element (%d,%d) = %g, want 0", scheme, r, c, v)
					}
				case cc == 2 && (cr == 1 || op.orders[2] == 1):
					if v != ci.Element(r, c) {
						t.Fatalf("%s: CI block element (%d,%d) = %g, CI %g", scheme, r, c, v, ci.Element(r, c))
					}
				case cr == 2:
					want := 0.0
					if r == c {
						want = op.zerothDiagonal(r)
					}
					if v != want {
						t.Fatalf("%s: zeroth-order 6h2p element (%d,%d) = %g, want %g", scheme, r, c, v, want)
					}
				}
				counts[[]string{"4h", "5h1p", "6h2p"}[cr]+"/"+[]string{"4h", "5h1p", "6h2p"}[cc]]++
			}
		}
		t.Logf("%s: %d rows, asymmetry %.1e, blocks %v", scheme, sp.Size(), asym, counts)
		if asym != 0 {
			t.Errorf("%s: matrix not symmetric (%.2e)", scheme, asym)
		}
	}
	if ran == 0 {
		t.Fatal("no scheme of quip is carried by isrgen/qip")
	}
}

// TestSigmaMatchesAssembly: NewSigma (the generated σ program, B12/B22 as plain CI) applies
// the matrix New assembles element by element, for every scheme the σ program covers; a
// scheme needing a block it was not derived to is refused rather than truncated.
func TestSigmaMatchesAssembly(t *testing.T) {
	g := loadPair(t, khciDir, "he3chain_631g")
	rng := rand.New(rand.NewPCG(61, 62))
	ran := 0
	for scheme, maxClass := range Schemes() {
		sp := space(t, g, maxClass)
		sop, err := NewSigma(sp, g.can, scheme, backend.Gonum{})
		orders := sigmaOrders[scheme]
		derived := true
		for b, o := range orders {
			if o > qip.SigmaOrders[b] {
				derived = false
			}
		}
		if !derived {
			if err == nil {
				t.Errorf("%s: NewSigma accepted a scheme the σ program was not derived to", scheme)
			}
			t.Logf("%s: σ refused as expected: %v", scheme, err)
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := qip.Schemes[scheme]; !ok {
			continue // no element assembly to compare with
		}
		op, err := New(sp, g.can, scheme)
		if err != nil {
			t.Fatal(err)
		}
		M := op.BuildMatrix()
		n, b := sp.Size(), 3
		x := make([]float64, n*b)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		be := backend.Gonum{}
		out := be.Alloc(n * b)
		sop.ApplyBlock(backend.BlockView{V: out, Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: be.Upload(x), Rows: n, Cols: b, Ld: n})
		got := be.Download(out)
		var dev, scale float64
		for c := range b {
			for r := range n {
				var s float64
				for k := range n {
					s += M.At(r, k) * x[c*n+k]
				}
				dev = math.Max(dev, math.Abs(got[c*n+r]-s))
				scale = math.Max(scale, math.Abs(s))
			}
		}
		t.Logf("%s: n=%d |σ - M·X| %.2e (scale %.2e)", scheme, n, dev, scale)
		if dev > 1e-11*math.Max(scale, 1) {
			t.Errorf("%s: σ differs from the assembled operator by %.2e", scheme, dev)
		}
		ran++
	}
	if ran == 0 {
		t.Fatal("no scheme compared")
	}
}
