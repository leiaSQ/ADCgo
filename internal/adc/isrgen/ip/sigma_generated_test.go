package ip

import (
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
// against the generated per-element evaluators. Elem.BuildDense is checked against
// adcgen's fully expanded expressions (fidelity_test.go) and against package sip
// (types_test.go), so agreement here pins the σ program — its mvp derivation, spin
// integration, contraction order and normalization — to the same matrix.

type sigmaSystem struct {
	name string
	ints *integrals.Store
	eps  []float64
	nocc int
	nvir int
}

func loadSigmaSystem(t *testing.T, rel ...string) sigmaSystem {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join(append([]string{"..", "..", "..", ".."}, rel...)...))
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	return sigmaSystem{name: rel[len(rel)-1], ints: integrals.New(d, nocc, nil),
		eps: mp.OrbitalEnergies(d, nocc), nocc: nocc, nvir: d.NORB - nocc}
}

// denseApply is M·X for a row-major M and a column-major panel X.
func denseApply(m backend.Mat, x []float64, n, b int) []float64 {
	y := make([]float64, n*b)
	for c := range b {
		for r := range n {
			var s float64
			row := m.Data[r*n : (r+1)*n]
			for k, v := range row {
				s += v * x[c*n+k]
			}
			y[c*n+r] = s
		}
	}
	return y
}

func maxDev(a, b []float64) (dev, scale float64) {
	for i := range a {
		dev = math.Max(dev, math.Abs(a[i]-b[i]))
		scale = math.Max(scale, math.Abs(b[i]))
	}
	return dev, scale
}

func TestSigmaMatchesDense(t *testing.T) {
	h2o := loadSigmaSystem(t, "testdata", "h2o.fcidump")
	he3 := loadSigmaSystem(t, "testdata", "khci", "he3chain_631g_canonical.fcidump")
	cases := []struct {
		sys      sigmaSystem
		scheme   string
		twoMs    int
		maxClass int
	}{
		{h2o, "strict:3", 1, 2},
		{h2o, "adc22x", 1, 2},
		{he3, "strict:3", 1, 3},
		{he3, "adc22m", 1, 3},
		{he3, "adc22x", 1, 3},
		{he3, "adc22f", 1, 3},
		{he3, "adc22f", 3, 3},
		{he3, "adc22f", 1, 2}, // the 3h2p blocks drop with their class
	}
	be := backend.Gonum{}
	rng := rand.New(rand.NewPCG(11, 12))
	for _, c := range cases {
		sp, err := khci.NewSpace(khci.Options{K: 1, NOcc: c.sys.nocc, NVir: c.sys.nvir,
			TwoMs: c.twoMs, MaxClass: c.maxClass})
		if err != nil {
			t.Fatal(err)
		}
		el, err := New(c.sys.ints, c.sys.eps, c.sys.nocc, c.scheme)
		if err != nil {
			t.Fatal(err)
		}
		op, err := NewSigma(sp, c.sys.ints, c.sys.eps, c.sys.nocc, c.scheme, be)
		if err != nil {
			t.Fatal(err)
		}
		n, b := sp.Size(), 3
		m := el.BuildDense(sp)
		x := make([]float64, n*b)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		out := be.Alloc(n * b)
		op.ApplyBlock(backend.BlockView{V: out, Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: be.Upload(x), Rows: n, Cols: b, Ld: n})
		dev, scale := maxDev(be.Download(out), denseApply(m, x, n, b))
		tag := c.sys.name + " " + c.scheme
		t.Logf("%s 2Ms=%d class<=%d: n=%d, |σ - M·X| max %.2e (scale %.2e)", tag, c.twoMs, c.maxClass, n, dev, scale)
		if dev > 1e-11*math.Max(scale, 1) {
			t.Errorf("%s 2Ms=%d: σ differs from BuildDense·X by %.3e", tag, c.twoMs, dev)
		}

		// the diagonal
		dg := be.Download(op.Diagonal(be))
		var ddev float64
		for r := range n {
			ddev = math.Max(ddev, math.Abs(dg[r]-m.Data[r*n+r]))
		}
		if ddev > 1e-11 {
			t.Errorf("%s 2Ms=%d: Diagonal differs from diag(BuildDense) by %.3e", tag, c.twoMs, ddev)
		}

		// the satellite-only apply: M with every row and column of the main class removed
		main := sp.MainBlockSize()
		ms := backend.NewMat(n, n)
		for r := main; r < n; r++ {
			for k := main; k < n; k++ {
				ms.Data[r*n+k] = m.Data[r*n+k]
			}
		}
		op.ApplyBlockSatellite(backend.BlockView{V: out, Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: be.Upload(x), Rows: n, Cols: b, Ld: n})
		if dev, _ := maxDev(be.Download(out), denseApply(ms, x, n, b)); dev > 1e-11*math.Max(scale, 1) {
			t.Errorf("%s 2Ms=%d: satellite apply differs by %.3e", tag, c.twoMs, dev)
		}

		// a restricted space: the operator on it is the sub-block of M
		var rows []int
		for r := range n {
			if rng.IntN(3) > 0 {
				rows = append(rows, r)
			}
		}
		sub, err := sp.Restrict(rows)
		if err != nil {
			t.Fatal(err)
		}
		ops, err := NewSigma(sub, c.sys.ints, c.sys.eps, c.sys.nocc, c.scheme, be)
		if err != nil {
			t.Fatal(err)
		}
		ns := len(rows)
		mr := backend.NewMat(ns, ns)
		for i, r := range rows {
			for j, k := range rows {
				mr.Data[i*ns+j] = m.Data[r*n+k]
			}
		}
		xs := x[:ns*b]
		os := be.Alloc(ns * b)
		ops.ApplyBlock(backend.BlockView{V: os, Rows: ns, Cols: b, Ld: ns},
			backend.BlockView{V: be.Upload(xs), Rows: ns, Cols: b, Ld: ns})
		if dev, _ := maxDev(be.Download(os), denseApply(mr, xs, ns, b)); dev > 1e-11*math.Max(scale, 1) {
			t.Errorf("%s 2Ms=%d: restricted σ differs by %.3e", tag, c.twoMs, dev)
		}
	}
}
