package sip

import (
	"math"
	"testing"
)

// detOfExc materializes an excitation as a sorted occupation list, so the O(N^2)
// oracle hamElem can be pointed at the same determinant as the O(1) hamNO.
func detOfExc(nocc int, x exc) det {
	d := make(det, 0, 2*nocc)
	for i := range 2 * nocc {
		if !has(x.H, i) {
			d = append(d, i)
		}
	}
	d = append(d, x.P...)
	sortSmall(d)
	return d
}

// TestHamNOMatchesOracle gates the normal-ordered closed forms against the general
// Slater-Condon oracle, which TestSlaterGateC22 has already tied to
// theADCcode-validated reference elements.
//
// The pairs are drawn from the real ADC(2,2) space rather than at random, and
// deliberately span every block the method needs: 2h1p/2h1p, 2h1p/3h2p and
// 3h2p/3h2p, and within those every excitation-difference case (0, 1 and 2
// differing spin orbitals) plus the ones that must vanish.
func TestHamNOMatchesOracle(t *testing.T) {
	d, nocc := h2o22(t)
	mx := buildH2OSpace22(t, 0)
	e := mx.el
	sp := mx.sp
	_ = d

	var terms []detTerm
	for _, cfg := range sp.Configs[sp.BeginSat:min(sp.BeginSat+40, sp.Begin3h2p)] {
		terms = append(terms, e.expandDet2(cfg)...)
	}
	n2 := len(terms)
	for _, cfg := range sp.Sat3[:min(60, len(sp.Sat3))] {
		terms = append(terms, e.expandDet3(cfg)...)
	}

	var maxDiff, maxAbs float64
	var nZero, nOne, nTwo, nNull int
	for i := range terms {
		for j := range terms {
			a, b := terms[i].E, terms[j].E
			want := e.hamElem(detOfExc(nocc, a), detOfExc(nocc, b))
			// hamElem returns <a|H|b>; hamNO returns <a|H - E_0|b>, so the
			// reference energy has to come off the diagonal.
			if _, _, _, ok := diffExc(nocc, a, b); ok {
				if from, _, _, _ := diffExc(nocc, a, b); len(from) == 0 {
					want -= e.refEnergy()
				}
			}
			got := e.hamNO(a, b)
			if from, _, _, ok := diffExc(nocc, a, b); !ok {
				nNull++
				want = 0
			} else {
				switch len(from) {
				case 0:
					nZero++
				case 1:
					nOne++
				default:
					nTwo++
				}
			}
			if math.Abs(want) > maxAbs {
				maxAbs = math.Abs(want)
			}
			if dd := math.Abs(got - want); dd > maxDiff {
				maxDiff = dd
			}
		}
	}
	t.Logf("%d determinants (%d from 2h1p, %d from 3h2p); difference cases: 0->%d 1->%d 2->%d none->%d",
		len(terms), n2, len(terms)-n2, nZero, nOne, nTwo, nNull)
	t.Logf("max |element| = %.6g, max deviation from the oracle = %.3g", maxAbs, maxDiff)
	if nZero == 0 || nOne == 0 || nTwo == 0 {
		t.Fatal("the sample did not cover all three difference cases")
	}
	if maxDiff > 1e-10 {
		t.Errorf("normal-ordered element deviates from the oracle by %.3g (limit 1e-10)", maxDiff)
	}
}

// TestExpandDetMatchesPrimitive re-runs the reference gate of TestSlaterGateC22
// through the fast path: expandDet2 + hamNO must reproduce c22diag / c22off. This
// is what licenses using expandDet3 + hamNO for the 3h2p blocks, where there is no
// reference to compare against.
func TestExpandDetMatchesPrimitive(t *testing.T) {
	mx := buildH2O(t, 3)
	e, sp := mx.el, mx.sp

	c22 := func(row, col Config) float64 {
		var sum float64
		for _, r := range e.expandDet2(row) {
			for _, c := range e.expandDet2(col) {
				sum += r.C * c.C * e.hamNO(r.E, c.E)
			}
		}
		return sum
	}

	var maxDiag, maxOff, maxAbs float64
	n := len(sp.Configs)
	for r := sp.BeginSat; r < n; r++ {
		for c := sp.BeginSat; c < n; c++ {
			var want float64
			if r == c {
				want = e.c22diag(sp.Configs[r])
			} else {
				want = e.c22off(sp.Configs[r], sp.Configs[c])
			}
			got := c22(sp.Configs[r], sp.Configs[c])
			if math.Abs(want) > maxAbs {
				maxAbs = math.Abs(want)
			}
			dd := math.Abs(got - want)
			if r == c {
				maxDiag = math.Max(maxDiag, dd)
			} else {
				maxOff = math.Max(maxOff, dd)
			}
		}
	}
	t.Logf("fast path over %d configs, max |element| = %.6g, deviation: diagonal %.3g, off-diagonal %.3g",
		n-sp.BeginSat, maxAbs, maxDiag, maxOff)
	if maxDiag > 1e-12 || maxOff > 1e-12 {
		t.Errorf("fast-path 2h1p block deviates: diagonal %.3g, off-diagonal %.3g (limit 1e-12)",
			maxDiag, maxOff)
	}
}
