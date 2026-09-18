package sip

import (
	"fmt"
	"math"
	"testing"
)

// primElem2 is the exact first-order primitive 2h1p/2h1p element via Slater-Condon.
func (e *elements) primElem2(a, k, l, ap, kp, lp sorb) float64 {
	da, sa := excite(e.nocc, []sorb{l, k}, []sorb{a})
	db, sb := excite(e.nocc, []sorb{lp, kp}, []sorb{ap})
	if da == nil || db == nil {
		return 0
	}
	xa := exc{H: []int{k.Idx(), l.Idx()}, P: []int{a.Idx()}}
	xb := exc{H: []int{kp.Idx(), lp.Idx()}, P: []int{ap.Idx()}}
	sortSmall(xa.H)
	sortSmall(xb.H)
	return sa * sb * e.hamNO(xa, xb)
}

func d(x, y sorb) float64 {
	if x == y {
		return 1
	}
	return 0
}

// core is the A8 expression WITHOUT any permutation applied.
func (e *elements) a8core(a, k, l, ap, kp, lp sorb) float64 {
	return d(a, ap)*e.eriSO(kp, lp, k, l) -
		(d(k, kp)*e.eriSO(lp, a, l, ap) + d(l, lp)*e.eriSO(kp, a, k, ap))
}

// TestA8PermutationConvention settles what the appendix's [k <-> l] operator
// means, by brute force rather than by reading.
//
// Eq. (A8) is a FIRST-order element, so its exact value is independently known
// from Slater-Condon (slater.go, gated against theADCcode-validated reference
// elements by TestSlaterGateC22). Every plausible reading of the permutation
// bracket is evaluated against that value, over samples that include the
// cross-pairing cases l == k' and k == l' which are the only place the
// permutation terms contribute at all.
//
// Exactly one reading survives: the already-antisymmetric leading term is left
// alone and the bracketed group is ANTISYMMETRIZED, -B(k,l) + B(l,k). That is the
// reading pt2.go applies to Eq. (A6), where TestPT2GateC12_2 then confirms it
// against c12_2. A10 and A13 need no inference — they print all four signs of the
// double antisymmetrizer explicitly.
func TestA8PermutationConvention(t *testing.T) {
	mx := buildH2O(t, 3)
	e := mx.el
	nocc := e.nocc

	type cand struct {
		name string
		f    func(a, k, l, ap, kp, lp sorb) float64
	}
	cands := []cand{
		{"core only", e.a8core},
		{"core + [k<->l] (plus)", func(a, k, l, ap, kp, lp sorb) float64 {
			return e.a8core(a, k, l, ap, kp, lp) + e.a8core(a, l, k, ap, kp, lp)
		}},
		{"core - [k<->l]", func(a, k, l, ap, kp, lp sorb) float64 {
			return e.a8core(a, k, l, ap, kp, lp) - e.a8core(a, l, k, ap, kp, lp)
		}},
		{"core + [k'<->l'] (plus)", func(a, k, l, ap, kp, lp sorb) float64 {
			return e.a8core(a, k, l, ap, kp, lp) + e.a8core(a, k, l, ap, lp, kp)
		}},
		{"core - [k'<->l']", func(a, k, l, ap, kp, lp sorb) float64 {
			return e.a8core(a, k, l, ap, kp, lp) - e.a8core(a, k, l, ap, lp, kp)
		}},
		{"core - cross-delta terms", func(a, k, l, ap, kp, lp sorb) float64 {
			return e.a8core(a, k, l, ap, kp, lp) -
				(d(l, kp)*e.eriSO(lp, a, k, ap) + d(k, lp)*e.eriSO(kp, a, l, ap))
		}},
		{"core + cross-delta terms", func(a, k, l, ap, kp, lp sorb) float64 {
			return e.a8core(a, k, l, ap, kp, lp) +
				(d(l, kp)*e.eriSO(lp, a, k, ap) + d(k, lp)*e.eriSO(kp, a, l, ap))
		}},
		{"delta-terms only get [k<->l]", func(a, k, l, ap, kp, lp sorb) float64 {
			del := func(k, l sorb) float64 {
				return -(d(k, kp)*e.eriSO(lp, a, l, ap) + d(l, lp)*e.eriSO(kp, a, k, ap))
			}
			return d(a, ap)*e.eriSO(kp, lp, k, l) + del(k, l) + del(l, k)
		}},
	}

	// Sample primitive configurations with k<l, k'<l'.
	var samples [][6]sorb
	mk := func(o int, dn bool) sorb { return sorb{Orb: o, Dn: dn} }
	for _, t1 := range [][6]sorb{
		{mk(nocc, false), mk(0, false), mk(1, true), mk(nocc, false), mk(0, false), mk(1, true)},
		{mk(nocc, false), mk(0, false), mk(1, true), mk(nocc, false), mk(0, true), mk(1, false)},
		{mk(nocc+1, true), mk(1, false), mk(2, true), mk(nocc+1, true), mk(1, false), mk(3, true)},
		{mk(nocc, false), mk(0, false), mk(2, false), mk(nocc+2, false), mk(0, false), mk(2, false)},
		{mk(nocc, true), mk(0, true), mk(3, false), mk(nocc, true), mk(1, true), mk(3, false)},
		{mk(nocc+1, false), mk(2, false), mk(4, true), mk(nocc+1, false), mk(2, false), mk(4, true)},
		// Cross-pairing cases: l == k', and k == l'. These are exactly what a
		// [k <-> l] permutation term would generate, so they discriminate.
		{mk(nocc, false), mk(0, false), mk(1, false), mk(nocc, false), mk(1, false), mk(2, false)},
		{mk(nocc, false), mk(1, false), mk(2, false), mk(nocc, false), mk(0, false), mk(1, false)},
		{mk(nocc, true), mk(0, true), mk(2, true), mk(nocc, true), mk(2, true), mk(3, true)},
		{mk(nocc+1, false), mk(0, false), mk(3, true), mk(nocc+1, false), mk(3, true), mk(4, false)},
	} {
		samples = append(samples, t1)
	}

	for _, c := range cands {
		var worst float64
		var detail string
		for _, s := range samples {
			want := e.primElem2(s[0], s[1], s[2], s[3], s[4], s[5])
			// A8 is the FIRST-order part; strip the zeroth order of A7.
			if s[0] == s[3] && s[1] == s[4] && s[2] == s[5] {
				want -= e.epsOf(s[0]) - e.epsOf(s[1]) - e.epsOf(s[2])
			}
			got := c.f(s[0], s[1], s[2], s[3], s[4], s[5])
			if dd := math.Abs(got - want); dd > worst {
				worst = dd
				detail = fmt.Sprintf("want %.10g got %.10g", want, got)
			}
		}
		t.Logf("%-32s max deviation %.4g   %s", c.name, worst, detail)
	}
}
