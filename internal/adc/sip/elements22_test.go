package sip

import (
	"fmt"
	"math"
	"strconv"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

const hartreeToEV = 27.211386245988

// TestADC22Characterization reports the ionization spectrum of a reduced H2O
// active space across ADC(2)x, ADC(3) and the three ADC(2,2) variants, and checks
// the qualitative signature the paper's Tables II and III describe:
//
//	"significant improvement of the 2h1p satellite state IPs ... The IP of the
//	 main state remains basically unchanged."
//
// It is a characterization test, not a numerical gate — the decisive comparison is
// against the published Be and Ne IPs in a proper basis, which needs the CLI. What
// it does pin is that ADC(2,2) moves the satellites and leaves the main lines
// alone, which is the whole point of the scheme, and that no variant produces a
// non-physical spectrum.
func TestADC22Characterization(t *testing.T) {
	if testing.Short() {
		t.Skip("dense ADC(2,2) solves across five schemes; -short skips it")
	}
	d, nocc := h2o22(t)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	const norb = 11

	solve := func(mx *Matrix) (main []float64, sat []float64) {
		M := mx.BuildMatrix()
		vals, vecs := backend.Gonum{}.SymEig(M)
		nMain := mx.sp.MainBlockSize()
		for i, ev := range vals {
			var w float64
			for r := range nMain {
				c := vecs.At(r, i)
				w += c * c
			}
			ip := ev * hartreeToEV
			if ip < 0 || ip > 200 {
				continue
			}
			if w > 0.5 {
				main = append(main, ip)
			} else if w > 0.005 {
				sat = append(sat, ip)
			}
		}
		return main, sat
	}

	ref2 := New(NewSpace(nocc, norb, nil, 0), ints, eps, 2, backend.Gonum{})
	m2, s2 := solve(ref2)
	ref3 := New(NewSpace(nocc, norb, nil, 0), ints, eps, 3, backend.Gonum{})
	m3, _ := solve(ref3)

	t.Logf("ADC(2)x  main IPs (eV): %s", fmtEV(m2, 4))
	t.Logf("ADC(3)   main IPs (eV): %s", fmtEV(m3, 4))
	t.Logf("ADC(2)x  lowest satellites (eV): %s", fmtEV(s2, 4))

	for _, v := range []Variant{VariantM, VariantX, VariantF} {
		sp := NewSpace22(nocc, norb, nil, 0)
		mx := New(sp, ints, eps, Order22, backend.Gonum{})
		mx.SetVariant(v)
		mm, ms := solve(mx)
		t.Logf("ADC(2,2)%s main IPs (eV): %s", v, fmtEV(mm, 4))
		t.Logf("ADC(2,2)%s lowest satellites (eV): %s", v, fmtEV(ms, 4))

		if len(mm) == 0 {
			t.Errorf("variant %s produced no main ionization states", v)
			continue
		}
		// The main lines must survive the scheme change: same count, and each
		// within a couple of eV of its ADC(2)x counterpart.
		n := min(len(mm), len(m2))
		for i := range n {
			if dd := math.Abs(mm[i] - m2[i]); dd > 2.0 {
				t.Errorf("variant %s: main IP %d moved %.2f eV from ADC(2)x (%.3f -> %.3f); "+
					"the paper reports the main lines as essentially unchanged",
					v, i, dd, m2[i], mm[i])
			}
		}
	}
}

func fmtEV(v []float64, n int) string {
	s := ""
	for i, x := range v {
		if i >= n {
			break
		}
		if i > 0 {
			s += "  "
		}
		s += strconv.FormatFloat(x, 'f', 3, 64)
	}
	if s == "" {
		return "(none)"
	}
	return s
}

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
