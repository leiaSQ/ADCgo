package fano

import (
	"math"
	"sort"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
	"github.com/leiaSQ/ADCgo/internal/adc/stieltjes"
)

// lorentzModel is a discretized continuum with an analytic width: PHP = diag(E_j) on a
// uniform grid of spacing delta, and |g_j|^2 = Gamma(E_j) delta / (2 pi) with
// Gamma(E) = gamma0 w^2 / ((E - e0)^2 + w^2). The exact width at e0 is gamma0.
type lorentzModel struct {
	e, g          []float64
	e0, w, gamma0 float64
}

func newLorentzModel(n int, gamma0 float64) lorentzModel {
	m := lorentzModel{e: make([]float64, n), g: make([]float64, n), e0: 2.0, w: 0.3, gamma0: gamma0}
	lo, hi := 0.5, 3.5
	delta := (hi - lo) / float64(n)
	for j := range n {
		e := lo + (float64(j)+0.5)*delta
		m.e[j] = e
		gam := gamma0 * m.w * m.w / ((e-m.e0)*(e-m.e0) + m.w*m.w)
		m.g[j] = math.Sqrt(gam * delta / (2 * math.Pi))
	}
	return m
}

func (m lorentzModel) apply(dst, src []float64) {
	for i := range dst {
		dst[i] = m.e[i] * src[i]
	}
}

// TestWidthEnginesRecoverLorentzian: W-B, W-B2 and W-C recover the analytic
// width of a Lorentzian coupling model, at every relative width from 1e-2 down to 1e-18
// of the coupling scale — widths of 1e-15 Eh and below. Nothing in the
// engines has an absolute cut, so the relative accuracy must not depend on the scale.
func TestWidthEnginesRecoverLorentzian(t *testing.T) {
	opts := stieltjes.Options{MaxOrder: 120}
	for _, g0 := range []float64{1e-2, 1e-6, 1e-10, 1e-14, 1e-18} {
		m := newLorentzModel(4000, g0)
		wb, _, err := GaussWidth(m.apply, m.g, 160, m.e0, opts)
		if err != nil {
			t.Fatal(err)
		}
		wb2, err := ShiftInvertWidth(m.apply, m.g, 60, m.e0,
			ShiftInvertOptions{Sigma: m.e0 + 1e-4, Diag: m.e}, stieltjes.Options{MaxOrder: 60})
		if err != nil {
			t.Fatal(err)
		}
		lo, hi, err := SpectralBounds(m.apply, m.g, 40, 0.05)
		if err != nil {
			t.Fatal(err)
		}
		wc, err := KPMWidth(m.apply, m.g, m.e0, 1000, lo, hi)
		if err != nil {
			t.Fatal(err)
		}
		rel := func(x float64) float64 { return math.Abs(x-g0) / g0 }
		t.Logf("Gamma0 %.0e: W-B %.6e (+/- %.1e, rel err %.1e), W-B2 %.6e (rel err %.1e), KPM %.6e (rel err %.1e)",
			g0, wb.Gamma, wb.Sigma, rel(wb.Gamma), wb2.Gamma, rel(wb2.Gamma), wc, rel(wc))
		for name, x := range map[string]float64{"W-B": wb.Gamma, "W-B2": wb2.Gamma, "W-C": wc} {
			if rel(x) > 1e-2 {
				t.Errorf("Gamma0 %.0e: %s gives %.6e, relative error %.2e", g0, name, x, rel(x))
			}
		}
	}
}

// TestGaussWidthHomogeneous: Gamma[lambda g] = lambda^2 Gamma[g] for lambda
// from 1e-8 to 1e3, to rounding. This is the property that makes the noise floor
// relative (decision D6) and that the eigen path's absolute gamma cut breaks.
func TestGaussWidthHomogeneous(t *testing.T) {
	m := newLorentzModel(2000, 1e-3)
	opts := stieltjes.Options{MaxOrder: 80}
	base, _, err := GaussWidth(m.apply, m.g, 100, m.e0, opts)
	if err != nil {
		t.Fatal(err)
	}
	for _, lam := range []float64{1e-8, 1e-3, 1e3} {
		g := make([]float64, len(m.g))
		for i := range g {
			g[i] = lam * m.g[i]
		}
		w, _, err := GaussWidth(m.apply, g, 100, m.e0, opts)
		if err != nil {
			t.Fatal(err)
		}
		r := w.Gamma / (lam * lam * base.Gamma)
		t.Logf("lambda %.0e: Gamma / (lambda^2 Gamma_1) - 1 = %.1e", lam, r-1)
		if math.Abs(r-1) > 1e-10 {
			t.Errorf("lambda %.0e: homogeneity broken by %.2e", lam, r-1)
		}
	}
}

// TestGaussRuleIsEigenMeasure: on the SIP ADC(2,2) fixture of
// TestSchemeAPipeline, with the O 2s vacancy (the O 1s one lies at 20 Eh, above this
// truncated 8-orbital space's pseudo-continuum, so no path can image it), the seeded Lanczos run over the whole P space reproduces the eigen
// path's discrete measure {E_j, gamma_j} exactly (their cumulative distributions agree to
// 1e-10 of the total), and the imaged widths of the two paths agree within their Stieltjes
// spreads.
func TestGaussRuleIsEigenMeasure(t *testing.T) {
	parent, pmx, build := h2o22(t, 8, sip.VariantF)
	const vacancy = 1
	sel, err := NewHoleLocalization(AnyHole, []int{vacancy}, "initial vacancy")
	if err != nil {
		t.Fatal(err)
	}
	part := NewPartition(parent, sel)
	be := backend.Gonum{}
	qsp, psp := parent.Restrict(part.Q), parent.Restrict(part.P)
	phi, err := SelectDiscrete(qsp, lanczos.SolveDense(build(qsp), be), vacancy, 0, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	g, err := Coupling(pmx, part, phi, be)
	if err != nil {
		t.Fatal(err)
	}
	pop := build(psp)
	pres := lanczos.SolveDense(pop, be)
	ps, err := Widths(psp, pres, g, Filter{EMax: -1, MinWeight: -1, MinGamma: -1})
	if err != nil {
		t.Fatal(err)
	}
	n := psp.Size()
	apply := func(dst, src []float64) {
		out := be.Alloc(n)
		pop.ApplyFull(out, be.Upload(src))
		copy(dst, be.Download(out))
	}
	sr, err := lanczos.SolveSeeded(apply, g, n)
	if err != nil {
		t.Fatal(err)
	}
	nodes, w, err := sr.GaussRule(sr.Steps())
	if err != nil {
		t.Fatal(err)
	}
	// cumulative distributions on the union of abscissae
	type pt struct{ e, w float64 }
	var eig, gau []pt
	var total float64
	for j := range ps.Energy {
		eig = append(eig, pt{ps.Energy[j], ps.Gamma[j]})
		total += ps.Gamma[j]
	}
	for i := range nodes {
		gau = append(gau, pt{nodes[i], 2 * math.Pi * w[i]})
	}
	sort.Slice(eig, func(a, b int) bool { return eig[a].e < eig[b].e })
	cum := func(ps []pt, e float64) float64 {
		var s float64
		for _, p := range ps {
			if p.e <= e+1e-9 {
				s += p.w
			}
		}
		return s
	}
	var worst float64
	for _, p := range append(append([]pt(nil), eig...), gau...) {
		worst = max(worst, math.Abs(cum(eig, p.e)-cum(gau, p.e)))
	}
	t.Logf("P %d rows: Lanczos closed after %d steps (invariant %v); max |F_gauss - F_eigen| %.1e of total %.4e",
		n, sr.Steps(), sr.Invariant, worst, total)
	if worst > 1e-10*total {
		t.Errorf("the full-length Gauss rule is not the eigen measure: cumulative differs by %.2e", worst)
	}
	we, err := ImageWidth(ps, phi.Energy, stieltjes.Options{})
	if err != nil {
		t.Fatal(err)
	}
	wg, _, err := GaussWidth(apply, g, n, phi.Energy, stieltjes.Options{MaxOrder: 60})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("eigen path %.4f +/- %.4f meV, Gauss engine %.4f +/- %.4f meV", we.MeV, we.SigmaMeV, wg.MeV, wg.SigmaMeV)
	if d := math.Abs(we.MeV - wg.MeV); d > 3*(we.SigmaMeV+wg.SigmaMeV)+0.05*we.MeV {
		t.Errorf("Gauss engine %.4f meV vs eigen path %.4f meV: differ by %.4f, beyond the spreads",
			wg.MeV, we.MeV, d)
	}
}

// TestDecomposeSumsToCoupling: the class pieces g_B of the coupling vector add up to
// g exactly (to rounding), on the SIP ADC(2,2) fixture, and each piece is nonzero only
// through its own block of Phi.
func TestDecomposeSumsToCoupling(t *testing.T) {
	parent, pmx, build := h2o22(t, 8, sip.VariantF)
	sel, err := NewHoleLocalization(AnyHole, []int{1}, "initial vacancy")
	if err != nil {
		t.Fatal(err)
	}
	part := NewPartition(parent, sel)
	be := backend.Gonum{}
	qsp := parent.Restrict(part.Q)
	phi, err := SelectDiscrete(qsp, lanczos.SolveDense(build(qsp), be), 1, 0, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	g, err := Coupling(pmx, part, phi, be)
	if err != nil {
		t.Fatal(err)
	}
	blocks := ClassBlocks(qsp)
	pieces, err := Decompose(pmx, part, phi, blocks, be)
	if err != nil {
		t.Fatal(err)
	}
	sum := make([]float64, len(g))
	var gmax float64
	for name, gb := range pieces {
		var n2 float64
		for i, v := range gb {
			sum[i] += v
			n2 += v * v
		}
		t.Logf("block %s: %d Q rows, ||g_B||^2 = %.6g", name, len(blocks[name]), n2)
	}
	var worst float64
	for i := range g {
		worst = max(worst, math.Abs(sum[i]-g[i]))
		gmax = max(gmax, math.Abs(g[i]))
	}
	t.Logf("%d blocks; max |sum_B g_B - g| = %.1e (max |g| %.3g)", len(pieces), worst, gmax)
	if worst > 1e-13*gmax {
		t.Errorf("the pieces do not add up to g: %.2e", worst)
	}
	if len(pieces) < 2 {
		t.Error("Q has a single class; the decomposition was not exercised")
	}
}
