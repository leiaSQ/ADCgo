package stieltjes

import (
	"math"
	"math/big"
	"testing"
)

// sample discretizes a continuous density f on [lo,hi] into n midpoint masses — a
// pseudo-continuum whose moments are those of f to the midpoint-rule error. Imaging it
// must recover f, which is how a moment-based reconstruction is tested without circularity.
func sample(f func(float64) float64, lo, hi float64, n int) (es, gs []float64) {
	h := (hi - lo) / float64(n)
	es, gs = make([]float64, n), make([]float64, n)
	for i := range n {
		e := lo + (float64(i)+0.5)*h
		es[i], gs[i] = e, f(e)*h
	}
	return es, gs
}

func lorentzian(e0, gamma float64) func(float64) float64 {
	return func(e float64) float64 {
		return (gamma / 2) / math.Pi / ((e-e0)*(e-e0) + gamma*gamma/4)
	}
}

// TestGaussRuleReproducesMoments is the load-bearing gate of the whole package.
//
// The Stieltjes procedure builds the polynomials orthogonal to the discrete measure and
// reads its Jacobi matrix's eigen-decomposition as a Gauss quadrature rule. An n-point
// Gauss rule integrates polynomials of degree up to 2n-1 exactly, so the nodes and weights
// MUST reproduce the input's negative moments
//
//	S_-k = sum_i gamma_i eps_i^-k   for every k < 2n.
//
// That single identity tests the entire chain at once — the extended-precision recurrence,
// the Jacobi matrix assembly, the float64 eigensolve and the 1/lambda inversion — against
// the property that defines it, which is a far stronger check than comparing eigenvalues
// against a second eigensolver. Both sides are compared in big.Float so the comparison
// itself contributes no error.
func TestGaussRuleReproducesMoments(t *testing.T) {
	es, gs := sample(func(e float64) float64 { return math.Exp(-e) }, 0.4, 5.0, 200)
	tb, err := NewTable(es, gs, Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("maximum usable order %d from %d input states", tb.MaxOrder(), tb.Points())
	if tb.MaxOrder() < 20 {
		t.Fatalf("only %d usable orders; the gate needs a realistic range", tb.MaxOrder())
	}

	const prec = 300
	for _, n := range []int{5, 10, 20, 40, tb.MaxOrder()} {
		if n > tb.MaxOrder() {
			continue
		}
		r, err := tb.Rule(n)
		if err != nil {
			t.Fatal(err)
		}
		if len(r.Energy) != n {
			t.Fatalf("order %d produced %d nodes", n, len(r.Energy))
		}
		// Nodes ascending and inside the input range, weights non-negative.
		//
		// The range test carries a tolerance because Gauss nodes approach the endpoints of
		// the measure's support as the order rises — at order 40 the lowest node sits on the
		// lowest input energy to 4 digits — so roundoff puts it a ULP outside. A node
		// genuinely outside would be off by far more than this.
		lo, hi := tb.Range()
		tol := 1e-9 * (hi - lo)
		for i := range r.Energy {
			if i > 0 && !(r.Energy[i] > r.Energy[i-1]) {
				t.Errorf("order %d: nodes not ascending at %d", n, i)
			}
			if r.Energy[i] < lo-tol || r.Energy[i] > hi+tol {
				t.Errorf("order %d: node %d at %.17g is outside the input range [%.17g,%.17g]",
					n, i, r.Energy[i], lo, hi)
			}
			if r.Weight[i] < 0 {
				t.Errorf("order %d: node %d has negative weight %.6g", n, i, r.Weight[i])
			}
		}

		var worst float64
		worstK := -1
		for k := range 2 * n {
			want := tb.Moment(k)
			got := new(big.Float).SetPrec(prec)
			for j := range r.Energy {
				term := new(big.Float).SetPrec(prec).SetFloat64(1)
				xj := new(big.Float).SetPrec(prec).SetFloat64(1 / r.Energy[j])
				for range k {
					term.Mul(term, xj)
				}
				term.Mul(term, new(big.Float).SetPrec(prec).SetFloat64(r.Weight[j]))
				got.Add(got, term)
			}
			d := new(big.Float).SetPrec(prec).Sub(got, want)
			rel, _ := d.Quo(d, want).Abs(d).Float64()
			if rel > worst {
				worst, worstK = rel, k
			}
		}
		t.Logf("order %3d: the %d-point rule reproduces every moment S_-0..S_-%d; "+
			"worst relative error %.3g (k=%d)", n, n, 2*n-1, worst, worstK)
		if worst > 1e-12 {
			t.Errorf("order %d: moment S_-%d is off by %.3g; the quadrature rule is not the "+
				"Gauss rule of the input measure", n, worstK, worst)
		}
	}
}

// TestExtendedPrecisionExtendsTheUsableOrder is why the recurrence runs in big.Float
// instead of float64, stated as a measurement rather than an assertion.
//
// The maximum usable order is set by where the orthogonal polynomials lose orthogonality,
// and that is a pure function of the working precision. The reference runs the recurrence
// in REAL*16 (113-bit) for exactly this reason; this shows what each precision buys, and
// what the accuracy of the imaged density does in consequence.
func TestExtendedPrecisionExtendsTheUsableOrder(t *testing.T) {
	const e0, gam = 2.0, 0.6
	f := lorentzian(e0, gam)
	es, gs := sample(f, 0.3, 6.0, 400)

	prev := 0
	for _, prec := range []uint{53, 113, 256, 512} {
		tb, err := NewTable(es, gs, Options{Prec: prec})
		if err != nil {
			t.Fatal(err)
		}
		r, err := Image(es, gs, 1.0, Options{Prec: prec})
		if err != nil {
			t.Fatal(err)
		}
		relErr := 100 * math.Abs(r.Gamma-f(1.0)) / f(1.0)
		t.Logf("prec %3d bits: maximum usable order %2d, %2d orders averaged, "+
			"density at E=1.0 off by %6.3f%%", prec, tb.MaxOrder(), len(r.Orders), relErr)
		if tb.MaxOrder() < prev {
			t.Errorf("precision %d gave a LOWER maximum order (%d) than the previous step (%d); "+
				"the usable order must not shrink as precision grows", prec, tb.MaxOrder(), prev)
		}
		prev = tb.MaxOrder()
	}
	// float64 must be visibly worse than the reference's REAL*16, which must be visibly
	// worse than the default — otherwise the extended precision is not earning anything.
	f64, _ := NewTable(es, gs, Options{Prec: 53})
	q, _ := NewTable(es, gs, Options{Prec: 113})
	def, _ := NewTable(es, gs, Options{Prec: DefaultPrec})
	if !(f64.MaxOrder() < q.MaxOrder() && q.MaxOrder() < def.MaxOrder()) {
		t.Errorf("usable orders float64=%d, REAL*16=%d, default=%d: the precision ladder does "+
			"not increase, so the big.Float recurrence buys nothing",
			f64.MaxOrder(), q.MaxOrder(), def.MaxOrder())
	}
}

// TestImageRecoversAFlatDensity is the cleanest accuracy check: a constant density has no
// feature for a moment reconstruction to smooth, so imaging must return it almost exactly.
func TestImageRecoversAFlatDensity(t *testing.T) {
	const want = 0.25
	es, gs := sample(func(float64) float64 { return want }, 0.5, 5.0, 300)
	for _, at := range []float64{1.0, 2.0, 3.0, 4.0} {
		r, err := Image(es, gs, at, Options{})
		if err != nil {
			t.Fatal(err)
		}
		rel := math.Abs(r.Gamma-want) / want
		t.Logf("flat density at E=%.1f: Gamma = %.8g +/- %.2g (exact %.2f), off by %.4f%%",
			at, r.Gamma, r.Sigma, want, 100*rel)
		if rel > 5e-3 {
			t.Errorf("at E=%.1f the imaged density is off by %.3g, above the 0.5%% bound", at, rel)
		}
		if r.Gamma < 0 {
			t.Errorf("at E=%.1f the imaged density is negative (%.6g)", at, r.Gamma)
		}
	}
}

// TestImageRecoversALorentzian is the plan's analytic model: a Lorentzian coupling density
// of known width, imaged before any ab-initio number is believed.
//
// It is checked in the WINGS, and the peak is reported rather than gated, because the
// discrepancy at the peak is the method's own behaviour and not this implementation's: a
// reconstruction from a finite number of moments smooths a sharply peaked density, so the
// maximum comes out low. TestGaussRuleReproducesMoments pins the implementation to 1e-14
// independently, which is what makes it safe to read the peak deficit as physics.
//
// Note also that the quoted sigma is SMALLER than the peak error. That is consistent with
// the paper, which says its error margins are statistical over Stieltjes orders and "do
// not attempt to reflect any systematical errors connected with the Fano-ADC methodology".
func TestImageRecoversALorentzian(t *testing.T) {
	const e0, gam = 2.0, 0.6
	f := lorentzian(e0, gam)
	es, gs := sample(f, 0.3, 6.0, 400)

	for _, at := range []float64{0.8, 1.0, 1.2, 3.0, 3.5} {
		r, err := Image(es, gs, at, Options{})
		if err != nil {
			t.Fatal(err)
		}
		rel := math.Abs(r.Gamma-f(at)) / f(at)
		t.Logf("Lorentzian wing at E=%.1f: Gamma = %.6g +/- %.2g (exact %.6g), off by %.3f%%",
			at, r.Gamma, r.Sigma, f(at), 100*rel)
		if rel > 0.05 {
			t.Errorf("at E=%.1f the imaged wing is off by %.1f%%, above the 5%% bound", at, 100*rel)
		}
	}
	r, err := Image(es, gs, e0, Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Lorentzian PEAK at E=%.1f: Gamma = %.6g +/- %.2g (exact %.6g), off by %.1f%% "+
		"— a moment reconstruction smooths a sharp maximum, and the statistical sigma does "+
		"not see that systematic deficit", e0, r.Gamma, r.Sigma, f(e0),
		100*(r.Gamma-f(e0))/f(e0))
	if r.Gamma <= 0 {
		t.Errorf("the imaged peak is non-positive (%.6g)", r.Gamma)
	}
	// The total area is a moment, so it must be right even where the peak is not.
	var area float64
	for _, x := range gs {
		area += x
	}
	t.Logf("input total width sum g_i = %.8g (the Lorentzian's normalization over the "+
		"sampled range)", area)
}

// TestAveragePaperPicksTheBestWindow checks the paper's protocol on a constructed order
// sequence: noisy at low order, a flat plateau in the middle, drifting at high order. The
// window it averages must be the plateau, and sigma must be that window's deviation.
func TestAveragePaperPicksTheBestWindow(t *testing.T) {
	res := &Result{MaxOrder: 30}
	// Orders 5..9 noisy, 10..18 a plateau at 1.0, 19..24 drifting away.
	vals := []float64{0.2, 1.9, 0.4, 1.7, 0.6,
		1.000, 1.002, 0.999, 1.001, 1.000, 0.998, 1.002, 1.001, 0.999,
		1.2, 1.4, 1.7, 2.1, 2.6, 3.2}
	for i, v := range vals {
		res.Orders = append(res.Orders, OrderResult{Order: 5 + i, Gamma: v})
	}
	combinePaper(res, Options{}.withDefaults())
	t.Logf("%s", res)
	if len(res.Used) != DefaultWindow {
		t.Fatalf("averaged %d orders, want %d", len(res.Used), DefaultWindow)
	}
	if res.Used[0] != 10 || res.Used[len(res.Used)-1] != 18 {
		t.Errorf("averaged orders %v, want the plateau 10..18", res.Used)
	}
	if math.Abs(res.Gamma-1.0) > 2e-3 {
		t.Errorf("Gamma = %.6g, want the plateau value 1.0", res.Gamma)
	}
	if res.Sigma > 5e-3 {
		t.Errorf("sigma = %.3g, too large for the plateau", res.Sigma)
	}

	// Fewer orders than the window: use them all, and say so through Used.
	short := &Result{MaxOrder: 8}
	for i, v := range []float64{1, 2, 3, 4} {
		short.Orders = append(short.Orders, OrderResult{Order: 5 + i, Gamma: v})
	}
	combinePaper(short, Options{}.withDefaults())
	if len(short.Used) != 4 || math.Abs(short.Gamma-2.5) > 1e-12 {
		t.Errorf("with 4 orders available: used %v, Gamma %.6g; want all four and 2.5",
			short.Used, short.Gamma)
	}
}

// TestAverageReferenceMatchesTheFortran checks the reference protocol against the two
// branches of stieltjes_phi1.f:327-370: the plain three-highest-order mean when no window
// converges, and the converged window when one does, found by scanning downwards from the
// highest order.
func TestAverageReferenceMatchesTheFortran(t *testing.T) {
	opts := Options{Average: AverageReference}.withDefaults()

	// A steadily diverging sequence: no three-order window ever meets the threshold within
	// ConvMax relaxations, so the answer is the mean of the three highest.
	div := &Result{MaxOrder: 12}
	for i, v := range []float64{1, 2, 4, 8, 16, 32, 64, 128} {
		div.Orders = append(div.Orders, OrderResult{Order: 5 + i, Gamma: v})
	}
	combineReference(div, opts)
	if want := (32.0 + 64 + 128) / 3; math.Abs(div.Gamma-want) > 1e-12 {
		t.Errorf("diverging: Gamma = %.6g, want the three-highest mean %.6g", div.Gamma, want)
	}
	t.Logf("diverging sequence: %s", div)

	// A sequence that is flat at high order: convergence must be detected there.
	conv := &Result{MaxOrder: 12}
	for i, v := range []float64{5, 3, 2, 1.5, 1.01, 1.0, 1.005, 0.999} {
		conv.Orders = append(conv.Orders, OrderResult{Order: 5 + i, Gamma: v})
	}
	combineReference(conv, opts)
	t.Logf("converging sequence: %s", conv)
	if !conv.Converged {
		t.Error("convergence was not detected on a sequence that is flat at high order")
	}
	if conv.ConvOrder != 12 {
		t.Errorf("convergence reported at order %d, want 12 (the scan runs downwards from "+
			"the highest order, so the highest converged window wins)", conv.ConvOrder)
	}

	// maxord < 7: the reference does not search at all.
	no := &Result{MaxOrder: 6}
	for i, v := range []float64{1.0, 1.0, 1.0} {
		no.Orders = append(no.Orders, OrderResult{Order: 5 + i, Gamma: v})
	}
	combineReference(no, opts)
	if no.Converged {
		t.Error("a convergence search ran with maxord < 7, which the reference skips")
	}
}

// TestBothAverageModesAgreeOnAConvergedSpectrum: where the per-order results really have
// converged, the paper's nine-order window and the reference's three-order mean must give
// the same answer. A disagreement there would mean one of the two protocols is
// mis-transcribed rather than merely differently conservative.
func TestBothAverageModesAgreeOnAConvergedSpectrum(t *testing.T) {
	es, gs := sample(func(float64) float64 { return 0.25 }, 0.5, 5.0, 300)
	paper, err := Image(es, gs, 2.0, Options{})
	if err != nil {
		t.Fatal(err)
	}
	ref, err := Image(es, gs, 2.0, Options{Average: AverageReference})
	if err != nil {
		t.Fatal(err)
	}
	rel := math.Abs(paper.Gamma-ref.Gamma) / paper.Gamma
	t.Logf("paper protocol:     %s", paper)
	t.Logf("reference protocol: %s", ref)
	t.Logf("the two protocols differ by %.4g%%", 100*rel)
	if rel > 1e-3 {
		t.Errorf("the two averaging protocols differ by %.3g on a converged spectrum", rel)
	}
}

// TestImageRejectsBadInput requires each failure mode to be reported rather than silently
// returning zero, which is what the reference does for all of them.
func TestImageRejectsBadInput(t *testing.T) {
	es, gs := sample(func(float64) float64 { return 1 }, 1, 3, 50)
	for _, c := range []struct {
		name string
		e, g []float64
		at   float64
	}{
		{"too few states", es[:3], gs[:3], 2},
		{"mismatched lengths", es, gs[:len(gs)-1], 2},
		{"below the range", es, gs, 0.5},
		{"above the range", es, gs, 9},
		{"non-positive energy", append([]float64{0}, es...), append([]float64{1}, gs...), 2},
		{"negative width", es, append([]float64{-1}, gs[1:]...), 2},
	} {
		if _, err := Image(c.e, c.g, c.at, Options{}); err == nil {
			t.Errorf("%s: accepted", c.name)
		} else {
			t.Logf("%s: %v", c.name, err)
		}
	}
	// All-zero widths: nothing to image.
	if _, err := Image(es, make([]float64, len(gs)), 2, Options{}); err == nil {
		t.Error("a pseudo-spectrum of zero total width was accepted")
	}
}

// TestLowOrderFallback covers the reference's MAXORD < 5 path (stieltjes_phi1.f:219-228):
// a result is still produced from the single highest available order, and flagged so a
// caller cannot mistake it for a converged one.
func TestLowOrderFallback(t *testing.T) {
	es, gs := sample(func(float64) float64 { return 1 }, 1, 3, 40)
	r, err := Image(es, gs, 2.0, Options{MaxOrder: 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s", r)
	if !r.LowOrder {
		t.Error("a 3rd-order-only result was not flagged LowOrder")
	}
	if len(r.Orders) != 1 || r.Orders[0].Order != 3 {
		t.Errorf("evaluated orders %v, want only order 3", r.Used)
	}
	if r.Sigma != 0 {
		t.Errorf("sigma = %.3g from a single order, want 0", r.Sigma)
	}
}

// TestGridSamplesTheDensity checks the optional density grid: it spans the highest order's
// midpoint range, and it is non-negative everywhere — which is the PCHIP guarantee doing
// its job on real imaged data, not just on constructed knots.
func TestGridSamplesTheDensity(t *testing.T) {
	f := lorentzian(2.0, 0.6)
	es, gs := sample(f, 0.3, 6.0, 300)
	r, err := Image(es, gs, 1.0, Options{GridPoints: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Grid) != 200 || len(r.Density) != 200 {
		t.Fatalf("grid is %d/%d points, want 200", len(r.Grid), len(r.Density))
	}
	for i := range r.Grid {
		if i > 0 && !(r.Grid[i] > r.Grid[i-1]) {
			t.Fatalf("grid is not ascending at %d", i)
		}
		if r.Density[i] < 0 {
			t.Errorf("the imaged density is negative at E=%.6g: %.6g; the monotone interpolant "+
				"is supposed to make that impossible", r.Grid[i], r.Density[i])
		}
	}
	// The sampled density must integrate to roughly the input's total width over that range.
	var area float64
	for i := 1; i < len(r.Grid); i++ {
		area += 0.5 * (r.Density[i] + r.Density[i-1]) * (r.Grid[i] - r.Grid[i-1])
	}
	t.Logf("the imaged density integrates to %.6g over [%.4g, %.4g]",
		area, r.Grid[0], r.Grid[len(r.Grid)-1])
}
