package fano

// gauss.go — width engines that need no eigenvector of PHP. The eigen path (Widths +
// ImageWidth) projects the coupling vector g onto the Ritz vectors of PHP, which needs
// lanczos.Options.WantFull over P: fine for SIP, infeasible once P holds a (k+2)h2p relaxation class (1e6-1e7 rows).
// All three engines below work from applications of PHP to vectors only, and all three
// feed the same Stieltjes imaging:
//
//   - W-B, GaussWidth: an m-step Lanczos run seeded by g gives the Gauss rule of every
//     order k <= m of mu(E) = sum_j |<psi_j|g>|^2 delta(E - E_j) (lanczos.SolveSeeded);
//     the rules go to stieltjes.ImageRules with weights 2 pi |.|^2, the gamma convention
//     of Widths.
//   - W-B2, ShiftInvertWidth: the same on (PHP - sigma)^-1 with sigma next to E_Phi, each
//     application a MINRES solve. The Gauss nodes mu_i map back to E_i = sigma + 1/mu_i
//     and crowd around sigma, which is where Gamma is evaluated; the weights are those of
//     the same measure.
//   - W-C, KPMWidth: the kernel polynomial method, Chebyshev moments of mu with a Jackson
//     kernel (Weisse et al., Rev. Mod. Phys. 78, 275 (2006)). No quadrature at all, so an
//     independent check of the imaged density, at the kernel's resolution.
//
// Every engine is homogeneous of degree two in g — scaling g by lambda scales Gamma by
// lambda^2 exactly (up to rounding) — because nothing here has an absolute cut. That is
// what lets a width of 1e-15 Eh be computed at all (the absolute Filter.MinGamma of the
// eigen path deletes it).

import (
	"fmt"
	"math"
	"sort"

	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/stieltjes"
)

// GaussRules turns a seeded Lanczos run into the Gauss rules of orders lo..hi (clamped to
// 2..Steps), weights in the gamma convention 2 pi |<psi|g>|^2.
func GaussRules(sr lanczos.SeededResult, lo, hi int) ([]stieltjes.Rule, error) {
	lo = max(lo, 2)
	if hi <= 0 || hi > sr.Steps() {
		hi = sr.Steps()
	}
	if hi < lo {
		return nil, fmt.Errorf("fano: %d Lanczos steps give no Gauss rule of order >= %d", sr.Steps(), lo)
	}
	var rules []stieltjes.Rule
	for k := lo; k <= hi; k++ {
		nodes, w, err := sr.GaussRule(k)
		if err != nil {
			return nil, err
		}
		for i := range w {
			w[i] *= 2 * math.Pi
		}
		rules = append(rules, stieltjes.Rule{Energy: nodes, Weight: w})
	}
	return rules, nil
}

// GaussWidth is W-B: Gamma(at) from m Lanczos steps of PHP (apply) seeded by g.
func GaussWidth(apply func(dst, src []float64), g []float64, m int, at float64,
	opts stieltjes.Options) (Width, lanczos.SeededResult, error) {

	sr, err := lanczos.SolveSeeded(apply, g, m)
	if err != nil {
		return Width{}, sr, err
	}
	rules, err := GaussRules(sr, opts.MinOrder, opts.MaxOrder)
	if err != nil {
		return Width{}, sr, err
	}
	r, err := stieltjes.ImageRules(rules, at, opts)
	if err != nil {
		return Width{}, sr, err
	}
	return newWidth(r), sr, nil
}

// ShiftInvertOptions controls W-B2's inner solves.
type ShiftInvertOptions struct {
	Sigma    float64   // the shift, next to (not on) E_Phi
	Diag     []float64 // diag(PHP), for the MINRES preconditioner
	InnerTol float64   // MINRES relative tolerance (default 1e-12)
	Inner    int       // MINRES iteration cap (default 2000)
}

// ShiftInvertWidth is W-B2: Gamma(at) from m Lanczos steps of (PHP - sigma)^-1 seeded
// by g, the nodes mapped back to energies.
func ShiftInvertWidth(apply func(dst, src []float64), g []float64, m int, at float64,
	so ShiftInvertOptions, opts stieltjes.Options) (Width, error) {

	if len(so.Diag) != len(g) {
		return Width{}, fmt.Errorf("fano: ShiftInvertWidth needs the diagonal (%d rows)", len(g))
	}
	if so.InnerTol <= 0 {
		so.InnerTol = 1e-12
	}
	if so.Inner <= 0 {
		so.Inner = 2000
	}
	sigma := so.Sigma
	shifted := func(dst, src []float64) {
		apply(dst, src)
		for i := range dst {
			dst[i] -= sigma * src[i]
		}
	}
	pre := func(dst, src []float64) {
		for i := range dst {
			dst[i] = src[i] / (math.Abs(so.Diag[i]-sigma) + 1e-3)
		}
	}
	var failed error
	inv := func(dst, src []float64) {
		x, info := lanczos.Minres(shifted, src, nil, lanczos.MinresOptions{
			RTol: so.InnerTol, MaxIter: so.Inner, Precond: pre})
		if !info.Converged && failed == nil {
			failed = fmt.Errorf("fano: shift-invert inner solve stopped at relative residual %.1e "+
				"after %d iterations (sigma %.8f)", info.RelResid, info.Iter, sigma)
		}
		copy(dst, x)
	}
	sr, err := lanczos.SolveSeeded(inv, g, m)
	if err != nil {
		return Width{}, err
	}
	if failed != nil {
		return Width{}, failed
	}
	lo, hi := max(opts.MinOrder, 2), opts.MaxOrder
	if hi <= 0 || hi > sr.Steps() {
		hi = sr.Steps()
	}
	var rules []stieltjes.Rule
	for k := lo; k <= hi; k++ {
		mu, w, err := sr.GaussRule(k)
		if err != nil {
			return Width{}, err
		}
		type node struct{ e, w float64 }
		nodes := make([]node, 0, k)
		for i := range mu {
			if mu[i] == 0 {
				continue // a node at infinite energy carries no density near sigma
			}
			nodes = append(nodes, node{sigma + 1/mu[i], 2 * math.Pi * w[i]})
		}
		sort.Slice(nodes, func(a, b int) bool { return nodes[a].e < nodes[b].e })
		r := stieltjes.Rule{Energy: make([]float64, len(nodes)), Weight: make([]float64, len(nodes))}
		for i, nd := range nodes {
			r.Energy[i], r.Weight[i] = nd.e, nd.w
		}
		rules = append(rules, r)
	}
	if len(rules) == 0 {
		return Width{}, fmt.Errorf("fano: %d shift-invert steps give no rule of order >= %d", sr.Steps(), lo)
	}
	r, err := stieltjes.ImageRules(rules, at, opts)
	if err != nil {
		return Width{}, err
	}
	return newWidth(r), nil
}

// SpectralBounds estimates [lo, hi] enclosing PHP's spectrum from a short seeded Lanczos
// run (extreme Ritz values bound the spectrum from inside), widened by margin times the
// span — what KPM needs to map the operator into [-1, 1].
func SpectralBounds(apply func(dst, src []float64), g []float64, steps int, margin float64) (lo, hi float64, err error) {
	sr, err := lanczos.SolveSeeded(apply, g, steps)
	if err != nil {
		return 0, 0, err
	}
	nodes, _, err := sr.GaussRule(sr.Steps())
	if err != nil {
		return 0, 0, err
	}
	lo, hi = nodes[0], nodes[len(nodes)-1]
	span := hi - lo
	if span == 0 {
		span = math.Max(math.Abs(hi), 1)
	}
	return lo - margin*span, hi + margin*span, nil
}

// KPMWidth is W-C: Gamma(at) = 2 pi rho(at) with rho the Jackson-damped Chebyshev
// expansion of mu to nMoments moments, the operator mapped into [-1, 1] from [lo, hi]
// (which must enclose its spectrum). Two moments per application (the doubling
// identities mu_2n = 2 <v_n|v_n> - mu_0, mu_2n+1 = 2 <v_n+1|v_n> - mu_1).
func KPMWidth(apply func(dst, src []float64), g []float64, at float64, nMoments int, lo, hi float64) (float64, error) {
	if nMoments < 2 || hi <= lo || at <= lo || at >= hi {
		return 0, fmt.Errorf("fano: KPM needs >= 2 moments and lo < at < hi (got %d, [%g, %g], %g)",
			nMoments, lo, hi, at)
	}
	n := len(g)
	a, b := (hi-lo)/2, (hi+lo)/2
	scaled := func(dst, src []float64) {
		apply(dst, src)
		for i := range dst {
			dst[i] = (dst[i] - b*src[i]) / a
		}
	}
	mom := make([]float64, nMoments)
	v0 := append([]float64(nil), g...)
	v1 := make([]float64, n)
	scaled(v1, v0)
	mom[0] = dotF(g, v0)
	if nMoments > 1 {
		mom[1] = dotF(g, v1)
	}
	vPrev, vCur := v0, v1
	next := make([]float64, n)
	// v_k known for k = 0, 1; each step makes v_{k+1} and fills mu_2k, mu_2k+1
	for k := 1; 2*k < nMoments; k++ {
		if 2*k < nMoments {
			mom[2*k] = 2*dotF(vCur, vCur) - mom[0]
		}
		scaled(next, vCur)
		for i := range next {
			next[i] = 2*next[i] - vPrev[i]
		}
		if 2*k+1 < nMoments {
			mom[2*k+1] = 2*dotF(next, vCur) - mom[1]
		}
		vPrev, vCur, next = vCur, next, vPrev
	}
	N := float64(nMoments)
	x := (at - b) / a
	var s float64
	tPrev, tCur := 1.0, x // T_0(x), T_1(x)
	for k := range nMoments {
		kf := float64(k)
		jackson := ((N-kf+1)*math.Cos(math.Pi*kf/(N+1)) + math.Sin(math.Pi*kf/(N+1))/math.Tan(math.Pi/(N+1))) / (N + 1)
		var tk float64
		switch k {
		case 0:
			tk = 1
		case 1:
			tk = x
		default:
			tk = 2*x*tCur - tPrev
			tPrev, tCur = tCur, tk
		}
		c := 2.0
		if k == 0 {
			c = 1
		}
		s += c * jackson * mom[k] * tk
	}
	rho := s / (math.Pi * math.Sqrt(1-x*x)) / a
	return 2 * math.Pi * rho, nil
}

func dotF(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
