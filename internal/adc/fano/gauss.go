package fano

// gauss.go — width engines that need no eigenvector of PHP. The eigen path (Widths +
// ImageWidth) projects the coupling vector g onto the Ritz vectors of PHP, which needs
// lanczos.Options.WantFull over P: fine for SIP, infeasible once P holds a (k+2)h2p relaxation class (1e6-1e7 rows).
// The engines below work from applications of PHP to vectors only; W-B feeds Stieltjes
// imaging, W-B2 and W-C smooth the measure with a kernel instead:
//
//   - W-B, GaussWidth: an m-step Lanczos run seeded by g gives the Gauss rule of every
//     order k <= m of mu(E) = sum_j |<psi_j|g>|^2 delta(E - E_j) (lanczos.SolveSeeded);
//     the rules go to stieltjes.ImageRules with weights 2 pi |.|^2, the gamma convention
//     of Widths.
//   - W-B2, ShiftInvertWidth: Lanczos on (PHP - sigma)^-1 with sigma next to E_Phi, PHP -
//     sigma LU-factorized once (dense P only). The Gauss nodes mu_i map back to
//     E_i = sigma + 1/mu_i and resolve the individual pseudo-continuum states around sigma;
//     that resolved measure is NOT Stieltjes-imaged but smoothed by a ladder of Gaussians
//     and extrapolated to zero width (see ShiftInvertWidth).
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
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
	"github.com/leiaSQ/ADCgo/internal/adc/stieltjes"
	"gonum.org/v1/gonum/mat"
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

// ShiftInvertOptions controls W-B2.
type ShiftInvertOptions struct {
	Sigma float64 // the shift, next to (not on) E_Phi
	// DenseMax is the largest PHP the engine accepts (default 8192 rows): PHP - sigma is
	// LU-factorized once. An iterative inner solve is no alternative, because sigma sits
	// inside the pseudo-continuum, PHP - sigma has eigenvalues far below the level spacing,
	// and MINRES runs to its cap on every application (6 h on 4392 rows, Ne2 C1).
	DenseMax int
	// WeightCut drops rule nodes whose weight is below WeightCut ||g||^2 (default 1e-12;
	// < 0 keeps every node). (PHP - sigma)^-1 amplifies the rounding-level components of
	// states g does not couple to (other irreps of a C1 run, degenerate partners) by
	// 1/|E - sigma|, so they surface as nodes of weight 1e-36 to 1e-8 next to sigma; their
	// degenerate copies collide with coupled nodes. Coupled nodes near sigma weigh
	// Gamma Delta / (2 pi ||g||^2), orders of magnitude above the cut.
	WeightCut float64
	// EtaSpacings is the Gaussian smoothing ladder in units of the local level spacing
	// (default 4 to 256). Widths below a few spacings are dominated by the fluctuation of
	// the individual gamma_i, widths on the scale of Gamma(E)'s own variation by its
	// curvature; the fit keeps the widths that lie on an eta^2 line and extrapolates it
	// to eta -> 0.
	EtaSpacings []float64
	// LocalNodes is how many nodes nearest E_Phi set the local spacing (default 40).
	LocalNodes int
}

// KernelDiag is W-B2's record of the smoothing and the extrapolation.
type KernelDiag struct {
	Sigma        float64   // the shift (hartree)
	Nodes        int       // nodes of the highest-order rule after the weight cut and merging
	LocalSpacing float64   // mean spacing of the LocalNodes nodes nearest E_Phi (hartree)
	Eta          []float64 // the converged smoothing widths used (hartree)
	Density      []float64 // Gamma_eta(E_Phi) at each (hartree)
	DensitySigma []float64 // its kernel-estimator standard deviation (hartree)
	Dropped      []float64 // ladder widths dropped: not converged in the Lanczos order
	EdgeDropped  []float64 // ladder widths dropped: 3 eta reaches past the lowest or highest node
	Curved       []float64 // ladder widths dropped: off the eta^2 line fitted to the narrower ones
	Bias         float64   // fitted b eta_min^2, the extrapolation's size at the smallest eta
	Chi2         float64   // reduced chi^2 of the a + b eta^2 fit (correlated points: a guide)
	// Below / Above: fewer than 3 ladder widths fit between E_Phi and the lowest / highest
	// node (a threshold), so the 3 smallest were used although their Gaussians straddle the
	// edge of the coupled pseudo-continuum, where the eta^2 law fails.
	Below, Above bool
}

// shiftedLU builds PHP - sigma column by column from apply (symmetrized, so the factor is
// that of the operator's symmetric part) and LU-factorizes it.
func shiftedLU(apply func(dst, src []float64), n int, sigma float64) (*mat.LU, error) {
	A := make([]float64, n*n) // column c of PHP in row c (PHP is symmetric)
	e := make([]float64, n)
	for c := range n {
		clear(e)
		e[c] = 1
		apply(A[c*n:(c+1)*n], e)
	}
	S := mat.NewDense(n, n, nil)
	parallel.Chunks(n, parallel.ChunkWorkers(n), func(_, lo, hi int) {
		for r := lo; r < hi; r++ {
			for c := range n {
				S.Set(r, c, 0.5*(A[r*n+c]+A[c*n+r]))
			}
			S.Set(r, r, S.At(r, r)-sigma)
		}
	})
	var lu mat.LU
	lu.Factorize(S)
	// LogDet, not Det: the product of thousands of pivots underflows for a regular matrix
	if ld, _ := lu.LogDet(); math.IsInf(ld, -1) {
		return nil, fmt.Errorf("fano: PHP - sigma is singular at sigma %.8f; change -fano-si-shift", sigma)
	}
	return &lu, nil
}

// shiftInvertNodes is the k-point Gauss rule of the shift-inverted measure mapped back to
// energies E = sigma + 1/mu, ascending, weights 2 pi |.|^2; nodes below cut ||g||^2 are
// dropped and coincident ones merged.
func shiftInvertNodes(sr lanczos.SeededResult, k int, sigma, cut float64) (e, w []float64, err error) {
	mu, wt, err := sr.GaussRule(k)
	if err != nil {
		return nil, nil, err
	}
	type node struct{ e, w float64 }
	nodes := make([]node, 0, k)
	for i := range mu {
		if mu[i] == 0 {
			continue // a node at infinite energy carries no density near sigma
		}
		if wt[i] < cut*sr.Norm2 {
			continue // an uncoupled state lifted out of rounding (WeightCut)
		}
		nodes = append(nodes, node{sigma + 1/mu[i], 2 * math.Pi * wt[i]})
	}
	sort.Slice(nodes, func(a, b int) bool { return nodes[a].e < nodes[b].e })
	for _, nd := range nodes {
		// 1/mu loses the digits that separate nodes far from sigma: merge coincident ones
		if j := len(e) - 1; j >= 0 && nd.e <= e[j] {
			w[j] += nd.w
			continue
		}
		e, w = append(e, nd.e), append(w, nd.w)
	}
	return e, w, nil
}

// gaussKernel is the normalized Gaussian of standard deviation eta at x.
func gaussKernel(x, eta float64) float64 {
	return math.Exp(-0.5*x*x/(eta*eta)) / (eta * math.Sqrt(2*math.Pi))
}

// kernelDensity is Gamma_eta(at) = sum_i w_i G_eta(at - e_i).
func kernelDensity(e, w []float64, at, eta float64) float64 {
	var s float64
	for i := range e {
		s += w[i] * gaussKernel(at-e[i], eta)
	}
	return s
}

// kernelVariance is the variance of the linear estimator sum_i w_i K_i: the w_i
// fluctuate state to state about their local mean wloc_i (the Porter-Thomas-like scatter
// of a discretized continuum), so var = sum_i (w_i - wloc_i)^2 K_i^2.
func kernelVariance(w, K, wloc []float64) float64 {
	var v float64
	for i := range w {
		d := w[i] - wloc[i]
		v += d * d * K[i] * K[i]
	}
	return v
}

// localMeans is each node's neighbours' kernel-weighted mean weight at scale h, the node
// itself excluded: a smooth coupling then has (almost) no scatter about it, and a
// fluctuating one keeps its full scatter.
func localMeans(e, w []float64, h float64) []float64 {
	out := make([]float64, len(e))
	parallel.Chunks(len(e), parallel.ChunkWorkers(len(e)), func(_, lo, hi int) {
		for i := lo; i < hi; i++ {
			var sw, sk float64
			for k := range e {
				if k == i {
					continue
				}
				g := gaussKernel(e[i]-e[k], h)
				sw += w[k] * g
				sk += g
			}
			if sk > 0 {
				out[i] = sw / sk
			} else {
				out[i] = w[i]
			}
		}
	})
	return out
}

// fitEta2 is the weighted least-squares line rho = a + b x (x = eta^2, weights 1/v) and
// the coefficients with a = sum_j c_j rho_j and b = sum_j d_j rho_j. The weights are the
// diagonal only: the rho_j of neighbouring widths share most of their nodes, and a
// generalized fit under that nearly singular covariance extrapolates past its data (on
// Porter-Thomas couplings it biased Gamma 15% low where this is unbiased to 5%). The
// correlations enter the error bar instead, through the exact kernel of a.
func fitEta2(xs, ys, vs []float64) (a, b float64, c, d []float64) {
	var S, Sx, Sxx float64
	for j := range xs {
		S += 1 / vs[j]
		Sx += xs[j] / vs[j]
		Sxx += xs[j] * xs[j] / vs[j]
	}
	D := S*Sxx - Sx*Sx
	c = make([]float64, len(xs))
	d = make([]float64, len(xs))
	for j := range xs {
		c[j] = (Sxx - Sx*xs[j]) / (vs[j] * D)
		d[j] = (S*xs[j] - Sx) / (vs[j] * D)
		a += c[j] * ys[j]
		b += d[j] * ys[j]
	}
	return a, b, c, d
}

// localSpacing is the mean spacing of the nloc nodes nearest at.
func localSpacing(e []float64, at float64, nloc int) float64 {
	near := make([]int, len(e))
	for i := range near {
		near[i] = i
	}
	sort.Slice(near, func(a, b int) bool { return math.Abs(e[near[a]]-at) < math.Abs(e[near[b]]-at) })
	near = near[:min(nloc, len(near))]
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, i := range near {
		lo, hi = math.Min(lo, e[i]), math.Max(hi, e[i])
	}
	return (hi - lo) / float64(len(near)-1)
}

// ShiftInvertWidth is W-B2: Gamma(at) from m Lanczos steps of (PHP - sigma)^-1 seeded by
// g, PHP - sigma LU-factorized once.
//
// Near sigma the shift-inverted rules converge onto the individual pseudo-continuum
// states, so they are NOT Stieltjes-imaged: the midpoint density of a resolved discrete
// spectrum is the coupling of whichever state pair brackets E_Phi, and the per-state
// gamma_i of a real continuum scatter by orders of magnitude (Ne2 C1: 0.12 meV against
// 3.9). Instead the resolved measure is smoothed with a Gaussian of width eta, a ladder of
// eta from a few to tens of local level spacings, and Gamma_eta = Gamma + b eta^2 + O(eta^4)
// is extrapolated to eta -> 0. A ladder width counts only if the rules of order 3m/4 and m
// agree on it within its statistical error (the Gaussian must not reach unconverged nodes).
// The error bar is the extrapolated estimator's own variance (the fit is linear in the
// node weights, so its kernel is exact, correlations included) plus the fitted bias at the
// smallest eta.
func ShiftInvertWidth(apply func(dst, src []float64), g []float64, m int, at float64,
	so ShiftInvertOptions) (Width, error) {

	n := len(g)
	if so.DenseMax == 0 {
		so.DenseMax = 8192
	}
	if n > so.DenseMax {
		return Width{}, fmt.Errorf("fano: shift-invert needs PHP factorized directly, and %d rows "+
			"exceed DenseMax %d; use -fano-engine kpm", n, so.DenseMax)
	}
	if so.WeightCut == 0 {
		so.WeightCut = 1e-12
	}
	if len(so.EtaSpacings) == 0 {
		so.EtaSpacings = []float64{4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256}
	}
	if so.LocalNodes <= 0 {
		so.LocalNodes = 40
	}
	sigma := so.Sigma
	lu, err := shiftedLU(apply, n, sigma)
	if err != nil {
		return Width{}, err
	}
	var failed error
	inv := func(dst, src []float64) {
		if err := lu.SolveVecTo(mat.NewVecDense(n, dst), false, mat.NewVecDense(n, src)); err != nil && failed == nil {
			failed = fmt.Errorf("fano: shift-invert direct solve at sigma %.8f: %w", sigma, err)
		}
	}
	sr, err := lanczos.SolveSeeded(inv, g, m)
	if err != nil {
		return Width{}, err
	}
	if failed != nil {
		return Width{}, failed
	}
	top := sr.Steps()
	if top < 8 {
		return Width{}, fmt.Errorf("fano: shift-invert needs at least 8 Lanczos steps, has %d", top)
	}
	e, w, err := shiftInvertNodes(sr, top, sigma, so.WeightCut)
	if err != nil {
		return Width{}, err
	}
	ePrev, wPrev, err := shiftInvertNodes(sr, 3*top/4, sigma, so.WeightCut)
	if err != nil {
		return Width{}, err
	}
	if len(e) < 3 {
		return Width{}, fmt.Errorf("fano: shift-invert kept %d nodes of %d; the coupling is "+
			"below the weight cut everywhere", len(e), top)
	}
	diag := &KernelDiag{Sigma: sigma, Nodes: len(e)}

	// local spacing and scatter. Rounding nodes above WeightCut can still sit between
	// coupled ones; next to their neighbours they weigh < 1e-6 of the local mean (a
	// Porter-Thomas weight does with probability 8e-4) and would fake both a finer spacing
	// and a large scatter, so they leave before either is measured.
	diag.LocalSpacing = localSpacing(e, at, so.LocalNodes)
	if !(diag.LocalSpacing > 0) {
		return Width{}, fmt.Errorf("fano: shift-invert found no level spacing near %.8f", at)
	}
	// A coupled state with an exactly degenerate uncoupled partner comes back as two nodes a
	// rounding error apart sharing its weight; one node at the kernel's resolution.
	keep := 0
	for i := range e {
		if keep > 0 && e[i]-e[keep-1] < 1e-3*diag.LocalSpacing {
			w[keep-1] += w[i]
			continue
		}
		e[keep], w[keep] = e[i], w[i]
		keep++
	}
	e, w = e[:keep], w[:keep]
	wloc := localMeans(e, w, 4*diag.LocalSpacing)
	keep = 0
	for i := range e {
		if w[i] >= 1e-6*wloc[i] {
			e[keep], w[keep] = e[i], w[i]
			keep++
		}
	}
	e, w = e[:keep], w[:keep]
	diag.Nodes = len(e)
	if len(e) < 3 {
		return Width{}, fmt.Errorf("fano: shift-invert kept %d coupled nodes", len(e))
	}
	diag.LocalSpacing = localSpacing(e, at, so.LocalNodes)
	if !(diag.LocalSpacing > 0) {
		return Width{}, fmt.Errorf("fano: shift-invert found no level spacing near %.8f", at)
	}
	// each node's scatter is measured against its neighbours' mean over 16 spacings: wide
	// enough that a run of weak states (a local hole in the coupling) does not pull the
	// mean down with it and hide its own scatter, narrow enough that a smooth Gamma(E)
	// leaves almost none
	wloc = localMeans(e, w, 16*diag.LocalSpacing)

	// Gamma(E) is not smooth at a threshold: a Gaussian that reaches past the lowest or
	// highest node is excluded, unless that leaves fewer than 3 widths
	ladder := make([]float64, 0, len(so.EtaSpacings))
	for _, f := range so.EtaSpacings {
		eta := f * diag.LocalSpacing
		if at-3*eta < e[0] || at+3*eta > e[len(e)-1] {
			diag.EdgeDropped = append(diag.EdgeDropped, eta)
			continue
		}
		ladder = append(ladder, eta)
	}
	if len(ladder) < 3 {
		ladder = ladder[:0]
		for _, f := range so.EtaSpacings[:min(3, len(so.EtaSpacings))] {
			ladder = append(ladder, f*diag.LocalSpacing)
		}
		diag.EdgeDropped = nil
		diag.Below = at-3*ladder[len(ladder)-1] < e[0]
		diag.Above = at+3*ladder[len(ladder)-1] > e[len(e)-1]
	}

	K := make([]float64, len(e))
	var xs, ys, vs []float64
	var G [][]float64 // G[j][i] = G_eta_j(at - e_i) for the converged widths
	for _, eta := range ladder {
		Kj := make([]float64, len(e))
		var rho float64
		for i := range e {
			Kj[i] = gaussKernel(at-e[i], eta)
			rho += w[i] * Kj[i]
		}
		v := kernelVariance(w, Kj, wloc)
		if math.Abs(rho-kernelDensity(ePrev, wPrev, at, eta)) > math.Max(math.Sqrt(v), 1e-6*math.Abs(rho)) {
			diag.Dropped = append(diag.Dropped, eta)
			continue
		}
		G = append(G, Kj)
		xs, ys = append(xs, eta*eta), append(ys, rho)
		vs = append(vs, math.Max(v, 1e-24*rho*rho+1e-300))
	}
	if len(xs) < 3 {
		return Width{}, fmt.Errorf("fano: shift-invert converged on %d smoothing widths of %d "+
			"(%d steps, %d nodes, local spacing %.3g Eh); raise -fano-order", len(xs),
			len(ladder), top, len(e), diag.LocalSpacing)
	}

	// combo is the kernel of sum_j coef_j rho_j over the first len(coef) widths
	combo := func(coef []float64) []float64 {
		out := make([]float64, len(e))
		for j, cj := range coef {
			for i := range out {
				out[i] += cj * G[j][i]
			}
		}
		return out
	}

	// grow the fit from the 3 narrowest widths while each wider one lies on the line
	// within 2 sigma of its deviation (rho_j - a - b x_j is linear in the node weights, so
	// its variance is exact, correlations between the widths included); past that,
	// O(eta^4) — the scale on which Gamma(E) itself varies — is resolved
	use := 3
	for ; use < len(xs); use++ {
		a, b, c, d := fitEta2(xs[:use], ys[:use], vs[:use])
		coef := make([]float64, use)
		for k := range coef {
			coef[k] = -(c[k] + xs[use]*d[k])
		}
		Kd := combo(coef)
		for i := range Kd {
			Kd[i] += G[use][i]
		}
		dev := ys[use] - a - b*xs[use]
		if math.Abs(dev) > 2*math.Sqrt(kernelVariance(w, Kd, wloc))+1e-9*math.Abs(ys[use]) {
			break
		}
	}
	for j := use; j < len(xs); j++ {
		diag.Curved = append(diag.Curved, math.Sqrt(xs[j]))
	}
	xs, ys, vs = xs[:use], ys[:use], vs[:use]
	for j := range xs {
		diag.Eta = append(diag.Eta, math.Sqrt(xs[j]))
		diag.Density = append(diag.Density, ys[j])
		diag.DensitySigma = append(diag.DensitySigma, math.Sqrt(vs[j]))
	}
	a, b, c, _ := fitEta2(xs, ys, vs)
	for j := range xs {
		r := ys[j] - a - b*xs[j]
		diag.Chi2 += r * r / vs[j]
	}
	if use > 2 {
		diag.Chi2 /= float64(use - 2)
	}
	diag.Bias = b * xs[0]

	// the extrapolated estimator's kernel K_i = sum_j c_j G_eta_j(at - e_i)
	K = combo(c)
	sigmaA := math.Sqrt(kernelVariance(w, K, wloc) + diag.Bias*diag.Bias)
	out := NewWidth(a, sigmaA)
	out.Kernel = diag
	return out, nil
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
// (which must enclose its spectrum).
func KPMWidth(apply func(dst, src []float64), g []float64, at float64, nMoments int, lo, hi float64) (float64, error) {
	w, err := KPMWidths(apply, g, at, []int{nMoments}, lo, hi)
	if err != nil {
		return 0, err
	}
	return w[0], nil
}

// KPMWidths is KPMWidth at every moment count in counts from one moment run to the
// largest: the moments of a shorter expansion are a prefix of the longer one's, only the
// Jackson kernel changes, so the convergence series costs no extra application. Two
// moments per application (the doubling identities mu_2n = 2 <v_n|v_n> - mu_0,
// mu_2n+1 = 2 <v_n+1|v_n> - mu_1).
func KPMWidths(apply func(dst, src []float64), g []float64, at float64, counts []int, lo, hi float64) ([]float64, error) {
	nMoments := 0
	for _, c := range counts {
		if c < 2 {
			return nil, fmt.Errorf("fano: KPM needs >= 2 moments (got %d)", c)
		}
		nMoments = max(nMoments, c)
	}
	if len(counts) == 0 || hi <= lo || at <= lo || at >= hi {
		return nil, fmt.Errorf("fano: KPM needs moment counts and lo < at < hi (got %v, [%g, %g], %g)",
			counts, lo, hi, at)
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
	mom[1] = dotF(g, v1)
	vPrev, vCur := v0, v1
	next := make([]float64, n)
	// v_k known for k = 0, 1; each step makes v_{k+1} and fills mu_2k, mu_2k+1
	for k := 1; 2*k < nMoments; k++ {
		mom[2*k] = 2*dotF(vCur, vCur) - mom[0]
		if 2*k+1 >= nMoments {
			break
		}
		scaled(next, vCur)
		for i := range next {
			next[i] = 2*next[i] - vPrev[i]
		}
		mom[2*k+1] = 2*dotF(next, vCur) - mom[1]
		vPrev, vCur, next = vCur, next, vPrev
	}
	x := (at - b) / a
	out := make([]float64, len(counts))
	for i, c := range counts {
		out[i] = 2 * math.Pi * kpmDensity(mom[:c], x) / a
	}
	return out, nil
}

// kpmDensity is the Jackson-damped Chebyshev series of the moments mom at x in (-1, 1).
func kpmDensity(mom []float64, x float64) float64 {
	N := float64(len(mom))
	var s float64
	tPrev, tCur := 1.0, x // T_0(x), T_1(x)
	for k := range mom {
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
	return s / (math.Pi * math.Sqrt(1-x*x))
}

func dotF(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
