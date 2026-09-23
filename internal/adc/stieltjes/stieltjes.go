// Package stieltjes performs Stieltjes imaging: it turns a discrete pseudo-spectrum of
// energy/width pairs into a continuous Gamma(E) and evaluates it at a chosen energy.
//
// This is the last step of a Fano decay-rate calculation. The L2 pseudo-continuum from
// internal/adc/fano is a finite set of normalizable states, not a continuum, so its
// individual gamma_i have no physical meaning on their own — only the cumulative
// distribution they sample does. Stieltjes imaging reconstructs the underlying density
// from that distribution's moments.
//
// Ported from ../ADC/adc2_pol/stieltjes_phi1.f (V. Averbukh, 2003), which implements the
// algorithm of F. Muller-Plathe and G. H. F. Diercksen, in *Electronic Structure of
// Atoms, Molecules and Solids* (World Scientific, 1990), p. 1; the recurrence numbering
// below is that tutorial's (3.3.20)-(3.3.23). The reference's REAL*16 TQL2 and its NAG
// interpolation calls are replaced as described at their use sites.
//
// # Why extended precision
//
// The method builds orthogonal polynomials in the INVERSE energy variable from the
// negative moments S_-k = sum_i gamma_i eps_i^-k. Those moments span an enormous dynamic
// range and the three-term recurrence differences them against each other, so the
// polynomials lose orthogonality after a few tens of orders — that loss is not a bug but
// the thing that LIMITS the achievable order, and how fast it arrives is set by the
// working precision. The reference runs the whole recurrence in REAL*16 for this reason;
// this port uses math/big.Float (256 bits by default, more on request), which also
// removes the overflow concern entirely: the running product of b coefficients would leave
// float64's exponent range long before its mantissa mattered.
//
// The small tridiagonal eigenproblem is then solved in float64, which is sound because
// the conditioning lives in the recurrence, not there — and TestGaussRuleReproducesMoments
// checks the whole chain at once against the property that defines it.
package stieltjes

import (
	"fmt"
	"math"
	"math/big"
	"sort"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// Defaults, all of them the reference's hard-coded constants (stieltjes_phi1.f:60-63)
// except DefaultPrec and DefaultWindow.
const (
	// DefaultPrec is the big.Float mantissa width. REAL*16 gives 113 bits; 256 is
	// comfortably beyond it, and the recurrence is O(orders x points) big-float
	// operations, so the extra width costs milliseconds.
	DefaultPrec = 256
	// DefaultOverMax is the reference's OVERMAX: the largest norm-to-overlap ratio of
	// adjacent orthogonal polynomials still considered orthogonal. Once the ratio falls
	// BELOW it the overlap has grown comparable to the norm and that order is unusable.
	DefaultOverMax = 100.0
	// DefaultMinOrder is the reference's lower bound on a usable Stieltjes order.
	DefaultMinOrder = 5
	// DefaultWindow is the paper's protocol: "decay widths are evaluated through
	// averaging of nine consecutive orders n_S of the imaging procedure in the region of
	// best convergence" (ADC22.pdf p. 10), with the quoted uncertainty the statistical
	// standard deviation over that window.
	DefaultWindow = 9
	// The reference's convergence-search constants (AverageReference only).
	DefaultConvThresh = 0.05
	DefaultConvFac    = 1.0
	DefaultConvMax    = 10
)

// AverageMode selects how the per-order results are combined into one width.
type AverageMode int

const (
	// AveragePaper is the default: the mean over DefaultWindow consecutive orders in the
	// region of best convergence, taken as the window with the smallest standard
	// deviation, with that deviation as the error bar. This is what the paper's tabulated
	// uncertainties are.
	AveragePaper AverageMode = iota
	// AverageReference reproduces stieltjes_phi1.f:327-370 instead: the mean of the three
	// highest orders, unless a three-order window satisfies a relative-spread threshold
	// that is progressively relaxed (x1.2, up to ConvMax times), scanning from the
	// highest order down so that convergence at high order wins.
	AverageReference
)

func (m AverageMode) String() string {
	if m == AverageReference {
		return "reference (3 highest orders / convergence search)"
	}
	return "paper (9 consecutive orders, best convergence)"
}

// Options tunes the imaging. A zero value means every default.
type Options struct {
	Prec     uint    // big.Float mantissa bits (0 -> DefaultPrec)
	OverMax  float64 // orthogonality bound (0 -> DefaultOverMax)
	MinOrder int     // lowest order evaluated (0 -> DefaultMinOrder)
	MaxOrder int     // cap on the order actually used (0 -> the discovered maximum)

	Average AverageMode
	Window  int // AveragePaper: orders per window (0 -> DefaultWindow)

	ConvThresh float64 // AverageReference (0 -> DefaultConvThresh)
	ConvFac    float64 // AverageReference (0 -> DefaultConvFac)
	ConvMax    int     // AverageReference (0 -> DefaultConvMax)

	// GridPoints samples the imaged density onto Result.Grid/Density at the highest order
	// used, for plotting. 0 disables it; the reference always builds 200 points.
	GridPoints int
}

func (o Options) withDefaults() Options {
	if o.Prec == 0 {
		o.Prec = DefaultPrec
	}
	if o.OverMax == 0 {
		o.OverMax = DefaultOverMax
	}
	if o.MinOrder == 0 {
		o.MinOrder = DefaultMinOrder
	}
	if o.Window == 0 {
		o.Window = DefaultWindow
	}
	if o.ConvThresh == 0 {
		o.ConvThresh = DefaultConvThresh
	}
	if o.ConvFac == 0 {
		o.ConvFac = DefaultConvFac
	}
	if o.ConvMax == 0 {
		o.ConvMax = DefaultConvMax
	}
	return o
}

// Table holds the orthogonal-polynomial recurrence for one discrete pseudo-spectrum,
// computed once in extended precision. Every Stieltjes order draws its quadrature rule
// from the same coefficients, so the expensive part is done once.
type Table struct {
	prec       uint
	n          int          // input points
	x, g       []*big.Float // inverse energies and weights
	a          []*big.Float // a[i] for i = 1..maxUsable+1 (a[0] unused)
	b          []*big.Float // b[i] for i = 0..maxUsable
	max        int          // the discovered maximum usable order
	eMin, eMax float64
}

// NewTable runs the recurrence (3.3.20)-(3.3.23) and discovers the maximum usable order.
//
// e and g are the pseudo-continuum energies and widths; energies must be strictly
// positive, since the whole construction lives in 1/E. The reference additionally refuses
// fewer than four points (stieltjes_phi1.f:74) and that bound is kept: with three points
// there is no order at which the procedure means anything.
//
// The orthogonality test is FUSED into the recurrence rather than run afterwards over a
// stored (num+1) x num polynomial table, as the reference does. That is not a shortcut —
// the test at order m needs only Q_m and Q_{m-1}, both in hand the moment Q_m is formed,
// so the result is identical while the memory drops from O(num^2) big floats (720 MB at
// the reference's own num = 3000 limit) to O(num), and the coefficients past the maximum
// usable order are never computed because they are never used.
func NewTable(e, g []float64, opts Options) (*Table, error) {
	opts = opts.withDefaults()
	if len(e) != len(g) {
		return nil, fmt.Errorf("stieltjes: %d energies and %d widths", len(e), len(g))
	}
	n := len(e)
	if n <= 3 {
		return nil, fmt.Errorf("stieltjes: not enough pseudo-continuum states (%d); "+
			"the procedure needs more than 3", n)
	}
	t := &Table{prec: opts.Prec, n: n, eMin: math.Inf(1), eMax: math.Inf(-1)}
	t.x = make([]*big.Float, n)
	t.g = make([]*big.Float, n)
	for i := range e {
		if !(e[i] > 0) {
			return nil, fmt.Errorf("stieltjes: state %d has energy %.17g; the moments are taken "+
				"in 1/E, so every energy must be strictly positive", i, e[i])
		}
		if g[i] < 0 {
			return nil, fmt.Errorf("stieltjes: state %d has a negative width %.17g", i, g[i])
		}
		t.eMin = math.Min(t.eMin, e[i])
		t.eMax = math.Max(t.eMax, e[i])
		t.x[i] = t.quo(t.one(), t.from(e[i]))
		t.g[i] = t.from(g[i])
	}
	if err := t.recur(opts.OverMax); err != nil {
		return nil, err
	}
	return t, nil
}

// Helpers for working at the table's precision.
func (t *Table) f() *big.Float                  { return new(big.Float).SetPrec(t.prec) }
func (t *Table) from(v float64) *big.Float      { return t.f().SetFloat64(v) }
func (t *Table) one() *big.Float                { return t.from(1) }
func (t *Table) quo(a, b *big.Float) *big.Float { return t.f().Quo(a, b) }

// recur fills a and b and sets max.
func (t *Table) recur(overMax float64) error {
	n := t.n
	// b_0 = sum g_i ; a_1 = (sum g_i x_i) / b_0.
	b0 := t.f()
	a1 := t.f()
	tmp := t.f()
	for i := range n {
		b0.Add(b0, t.g[i])
		a1.Add(a1, tmp.Mul(t.g[i], t.x[i]))
	}
	if b0.Sign() <= 0 {
		return fmt.Errorf("stieltjes: the total width is %s; there is nothing to image", b0.Text('g', 6))
	}
	a1.Quo(a1, b0)

	// Q_0 = 1 ; Q_1 = x - a_1.
	qPrev := make([]*big.Float, n) // Q_{m-2}
	qCur := make([]*big.Float, n)  // Q_{m-1}
	qNew := make([]*big.Float, n)  // Q_m
	xp := make([]*big.Float, n)    // x^m at the current m
	for i := range n {
		qPrev[i] = t.one()
		qCur[i] = t.f().Sub(t.x[i], a1)
		qNew[i] = t.f()
		xp[i] = t.f().Mul(t.x[i], t.x[i]) // m starts at 2
	}

	// b_1 and a_2 (the m = 1 instance of the general step below).
	b1 := t.f()
	a2 := t.f()
	for i := range n {
		w := t.f().Mul(qCur[i], t.g[i])
		b1.Add(b1, tmp.Mul(w, t.x[i]))
		a2.Add(a2, t.f().Mul(w, t.f().Mul(t.x[i], t.x[i])))
	}
	b1.Quo(b1, b0)
	if b1.Sign() == 0 {
		return fmt.Errorf("stieltjes: b_1 vanished; the pseudo-spectrum is degenerate " +
			"(all states at one energy?)")
	}
	a2.Quo(a2, t.f().Mul(b0, b1))
	a2.Sub(a2, a1)

	t.a = []*big.Float{nil, a1, a2}
	t.b = []*big.Float{b0, b1}
	t.max = n

	// Order 1's orthogonality, then the loop.
	if ok, _, _ := t.orthoOK(qCur, qPrev, overMax); !ok {
		t.max = 0
		return nil
	}

	bprod := t.f().Mul(b0, b1) // prod_{k=0}^{m-1} b_k at m = 2
	asum := t.f().Add(a1, a2)  // sum_{k=1}^{m} a_k at m = 2

	W := parallel.ChunkWorkers(n)
	for m := 2; m <= n; m++ {
		am, bm1 := t.a[m], t.b[m-1]
		// Q_m[j] = (x_j - a_m) Q_{m-1}[j] - b_{m-1} Q_{m-2}[j], with the two weighted
		// sums the next coefficients need accumulated in the same pass. Parallel over
		// points with per-worker accumulators reduced in worker order, so the
		// coefficients are reproducible for a given worker count; the arithmetic per
		// point is untouched.
		sumB := make([]*big.Float, W)
		sumA := make([]*big.Float, W)
		qn := make([]*big.Float, W)
		qo := make([]*big.Float, W)
		parallel.Chunks(n, W, func(w, lo, hi int) {
			sB, sA, nn, oo := t.f(), t.f(), t.f(), t.f()
			s1, s2 := t.f(), t.f()
			for j := lo; j < hi; j++ {
				s1.Sub(t.x[j], am)
				s1.Mul(s1, qCur[j])
				s2.Mul(bm1, qPrev[j])
				qNew[j].Sub(s1, s2)

				s1.Mul(qNew[j], t.g[j])         // Q_m g
				sB.Add(sB, s2.Mul(s1, xp[j]))   // += Q_m g x^m
				sA.Add(sA, s2.Mul(s2, t.x[j]))  // += Q_m g x^{m+1}
				nn.Add(nn, s2.Mul(s1, qNew[j])) // += Q_m^2 g
				oo.Add(oo, s2.Mul(s1, qCur[j])) // += Q_m Q_{m-1} g
				xp[j].Mul(xp[j], t.x[j])        // advance to x^{m+1}
			}
			sumB[w], sumA[w], qn[w], qo[w] = sB, sA, nn, oo
		})
		qnorm, qover, sB, sA := t.f(), t.f(), t.f(), t.f()
		for w := range W {
			qnorm.Add(qnorm, qn[w])
			qover.Add(qover, qo[w])
			sB.Add(sB, sumB[w])
			sA.Add(sA, sumA[w])
		}

		if ok, _, _ := t.orthoRatio(qnorm, qover, overMax); !ok {
			t.max = m - 1
			return nil
		}
		if m == n {
			break // Q_n exists only for this check; there are no further coefficients
		}

		// b_m and a_{m+1}.
		bm := t.quo(sB, bprod)
		if bm.Sign() == 0 {
			t.max = m
			return nil
		}
		bprod.Mul(bprod, bm)
		amp1 := t.quo(sA, bprod)
		amp1.Sub(amp1, asum)
		t.b = append(t.b, bm)
		t.a = append(t.a, amp1)
		asum.Add(asum, amp1)

		qPrev, qCur, qNew = qCur, qNew, qPrev
	}
	return nil
}

// orthoOK is orthoRatio over an explicit polynomial pair.
func (t *Table) orthoOK(q, qPrev []*big.Float, overMax float64) (bool, *big.Float, *big.Float) {
	qnorm, qover, tmp := t.f(), t.f(), t.f()
	for j := range t.n {
		w := t.f().Mul(q[j], t.g[j])
		qnorm.Add(qnorm, tmp.Mul(w, q[j]))
		qover.Add(qover, tmp.Mul(w, qPrev[j]))
	}
	return t.orthoRatio(qnorm, qover, overMax)
}

// orthoRatio applies the reference's orthogonality test (stieltjes_phi1.f:196-209):
// the order is unusable once norm/|overlap| has fallen TO OR BELOW OverMax, with the
// overlap floored at 1e-50 so an exactly orthogonal pair does not divide by zero.
func (t *Table) orthoRatio(qnorm, qover *big.Float, overMax float64) (bool, *big.Float, *big.Float) {
	absOver := t.f().Abs(qover)
	if floor := t.from(1e-50); absOver.Cmp(floor) < 0 {
		absOver = floor
	}
	ratio := t.quo(qnorm, absOver)
	return ratio.Cmp(t.from(overMax)) > 0, qnorm, qover
}

// MaxOrder is the highest Stieltjes order whose orthogonal polynomials survived the test.
func (t *Table) MaxOrder() int { return t.max }

// Points is the number of input pseudo-continuum states.
func (t *Table) Points() int { return t.n }

// Range is the input energy range, the interval outside which imaging is meaningless.
func (t *Table) Range() (lo, hi float64) { return t.eMin, t.eMax }

// Moment returns the negative moment S_-k = sum_i gamma_i eps_i^-k of the input
// pseudo-spectrum, at the table's precision. It exists for the gate: an n-point Gauss
// rule must reproduce S_-k exactly for every k < 2n.
func (t *Table) Moment(k int) *big.Float {
	s, term := t.f(), t.f()
	for i := range t.n {
		term.SetInt64(1)
		for range k {
			term.Mul(term, t.x[i])
		}
		s.Add(s, term.Mul(term, t.g[i]))
	}
	return s
}

// Rule is an n-point Gauss quadrature rule for the discrete measure: the Stieltjes
// energies and their weights, ascending in energy.
type Rule struct {
	Energy []float64
	Weight []float64
}

// Rule builds the n-point rule by diagonalizing the Jacobi matrix
//
//	diag(i) = a_i,  offdiag(i) = -sqrt(b_{i-1}),
//
// whose eigenvalues are INVERSE energies in ascending order, so e_new(i) = 1/lambda
// reverses them, and whose weights are b_0 times the squared first eigenvector component
// (stieltjes_phi1.f:238-265).
//
// Solved in float64 with backend.Gonum's symmetric eigensolver rather than the
// reference's REAL*16 TQL2, and on the host unconditionally: the matrix is at most a few
// hundred square, so a device round trip would cost more than the solve. The precision is
// sound because the ill-conditioning of this procedure is in the recurrence that produced
// a and b, not in the eigenproblem — which is what the moment gate verifies rather than
// assumes.
func (t *Table) Rule(n int) (Rule, error) {
	if n < 1 || n > t.max {
		return Rule{}, fmt.Errorf("stieltjes: order %d is outside the usable range 1..%d", n, t.max)
	}
	T := backend.NewMat(n, n)
	for i := range n {
		ai, _ := t.a[i+1].Float64()
		T.Set(i, i, ai)
		if i > 0 {
			bi, _ := t.b[i].Float64()
			if bi < 0 {
				return Rule{}, fmt.Errorf("stieltjes: b_%d = %.6g is negative, so the Jacobi "+
					"matrix is not real symmetric; the recurrence has broken down at order %d",
					i, bi, n)
			}
			off := -math.Sqrt(bi)
			T.Set(i, i-1, off)
			T.Set(i-1, i, off)
		}
	}
	lam, vec := backend.Gonum{}.SymEig(T)
	b0, _ := t.b[0].Float64()

	r := Rule{Energy: make([]float64, n), Weight: make([]float64, n)}
	for i := range n {
		k := n - 1 - i // eigenvalues ascending in 1/E, so reverse for ascending E
		if lam[k] <= 0 {
			return Rule{}, fmt.Errorf("stieltjes: order %d produced a non-positive inverse "+
				"energy %.6g; the quadrature nodes must lie inside the input energy range", n, lam[k])
		}
		v := vec.At(0, k)
		r.Energy[i] = 1 / lam[k]
		r.Weight[i] = b0 * v * v
	}
	return r, nil
}

// Density differentiates a rule's cumulative distribution at the midpoint of each node
// interval (stieltjes_phi1.f:271-276):
//
//	E0_i     = (e_i + e_{i+1})/2
//	gamma_i  = (g_i + g_{i+1}) / (2 (e_{i+1} - e_i))
//
// which is the cumulative F, stepped by g_i/2 at each node, differenced across the
// interval. An n-point rule yields n-1 density points.
func (r Rule) Density() (grid, dens []float64) {
	n := len(r.Energy)
	if n < 2 {
		return nil, nil
	}
	grid = make([]float64, n-1)
	dens = make([]float64, n-1)
	for i := range n - 1 {
		grid[i] = 0.5 * (r.Energy[i] + r.Energy[i+1])
		dens[i] = 0.5 * (r.Weight[i+1] + r.Weight[i]) / (r.Energy[i+1] - r.Energy[i])
	}
	return grid, dens
}

// OrderResult is one Stieltjes order's outcome.
type OrderResult struct {
	Order int
	Gamma float64
	// Below / Above record the reference's two boundary cases: the requested energy fell
	// under the order's first density point (Gamma is then the crude 0.5 g_1/e_1 estimate
	// and "large inaccuracy expected"), or over its last (Gamma is 0).
	//
	// Both are traps for the averaging step, since the reference folds such values into
	// its mean regardless. They are recorded so that cannot happen silently; the paper's
	// protocol is immune by construction, because a window containing one of them has a
	// large spread and so is not "the region of best convergence".
	Below, Above bool
}

// Result is the imaged width.
type Result struct {
	Gamma float64 // Gamma(E) at the requested energy, in the input's energy unit
	Sigma float64 // the standard deviation over the averaged orders — the error bar

	Orders []OrderResult // every order evaluated, ascending
	Used   []int         // the orders that went into Gamma

	MaxOrder int  // the discovered maximum usable order
	Points   int  // input pseudo-continuum states
	LowOrder bool // MaxOrder < MinOrder: only a low-order approximation exists

	// Converged / ConvOrder report AverageReference's convergence search.
	Converged bool
	ConvOrder int

	Grid, Density []float64 // the imaged density at the highest order used, when requested
}

// Image runs the whole procedure: build the recurrence, evaluate Gamma at `at` for every
// available Stieltjes order, and combine the orders into one width with an error bar.
func Image(e, g []float64, at float64, opts Options) (*Result, error) {
	opts = opts.withDefaults()
	t, err := NewTable(e, g, opts)
	if err != nil {
		return nil, err
	}
	lo, hi := t.Range()
	if at < lo || at > hi {
		return nil, fmt.Errorf("stieltjes: the requested energy %.6g is outside the "+
			"pseudo-continuum's range [%.6g, %.6g]; imaging cannot extrapolate", at, lo, hi)
	}

	res := &Result{MaxOrder: t.max, Points: t.n}
	minOrd, maxOrd := opts.MinOrder, t.max
	if opts.MaxOrder > 0 && opts.MaxOrder < maxOrd {
		maxOrd = opts.MaxOrder
	}
	if maxOrd < opts.MinOrder {
		// The reference's low-order fallback (stieltjes_phi1.f:219-228): only the single
		// highest available order, flagged, rather than nothing.
		res.LowOrder = true
		minOrd = maxOrd
	}
	if maxOrd < 2 {
		return nil, fmt.Errorf("stieltjes: the polynomials lost orthogonality at order %d, "+
			"so no quadrature rule is available; %d input states may be too few, or their "+
			"energy range too narrow", t.max, t.n)
	}

	var rules []Rule
	for n := minOrd; n <= maxOrd; n++ {
		rule, err := t.Rule(n)
		if err != nil {
			return nil, err
		}
		rules = append(rules, rule)
	}
	if err := evaluateRules(res, rules, at, opts); err != nil {
		return nil, err
	}
	return res, nil
}

// ImageRules images Gauss quadrature rules of one measure that did not come from the
// moment recurrence: rules[i] must be the rule of order len(rules[i].Energy), ascending
// in order. It applies Image's per-order evaluation, density export and order averaging
// unchanged. The Fano Gauss engine uses it: a Lanczos run on PHP seeded by the coupling
// vector g yields the Gauss rule of every order for the measure sum_j |<psi_j|g>|^2
// delta(E - E_j) directly, in float64, with usable orders set by the Lanczos length
// rather than by the moment recurrence's loss of orthogonality.
//
// at must lie inside the highest-order rule's node range, as for Image.
func ImageRules(rules []Rule, at float64, opts Options) (*Result, error) {
	opts = opts.withDefaults()
	if len(rules) == 0 {
		return nil, fmt.Errorf("stieltjes: no quadrature rules to image")
	}
	top := rules[len(rules)-1]
	if len(top.Energy) < 2 {
		return nil, fmt.Errorf("stieltjes: the highest rule has %d node(s); imaging needs at least 2",
			len(top.Energy))
	}
	lo, hi := top.Energy[0], top.Energy[len(top.Energy)-1]
	if at < lo || at > hi {
		return nil, fmt.Errorf("stieltjes: the requested energy %.6g is outside the "+
			"pseudo-continuum's range [%.6g, %.6g]; imaging cannot extrapolate", at, lo, hi)
	}
	for i, r := range rules {
		if len(r.Energy) != len(r.Weight) || (i > 0 && len(r.Energy) <= len(rules[i-1].Energy)) {
			return nil, fmt.Errorf("stieltjes: rule %d is malformed or not of increasing order", i)
		}
		for k := 1; k < len(r.Energy); k++ {
			if r.Energy[k] <= r.Energy[k-1] {
				return nil, fmt.Errorf("stieltjes: rule of order %d has nodes out of order", len(r.Energy))
			}
		}
	}
	res := &Result{MaxOrder: len(top.Energy), Points: len(top.Energy)}
	if err := evaluateRules(res, rules, at, opts); err != nil {
		return nil, err
	}
	return res, nil
}

// evaluateRules evaluates Gamma(at) from every rule, exports the density of the last one
// when requested, and combines the orders.
func evaluateRules(res *Result, rules []Rule, at float64, opts Options) error {
	for _, rule := range rules {
		n := len(rule.Energy)
		grid, dens := rule.Density()
		or := OrderResult{Order: n}
		switch {
		case len(grid) == 0:
			continue // a one-node rule has no density
		case at < grid[0]:
			// Under the first midpoint: the reference's crude estimate, with its warning.
			or.Below = true
			or.Gamma = 0.5 * rule.Weight[0] / rule.Energy[0]
		case at > grid[len(grid)-1]:
			or.Above = true
			or.Gamma = 0
		default:
			p, err := NewPCHIP(grid, dens)
			if err != nil {
				return fmt.Errorf("stieltjes: order %d: %w", n, err)
			}
			or.Gamma = p.At(at)
		}
		res.Orders = append(res.Orders, or)
	}

	last := rules[len(rules)-1]
	if opts.GridPoints > 0 && len(last.Energy) >= 3 {
		grid, dens := last.Density()
		if p, err := NewPCHIP(grid, dens); err == nil {
			g0, g1 := grid[0], grid[len(grid)-1]
			res.Grid = make([]float64, opts.GridPoints)
			res.Density = make([]float64, opts.GridPoints)
			for i := range opts.GridPoints {
				x := g0 + (g1-g0)*float64(i)/float64(opts.GridPoints-1)
				res.Grid[i], res.Density[i] = x, p.At(x)
			}
		}
	}

	if opts.Average == AverageReference {
		combineReference(res, opts)
	} else {
		combinePaper(res, opts)
	}
	return nil
}

// combinePaper implements the paper's protocol (ADC22.pdf p. 10): average Window
// consecutive orders in the region of best convergence, and quote the statistical
// standard deviation over that window.
//
// "The region of best convergence" is read as the window of consecutive orders with the
// smallest standard deviation, which is the operational meaning of the phrase and also
// the definition that makes the quoted uncertainty the deviation of the window that was
// averaged. With fewer than Window orders available, every order is used and the window
// shrinks — reported through Result.Used, so a thin average is visible rather than
// presented as a nine-order one.
func combinePaper(res *Result, opts Options) {
	n := len(res.Orders)
	if n == 0 {
		return
	}
	w := min(opts.Window, n)
	best, bestSD := 0, math.Inf(1)
	for s := 0; s+w <= n; s++ {
		_, sd := meanSD(res.Orders[s : s+w])
		if sd < bestSD {
			best, bestSD = s, sd
		}
	}
	mean, sd := meanSD(res.Orders[best : best+w])
	res.Gamma, res.Sigma = mean, sd
	for _, or := range res.Orders[best : best+w] {
		res.Used = append(res.Used, or.Order)
	}
}

// combineReference reproduces stieltjes_phi1.f:327-370: the mean of the three highest
// orders, superseded by any three-order window whose mean relative spread meets a
// threshold, scanning from the highest order downwards. The threshold is relaxed by 1.2x
// per pass, up to ConvMax passes.
//
// Note that with the reference's own ConvFac = 1.0 the conv_fac^(max-imax) factor is
// identically 1, so the documented "stricter at lower orders" behaviour is inert at the
// default settings; the downward scan is what actually prefers high orders.
func combineReference(res *Result, opts Options) {
	n := len(res.Orders)
	if n == 0 {
		return
	}
	if n < 3 {
		// The reference's converge == 0 branch: take the highest order alone.
		res.Gamma = res.Orders[n-1].Gamma
		res.Used = []int{res.Orders[n-1].Order}
		return
	}
	mean, sd := meanSD(res.Orders[n-3:])
	res.Gamma, res.Sigma = mean, sd
	res.Used = []int{res.Orders[n-3].Order, res.Orders[n-2].Order, res.Orders[n-1].Order}
	if res.MaxOrder < 7 {
		return // the reference only searches when maxord >= 7
	}

	thresh := opts.ConvThresh
	for pass := 1; pass < opts.ConvMax; pass++ {
		for i := n - 1; i >= 2; i-- {
			g0, g1, g2 := res.Orders[i-2].Gamma, res.Orders[i-1].Gamma, res.Orders[i].Gamma
			gmean := (g0 + g1 + g2) / 3
			if gmean == 0 {
				continue
			}
			difmean := (math.Abs(g2-g1) + math.Abs(g1-g0) + math.Abs(g2-g0)) / 3
			if difmean/gmean <= thresh*math.Pow(opts.ConvFac, float64(n-1-i)) {
				_, sd := meanSD(res.Orders[i-2 : i+1])
				res.Gamma, res.Sigma = gmean, sd
				res.Converged, res.ConvOrder = true, res.Orders[i].Order
				res.Used = []int{res.Orders[i-2].Order, res.Orders[i-1].Order, res.Orders[i].Order}
				return
			}
		}
		thresh *= 1.2
	}
}

// meanSD is the mean and the sample standard deviation of a window's widths. The sample
// (n-1) denominator is the "statistical standard deviation" the paper quotes.
func meanSD(ors []OrderResult) (mean, sd float64) {
	n := len(ors)
	if n == 0 {
		return 0, 0
	}
	for _, or := range ors {
		mean += or.Gamma
	}
	mean /= float64(n)
	if n < 2 {
		return mean, 0
	}
	var v float64
	for _, or := range ors {
		d := or.Gamma - mean
		v += d * d
	}
	return mean, math.Sqrt(v / float64(n-1))
}

// PerOrder returns the per-order widths, for a convergence plot.
func (r *Result) PerOrder() (orders []int, gammas []float64) {
	for _, or := range r.Orders {
		orders = append(orders, or.Order)
		gammas = append(gammas, or.Gamma)
	}
	return orders, gammas
}

// Boundary counts the orders that fell outside their own density grid, which is the
// diagnostic that says an averaged width may be diluted by the reference's fallbacks.
func (r *Result) Boundary() (below, above int) {
	for _, or := range r.Orders {
		if or.Below {
			below++
		}
		if or.Above {
			above++
		}
	}
	return below, above
}

// String summarizes the result.
func (r *Result) String() string {
	below, above := r.Boundary()
	s := fmt.Sprintf("Gamma = %.6g +/- %.3g (%.2f%%), averaged over %d order(s) %v; "+
		"%d order(s) evaluated, maximum usable order %d, %d input states",
		r.Gamma, r.Sigma, 100*r.Sigma/nonZero(r.Gamma), len(r.Used), orderRange(r.Used),
		len(r.Orders), r.MaxOrder, r.Points)
	if r.LowOrder {
		s += "; LOW ORDER, below the usual minimum"
	}
	if r.Converged {
		s += fmt.Sprintf("; convergence detected at order %d", r.ConvOrder)
	}
	if below+above > 0 {
		s += fmt.Sprintf("; %d order(s) below and %d above their own density grid", below, above)
	}
	return s
}

func orderRange(used []int) string {
	if len(used) == 0 {
		return "[]"
	}
	u := append([]int(nil), used...)
	sort.Ints(u)
	if len(u) == 1 {
		return fmt.Sprintf("[%d]", u[0])
	}
	return fmt.Sprintf("[%d..%d]", u[0], u[len(u)-1])
}

func nonZero(x float64) float64 {
	if x == 0 {
		return 1
	}
	return x
}
