package stieltjes

import (
	"fmt"
	"math"
)

// pchip.go — monotonicity-preserving piecewise cubic Hermite interpolation.
//
// This replaces the reference's NAG calls E01BEF (compute the derivatives) and E01BFF
// (evaluate), which implement the method of F. N. Fritsch and J. Butland, "A method for
// constructing local monotone piecewise cubic interpolants", SIAM J. Sci. Stat. Comput.
// 5, 300 (1984) — the same rule as SciPy's PchipInterpolator.
//
// It is not cosmetic, and not interchangeable with a spline. A cubic spline through the
// numerically differentiated cumulative distribution overshoots between points, and an
// overshoot below zero is a NEGATIVE DECAY WIDTH. The monotone interpolant cannot
// overshoot: on each interval it stays within [min(y_i,y_{i+1}), max(y_i,y_{i+1})], so
// non-negative data gives a non-negative gamma(E) everywhere. That guarantee is the only
// reason this particular interpolant is specified rather than "some smooth curve".

// PCHIP is a monotone cubic Hermite interpolant through ascending knots.
type PCHIP struct {
	x, y, d []float64
}

// NewPCHIP builds the interpolant. x must be strictly ascending with at least two
// points; y may be any values, but the non-negativity guarantee only follows for
// non-negative y.
func NewPCHIP(x, y []float64) (*PCHIP, error) {
	if len(x) != len(y) {
		return nil, fmt.Errorf("stieltjes: pchip got %d abscissae and %d ordinates", len(x), len(y))
	}
	n := len(x)
	if n < 2 {
		return nil, fmt.Errorf("stieltjes: pchip needs at least 2 points, got %d", n)
	}
	for i := 1; i < n; i++ {
		if !(x[i] > x[i-1]) {
			return nil, fmt.Errorf("stieltjes: pchip abscissae are not strictly ascending at %d "+
				"(%.17g then %.17g)", i, x[i-1], x[i])
		}
	}

	h := make([]float64, n-1)
	del := make([]float64, n-1)
	for i := range h {
		h[i] = x[i+1] - x[i]
		del[i] = (y[i+1] - y[i]) / h[i]
	}

	d := make([]float64, n)
	if n == 2 {
		d[0], d[1] = del[0], del[0]
		return &PCHIP{x: x, y: y, d: d}, nil
	}

	// Interior nodes: the weighted harmonic mean of the neighbouring secant slopes, and
	// zero at a local extremum (where the secants differ in sign). The harmonic mean is
	// what bounds |d_i| by 3·min(|del|), which is Fritsch-Carlson's sufficient condition
	// for monotonicity — an arithmetic mean does not satisfy it.
	for i := 1; i < n-1; i++ {
		if del[i-1]*del[i] <= 0 {
			d[i] = 0
			continue
		}
		w1 := 2*h[i] + h[i-1]
		w2 := h[i] + 2*h[i-1]
		d[i] = (w1 + w2) / (w1/del[i-1] + w2/del[i])
	}
	d[0] = endSlope(h[0], h[1], del[0], del[1])
	d[n-1] = endSlope(h[n-2], h[n-3], del[n-2], del[n-3])
	return &PCHIP{x: x, y: y, d: d}, nil
}

// endSlope is the shape-preserving one-sided end derivative: the three-point one-sided
// estimate, then clamped so the end interval stays monotone — zeroed if it disagrees in
// sign with the adjacent secant, and capped at 3x that secant if the data turns there.
func endSlope(h0, h1, d0, d1 float64) float64 {
	d := ((2*h0+h1)*d0 - h0*d1) / (h0 + h1)
	switch {
	case sign(d) != sign(d0):
		return 0
	case sign(d0) != sign(d1) && math.Abs(d) > math.Abs(3*d0):
		return 3 * d0
	}
	return d
}

func sign(x float64) int {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	}
	return 0
}

// At evaluates the interpolant. Outside [x[0], x[n-1]] it clamps to the nearest knot
// value rather than extrapolating a cubic, which would leave the non-negativity
// guarantee behind; callers that care about the distinction test Contains first.
func (p *PCHIP) At(t float64) float64 {
	n := len(p.x)
	if t <= p.x[0] {
		return p.y[0]
	}
	if t >= p.x[n-1] {
		return p.y[n-1]
	}
	// Binary search for the interval containing t.
	lo, hi := 0, n-1
	for hi-lo > 1 {
		mid := (lo + hi) / 2
		if p.x[mid] <= t {
			lo = mid
		} else {
			hi = mid
		}
	}
	h := p.x[hi] - p.x[lo]
	s := (t - p.x[lo]) / h
	// Hermite basis on [0,1].
	s2, s3 := s*s, s*s*s
	h00 := 2*s3 - 3*s2 + 1
	h10 := s3 - 2*s2 + s
	h01 := -2*s3 + 3*s2
	h11 := s3 - s2
	return h00*p.y[lo] + h*h10*p.d[lo] + h01*p.y[hi] + h*h11*p.d[hi]
}

// Contains reports whether t lies inside the knot range, i.e. whether At interpolates
// rather than clamps.
func (p *PCHIP) Contains(t float64) bool {
	return t >= p.x[0] && t <= p.x[len(p.x)-1]
}

// Range returns the knot range.
func (p *PCHIP) Range() (lo, hi float64) { return p.x[0], p.x[len(p.x)-1] }
