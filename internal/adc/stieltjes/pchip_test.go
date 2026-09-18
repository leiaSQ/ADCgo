package stieltjes

import (
	"math"
	"math/rand"
	"testing"
)

// TestPCHIPInterpolatesItsKnots: the interpolant must pass through every data point
// exactly. A Hermite basis makes that structural, so this is the check that the basis and
// the interval search are right.
func TestPCHIPInterpolatesItsKnots(t *testing.T) {
	x := []float64{0.1, 0.4, 0.5, 1.7, 2.0, 5.5, 9.0}
	y := []float64{3.0, 1.0, 1.0, 0.0, 4.0, 4.0, 0.5}
	p, err := NewPCHIP(x, y)
	if err != nil {
		t.Fatal(err)
	}
	for i := range x {
		if got := p.At(x[i]); math.Abs(got-y[i]) > 1e-14 {
			t.Errorf("At(x[%d]=%.3g) = %.17g, want %.17g", i, x[i], got, y[i])
		}
	}
}

// TestPCHIPReproducesLinearData: on collinear points every secant slope is equal, so the
// weighted harmonic mean and both end formulas must return that same slope and the
// interpolant must be the line. This catches a sign or weight error in the derivative rule
// that knot interpolation alone would not.
func TestPCHIPReproducesLinearData(t *testing.T) {
	x := []float64{0.2, 0.9, 1.1, 3.0, 7.5}
	y := make([]float64, len(x))
	line := func(t float64) float64 { return 2.5*t + 0.75 }
	for i := range x {
		y[i] = line(x[i])
	}
	p, err := NewPCHIP(x, y)
	if err != nil {
		t.Fatal(err)
	}
	var worst float64
	for s := 0.2; s <= 7.5; s += 0.01 {
		worst = math.Max(worst, math.Abs(p.At(s)-line(s)))
	}
	t.Logf("linear data reproduced to %.3g over the whole range", worst)
	if worst > 1e-12 {
		t.Errorf("linear data is not reproduced: worst deviation %.3g", worst)
	}
}

// TestPCHIPNeverOvershoots is the property the whole choice of interpolant rests on: on
// each interval the curve stays between the two knot values, so non-negative data cannot
// produce a negative gamma(E). A cubic spline through the same points would overshoot, and
// an overshoot below zero is a negative decay width.
func TestPCHIPNeverOvershoots(t *testing.T) {
	rng := rand.New(rand.NewSource(20200521))
	for trial := range 200 {
		n := 4 + rng.Intn(12)
		x := make([]float64, n)
		y := make([]float64, n)
		v := rng.Float64()
		for i := range n {
			if i > 0 {
				v += 0.01 + rng.Float64()
			}
			x[i] = v
			// Non-negative, and deliberately spiky: the hard case is a sharp local maximum
			// between two near-zero neighbours, which is what makes a spline ring.
			y[i] = math.Max(0, rng.Float64()*math.Pow(10, float64(rng.Intn(5)-2)))
			if rng.Intn(4) == 0 {
				y[i] = 0
			}
		}
		p, err := NewPCHIP(x, y)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		for i := range n - 1 {
			lo := math.Min(y[i], y[i+1])
			hi := math.Max(y[i], y[i+1])
			span := hi - lo
			for k := range 21 {
				s := x[i] + (x[i+1]-x[i])*float64(k)/20
				got := p.At(s)
				if got < lo-1e-12*(1+span) || got > hi+1e-12*(1+span) {
					t.Fatalf("trial %d, interval %d [%.6g,%.6g] with knots %.6g,%.6g: "+
						"At(%.6g) = %.6g overshoots", trial, i, x[i], x[i+1], y[i], y[i+1], s, got)
				}
				if got < 0 {
					t.Fatalf("trial %d: At(%.6g) = %.6g is negative on non-negative data",
						trial, s, got)
				}
			}
		}
	}
	t.Log("200 random non-negative data sets: no interval overshoots its knot values, " +
		"so the interpolated density is never negative")
}

// TestPCHIPPreservesMonotonicity: on monotone data the interpolant must be monotone in the
// same direction everywhere, which is the Fritsch-Carlson condition itself.
func TestPCHIPPreservesMonotonicity(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := range 100 {
		n := 4 + rng.Intn(10)
		x := make([]float64, n)
		y := make([]float64, n)
		xv, yv := 0.0, 0.0
		for i := range n {
			xv += 0.05 + rng.Float64()
			yv += rng.Float64() * rng.Float64() // non-decreasing, often nearly flat
			x[i], y[i] = xv, yv
		}
		p, err := NewPCHIP(x, y)
		if err != nil {
			t.Fatal(err)
		}
		prev := math.Inf(-1)
		for s := x[0]; s <= x[n-1]; s += (x[n-1] - x[0]) / 500 {
			got := p.At(s)
			if got < prev-1e-11*(1+math.Abs(prev)) {
				t.Fatalf("trial %d: the interpolant decreased at E=%.6g (%.12g after %.12g) "+
					"on non-decreasing data", trial, s, got, prev)
			}
			prev = got
		}
	}
	t.Log("100 random non-decreasing data sets: the interpolant is non-decreasing throughout")
}

// TestPCHIPFlatWhereDataTurns: at a local extremum the derivative must be exactly zero, so
// the curve cannot cross the knot value there. That is the specific clause that keeps a
// spike from ringing negative on the far side.
func TestPCHIPFlatWhereDataTurns(t *testing.T) {
	x := []float64{0, 1, 2, 3}
	y := []float64{0, 5, 0, 0}
	p, err := NewPCHIP(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if p.d[1] != 0 {
		t.Errorf("the derivative at the local maximum is %.6g, want 0", p.d[1])
	}
	if p.d[2] != 0 {
		t.Errorf("the derivative at the local minimum is %.6g, want 0", p.d[2])
	}
	for s := 0.0; s <= 3.0; s += 0.005 {
		if v := p.At(s); v < 0 || v > 5 {
			t.Fatalf("At(%.4g) = %.6g escaped [0,5]", s, v)
		}
	}
}

// TestPCHIPClampsOutsideItsRange: outside the knots it returns the nearest knot value
// rather than extrapolating a cubic, which would leave the non-negativity guarantee behind.
func TestPCHIPClampsOutsideItsRange(t *testing.T) {
	p, err := NewPCHIP([]float64{1, 2, 3, 4}, []float64{9, 4, 1, 0.25})
	if err != nil {
		t.Fatal(err)
	}
	if got := p.At(-5); got != 9 {
		t.Errorf("At(-5) = %.6g, want the first knot value 9", got)
	}
	if got := p.At(99); got != 0.25 {
		t.Errorf("At(99) = %.6g, want the last knot value 0.25", got)
	}
	if p.Contains(-5) || p.Contains(99) || !p.Contains(2.5) {
		t.Error("Contains disagrees with the knot range")
	}
	if lo, hi := p.Range(); lo != 1 || hi != 4 {
		t.Errorf("Range() = (%.6g, %.6g), want (1, 4)", lo, hi)
	}
}

func TestPCHIPRejectsBadInput(t *testing.T) {
	for _, c := range []struct {
		name string
		x, y []float64
	}{
		{"length mismatch", []float64{1, 2, 3}, []float64{1, 2}},
		{"one point", []float64{1}, []float64{1}},
		{"not ascending", []float64{1, 3, 2}, []float64{1, 2, 3}},
		{"repeated abscissa", []float64{1, 2, 2}, []float64{1, 2, 3}},
	} {
		if _, err := NewPCHIP(c.x, c.y); err == nil {
			t.Errorf("%s: accepted", c.name)
		}
	}
	// Two points is the minimum, and gives the straight line through them.
	p, err := NewPCHIP([]float64{1, 3}, []float64{2, 6})
	if err != nil {
		t.Fatal(err)
	}
	if got := p.At(2); math.Abs(got-4) > 1e-14 {
		t.Errorf("two-point interpolant At(2) = %.6g, want 4", got)
	}
}
