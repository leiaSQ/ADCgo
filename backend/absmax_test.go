package backend

import (
	"math"
	"math/rand"
	"testing"
)

// TestAbsMax2MatchesTheDefinition pins the free function to max(re^2 + im^2), written
// out here rather than borrowed from the implementation.
//
// The exactness claim is the point: a max reduction associates any way it likes and
// still lands on the same element, so a device implementation must agree with this one
// to the last bit, not to a tolerance. The comparison below is therefore ==.
func TestAbsMax2MatchesTheDefinition(t *testing.T) {
	be := Gonum{}
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{1, 2, 17, 1000} {
		re, im := make([]float64, n), make([]float64, n)
		for i := range re {
			re[i], im[i] = rng.NormFloat64(), rng.NormFloat64()
		}
		want := 0.0
		for i := range re {
			if v := re[i]*re[i] + im[i]*im[i]; v > want {
				want = v
			}
		}
		if got := AbsMax2(be, be.Upload(re), be.Upload(im)); got != want {
			t.Fatalf("n=%d: AbsMax2 = %g, want %g", n, got, want)
		}
		if got, w := AbsMax(be, be.Upload(re), be.Upload(im)), math.Sqrt(want); got != w {
			t.Fatalf("n=%d: AbsMax = %g, want %g", n, got, w)
		}
	}
}

// TestAbsMax2HandlesTheDegenerateCases: an empty vector has no largest element, and a
// purely real or purely imaginary one must still find it.
func TestAbsMax2HandlesTheDegenerateCases(t *testing.T) {
	be := Gonum{}
	if got := AbsMax2(be, be.Alloc(0), be.Alloc(0)); got != 0 {
		t.Fatalf("empty: got %g, want 0", got)
	}
	re := []float64{1, -3, 2}
	zero := []float64{0, 0, 0}
	if got := AbsMax2(be, be.Upload(re), be.Upload(zero)); got != 9 {
		t.Fatalf("real only: got %g, want 9", got)
	}
	if got := AbsMax2(be, be.Upload(zero), be.Upload(re)); got != 9 {
		t.Fatalf("imaginary only: got %g, want 9", got)
	}
}
