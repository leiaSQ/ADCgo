package backend

import "math"

// Largest squared modulus of a planar complex vector.
//
// MCTDH's Bulirsch-Stoer error estimate is AbsBSError (source/lib/ode/bslib.f:463),
// `Error = Max(Error, Dble(Abs(PsiError(D))))` -- the largest complex modulus over the
// extrapolation difference. It is called once per extrapolation column, several columns
// per large step, several large steps per constant-mean-field interval, and it wants one
// number. Getting it by downloading the whole vector makes every column a device-to-host
// stall on a quantity that is eight bytes.
//
// AbsMax2 returns the largest re[i]^2 + im[i]^2 rather than the largest modulus, and
// leaves the square root to the caller. That is not a shortcut, it is what makes a
// device implementation reproduce the host one exactly: x -> sqrt(x) is monotone, so the
// maximum is attained at the same element either way, and a max reduction is exact
// regardless of the order it associates in -- unlike a sum. Taking the modulus per
// element first would instead pin the answer to whichever hypot() the two sides happen
// to use, and no CUDA reduction reproduces Go's math.Hypot bit for bit.

// AbsMax2er is an optional capability: the largest squared modulus of a planar complex
// vector, computed without moving the vector.
type AbsMax2er interface {
	AbsMax2(re, im Vector) float64
}

// AbsMax2 returns max(re[i]^2 + im[i]^2), using the backend's own reduction when it has
// one and falling back to a download and a host loop when it does not.
//
// re and im must have the same length. An empty vector gives 0.
func AbsMax2(be Backend, re, im Vector) float64 {
	if re.Len() != im.Len() {
		panic("backend: AbsMax2 halves differ in length")
	}
	if re.Len() == 0 {
		return 0
	}
	if a, ok := be.(AbsMax2er); ok {
		return a.AbsMax2(re, im)
	}
	hr, hi := be.Download(re), be.Download(im)
	m := 0.0
	for i, r := range hr {
		if v := r*r + hi[i]*hi[i]; v > m {
			m = v
		}
	}
	return m
}

// AbsMax returns max |re[i] + i*im[i]|, the complex modulus AbsBSError takes.
func AbsMax(be Backend, re, im Vector) float64 { return math.Sqrt(AbsMax2(be, re, im)) }
