package sip

import (
	"math"
	"testing"
)

// detOf turns a primitive 2h1p operator string c†_A c_K c_L into its sorted
// determinant and the sign relating the two. The holes are applied rightmost
// first, which for this string means L before K.
func (e *elements) detOf(p prim2) (det, float64) {
	return excite(e.nocc, []sorb{p.L, p.K}, []sorb{p.A})
}

// c22_01SO evaluates the zeroth- plus first-order 2h1p/2h1p element through the
// Slater-Condon oracle: expand both spin-adapted configurations over primitives,
// map each primitive to a determinant, and contract <D|H - E_0|D'>.
func (e *elements) c22_01SO(row, col Config, e0 float64) float64 {
	type dc struct {
		d det
		c float64
	}
	conv := func(cfg Config) []dc {
		var out []dc
		for _, t := range e.expand2(cfg) {
			d, s := e.detOf(t.P)
			if d == nil {
				continue
			}
			out = append(out, dc{d: d, c: t.C * s})
		}
		return out
	}
	rs, cs := conv(row), conv(col)
	var sum float64
	for _, r := range rs {
		for _, c := range cs {
			sum += r.c * c.c * e.shiftedHamElem(r.d, c.d, e0)
		}
	}
	return sum
}

// TestSlaterGateC22 is the decisive F0 gate: the Slater-Condon oracle must
// reproduce the existing order-3 2h1p/2h1p block (c22diag on the diagonal,
// c22off off it) — code that is bit-exact against theADCcode — over the whole H2O
// satellite space.
//
// This is a stronger statement than the c12_1 gate, because it exercises the
// spin-adapted 2h1p basis on BOTH sides of the element, which is the situation
// every new ADC(2,2) block is in. A single global sign is allowed (the paper's
// M = -(K+C) against ADCgo's own convention) but it is pinned on the DIAGONAL,
// where no basis phase can hide, and then held fixed for every off-diagonal.
func TestSlaterGateC22(t *testing.T) {
	mx := buildH2O(t, 3)
	e, sp := mx.el, mx.sp
	e0 := e.refEnergy()

	// Pin the convention sign on the diagonal, which is phase-invariant.
	r0 := sp.BeginSat
	dRef, dGot := e.c22diag(sp.Configs[r0]), e.c22_01SO(sp.Configs[r0], sp.Configs[r0], e0)
	conv := 1.0
	if dRef*dGot < 0 {
		conv = -1
	}
	t.Logf("convention sign pinned on the diagonal to %+.0f (reference %.12g, oracle %.12g)",
		conv, dRef, dGot)

	var maxDiag, maxOff, maxAbs float64
	n := len(sp.Configs)
	for r := sp.BeginSat; r < n; r++ {
		for c := sp.BeginSat; c < n; c++ {
			var want float64
			if r == c {
				want = e.c22diag(sp.Configs[r])
			} else {
				want = e.c22off(sp.Configs[r], sp.Configs[c])
			}
			got := conv * e.c22_01SO(sp.Configs[r], sp.Configs[c], e0)
			if math.Abs(want) > maxAbs {
				maxAbs = math.Abs(want)
			}
			d := math.Abs(got - want)
			if r == c {
				if d > maxDiag {
					maxDiag = d
				}
			} else if d > maxOff {
				maxOff = d
			}
		}
	}
	t.Logf("satellite space %d configs, max |element| = %.6g, max deviation: diagonal %.3g, off-diagonal %.3g",
		n-sp.BeginSat, maxAbs, maxDiag, maxOff)
	if maxDiag > 1e-12 {
		t.Errorf("diagonal deviates by %.3g (limit 1e-12)", maxDiag)
	}
	if maxOff > 1e-12 {
		t.Errorf("off-diagonal deviates by %.3g (limit 1e-12)", maxOff)
	}
}
