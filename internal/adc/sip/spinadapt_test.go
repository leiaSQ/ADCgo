package sip

import (
	"math"
	"testing"
)

// c12_1SO evaluates the first-order 1h/2h1p coupling from the paper's
// spin-orbital expression (ADC22.pdf Eq. A5),
//
//	M^1_{i,akl} = V_kl[ia] = <kl||ia>,
//
// by expanding both sides over primitive configurations. It is the spin-adapted
// element that c12_1 computes directly from a hand-derived spin table, so the two
// must agree — that agreement is what licenses using the same machinery for the
// blocks (A9-A22) where no hand-derived table exists.
func (e *elements) c12_1SO(j int, cfg Config) float64 {
	bra, braSign := expand1(j)
	var sum float64
	for _, t := range e.expand2(cfg) {
		sum += braSign * t.C * e.eriSO(t.P.K, t.P.L, bra, t.P.A)
	}
	return sum
}

// TestSpinAdaptGateC12_1 is the F0 gate. The spin-orbital route must reproduce
// the existing order-3 c12_1 — which is itself bit-exact against theADCcode — for
// every (1h, 2h1p) pair of the H2O sector, up to one global basis phase that is
// required to be the same for all of them.
//
// A global phase is legitimate: it is the freedom in how the 1h and 2h1p basis
// functions are signed, it flips the whole off-diagonal block at once, and it
// leaves the spectrum unchanged. A phase that varied element to element would not
// be a phase at all, and the test fails on that.
func TestSpinAdaptGateC12_1(t *testing.T) {
	mx := buildH2O(t, 3)
	e, sp := mx.el, mx.sp

	phase := 0.0
	var maxAbs, maxDiff float64
	var nCompared, nNonzero int

	for r := range sp.BeginSat {
		j := sp.Configs[r].Occ[0]
		for c := sp.BeginSat; c < len(sp.Configs); c++ {
			cfg := sp.Configs[c]
			want := e.c12_1(j, cfg)
			got := e.c12_1SO(j, cfg)
			nCompared++

			if math.Abs(want) < 1e-12 && math.Abs(got) < 1e-12 {
				continue
			}
			nNonzero++
			if math.Abs(want) > maxAbs {
				maxAbs = math.Abs(want)
			}
			// Pin the global phase on the first element with real magnitude.
			if phase == 0 && math.Abs(want) > 1e-8 {
				if got*want > 0 {
					phase = 1
				} else {
					phase = -1
				}
				t.Logf("global basis phase pinned to %+.0f "+
					"(j=%d cfg=%v: reference %.12g, spin-orbital %.12g)",
					phase, j, cfg, want, got)
			}
			if d := math.Abs(phase*got - want); d > maxDiff {
				maxDiff = d
			}
		}
	}

	t.Logf("compared %d pairs (%d nonzero), max |element| = %.6g, max deviation = %.3g",
		nCompared, nNonzero, maxAbs, maxDiff)
	if nNonzero == 0 {
		t.Fatal("no nonzero couplings compared — the gate proved nothing")
	}
	if maxDiff > 1e-13 {
		t.Errorf("spin-orbital c12_1 deviates from the reference by %.3g (limit 1e-13)", maxDiff)
	}
}
