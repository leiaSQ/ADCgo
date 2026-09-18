package sip

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestPT2GateC11_2 checks Eq. (A4) against c11_2, the spin-free second-order
// self-energy that is bit-exact against theADCcode. This pins the v-amplitude
// convention of Eq. (A1), the antisymmetrized "1212" integral, the 1/2 prefactor
// and the epsilon combination — all of which A9-A14 reuse.
func TestPT2GateC11_2(t *testing.T) {
	mx := buildH2O(t, 2)
	e := mx.el
	var maxAbs, maxDiff float64
	for i := range e.nocc {
		for j := range e.nocc {
			want := e.c11_2(i, j)
			// The 1h spin function is the single determinant that removes the
			// beta electron; bra and ket carry the same operator-string sign, so
			// it cancels and no contraction is needed.
			got := e.m2_1h(sorb{Orb: i, Dn: true}, sorb{Orb: j, Dn: true})
			maxAbs = math.Max(maxAbs, math.Abs(want))
			maxDiff = math.Max(maxDiff, math.Abs(got-want))
		}
	}
	t.Logf("A4 vs c11_2 over %dx%d occupied pairs: max |element| = %.6g, max deviation = %.3g",
		e.nocc, e.nocc, maxAbs, maxDiff)
	if maxDiff > 1e-12 {
		t.Errorf("A4 deviates from c11_2 by %.3g (limit 1e-12)", maxDiff)
	}
}

// TestPT2GateC12_2 checks Eq. (A6) against c12_2. This is the closer analogue of
// the new block: it contracts over the spin-adapted 2h1p basis and carries the
// same [k <-> l] permutation operator that A10 and A13 use.
func TestPT2GateC12_2(t *testing.T) {
	mx := buildH2O(t, 3)
	e, sp := mx.el, mx.sp

	spinAdapted := func(j int, cfg Config) float64 {
		bra, braSign := expand1(j)
		var sum float64
		for _, tm := range e.expand2(cfg) {
			sum += braSign * tm.C * e.m2_1h2h1p(bra, tm.P.A, tm.P.K, tm.P.L)
		}
		return sum
	}

	phase := 0.0
	var maxAbs, maxDiff float64
	var n int
	for r := range sp.BeginSat {
		j := sp.Configs[r].Occ[0]
		for c := sp.BeginSat; c < len(sp.Configs); c++ {
			want := e.c12_2(j, sp.Configs[c])
			got := spinAdapted(j, sp.Configs[c])
			if math.Abs(want) < 1e-12 && math.Abs(got) < 1e-12 {
				continue
			}
			n++
			maxAbs = math.Max(maxAbs, math.Abs(want))
			if phase == 0 && math.Abs(want) > 1e-8 {
				phase = 1
				if got*want < 0 {
					phase = -1
				}
				t.Logf("global basis phase pinned to %+.0f", phase)
			}
			maxDiff = math.Max(maxDiff, math.Abs(phase*got-want))
		}
	}
	t.Logf("A6 vs c12_2 over %d nonzero couplings: max |element| = %.6g, max deviation = %.3g",
		n, maxAbs, maxDiff)
	if n == 0 {
		t.Fatal("no nonzero couplings compared")
	}
	if maxDiff > 1e-12 {
		t.Errorf("A6 deviates from c12_2 by %.3g (limit 1e-12)", maxDiff)
	}
}

// TestC22_2Hermitian requires the second-order 2h1p/2h1p block to be symmetric.
// Each of A10-A14 is manifestly invariant under exchanging the primed and
// unprimed configurations for real orbitals, so any asymmetry is a transposed
// index in the transcription.
func TestC22_2Hermitian(t *testing.T) {
	mx := buildH2O(t, 3)
	e, sp := mx.el, mx.sp
	cfgs := sp.Configs[sp.BeginSat:min(sp.BeginSat+24, len(sp.Configs))]
	var maxAsym, maxAbs float64
	for i, r := range cfgs {
		for _, c := range cfgs[i:] {
			x, y := e.c22_2(r, c), e.c22_2(c, r)
			maxAbs = math.Max(maxAbs, math.Abs(x))
			maxAsym = math.Max(maxAsym, math.Abs(x-y))
		}
	}
	t.Logf("second-order 2h1p block over %d configs: max |element| = %.6g, max asymmetry = %.3g",
		len(cfgs), maxAbs, maxAsym)
	if maxAbs == 0 {
		t.Fatal("the whole block is zero — nothing was tested")
	}
	if maxAsym > 1e-12 {
		t.Errorf("block asymmetry %.3g exceeds 1e-12", maxAsym)
	}
}

// TestC22_2SymmetryBlocked requires the block to vanish between configurations of
// different irreps. The element sums over spin orbitals with no explicit symmetry
// screening, so this checks that the integrals themselves enforce the blocking —
// an index slip between the primed and unprimed sides would leak across it.
func TestC22_2SymmetryBlocked(t *testing.T) {
	d, nocc := h2o22(t)
	eps := mp.OrbitalEnergies(d, nocc)
	// Build one space per irrep, then cross elements between different sectors.
	spA := NewSpace(nocc, d.NORB, d.OrbSym, 0)
	spB := NewSpace(nocc, d.NORB, d.OrbSym, 1)
	e := newElements(spA, integrals.New(d, nocc, d.OrbSym), eps, 3)

	var maxLeak float64
	na := min(20, len(spA.Configs)-spA.BeginSat)
	nb := min(20, len(spB.Configs)-spB.BeginSat)
	for _, r := range spA.Configs[spA.BeginSat : spA.BeginSat+na] {
		for _, c := range spB.Configs[spB.BeginSat : spB.BeginSat+nb] {
			maxLeak = math.Max(maxLeak, math.Abs(e.c22_2(r, c)))
		}
	}
	t.Logf("cross-irrep leakage over %dx%d configs: %.3g", na, nb, maxLeak)
	if maxLeak > 1e-12 {
		t.Errorf("second-order block leaks %.3g between irreps 0 and 1 (limit 1e-12)", maxLeak)
	}
}
