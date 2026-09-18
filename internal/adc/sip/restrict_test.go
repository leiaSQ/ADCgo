package sip

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestRestrictIsTheParentSubBlock is the gate the whole Fano design rests on: the
// matrix assembled over a restricted space must BE the corresponding sub-block of the
// parent's matrix, element for element.
//
// That is what lets the Fano layer hand row subsets to Restrict and treat QMQ and PMP
// as ordinary ADC matrices — solvers, matrix-free appliers and GPU backends all
// unchanged — instead of wrapping the operator in a projector. If it did not hold,
// every width computed through that path would be wrong with nothing to show it.
//
// The agreement should be exact, not merely close. Restrict preserves the parent's row
// order, so the directional elements (c22off, which evaluates its k==l and m==n
// branches by argument position) are called with the same arguments in the same order
// in both matrices. The test asserts exactness and logs any deviation, so a future
// change that reorders rows shows up here rather than as a shifted spectrum.
func TestRestrictIsTheParentSubBlock(t *testing.T) {
	d, nocc := h2o22(t)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	const norb = 8

	cases := []struct {
		name  string
		space func() *Space
		order int
	}{
		{"order 3", func() *Space { return NewSpace(nocc, norb, nil, 0) }, 3},
		{"ADC(2,2)f", func() *Space { return NewSpace22(nocc, norb, nil, 0) }, Order22},
	}
	for _, tc := range cases {
		parent := tc.space()
		pmx := New(parent, ints, eps, tc.order, backend.Gonum{})
		pmx.SetVariant(VariantF)
		M := pmx.BuildMatrix()

		// A row subset with a genuinely mixed character: every class, contiguous runs
		// and isolated rows, so the class-boundary recount and the group rebuild are
		// both exercised. "Holes all below nocc/2" is the shape a Fano scheme A
		// selector produces.
		var rows []int
		holes := make([]int, 0, 3)
		for r := range parent.Size() {
			keep := true
			for _, h := range parent.Holes(r, holes[:0]) {
				if h >= nocc/2 {
					keep = false
					break
				}
			}
			if keep {
				rows = append(rows, r)
			}
		}
		if len(rows) < 3 || len(rows) == parent.Size() {
			t.Fatalf("%s: the row subset is degenerate (%d of %d)", tc.name, len(rows), parent.Size())
		}

		sub := parent.Restrict(rows)
		if sub.Size() != len(rows) {
			t.Fatalf("%s: restricted space has %d rows, want %d", tc.name, sub.Size(), len(rows))
		}
		smx := New(sub, ints, eps, tc.order, backend.Gonum{})
		smx.SetVariant(VariantF)
		S := smx.BuildMatrix()

		var maxDiff, maxAbs float64
		for i := range rows {
			for j := range rows {
				want := M.At(rows[i], rows[j])
				got := S.At(i, j)
				maxAbs = math.Max(maxAbs, math.Abs(want))
				maxDiff = math.Max(maxDiff, math.Abs(got-want))
			}
		}
		t.Logf("%s: restricted %d of %d rows (1h %d, 2h1p %d, 3h2p %d); "+
			"max |element| = %.6g, max deviation from the parent sub-block = %.3g",
			tc.name, len(rows), parent.Size(), sub.BeginSat,
			len(sub.Configs)-sub.BeginSat, len(sub.Sat3), maxAbs, maxDiff)
		if maxDiff != 0 {
			t.Errorf("%s: restricted matrix deviates from the parent sub-block by %.3g "+
				"(expected exact agreement)", tc.name, maxDiff)
		}
	}
}

// TestRestrictAllRowsIsIdentity requires restricting to every row to reproduce the
// space itself, group boundaries included. That is what pins regroup() against the
// constructors: it recovers Group and Group3 by scanning for a change of key, and this
// is the case where the answer is already known.
func TestRestrictAllRowsIsIdentity(t *testing.T) {
	d, nocc := h2o22(t)
	for _, tc := range []struct {
		name string
		sp   *Space
	}{
		{"order 3", NewSpace(nocc, d.NORB, d.OrbSym, 0)},
		{"ADC(2,2)", NewSpace22(nocc, 10, d.OrbSym, 0)},
		{"ADC(2,2) C1", NewSpace22(nocc, 9, nil, 0)},
	} {
		all := make([]int, tc.sp.Size())
		for i := range all {
			all[i] = i
		}
		sub := tc.sp.Restrict(all)
		if sub.Size() != tc.sp.Size() || sub.BeginSat != tc.sp.BeginSat ||
			sub.Begin3h2p != tc.sp.Begin3h2p || len(sub.Sat3) != len(tc.sp.Sat3) {
			t.Errorf("%s: boundaries differ: got size=%d main=%d begin3=%d sat3=%d, "+
				"want size=%d main=%d begin3=%d sat3=%d", tc.name,
				sub.Size(), sub.BeginSat, sub.Begin3h2p, len(sub.Sat3),
				tc.sp.Size(), tc.sp.BeginSat, tc.sp.Begin3h2p, len(tc.sp.Sat3))
		}
		for i := range tc.sp.Configs {
			if sub.Configs[i] != tc.sp.Configs[i] {
				t.Fatalf("%s: config %d differs", tc.name, i)
			}
		}
		for i := range tc.sp.Sat3 {
			if sub.Sat3[i] != tc.sp.Sat3[i] {
				t.Fatalf("%s: 3h2p config %d differs", tc.name, i)
			}
		}
		eq := func(what string, got, want []int) {
			if len(got) != len(want) {
				t.Errorf("%s: %s has %d boundaries, want %d", tc.name, what, len(got), len(want))
				return
			}
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("%s: %s[%d] = %d, want %d", tc.name, what, i, got[i], want[i])
					return
				}
			}
		}
		eq("Group", sub.Group, tc.sp.Group)
		eq("Group3", sub.Group3, tc.sp.Group3)
		t.Logf("%s: identity restriction reproduces %d rows, %d satellite groups, %d 3h2p groups",
			tc.name, sub.Size(), len(sub.Group), len(sub.Group3))
	}
}

// TestHolesMatchesConfigs checks Holes against the configuration structs directly, for
// every class. It is the one function the Fano selector sees the space through, and an
// off-by-one in the class boundaries would silently reclassify whole bands — which
// would not fail any other test, just partition the space wrongly.
func TestHolesMatchesConfigs(t *testing.T) {
	d, nocc := h2o22(t)
	for _, sp := range []*Space{
		NewSpace(nocc, d.NORB, d.OrbSym, 0),
		NewSpace22(nocc, 10, d.OrbSym, 0),
	} {
		buf := make([]int, 0, 3)
		for r := range sp.Size() {
			h := sp.Holes(r, buf[:0])
			switch {
			case r < sp.BeginSat:
				if len(h) != 1 || h[0] != sp.Configs[r].Occ[0] {
					t.Fatalf("row %d (1h): Holes = %v, want [%d]", r, h, sp.Configs[r].Occ[0])
				}
			case r < len(sp.Configs):
				c := sp.Configs[r]
				if len(h) != 2 || h[0] != c.Occ[0] || h[1] != c.Occ[1] {
					t.Fatalf("row %d (2h1p): Holes = %v, want %v", r, h, c.Occ)
				}
			default:
				c := sp.Sat3[r-len(sp.Configs)]
				if len(h) != 3 || h[0] != c.Core || h[1] != c.L || h[2] != c.M {
					t.Fatalf("row %d (3h2p): Holes = %v, want [%d %d %d]", r, h, c.Core, c.L, c.M)
				}
			}
		}
	}
}
