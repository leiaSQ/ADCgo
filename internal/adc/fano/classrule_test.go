package fano

import "testing"

// classrule_test.go — the per-class Q/P rules of the supplementary material to
// Kolorenč/Averbukh, J. Chem. Phys. 152, 214107 (2020), checked configuration by
// configuration against its wording.
//
// These are not round-trip tests of the parser. Each case names a physical
// configuration and asserts which subspace it must land in, with the reason, because
// that is what a misreading of the supplement would get wrong — and a Q/P misassignment
// does not fail loudly, it changes the width.

// TestMgClassRule covers "P subspace spanned by 2h1p ISs with at least one 3s hole and
// by 3h2p ISs with two 3s holes, with the exception of ISs with at least one 2s vacancy,
// which belong naturally to the Q subspace".
//
// Orbital indices are Mg's occupied set after freezing 1s: 2s = 0, 2p = 1,2,3, 3s = 4.
func TestMgClassRule(t *testing.T) {
	r, err := ParseClassRule("q:0:1;p:2/4:1;p:3/4:2", "Mg 2s")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	cases := []struct {
		name  string
		holes []int
		bound bool
		why   string
	}{
		{"1h 2s^-1 (the decaying state)", []int{0}, true,
			"Phi itself; it carries the 2s vacancy"},
		{"1h 3s^-1 (ground ionic state)", []int{4}, true,
			"bound: a single 3s hole is Mg+(3s^-1), not a decay channel"},
		{"2h1p 3s^-2", []int{4, 4}, false,
			"the open Auger channel 3s^-2 + e-"},
		{"2h1p 2p^-1 3s^-1", []int{1, 4}, false,
			"the other open channel, 2p^-1 3s^-1 + e-"},
		{"2h1p 2p^-2", []int{1, 2}, true,
			"CLOSED: 2p^-2 lies at 4.6 Eh, above the 3.6 Eh vacancy, so it is bound"},
		{"2h1p 2s^-1 3s^-1", []int{0, 4}, true,
			"the stated exception: anything with a 2s vacancy is Q despite the 3s hole"},
		{"3h2p 2p^-1 3s^-2", []int{1, 4, 4}, false,
			"two 3s holes, so P — the assignment the paper says it made deliberately"},
		{"3h2p 2p^-2 3s^-1", []int{1, 2, 4}, true,
			"only one 3s hole, so not continuum by the stated rule"},
		{"3h2p 2s^-1 3s^-2", []int{0, 4, 4}, true,
			"two 3s holes but a 2s vacancy, so the exception wins"},
	}
	for _, c := range cases {
		if got := r.Bound(c.holes); got != c.bound {
			t.Errorf("%s: Bound=%v want %v (%s)", c.name, got, c.bound, c.why)
		}
	}
}

// TestKrClassRule covers "Q subspace is defined by configurations with at least one 3d
// vacancy plus all 3h2p ISs associated with Kr+(4s^-2 4p^-1 nl^+2) configurations".
//
// Indices are Kr's occupied set after the [Ne] ECP and freezing 3s/3p: 3d = 0..4,
// 4s = 5, 4p = 6,7,8.
func TestKrClassRule(t *testing.T) {
	r, err := ParseClassRule("q:0-4:1;q:3/5:2:2&3/6-8:1", "Kr 3d")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	cases := []struct {
		name  string
		holes []int
		bound bool
		why   string
	}{
		{"1h 3d^-1", []int{0}, true, "Phi: carries the 3d vacancy"},
		{"2h1p 4s^-1 4p^-1", []int{5, 6}, false, "an open M4,5-NN Auger channel"},
		{"2h1p 4p^-2", []int{6, 7}, false, "the dominant open channel"},
		{"2h1p 3d^-1 4p^-1", []int{0, 6}, true, "still carries a 3d hole"},
		{"3h2p 4s^-2 4p^-1", []int{5, 5, 6}, true,
			"the named shake-up family, explicitly placed in Q"},
		{"3h2p 4s^-1 4p^-2", []int{5, 6, 7}, false,
			"only one 4s hole, so not the named family: continuum"},
		{"3h2p 4p^-3", []int{6, 7, 8}, false, "double-Auger continuum"},
		{"3h2p 3d^-1 4p^-2", []int{0, 6, 7}, true, "3d hole present"},
	}
	for _, c := range cases {
		if got := r.Bound(c.holes); got != c.bound {
			t.Errorf("%s: Bound=%v want %v (%s)", c.name, got, c.bound, c.why)
		}
	}
}

// TestClassRuleReducesToHoleLocalization pins that Ne's and Ar's rules — which the
// supplement states as plain any-hole criteria — agree with HoleLocalization
// configuration for configuration. If the general machinery did not reduce to the
// special case it would silently change the two ions that were already right.
func TestClassRuleReducesToHoleLocalization(t *testing.T) {
	for _, tc := range []struct {
		name string
		spec string
		orbs []int
	}{
		{"Ne 1s", "q:0:1", []int{0}},
		{"Ar 2s+2p", "q:0-3:1", []int{0, 1, 2, 3}},
	} {
		cr, err := ParseClassRule(tc.spec, tc.name)
		if err != nil {
			t.Fatalf("%s: parse: %v", tc.name, err)
		}
		hl, err := NewHoleLocalization(AnyHole, tc.orbs, tc.name)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		// Every hole multiset over 0..5 up to three holes.
		var holes []int
		for a := range 6 {
			for b := range 6 {
				for c := range 6 {
					for _, h := range [][]int{{a}, {a, b}, {a, b, c}} {
						holes = append(holes[:0], h...)
						if cr.Bound(holes) != hl.Bound(holes) {
							t.Fatalf("%s: disagree on %v: class=%v hole=%v",
								tc.name, holes, cr.Bound(holes), hl.Bound(holes))
						}
					}
				}
			}
		}
	}
}

// TestClassRuleRejectsBadSpecs — a malformed rule must fail loudly. A rule that parses
// to nothing would send every configuration to P, which shows up as "Q holds no main
// configuration" much later, after the integrals have been read.
func TestClassRuleRejectsBadSpecs(t *testing.T) {
	for _, spec := range []string{
		"", "0:1", "x:0:1", "q:0", "q::1", "q:0:1:2:3", "q:a:1", "q:0:b",
		"q:3-1:1", "q:0:-1", "q:0:2:1", "q:z/0:1", "p:", "q:0:1&",
	} {
		if r, err := ParseClassRule(spec, ""); err == nil {
			t.Errorf("spec %q parsed to %v, want an error", spec, r)
		}
	}
}

// TestEmptyClauseNeverMatches guards the one way a bug here could silently widen Q: an
// empty conjunction is vacuously true in logic, which would make every configuration
// bound. Clause.match returns false instead.
func TestEmptyClauseNeverMatches(t *testing.T) {
	if (Clause{}).match([]int{0}) {
		t.Fatal("an empty clause matched; it must not")
	}
	if _, err := NewClassRule([]Clause{{}}, nil, ""); err == nil {
		t.Fatal("NewClassRule accepted an empty clause")
	}
}
