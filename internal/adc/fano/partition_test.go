package fano

import (
	"math"
	"runtime"
	"slices"
	"testing"
)

// fakeSpace is a Space with hand-written hole lists, so the partition and selector
// logic can be tested without an integral file.
type fakeSpace struct {
	main  int
	holes [][]int
}

func (s *fakeSpace) Size() int          { return len(s.holes) }
func (s *fakeSpace) MainBlockSize() int { return s.main }
func (s *fakeSpace) Holes(row int, dst []int) []int {
	return append(dst, s.holes[row]...)
}

func TestHoleLocalizationRules(t *testing.T) {
	all, err := NewHoleLocalization(AllHoles, []int{0, 2}, "site A")
	if err != nil {
		t.Fatal(err)
	}
	any, err := NewHoleLocalization(AnyHole, []int{0}, "core")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		holes            []int
		wantAll, wantAny bool
	}{
		{[]int{0}, true, true},
		{[]int{2}, true, false},
		{[]int{0, 2}, true, true},
		{[]int{0, 1}, false, true},   // one hole outside {0,2}, but retains 0
		{[]int{1, 3}, false, false},  // nothing shared
		{[]int{0, 2, 0}, true, true}, // a coincident hole pair
		{[]int{}, false, false},      // vacuous: not bound under either rule
	} {
		if got := all.Bound(c.holes); got != c.wantAll {
			t.Errorf("AllHoles{0,2}.Bound(%v) = %v, want %v", c.holes, got, c.wantAll)
		}
		if got := any.Bound(c.holes); got != c.wantAny {
			t.Errorf("AnyHole{0}.Bound(%v) = %v, want %v", c.holes, got, c.wantAny)
		}
	}
	if _, err := NewHoleLocalization(AllHoles, nil, ""); err == nil {
		t.Error("an empty Q orbital set was accepted; it makes Q empty, so the run cannot produce a rate")
	}
	t.Logf("%s", all)
	t.Logf("%s", any)
}

// TestPartitionCoversEveryRow pins the Feshbach invariants: Q and P are disjoint,
// together they are every row, both are ascending, and the index maps invert the lists.
// QP = 0 downstream depends on exactly this.
func TestPartitionCoversEveryRow(t *testing.T) {
	sp := &fakeSpace{main: 3, holes: [][]int{
		{0}, {1}, {2},
		{0, 0}, {0, 1}, {1, 2}, {0, 2}, {2, 2},
		{0, 0, 1}, {1, 2, 2}, {0, 1, 2},
	}}
	sel, err := NewHoleLocalization(AllHoles, []int{0, 1}, "")
	if err != nil {
		t.Fatal(err)
	}
	p := NewPartition(sp, sel)
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}

	seen := make([]int, sp.Size())
	for _, r := range p.Q {
		seen[r]++
	}
	for _, r := range p.P {
		seen[r]++
	}
	for r, n := range seen {
		if n != 1 {
			t.Errorf("row %d appears in %d subspaces, want exactly 1", r, n)
		}
	}
	if !slices.IsSorted(p.Q) || !slices.IsSorted(p.P) {
		t.Errorf("subspace row lists are not ascending: Q=%v P=%v", p.Q, p.P)
	}
	for i, r := range p.Q {
		if p.QIndex(r) != i || p.PIndex(r) != -1 {
			t.Errorf("row %d: QIndex=%d (want %d), PIndex=%d (want -1)", r, p.QIndex(r), i, p.PIndex(r))
		}
	}
	for i, r := range p.P {
		if p.PIndex(r) != i || p.QIndex(r) != -1 {
			t.Errorf("row %d: PIndex=%d (want %d), QIndex=%d (want -1)", r, p.PIndex(r), i, p.QIndex(r))
		}
	}
	// {0,1} under AllHoles: rows 0,1 (1h), 3,4 (2h1p) and 8 (3h2p).
	wantQ := []int{0, 1, 3, 4, 8}
	if !slices.Equal(p.Q, wantQ) {
		t.Errorf("Q = %v, want %v", p.Q, wantQ)
	}
	if p.QMain != 2 || p.QSat != 3 {
		t.Errorf("Q class counts: main=%d sat=%d, want 2 and 3", p.QMain, p.QSat)
	}
	t.Logf("%s", p)
}

// TestPartitionIndependentOfWorkerCount requires the classification to be identical
// whatever GOMAXPROCS is. It is parallel over row chunks with per-worker lists
// concatenated afterwards, so a concatenation in the wrong order would reorder Q and P
// and silently permute the restricted spaces.
func TestPartitionIndependentOfWorkerCount(t *testing.T) {
	holes := make([][]int, 4000)
	for i := range holes {
		holes[i] = []int{i % 7, (i / 7) % 5, (i / 35) % 3}
	}
	sp := &fakeSpace{main: 11, holes: holes}
	sel, err := NewHoleLocalization(AnyHole, []int{0, 3}, "")
	if err != nil {
		t.Fatal(err)
	}

	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	var ref *Partition
	for _, procs := range []int{1, 2, 3, 8, 17} {
		runtime.GOMAXPROCS(procs)
		p := NewPartition(sp, sel)
		if ref == nil {
			ref = p
			continue
		}
		if !slices.Equal(p.Q, ref.Q) || !slices.Equal(p.P, ref.P) {
			t.Fatalf("GOMAXPROCS=%d produced a different partition (|Q|=%d vs %d)",
				procs, len(p.Q), len(ref.Q))
		}
	}
	t.Logf("partition of %d rows is identical across GOMAXPROCS 1..17: |Q|=%d |P|=%d",
		sp.Size(), len(ref.Q), len(ref.P))
}

// TestEmbedGatherRoundTrip checks the scatter/gather pair that stands in for the
// Feshbach projectors, including the property the coupling relies on: an embedded Q
// vector is identically zero on every P row.
func TestEmbedGatherRoundTrip(t *testing.T) {
	sp := &fakeSpace{main: 2, holes: [][]int{{0}, {1}, {0, 0}, {0, 1}, {1, 1}, {0, 0, 1}}}
	sel, err := NewHoleLocalization(AllHoles, []int{0}, "")
	if err != nil {
		t.Fatal(err)
	}
	p := NewPartition(sp, sel)

	q := make([]float64, p.QSize())
	for i := range q {
		q[i] = float64(i) + 1.5
	}
	full := p.EmbedQ(q, nil)
	if len(full) != sp.Size() {
		t.Fatalf("EmbedQ gave %d rows, want %d", len(full), sp.Size())
	}
	for _, r := range p.P {
		if full[r] != 0 {
			t.Errorf("embedded Q vector is nonzero on P row %d (%g); QP must vanish", r, full[r])
		}
	}
	if back := p.GatherQ(full, nil); !slices.Equal(back, q) {
		t.Errorf("GatherQ(EmbedQ(q)) = %v, want %v", back, q)
	}
	if pv := p.GatherP(full, nil); len(pv) != p.PSize() {
		t.Fatalf("GatherP gave %d rows, want %d", len(pv), p.PSize())
	}

	// Buffer reuse must not leak the previous contents.
	buf := make([]float64, sp.Size())
	for i := range buf {
		buf[i] = math.Pi
	}
	full = p.EmbedQ(q, buf)
	for _, r := range p.P {
		if full[r] != 0 {
			t.Errorf("EmbedQ into a reused buffer left %g on P row %d", full[r], r)
		}
	}
}

// TestValidateRejectsDegeneratePartitions requires the failure modes that otherwise
// surface as a silently zero or missing width to be reported as errors.
func TestValidateRejectsDegeneratePartitions(t *testing.T) {
	sel := func(rule HoleRule, orbs ...int) Selector {
		s, err := NewHoleLocalization(rule, orbs, "")
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	sp := &fakeSpace{main: 1, holes: [][]int{{0}, {0, 1}, {1, 1}}}

	// Nothing in Q: the Q orbital set shares no hole with any configuration.
	if err := NewPartition(sp, sel(AnyHole, 9)).Validate(); err == nil {
		t.Error("an empty Q was accepted")
	}
	// Everything in Q: no decay channels left.
	if err := NewPartition(sp, sel(AnyHole, 0, 1)).Validate(); err == nil {
		t.Error("an empty P was accepted")
	}
	// Q with satellites but no main-class row: no discrete state to select.
	noMain := &fakeSpace{main: 1, holes: [][]int{{5}, {0, 1}, {1, 1}}}
	if err := NewPartition(noMain, sel(AllHoles, 0, 1)).Validate(); err == nil {
		t.Error("a Q without a main-class configuration was accepted")
	}
}

// partition_test.go — the per-class Q/P rules of the supplementary material to
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
