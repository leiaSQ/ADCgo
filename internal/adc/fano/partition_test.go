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
