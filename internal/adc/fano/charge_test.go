package fano

import (
	"math"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
)

// Five-atom toy: atoms A1..A5 (0..4) each own one occupied orbital (spatial i = atom)
// and four compact virtuals (spatial 4a..4a+3); virtuals 20..25 are free. A sixth,
// ghost, atom G owns nothing.
const (
	toyNOcc    = 5
	toyCompact = 20
	toyNVir    = 26
)

func toyAtomMap() *AtomMap {
	am := &AtomMap{
		Names:   []string{"A1", "A2", "A3", "A4", "A5", "G"},
		Ghost:   []bool{false, false, false, false, false, true},
		OccAtom: []int{0, 1, 2, 3, 4},
	}
	for v := range toyNVir {
		if v < toyCompact {
			am.VirAtom = append(am.VirAtom, v/4)
		} else {
			am.VirAtom = append(am.VirAtom, -1)
		}
	}
	return am
}

func toySpace(t *testing.T) *khci.Space {
	t.Helper()
	kinds := make([]khci.VirKind, toyNVir)
	for v := toyCompact; v < toyNVir; v++ {
		kinds[v] = khci.Free
	}
	sp, err := khci.NewSpace(khci.Options{K: 4, NOcc: toyNOcc, NVir: toyNVir, TwoMs: 0,
		VirKind: kinds, LimitFree: true, MaxFreeTop: 1})
	if err != nil {
		t.Fatal(err)
	}
	return sp
}

// so(spatial, beta) is an occupied spin orbital; vso a relative virtual one.
func so(i int, beta bool) int {
	if beta {
		return 2*i + 1
	}
	return 2 * i
}

func mask(ps ...int) uint64 {
	var m uint64
	for _, p := range ps {
		m |= 1 << uint(p)
	}
	return m
}

const oneEachRule = "p: charge A1=1,A2=1,A5=1,A4=1,A3=1 & free=1"

// TestNetChargePartition: hand-built configurations of a four-hole initial state on the
// five-atom toy (open channel: every atom singly charged, one free electron) land where
// the physics puts them.
func TestNetChargePartition(t *testing.T) {
	sp := toySpace(t)
	am := toyAtomMap()
	rule, err := ParseConfigRule(oneEachRule, "one charge per atom", am)
	if err != nil {
		t.Fatal(err)
	}
	part := NewPartition(sp, rule)
	if err := part.Validate(); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s", part)
	t.Logf("census: %s", part.CensusString())

	const a1, a2, a3, a4, a5 = 0, 1, 2, 3, 4
	compactOn := func(atom, k int, beta bool) int { // relative virtual spin orbital
		v := 4*atom + k
		if beta {
			return 2*v + 1
		}
		return 2 * v
	}
	free := func(k int, beta bool) int {
		v := toyCompact + k
		if beta {
			return 2*v + 1
		}
		return 2 * v
	}
	cases := []struct {
		name  string
		holes uint64
		parts []int
		inQ   bool
	}{
		{"main 4h A1^2 A5^2", mask(so(a1, false), so(a1, true), so(a5, false), so(a5, true)), nil, true},
		{"relaxed-ion 6h2p (A3+ relaxed, one free electron)",
			mask(so(a1, false), so(a2, true), so(a3, false), so(a3, true), so(a4, false), so(a5, true)),
			[]int{compactOn(a3, 0, false), free(0, true)}, false},
		{"excited intermediate 5h1p(A1,A2,A5,A5,A3; compact on A3)",
			mask(so(a1, false), so(a2, true), so(a5, false), so(a5, true), so(a3, false)),
			[]int{compactOn(a3, 1, false)}, true},
		{"one hole per atom + free electron, 5h1p",
			mask(so(a1, false), so(a2, true), so(a3, false), so(a4, true), so(a5, false)),
			[]int{free(2, false)}, false},
		{"doubly charged look-alike: same holes as the relaxed ion, compact particle on A1",
			mask(so(a1, false), so(a2, true), so(a3, false), so(a3, true), so(a4, false), so(a5, true)),
			[]int{compactOn(a1, 0, false), free(0, true)}, true},
		{"one hole per atom, particle compact on A3 (bound excitation)",
			mask(so(a1, false), so(a2, true), so(a3, false), so(a4, true), so(a5, false)),
			[]int{compactOn(a3, 2, false)}, true},
	}
	for _, c := range cases {
		r, ok := sp.Index(c.holes, c.parts)
		if !ok {
			t.Fatalf("%s: configuration not in the space", c.name)
		}
		gotQ := part.QIndex(r) >= 0
		if gotQ != c.inQ {
			t.Errorf("%s: in Q = %v, want %v", c.name, gotQ, c.inQ)
		}
	}
	// no main-class configuration can be in P: four holes cannot give five +1 atoms
	for r := range sp.MainBlockSize() {
		if part.PIndex(r) >= 0 {
			t.Fatalf("main-class row %d landed in P", r)
		}
	}
	cen := part.Census()
	if cen[4][1] != 0 || cen[5][1] == 0 || cen[6][1] == 0 {
		t.Errorf("unexpected census %v: P must hold 5h1p and 6h2p rows and no 4h ones", cen)
	}
}

// TestConfigRuleWildcard: "charge *=1" names every real atom and never the ghost.
func TestConfigRuleWildcard(t *testing.T) {
	am := toyAtomMap()
	a, err := ParseConfigRule("p: charge *=1 & free=1", "", am)
	if err != nil {
		t.Fatal(err)
	}
	b, err := ParseConfigRule(oneEachRule, "", am)
	if err != nil {
		t.Fatal(err)
	}
	sp := toySpace(t)
	pa, pb := NewPartition(sp, a), NewPartition(sp, b)
	if pa.QSize() != pb.QSize() {
		t.Fatalf("wildcard and explicit rules differ: Q %d vs %d", pa.QSize(), pb.QSize())
	}
	if strings.Contains(a.String(), "G=") {
		t.Errorf("wildcard expanded to the ghost: %s", a)
	}
}

// TestConfigRuleErrors: ghosts, unknown atoms and malformed terms are rejected, and
// the hole-only parser points charge terms to ParseConfigRule.
func TestConfigRuleErrors(t *testing.T) {
	am := toyAtomMap()
	for _, spec := range []string{
		"p: charge G=1",       // ghost
		"p: charge XX=1",      // unknown
		"p: charge A1=1,A1=1", // duplicate
		"p: charge A1",        // no value
		"p: free=2:1",         // max < min
		"p: free 1",           // syntax
		"x: charge A1=1",      // bad prefix
		"",                    // empty
	} {
		if _, err := ParseConfigRule(spec, "", am); err == nil {
			t.Errorf("spec %q accepted", spec)
		}
	}
	if _, err := ParseClassRule(oneEachRule, ""); err == nil ||
		!strings.Contains(err.Error(), "ParseConfigRule") {
		t.Errorf("ParseClassRule on a charge rule: err = %v, want a pointer to ParseConfigRule", err)
	}
	// a hole-only space cannot take a ConfigSelector
	rule, _ := ParseConfigRule(oneEachRule, "", am)
	defer func() {
		if recover() == nil {
			t.Error("NewPartition accepted a ConfigSelector on a hole-only space")
		}
	}()
	NewPartition(holeOnly{toySpace(t)}, rule)
}

type holeOnly struct{ s *khci.Space }

func (h holeOnly) Size() int                      { return h.s.Size() }
func (h holeOnly) MainBlockSize() int             { return h.s.MainBlockSize() }
func (h holeOnly) Holes(row int, dst []int) []int { return h.s.Holes(row, dst) }

// TestAtomMapFromMO: the map built from a labelled sidecar, and its guards.
func TestAtomMapFromMO(t *testing.T) {
	md := &mo.Data{
		NMO:       4,
		AtomNames: []string{"He1", "He2", "X1"},
		HasLabels: true,
		OrbKind:   []mo.OrbKind{mo.OrbOcc, mo.OrbOcc, mo.OrbCompact, mo.OrbFree},
		OrbAtom:   []int{0, 1, 1, -1},
		GhostAtom: []bool{false, false, true},
	}
	am, err := AtomMapFromMO(md, 2)
	if err != nil {
		t.Fatal(err)
	}
	if am.OccAtom[1] != 1 || am.VirAtom[0] != 1 || am.VirAtom[1] != -1 {
		t.Errorf("bad map %+v", am)
	}
	if _, err := AtomMapFromMO(md, 3); err == nil {
		t.Error("occupied-count mismatch accepted")
	}
	md.HasLabels = false
	if _, err := AtomMapFromMO(md, 2); err == nil {
		t.Error("unlabelled sidecar accepted")
	}
}

// TestAtomMapFromLabelledDump: the atom map of a real localized-orbital dump
// (testdata/khci/he3_ghost), and its ghost centres refused in charge rules.
func TestAtomMapFromLabelledDump(t *testing.T) {
	md, err := mo.ReadFile("../../../testdata/khci/he3_ghost.mo.json")
	if err != nil {
		t.Fatal(err)
	}
	am, err := AtomMapFromMO(md, 3)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseConfigRule("p: charge He1=1,He2=1,He3=1 & free=1", "", am); err != nil {
		t.Errorf("real-atom rule rejected: %v", err)
	}
	for _, g := range []string{"GT", "GOFF"} {
		if _, err := ParseConfigRule("p: charge "+g+"=1", "", am); err == nil {
			t.Errorf("ghost %s accepted in a charge rule", g)
		}
	}
}

// TestChannelsFoldGhosts: with a labelled sidecar that declares ghost
// centres (testdata/khci/he3_ghost), channel routing takes each occupied orbital's atom
// from the labels (IAO-assigned by the dump), so every orbital sits wholly on its own atom, no
// population lands on a ghost, each row sums to 1, and a ghost can be neither a site
// member nor the initial site.
func TestChannelsFoldGhosts(t *testing.T) {
	md, err := mo.ReadFile("../../../testdata/khci/he3_ghost.mo.json")
	if err != nil {
		t.Fatal(err)
	}
	ch, err := NewChannels(md, 3, nil, "He1", spectrum.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for i, q := range ch.pop {
		var sum float64
		for a, v := range q {
			sum += v
			if md.GhostAtom[a] && v != 0 {
				t.Errorf("orbital %d: %.3g of its population on ghost %s", i, v, md.AtomNames[a])
			}
		}
		own := md.OrbAtom[i]
		t.Logf("orbital %d: %.8f on %s (row sum %.15f)", i, q[own], md.AtomNames[own], sum)
		if math.Abs(sum-1) > 1e-12 || q[own] < 0.999 {
			t.Errorf("orbital %d: %.6f on its atom %s, row sum %.15f", i, q[own], md.AtomNames[own], sum)
		}
	}
	if _, err := NewChannels(md, 3, []spectrum.Site{{Name: "X", Members: []string{"He1", "GT"}}}, "He1",
		spectrum.Options{}); err == nil {
		t.Error("a site naming a ghost centre was accepted")
	}
	if _, err := NewChannels(md, 3, nil, "GOFF", spectrum.Options{}); err == nil {
		t.Error("a ghost centre was accepted as the initial site")
	}
}
