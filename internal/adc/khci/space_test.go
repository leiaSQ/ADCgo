package khci

import (
	"math/bits"
	"math/rand/v2"
	"runtime"
	"slices"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/fano"
)

var (
	_ fano.Space         = (*Space)(nil)
	_ fano.ParticleSpace = (*Space)(nil)
)

// config is an independent representation for the brute-force oracle: sorted hole and
// particle spin-orbital lists.
type config struct {
	holes, parts []int
}

// bruteForce enumerates the space with nested loops and explicit filters, in the
// documented row order (class, sorted hole list lexicographic, particle tuple
// lexicographic). It shares no code with NewSpace.
func bruteForce(o Options) []config {
	if o.MaxClass == 0 {
		o.MaxClass = o.K + 2
	}
	nso, nv2 := 2*o.NOcc, 2*o.NVir
	irr := func(orb int) int {
		if o.OrbSym == nil {
			return 0
		}
		return o.OrbSym[orb]
	}
	ms := func(p int) int {
		if p%2 == 0 {
			return 1
		}
		return -1
	}
	var out []config
	var combos func(n, from int, cur []int, emit func([]int))
	combos = func(n, from int, cur []int, emit func([]int)) {
		if len(cur) == n {
			emit(slices.Clone(cur))
			return
		}
		for p := from; p < nso; p++ {
			combos(n, p+1, append(cur, p), emit)
		}
	}
	for c := o.K; c <= o.MaxClass; c++ {
		m := c - o.K
		combos(c, 0, nil, func(h []int) {
			var partsets [][]int
			switch m {
			case 0:
				partsets = [][]int{{}}
			case 1:
				for a := range nv2 {
					partsets = append(partsets, []int{a})
				}
			case 2:
				for a := range nv2 {
					for b := a + 1; b < nv2; b++ {
						partsets = append(partsets, []int{a, b})
					}
				}
			}
			for _, p := range partsets {
				spin, sym := 0, 0
				for _, x := range h {
					spin -= ms(x)
					sym ^= irr(x / 2)
				}
				nfree := 0
				for _, a := range p {
					spin += ms(a)
					sym ^= irr(o.NOcc + a/2)
					if o.VirKind != nil && o.VirKind[a/2] == Free {
						nfree++
					}
				}
				if !o.AllMs && spin != o.TwoMs {
					continue
				}
				if sym != o.TargetIrrep {
					continue
				}
				if o.LimitFree && m == 2 && nfree > o.MaxFreeTop {
					continue
				}
				out = append(out, config{h, p})
			}
		})
	}
	return out
}

func rowConfig(s *Space, r int) config {
	var h []int
	for q := s.HoleMask(r); q != 0; q &= q - 1 {
		h = append(h, bits.TrailingZeros64(q))
	}
	return config{h, s.PartSO(r, nil)}
}

func testOptions() []Options {
	sym := func(n int, seed uint64) []int {
		rng := rand.New(rand.NewPCG(seed, 7))
		out := make([]int, n)
		for i := range out {
			out[i] = rng.IntN(2)
		}
		return out
	}
	kinds := []VirKind{Compact, Free, Compact, Free, Free}
	return []Options{
		{K: 1, NOcc: 3, NVir: 3, TwoMs: 1},
		{K: 1, NOcc: 3, NVir: 3, AllMs: true},
		{K: 2, NOcc: 3, NVir: 4, TwoMs: 0},
		{K: 2, NOcc: 3, NVir: 4, TwoMs: 2, OrbSym: sym(7, 1), TargetIrrep: 1},
		{K: 3, NOcc: 4, NVir: 3, TwoMs: 1, OrbSym: sym(7, 2)},
		{K: 4, NOcc: 5, NVir: 5, TwoMs: 0, VirKind: kinds, LimitFree: true, MaxFreeTop: 1},
		{K: 4, NOcc: 5, NVir: 5, TwoMs: 0, VirKind: kinds, LimitFree: true, MaxFreeTop: 0},
		{K: 4, NOcc: 5, NVir: 5, AllMs: true, MaxClass: 5},
		{K: 3, NOcc: 3, NVir: 0, TwoMs: 1},
	}
}

// TestRowsMatchBruteForce: NewSpace produces exactly the brute-force configuration list,
// in the documented order.
func TestRowsMatchBruteForce(t *testing.T) {
	for _, o := range testOptions() {
		s, err := NewSpace(o)
		if err != nil {
			t.Fatalf("%+v: %v", o, err)
		}
		want := bruteForce(o)
		if s.Size() != len(want) {
			t.Fatalf("%v: %d rows, brute force %d", s, s.Size(), len(want))
		}
		for r := range want {
			got := rowConfig(s, r)
			if !slices.Equal(got.holes, want[r].holes) || !slices.Equal(got.parts, want[r].parts) {
				t.Fatalf("%v: row %d is %+v, want %+v", s, r, got, want[r])
			}
		}
	}
}

func binom(n, k int) int {
	if k < 0 || k > n {
		return 0
	}
	r := 1
	for i := range k {
		r = r * (n - i) / (i + 1)
	}
	return r
}

// TestClosedFormCounts: with every filter off, class c has C(2*NOcc, c) *
// C(2*NVir, c-K) rows. With NOcc = 5 the 5h1p count is C(10,5)*2*n_v.
func TestClosedFormCounts(t *testing.T) {
	o := Options{K: 4, NOcc: 5, NVir: 7, AllMs: true}
	s, err := NewSpace(o)
	if err != nil {
		t.Fatal(err)
	}
	for c := 4; c <= 6; c++ {
		got := s.ClassEnd(c) - s.ClassStart(c)
		want := binom(10, c) * binom(14, c-4)
		if got != want {
			t.Errorf("class %d: %d rows, want C(10,%d)*C(14,%d) = %d", c, got, c, c-4, want)
		}
	}
	if n5 := s.ClassEnd(5) - s.ClassStart(5); n5 != binom(10, 5)*2*7 {
		t.Errorf("5h1p: %d rows, want C(10,5)*2n_v = %d", n5, binom(10, 5)*14)
	}
	if s.MainBlockSize() != binom(10, 4) {
		t.Errorf("MainBlockSize %d, want %d", s.MainBlockSize(), binom(10, 4))
	}
	// the at-most-one-free restriction removes exactly the free-free pairs of 6h2p
	kinds := []VirKind{Compact, Free, Free, Compact, Free, Compact, Compact}
	o2 := Options{K: 4, NOcc: 5, NVir: 7, AllMs: true, VirKind: kinds, LimitFree: true, MaxFreeTop: 1}
	s2, err := NewSpace(o2)
	if err != nil {
		t.Fatal(err)
	}
	nfree := 2 * 3 // free virtual spin orbitals
	want6 := binom(10, 6) * (binom(14, 2) - binom(nfree, 2))
	if got := s2.ClassEnd(6) - s2.ClassStart(6); got != want6 {
		t.Errorf("6h2p with <=1 free: %d rows, want %d", got, want6)
	}
}

// TestIndexRoundTrip: Index inverts the row map on every row, and rejects
// configurations that are not in the space.
func TestIndexRoundTrip(t *testing.T) {
	for _, o := range testOptions() {
		s, _ := NewSpace(o)
		for r := range s.Size() {
			got, ok := s.Index(s.HoleMask(r), s.PartSO(r, nil))
			if !ok || got != r {
				t.Fatalf("%v: Index(row %d) = %d, %v", s, r, got, ok)
			}
			// reversed particle order is the same configuration
			p := s.PartSO(r, nil)
			slices.Reverse(p)
			if got, ok := s.Index(s.HoleMask(r), p); !ok || got != r {
				t.Fatalf("%v: Index with reversed particles failed on row %d", s, r)
			}
		}
		// a hole set of the wrong size, and duplicate particles, are never rows
		if _, ok := s.Index(1, nil); ok && o.K != 1 {
			t.Errorf("%v: a one-hole configuration was found in a k=%d space", s, o.K)
		}
		if o.NVir > 0 {
			if _, ok := s.Index(0b111111, []int{0, 0}); ok {
				t.Errorf("%v: duplicate particles accepted", s)
			}
		}
	}
}

// TestHolesParticlesConsistent: Holes and Particles are the spatial projections of
// the spin-orbital lists, of lengths class and class-K.
func TestHolesParticlesConsistent(t *testing.T) {
	for _, o := range testOptions() {
		s, _ := NewSpace(o)
		for r := range s.Size() {
			c := rowConfig(s, r)
			h := s.Holes(r, nil)
			p := s.Particles(r, nil)
			if len(h) != s.Class(r) || len(p) != s.Class(r)-o.K {
				t.Fatalf("%v row %d: %d holes %d particles for class %d", s, r, len(h), len(p), s.Class(r))
			}
			for i, x := range c.holes {
				if h[i] != x/2 {
					t.Fatalf("row %d: Holes %v vs spin orbitals %v", r, h, c.holes)
				}
			}
			for i, a := range c.parts {
				if p[i] != a/2 {
					t.Fatalf("row %d: Particles %v vs spin orbitals %v", r, p, c.parts)
				}
			}
			if r > 0 && s.Class(r) < s.Class(r-1) {
				t.Fatalf("rows not class-major at %d", r)
			}
		}
		for c := o.K; c <= s.Options().MaxClass; c++ {
			for r := s.ClassStart(c); r < s.ClassEnd(c); r++ {
				if s.Class(r) != c {
					t.Fatalf("row %d in class band %d has class %d", r, c, s.Class(r))
				}
			}
		}
	}
}

// TestRestrictRoundTrip: a restricted space keeps the chosen rows in order, maps them
// back to the parent, and indexes them; restricting twice composes.
func TestRestrictRoundTrip(t *testing.T) {
	o := Options{K: 4, NOcc: 5, NVir: 4, TwoMs: 0}
	s, _ := NewSpace(o)
	rng := rand.New(rand.NewPCG(3, 4))
	var keep []int
	for r := range s.Size() {
		if rng.IntN(3) == 0 {
			keep = append(keep, r)
		}
	}
	sub, err := s.Restrict(keep)
	if err != nil {
		t.Fatal(err)
	}
	if sub.Size() != len(keep) {
		t.Fatalf("sub size %d, want %d", sub.Size(), len(keep))
	}
	for i, r := range keep {
		if sub.Parent(i) != r || sub.HoleMask(i) != s.HoleMask(r) ||
			!slices.Equal(sub.PartSO(i, nil), s.PartSO(r, nil)) {
			t.Fatalf("sub row %d does not match parent row %d", i, r)
		}
		if j, ok := sub.Index(s.HoleMask(r), s.PartSO(r, nil)); !ok || j != i {
			t.Fatalf("sub.Index(parent row %d) = %d, %v; want %d", r, j, ok, i)
		}
	}
	mainWant := 0
	for _, r := range keep {
		if r < s.MainBlockSize() {
			mainWant++
		}
	}
	if sub.MainBlockSize() != mainWant {
		t.Errorf("sub MainBlockSize %d, want %d", sub.MainBlockSize(), mainWant)
	}
	// a dropped parent row is not found
	for r := range s.Size() {
		if !slices.Contains(keep, r) {
			if _, ok := sub.Index(s.HoleMask(r), s.PartSO(r, nil)); ok {
				t.Fatalf("dropped parent row %d found in the sub-space", r)
			}
			break
		}
	}
	// restrict again: Parent still maps to the unrestricted space
	sub2, err := sub.Restrict([]int{0, sub.Size() - 1})
	if err != nil {
		t.Fatal(err)
	}
	if sub2.Parent(0) != keep[0] || sub2.Parent(1) != keep[len(keep)-1] {
		t.Errorf("nested Restrict parents %d,%d; want %d,%d", sub2.Parent(0), sub2.Parent(1),
			keep[0], keep[len(keep)-1])
	}
	if _, err := s.Restrict([]int{3, 3}); err == nil {
		t.Error("Restrict accepted duplicate rows")
	}
}

// bruteDet applies the operator string c†_{a1} c†_{a2} c_{i1} ... c_{in} to Phi_0 by
// explicit list manipulation (the operator acting rightmost first), returning the final
// ordered occupation list and the sign relative to its sorted order. Independent of
// Space.Det, which counts bits.
func bruteDet(nso int, holes, parts []int) ([]int, float64) {
	occ := make([]int, nso) // ordered list of created orbitals, as c†_{occ[0]} c†_{occ[1]}...
	for i := range occ {
		occ[i] = i
	}
	sign := 1.0
	for j := len(holes) - 1; j >= 0; j-- {
		pos := slices.Index(occ, holes[j])
		// move c_p through the pos creators to its left
		if pos%2 == 1 {
			sign = -sign
		}
		occ = slices.Delete(occ, pos, pos+1)
	}
	for j := len(parts) - 1; j >= 0; j-- {
		// c†_x prepended to the string, then bubble-sorted into place
		occ = append([]int{parts[j]}, occ...)
	}
	// sort the creator string by adjacent swaps, flipping the sign per swap
	for i := range occ {
		for j := len(occ) - 1; j > i; j-- {
			if occ[j-1] > occ[j] {
				occ[j-1], occ[j] = occ[j], occ[j-1]
				sign = -sign
			}
		}
	}
	return occ, sign
}

// TestDetSign: Det agrees with explicit operator application on every row.
func TestDetSign(t *testing.T) {
	for _, o := range testOptions() {
		s, _ := NewSpace(o)
		nso := 2 * o.NOcc
		for r := range s.Size() {
			occ, parts, sign := s.Det(r)
			c := rowConfig(s, r)
			abs := make([]int, len(c.parts))
			for i, a := range c.parts {
				abs[i] = nso + a
			}
			wantOcc, wantSign := bruteDet(nso, c.holes, abs)
			var got []int
			for q := occ; q != 0; q &= q - 1 {
				got = append(got, bits.TrailingZeros64(q))
			}
			got = append(got, parts...)
			if !slices.Equal(got, wantOcc) || sign != wantSign {
				t.Fatalf("%v row %d (%+v): Det %v sign %v, brute %v sign %v",
					s, r, c, got, sign, wantOcc, wantSign)
			}
		}
	}
}

// TestDeterministic: the enumeration is independent of the worker count.
func TestDeterministic(t *testing.T) {
	o := Options{K: 4, NOcc: 5, NVir: 6, TwoMs: 0}
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	a, _ := NewSpace(o)
	runtime.GOMAXPROCS(8)
	b, _ := NewSpace(o)
	if a.Size() != b.Size() {
		t.Fatalf("sizes differ: %d vs %d", a.Size(), b.Size())
	}
	for r := range a.Size() {
		if a.rows[r] != b.rows[r] {
			t.Fatalf("row %d differs between GOMAXPROCS 1 and 8", r)
		}
	}
}

// TestOptionValidation: malformed options are rejected rather than silently shrunk.
func TestOptionValidation(t *testing.T) {
	bad := []Options{
		{K: 0, NOcc: 2, NVir: 2},
		{K: 5, NOcc: 5, NVir: 2},
		{K: 2, NOcc: 33, NVir: 2},
		{K: 2, NOcc: 2, NVir: 2, TwoMs: 1},            // parity
		{K: 2, NOcc: 2, NVir: 2, MaxClass: 5},         // above K+2
		{K: 2, NOcc: 2, NVir: 2, OrbSym: []int{0, 1}}, // wrong length
		{K: 2, NOcc: 2, NVir: 2, LimitFree: true},     // no VirKind
		{K: 4, NOcc: 2, NVir: 2, AllMs: true},         // 6 holes > 4 spin orbitals
	}
	for _, o := range bad {
		if _, err := NewSpace(o); err == nil {
			t.Errorf("options %+v accepted", o)
		}
	}
}
