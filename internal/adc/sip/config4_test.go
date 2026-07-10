package sip

import (
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// choose2 is C(n,2).
func choose2(n int) int { return n * (n - 1) / 2 }

// dim3h2pOff is an independent oracle for the symmetry-off CVS 3h2p count: one
// core hole K (from ncore core orbitals), a valence-hole pair L<=M from nvalocc
// valence occupieds, a particle pair a<=b from nvir virtuals, with spin
// multiplicity MAXS(L==M, a==b) = {1,2,2,5} (state.F rules).
func dim3h2pOff(ncore, nvalocc, nvir int) int {
	holesEq := nvalocc          // L == M
	holesNe := choose2(nvalocc) // L <  M
	virsEq := nvir              // a == b
	virsNe := choose2(nvir)     // a <  b
	per := holesEq*virsEq*1 + holesEq*virsNe*2 + holesNe*virsEq*2 + holesNe*virsNe*5
	return ncore * per
}

func TestSpace4CountsSymmetryOff(t *testing.T) {
	core := []int{0} // O 1s
	s := NewSpace4(testNocc, testNorb, nil, 0, core)

	nvalocc := testNocc - len(core)

	// 1h main = core orbitals of the target irrep.
	if s.BeginSat != len(core) {
		t.Errorf("main (1h core) block = %d, want %d", s.BeginSat, len(core))
	}
	// 2h1p = ncore * nvalocc * nvir * 2 spin functions.
	want2h1p := len(core) * nvalocc * testNvir * 2
	if got := s.Begin3h2p - s.BeginSat; got != want2h1p {
		t.Errorf("2h1p dim = %d, want %d", got, want2h1p)
	}
	// 3h2p matches the independent oracle.
	want3h2p := dim3h2pOff(len(core), nvalocc, testNvir)
	if got := len(s.Sat3); got != want3h2p {
		t.Errorf("3h2p dim = %d, want %d", got, want3h2p)
	}
	if s.Begin3h2p != len(s.Configs) {
		t.Errorf("Begin3h2p=%d, want len(Configs)=%d", s.Begin3h2p, len(s.Configs))
	}
	if s.Size() != len(s.Configs)+want3h2p {
		t.Errorf("Size=%d, want %d", s.Size(), len(s.Configs)+want3h2p)
	}
}

// check3h2pInvariants verifies each 3h2p config: one core hole, ordered pairs,
// valence holes, target symmetry, and spin index in range.
func check3h2pInvariants(t *testing.T, s *Space, target int) {
	t.Helper()
	for i, c := range s.Sat3 {
		k, l, m := c.Core, c.L, c.M
		a, b := s.Nocc+c.I, s.Nocc+c.J
		if !s.isCore(k) {
			t.Fatalf("cfg %d: core hole K=%d not core", i, k)
		}
		if s.isCore(l) || s.isCore(m) {
			t.Fatalf("cfg %d: valence holes L=%d M=%d include a core orbital", i, l, m)
		}
		if p := symProduct(s.irrep(k), s.irrep(l), s.irrep(m), s.irrep(a), s.irrep(b)); p != target {
			t.Fatalf("cfg %d: symmetry product %d != target %d", i, p, target)
		}
		ms := maxS3(l == m, c.I == c.J)
		if c.Spin < 1 || c.Spin > ms {
			t.Fatalf("cfg %d: spin %d out of range 1..%d", i, c.Spin, ms)
		}
	}
}

func TestSpace4PerIrrepUnion(t *testing.T) {
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	if d.OrbSym == nil {
		t.Fatal("testdata FCIDUMP has no ORBSYM; regenerate with symmetry")
	}
	nocc := mp.NOcc(d)
	nSym := numIrreps(d.OrbSym, d.NORB)
	core := []int{0} // O 1s (a1)

	var union int
	for sym := range nSym {
		s := NewSpace4(nocc, d.NORB, d.OrbSym, sym, core)
		check3h2pInvariants(t, s, sym)
		// Group3 boundaries: start at Begin3h2p, strictly increasing.
		if len(s.Group3) > 0 && s.Group3[0] != s.Begin3h2p {
			t.Errorf("sym %d: Group3[0]=%d, want Begin3h2p=%d", sym, s.Group3[0], s.Begin3h2p)
		}
		for g := 1; g < len(s.Group3); g++ {
			if s.Group3[g] <= s.Group3[g-1] {
				t.Errorf("sym %d: Group3 not increasing at %d", sym, g)
			}
		}
		union += len(s.Sat3)
	}

	// Union over irreps reproduces the symmetry-off 3h2p count.
	off := NewSpace4(nocc, d.NORB, nil, 0, core)
	if union != len(off.Sat3) {
		t.Errorf("union of 3h2p dims = %d, want symmetry-off %d", union, len(off.Sat3))
	}
}
