package sip

import (
	"fmt"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// h2o22 is the ADC(2,2) fixture: the H2O FCIDUMP and its occupied count. It reuses
// the package's existing h2oData reader.
func h2o22(t *testing.T) (*fcidump.Data, int) {
	t.Helper()
	d := h2oData(t)
	return d, mp.NOcc(d)
}

// TestSpin3MatchesMaxS3 checks the general open-shell spin count against maxS3 on
// exactly the cases the CVS space can produce — where the core hole K is always
// distinct from the valence pair, so only L == M can coincide. Anywhere both
// agree they must agree exactly; the general count additionally covers K == L and
// K == M, which CVS never reaches.
func TestSpin3MatchesMaxS3(t *testing.T) {
	const k, l, m = 0, 1, 2
	for _, c := range []struct{ lEqM, iEqJ bool }{
		{false, false}, {false, true}, {true, false}, {true, true},
	} {
		ll, mm := l, m
		if c.lEqM {
			mm = ll
		}
		i, j := 0, 1
		if c.iEqJ {
			j = i
		}
		got, want := nSpin3(k, ll, mm, i, j), maxS3(c.lEqM, c.iEqJ)
		if got != want {
			t.Errorf("lEqM=%v iEqJ=%v: nSpin3=%d maxS3=%d", c.lEqM, c.iEqJ, got, want)
		}
	}
	// A coincident pair involving the first hole must give the same count as one
	// involving the last two — the label carries no physics.
	if a, b := nSpin3(0, 0, 2, 0, 1), nSpin3(2, 0, 0, 0, 1); a != b {
		t.Errorf("K==L gives %d spin functions but L==M gives %d", a, b)
	}
}

// TestSpace22NoDuplicates requires every 3h2p configuration to be enumerated
// exactly once: one entry per (unordered hole triple, unordered particle pair,
// spin function). A duplicate would silently double-count that configuration in
// every matrix element.
func TestSpace22NoDuplicates(t *testing.T) {
	d, nocc := h2o22(t)
	sp := NewSpace22(nocc, d.NORB, d.OrbSym, 0)
	seen := map[string]bool{}
	for _, c := range sp.Sat3 {
		h := []int{c.Core, c.L, c.M}
		sortInts3(h)
		p := []int{c.I, c.J}
		if p[0] > p[1] {
			p[0], p[1] = p[1], p[0]
		}
		key := fmt.Sprintf("%v|%v|%d", h, p, c.Spin)
		if seen[key] {
			t.Fatalf("3h2p configuration enumerated twice: %s", key)
		}
		seen[key] = true
	}
	t.Logf("H2O sector 0: 1h=%d 2h1p=%d 3h2p=%d (total %d)",
		sp.BeginSat, sp.Begin3h2p-sp.BeginSat, len(sp.Sat3), sp.Size())
	if len(sp.Sat3) == 0 {
		t.Fatal("no 3h2p configurations enumerated")
	}
}

func sortInts3(a []int) {
	if a[0] > a[1] {
		a[0], a[1] = a[1], a[0]
	}
	if a[1] > a[2] {
		a[1], a[2] = a[2], a[1]
	}
	if a[0] > a[1] {
		a[0], a[1] = a[1], a[0]
	}
}

// TestSpace22Symmetry requires every enumerated configuration to carry the target
// irrep, and no three-electron-from-one-orbital configuration to survive.
func TestSpace22Symmetry(t *testing.T) {
	d, nocc := h2o22(t)
	for sym := range 4 {
		sp := NewSpace22(nocc, d.NORB, d.OrbSym, sym)
		for _, c := range sp.Sat3 {
			if c.Core == c.L && c.L == c.M {
				t.Fatalf("sym %d: three holes in one orbital %d", sym, c.Core)
			}
			p := symProduct(sp.irrep(c.Core), sp.irrep(c.L), sp.irrep(c.M),
				sp.irrep(nocc+c.I), sp.irrep(nocc+c.J))
			if p != sym {
				t.Fatalf("sym %d: config %+v has irrep product %d", sym, c, p)
			}
		}
	}
}

// TestSpace22UnionMatchesC1 checks that the per-irrep sectors partition the same
// space the symmetry-off run enumerates — the standard per-irrep union gate used
// across this package. Symmetry blocking must lose no configuration and invent
// none.
func TestSpace22UnionMatchesC1(t *testing.T) {
	d, nocc := h2o22(t)
	c1 := NewSpace22(nocc, d.NORB, nil, 0)
	nIrreps := numIrreps(d.OrbSym, d.NORB)
	total := 0
	for sym := range nIrreps {
		total += len(NewSpace22(nocc, d.NORB, d.OrbSym, sym).Sat3)
	}
	if total != len(c1.Sat3) {
		t.Errorf("per-irrep 3h2p total %d != symmetry-off total %d", total, len(c1.Sat3))
	}
	t.Logf("3h2p configurations: %d, identical across %d irrep sectors and the C1 run",
		total, nIrreps)
}

// buildH2OSpace22 is the ADC(2,2) matrix engine for one H2O sector.
func buildH2OSpace22(t *testing.T, sym int) *Matrix {
	t.Helper()
	d, nocc := h2o22(t)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace22(nocc, d.NORB, nil, sym)
	return New(sp, integrals.New(d, nocc, nil), eps, 22, backend.Gonum{})
}
