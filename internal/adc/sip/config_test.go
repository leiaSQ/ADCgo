package sip

import (
	"math"
	"path/filepath"
	"sort"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// H2O/cc-pVDZ (symmetry off): nocc=5, norb=24, nvir=19.
const (
	testNocc = 5
	testNorb = 24
	testNvir = 19
)

// dim2h1p reimplements ndadc3_ip/calc_dim_2h1p.c as an independent oracle: the
// number of 2h1p configurations for the given target symmetry. nOcc/nVir are
// per-0-based-irrep counts; nSym is the group order.
func dim2h1p(nOcc, nVir []int, nSym, target int) int {
	d := 0
	for si := range nSym {
		for sj := range si { // sj < si
			a := si ^ sj ^ target
			d += 2 * nOcc[si] * nOcc[sj] * nVir[a]
		}
	}
	for si := range nSym {
		d += 2*nOcc[si]*(nOcc[si]-1)/2*nVir[target] + nOcc[si]*nVir[target]
	}
	return d
}

func TestConfigCountsSymmetryOff(t *testing.T) {
	s := NewSpace(testNocc, testNorb, nil, 0)

	if s.BeginSat != testNocc {
		t.Errorf("main (1h) block = %d, want %d", s.BeginSat, testNocc)
	}
	// 2h1p = 2*C(nocc,2)*nvir + nocc*nvir.
	wantSat := 2*testNocc*(testNocc-1)/2*testNvir + testNocc*testNvir
	if got := s.Size() - s.BeginSat; got != wantSat {
		t.Errorf("2h1p satellite dim = %d, want %d", got, wantSat)
	}
	oracle := dim2h1p([]int{testNocc}, []int{testNvir}, 1, 0)
	if got := s.Size() - s.BeginSat; got != oracle {
		t.Errorf("2h1p dim = %d, oracle %d", got, oracle)
	}
}

// perSymCounts returns per-0-based-irrep occupied and virtual counts.
func perSymCounts(nocc, norb int, orbSym []int, nSym int) (nOcc, nVir []int) {
	nOcc = make([]int, nSym)
	nVir = make([]int, nSym)
	for o := range norb {
		g := orbSym[o] - 1
		if o < nocc {
			nOcc[g]++
		} else {
			nVir[g]++
		}
	}
	return
}

func TestConfigCountsPerIrrep(t *testing.T) {
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
	nOcc, nVir := perSymCounts(nocc, d.NORB, d.OrbSym, nSym)

	var unionMain, unionSat int
	for sym := range nSym {
		s := NewSpace(nocc, d.NORB, d.OrbSym, sym)

		// Main dim = occupied orbitals of this irrep.
		if s.BeginSat != nOcc[sym] {
			t.Errorf("sym %d: main dim %d, want %d", sym, s.BeginSat, nOcc[sym])
		}
		// Satellite dim matches the independent calc_dim_2h1p oracle.
		if got, want := s.Size()-s.BeginSat, dim2h1p(nOcc, nVir, nSym, sym); got != want {
			t.Errorf("sym %d: 2h1p dim %d, oracle %d", sym, got, want)
		}
		// Every 2h1p config obeys sym(k)⊗sym(l)⊗sym(a) == target.
		for idx := s.BeginSat; idx < s.Size(); idx++ {
			c := s.Configs[idx]
			k, l, a := c.Occ[0], c.Occ[1], nocc+c.Vir
			if p := symProduct(s.irrep(k), s.irrep(l), s.irrep(a)); p != sym {
				t.Fatalf("sym %d cfg %d: <%d,%d,%d> product %d != target", sym, idx, k, l, a, p)
			}
		}
		// Group boundaries: strictly increasing, span the satellite region.
		if s.Group[0] != s.BeginSat {
			t.Errorf("sym %d: Group[0]=%d, want BeginSat=%d", sym, s.Group[0], s.BeginSat)
		}
		for g := 1; g < len(s.Group); g++ {
			if s.Group[g] <= s.Group[g-1] {
				t.Errorf("sym %d: Group not increasing at %d (%d<=%d)", sym, g, s.Group[g], s.Group[g-1])
			}
		}

		unionMain += s.BeginSat
		unionSat += s.Size() - s.BeginSat
	}

	// Union over irreps reproduces the symmetry-off dimensions.
	if unionMain != nocc {
		t.Errorf("union of main dims = %d, want nocc=%d", unionMain, nocc)
	}
	off := NewSpace(nocc, d.NORB, nil, 0)
	if unionSat != off.Size()-off.BeginSat {
		t.Errorf("union of 2h1p dims = %d, want symmetry-off %d", unionSat, off.Size()-off.BeginSat)
	}
}

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

func symSetup(t *testing.T) (*fcidump.Data, int, []float64) {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	if d.OrbSym == nil {
		t.Fatal("testdata FCIDUMP has no ORBSYM; regenerate with symmetry")
	}
	nocc := mp.NOcc(d)
	return d, nocc, mp.OrbitalEnergies(d, nocc)
}

// TestSymmetryPartitionsSpace: the per-irrep sectors tile the full symmetry-off
// configuration space (no configuration lost or double-counted).
func TestSymmetryPartitionsSpace(t *testing.T) {
	d, nocc, _ := symSetup(t)
	nsym := integrals.New(d, nocc, d.OrbSym).NSym()
	full := NewSpace(nocc, d.NORB, nil, 0).Size()
	sum := 0
	for sym := range nsym {
		sum += NewSpace(nocc, d.NORB, d.OrbSym, sym).Size()
	}
	if sum != full {
		t.Errorf("per-irrep sizes sum to %d, want full size %d", sum, full)
	}
}

// TestSymmetryBlockingSpectrum is the M2 gate for SIP: because the H2O integrals
// carry C2v symmetry, the symmetry-off IP matrix is exactly block-diagonal by
// irrep, so the union of the per-irrep spectra reproduces the full symmetry-off
// spectrum eigenvalue-for-eigenvalue (both ADC orders).
func TestSymmetryBlockingSpectrum(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full-vs-blocked dense spectrum comparison in -short mode")
	}
	d, nocc, eps := symSetup(t)
	be := backend.Gonum{}
	intsOff := integrals.New(d, nocc, nil)
	intsSym := integrals.New(d, nocc, d.OrbSym)
	nsym := intsSym.NSym()

	spectrum := func(mx *Matrix) []float64 { ev, _ := be.SymEig(mx.BuildMatrix()); return ev }

	for _, order := range []int{2, 3} {
		full := spectrum(New(NewSpace(nocc, d.NORB, nil, 0), intsOff, eps, order, be))

		var blocked []float64
		for sym := range nsym {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym)
			if sp.Size() == 0 {
				continue
			}
			blocked = append(blocked, spectrum(New(sp, intsSym, eps, order, be))...)
		}
		if len(blocked) != len(full) {
			t.Fatalf("order %d: blocked dim %d != full dim %d", order, len(blocked), len(full))
		}
		sort.Float64s(blocked)
		var maxErr float64
		for i := range full {
			if e := math.Abs(full[i] - blocked[i]); e > maxErr {
				maxErr = e
			}
		}
		if maxErr > 1e-8 {
			t.Errorf("order %d: symmetry-blocked spectrum differs from full by %g (>1e-8)", order, maxErr)
		}
	}
}
