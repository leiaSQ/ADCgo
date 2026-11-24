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
