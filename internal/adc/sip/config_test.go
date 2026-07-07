package sip

import (
	"path/filepath"
	"testing"

	"adcgo/internal/adc/fcidump"
	"adcgo/internal/adc/mp"
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
