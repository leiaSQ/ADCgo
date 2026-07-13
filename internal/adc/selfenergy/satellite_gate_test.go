package selfenergy

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// satellite_gate_test.go — stage gates for the Σ(∞) machinery against theADCcode's own
// intermediates, dumped per (irrep, block) by ../ADC/self_energy/constanti/constanti/dmpsat.f
// (called from INVERT before the Jacobi loop overwrites the raw amplitudes).
//
// testdata/reference/h2o_dzp_sip_sigma/satellite/SAT_<psym>_<iab>.dat:
//
//	NDIM MAXP MAXOCC
//	MAXP × [ NP ; NDIM values ]        the coupling amplitudes U(:,p) (KOPP1+KOPP2)
//	NDIM × [ I ; value ]               the (K+C) diagonal
//	rest  × [ I J value ]              the strict lower-triangle (K+C) off-diagonal
//
// psym is theADCcode's 1-based GAMESS irrep, so ADCgo's 0-based irrep is psym−1.

type satRef struct {
	ndim, maxp, maxocc int
	u                  []float64 // [ndim × maxp], column-major by p (u[p*ndim+i])
	diag               []float64
	off                []offElem
}

type offElem struct {
	i, j int // 1-based, as the reference stores them
	v    float64
}

func readSatRef(t *testing.T, psym int, blk iab) *satRef {
	t.Helper()
	fn := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp_sip_sigma",
		"satellite", fmt.Sprintf("SAT_%d_%d.dat", psym, int(blk)))
	f, err := os.Open(fn)
	if err != nil {
		t.Skipf("satellite dump unavailable: %v", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)

	next := func() string {
		if !sc.Scan() {
			t.Fatalf("%s: unexpected EOF", fn)
		}
		return sc.Text()
	}

	r := &satRef{}
	if _, err := fmt.Sscan(next(), &r.ndim, &r.maxp, &r.maxocc); err != nil {
		t.Fatalf("%s: header: %v", fn, err)
	}
	r.u = make([]float64, r.ndim*r.maxp)
	for range r.maxp {
		var np int
		if _, err := fmt.Sscan(next(), &np); err != nil {
			t.Fatalf("%s: U column header: %v", fn, err)
		}
		for i := range r.ndim {
			var v float64
			if _, err := fmt.Sscan(next(), &v); err != nil {
				t.Fatalf("%s: U value: %v", fn, err)
			}
			r.u[(np-1)*r.ndim+i] = v
		}
	}
	r.diag = make([]float64, r.ndim)
	for range r.ndim {
		var i int
		var v float64
		if _, err := fmt.Sscan(next(), &i, &v); err != nil {
			t.Fatalf("%s: diagonal: %v", fn, err)
		}
		r.diag[i-1] = v
	}
	for sc.Scan() {
		var e offElem
		if _, err := fmt.Sscan(sc.Text(), &e.i, &e.j, &e.v); err != nil {
			t.Fatalf("%s: off-diagonal: %v", fn, err)
		}
		r.off = append(r.off, e)
	}
	return r
}

// psymCases maps theADCcode's 1-based irrep to ADCgo's 0-based one, for the irreps h2o/DZP
// actually has occupied orbitals in (A2 is empty, so constanti never builds it).
var psymCases = []struct {
	psym, sym int
	label     string
}{
	{1, 0, "A1"},
	{3, 2, "B1"},
	{4, 3, "B2"},
}

// TestSatSpaceDims gates the configuration-space enumeration: the spin-resolved dimension must
// equal theADCcode's NDIM for every (irrep, block).
func TestSatSpaceDims(t *testing.T) {
	ints, eps, nocc, norb := loadH2O(t)
	e := newEngine(ints, eps, nocc, norb)

	for _, tc := range psymCases {
		for _, blk := range []iab{iab2h1p, iab2p1h} {
			ref := readSatRef(t, tc.psym, blk)
			sp := e.buildSatSpace(blk, tc.sym)

			if sp.dim != ref.ndim {
				t.Errorf("%s block %d: dim = %d, want %d", tc.label, blk, sp.dim, ref.ndim)
			}
			if got := len(e.orbsOfSym(tc.sym)); got != ref.maxp {
				t.Errorf("%s: orbitals in irrep = %d, want %d", tc.label, got, ref.maxp)
			}
			if got := len(e.occs[tc.sym]); got != ref.maxocc {
				t.Errorf("%s: occupied in irrep = %d, want %d", tc.label, got, ref.maxocc)
			}
			t.Logf("%s block %d: dim=%d (ref %d)", tc.label, blk, sp.dim, ref.ndim)
		}
	}
}

// TestCouplingU gates the KOPP1+KOPP2 amplitudes elementwise against theADCcode.
//
// The spin-adapted basis carries an arbitrary sign per basis vector, so a mismatch that is a
// pure per-row sign flip would be physically harmless (ρ is quadratic in the amplitudes). We
// still demand an exact match: the port uses constanti's own spin tables, so the signs must
// agree too — anything else means a genuine transcription error.
func TestCouplingU(t *testing.T) {
	ints, eps, nocc, norb := loadH2O(t)
	e := newEngine(ints, eps, nocc, norb)

	for _, tc := range psymCases {
		for _, blk := range []iab{iab2h1p, iab2p1h} {
			ref := readSatRef(t, tc.psym, blk)
			sp := e.buildSatSpace(blk, tc.sym)
			if sp.dim != ref.ndim {
				t.Fatalf("%s block %d: dim %d != ref %d — fix the space first",
					tc.label, blk, sp.dim, ref.ndim)
			}
			orbs := e.orbsOfSym(tc.sym)
			u := e.coupling(sp)

			var maxd float64
			for np := range orbs {
				for i := range sp.dim {
					d := math.Abs(u[i*len(orbs)+np] - ref.u[np*ref.ndim+i])
					if d > maxd {
						maxd = d
					}
				}
			}
			if maxd > 1e-10 {
				t.Errorf("%s block %d: max |U − U_theADCcode| = %.3e", tc.label, blk, maxd)
			} else {
				t.Logf("%s block %d: U matches to %.2e over %d×%d", tc.label, blk, maxd,
					sp.dim, len(orbs))
			}
		}
	}
}
