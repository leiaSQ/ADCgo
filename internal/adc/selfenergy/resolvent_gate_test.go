package selfenergy

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// readResRef loads theADCcode's converged resolvent amplitudes for one block:
// RES_<psym>_<iab>.dat = "NDIM INDSUP ISHIFT" then INDSUP × [ NP ; NDIM values ].
// Columns are returned indexed by position within the irrep (ISHIFT + NP − 1).
func readResRef(t *testing.T, psym int, blk iab) (map[int][]float64, int) {
	t.Helper()
	fn := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp_sip_sigma",
		"satellite", fmt.Sprintf("RES_%d_%d.dat", psym, int(blk)))
	f, err := os.Open(fn)
	if err != nil {
		t.Skipf("resolvent dump unavailable: %v", err)
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
	var ndim, indsup, ishift int
	if _, err := fmt.Sscan(next(), &ndim, &indsup, &ishift); err != nil {
		t.Fatalf("%s: header: %v", fn, err)
	}
	out := map[int][]float64{}
	for range indsup {
		var np int
		if _, err := fmt.Sscan(next(), &np); err != nil {
			t.Fatalf("%s: column header: %v", fn, err)
		}
		col := make([]float64, ndim)
		for i := range ndim {
			if _, err := fmt.Sscan(next(), &col[i]); err != nil {
				t.Fatalf("%s: value: %v", fn, err)
			}
		}
		out[ishift+np-1] = col
	}
	return out, ndim
}

// TestResolvent gates the truncated Jacobi iteration itself: with theADCcode's own Akrit/MaxIt,
// the amplitudes y = (ε_p − K − C)⁻¹U must match its converged VDYSON. This isolates the
// iteration from the density contraction that consumes it.
func TestResolvent(t *testing.T) {
	ints, eps, nocc, norb := loadH2O(t)
	e := newEngine(ints, eps, nocc, norb)

	for _, tc := range psymCases {
		for _, blk := range []iab{iab2h1p, iab2p1h} {
			ref, ndim := readResRef(t, tc.psym, blk)
			sp := e.buildSatSpace(blk, tc.sym)
			if sp.dim != ndim {
				t.Fatalf("%s block %d: dim %d != ref %d", tc.label, blk, sp.dim, ndim)
			}
			m := e.buildSatMatrix(sp)
			u := e.coupling(sp)
			y := e.solveResolvent(sp, m, u, TheADCcodeDefaults)

			var maxd float64
			var cols int
			for np, want := range ref {
				got := y[np]
				if got == nil {
					t.Fatalf("%s block %d: column %d not solved for", tc.label, blk, np)
				}
				cols++
				for i := range want {
					if d := math.Abs(got[i] - want[i]); d > maxd {
						maxd = d
					}
				}
			}
			if maxd > 1e-10 {
				t.Errorf("%s block %d: resolvent max |y − y_theADCcode| = %.3e over %d columns",
					tc.label, blk, maxd, cols)
			} else {
				t.Logf("%s block %d: resolvent matches to %.2e (%d columns)", tc.label, blk, maxd, cols)
			}
		}
	}
}
