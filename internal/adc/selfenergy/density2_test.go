package selfenergy

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestDensityAllOrder gates the all-order correlation density against theADCcode's own QKL,
// irrep by irrep, in the reference's triangular packing over [occ..., vir...] (the ½ diagonal factor
// applies only within the occ/occ and vir/vir blocks).
func TestDensityAllOrder(t *testing.T) {
	ints, eps, nocc, norb := loadH2O(t)
	e := newEngine(ints, eps, nocc, norb)
	rho := e.densityAllOrder(TheADCcodeDefaults)

	for _, tc := range psymCases {
		fn := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp_sip_sigma",
			"satellite", fmt.Sprintf("QKL_%d.dat", tc.psym))
		f, err := os.Open(fn)
		if err != nil {
			t.Skipf("no QKL dump: %v", err)
		}
		sc := bufio.NewScanner(f)
		sc.Scan()
		var maxp int
		fmt.Sscan(sc.Text(), &maxp)
		var ref []float64
		for sc.Scan() {
			var v float64
			fmt.Sscan(sc.Text(), &v)
			ref = append(ref, v)
		}
		f.Close()

		orbs := e.orbsOfSym(tc.sym)
		no := len(e.occs[tc.sym])
		idx := 0
		var worst float64
		var wi, wj int
		for li := range orbs {
			for ki := 0; ki <= li; ki++ {
				k, l := orbs[ki], orbs[li]
				got := rho.At(k, l)
				// triangular packing: ½ on the diagonal of the oo and vv blocks only
				sameBlock := (ki < no) == (li < no)
				if k == l && sameBlock {
					got *= 0.5
				}
				d := math.Abs(got - ref[idx])
				if d > worst {
					worst, wi, wj = d, ki, li
				}
				idx++
			}
		}
		if worst > 1e-10 {
			t.Errorf("%s: worst |ρ − QKL| = %.3e at position (%d,%d)", tc.label, worst, wi, wj)
		}
		t.Logf("%s: density matches QKL to %.2e (%d orbitals)", tc.label, worst, maxp)
	}
}
