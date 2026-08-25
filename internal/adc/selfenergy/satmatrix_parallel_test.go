package selfenergy

import (
	"runtime"
	"testing"
)

// TestSatMatrixParallelMatchesSerial pins the two schedules buildSatMatrix can take against each
// other: the single-pass serial walk it keeps for small spaces, and the two-pass counted fill it
// takes once a space is worth parallelizing. Which one runs is chosen by n vs 2*GOMAXPROCS, so
// the choice is forced here by moving GOMAXPROCS rather than by touching the space.
//
// Byte-identical output is the contract, not merely equal values. The parallel pass decides each
// column's slot from a prefix sum over column index, so neither the ordering nor the length may
// depend on which worker reached a column first — and the counting walk and the filling walk must
// agree exactly, since a disagreement would truncate or misplace a column's triplets rather than
// fail loudly. TestSatMatrix gates the values themselves against theADCcode's dump.
func TestSatMatrixParallelMatchesSerial(t *testing.T) {
	ints, eps, nocc, norb := loadH2O(t)
	e := newEngine(ints, eps, nocc, norb)
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))

	checked := 0
	for _, tc := range psymCases {
		for _, blk := range []iab{iab2h1p, iab2p1h} {
			sp := e.buildSatSpace(blk, tc.sym)
			n := len(sp.confs)
			if n < 4 {
				continue // cannot reach the parallel path at the lowest usable GOMAXPROCS
			}

			// n < 2*GOMAXPROCS -> the serial single-pass walk.
			runtime.GOMAXPROCS(n)
			serial := e.buildSatMatrix(sp)

			// n >= 2*GOMAXPROCS -> the two-pass counted parallel fill.
			runtime.GOMAXPROCS(2)
			par := e.buildSatMatrix(sp)
			checked++

			if len(par.diag) != len(serial.diag) {
				t.Errorf("%s block %d: parallel diag has %d entries, serial %d",
					tc.label, blk, len(par.diag), len(serial.diag))
				continue
			}
			for i := range serial.diag {
				if par.diag[i] != serial.diag[i] {
					t.Errorf("%s block %d: diag[%d] = %g parallel, %g serial",
						tc.label, blk, i, par.diag[i], serial.diag[i])
					break
				}
			}

			if len(par.off) != len(serial.off) {
				t.Errorf("%s block %d: parallel has %d off-diagonal triplets, serial %d",
					tc.label, blk, len(par.off), len(serial.off))
				continue
			}
			for i := range serial.off {
				if par.off[i] != serial.off[i] {
					t.Errorf("%s block %d: off[%d] = %+v parallel, %+v serial",
						tc.label, blk, i, par.off[i], serial.off[i])
					break
				}
			}
		}
	}
	if checked == 0 {
		t.Fatal("no satellite space was large enough to exercise the parallel path")
	}
	t.Logf("compared %d satellite spaces across both schedules", checked)
}
