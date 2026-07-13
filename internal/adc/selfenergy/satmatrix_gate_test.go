package selfenergy

import (
	"math"
	"testing"
)

// TestSatMatrix gates the (K+C) satellite matrix — the diagonal and every sparse off-diagonal
// triplet — against theADCcode's own dump, for both blocks and every irrep.
//
// The reference stores the off-diagonal once per symmetric pair (row < col in its enumeration),
// applying it to both rows during the iteration, and drops |v| < TOLMAT. Both are replicated, so
// the triplet SET must match exactly — not merely the dense matrix they imply. That matters:
// the Jacobi iteration is truncated, so a different sparsity pattern would give a different
// (still converged-to-the-same-fixed-point) answer at the 1e-6 level.
func TestSatMatrix(t *testing.T) {
	ints, eps, nocc, norb := loadH2O(t)
	e := newEngine(ints, eps, nocc, norb)

	for _, tc := range psymCases {
		for _, blk := range []iab{iab2h1p, iab2p1h} {
			ref := readSatRef(t, tc.psym, blk)
			sp := e.buildSatSpace(blk, tc.sym)
			m := e.buildSatMatrix(sp)

			if len(m.diag) != ref.ndim {
				t.Errorf("%s block %d: %d diagonal entries, want %d (a TOLMAT drop would shrink this)",
					tc.label, blk, len(m.diag), ref.ndim)
				continue
			}
			var maxDiag float64
			for i := range m.diag {
				if d := math.Abs(m.diag[i] - ref.diag[i]); d > maxDiag {
					maxDiag = d
				}
			}
			if maxDiag > 1e-10 {
				t.Errorf("%s block %d: (K+C) diagonal max diff %.3e", tc.label, blk, maxDiag)
			}

			if len(m.off) != len(ref.off) {
				t.Errorf("%s block %d: %d off-diagonal triplets, want %d",
					tc.label, blk, len(m.off), len(ref.off))
				continue
			}
			// The reference's indices are 1-based and its traversal order is the one we
			// reproduce, so compare triplet-for-triplet in sequence.
			var maxOff float64
			var misplaced int
			for k, o := range m.off {
				r := ref.off[k]
				if o.i+1 != r.i || o.j+1 != r.j {
					misplaced++
					continue
				}
				if d := math.Abs(o.v - r.v); d > maxOff {
					maxOff = d
				}
			}
			if misplaced > 0 {
				t.Errorf("%s block %d: %d/%d off-diagonal triplets at a different index",
					tc.label, blk, misplaced, len(m.off))
				continue
			}
			if maxOff > 1e-10 {
				t.Errorf("%s block %d: (K+C) off-diagonal max diff %.3e", tc.label, blk, maxOff)
			}
			t.Logf("%s block %d: (K+C) diag %d (%.1e), off %d (%.1e)",
				tc.label, blk, len(m.diag), maxDiag, len(m.off), maxOff)
		}
	}
}
