package lanczos

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
)

// TestDavidsonMatchesDense is the primary correctness gate: for a tight residual
// threshold the block-Davidson driver must reproduce the algebraically lowest NRoots
// eigenvalues and pole strengths of the dense path — exercised both with a small
// subspace cap (restart-heavy) and an uncapped subspace (no restart). It covers the
// dip.Matrix.Diagonal() preconditioner surface at the same time.
func TestDavidsonMatchesDense(t *testing.T) {
	be := backend.Gonum{}
	const nr = 6
	const tolE, tolPS = 1e-7, 1.0
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		ref := SolveDense(buildH2O(t, spin), be)
		if len(ref.Values) < nr {
			t.Fatalf("spin %d: dense spectrum has only %d states, need %d", spin, len(ref.Values), nr)
		}
		// maxsp 4*nr exercises thick restarts; 0 → uncapped, grows freely.
		for _, maxsp := range []int{4 * nr, 0} {
			mx := buildH2O(t, spin)
			res := SolveDavidson(mx, be, Options{NRoots: nr, ConvThr: 1e-9, MaxDim: maxsp, MaxIters: 1000})
			if len(res.Values) != nr {
				t.Fatalf("spin %d maxsp %d: got %d roots, want %d", spin, maxsp, len(res.Values), nr)
			}
			for k := range nr {
				if e := math.Abs(res.Values[k] - ref.Values[k]); e > tolE {
					t.Errorf("spin %d maxsp %d root %d: davidson %.10f dense %.10f Δ=%.2e",
						spin, maxsp, k, res.Values[k], ref.Values[k], e)
				}
				if p := math.Abs(res.PS[k] - ref.PS[k]); p > tolPS {
					t.Errorf("spin %d maxsp %d root %d PS: davidson %.4f dense %.4f",
						spin, maxsp, k, res.PS[k], ref.PS[k])
				}
				if res.Residual[k] > 1e-8 {
					t.Errorf("spin %d maxsp %d root %d: residual %.2e exceeds threshold",
						spin, maxsp, k, res.Residual[k])
				}
			}
		}
	}
}
