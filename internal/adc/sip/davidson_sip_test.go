package sip

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
)

// TestDavidsonMatchesDenseSIP checks the block-Davidson driver against the dense path on
// an IP-ADC(2) SIP sector, covering the sip.Matrix.Diagonal() preconditioner surface (the
// c11/c22diag block-diagonal assembly). The lowest NRoots eigenvalues and pole strengths
// must agree to a tight tolerance when converged to a small residual.
func TestDavidsonMatchesDenseSIP(t *testing.T) {
	be := backend.Gonum{}
	const nr = 6
	const tolE, tolPS = 1e-7, 1.0

	_, mxRef, _ := dipoleSpace(t)
	ref := lanczos.SolveDense(mxRef, be)
	if len(ref.Values) < nr {
		t.Fatalf("dense SIP spectrum has only %d states, need %d", len(ref.Values), nr)
	}

	_, mx, _ := dipoleSpace(t)
	res := lanczos.SolveDavidson(mx, be, lanczos.Options{NRoots: nr, ConvThr: 1e-9, MaxDim: 2 * nr, MaxIters: 500})
	if len(res.Values) != nr {
		t.Fatalf("davidson returned %d roots, want %d", len(res.Values), nr)
	}
	for k := range nr {
		if e := math.Abs(res.Values[k] - ref.Values[k]); e > tolE {
			t.Errorf("root %d: davidson %.10f dense %.10f Δ=%.2e", k, res.Values[k], ref.Values[k], e)
		}
		if p := math.Abs(res.PS[k] - ref.PS[k]); p > tolPS {
			t.Errorf("root %d PS: davidson %.4f dense %.4f", k, res.PS[k], ref.PS[k])
		}
	}
}
