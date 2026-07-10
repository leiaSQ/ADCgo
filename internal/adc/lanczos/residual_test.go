package lanczos_test

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestRitzResidualBoundsEigenvalueError checks that Result.Residual is a genuine
// convergence measure and not decoration.
//
// For a real symmetric M and any unit y with θ = yᵀMy, the residual r = ‖My − θy‖
// bounds the distance from θ to the spectrum: min_i |θ − λ_i| ≤ r. So every reported
// Ritz value must sit within its own residual of some exact eigenvalue. A truncated
// run (small -blocks) produces both converged and unconverged roots, which is exactly
// what makes this test discriminating: a residual that were always ~0, or one
// unrelated to the error, would fail.
func TestRitzResidualBoundsEigenvalueError(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	be := backend.Gonum{}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)

	// A sector large enough that -blocks 6 leaves real convergence spread.
	sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, 0, dip.Singlet)
	if sp.Size() == 0 {
		t.Skip("empty sector")
	}
	mx := dip.New(sp, ints, eps, be)
	exact := lanczos.SolveDense(mx, be).Values

	res := lanczos.Solve(mx, be, lanczos.Options{MaxBlocks: 6})
	if len(res.Residual) != len(res.Values) {
		t.Fatalf("Residual has %d entries, want %d", len(res.Residual), len(res.Values))
	}

	var maxSlack, maxResid float64
	nonTrivial := 0
	for k, theta := range res.Values {
		best := math.Inf(1)
		for _, lam := range exact {
			best = math.Min(best, math.Abs(theta-lam))
		}
		r := res.Residual[k]
		maxResid = math.Max(maxResid, r)
		if r > 1e-9 {
			nonTrivial++
		}
		// |θ − λ| ≤ r, with a little slack for round-off.
		if best > r+1e-8 {
			t.Errorf("Ritz %d: θ=%.10f is %.3e from the spectrum but residual is only %.3e",
				k, theta, best, r)
		}
		maxSlack = math.Max(maxSlack, best-r)
	}
	if nonTrivial == 0 {
		t.Fatal("every residual was ~0; the truncated run should leave unconverged roots")
	}
	t.Logf("dim=%d roots=%d, %d with residual>1e-9, max residual %.3e, max (err - residual) %.3e",
		sp.Size(), len(res.Values), nonTrivial, maxResid, maxSlack)
}

// TestRitzResidualVanishesAtFullSubspace: once the Krylov space is the whole space,
// every Ritz pair is exact and the residual must collapse to round-off.
func TestRitzResidualVanishesAtFullSubspace(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	be := backend.Gonum{}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)

	sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, 1, dip.Triplet)
	if sp.Size() == 0 {
		t.Skip("empty sector")
	}
	mx := dip.New(sp, ints, eps, be)
	res := lanczos.Solve(mx, be, lanczos.Options{}) // run to full subspace

	var maxResid float64
	for _, r := range res.Residual {
		maxResid = math.Max(maxResid, r)
	}
	if maxResid > 1e-8 {
		t.Errorf("full-subspace max residual %.3e, want ~0", maxResid)
	}
	t.Logf("dim=%d: max residual at full subspace %.3e", sp.Size(), maxResid)
}
