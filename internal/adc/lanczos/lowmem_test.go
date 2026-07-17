package lanczos

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
)

// TestLowMemProbe logs the H2O sector geometry, so the Mode A/B tolerances below are chosen
// against real dimensions rather than guesses.
func TestLowMemProbe(t *testing.T) {
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		mx := buildH2O(t, spin)
		t.Logf("spin=%d: n=%d main=%d", spin, mx.Size(), mx.MainBlockSize())
	}
}

// TestSolveLowMemModeB_MatchesDense: the faithful block=main short-recurrence solve
// (Tarantelli gate + banded eigensolver) reproduces the dense main-weighted spectrum.
func TestSolveLowMemModeB_MatchesDense(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping heavy low-memory Lanczos test in -short mode")
	}
	be := backend.Gonum{}
	const tolE, tolPS = 5e-3, 5.0
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		mx := buildH2O(t, spin)
		ref := denseMainStates(mx, 65)
		// LowMemBlock 0 → block = main → Mode B (dip.Matrix is a SatelliteOperator).
		res := SolveLowMem(mx, be, Options{MaxBlocks: 24})
		matchMainLines(t, "modeB spin="+spinName(spin), res, ref, tolE, tolPS)
	}
}

// TestSolveLowMemModeA_Sound: the small-block, device-frugal full-reorth mode is numerically
// sound — every non-spurious Ritz value it returns is a genuine eigenvalue, and the pole
// strengths of the main lines it does reach are correct. It is NOT tested for completeness:
// a block smaller than the main space cannot span every pole-carrying direction, so some main
// lines are unreachable by construction (that is the documented Mode A trade-off, and why the
// faithful melanin path is Mode B with block = main). Here we assert correctness of what it
// returns and that it recovers the reachable main lines' strengths.
func TestSolveLowMemModeA_Sound(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping heavy low-memory Lanczos test in -short mode")
	}
	be := backend.Gonum{}
	const tolE, tolPS = 5e-3, 5.0
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		mx := buildH2O(t, spin)
		M := mx.BuildMatrix()
		dense, _ := be.SymEig(M)
		res := SolveLowMem(mx, be, Options{MaxBlocks: 120, LowMemBlock: 5})

		// Soundness: every returned root lies close to some dense eigenvalue — no ghosts leak
		// through. Full reorthogonalization makes every Ritz value a valid Rayleigh quotient,
		// so unconverged interior values sit within convergence tolerance (~1e-4) of a true
		// eigenvalue; a genuine ghost would be O(0.1) off. 5e-3 separates the two.
		for k := range res.Values {
			best := math.Inf(1)
			for _, e := range dense {
				if d := math.Abs(res.Values[k] - e); d < best {
					best = d
				}
			}
			if best > 5e-3 {
				t.Errorf("modeA %s: Ritz value %.6f is far from any dense eigenvalue (Δ=%.2e) — ghost leak",
					spinName(spin), res.Values[k], best)
			}
		}

		// Pole strengths correct for the reachable main lines.
		ref := denseMainStates(mx, 65)
		reached := 0
		for _, d := range ref {
			bestErr, bestPS := math.Inf(1), 0.0
			for k := range res.Values {
				if e := math.Abs(res.Values[k] - d.e); e < bestErr {
					bestErr, bestPS = e, res.PS[k]
				}
			}
			if bestErr > tolE {
				continue // unreachable from this small start block — expected
			}
			reached++
			if math.Abs(bestPS-d.ps) > tolPS {
				t.Errorf("modeA %s: reached state %.4f Ha ps dense=%.2f lowmem=%.2f",
					spinName(spin), d.e, d.ps, bestPS)
			}
		}
		if reached == 0 {
			t.Errorf("modeA %s: reached no main lines at all", spinName(spin))
		}
		t.Logf("modeA %s: reached %d/%d main lines, all pole strengths correct", spinName(spin), reached, len(ref))
	}
}

// TestSolveLowMemModeA_FullExact: run Mode A to the invariant subspace; every root it
// returns must match a dense eigenvalue exactly (no spurious leakage past the ghost filter).
func TestSolveLowMemModeA_FullExact(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full-subspace low-memory exactness test in -short mode")
	}
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Triplet)
	M := mx.BuildMatrix()
	dense, _ := be.SymEig(M)

	res := SolveLowMem(mx, be, Options{LowMemBlock: 2}) // no MaxBlocks cap → grows out
	var maxErr float64
	for _, r := range res.Values {
		best := math.Inf(1)
		for _, e := range dense {
			if d := math.Abs(r - e); d < best {
				best = d
			}
		}
		if best > maxErr {
			maxErr = best
		}
	}
	if maxErr > 1e-8 {
		t.Errorf("Mode A full-subspace: returned Ritz value off dense by %g", maxErr)
	}
}

func spinName(s dip.Spin) string {
	if s == dip.Triplet {
		return "T"
	}
	return "S"
}

// matchMainLines checks that every clear dense main line is matched by a non-spurious Ritz
// value within tolE (energy) and tolPS (pole strength).
func matchMainLines(t *testing.T, label string, res Result, ref []denseState, tolE, tolPS float64) {
	t.Helper()
	for _, d := range ref {
		best, bestPS, bestErr := 0.0, 0.0, math.Inf(1)
		for k := range res.Values {
			if e := math.Abs(res.Values[k] - d.e); e < bestErr {
				bestErr, best, bestPS = e, res.Values[k], res.PS[k]
			}
		}
		if bestErr > tolE {
			t.Errorf("%s: dense state %.4f Ha (ps %.1f%%) unmatched; nearest Ritz %.4f (Δ=%.2e)",
				label, d.e, d.ps, best, bestErr)
			continue
		}
		if math.Abs(bestPS-d.ps) > tolPS {
			t.Errorf("%s: state %.4f Ha ps mismatch dense=%.2f lowmem=%.2f", label, d.e, d.ps, bestPS)
		}
	}
}
