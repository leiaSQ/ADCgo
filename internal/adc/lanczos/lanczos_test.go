package lanczos

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

func buildH2O(t *testing.T, spin dip.Spin) *dip.Matrix {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := dip.NewSpace(nocc, d.NORB, nil, 0, spin)
	return dip.New(sp, integrals.New(d, nocc, nil), eps, backend.Gonum{})
}

// denseState is a reference (energy, ps) main-line from the dense path.
type denseState struct{ e, ps float64 }

func denseMainStates(mx *dip.Matrix, psMin float64) []denseState {
	be := backend.Gonum{}
	M := mx.BuildMatrix()
	evals, evecs := be.SymEig(M)
	main := mx.MainBlockSize()
	var out []denseState
	for k := range evals {
		var ps float64
		for c := range main {
			ps += evecs.At(c, k) * evecs.At(c, k)
		}
		ps *= 100
		if ps >= psMin {
			out = append(out, denseState{evals[k], ps})
		}
	}
	return out
}

// TestLanczosMatchesDense is Gate 2: the block-Lanczos spectrum must reproduce
// the dense eigenvalues and pole strengths of every main-space-weighted state.
func TestLanczosMatchesDense(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping heavy block-Lanczos convergence test in -short mode")
	}
	be := backend.Gonum{}
	// Tolerances reflect finite Lanczos convergence (not algorithm error — the
	// full-subspace test below shows exactness): energies to ~5e-3 Ha (~0.14 eV)
	// and pole strengths to 5% for the clear main lines.
	const tolE, tolPS = 5e-3, 5.0
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		mx := buildH2O(t, spin)
		ref := denseMainStates(mx, 65) // clear main lines
		res := Solve(mx, be, Options{MaxBlocks: 24})

		for _, d := range ref {
			// nearest non-spurious Ritz value.
			best, bestPS, bestErr := 0.0, 0.0, math.Inf(1)
			for k := range res.Values {
				if res.Spurious(k, 1e-9) {
					continue
				}
				if e := math.Abs(res.Values[k] - d.e); e < bestErr {
					bestErr, best, bestPS = e, res.Values[k], res.PS[k]
				}
			}
			if bestErr > tolE {
				t.Errorf("spin %d: dense state %.4f Ha (ps %.1f%%) unmatched; nearest Ritz %.4f (Δ=%.2e)",
					spin, d.e, d.ps, best, bestErr)
				continue
			}
			if math.Abs(bestPS-d.ps) > tolPS {
				t.Errorf("spin %d: state %.4f Ha ps mismatch dense=%.2f lanczos=%.2f",
					spin, d.e, d.ps, bestPS)
			}
		}
	}
}

// TestLanczosInvariance: full subspace reproduces the dense spectrum exactly.
func TestLanczosFullExact(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full-subspace exactness test in -short mode")
	}
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Triplet) // smaller main block → cheaper full build
	M := mx.BuildMatrix()
	dense, _ := be.SymEig(M)

	res := Solve(mx, be, Options{}) // no caps → grows to invariant subspace
	// Every dense eigenvalue must appear among the Ritz values.
	var maxErr float64
	for _, e := range dense {
		best := math.Inf(1)
		for _, r := range res.Values {
			if d := math.Abs(r - e); d < best {
				best = d
			}
		}
		if best > maxErr {
			maxErr = best
		}
	}
	if maxErr > 1e-8 {
		t.Errorf("full-subspace Lanczos vs dense max eigenvalue error %g", maxErr)
	}
}
