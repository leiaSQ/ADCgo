//go:build hip || cuda

package lanczos

import (
	"math"
	"path/filepath"
	"testing"

	"adcgo/internal/adc/backend"
	"adcgo/internal/adc/dip"
	"adcgo/internal/adc/fcidump"
	"adcgo/internal/adc/integrals"
	"adcgo/internal/adc/mp"
)

func gpuBackend(t *testing.T) (backend.Backend, string) {
	t.Helper()
	for _, name := range backend.Available() {
		if name == "gonum" {
			continue
		}
		be, err := backend.New(name)
		if err != nil {
			t.Fatalf("New(%q): %v", name, err)
		}
		return be, name
	}
	t.Skip("no accelerated backend registered in this build")
	return nil, ""
}

func buildH2OWith(t *testing.T, spin dip.Spin, be backend.Backend) *dip.Matrix {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := dip.NewSpace(nocc, d.NORB, nil, 0, spin)
	return dip.New(sp, integrals.New(d, nocc, nil), eps, be)
}

// TestGPULanczosMatchesDense is the M3 spectrum-parity gate: a full block-Lanczos
// solve with the device mat-vec must reproduce the pure-Go dense spectrum. This
// exercises the resident GEMV (ApplyFull) plus the on-device AXPY/DOT/NRM2/SCAL of
// the Gram–Schmidt recurrence end-to-end.
func TestGPULanczosMatchesDense(t *testing.T) {
	gpu, name := gpuBackend(t)
	ref := backend.Gonum{}

	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		dense, _ := ref.SymEig(buildH2OWith(t, spin, ref).BuildMatrix())

		res := Solve(buildH2OWith(t, spin, gpu), gpu, Options{}) // full subspace → exact

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
		if maxErr > 1e-7 {
			t.Errorf("%s spin %d: device Lanczos vs dense max eigenvalue error %g (>1e-7)", name, spin, maxErr)
		}
	}
}
