//go:build cuda

package sip

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestC22MatFreeO3DeviceParity validates the CUDA matrix-free order-3 2h1p×2h1p satellite
// kernel (sip_kernels.cu c22_apply) against the dense BuildMatrix on real NVIDIA hardware:
// the device ApplyFull with -matfree on must reproduce the dense order-3 operator column by
// column. This is the production SIP-ADC(3) path — the dense satBlock is terabytes, so the GPU
// c22 kernel is what makes the full active space tractable. Skips when no CUDA device present.
func TestC22MatFreeO3DeviceParity(t *testing.T) {
	be, err := backend.New("cuda")
	if err != nil {
		t.Skipf("no cuda backend/device: %v", err)
	}
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace(nocc, d.NORB, d.OrbSym, 0) // order-3 1h + 2h1p sector
	ints := integrals.New(d, nocc, d.OrbSym)

	// Dense reference on the CPU backend.
	ref := New(sp, ints, eps, 3, backend.Gonum{})
	M := ref.BuildMatrix()
	n := ref.Size()

	// Device operator with the 2h1p satellite block applied matrix-free.
	mx := New(sp, ints, eps, 3, be)
	mx.SetMatFree(MatFreeOn, 0)

	// Sample columns across 1h / 2h1p.
	var cols []int
	for j := 0; j < sp.BeginSat; j++ {
		cols = append(cols, j)
	}
	step := max((n-sp.BeginSat)/60, 1)
	for j := sp.BeginSat; j < n; j += step {
		cols = append(cols, j)
	}

	e := make([]float64, n)
	out := be.Alloc(n)
	var maxErr float64
	for _, j := range cols {
		e[j] = 1
		in := be.Upload(e)
		e[j] = 0
		mx.ApplyFull(out, in)
		col := be.Download(out)
		for i := 0; i < n; i++ {
			if dd := math.Abs(col[i] - M.At(i, j)); dd > maxErr {
				maxErr = dd
			}
		}
		be.Free(in)
	}
	if maxErr > 1e-10 {
		t.Fatalf("device c22 order-3 matfree vs dense: max diff %g exceeds 1e-10", maxErr)
	}
	t.Logf("device c22 order-3 matfree parity: max diff %g  (n=%d, nSat=%d)", maxErr, n, n-sp.BeginSat)
}
