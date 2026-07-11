//go:build cuda

package sip

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestWert2MatFreeDeviceParity validates the CUDA matrix-free 2h1p×3h2p coupling kernel
// (adc4_kernels.cu) against the dense BuildMatrix on real NVIDIA hardware: the device
// ApplyFull with -matfree on must reproduce the dense operator column-by-column. Skips
// when no CUDA device is present (e.g. a build host with the toolkit but no driver).
// c22 (satBlock2_4) has no device kernel yet, so it stays dense on the GPU — this test
// isolates the wert2 device path.
func TestWert2MatFreeDeviceParity(t *testing.T) {
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
	sp := NewSpace4(nocc, d.NORB, d.OrbSym, 0, []int{0}) // A1 sector: 1h+2h1p+3h2p
	ints := integrals.New(d, nocc, d.OrbSym)

	// Dense reference on the CPU backend.
	ref := New(sp, ints, eps, 4, backend.Gonum{})
	M := ref.BuildMatrix()
	n := ref.Size()

	// Device operator with the wert2 coupling applied matrix-free.
	mx := New(sp, ints, eps, 4, be)
	mx.SetMatFree(MatFreeOn, 0)

	// Sample columns across 1h / 2h1p / 3h2p.
	var cols []int
	for j := 0; j < sp.BeginSat; j++ {
		cols = append(cols, j)
	}
	stride := func(lo, hi, k int) {
		if hi <= lo {
			return
		}
		step := max((hi-lo)/k, 1)
		for j := lo; j < hi; j += step {
			cols = append(cols, j)
		}
	}
	stride(sp.BeginSat, sp.Begin3h2p, 40)
	stride(sp.Begin3h2p, n, 40)

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
		t.Fatalf("device wert2 matfree vs dense: max diff %g exceeds 1e-10", maxErr)
	}
	t.Logf("device wert2 matfree parity: max diff %g", maxErr)
}
