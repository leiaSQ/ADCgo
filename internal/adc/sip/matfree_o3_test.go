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

// TestC22MatFreeO3HostParity validates the order-3 matrix-free 2h1p×2h1p satellite applier
// (newC22MatFreeO3, host) against the dense BuildMatrix: with -matfree on, ApplyFull must
// reproduce the dense operator column-by-column. This runs on the CPU (Gonum) backend, so it
// pins the element math and — crucially — the c22off directionality (element(r,c) evaluated
// with the lower-indexed config as the row), independent of any GPU. The GPU kernel is a
// bit-for-bit port of the same evaluation, checked separately by TestC22MatFreeO3DeviceParity.
func TestC22MatFreeO3HostParity(t *testing.T) {
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)
	sp := NewSpace(nocc, d.NORB, d.OrbSym, 0)
	if sp.Size() == sp.BeginSat {
		t.Fatalf("no 2h1p satellite configs to exercise")
	}

	be := backend.Gonum{}
	ref := New(sp, ints, eps, 3, be)
	M := ref.BuildMatrix() // dense satBlock reference
	n := ref.Size()

	mx := New(sp, ints, eps, 3, be)
	mx.SetMatFree(MatFreeOn, 0) // force the matrix-free satellite path

	e := make([]float64, n)
	out := be.Alloc(n)
	var maxErr float64
	for j := 0; j < n; j++ {
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
		t.Fatalf("order-3 c22 matfree(host) vs dense: max diff %g exceeds 1e-10", maxErr)
	}
	t.Logf("order-3 c22 matfree(host) parity: max diff %g  (n=%d, nSat=%d)", maxErr, n, n-sp.BeginSat)
}
