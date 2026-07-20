//go:build cuda

package dip

import (
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestSatelliteMatFreeDeviceParity validates the CUDA DIP 3h1p↔3h1p satellite kernel
// (backend/adc2dip_kernels.cu) against the dense operator on real NVIDIA hardware: a cuda
// Matrix with -matfree on must reproduce the dense (host) operator for both ApplyFull and a
// panel ApplyBlock, over every h2o DIP sector (singlet + triplet). It skips when no CUDA
// device is present. The host tests already pin the physics (TestSatelliteScalarMatchesDense,
// TestSatelliteScalarApplyEqualsDense); this isolates the C transcription + the device marshal.
func TestSatelliteMatFreeDeviceParity(t *testing.T) {
	dev, err := backend.New("cuda")
	if err != nil {
		t.Skipf("no cuda backend/device: %v", err)
	}
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)
	host := backend.Gonum{}
	rng := rand.New(rand.NewSource(23))

	tested := 0
	for _, spin := range []Spin{Singlet, Triplet} {
		for sym := range 4 {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			if sp.Size() == 0 || sp.Size() == sp.MainBlockSize() {
				continue
			}
			n := sp.Size()
			dense := New(sp, ints, eps, host)
			free := New(sp, ints, eps, dev)
			free.SetMatFree(matfree.On, 0)
			if !free.matFreeSatellite() {
				t.Fatalf("spin=%v sym=%d: matFreeSatellite() false on cuda backend", spin, sym)
			}

			// ApplyFull, single vector.
			x := make([]float64, n)
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			wantV := host.Alloc(n)
			dense.ApplyFull(wantV, host.Upload(x))
			gotV := dev.Alloc(n)
			free.ApplyFull(gotV, dev.Upload(x))
			assertClose(t, spin, sym, "cuda ApplyFull", host.Download(wantV), dev.Download(gotV))

			// ApplyBlock, a 4-column panel.
			const b = 4
			panel := make([]float64, n*b)
			for i := range panel {
				panel[i] = rng.NormFloat64()
			}
			wantB := backend.BlockView{V: host.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			dense.ApplyBlock(wantB, backend.BlockView{V: host.Upload(panel), Rows: n, Cols: b, Ld: n})
			gotB := backend.BlockView{V: dev.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			free.ApplyBlock(gotB, backend.BlockView{V: dev.Upload(panel), Rows: n, Cols: b, Ld: n})
			assertClose(t, spin, sym, "cuda ApplyBlock", host.Download(wantB.V), dev.Download(gotB.V))

			free.Release()
			tested++
		}
	}
	if tested == 0 {
		t.Fatal("no sectors with a satellite space exercised")
	}
}
