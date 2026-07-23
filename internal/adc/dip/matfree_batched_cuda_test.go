//go:build cuda

package dip

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
)

// TestJIIFillDeviceMatchesHost validates the CUDA jiiLKK fill kernel (adc2dip_kernels.cu
// dip_fill_jii) against the host block builder, block by block.
//
// This is the per-element oracle for the contraction path, and it is deliberately finer-grained
// than a whole-apply comparison: the fill kernel's output IS the operator block, so it can be
// compared entry-by-entry to mx.blk.jiiLKK — recovering, for the device contraction path, the
// per-element verifiability that docs/sigma_build_contractions.md warned a contraction rewrite
// would give up. The kernel calls the same d_jii_s/d_jii_t as dip_sat_apply, so a mismatch here
// is a marshaling/indexing bug (SoA offsets, row-major layout, orientation), not physics.
func TestJIIFillDeviceMatchesHost(t *testing.T) {
	be, err := backend.New("cuda")
	if err != nil {
		t.Skipf("no cuda backend/device: %v", err)
	}
	dk, ok := be.(backend.DeviceKernels)
	if !ok {
		t.Skip("cuda backend does not expose DeviceKernels")
	}

	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, host backend.Backend) {
		// Host reference (Gonum) and the device matrix under test.
		ref := New(sp, ints, eps, host)
		mx := New(sp, ints, eps, be)

		s := mx.buildSatDeviceSoA(mx.buildSatScalarPlan())
		dERI, dEps, dOsym := dk.DeviceERI(s.eri), dk.UploadFloats(s.eps), dk.UploadInts(s.osym)
		defer func() {
			dk.FreeDev(dERI)
			dk.FreeDev(dEps)
			dk.FreeDev(dOsym)
		}()

		p := mx.buildJIIBatchPlan()
		bufs := mx.buildJIIDeviceBufs(dk, p, s)
		defer bufs.free()
		bufs.args.ERI, bufs.args.Eps, bufs.args.OrbSym = dERI, dEps, dOsym

		mats := dk.DipSatFillJII(bufs.args)
		if len(mats) != len(p.slots) {
			t.Fatalf("spin=%v sym=%d: fill returned %d handles, want %d", spin, sym, len(mats), len(p.slots))
		}

		var maxErr, scale float64
		for i, sl := range p.slots {
			want, ok := ref.buildSlot(sl)
			if !ok {
				t.Fatalf("spin=%v sym=%d: slot %d has no host block", spin, sym, i)
			}
			r, c := mats[i].Dims()
			if r != want.Rows || c != want.Cols {
				t.Fatalf("spin=%v sym=%d: slot %d dims %dx%d, host %dx%d",
					spin, sym, i, r, c, want.Rows, want.Cols)
			}
			got := dk.DownloadMat(mats[i])
			for k := range want.Data {
				if d := math.Abs(got[k] - want.Data[k]); d > maxErr {
					maxErr = d
				}
				if a := math.Abs(want.Data[k]); a > scale {
					scale = a
				}
			}
		}
		if maxErr > 1e-12*(1+scale) {
			t.Errorf("spin=%v sym=%d: device fill vs host block: max |Δ| = %g (scale %g)",
				spin, sym, maxErr, scale)
		}
	})
}

// TestSatBatchedPerDeviceParity validates the MULTI-GPU contraction path (newSatBatchedPerDevice)
// against the host loop applier. This is the configuration melanin actually runs in, and the one
// piece with genuinely new logic: each planned batch is split by which device owns its write
// offset, and output offsets are rebased into that device's local partition.
//
// The split is only sound because PlanBatches guarantees distinct write offsets within a batch and
// dip.PartitionBounds is group-aligned, so no block's band straddles two partitions. A bug there
// drops or double-counts whole blocks, which shows up here as a large error rather than drift.
//
// Wants >= 4 peered devices: at 2 the ownership split is nearly trivial.
func TestSatBatchedPerDeviceParity(t *testing.T) {
	const minDev = 4
	if c := backend.DeviceCount("cuda"); c < minDev {
		t.Skipf("need >= %d cuda devices, have %d", minDev, c)
	}
	subs, err := backend.NewAll("cuda", minDev)
	if err != nil {
		t.Skipf("no cuda devices: %v", err)
	}
	rng := rand.New(rand.NewSource(505))

	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, host backend.Backend) {
		n, main := sp.Size(), sp.MainBlockSize()
		bounds := sp.PartitionBounds(len(subs))
		npart := len(bounds) - 1
		if npart < 2 || n <= 2*main*main {
			return
		}
		dist, err := backend.NewDistributed(subs[:npart], n, main, bounds)
		if err != nil {
			t.Fatalf("NewDistributed: %v", err)
		}
		pd, ok := dist.(backend.PartitionedDevices)
		if !ok || !perDeviceSatelliteOK(pd) {
			t.Skip("cuda sub-backends not fully peered; batched per-device path unavailable")
		}

		const b = 3
		panel := make([]float64, n*b)
		for i := range panel {
			panel[i] = rng.NormFloat64()
		}

		// Reference: full host loop applier.
		refV := host.Alloc(n * b)
		host.Zero(refV)
		New(sp, ints, eps, host).newSatelliteMatFreeExcept(false).apply(
			backend.BlockView{V: host.Upload(panel), Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: refV, Rows: n, Cols: b, Ld: n},
		)
		want := host.Download(refV)

		mx := New(sp, ints, eps, dist)
		part := mx.newSatBatchedPerDevice(pd)
		defer part.release()
		outV := dist.Alloc(n * b)
		dist.Zero(outV)
		part.apply(
			backend.BlockView{V: dist.Upload(panel), Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: outV, Rows: n, Cols: b, Ld: n},
		)
		got := dist.Download(outV)

		var maxErr, scale float64
		for i := range want {
			if d := math.Abs(got[i] - want[i]); d > maxErr {
				maxErr = d
			}
			if a := math.Abs(want[i]); a > scale {
				scale = a
			}
		}
		if maxErr > 1e-10*(1+scale) {
			t.Errorf("spin=%v sym=%d parts=%d: batched per-device vs host loop: max |Δ| = %g (scale %g)",
				spin, sym, npart, maxErr, scale)
		}
		t.Logf("spin=%v sym=%d: n=%d parts=%d max|Δ|=%g", spin, sym, n, npart, maxErr)
	})
}

// TestJIIMatFreeBatchedDeviceParity checks the whole device jiiLKK contraction path — fill kernel
// plus batched GEMM — against the host loop applier on the same panel. Reassociation differs
// (cuBLAS sums within a block in its own order), so the bar is relative, not bitwise.
func TestJIIMatFreeBatchedDeviceParity(t *testing.T) {
	be, err := backend.New("cuda")
	if err != nil {
		t.Skipf("no cuda backend/device: %v", err)
	}
	dk, ok := be.(backend.DeviceKernels)
	if !ok {
		t.Skip("cuda backend does not expose DeviceKernels")
	}
	rng := rand.New(rand.NewSource(303))

	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, host backend.Backend) {
		n := sp.Size()
		const b = 3
		panel := make([]float64, n*b)
		for i := range panel {
			panel[i] = rng.NormFloat64()
		}

		// Host reference: the full loop applier over all three satellite blocks.
		refOutV := host.Alloc(n * b)
		host.Zero(refOutV)
		New(sp, ints, eps, host).newSatelliteMatFreeExcept(false).apply(
			backend.BlockView{V: host.Upload(panel), Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: refOutV, Rows: n, Cols: b, Ld: n},
		)
		refOut := host.Download(refOutV)

		mx := New(sp, ints, eps, be)
		s := mx.buildSatDeviceSoA(mx.buildSatScalarPlan())
		dERI, dEps, dOsym := dk.DeviceERI(s.eri), dk.UploadFloats(s.eps), dk.UploadInts(s.osym)
		defer func() {
			dk.FreeDev(dERI)
			dk.FreeDev(dEps)
			dk.FreeDev(dOsym)
		}()

		part, _ := mx.newJIIMatFreeBatchedDevice(dk, s, dERI, dEps, dOsym)
		defer part.release()

		inV := be.Upload(panel)
		outV := be.Alloc(n * b)
		be.Zero(outV)
		part.apply(
			backend.BlockView{V: inV, Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: outV, Rows: n, Cols: b, Ld: n},
		)
		got := be.Download(outV)

		var maxErr, scale float64
		for i := range refOut {
			if d := math.Abs(got[i] - refOut[i]); d > maxErr {
				maxErr = d
			}
			if a := math.Abs(refOut[i]); a > scale {
				scale = a
			}
		}
		if maxErr > 1e-10*(1+scale) {
			t.Errorf("spin=%v sym=%d: device batched jiiLKK vs host loop: max |Δ| = %g (scale %g)",
				spin, sym, maxErr, scale)
		}

		// Main-space rows must be untouched (asserted literally zero elsewhere).
		for j := range b {
			for i := range sp.MainBlockSize() {
				if got[i+j*n] != 0 {
					t.Fatalf("spin=%v sym=%d: device path wrote main-space row %d col %d (%g)",
						spin, sym, i, j, got[i+j*n])
				}
			}
		}
	})
}
