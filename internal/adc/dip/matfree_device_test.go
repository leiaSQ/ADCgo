//go:build cuda

package dip

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
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

		// Force many chunks even on this tiny system so the chunk-boundary logic (pointer
		// offsetting, chunk-local BufOff, per-chunk buffer reuse) is exercised, not just the
		// single-chunk path a 4 GiB budget takes here. budget=1 makes every block its own chunk.
		defer func(b int) { JIIFillBudgetElems = b }(JIIFillBudgetElems)
		JIIFillBudgetElems = 1

		p := mx.buildJIIBatchPlan()
		bufs := mx.buildJIIDeviceBufs(dk, p, s)
		defer bufs.free()
		bufs.args.ERI, bufs.args.Eps, bufs.args.OrbSym = dERI, dEps, dOsym

		if len(p.chunks) < 2 && len(p.apps) > 1 {
			t.Fatalf("spin=%v sym=%d: expected multiple chunks at budget=1 (%d applications), got %d",
				spin, sym, len(p.apps), len(p.chunks))
		}

		// Materialize chunk by chunk, comparing each chunk's handles to the host block builder.
		var maxErr, scale float64
		for _, ch := range p.chunks {
			bufs.args.SlotLo, bufs.args.SlotHi, bufs.args.ChunkElems = ch.lo, ch.hi, ch.elems
			mats := dk.DipSatFillJII(bufs.args)
			if len(mats) != ch.hi-ch.lo {
				t.Fatalf("spin=%v sym=%d: chunk [%d,%d) returned %d handles, want %d",
					spin, sym, ch.lo, ch.hi, len(mats), ch.hi-ch.lo)
			}
			for k, si := range p.apps[ch.lo:ch.hi] {
				sl := p.slots[si]
				want, ok := ref.buildSlot(sl)
				if !ok {
					t.Fatalf("spin=%v sym=%d: slot %d has no host block", spin, sym, si)
				}
				r, c := mats[k].Dims()
				if r != want.Rows || c != want.Cols {
					t.Fatalf("spin=%v sym=%d: slot %d dims %dx%d, host %dx%d",
						spin, sym, si, r, c, want.Rows, want.Cols)
				}
				got := dk.DownloadMat(mats[k])
				for j := range want.Data {
					if d := math.Abs(got[j] - want.Data[j]); d > maxErr {
						maxErr = d
					}
					if a := math.Abs(want.Data[j]); a > scale {
						scale = a
					}
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
// against the host loop applier. This is the configuration the production system actually runs in, and the one
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

	// budget=1 forces one block per fill chunk, so the whole applier exercises the chunk loop and
	// buffer reuse on this small system, not just the single-chunk path a 4 GiB default would take.
	defer func(b int) { JIIFillBudgetElems = b }(JIIFillBudgetElems)
	JIIFillBudgetElems = 1

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

	// budget=1 forces multi-chunk fills through the applier on this small system.
	defer func(b int) { JIIFillBudgetElems = b }(JIIFillBudgetElems)
	JIIFillBudgetElems = 1

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

// TestSatelliteMatFreePerDeviceParity validates the per-device on-device satellite apply — each
// GPU recomputing only its own output row band, reading a gathered full-height slab over NVLink —
// against BOTH references that must agree with it:
//
//   - the dense single-node operator (the physics), and
//   - the host gather-apply-scatter path it replaces (the thing being optimized away).
//
// Bit-exactness is required, not just closeness: every output row is summed by one owner in a
// fixed candidate order regardless of how the rows are partitioned (satscalar.go elem() resolves
// orientation per scalar pair), so partitioning must not perturb the arithmetic at all. A drift
// here would mean rows are being double-counted or dropped, not that floating point moved.
//
// It wants >= 4 devices: at 2 the partition-resolution and row-band clamping are trivial and a
// boundary bug can hide. Skips otherwise.
func TestSatelliteMatFreePerDeviceParity(t *testing.T) {
	const minDev = 4
	if c := backend.DeviceCount("cuda"); c < minDev {
		t.Skipf("need >= %d cuda devices for a meaningful partitioning, have %d", minDev, c)
	}
	subs, err := backend.NewAll("cuda", minDev)
	if err != nil {
		t.Skipf("no cuda devices: %v", err)
	}
	npartWant := len(subs)

	rng := rand.New(rand.NewSource(31))
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		n, main := sp.Size(), sp.MainBlockSize()
		bounds := sp.PartitionBounds(npartWant)
		npart := len(bounds) - 1
		if npart < 2 || n <= 2*main*main {
			return // sector too small to partition; the shape invariant would reject it
		}
		dist, err := backend.NewDistributed(subs[:npart], n, main, bounds)
		if err != nil {
			t.Fatalf("spin=%v sym=%d: NewDistributed: %v", spin, sym, err)
		}
		pd, ok := dist.(backend.PartitionedDevices)
		if !ok {
			t.Fatal("distributed backend does not expose PartitionedDevices")
		}
		if !perDeviceSatelliteOK(pd) {
			t.Skipf("cuda sub-backends are not fully peered; per-device path unavailable")
		}

		// Report the band layout: a run where some partition owns no satellite rows, or where
		// one straddles the main boundary, is exactly the interesting case — surface it so a
		// green run on a trivial layout is not mistaken for coverage.
		lo, hi := satRowBands(bounds, main, n)
		empty, straddle := 0, 0
		for d := range npart {
			if hi[d] <= lo[d] {
				empty++
			}
			if bounds[d] < main && bounds[d+1] > main {
				straddle++
			}
		}
		t.Logf("spin=%v sym=%d: n=%d main=%d parts=%d (empty bands=%d, straddling=%d)",
			spin, sym, n, main, npart, empty, straddle)

		dense := New(sp, ints, eps, be)
		perDev := New(sp, ints, eps, dist)
		perDev.SetMatFree(matfree.On, 0)
		if !perDev.matFreeSatellite() {
			t.Fatalf("spin=%v sym=%d: matFreeSatellite() false on distributed cuda backend", spin, sym)
		}

		// Deflation is real for this backend (panels are sized to the max block width but an
		// apply commonly uses fewer columns), and the chunk loop must handle a final short
		// chunk, so exercise a width that is not a multiple of SatChunkCols.
		for _, b := range []int{3, SatChunkCols + 1} {
			panel := make([]float64, n*b)
			for i := range panel {
				panel[i] = rng.NormFloat64()
			}

			want := backend.BlockView{V: be.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			dense.ApplyBlockSatellite(want, backend.BlockView{V: be.Upload(panel), Rows: n, Cols: b, Ld: n})
			wantH := be.Download(want.V)

			got := backend.BlockView{V: dist.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			perDev.ApplyBlockSatellite(got, backend.BlockView{V: dist.Upload(panel), Rows: n, Cols: b, Ld: n})
			gotH := dist.Download(got.V)

			var maxErr float64
			for i := range wantH {
				if d := math.Abs(gotH[i] - wantH[i]); d > maxErr {
					maxErr = d
				}
			}
			if maxErr > 1e-10 {
				t.Errorf("spin=%v sym=%d b=%d: per-device vs dense max diff %g exceeds 1e-10",
					spin, sym, b, maxErr)
			}

			// Same panel through the host gather-apply-scatter fallback: the two -mgpu paths
			// must agree with each other, not merely both be near the dense reference.
			hostSubs := make([]backend.Backend, npart)
			for i := range hostSubs {
				hostSubs[i] = backend.Gonum{}
			}
			hdist, err := backend.NewDistributed(hostSubs, n, main, bounds)
			if err != nil {
				t.Fatalf("NewDistributed(host): %v", err)
			}
			hmx := New(sp, ints, eps, hdist)
			hmx.SetMatFree(matfree.On, 0)
			hgot := backend.BlockView{V: hdist.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			hmx.ApplyBlockSatellite(hgot, backend.BlockView{V: hdist.Upload(panel), Rows: n, Cols: b, Ld: n})
			hostH := hdist.Download(hgot.V)

			var maxPath float64
			for i := range hostH {
				if d := math.Abs(gotH[i] - hostH[i]); d > maxPath {
					maxPath = d
				}
			}
			if maxPath > 1e-10 {
				t.Errorf("spin=%v sym=%d b=%d: per-device vs gather-apply-scatter max diff %g exceeds 1e-10",
					spin, sym, b, maxPath)
			}
		}
	})
}
