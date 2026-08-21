package dip

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"
	"unsafe"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// matfree_dist.go — matrix-free 3h1p↔3h1p satellite apply under the row-partitioned (-mgpu)
// backend. Composes the matrix-free satellite region with the distributed Mode-B solver so a
// whole-band DIP sector fits a multi-device node: the dense main/coupling blocks and the Krylov
// panels stay partitioned across the devices (distBackend.GemmMat), and the satellite region —
// the multi-TB memory hog that made -mgpu materialize densely — is recomputed instead of stored.
//
// Two appliers live here:
//
//   - newSatelliteMatFreePerDevice (preferred): each device recomputes ONLY its own output row
//     band, on-device, with the CUDA kernel. Requires every partition to expose device kernels
//     and every pair to be peered (backend.PartitionedDevices).
//   - newSatelliteMatFreeDistributed (fallback): gather-apply-scatter through the host. Correct
//     everywhere — including gonum sub-backends, which have no kernel — but it downloads the
//     whole panel, contracts on CPU and uploads it back (~137 GB each way for the production system), which is
//     the bottleneck docs/dip_operator_memory.md flags as the Phase-C follow-up.
//
// TestSatelliteMatFreeDistributedEqualsDense validates the fallback over gonum sub-backends.

// SatChunkCols is the column-chunk width of the per-device gather. Each device stages a
// full-height n×SatChunkCols input slab, so the slab costs n·SatChunkCols·8 bytes per device
// (production triplet, 64: ~7.6 GB).
//
// It must not be small: chunking repeats the element evaluation ceil(b/w) times. The column loop
// dominates a candidate element (~b ops of apply against ~30 to evaluate), so at w=64 the
// recompute overhead is ~15% of chunk work and falls as w grows — the trade is recompute against
// slab residency. It also sets the barrier count: the applier fences twice per chunk, so ceil(b/w)
// chunks means 2·ceil(b/w) multi-device syncs per mat-vec, and doubling w halves them.
//
// The measured production budget (docs/dip_operator_memory.md) leaves ~26 GB spare per device on the
// triplet at -mgpu 8, enough to raise this to 128 and roughly halve both the recompute overhead
// and the barrier count. It stays 64 by default because that is what has been run; -satchunk
// overrides it.
//
// A var, not a const, so cmd/adcgo can set it and cmd/sizeprobe can report the real value rather
// than a hand-copied mirror. Read when an applier is CONSTRUCTED (it sizes the slab allocation),
// so set it before building a Matrix; changing it mid-solve does nothing useful.
var SatChunkCols = 64

// SatTrace makes the -mgpu satellite appliers print a per-COLUMN-CHUNK timing breakdown to stderr.
//
// It exists because a whole-band production mat-vec costs 40 h (job 14158038 measured block 0 at
// apply=40h15m18s), and lanczos.Timing buckets all of that into a single "apply" number that only
// appears once the block finishes. The apply is a loop of ceil(b/SatChunkCols) column chunks —
// 27 of them at the production system's b=1711 — so tracing per chunk turns a 40 h measurement into a ~90 min
// one, and splits it into the parts that can actually be acted on: the NVLink gather, the two
// fences around it, the operator fill, and the batched GEMM.
//
// Off by default: it writes a line per chunk per device.
var SatTrace = false

// satTracef prints one trace line when SatTrace is on. Kept here so the appliers stay readable and
// the package has exactly one place that writes to stderr.
func satTracef(format string, a ...any) {
	if SatTrace {
		fmt.Fprintf(os.Stderr, "sattrace "+format+"\n", a...)
	}
}

// checkSatChunkFits guards the one 32-bit index in the satellite kernel. dip_sat_apply addresses
// its FULL-HEIGHT input as xin[C + jc*ldIn] with every operand a C `int`, where ldIn is the whole
// sector height n and jc < w. The largest element offset it forms is therefore w·n − 1, which must
// stay inside int32 — a wrapped index reads out of bounds, silently on some allocation layouts and
// as a cudaErrorLaunchFailure on others, and neither is caught at launch (adc2_dip_sat_apply
// returns only cudaGetLastError(), which does not see execution faults).
//
// The default w=64 is comfortable — the production system's n=10,014,483 sector sits at 6.4e8 of 2.1e9 — but the
// ceiling for that sector is w ≤ MaxInt32/n = 214, and both SatChunkCols' own doc comment above
// and the -satchunk flag invite raising w to 128 "and beyond". 256 would silently corrupt it. Fail
// at construction instead, where the message can name the limit, rather than days into a run.
func checkSatChunkFits(w, n int) {
	if int64(w)*int64(n) > math.MaxInt32 {
		panic(fmt.Sprintf("dip: -satchunk %d with n=%d overflows the satellite kernel's 32-bit input "+
			"index (w·n = %d > %d); use -satchunk %d or less for this sector",
			w, n, int64(w)*int64(n), int64(math.MaxInt32), math.MaxInt32/n))
	}
}

// syncAll drains every partition's device stream concurrently. A peer read does not synchronize
// the source stream, so the gather must be fenced on both sides: after the producers have written
// the input, and after the kernels have written their output bands. The syncs are issued in
// parallel because each is a blocking round-trip through that device's owning goroutine —
// serialized over 8 devices, twice per apply, that is pure added latency.
func syncAll(pd backend.PartitionedDevices) {
	var wg sync.WaitGroup
	for d := range pd.NumParts() {
		pc, ok := pd.PartBackend(d).(backend.PeerCopier)
		if !ok {
			continue
		}
		wg.Go(pc.Sync)
	}
	wg.Wait()
}

// gatherSlabs stages, on every partition, the full-height n×cw input slab for panel columns
// [c0, c0+cw): device d's slab receives every partition's row band, because a satellite block's
// candidate COLUMN can live on any partition (backend/README.md, "The slab must be full height").
//
// The DESTINATION loop is issued concurrently. PeerCopy2D (backend/gpu_device.go) resolves both
// pointers on the calling goroutine and then blocks on a round-trip through the *destination*
// backend's owning thread — the source is touched only as a pointer, so distinct destinations
// share neither channel nor BLAS handle. Run serially this is nd² blocking transfers per chunk
// (~1,664 per mat-vec at nd=8 over the production system's 26 chunks) on a fabric built to overlap them. The
// inner src loop stays serial: those copies DO all funnel through the one destination thread.
//
// Copies only — no arithmetic is reordered, so the gathered slab is bit-identical to the serial
// version regardless of completion order.
func gatherSlabs(pd backend.PartitionedDevices, slabOf func(int) backend.Vector,
	in backend.BlockView, bounds []int, n, c0, cw int) {
	nd := pd.NumParts()
	var wg sync.WaitGroup
	for d := range nd {
		// Assert on this goroutine, not inside the worker: this applier is only selected when
		// every partition is a peered PeerCopier, so a failure here is a selection bug and
		// should not surface as a panic on an anonymous goroutine.
		dst := pd.PartBackend(d).(backend.PeerCopier)
		slab := slabOf(d)
		wg.Go(func() {
			for src := range nd {
				rows := bounds[src+1] - bounds[src]
				if rows == 0 {
					continue
				}
				// Panel vectors are resolved fresh per call — the Mode-B ring buffer moves the
				// input's column offset between iterations, so a cached pointer would read the
				// wrong half of the ring.
				dst.PeerCopy2D(
					slab.Slice(bounds[src], n*cw-bounds[src]),
					pd.PartVector(in.V, src).Slice(c0*rows, cw*rows),
					pd.PartBackend(src),
					rows, cw, n, rows)
			}
		})
	}
	wg.Wait()
}

// satRowBands derives each partition's 3h1p row band from the SAME bounds the panels were
// allocated against. A partition may legitimately straddle the main/satellite boundary (normal
// for partition 0, main being tiny relative to n), so the band is the clamped intersection of
// [bound[d], bound[d+1]) with [main, n) expressed satellite-relative; a partition lying wholly
// inside the main block yields an empty band and must be skipped, not launched.
//
// Unlike the dense blocks — which route through distVec.Slice and get its crossing-boundary
// panic — this path hands raw offsets to the launcher, so nothing downstream would catch a bad
// band. Hence the derivation lives in one place and is unit-tested (TestSatRowBands).
func satRowBands(bounds []int, main, n int) (lo, hi []int) {
	nd := len(bounds) - 1
	lo, hi = make([]int, nd), make([]int, nd)
	for d := range nd {
		l := max(bounds[d], main) - main
		// max(l): an empty band when the partition lies entirely inside the main block.
		h := max(min(bounds[d+1], n)-main, l)
		lo[d], hi[d] = l, h
	}
	return lo, hi
}

// newSatelliteMatFreePerDevice builds the per-device on-device satellite applier: no host
// round-trip, no scatter. Each device stages a full-height input slab for a chunk of columns
// (gathered from every partition over NVLink) and runs the kernel over its own output band.
func (mx *Matrix) newSatelliteMatFreePerDevice(pd backend.PartitionedDevices) matFreePart {
	p := mx.buildSatScalarPlan()
	s := mx.buildSatDeviceSoA(p)
	n := mx.sp.Size()
	nd := pd.NumParts()
	bounds := pd.Bounds()
	rowLo, rowHi := satRowBands(bounds, s.main, n)

	// Latch the chunk width once: it sizes the slab allocated just below, so the apply loop
	// must not read a value someone changed afterwards.
	w := SatChunkCols
	checkSatChunkFits(w, n) // this applier is the one that launches dip_sat_apply

	// Per-device: the uploaded plan, and the staging slab for one column chunk.
	bufs := make([]*satDeviceBufs, nd)
	slab := make([]backend.Vector, nd)
	for d := range nd {
		dk, ok := pd.PartKernels(d)
		if !ok {
			panic("dip: per-device satellite apply selected without device kernels on every partition")
		}
		bufs[d] = uploadSatSoA(dk, s)
		slab[d] = pd.PartBackend(d).Alloc(n * w)
	}

	apply := func(in, out backend.BlockView) {
		for c0 := 0; c0 < in.Cols; c0 += w {
			cw := min(w, in.Cols-c0)

			// Fence the producers: the input panel may still be mid-write from an async panel
			// kernel, and a peer read would not drain that stream.
			syncAll(pd)

			// Gather: every device assembles the full-height slab for columns [c0, c0+cw).
			gatherSlabs(pd, func(d int) backend.Vector { return slab[d] }, in, bounds, n, c0, cw)

			// Fence the gather before any kernel reads a slab.
			syncAll(pd)

			// Concurrent for the same reason, and with the same disjointness argument, as the
			// batched applier: one blocking round-trip per device otherwise leaves the rest idle.
			var wg sync.WaitGroup
			for d := range nd {
				if rowHi[d] <= rowLo[d] {
					continue // partition owns no satellite rows
				}
				rd := bounds[d+1] - bounds[d]
				inView := backend.BlockView{V: slab[d], Rows: n, Cols: cw, Ld: n}
				outView := backend.BlockView{
					V:    pd.PartVector(out.V, d).Slice(c0*rd, cw*rd),
					Rows: rd, Cols: cw, Ld: rd,
				}
				args := bufs[d].args(s, inView, outView, rowLo[d], rowHi[d], bounds[d])
				dk := bufs[d].dk
				wg.Go(func() { dk.DipSatApply(args) })
			}
			wg.Wait()
		}
		// Fence the outputs before the caller consumes them.
		syncAll(pd)
	}

	release := func() {
		for d := range nd {
			bufs[d].free()
			pd.PartBackend(d).Free(slab[d])
		}
	}
	return matFreePart{apply: apply, release: release}
}

// ownerOf returns the partition holding global row r. Bounds are group-aligned
// (dip.PartitionBounds), so a whole block band never straddles two partitions.
func ownerOf(bounds []int, r int) int {
	for d := 0; d < len(bounds)-1; d++ {
		if r < bounds[d+1] {
			return d
		}
	}
	return len(bounds) - 2
}

// newSatBatchedPerDevice is the multi-GPU contraction path: the -mgpu twin of
// newSatBatchedDevice.
//
// It composes the two mechanisms already in place. From the per-scalar -mgpu path it keeps the
// chunked NVLink gather (each device stages a full-height n×w input slab, because a block's
// COLUMN band can live on any partition). From the batched path it keeps the plan and
// GemmMatBatched. What is new is the split: each planned batch is partitioned by which device
// owns its WRITE offset, and each device issues only its own share.
//
// That split is exact rather than approximate because PlanBatches already guarantees write
// offsets within a batch are distinct, and dip.PartitionBounds is group-aligned so a block's
// output band lies wholly inside one partition. So assigning a slot to ownerOf(writeOffset)
// partitions the batch without splitting any block.
func (mx *Matrix) newSatBatchedPerDevice(pd backend.PartitionedDevices) matFreePart {
	s := mx.buildSatDeviceSoA(mx.buildSatScalarPlan())
	n := mx.sp.Size()
	nd := pd.NumParts()
	bounds := pd.Bounds()
	w := SatChunkCols // latched at construction: it sizes the per-device slab allocated below

	type devState struct {
		be   backend.Backend // the sub-backend that issues this device's GemmMatBatched
		dk   backend.DeviceKernels
		bufs *jiiDeviceBufs
		plan *jiiBatchPlan
		slab backend.Vector
		eri  unsafe.Pointer
		eps  unsafe.Pointer
		osym unsafe.Pointer
		// members[b] lists the indices (into plan.batches[b].Blocks) this device owns.
		members [][]int
	}
	st := make([]*devState, nd)

	// One plan for the whole pool. It depends only on the Space and the virtual-symmetry groups,
	// so building it per device produced N byte-identical copies — ~10.5 GB of []jiiSlot each at
	// the production system's 82 M slots, ~84 GB of host RAM across 8 devices. Each device gets a clone that
	// shares the planning data and owns its own issue scratch (jiiBatchPlan.clone).
	basePlan := mx.buildJIIBatchPlan()

	for d := range nd {
		dk, ok := pd.PartKernels(d)
		if !ok {
			panic("dip: batched per-device satellite selected without device kernels")
		}
		plan := basePlan.clone()
		bufs := mx.buildJIIDeviceBufs(dk, plan, s)
		eri, eps, osym := dk.DeviceERI(s.eri), dk.UploadFloats(s.eps), dk.UploadInts(s.osym)
		bufs.args.ERI, bufs.args.Eps, bufs.args.OrbSym = eri, eps, osym

		// Precompute this device's share of every batch — the plan is apply-invariant, so the
		// ownership split is too.
		members := make([][]int, len(plan.batches))
		for bi, bt := range plan.batches {
			for _, si := range bt.Blocks {
				sl := plan.slots[si]
				off := sl.rowOff
				if bt.Trans {
					off = sl.colOff
				}
				if ownerOf(bounds, off) == d {
					members[bi] = append(members[bi], si)
				}
			}
		}
		st[d] = &devState{
			be: pd.PartBackend(d), dk: dk, bufs: bufs, plan: plan, members: members,
			slab: pd.PartBackend(d).Alloc(n * w),
			eri:  eri, eps: eps, osym: osym,
		}
	}

	apply := func(in, out backend.BlockView) {
		nchunkCols := (in.Cols + w - 1) / w
		for c0 := 0; c0 < in.Cols; c0 += w {
			cw := min(w, in.Cols-c0)
			tChunk := time.Now()

			t0 := time.Now()
			syncAll(pd) // producers may still be mid-write; a peer read does not drain them
			tSync1 := time.Since(t0)

			// Gather the full-height slab on every device (identical to the per-scalar path).
			t0 = time.Now()
			gatherSlabs(pd, func(d int) backend.Vector { return st[d].slab }, in, bounds, n, c0, cw)
			tGather := time.Since(t0)

			t0 = time.Now()
			syncAll(pd) // fence the gather before any kernel reads a slab
			tSync2 := time.Since(t0)

			// Run the devices CONCURRENTLY, as gatherSlabs and syncAll above already do. Every
			// call into a sub-backend blocks on a round-trip through that device's owning
			// goroutine, so a serial loop here idles seven GPUs while one works: job 14211868
			// measured the summed per-device GEMM time equal to this chunk's wall time, i.e. no
			// overlap at all.
			//
			// Safe because the per-device state is disjoint by construction — each devState owns
			// its backend, kernels, device buffers, slab, and (since the plan is shared read-only)
			// its own issue scratch via jiiBatchPlan.clone. Output bands are disjoint too:
			// PartitionBounds is group-aligned, so a block's output lies wholly inside one
			// partition. Concurrency changes WHEN a row is summed, never in what order, so this is
			// bit-exact.
			fills := make([]time.Duration, nd)
			gemms := make([]time.Duration, nd)
			nfills := make([]int, nd)
			stats := make([]satStats, nd)
			var wg sync.WaitGroup

			for d := range nd {
				ds := st[d]
				rd := bounds[d+1] - bounds[d]
				// Input: the full-height local slab. Output: this device's partition, rebased so
				// a block's global row offset addresses local storage. The satellite blocks total
				// hundreds of GB, so the fill runs in bounded chunks (fillAndRun) rather than
				// materializing the whole operator per column chunk — the 424 GB OOM (job 14026481).
				inView := backend.BlockView{V: ds.slab, Rows: n, Cols: cw, Ld: n}
				outLocal := backend.BlockView{
					V:    pd.PartVector(out.V, d).Slice(c0*rd, cw*rd),
					Rows: rd, Cols: cw, Ld: rd,
				}
				wg.Go(func() {
					fills[d], gemms[d], nfills[d], stats[d] = ds.plan.fillAndRun(
						ds.dk, ds.bufs.args, ds.be, inView, outLocal, ds.members, bounds[d])
				})
			}
			wg.Wait()

			var tFill, tGemm time.Duration
			var nFill int
			var agg satStats
			for d := range nd {
				tFill += fills[d]
				tGemm += gemms[d]
				nFill += nfills[d]
				agg.calls += stats[d].calls
				agg.members += stats[d].members
			}
			perCall := 0.0
			if agg.calls > 0 {
				perCall = float64(agg.members) / float64(agg.calls)
			}
			// gemm/fill/fillchunks/calls are SUMMED over the devices, which now run concurrently:
			// a summed time well above `wall` is the proof that they overlap (before the parallel
			// loop landed, summed gemm equalled wall exactly). members/call is the batching health
			// check — ~1 means chunking has shredded the plan's batches.
			satTracef("batched colchunk %d/%d cw=%d wall=%s | sync1=%s gather=%s sync2=%s fill=%s gemm=%s "+
				"| gemmcalls=%d members=%d members/call=%.1f (fillchunks=%d, summed over %d devices)",
				c0/w+1, nchunkCols, cw, time.Since(tChunk).Round(time.Millisecond),
				tSync1.Round(time.Millisecond), tGather.Round(time.Millisecond), tSync2.Round(time.Millisecond),
				tFill.Round(time.Millisecond), tGemm.Round(time.Millisecond),
				agg.calls, agg.members, perCall, nFill, nd)
		}
		syncAll(pd) // fence outputs before the caller reads them
	}

	release := func() {
		for d := range nd {
			st[d].bufs.free()
			st[d].dk.FreeDev(st[d].eri)
			st[d].dk.FreeDev(st[d].eps)
			st[d].dk.FreeDev(st[d].osym)
			pd.PartBackend(d).Free(st[d].slab)
		}
	}
	return matFreePart{apply: apply, release: release}
}

// newSatelliteMatFreeDistributed builds the gather-apply-scatter satellite applier over a
// row-partitioned backend — the portable fallback when the partitions have no device kernels or
// are not fully peered. The plan is backend-independent (built from the space + block physics);
// only the gather (Download) and scatter (AddPanel) touch the distribution.
func (mx *Matrix) newSatelliteMatFreeDistributed(pg backend.PanelScatterAdd) matFreePart {
	plan := mx.buildSatScalarPlan()
	n := mx.sp.Size()
	apply := func(in, out backend.BlockView) {
		cols := in.Cols
		xfull := mx.be.Download(in.V) // full n×cols column-major host panel
		yfull := make([]float64, n*cols)
		plan.applyHost(xfull, yfull, cols, n, n)
		pg.AddPanel(out.V, yfull)
	}
	return matFreePart{apply: apply, release: func() {}}
}
