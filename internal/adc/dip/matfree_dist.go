package dip

import (
	"fmt"
	"math"
	"os"
	"runtime/debug"
	"sort"
	"sync"
	"time"
	"unsafe"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// matfree_dist.go — matrix-free 3h1p↔3h1p satellite apply under the row-partitioned (-mgpu)
// backend. Composes the matrix-free satellite region with the distributed Mode-B solver so a
// whole-band DIP sector fits a multi-device node: the dense main/coupling blocks and the Krylov
// panels stay partitioned across the devices (distBackend.GemmMat), and the satellite region —
// the multi-TB memory hog that made -mgpu materialize densely — is recomputed instead of stored.
//
// Three appliers live here:
//
//   - newSatBatchedPerDevice (production, what -mgpu -matfree on selects): the batched
//     contraction path — per-device plan split, on-device fill, cuBLAS batched GEMM.
//   - newSatelliteMatFreePerDevice: the per-SCALAR ancestor of the above — each device recomputes
//     ONLY its own output row band, on-device, with dip_sat_apply. Currently UNREACHABLE: the two
//     places that could select it (newSatelliteMatFreeParts, then newSatelliteMatFreePart) are
//     both guarded by perDeviceSatelliteOK, and newSatelliteMatFreeParts already returns the
//     batched applier whenever that predicate holds — so by the time newSatelliteMatFreePart
//     re-tests it, it is false by construction. It is kept as the per-scalar reference the
//     batched path was derived from; anything that revives it must re-check that guard chain.
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

// goDevices runs body(d) for every partition concurrently and blocks until all have stopped.
//
// It exists for the panics. Every call into a sub-backend blocks on a round-trip through that
// device's owning goroutine, and gpuBackend.do re-raises a device fault (a failed cudaMalloc in
// a fill, a nonzero cudaGetLastError from a launch) as a panic on whichever goroutine called it.
// Raised on a bare wg.Go goroutine that panic has no path back to solveDIPSectorMGPU — it aborts
// the process, so a fault 30 h into a production mat-vec dies with a goroutine dump and, unless the
// block happened to have reached cp.Every, no checkpoint. The serial applier's equivalent fault
// unwound through apply -> ApplyBlock -> SolveLowMem, where the errInterrupted / checkpoint
// handling lives. Recovering per device and re-raising on the CALLER's goroutine restores that.
//
// Every device is allowed to stop before anything is re-raised: a sibling still writing to a
// slab that a unwinding goroutine is about to free would be a use-after-free. The lowest device
// index wins so a reproducible fault reports reproducibly; the others are logged, not lost.
func goDevices(nd int, body func(d int)) {
	panics := make([]any, nd)
	stacks := make([][]byte, nd)
	var wg sync.WaitGroup
	for d := range nd {
		wg.Go(func() {
			defer func() {
				if r := recover(); r != nil {
					panics[d], stacks[d] = r, debug.Stack()
				}
			}()
			body(d)
		})
	}
	wg.Wait()

	first := -1
	for d := range nd {
		if panics[d] == nil {
			continue
		}
		if first < 0 {
			first = d
			continue
		}
		fmt.Fprintf(os.Stderr, "dip: partition %d also failed: %v\n%s\n", d, panics[d], stacks[d])
	}
	if first >= 0 {
		// Re-raise the original value, not a wrapper: any type-based handling upstream still
		// sees what the device backend raised. The partition and its worker stack — which the
		// re-panic's own stack no longer shows — go to stderr first.
		fmt.Fprintf(os.Stderr, "dip: partition %d failed:\n%s\n", first, stacks[first])
		panic(panics[first])
	}
}

// syncAll drains every partition's device stream concurrently. A peer read does not synchronize
// the source stream, so the gather must be fenced on both sides: after the producers have written
// the input, and after the kernels have written their output bands. The syncs are issued in
// parallel because each is a blocking round-trip through that device's owning goroutine —
// serialized over 8 devices, twice per apply, that is pure added latency.
func syncAll(pd backend.PartitionedDevices) {
	goDevices(pd.NumParts(), func(d int) {
		if pc, ok := pd.PartBackend(d).(backend.PeerCopier); ok {
			pc.Sync()
		}
	})
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
// active[d] false marks a partition that consumes no slab — one whose satellite row band is
// empty, or which owns no block in any batch. Staging for it is pure waste: nd-1 peer transfers
// of the full n x cw slab per column chunk (~7.6 GB per chunk at the production system's n and w=64, over 27
// chunks per mat-vec) into a buffer nothing ever reads. The kernel loops already skip these
// partitions; the gather has to be told. nil means every partition is active.
//
// Copies only — no arithmetic is reordered, so the gathered slab is bit-identical to the serial
// version regardless of completion order.
func gatherSlabs(pd backend.PartitionedDevices, slabOf func(int) backend.Vector,
	in backend.BlockView, bounds []int, n, c0, cw int, active []bool) {
	nd := pd.NumParts()
	// Assert on this goroutine, not inside a worker: this applier is only selected when every
	// partition is a peered PeerCopier, so a failure here is a selection bug and should not
	// surface as a panic from a device goroutine.
	dsts := make([]backend.PeerCopier, nd)
	slabs := make([]backend.Vector, nd)
	for d := range nd {
		if active != nil && !active[d] {
			continue
		}
		dsts[d] = pd.PartBackend(d).(backend.PeerCopier)
		slabs[d] = slabOf(d)
	}
	goDevices(nd, func(d int) {
		if dsts[d] == nil {
			return // stages nothing
		}
		for src := range nd {
			rows := bounds[src+1] - bounds[src]
			if rows == 0 {
				continue
			}
			// Panel vectors are resolved fresh per call — the Mode-B ring buffer moves the
			// input's column offset between iterations, so a cached pointer would read the
			// wrong half of the ring.
			dsts[d].PeerCopy2D(
				slabs[d].Slice(bounds[src], n*cw-bounds[src]),
				pd.PartVector(in.V, src).Slice(c0*rows, cw*rows),
				pd.PartBackend(src),
				rows, cw, n, rows)
		}
	})
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
//
// UNREACHABLE as wired today — newSatBatchedPerDevice takes every case that would reach it; see
// the file header for the guard chain. Kept correct and kept concurrent so it is not a trap for
// whoever re-wires it, not because it runs.
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

	// A partition lying wholly inside the main block owns no satellite row (satRowBands returns
	// an empty band) and never launches. It needs neither the uploaded SoA nor a slab: both are
	// device memory written every chunk and read never.
	active := make([]bool, nd)
	for d := range nd {
		active[d] = rowHi[d] > rowLo[d]
	}

	// Per-device: the uploaded plan, and the staging slab for one column chunk.
	//
	// Concurrent, for exactly the reason the batched twin's construction loop is
	// (newSatBatchedPerDevice): uploadSatSoA pushes the norb⁴ ERI tensor — 16 GB at the production system's
	// norb=212 — plus the 21-array config SoA, and Alloc reserves the n×w slab (~5.1 GB on the
	// production singlet at w=64), and every one of those calls blocks on a round-trip through that
	// device's owning goroutine. Serially that is eight 16 GB uploads one after another with
	// seven GPUs idle. goDevices, not parallel.Rows: Rows runs serially below 2*GOMAXPROCS rows
	// and nd=8 is far under that, and goDevices is what re-raises a failed cudaMalloc on the
	// caller's goroutine instead of aborting the process from a bare worker.
	//
	// Disjoint by construction — device d touches only its own sub-backend, its own kernels and
	// slot d of bufs/slab — and this is allocation and upload only, no floating-point reduction,
	// so it cannot move a bit of the result.
	bufs := make([]*satDeviceBufs, nd)
	slab := make([]backend.Vector, nd)
	goDevices(nd, func(d int) {
		if !active[d] {
			return
		}
		dk, ok := pd.PartKernels(d)
		if !ok {
			panic("dip: per-device satellite apply selected without device kernels on every partition")
		}
		bufs[d] = uploadSatSoA(dk, s)
		slab[d] = pd.PartBackend(d).Alloc(n * w)
	})

	apply := func(in, out backend.BlockView) {
		for c0 := 0; c0 < in.Cols; c0 += w {
			cw := min(w, in.Cols-c0)

			// Fence the producers: the input panel may still be mid-write from an async panel
			// kernel, and a peer read would not drain that stream.
			syncAll(pd)

			// Gather: every device assembles the full-height slab for columns [c0, c0+cw).
			gatherSlabs(pd, func(d int) backend.Vector { return slab[d] }, in, bounds, n, c0, cw, active)

			// Fence the gather before any kernel reads a slab.
			syncAll(pd)

			// Concurrent for the same reason, and with the same disjointness argument, as the
			// batched applier: one blocking round-trip per device otherwise leaves the rest idle.
			goDevices(nd, func(d int) {
				if !active[d] {
					return // partition owns no satellite rows
				}
				rd := bounds[d+1] - bounds[d]
				inView := backend.BlockView{V: slab[d], Rows: n, Cols: cw, Ld: n}
				outView := backend.BlockView{
					V:    pd.PartVector(out.V, d).Slice(c0*rd, cw*rd),
					Rows: rd, Cols: cw, Ld: rd,
				}
				args := bufs[d].args(s, inView, outView, rowLo[d], rowHi[d], bounds[d])
				bufs[d].dk.DipSatApply(args)
			})
		}
		// Fence the outputs before the caller consumes them.
		syncAll(pd)
	}

	release := func() {
		for d := range nd {
			if !active[d] {
				continue // nothing was allocated for it
			}
			bufs[d].free()
			pd.PartBackend(d).Free(slab[d])
		}
	}
	return matFreePart{apply: apply, release: release}
}

// ownerOf returns the partition holding global row r. Bounds are group-aligned
// (dip.PartitionBounds), so a whole block band never straddles two partitions.
//
// Binary search, not a scan: the ownership split calls this once per block reference per
// partition — nd·2S ≈ 1.3e9 times for a production sector at -mgpu 8 — and bounds is ascending by
// construction. sort.SearchInts(bounds, r+1)-1 is the index of the last bound ≤ r.
func ownerOf(bounds []int, r int) int {
	d := sort.SearchInts(bounds, r+1) - 1
	return min(max(d, 0), len(bounds)-2)
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
	}
	st := make([]*devState, nd)

	// One plan for the whole pool. It depends only on the Space and the virtual-symmetry groups,
	// so building it per device produced N byte-identical copies — ~10.5 GB of []jiiSlot each at
	// the production system's 82 M slots, ~84 GB of host RAM across 8 devices. Each device gets a clone that
	// shares the planning data and owns its own application order and issue scratch (cloneWith).
	basePlan := mx.buildJIIBatchPlan()

	// Split the batches by owner up front. The plan is apply-invariant, so the ownership split
	// is too — and settling it before the device loop is what lets a partition that owns no
	// block in any batch be skipped entirely: no plan clone, no ERI/eps/osym upload, no slab,
	// and no share of the gather. Such a partition never issued a GEMM in the first place.
	//
	// One pass over the block references, not one per device: ownerOf is evaluated once per
	// reference and the result selects the destination list, instead of nd passes each testing
	// every reference against one device (nd·2S ≈ 1.3e9 for the production system at -mgpu 8). Parallel over
	// BATCHES, so every worker owns memberOf[*][bi] for its own bi and no two ever touch the same
	// slice; within a batch the walk stays in bt.Blocks' ascending order, which appChunks and the
	// device fill both rely on.
	memberOf := make([][][]int, nd)
	for d := range nd {
		memberOf[d] = make([][]int, len(basePlan.batches))
	}
	parallel.HeavyRows(len(basePlan.batches), func(bi int) {
		bt := basePlan.batches[bi]
		for _, si := range bt.Blocks {
			sl := basePlan.slots[si]
			off := sl.rowOff
			if bt.Trans {
				off = sl.colOff
			}
			d := ownerOf(bounds, off)
			memberOf[d][bi] = append(memberOf[d][bi], si)
		}
	})

	// Each device fills ONLY the applications it will issue. The whole-plan fill this replaces
	// materialized all 2S blocks on every device and discarded the (nd-1)/nd it did not own —
	// nd-fold redundant work in the phase that is second-largest after the GEMM.
	//
	// Concurrent over devices, for the same reason as the goDevices loop two statements below:
	// this is ~1.64e8 block applications summed over the pool at production scale (~2.05e7 per
	// device at nd=8), each one a slot lookup plus a rows·cols multiply, and run serially the
	// whole pass sat on one core while construction stalled. HeavyRows, not parallel.Rows: Rows
	// falls back to a serial loop below 2*GOMAXPROCS rows and nd=8 is far below 256, so it would
	// have changed nothing.
	//
	// Disjoint by construction — device d reads only memberOf[d] (the ownership split above gave
	// each device its own member lists) plus the shared read-only basePlan, and writes only slot
	// d of appsOf/chunksOf/active. It is enumeration and bookkeeping, no floating-point
	// reduction, and appChunks walks each device's members in the same ascending order whatever
	// the schedule, so the emitted apps/chunks are bit-identical to the serial version.
	active := make([]bool, nd)
	appsOf := make([][]int32, nd)
	chunksOf := make([][]jiiChunk, nd)
	parallel.HeavyRows(nd, func(d int) {
		appsOf[d], chunksOf[d] = basePlan.appChunks(memberOf[d], JIIFillBudgetElems)
		active[d] = len(appsOf[d]) > 0
	})

	// Build the per-device state CONCURRENTLY. Each device uploads the norb⁴ ERI tensor (16 GB at
	// the production system's norb=212), the 21-array config SoA, the plan SoA and its n×w slab, and every one
	// of those calls blocks on a round-trip through that device's owning goroutine — so a serial
	// loop here uploads 8 GPUs one after another while 7 idle. The state is disjoint by
	// construction (own backend, own kernels, own buffers, own slab, own plan clone), and
	// goDevices recovers per device so a failed cudaMalloc still unwinds through the caller.
	goDevices(nd, func(d int) {
		if !active[d] {
			return
		}
		dk, ok := pd.PartKernels(d)
		if !ok {
			panic("dip: batched per-device satellite selected without device kernels")
		}
		plan := basePlan.cloneWith(appsOf[d], chunksOf[d])
		bufs := mx.buildJIIDeviceBufs(dk, plan, s)
		eri, eps, osym := dk.DeviceERI(s.eri), dk.UploadFloats(s.eps), dk.UploadInts(s.osym)
		bufs.args.ERI, bufs.args.Eps, bufs.args.OrbSym = eri, eps, osym

		st[d] = &devState{
			be: pd.PartBackend(d), dk: dk, bufs: bufs, plan: plan,
			slab: pd.PartBackend(d).Alloc(n * w),
			eri:  eri, eps: eps, osym: osym,
		}
	})

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
			gatherSlabs(pd, func(d int) backend.Vector { return st[d].slab }, in, bounds, n, c0, cw, active)
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
			goDevices(nd, func(d int) {
				ds := st[d]
				if ds == nil {
					return // owns no block in any batch
				}
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
				fills[d], gemms[d], nfills[d], stats[d] = ds.plan.fillAndRun(
					ds.dk, ds.bufs.args, ds.be, inView, outLocal, bounds[d])
			})

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
			if st[d] == nil {
				continue // skipped at construction: owns no block in any batch
			}
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
