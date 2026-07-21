package dip

import (
	"sync"

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
//     whole panel, contracts on CPU and uploads it back (~137 GB each way for melanin), which is
//     the bottleneck docs/dip_operator_memory.md flags as the Phase-C follow-up.
//
// TestSatelliteMatFreeDistributedEqualsDense validates the fallback over gonum sub-backends.

// satChunkCols is the column-chunk width of the per-device gather. Each device stages a
// full-height n×satChunkCols input slab, so the slab costs n·satChunkCols·8 bytes per device
// (melanin triplet, 64: ~7.6 GB).
//
// It must not be small: chunking repeats the element evaluation ceil(b/w) times. The column loop
// dominates a candidate element (~b ops of apply against ~30 to evaluate), so at w=64 the
// recompute overhead is ~15% of chunk work and falls as w grows — the trade is recompute against
// slab residency, and 64 is the conservative starting point pending the melanin budget measurement.
const satChunkCols = 64

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

	// Per-device: the uploaded plan, and the staging slab for one column chunk.
	bufs := make([]*satDeviceBufs, nd)
	slab := make([]backend.Vector, nd)
	for d := range nd {
		dk, ok := pd.PartKernels(d)
		if !ok {
			panic("dip: per-device satellite apply selected without device kernels on every partition")
		}
		bufs[d] = uploadSatSoA(dk, s)
		slab[d] = pd.PartBackend(d).Alloc(n * satChunkCols)
	}

	apply := func(in, out backend.BlockView) {
		for c0 := 0; c0 < in.Cols; c0 += satChunkCols {
			cw := min(satChunkCols, in.Cols-c0)

			// Fence the producers: the input panel may still be mid-write from an async panel
			// kernel, and a peer read would not drain that stream.
			syncAll(pd)

			// Gather: every device assembles the full-height slab for columns [c0, c0+cw).
			// Panel vectors are resolved fresh here — the Mode-B ring buffer moves the input's
			// column offset between iterations, so a cached pointer would read the wrong half.
			for d := range nd {
				dst := pd.PartBackend(d).(backend.PeerCopier)
				for src := range nd {
					rows := bounds[src+1] - bounds[src]
					if rows == 0 {
						continue
					}
					srcPart := pd.PartVector(in.V, src)
					// Columns [c0,c0+cw) of the source band, and the slab row-offset to that
					// partition's global rows. dstLd = n makes the slab full height.
					dst.PeerCopy2D(
						slab[d].Slice(bounds[src], n*cw-bounds[src]),
						srcPart.Slice(c0*rows, cw*rows),
						pd.PartBackend(src),
						rows, cw, n, rows)
				}
			}

			// Fence the gather before any kernel reads a slab.
			syncAll(pd)

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
				bufs[d].dk.DipSatApply(bufs[d].args(s, inView, outView, rowLo[d], rowHi[d], bounds[d]))
			}
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
