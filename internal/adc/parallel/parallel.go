// Package parallel provides small goroutine work-pool helpers shared by the ADC
// block-assembly code (sip, dip).
//
// Assembling the ADC blocks is heavy integral arithmetic per element over a large
// configuration space — the dominant cost of a build, and pure Go (no BLAS). It
// runs in the assemble() phase, before the block-Lanczos iterations, so it does
// not overlap the threaded GEMM solve: parallelizing it fills cores that would
// otherwise sit idle and does not oversubscribe OpenBLAS/cuBLAS (the "separate
// phase" case). The integral store, orbital energies and symmetry data are
// immutable after construction, so concurrent reads are safe; each work item must
// write a disjoint output region (one matrix row / one config's / one group's
// elements).
package parallel

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// Rows runs body(r) for r in [0,rows) across up to GOMAXPROCS workers. body must
// be safe for concurrent calls on distinct r — it must only write output cells
// owned by row r (or otherwise-disjoint storage). For small row counts it runs
// serially to avoid goroutine overhead.
func Rows(rows int, body func(r int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || rows < 2*workers {
		for r := range rows {
			body(r)
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Go(func() {
			for {
				r := int(next.Add(1)) - 1
				if r >= rows {
					return
				}
				body(r)
			}
		})
	}
	wg.Wait()
}

// ChunkWorkers is the worker count Chunks will use for n items: GOMAXPROCS,
// capped at n and floored at 1. Exposed so a caller can pre-size per-worker
// accumulators before the parallel region.
func ChunkWorkers(n int) int {
	w := runtime.GOMAXPROCS(0)
	if w < 1 {
		w = 1
	}
	if w > n {
		w = n
	}
	if w < 1 {
		w = 1
	}
	return w
}

// Chunks statically partitions [0,n) into `workers` contiguous ranges and runs
// body(worker, lo, hi) once per range, each on its own goroutine. Unlike Rows the
// partition is fixed (not work-stealing), so a reduction over per-worker
// accumulators indexed by `worker` is deterministic run-to-run. Pass
// workers = ChunkWorkers(n). body(worker, lo, hi) must write only storage owned
// by items [lo,hi) or by its own worker slot. Runs serially when workers <= 1.
func Chunks(n, workers int, body func(worker, lo, hi int)) {
	if workers <= 1 || n == 0 {
		if n > 0 {
			body(0, 0, n)
		}
		return
	}
	var wg sync.WaitGroup
	for w := range workers {
		lo, hi := w*n/workers, (w+1)*n/workers
		wg.Go(func() { body(w, lo, hi) })
	}
	wg.Wait()
}
