package sip

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// parallel.go — goroutine-parallel block assembly for CVS IP-ADC(4).
//
// Assembling the ADC(4) blocks is O(nvir⁴·…) integral arithmetic per element over a
// large 2h1p/3h2p space — the dominant cost of an order-4 build, and pure Go (no
// BLAS). It runs in the assemble() phase, before the block-Lanczos iterations, so
// it does not overlap the threaded GEMM solve: parallelizing it fills cores that
// would otherwise sit idle and does not oversubscribe OpenBLAS (see
// ADCgo_goroutines_deferred.md, the "separate phase" case the deferral did not
// cover). The integral store, orbital energies and symmetry data are immutable
// after construction, so concurrent reads are safe; each work item writes a
// disjoint output region (one matrix row / one config's elements).
//
// This is the only place the sip package spawns goroutines. Order 2/3 assembly is
// cheap and untouched.

// parRows runs body(r) for r in [0,rows) across up to GOMAXPROCS workers. body must
// be safe for concurrent calls on distinct r — it must only write output cells owned
// by row r (or otherwise-disjoint storage). For small row counts it runs serially to
// avoid goroutine overhead.
func parRows(rows int, body func(r int)) {
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

// chunkWorkers is the worker count parChunks will use for n items: GOMAXPROCS,
// capped at n and floored at 1. Exposed so a caller can pre-size per-worker
// accumulators before the parallel region.
func chunkWorkers(n int) int {
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

// parChunks statically partitions [0,n) into `workers` contiguous ranges and runs
// body(worker, lo, hi) once per range, each on its own goroutine. Unlike parRows the
// partition is fixed (not work-stealing), so a reduction over per-worker accumulators
// indexed by `worker` is deterministic run-to-run. Pass workers = chunkWorkers(n).
// body(worker, lo, hi) must write only storage owned by columns [lo,hi) or by its own
// worker slot. Runs serially when workers <= 1.
func parChunks(n, workers int, body func(worker, lo, hi int)) {
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
