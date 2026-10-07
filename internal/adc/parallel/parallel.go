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
//
// That fallback assumes a CHEAP body: it trades away parallelism whenever there are
// fewer than 2*GOMAXPROCS rows, which on a 128-core node means anything under 256.
// When one body call is expensive — a block build, a device upload, an iterative
// solve — the trade is inverted and HeavyRows is the one to use.
func Rows(rows int, body func(r int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || rows < 2*workers {
		for r := range rows {
			body(r)
		}
		return
	}
	stealRows(rows, workers, body)
}

// HeavyRows is Rows without the small-count serial fallback: it parallelizes whenever
// there is more than one row and more than one core, capping the workers at rows.
//
// Use it when a single body call is costly enough that goroutine overhead cannot matter.
// Rows' heuristic reads "few items means the overhead dominates", which is exactly wrong
// for the coarse work in this tree: an irrep's orbital count, a sector's group count and a
// satellite block list are all in the tens-to-low-hundreds, while one item is millions of
// flops or a device round-trip. Under Rows a 128-core node ran those on one core.
//
// Same contract as Rows — body must own row r's output — and the same determinism argument:
// the work is stolen in arbitrary order, so the schedule must not affect the result.
func HeavyRows(rows int, body func(r int)) {
	workers := min(runtime.GOMAXPROCS(0), rows)
	if workers <= 1 {
		for r := range rows {
			body(r)
		}
		return
	}
	stealRows(rows, workers, body)
}

// stealRows runs body over [0,rows) on `workers` goroutines that pull the next index off a
// shared counter, so an uneven cost per row balances itself rather than stranding one worker
// on the expensive tail.
func stealRows(rows, workers int, body func(r int)) {
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

// FixedPool runs many small parallel rounds on persistent goroutines.
//
// Chunks spawns a goroutine per range per call. That is invisible when a call does milliseconds
// of work and ruinous when it does microseconds: measured on a 64-core Helix node, one Chunks
// round over 64 ranges costs 80 us. The banded eigensolver's band→tridiagonal reduction has to
// synchronize once per chase, ~9.5e8 times at the production DIP shape, so 80 us a round is
// 21 hours of pure overhead against ~2 hours of actual work. A pool whose workers are already
// running and spin on a round counter is the only way that arithmetic works.
//
// HOW THE BARRIER IS BUILT, because the obvious version is far slower than Chunks. Two things
// cost milliseconds rather than microseconds at 64 threads and both are avoided here:
//
//   - runtime.Gosched() in the wait loop. Yielding requeues the goroutine and sets off
//     work-stealing across every P; with 63 waiters that dominates everything. The loops below
//     spin on a plain atomic load, which on the sharers' side is a cache-line broadcast, and only
//     yield after spinYield iterations — long enough that it never happens in a healthy round and
//     still guarantees progress if the pool is oversubscribed.
//   - a single shared completion counter. 63 atomic decrements of one line serialize on that
//     line. Each worker instead publishes the round it finished into its OWN cache-line-padded
//     slot, and Run reads the 63 slots; reads do not invalidate.
//
// Every worker acknowledges every round, including rounds whose split leaves it nothing, so Run
// does not return until all of them have. That is what makes the plain body/n/w fields safe to
// overwrite for the next round. An earlier version let idle workers skip the acknowledgement, and
// Run could then republish n while such a worker was still reading it, handing one range the
// previous round's bounds — the race detector caught it as a wrong partition, not as a hang.
//
// A pool is only worth creating when rounds follow each other closely AND the caller owns the
// cores, because idle workers spin. Close it when the phase ends.
//
// The caller's body must be safe for concurrent calls on disjoint ranges, exactly as for Chunks,
// must tolerate an empty range, and must not depend on how [0,n) is split — Run varies the split
// with n.
type FixedPool struct {
	workers int

	// Published under round; see the note above on why every worker acknowledges every round.
	body func(lo, hi int)
	n    int
	w    int // workers that get a non-empty range this round, or -1 to stop

	round atomic.Uint64
	done  []poolSlot
	wg    sync.WaitGroup
}

// poolSlot is one worker's completion counter, padded to a cache line so that a worker publishing
// its progress does not invalidate its neighbours' slots.
type poolSlot struct {
	round atomic.Uint64
	_     [56]byte
}

// spinYield is how many polls a waiter makes before yielding. A healthy round completes in a few
// microseconds, far inside this, so the yield is a liveness escape for an oversubscribed pool
// rather than part of the normal path.
const spinYield = 1 << 12

// poolPause delays between polls of the round counter, so a waiter loads it thousands of times a
// millisecond rather than millions. It costs at most a few tens of nanoseconds of wake latency
// against a barrier that takes microseconds, and it buys two things:
//
//   - the round counter's cache line is not hammered by every waiter at full rate;
//   - `go test -race` stays usable. The detector instruments every atomic load, and a bare spin
//     loop issues them as fast as the core can retire them; with the eigensolver driving ~1e6
//     rounds per test the package timed out at 14 minutes before this existed.
//
// The state is threaded through a pointer so the arithmetic cannot be optimized away.
func poolPause(state *uint32) {
	x := *state
	for range 64 {
		x = x*1664525 + 1013904223
	}
	*state = x
}

// fixedPoolMaxWorkers is the largest pool whose barrier is actually fast, and FixedPoolWorkers is
// how a caller should size one.
//
// The barrier does not degrade gracefully — it falls off a cliff. Measured on a 64-core Helix node
// (2 sockets x 32 cores, GOMAXPROCS=64 throughout, so the runtime always had spare Ps):
//
//	workers   4      8      16     24      32       48       64
//	round     2.0us  2.4us  2.5us  10.1us  1.59ms   3.85ms   3.95ms
//
// Flat to 16, an order of magnitude worse by 24, and three orders worse by 32 — where the pool is
// far slower than the Chunks it exists to replace (72us at 64 ranges). The knee sits below this
// node's 32-core socket, and it is not false sharing in the benchmark body (each range writes its
// own cache line) nor starved Ps (48 were free). Whatever the mechanism, the consequence for a
// caller is the same: ask for more than this and every round costs milliseconds.
//
// So a phase built on a per-round barrier gets 16 cores, not the machine. Re-measure with
// BenchmarkFixedPoolRound before raising it — the knee is a property of the hardware and the Go
// runtime version, not of this package.
const fixedPoolMaxWorkers = 16

// FixedPoolWorkers is the pool size to use: GOMAXPROCS, capped at fixedPoolMaxWorkers.
func FixedPoolWorkers() int { return min(max(runtime.GOMAXPROCS(0), 1), fixedPoolMaxWorkers) }

// NewFixedPool starts workers-1 goroutines; the calling goroutine is worker 0 and takes a share of
// every round itself. One pool should serve every parallel region of a phase: a second pool's
// workers would spin against the first pool's.
func NewFixedPool(workers int) *FixedPool {
	// Never more workers than there are Ps. These workers spin, so asking for more than the
	// runtime can run at once does not add parallelism — it takes it away: the surplus spinners
	// contend for the same Ps as the workers doing the work, and every waiter then reaches its
	// yield budget and thrashes the scheduler. On the 4-CPU login cgroup this turned a 25-solve
	// test sweep that asks for 7 and 16 workers into minutes of Gosched churn.
	if p := max(runtime.GOMAXPROCS(0), 1); workers > p {
		workers = p
	}
	if workers < 1 {
		workers = 1
	}
	p := &FixedPool{workers: workers, done: make([]poolSlot, workers)}
	for id := 1; id < workers; id++ {
		p.wg.Go(func() { p.serve(id) })
	}
	return p
}

// Run splits [0,n) into contiguous ranges and calls body on each, then returns once every range
// has completed. grain is the fewest items a worker will be given, so a round with fewer than
// 2*grain items runs entirely on the caller and pays no barrier at all; it is per call because one
// pool serves regions whose items cost very different amounts.
func (p *FixedPool) Run(n, grain int, body func(lo, hi int)) {
	if n <= 0 {
		return
	}
	if grain < 1 {
		grain = 1
	}
	w := min(p.workers, max(1, n/grain))
	if w <= 1 || p.workers == 1 {
		body(0, n)
		return
	}
	p.body, p.n, p.w = body, n, w
	r := p.round.Add(1)
	lo, hi := split(n, w, 0)
	body(lo, hi)
	var pause uint32
	for id := 1; id < p.workers; id++ {
		for spins := 0; p.done[id].round.Load() < r; spins++ {
			if spins&(spinYield-1) == spinYield-1 {
				runtime.Gosched()
			}
			poolPause(&pause)
		}
	}
	p.body = nil
}

// Close stops the workers. The pool must not be used afterwards.
func (p *FixedPool) Close() {
	p.w = -1
	p.round.Add(1)
	p.wg.Wait()
}

func (p *FixedPool) serve(id int) {
	// Rounds are numbered from 1 and last counts them rather than being reloaded from the counter:
	// a worker may not be scheduled until after Run has already published a round, and if it took
	// the current value as "already seen" it would wait for the next round while Run waited for
	// its acknowledgement.
	var pause uint32
	for last := uint64(0); ; last++ {
		for spins := 0; p.round.Load() == last; spins++ {
			if spins&(spinYield-1) == spinYield-1 {
				runtime.Gosched()
			}
			poolPause(&pause)
		}
		if p.w < 0 {
			return
		}
		if id < p.w {
			lo, hi := split(p.n, p.w, id)
			p.body(lo, hi)
		}
		p.done[id].round.Store(last + 1)
	}
}

// split is Chunks' partition of [0,n) into w contiguous ranges, for range id.
func split(n, w, id int) (lo, hi int) { return id * n / w, (id + 1) * n / w }
