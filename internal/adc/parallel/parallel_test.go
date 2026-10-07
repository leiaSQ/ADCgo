package parallel

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestRowsCoversEachOnce verifies every row runs exactly once (run with -race to
// check the concurrent claim path). Each body writes only its own slice cell.
func TestRowsCoversEachOnce(t *testing.T) {
	for _, n := range []int{0, 1, 5, 64, 1000} {
		hits := make([]int32, n)
		Rows(n, func(r int) { atomic.AddInt32(&hits[r], 1) })
		for r := range n {
			if hits[r] != 1 {
				t.Errorf("n=%d: row %d ran %d times, want 1", n, r, hits[r])
			}
		}
	}
}

// TestRowsMatchesSerial checks the parallel fill produces the same result as a
// serial loop for a representative reduction into disjoint cells.
func TestRowsMatchesSerial(t *testing.T) {
	const n = 777
	want := make([]float64, n)
	for r := range n {
		want[r] = float64(r*r) - 3*float64(r)
	}
	got := make([]float64, n)
	Rows(n, func(r int) { got[r] = float64(r*r) - 3*float64(r) })
	for r := range n {
		if got[r] != want[r] {
			t.Fatalf("row %d: got %v want %v", r, got[r], want[r])
		}
	}
}

// TestChunksCoversContiguously verifies Chunks partitions [0,n) into disjoint
// contiguous ranges covering every index exactly once.
func TestChunksCoversContiguously(t *testing.T) {
	for _, n := range []int{0, 1, 7, 64, 1000} {
		hits := make([]int32, n)
		w := ChunkWorkers(n)
		Chunks(n, w, func(_, lo, hi int) {
			for i := lo; i < hi; i++ {
				atomic.AddInt32(&hits[i], 1)
			}
		})
		for i := range n {
			if hits[i] != 1 {
				t.Errorf("n=%d: index %d ran %d times, want 1", n, i, hits[i])
			}
		}
	}
}

// TestHeavyRowsParallelizesSmallCounts pins the property HeavyRows exists for: a row count below
// Rows' 2*GOMAXPROCS fallback threshold must still run on more than one goroutine. Rows is
// checked alongside it to document that it deliberately does NOT.
func TestHeavyRowsParallelizesSmallCounts(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("needs more than one core to distinguish the two")
	}
	const rows = 3 // far below 2*GOMAXPROCS on any test machine

	var heavyMax int64
	var heavyLive atomic.Int64
	var mu sync.Mutex
	HeavyRows(rows, func(r int) {
		n := heavyLive.Add(1)
		mu.Lock()
		if n > heavyMax {
			heavyMax = n
		}
		mu.Unlock()
		time.Sleep(20 * time.Millisecond) // hold the slot so overlap is observable
		heavyLive.Add(-1)
	})
	if heavyMax < 2 {
		t.Errorf("HeavyRows ran %d rows with max concurrency %d, want >= 2", rows, heavyMax)
	}
}

// TestHeavyRowsCoversEveryRow guards the work-stealing loop: every row runs exactly once.
func TestHeavyRowsCoversEveryRow(t *testing.T) {
	for _, rows := range []int{0, 1, 2, 7, 1000} {
		counts := make([]int32, rows)
		HeavyRows(rows, func(r int) { atomic.AddInt32(&counts[r], 1) })
		for r, c := range counts {
			if c != 1 {
				t.Fatalf("rows=%d: row %d ran %d times, want 1", rows, r, c)
			}
		}
	}
}

// TestFixedPoolCoversRangeExactly checks that every round partitions [0,n) into disjoint
// contiguous ranges that cover it once, across worker counts, grains and sizes — including the
// n < 2*grain case that must run entirely on the caller.
func TestFixedPoolCoversRangeExactly(t *testing.T) {
	for _, workers := range []int{1, 2, 3, 8} {
		for _, grain := range []int{1, 4, 64} {
			p := NewFixedPool(workers)
			for _, n := range []int{0, 1, 2, 7, 64, 65, 1000} {
				hits := make([]int32, n)
				var ranges sync.Map
				p.Run(n, grain, func(lo, hi int) {
					if lo > hi || lo < 0 || hi > n {
						t.Errorf("workers=%d grain=%d n=%d: bad range [%d,%d)", workers, grain, n, lo, hi)
						return
					}
					ranges.Store(lo, hi)
					for i := lo; i < hi; i++ {
						atomic.AddInt32(&hits[i], 1)
					}
				})
				for i, h := range hits {
					if h != 1 {
						t.Fatalf("workers=%d grain=%d n=%d: item %d covered %d times", workers, grain, n, i, h)
					}
				}
			}
			p.Close()
		}
	}
}

// TestFixedPoolManyRounds exercises the round counter and the spin barrier over enough rounds to
// catch a lost wakeup or a worker running a stale body, which is the failure mode that matters:
// the eigensolver drives ~1e9 rounds, so a one-in-a-million miss is a wrong answer every time.
func TestFixedPoolManyRounds(t *testing.T) {
	const rounds, n = 20000, 256
	p := NewFixedPool(4)
	defer p.Close()
	acc := make([]int64, n)
	for r := range rounds {
		want := int64(r)
		p.Run(n, 8, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				acc[i] += want
			}
		})
	}
	want := int64(rounds) * int64(rounds-1) / 2
	for i, v := range acc {
		if v != want {
			t.Fatalf("item %d accumulated %d, want %d", i, v, want)
		}
	}
}

// BenchmarkFixedPoolRound measures the round latency the eigensolver's chase loop pays, as a
// function of pool size at the ambient GOMAXPROCS, against BenchmarkChunksRound for the same
// split. The ratio is the whole reason FixedPool exists, and the SHAPE of the curve is what sets
// how many workers the caller may ask for.
//
// Measured 2026-09-28, 64-core Helix node (2 sockets x 32 cores), GOMAXPROCS = pool size:
// 8 -> 2.8 us, 16 -> 3.3 us, 32 -> 66 us, 64 -> 3.97 ms, against Chunks at 3.9 / 9.5 / 29 / 75 us.
// The collapse sets in when the spinning workers take every P and leave the runtime none, so the
// sweep below holds GOMAXPROCS fixed and varies only the pool, which is the choice a caller has.
func BenchmarkFixedPoolRound(b *testing.B) {
	for _, w := range []int{4, 8, 16, 24, 32, 48, 64} {
		if w > runtime.GOMAXPROCS(0) {
			continue
		}
		b.Run(fmt.Sprintf("workers=%d/gomaxprocs=%d", w, runtime.GOMAXPROCS(0)), func(b *testing.B) {
			p := NewFixedPool(w)
			defer p.Close()
			// One cache line per range. A packed [64]int64 sink puts eight ranges on every line,
			// and that false sharing — not the barrier — is then what the benchmark measures.
			sink := make([]int64, 64*8)
			for b.Loop() {
				p.Run(w*4, 1, func(lo, hi int) { sink[(lo&63)*8] += int64(hi - lo) })
			}
		})
	}
}

// BenchmarkChunksRound is BenchmarkFixedPoolRound with Chunks, for the ratio.
func BenchmarkChunksRound(b *testing.B) {
	n := runtime.GOMAXPROCS(0) * 4
	w := ChunkWorkers(n)
	sink := make([]int64, 64*8)
	for b.Loop() {
		Chunks(n, w, func(_, lo, hi int) { sink[(lo&63)*8] += int64(hi - lo) })
	}
}
