package parallel

import (
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
