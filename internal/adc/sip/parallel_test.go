package sip

import (
	"sync/atomic"
	"testing"
)

// TestParRowsCoversEachOnce verifies every row runs exactly once (run with -race to
// check the concurrent claim path). Each body writes only its own slice cell.
func TestParRowsCoversEachOnce(t *testing.T) {
	for _, n := range []int{0, 1, 5, 64, 1000} {
		hits := make([]int32, n)
		parRows(n, func(r int) { atomic.AddInt32(&hits[r], 1) })
		for r := range n {
			if hits[r] != 1 {
				t.Errorf("n=%d: row %d ran %d times, want 1", n, r, hits[r])
			}
		}
	}
}

// TestParRowsMatchesSerial checks the parallel fill produces the same result as a
// serial loop for a representative reduction into disjoint cells.
func TestParRowsMatchesSerial(t *testing.T) {
	const n = 777
	want := make([]float64, n)
	for r := range n {
		want[r] = float64(r*r) - 3*float64(r)
	}
	got := make([]float64, n)
	parRows(n, func(r int) { got[r] = float64(r*r) - 3*float64(r) })
	for r := range n {
		if got[r] != want[r] {
			t.Fatalf("row %d: got %v want %v", r, got[r], want[r])
		}
	}
}
