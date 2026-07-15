package main

import (
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// poolChooser builds a chooser whose pool is n host (gonum) backends, so the concurrent
// dispatch harness (workerChoosers/runConcurrent) can be exercised on CPU without a GPU.
// On real hardware the pool holds one gpuBackend per device instead; the harness logic
// under test — order-preserving result placement and error propagation — is identical.
func poolChooser(t *testing.T, n int) *chooser {
	t.Helper()
	pool := make([]backend.Backend, n)
	for i := range pool {
		be, err := backend.New("gonum")
		if err != nil {
			t.Fatal(err)
		}
		pool[i] = be
	}
	return &chooser{cands: []candidate{{name: "gonum", be: pool[0]}}, pool: pool}
}

func TestRunConcurrentOrderAndCoverage(t *testing.T) {
	ch := poolChooser(t, 4)
	const N = 37 // more items than workers -> the pool queues
	got := make([]int, N)
	var count int32
	err := ch.runConcurrent(N, func(w *chooser, i int) error {
		atomic.AddInt32(&count, 1)
		got[i] = i * i // workers touch disjoint indices
		return nil
	})
	if err != nil {
		t.Fatalf("runConcurrent: %v", err)
	}
	if int(count) != N {
		t.Fatalf("processed %d items, want %d", count, N)
	}
	for i := 0; i < N; i++ {
		if got[i] != i*i { // result[i] lands at index i regardless of completion order
			t.Fatalf("item %d: got %d, want %d", i, got[i], i*i)
		}
	}
}

func TestRunConcurrentErrorPropagates(t *testing.T) {
	ch := poolChooser(t, 3)
	err := ch.runConcurrent(10, func(w *chooser, i int) error {
		if i == 5 {
			return fmt.Errorf("boom at %d", i)
		}
		return nil
	})
	if err == nil {
		t.Fatal("expected the injected error to propagate")
	}
}

func TestWorkerChoosersOnePerDevice(t *testing.T) {
	ch := poolChooser(t, 3)
	ws := ch.workerChoosers()
	if len(ws) != 3 {
		t.Fatalf("got %d worker choosers, want 3", len(ws))
	}
	for i, w := range ws {
		if be, ok := w.single(); !ok || be != ch.pool[i] {
			t.Fatalf("worker %d not pinned to pool[%d]", i, i)
		}
	}
}
