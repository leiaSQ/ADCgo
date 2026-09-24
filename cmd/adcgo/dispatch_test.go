package main

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestConcurrentDIPSectorsShareReadOnlyState exercises the data-safety claim behind the
// multi-GPU sector loops: multiple sectors solved concurrently share one integrals.Store
// and eps slice, which must be read-only. It runs the real per-sector body
// (solveDIPSector) from several goroutines on host backends and compares each concurrent
// result to a serial reference. Run under `go test -race ./cmd/adcgo/` to catch races in
// the shared state; on a multi-GPU node the same body runs on per-device GPU backends.
func TestConcurrentDIPSectorsShareReadOnlyState(t *testing.T) {
	d, err := fcidump.ReadFile("../../testdata/h2o_dzp.fcidump")
	if err != nil {
		t.Skipf("fixture not available: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	orbSym, syms, err := selectSymmetry("all", d)
	if err != nil {
		t.Fatal(err)
	}
	ints := integrals.New(d, nocc, orbSym) // shared read-only across goroutines

	cfg := dipConfig{solver: "lanczos", blocks: 30}
	opts := analyze.Options{}

	// Enumerate the non-empty (spin, irrep) sectors.
	type item struct {
		spin      dip.Spin
		targetSym int
		sp        *dip.Space
	}
	var items []item
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		for _, ts := range syms {
			sp := dip.NewSpace(nocc, d.NORB, orbSym, ts, spin)
			if sp.Size() == 0 {
				continue
			}
			items = append(items, item{spin, ts, sp})
		}
	}
	if len(items) < 2 {
		t.Skip("need >=2 sectors to test concurrency")
	}

	// Serial reference: one backend, one sector at a time.
	refBE, _ := backend.New("gonum")
	want := make([]float64, len(items))
	for i, it := range items {
		sec, err := solveDIPSector(&chooser{cands: []candidate{{name: "gonum", be: refBE}}},
			cfg, it.sp, ints, eps, it.spin, it.targetSym, nil, opts)
		if err != nil {
			t.Fatalf("sector %d: %v", i, err)
		}
		want[i] = firstEnergy(sec)
	}

	// Concurrent: each goroutine owns its own host backend (mirrors one-GPU-per-worker),
	// all sharing ints/eps. Results must match the serial reference exactly.
	got := make([]float64, len(items))
	var wg sync.WaitGroup
	for i, it := range items {
		wg.Add(1)
		go func(i int, it item) {
			defer wg.Done()
			be, _ := backend.New("gonum")
			sec, err := solveDIPSector(&chooser{cands: []candidate{{name: "gonum", be: be}}},
				cfg, it.sp, ints, eps, it.spin, it.targetSym, nil, opts)
			if err != nil {
				t.Errorf("sector %d: %v", i, err)
				return
			}
			got[i] = firstEnergy(sec)
		}(i, it)
	}
	wg.Wait()

	for i := range items {
		if got[i] != want[i] {
			t.Errorf("sector %d: concurrent E0=%v, serial E0=%v", i, got[i], want[i])
		}
	}
}

// firstEnergy returns the lowest state's energy of a sector, or 0 if empty.
func firstEnergy(s analyze.Sector) float64 {
	if len(s.States) == 0 {
		return 0
	}
	return s.States[0].EnergyEV
}

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
