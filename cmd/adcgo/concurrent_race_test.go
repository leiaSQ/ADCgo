package main

import (
	"sync"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
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
		sec := solveDIPSector(&chooser{cands: []candidate{{name: "gonum", be: refBE}}},
			cfg, it.sp, ints, eps, it.spin, it.targetSym, nil, opts)
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
			sec := solveDIPSector(&chooser{cands: []candidate{{name: "gonum", be: be}}},
				cfg, it.sp, ints, eps, it.spin, it.targetSym, nil, opts)
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
