package main

import (
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// Per-sector backend selection for -backend auto.
//
// Neither backend wins everywhere. Measured on this box (Ryzen 5600X + RTX 3060 Ti),
// blocks=200, both after the level-3 rewrite:
//
//	CH2O  (8 sectors, n ≈ 1–2k)   CPU  7.6 s   GPU 22.8 s
//	HCOOH (4 sectors, n ≈ 10–18k) CPU 630 s    GPU 294 s
//
// The crossover is a property of the hardware, not the molecule: consumer Ampere runs
// FP64 at 1/64 of FP32, so this card's DGEMM is only ~1.6–2× the CPU's, while a
// datacenter card is 40–60×. A size threshold tuned here would be wrong there. So the
// chooser calibrates each backend once and then, per sector, adds a *measured* apply time
// to the modelled GEMM/eigensolver time.
//
// Measuring apply is not a refinement, it is the point. A flop-based estimate of the
// mat-vec gets the sign of the CH2O comparison wrong: the GPU's apply is launch-bound on
// small sectors (thousands of tiny batched GEMMs) and bandwidth-bound on large ones, and
// only the second regime favours the device.
type chooser struct {
	cands   []candidate
	verbose bool

	// pool holds one backend per visible device when a homogeneous GPU backend was
	// pinned (-backend cuda|hip on a multi-GPU node). len(pool) >= 2 enables concurrent
	// per-sector dispatch (one GPU per sector); nil/len<2 means run sectors serially
	// through cands (the single-backend or -backend auto cost-picking path).
	pool []backend.Backend
}

type candidate struct {
	name string
	be   backend.Backend
	perf backend.Perf
}

// applyProbe assembles the sector's operator on be, times one ApplyBlock, and releases it.
// Returns the per-block-iteration apply cost.
type applyProbe func(be backend.Backend) time.Duration

// newChooser resolves cfgName into a chooser. Any name other than "auto" pins that
// backend and runs no calibration; on a multi-GPU node it binds one instance per
// visible device (capped by maxGPUs, 0 = all) so independent sectors run concurrently,
// one per GPU. "auto" keeps the per-sector cost-picking path (single device per
// backend); "auto" with a single available backend short-circuits.
func newChooser(cfgName string, verbose bool, maxGPUs int) (*chooser, error) {
	if cfgName != "auto" {
		bes, err := backend.NewAll(cfgName, maxGPUs)
		if err != nil {
			return nil, err
		}
		// One representative candidate drives the serial/single path (single() short-
		// circuits pickLanczos/pickDense — no probe, no calibration). When more than one
		// device is visible, pool holds them all for concurrent per-sector dispatch.
		c := &chooser{cands: []candidate{{name: cfgName, be: bes[0]}}, verbose: verbose}
		if len(bes) > 1 {
			c.pool = bes
			if verbose {
				fmt.Fprintf(os.Stderr, "dispatch %s: %d devices, sectors run concurrently (one per GPU)\n",
					cfgName, len(bes))
			}
		}
		return c, nil
	}

	names := backend.Available()
	if len(names) == 1 {
		be, err := backend.New(names[0])
		if err != nil {
			return nil, err
		}
		return &chooser{cands: []candidate{{name: names[0], be: be}}, verbose: verbose}, nil
	}

	c := &chooser{verbose: verbose}
	for _, name := range names {
		be, err := backend.New(name)
		if err != nil {
			return nil, err
		}
		perf := backend.Calibrate(be)
		c.cands = append(c.cands, candidate{name: name, be: be, perf: perf})
		if verbose {
			fmt.Fprintf(os.Stderr, "calibrate %-6s gemm=%6.1f GFLOP/s  symeig=%5.1f GFLOP/s\n",
				name, perf.GemmFlops/1e9, perf.EigFlops/1e9)
		}
	}
	return c, nil
}

// single reports whether there is nothing to choose between.
func (c *chooser) single() (backend.Backend, bool) {
	if len(c.cands) == 1 {
		return c.cands[0].be, true
	}
	return nil, false
}

// fits reports whether a device backend has room for the sector. Host backends always
// fit. The estimate is deliberately generous (see backend.SectorBytes), so a refusal
// means "definitely does not fit".
func (c *chooser) fits(cand candidate, n, b, dim int, label string) bool {
	dm, isDev := cand.be.(backend.DeviceMemory)
	if !isDev {
		return true
	}
	need := backend.SectorBytes(n, dim, b)
	const margin = 256 << 20
	free, _ := dm.DeviceMem()
	if free < need+margin {
		if c.verbose {
			fmt.Fprintf(os.Stderr, "dispatch %-18s skip %-6s (needs ~%.2f GB, %.2f GB free)\n",
				label, cand.name, float64(need)/(1<<30), float64(free)/(1<<30))
		}
		return false
	}
	return true
}

// checkDeviceFit verifies the chosen backend has room for a sector whose true resident
// device footprint is needBytes. Host backends always pass. It exists because the pinned-
// backend path (single(), below) skips the cost-picker's fits() check, and because the
// cost model's dense-n² operator term is meaningless at production sizes — the caller
// supplies the exact block-sparse operator size instead. On a device short of room it
// returns an actionable error rather than letting a later cudaMalloc panic tear down the run.
func (c *chooser) checkDeviceFit(label string, be backend.Backend, needBytes uint64) error {
	dm, isDev := be.(backend.DeviceMemory)
	if !isDev {
		return nil
	}
	const margin = 256 << 20
	free, _ := dm.DeviceMem()
	if free >= needBytes+margin {
		return nil
	}
	return fmt.Errorf("%s: sector needs ~%.1f GB of GPU memory (block-sparse operator + Lanczos panels) "+
		"but only %.1f GB is free on %s; submit on a larger-memory GPU (e.g. ADCGO_DIP_GRES=gpu:H200:2) "+
		"or run with -backend gonum",
		label, float64(needBytes)/(1<<30), float64(free)/(1<<30), c.cands[0].name)
}

// checkSubsFit verifies each sub-backend of a multi-GPU (row-partitioned) sector has room for
// its per-partition resident share needPerGPU. Host sub-backends always pass. It mirrors
// checkDeviceFit for the -mgpu path, which row-partitions across a set of sub-backends and has
// no single chooser device in hand; needPerGPU is an approximate per-GPU share (the exact
// per-partition operator split is not modeled), enough to catch a plainly-too-small GPU before
// a mid-assembly cudaMalloc panic.
func checkSubsFit(label string, subs []backend.Backend, needPerGPU uint64) error {
	const margin = 256 << 20
	for i, be := range subs {
		dm, isDev := be.(backend.DeviceMemory)
		if !isDev {
			continue
		}
		free, _ := dm.DeviceMem()
		if free < needPerGPU+margin {
			return fmt.Errorf("%s: partition %d/%d needs ~%.1f GB of GPU memory (operator row-band + Lanczos panels) "+
				"but only %.1f GB is free; use fewer -mgpu partitions on larger-memory GPUs or run with -backend gonum",
				label, i+1, len(subs), float64(needPerGPU)/(1<<30), float64(free)/(1<<30))
		}
	}
	return nil
}

// pickLanczos chooses the backend predicted fastest for one block-Lanczos sector.
// probe measures the real apply cost on each surviving candidate.
func (c *chooser) pickLanczos(label string, n, b, dim int, probe applyProbe) backend.Backend {
	if be, ok := c.single(); ok {
		return be
	}
	iters := float64(dim) / float64(b)

	best, bestT := -1, 0.0
	est := make([]float64, len(c.cands))
	for i, cand := range c.cands {
		if !c.fits(cand, n, b, dim, label) {
			est[i] = -1
			continue
		}
		apply := probe(cand.be).Seconds() * iters
		est[i] = apply + cand.perf.SolveSeconds(n, dim)
		if best < 0 || est[i] < bestT {
			best, bestT = i, est[i]
		}
	}
	return c.commit(label, n, dim, best, est)
}

// pickDense chooses for the dense path, which is one SymEig of the whole sector; there is
// no mat-vec to measure.
func (c *chooser) pickDense(label string, n int) backend.Backend {
	if be, ok := c.single(); ok {
		return be
	}
	best, bestT := -1, 0.0
	est := make([]float64, len(c.cands))
	for i, cand := range c.cands {
		if !c.fits(cand, n, 1, n, label) {
			est[i] = -1
			continue
		}
		est[i] = cand.perf.EigSeconds(n)
		if best < 0 || est[i] < bestT {
			best, bestT = i, est[i]
		}
	}
	return c.commit(label, n, n, best, est)
}

// commit resolves the winner, falling back to a host backend if every device refused on
// memory, and reports the decision.
func (c *chooser) commit(label string, n, dim, best int, est []float64) backend.Backend {
	if best < 0 {
		for i, cand := range c.cands {
			if _, isDev := cand.be.(backend.DeviceMemory); !isDev {
				best = i
				break
			}
		}
		if best < 0 {
			best = 0 // nothing else to offer
		}
	}
	if c.verbose {
		var alt string
		for i, cand := range c.cands {
			if i == best {
				continue
			}
			if est[i] < 0 {
				alt += fmt.Sprintf("  %s=skipped", cand.name)
			} else {
				alt += fmt.Sprintf("  %s=%.1fs", cand.name, est[i])
			}
		}
		fmt.Fprintf(os.Stderr, "dispatch %-18s n=%-6d dim=%-6d -> %-6s (%.1fs predicted)%s\n",
			label, n, dim, c.cands[best].name, est[best], alt)
	}
	return c.cands[best].be
}

// timeApplyBlock assembles op on be (the first ApplyBlock triggers assembly), then times a
// second one. The Nrm2 forces the device queue to drain: ApplyBlock's GEMMs are
// asynchronous, and timing the launches alone would report a fraction of the true cost —
// the same trap that made BenchmarkGemm report throughput above the card's FP64 peak.
func timeApplyBlock(be backend.Backend, n, b int, apply func(out, in backend.BlockView)) time.Duration {
	wbuf := be.Alloc(n * b)
	qbuf := be.Alloc(n * b)
	defer be.Free(wbuf)
	defer be.Free(qbuf)
	w := backend.BlockView{V: wbuf, Rows: n, Cols: b, Ld: n}
	q := backend.BlockView{V: qbuf, Rows: n, Cols: b, Ld: n}

	apply(w, q) // warm-up: assembles and uploads the operator
	_ = be.Nrm2(w.V)

	t0 := time.Now()
	apply(w, q)
	_ = be.Nrm2(w.V)
	return time.Since(t0)
}

// workerChoosers derives one single-backend chooser per pooled device. Each has a
// single candidate, so its pickLanczos/pickDense short-circuits (single()) to that
// device's backend — no cross-device probe, exactly what a per-GPU worker wants.
func (c *chooser) workerChoosers() []*chooser {
	out := make([]*chooser, len(c.pool))
	for i, be := range c.pool {
		out[i] = &chooser{
			cands:   []candidate{{name: fmt.Sprintf("%s#%d", c.cands[0].name, i), be: be}},
			verbose: c.verbose,
		}
	}
	return out
}

// runConcurrent dispatches nItems work items across the device pool, one item per free
// device at a time, and blocks until all complete. job runs on a worker's own single-
// device chooser and must write its result into caller-owned storage indexed by item
// (workers touch disjoint indices, so no locking is needed there). Output ordering is
// the caller's responsibility; runConcurrent imposes none. Returns the first error.
func (c *chooser) runConcurrent(nItems int, job func(w *chooser, item int) error) error {
	workers := c.workerChoosers()
	jobs := make(chan int)
	errs := make([]error, nItems)
	var wg sync.WaitGroup
	for _, w := range workers {
		wg.Add(1)
		go func(w *chooser) {
			defer wg.Done()
			for i := range jobs {
				errs[i] = job(w, i)
			}
		}(w)
	}
	for i := 0; i < nItems; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	for _, e := range errs {
		if e != nil {
			return e
		}
	}
	return nil
}
