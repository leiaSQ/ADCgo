package main

import (
	"fmt"
	"os"
	"time"

	"adcgo/internal/adc/backend"
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
}

type candidate struct {
	name string
	be   backend.Backend
	perf backend.Perf
}

// applyProbe assembles the sector's operator on be, times one ApplyBlock, and releases it.
// Returns the per-block-iteration apply cost.
type applyProbe func(be backend.Backend) time.Duration

// newChooser resolves cfgName into a chooser. Any name other than "auto" pins that one
// backend, and no calibration runs. "auto" with a single available backend likewise
// short-circuits: there is nothing to choose between, and calibration is not free.
func newChooser(cfgName string, verbose bool) (*chooser, error) {
	if cfgName != "auto" {
		be, err := backend.New(cfgName)
		if err != nil {
			return nil, err
		}
		return &chooser{cands: []candidate{{name: cfgName, be: be}}, verbose: verbose}, nil
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
