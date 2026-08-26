package lanczos

import (
	"path/filepath"
	"runtime"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
)

// TestCheckpointAllocGrowth pins the allocation behaviour of the checkpoint writer, which is what
// OOM-killed the production SIP run (deterministic ~733 GB RSS after ~25 h, jobs 14040959/14075367).
//
// saveKrylov streams the basis off the backend in ckptBasisChunkCols-wide chunks. The question this
// test answers is whether the bytes allocated per checkpoint scale with the CURRENT BASIS WIDTH
// (dim) or stay bounded by one reusable chunk buffer. Downloading the whole basis every checkpoint
// is O(dim·n) per checkpoint and O(dim²·n) over a run — for the production system that is ~1 TB of large
// transient spans, which Go's scavenger returns to the OS only lazily, so RSS climbs until the
// cgroup kills the job. The allocation SEQUENCE is a deterministic function of n/main/blocks, which
// is why the observed peak reproduced to within 40 KB across two independent runs.
//
// The invariant asserted here — per-checkpoint allocation is bounded by a constant rather than
// growing with dim — is scale-free, so a synthetic basis exercises it without production-scale memory.
// No operator is needed: saveKrylov consumes a BlockView, so the basis can be allocated directly,
// which also keeps the test independent of any fixture's size.
func TestCheckpointAllocGrowth(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping checkpoint allocation-growth test in -short mode")
	}
	be := backend.Gonum{}

	// n is the (synthetic) sector size; the two widths are a factor of 8 apart so a writer that
	// downloads the whole basis allocates ~8x more at the wider one, while a writer reusing one
	// chunk buffer allocates about the same.
	const n = 20000
	const main = 64
	small := ckptBasisChunkCols
	large := small * 8

	path := filepath.Join(t.TempDir(), "growth.ckpt")

	alloced := func() uint64 { // cumulative bytes allocated; GC-independent
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.TotalAlloc
	}

	measure := func(dim int) uint64 {
		basisBuf := be.Alloc(n * dim)
		defer be.Free(basisBuf)
		basis := backend.BlockView{V: basisBuf, Rows: n, Cols: dim, Ld: n}
		tm := backend.NewMat(dim, dim)

		runtime.GC()
		before := alloced()
		if err := saveKrylov(be, path, basis, tm, n, main, dim, 1, dim, 0, main, 0); err != nil {
			t.Fatalf("saveKrylov(dim=%d): %v", dim, err)
		}
		return alloced() - before
	}

	aSmall := measure(small)
	aLarge := measure(large)
	t.Logf("bytes allocated per checkpoint (n=%d): dim=%d -> %.1f MiB, dim=%d -> %.1f MiB (%.1fx)",
		n, small, float64(aSmall)/(1<<20), large, float64(aLarge)/(1<<20),
		float64(aLarge)/float64(max(aSmall, 1)))

	// The basis payload goes straight to the file, so a bounded writer's allocation must not grow
	// proportionally with dim. 2x slack covers the T rows and bufio; the pre-fix full-basis
	// download grows ~8x here and fails this comfortably.
	if aLarge > 2*aSmall+(1<<20) {
		t.Errorf("checkpoint allocation grows with basis width: dim=%d allocated %d B, dim=%d allocated %d B "+
			"(~%.1fx). saveKrylov must reuse one chunk buffer instead of allocating a fresh one per chunk — "+
			"this is the production SIP OOM (see the test doc).",
			small, aSmall, large, aLarge, float64(aLarge)/float64(max(aSmall, 1)))
	}
}
