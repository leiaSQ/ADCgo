package lanczos

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
)

// sampleLowMemState builds a small but structurally realistic checkpoint state: three blocks with
// the invariants SolveLowMem maintains at the top of its loop — ascending offsets, the first
// block's β empty, and the LAST block's α unset (it is written partway through the iteration a
// checkpoint re-does).
func sampleLowMemState(n, b int) *lmCkptState {
	mk := func(rows, cols int, seed float64) backend.Mat {
		m := backend.NewMat(rows, cols)
		for i := range m.Data {
			m.Data[i] = seed + float64(i)
		}
		return m
	}
	alpha := func(size int, seed float64) []float64 {
		a := make([]float64, size*size)
		for i := range a {
			a[i] = seed + 0.5*float64(i)
		}
		return a
	}
	blocks := []lmBlock{
		{off: 0, size: b, alpha: alpha(b, 1), beta: backend.NewMat(0, 0)},
		{off: b, size: b, alpha: alpha(b, 2), beta: mk(b, b, 10)},
		{off: 2 * b, size: b, alpha: nil, beta: mk(b, b, 20)},
	}
	panel := make([]float64, n*2*b)
	for i := range panel {
		panel[i] = float64(i) * 0.25
	}
	return &lmCkptState{
		N: n, Main: b, B: b, Maxdim: 8 * b, MaxBlocks: 8, ModeB: true, DeflTol: 1e-8,
		Dim: 3 * b, RPrev: b, RCur: b, Iter: 2,
		Timing: Timing{Apply: 7, Orth: 11, Proj: 13, Eig: 17, Back: 19},
		Blocks: blocks, Panel: panel,
	}
}

// TestLowMemCheckpointRoundTrip is the twin of TestCheckpointRoundTrip for the low-memory format:
// everything written must come back identically, including the last block's deliberately-absent α
// and the timings carried across a daisychain.
func TestLowMemCheckpointRoundTrip(t *testing.T) {
	const n, b = 7, 2
	s := sampleLowMemState(n, b)
	p := filepath.Join(t.TempDir(), "lowmem.ckpt")

	if err := writeLowMemCheckpoint(p, s, len(s.Panel), nil); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := readLowMemCheckpoint(p, nil)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	// A nil alpha round-trips as a zero-length slice; normalize before comparing.
	for i := range got.Blocks {
		if len(got.Blocks[i].alpha) == 0 {
			got.Blocks[i].alpha = nil
		}
		if len(got.Blocks[i].beta.Data) == 0 {
			got.Blocks[i].beta = backend.NewMat(got.Blocks[i].beta.Rows, got.Blocks[i].beta.Cols)
			s.Blocks[i].beta = backend.NewMat(s.Blocks[i].beta.Rows, s.Blocks[i].beta.Cols)
		}
	}
	if !reflect.DeepEqual(got, s) {
		t.Errorf("round trip mismatch:\n got %+v\nwant %+v", got, s)
	}
	if !got.matches(n, b, b, 8*b, 8, true, 1e-8) {
		t.Errorf("round-tripped state fails its own guard")
	}
}

// TestLowMemCheckpointGuardRejects pins every mismatch the guard must catch. Each of these would
// otherwise resume a multi-day solve onto an incompatible problem.
func TestLowMemCheckpointGuardRejects(t *testing.T) {
	const n, b = 7, 2
	base := sampleLowMemState(n, b)

	cases := []struct {
		name                           string
		n, main, bb, maxdim, maxBlocks int
		modeB                          bool
		deflTol                        float64
		mutate                         func(s *lmCkptState)
	}{
		{name: "different n", n: n + 1, main: b, bb: b, maxdim: 8 * b, maxBlocks: 8, modeB: true, deflTol: 1e-8},
		{name: "different block width", n: n, main: b, bb: b + 1, maxdim: 8 * b, maxBlocks: 8, modeB: true, deflTol: 1e-8},
		{name: "different maxBlocks", n: n, main: b, bb: b, maxdim: 8 * b, maxBlocks: 9, modeB: true, deflTol: 1e-8},
		{name: "mode flipped", n: n, main: b, bb: b, maxdim: 8 * b, maxBlocks: 8, modeB: false, deflTol: 1e-8},
		{name: "different deflation tolerance", n: n, main: b, bb: b, maxdim: 8 * b, maxBlocks: 8, modeB: true, deflTol: 1e-10},
		{
			name: "dim disagrees with block sizes", n: n, main: b, bb: b, maxdim: 8 * b, maxBlocks: 8, modeB: true, deflTol: 1e-8,
			mutate: func(s *lmCkptState) { s.Dim = 99 },
		},
		{
			name: "last block already carries alpha", n: n, main: b, bb: b, maxdim: 8 * b, maxBlocks: 8, modeB: true, deflTol: 1e-8,
			mutate: func(s *lmCkptState) { s.Blocks[2].alpha = make([]float64, b*b) },
		},
		{
			name: "block count disagrees with iter", n: n, main: b, bb: b, maxdim: 8 * b, maxBlocks: 8, modeB: true, deflTol: 1e-8,
			mutate: func(s *lmCkptState) { s.Iter = 5 },
		},
		{
			name: "first block has a beta", n: n, main: b, bb: b, maxdim: 8 * b, maxBlocks: 8, modeB: true, deflTol: 1e-8,
			mutate: func(s *lmCkptState) { s.Blocks[0].beta = backend.NewMat(b, b) },
		},
	}
	for _, c := range cases {
		s := sampleLowMemState(n, b)
		if c.mutate != nil {
			c.mutate(s)
		}
		if s.matches(c.n, c.main, c.bb, c.maxdim, c.maxBlocks, c.modeB, c.deflTol) {
			t.Errorf("%s: guard accepted a checkpoint it must reject", c.name)
		}
	}
	// Sanity: the unmutated state against its own parameters must pass, or the cases above prove
	// nothing.
	if !base.matches(n, b, b, 8*b, 8, true, 1e-8) {
		t.Fatalf("baseline state rejected by its own guard")
	}
}

// TestSaveLowMemStreamMatchesInMemory pins the chunked panel write against the whole-panel one:
// saveLowMem streams pc off the backend in byte-budgeted chunks so the host never holds the whole
// panel (274 GB at production scale), and the file it produces must be byte-identical to writing the
// panel as one blob. The column count deliberately crosses a chunk boundary and does not divide
// evenly by it, so an off-by-one in the chunk loop cannot pass.
func TestSaveLowMemStreamMatchesInMemory(t *testing.T) {
	be := backend.Gonum{}
	const n, b = 5, 3

	// Force a chunk width of 2 columns so 2b = 6 columns spans three chunks unevenly.
	oldBudget := lmPanelChunkBytes
	lmPanelChunkBytes = 2 * n * 8
	defer func() { lmPanelChunkBytes = oldBudget }()
	if got := lmPanelChunkCols(n); got != 2 {
		t.Fatalf("chunk width %d, want 2 — the test would not cross a boundary", got)
	}

	s := sampleLowMemState(n, b)
	s.Maxdim, s.MaxBlocks = 8*b, 8

	buf := be.Upload(s.Panel)
	defer be.Free(buf)
	pc := backend.BlockView{V: buf, Rows: n, Cols: 2 * b, Ld: n}

	dir := t.TempDir()
	streamed := filepath.Join(dir, "streamed.ckpt")
	if err := saveLowMem(be, streamed, pc, s.Blocks, n, s.Main, b, s.Maxdim, s.MaxBlocks,
		true, s.DeflTol, s.Dim, s.RPrev, s.RCur, s.Iter, s.Timing); err != nil {
		t.Fatalf("saveLowMem: %v", err)
	}
	blob := filepath.Join(dir, "blob.ckpt")
	if err := writeLowMemCheckpoint(blob, s, len(s.Panel), nil); err != nil {
		t.Fatalf("writeLowMemCheckpoint: %v", err)
	}

	a, err := os.ReadFile(streamed)
	if err != nil {
		t.Fatal(err)
	}
	c, err := os.ReadFile(blob)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, c) {
		t.Errorf("streamed checkpoint (%d B) differs from the single-blob write (%d B)", len(a), len(c))
	}

	// And it must restore through the chunked reader onto a zeroed panel.
	dst := be.Alloc(n * 2 * b)
	defer be.Free(dst)
	dstView := backend.BlockView{V: dst, Rows: n, Cols: 2 * b, Ld: n}
	if st := loadLowMem(be, streamed, dstView, n, s.Main, b, s.Maxdim, s.MaxBlocks, true, s.DeflTol); st == nil {
		t.Fatal("loadLowMem rejected a checkpoint it had just written")
	}
	if got := be.Download(dst); !reflect.DeepEqual(got, s.Panel) {
		t.Errorf("restored panel differs from the saved one")
	}
}

// stopAfterLowMem trips a stop flag after `after` operator applies of EITHER kind.
//
// The existing stopAfter (checkpoint_test.go) cannot be reused here, for two independent reasons.
// It embeds Operator, but SolveLowMem selects Mode B with `op.(SatelliteOperator)` (lowmem.go),
// so a wrapper that only satisfies the narrower interface silently demotes the run to Mode A —
// the mode that does not checkpoint at all, so the test would pass while testing nothing. And
// from iter >= 2 Mode B calls ApplyBlockSatellite rather than ApplyBlock, so a counter on
// ApplyBlock alone stops advancing after the second block and never fires.
type stopAfterLowMem struct {
	SatelliteOperator
	calls int
	after int
	stop  *atomic.Bool
}

func (s *stopAfterLowMem) tick() {
	s.calls++
	if s.calls >= s.after {
		s.stop.Store(true)
	}
}

func (s *stopAfterLowMem) ApplyBlock(out, in backend.BlockView) {
	s.SatelliteOperator.ApplyBlock(out, in)
	s.tick()
}

func (s *stopAfterLowMem) ApplyBlockSatellite(out, in backend.BlockView) {
	s.SatelliteOperator.ApplyBlockSatellite(out, in)
	s.tick()
}

// TestLowMemResumeMatches: a Mode B solve interrupted by a stop signal and resumed from its
// checkpoint reproduces an uninterrupted run EXACTLY.
//
// Bit-equality, not a tolerance: the short recurrence reads only the [prev|cur] window and the
// accumulated α/β, all of which are restored verbatim, so on a deterministic backend the
// continuation is the same arithmetic in the same order. Anything less than exact equality means
// the restored state is not the state that was saved.
func TestLowMemResumeMatches(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping low-memory resume test in -short mode")
	}
	be := backend.Gonum{}
	const blocks = 12

	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		ref := SolveLowMem(buildH2O(t, spin), be, Options{MaxBlocks: blocks})

		p := filepath.Join(t.TempDir(), "resume.ckpt")
		stop := new(atomic.Bool)

		// Phase A: interrupt a few blocks in; a checkpoint must be left behind.
		mx := buildH2O(t, spin)
		a := SolveLowMem(&stopAfterLowMem{SatelliteOperator: mx, after: 4, stop: stop}, be,
			Options{MaxBlocks: blocks, Checkpoint: &Checkpoint{Path: p, Stop: stop}})
		if !a.Interrupted {
			t.Fatalf("spin %d: phase A did not interrupt", spin)
		}
		st, err := readLowMemCheckpoint(p, nil)
		if err != nil || st == nil {
			t.Fatalf("spin %d: no checkpoint after interrupt (%v)", spin, err)
		}
		if st.Iter == 0 {
			t.Fatalf("spin %d: checkpoint written at iter 0 — nothing was actually resumed", spin)
		}

		// Phase B: resume to completion.
		got := SolveLowMem(buildH2O(t, spin), be,
			Options{MaxBlocks: blocks, Checkpoint: &Checkpoint{Path: p}})
		if got.Interrupted {
			t.Fatalf("spin %d: resumed solve reported interrupted", spin)
		}
		if len(got.Values) != len(ref.Values) {
			t.Fatalf("spin %d: resumed %d values, reference %d", spin, len(got.Values), len(ref.Values))
		}
		for k := range ref.Values {
			if got.Values[k] != ref.Values[k] {
				t.Errorf("spin %d: value %d resume=%.17g ref=%.17g", spin, k, got.Values[k], ref.Values[k])
			}
			if got.PS[k] != ref.PS[k] {
				t.Errorf("spin %d: ps %d resume=%.17g ref=%.17g", spin, k, got.PS[k], ref.PS[k])
			}
		}

		// A completed solve removes its checkpoint.
		if st, err := readLowMemCheckpoint(p, nil); err != nil || st != nil {
			t.Errorf("spin %d: checkpoint not cleaned up after completion (%v, %v)", spin, st, err)
		}
	}
}

// TestLowMemCheckpointRejectsModeA: Mode A retains the whole basis on the host, so the
// short-recurrence format does not describe it. The combination must be refused rather than
// silently ignored — a run that believes it is checkpointing and is not is worse than one that
// knows it is not.
func TestLowMemCheckpointRejectsModeA(t *testing.T) {
	mx := buildH2O(t, dip.Singlet)
	main := mx.MainBlockSize()
	if main < 3 {
		t.Skipf("main block %d too small to select Mode A", main)
	}

	if err := LowMemCheckpointable(mx, Options{LowMemBlock: 0}); err != nil {
		t.Errorf("Mode B reported as not checkpointable: %v", err)
	}
	err := LowMemCheckpointable(mx, Options{LowMemBlock: main - 1})
	if err == nil {
		t.Fatal("Mode A reported as checkpointable")
	}
	t.Logf("Mode A correctly refused: %v", err)

	// And the driver itself refuses rather than running for days writing nothing.
	defer func() {
		if recover() == nil {
			t.Error("SolveLowMem accepted a checkpoint in Mode A instead of refusing")
		}
	}()
	p := filepath.Join(t.TempDir(), "modeA.ckpt")
	SolveLowMem(mx, backend.Gonum{}, Options{
		MaxBlocks: 4, LowMemBlock: main - 1, Checkpoint: &Checkpoint{Path: p},
	})
}

// TestLowMemCheckpointDistributedPanel exercises save/restore over the ROW-PARTITIONED backend,
// which is what -mgpu actually runs and what every other test in this file misses: they all pass
// backend.Gonum{} directly, so they only ever build a hostVec.
//
// That gap is not hypothetical. The production DIP probe (job 14158038) completed a 40 h block and
// then panicked in its first checkpoint with "interface conversion: backend.Vector is
// backend.distVec, not backend.hostVec" — distBackend embeds Gonum, so saveLowMem's
// be.(BufferedDownloader) assertion succeeded through the embedded method and handed a distVec to
// code that only handles host vectors. 40 h of compute lost to a path no test covered.
func TestLowMemCheckpointDistributedPanel(t *testing.T) {
	const n, b, main = 12, 2, 2 // n > 2·main²
	subs := []backend.Backend{backend.Gonum{}, backend.Gonum{}, backend.Gonum{}}
	be, err := backend.NewDistributed(subs, n, main, []int{0, 3, 7, n}) // uneven bands on purpose
	if err != nil {
		t.Fatalf("NewDistributed: %v", err)
	}

	s := sampleLowMemState(n, b)
	s.Main, s.Maxdim, s.MaxBlocks = main, 8*b, 8

	src := be.Upload(s.Panel)
	defer be.Free(src)
	pc := backend.BlockView{V: src, Rows: n, Cols: 2 * b, Ld: n}

	p := filepath.Join(t.TempDir(), "dist.ckpt")
	if err := saveLowMem(be, p, pc, s.Blocks, n, main, b, s.Maxdim, s.MaxBlocks,
		true, s.DeflTol, s.Dim, s.RPrev, s.RCur, s.Iter, s.Timing); err != nil {
		t.Fatalf("saveLowMem over distBackend: %v", err)
	}

	dst := be.Alloc(n * 2 * b)
	defer be.Free(dst)
	dstView := backend.BlockView{V: dst, Rows: n, Cols: 2 * b, Ld: n}
	st := loadLowMem(be, p, dstView, n, main, b, s.Maxdim, s.MaxBlocks, true, s.DeflTol)
	if st == nil {
		t.Fatal("loadLowMem rejected a checkpoint it had just written")
	}
	if st.Iter != s.Iter || st.Dim != s.Dim || st.RPrev != s.RPrev || st.RCur != s.RCur {
		t.Errorf("scalars differ: got iter=%d dim=%d rPrev=%d rCur=%d", st.Iter, st.Dim, st.RPrev, st.RCur)
	}
	if got := be.Download(dst); !reflect.DeepEqual(got, s.Panel) {
		t.Errorf("restored distributed panel differs from the saved one")
	}
}
