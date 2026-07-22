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

// TestCheckpointRoundTrip: writeCheckpoint → readCheckpoint reproduces the state exactly,
// including the raw float64 payloads.
func TestCheckpointRoundTrip(t *testing.T) {
	s := &ckptState{
		N: 7, Main: 2, Maxdim: 6, MaxBlocks: 3,
		Dim: 4, BlkStart: 2, BlkSize: 2, Iter: 1,
		Basis: []float64{1, -2, 3.5, 4, 5, -6.25, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28},
		T:     []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6},
	}
	if len(s.Basis) != s.N*s.Dim || len(s.T) != s.Dim*s.Dim {
		t.Fatalf("test setup: payload lengths inconsistent with dims")
	}
	p := filepath.Join(t.TempDir(), "rt.ckpt")
	// nil writeBasis = the in-memory form, writing s.Basis directly.
	if err := writeCheckpoint(p, s, len(s.Basis), nil); err != nil {
		t.Fatalf("writeCheckpoint: %v", err)
	}
	got, err := readCheckpoint(p)
	if err != nil {
		t.Fatalf("readCheckpoint: %v", err)
	}
	if !reflect.DeepEqual(s, got) {
		t.Fatalf("round-trip mismatch:\n want %+v\n got  %+v", s, got)
	}

	// A missing file is a fresh run, not an error.
	if st, err := readCheckpoint(filepath.Join(t.TempDir(), "absent")); err != nil || st != nil {
		t.Fatalf("missing checkpoint: got (%v, %v), want (nil, nil)", st, err)
	}

	// The guard rejects a mismatched problem.
	if got.matches(s.N+1, s.Main, s.Maxdim, s.MaxBlocks) {
		t.Error("matches() accepted a different N")
	}
	if !got.matches(s.N, s.Main, s.Maxdim, s.MaxBlocks) {
		t.Error("matches() rejected the correct dims")
	}
}

// stopAfter wraps an Operator and trips a stop flag after `after` ApplyBlock calls, so a real
// solve can be interrupted deterministically mid-build.
type stopAfter struct {
	Operator
	calls int
	after int
	stop  *atomic.Bool
}

func (s *stopAfter) ApplyBlock(out, in backend.BlockView) {
	s.Operator.ApplyBlock(out, in)
	s.calls++
	if s.calls >= s.after {
		s.stop.Store(true)
	}
}

// TestLanczosResumeMatches: a solve interrupted by a stop signal and resumed from its
// checkpoint reproduces the spectrum of an uninterrupted run to machine precision — the
// retained basis makes the continuation bit-reproducible.
func TestLanczosResumeMatches(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping block-Lanczos resume test in -short mode")
	}
	be := backend.Gonum{}
	const blocks = 24

	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		ref := Solve(buildH2O(t, spin), be, Options{MaxBlocks: blocks})

		p := filepath.Join(t.TempDir(), "resume.ckpt")
		stop := new(atomic.Bool)

		// Phase A: interrupt after a few blocks; a checkpoint must be left behind.
		a := Solve(&stopAfter{Operator: buildH2O(t, spin), after: 3, stop: stop},
			be, Options{MaxBlocks: blocks, Checkpoint: &Checkpoint{Path: p, Stop: stop}})
		if !a.Interrupted {
			t.Fatalf("spin %d: phase A did not interrupt", spin)
		}
		if st, err := readCheckpoint(p); err != nil || st == nil {
			t.Fatalf("spin %d: no checkpoint after interrupt (%v)", spin, err)
		}

		// Phase B: resume from the checkpoint to completion.
		got := Solve(buildH2O(t, spin), be, Options{MaxBlocks: blocks, Checkpoint: &Checkpoint{Path: p}})
		if got.Interrupted {
			t.Fatalf("spin %d: resumed solve reported interrupted", spin)
		}
		if len(got.Values) != len(ref.Values) {
			t.Fatalf("spin %d: resumed %d values, reference %d", spin, len(got.Values), len(ref.Values))
		}
		const tol = 1e-10
		for k := range ref.Values {
			if de := abs(got.Values[k] - ref.Values[k]); de > tol {
				t.Errorf("spin %d: value %d resume=%.12f ref=%.12f Δ=%.2e", spin, k, got.Values[k], ref.Values[k], de)
			}
			if dp := abs(got.PS[k] - ref.PS[k]); dp > tol {
				t.Errorf("spin %d: ps %d resume=%.12f ref=%.12f Δ=%.2e", spin, k, got.PS[k], ref.PS[k], dp)
			}
		}

		// A completed solve removes its checkpoint.
		if st, err := readCheckpoint(p); err != nil || st != nil {
			t.Errorf("spin %d: checkpoint not cleaned up after completion (%v, %v)", spin, st, err)
		}
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// TestSaveKrylovStreamMatchesInMemory pins the chunked basis write against the whole-basis one:
// saveKrylov streams the basis off the backend in ckptBasisChunkCols-wide pieces (so the host
// never holds the 48 GB melanin SIP basis at once), and the file it produces must be
// byte-identical to writing it in a single blob. It also crosses a chunk boundary — dim is set
// past ckptBasisChunkCols — so an off-by-one in the chunk loop cannot pass.
func TestSaveKrylovStreamMatchesInMemory(t *testing.T) {
	const n, main = 7, 2
	dim := ckptBasisChunkCols + 3 // > one chunk, and not a multiple of the chunk width
	maxdim := dim + 5             // basis panel is wider than the populated part
	const maxBlocks, blkStart, blkSize, iter = 99, 4, 2, 3

	be := backend.Gonum{}
	host := make([]float64, n*maxdim)
	for i := range host {
		host[i] = float64(i%97) - 48.5
	}
	basis := backend.BlockView{V: be.Upload(host), Rows: n, Cols: maxdim, Ld: n}

	tm := backend.NewMat(maxdim, maxdim)
	for i := range tm.Data {
		tm.Data[i] = float64(i%13) * 0.25
	}

	streamed := filepath.Join(t.TempDir(), "stream.ckpt")
	if err := saveKrylov(be, streamed, basis, tm, n, main, maxdim, maxBlocks, dim, blkStart, blkSize, iter); err != nil {
		t.Fatalf("saveKrylov: %v", err)
	}

	// Reference: the same state written from a single in-memory basis slice.
	tHost := make([]float64, dim*dim)
	for i := range dim {
		copy(tHost[i*dim:(i+1)*dim], tm.Data[i*maxdim:i*maxdim+dim])
	}
	ref := filepath.Join(t.TempDir(), "ref.ckpt")
	s := &ckptState{
		N: n, Main: main, Maxdim: maxdim, MaxBlocks: maxBlocks,
		Dim: dim, BlkStart: blkStart, BlkSize: blkSize, Iter: iter,
		Basis: host[:n*dim], T: tHost,
	}
	if err := writeCheckpoint(ref, s, len(s.Basis), nil); err != nil {
		t.Fatalf("writeCheckpoint(ref): %v", err)
	}

	a, err := os.ReadFile(streamed)
	if err != nil {
		t.Fatalf("read streamed: %v", err)
	}
	b, err := os.ReadFile(ref)
	if err != nil {
		t.Fatalf("read ref: %v", err)
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("streamed checkpoint differs from in-memory one (%d vs %d bytes)", len(a), len(b))
	}

	// And it must still load back correctly through the normal path.
	got, err := readCheckpoint(streamed)
	if err != nil {
		t.Fatalf("readCheckpoint: %v", err)
	}
	if !got.matches(n, main, maxdim, maxBlocks) {
		t.Fatal("streamed checkpoint fails its own guard")
	}
	for i := range n * dim {
		if got.Basis[i] != host[i] {
			t.Fatalf("basis[%d] = %g, want %g", i, got.Basis[i], host[i])
		}
	}
}
