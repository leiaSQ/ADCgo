package lanczos

import (
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
	if err := writeCheckpoint(p, s); err != nil {
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
