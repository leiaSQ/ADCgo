package lanczos

import (
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
)

// TestBandEigCheckpointResumeIsBitExact is the test that makes the reduction checkpoint worth
// having, and the only one that would catch a missing piece of saved state.
//
// It drives the reduction with the stop flag permanently set, so every call saves at the end of one
// column and gives up. Resuming in a loop therefore restarts the reduction at EVERY column
// boundary, and the final eigenvalues and eigenvector rows must come out bit-identical to a single
// uninterrupted run. Anything the snapshot forgets — a scaling vector, a band entry, the
// accumulator — shows up here as a differing bit; a tolerance-based check would hide exactly the
// small omissions that are easiest to make.
//
// It also proves progress is guaranteed: the stop is tested at the END of a column, so each
// generation advances at least one, and a chain of them cannot spin forever. That is the property
// the production daisychain depends on.
func TestBandEigCheckpointResumeIsBitExact(t *testing.T) {
	for _, tc := range []struct{ dim, band int }{{40, 7}, {60, 3}, {80, 12}} {
		seed := int64(7000 + tc.dim*100 + tc.band)
		main := (tc.band + 1) / 2

		// The uninterrupted reference.
		_, plain := buildBanded(tc.dim, tc.band, seed)
		wantD, wantZ, wantLD, stopped := bandSymDiagFastOpts(plain, bandEigOpts{topRows: main, botRows: main})
		if stopped {
			t.Fatalf("dim=%d band=%d: the uninterrupted run reported itself interrupted", tc.dim, tc.band)
		}

		path := filepath.Join(t.TempDir(), "reduction.eig")
		var stop atomic.Bool
		stop.Store(true)

		var gotD, gotZ []float64
		gotLD, gens := 0, 0
		for {
			gens++
			if gens > tc.dim+2 {
				t.Fatalf("dim=%d band=%d: %d generations without finishing — resume is not advancing",
					tc.dim, tc.band, gens)
			}
			_, bs := buildBanded(tc.dim, tc.band, seed)
			d, z, ld, interrupted := bandSymDiagFastOpts(bs, bandEigOpts{
				topRows: main, botRows: main,
				ckpt: &bandEigCkpt{path: path, stop: &stop},
			})
			if !interrupted {
				gotD, gotZ, gotLD = d, z, ld
				break
			}
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("dim=%d band=%d generation %d: reported interrupted but wrote no checkpoint: %v",
					tc.dim, tc.band, gens, err)
			}
		}
		if gens < 3 {
			t.Errorf("dim=%d band=%d: finished in %d generations; the loop cannot be exercising resume",
				tc.dim, tc.band, gens)
		}
		// A completed reduction must leave no restart point behind, or a rerun would resume a
		// finished computation.
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Errorf("dim=%d band=%d: checkpoint still present after the reduction completed (%v)",
				tc.dim, tc.band, err)
		}

		if gotLD != wantLD {
			t.Fatalf("dim=%d band=%d: resumed ld=%d, uninterrupted ld=%d", tc.dim, tc.band, gotLD, wantLD)
		}
		for k := range wantD {
			if gotD[k] != wantD[k] {
				t.Fatalf("dim=%d band=%d (%d generations): eval[%d] resumed %.17g, uninterrupted %.17g",
					tc.dim, tc.band, gens, k, gotD[k], wantD[k])
			}
		}
		for i := range wantZ {
			if gotZ[i] != wantZ[i] {
				t.Fatalf("dim=%d band=%d (%d generations): z[%d] resumed %.17g, uninterrupted %.17g",
					tc.dim, tc.band, gens, i, gotZ[i], wantZ[i])
			}
		}
	}
}

// TestBandEigCheckpointRoundTrip checks the binary format on its own: every field survives a write
// and a read, including the payloads and the fingerprint the guard turns on.
func TestBandEigCheckpointRoundTrip(t *testing.T) {
	want := &beCkptState{
		Dim: 6, Mb: 3, Top: 2, Bot: 2, Ld: 8,
		Fingerprint: 0xfeedfacecafebeef,
		K:           4,
		D:           []float64{1, 2, 3, 4, 5, 6},
		A:           make([]float64, 3*6),
		Z:           make([]float64, 8*6),
	}
	for i := range want.A {
		want.A[i] = float64(i) * 0.5
	}
	for i := range want.Z {
		want.Z[i] = float64(i) * -0.25
	}
	path := filepath.Join(t.TempDir(), "rt.eig")
	if err := writeBandEigCheckpoint(path, want); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := readBandEigCheckpoint(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got.Dim != want.Dim || got.Mb != want.Mb || got.Top != want.Top || got.Bot != want.Bot ||
		got.Ld != want.Ld || got.K != want.K || got.Fingerprint != want.Fingerprint {
		t.Errorf("header round-trip: got %+v", *got)
	}
	for name, pair := range map[string][2][]float64{
		"d": {got.D, want.D}, "a": {got.A, want.A}, "z": {got.Z, want.Z},
	} {
		if len(pair[0]) != len(pair[1]) {
			t.Errorf("%s: got %d elements, want %d", name, len(pair[0]), len(pair[1]))
			continue
		}
		for i := range pair[1] {
			if pair[0][i] != pair[1][i] {
				t.Errorf("%s[%d] = %v, want %v", name, i, pair[0][i], pair[1][i])
				break
			}
		}
	}
	if !got.matches(want.Dim, want.Mb, want.Top, want.Bot, want.Ld, want.Fingerprint) {
		t.Error("a freshly round-tripped checkpoint does not match its own problem")
	}
}

// TestBandEigCheckpointRejectsForeignState covers the guard. The dangerous failure is not a corrupt
// file — that is obvious — but a well-formed checkpoint from a DIFFERENT matrix of the same shape:
// resuming onto it would produce a plausible, wrong spectrum. The fingerprint is a CRC-64 of the
// input band matrix precisely so that case is rejected, and a shape check alone would not do it.
func TestBandEigCheckpointRejectsForeignState(t *testing.T) {
	const dim, band = 40, 7
	mb, ld := band+1, 8
	_, a := buildBanded(dim, band, 1)
	_, b := buildBanded(dim, band, 2)
	fpA, fpB := bandEigFingerprint(a), bandEigFingerprint(b)
	if fpA == fpB {
		t.Fatal("two different random band matrices fingerprinted the same; the guard is useless")
	}
	st := &beCkptState{
		Dim: dim, Mb: mb, Top: 4, Bot: 4, Ld: ld, Fingerprint: fpA, K: 5,
		D: make([]float64, dim), A: make([]float64, mb*dim), Z: make([]float64, ld*dim),
	}
	if !st.matches(dim, mb, 4, 4, ld, fpA) {
		t.Error("rejected its own matrix")
	}
	if st.matches(dim, mb, 4, 4, ld, fpB) {
		t.Error("accepted a checkpoint from a different matrix of the same shape")
	}
	if st.matches(dim+1, mb, 4, 4, ld, fpA) {
		t.Error("accepted a different dimension")
	}
	if st.matches(dim, mb, 3, 4, ld, fpA) {
		t.Error("accepted a different accumulated-row count")
	}
	st.A = st.A[:len(st.A)-1]
	if st.matches(dim, mb, 4, 4, ld, fpA) {
		t.Error("accepted a truncated band-matrix payload")
	}
}

// TestBandEigCheckpointMissingAndCorrupt pins the two I/O edges the solve relies on: a missing file
// is a fresh run rather than an error, and a corrupt one is an error the solve can report and then
// ignore. Neither may be fatal — losing a restart point should cost time, not the run.
func TestBandEigCheckpointMissingAndCorrupt(t *testing.T) {
	dir := t.TempDir()

	st, err := readBandEigCheckpoint(filepath.Join(dir, "absent.eig"))
	if st != nil || err != nil {
		t.Errorf("missing file: got (%v, %v), want (nil, nil)", st, err)
	}

	bad := filepath.Join(dir, "garbage.eig")
	if err := os.WriteFile(bad, []byte("not a checkpoint at all"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readBandEigCheckpoint(bad); err == nil {
		t.Error("a file with the wrong magic was accepted")
	}

	// A valid header with a truncated payload: the length the header declares is not there.
	short := filepath.Join(dir, "short.eig")
	full := &beCkptState{
		Dim: 4, Mb: 2, Top: 1, Bot: 1, Ld: 8, K: 1,
		D: make([]float64, 4), A: make([]float64, 8), Z: make([]float64, 32),
	}
	if err := writeBandEigCheckpoint(short, full); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(short)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(short, raw[:len(raw)-64], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readBandEigCheckpoint(short); err == nil {
		t.Error("a truncated payload was accepted")
	}

	// And the solve survives all of it: pointed at the garbage file it reports the problem, starts
	// the reduction from scratch, and returns the same answer as an unconfigured run.
	const dim, band = 40, 7
	main := (band + 1) / 2
	_, ref := buildBanded(dim, band, 11)
	wantD, _, _, _ := bandSymDiagFastOpts(ref, bandEigOpts{topRows: main, botRows: main})
	_, bs := buildBanded(dim, band, 11)
	gotD, _, _, interrupted := bandSymDiagFastOpts(bs, bandEigOpts{
		topRows: main, botRows: main,
		ckpt: &bandEigCkpt{path: bad, interval: time.Hour},
	})
	if interrupted {
		t.Fatal("a solve with no stop request reported itself interrupted")
	}
	for k := range wantD {
		if gotD[k] != wantD[k] {
			t.Fatalf("after ignoring a corrupt checkpoint: eval[%d] = %.17g, want %.17g",
				k, gotD[k], wantD[k])
		}
	}
}

// TestSolveLowMemSurfacesReductionInterrupt covers the one thing the unit tests above cannot reach:
// that an interrupted BAND REDUCTION propagates out of SolveLowMem as Result.Interrupted, which is
// what cmd/adcgo turns into exit 64 and what makes the daisychain resume rather than analyze an
// empty spectrum. Before the reduction had a checkpoint, a stop signal arriving during it was simply
// ignored and the generation ran to its walltime kill.
//
// The stop must be armed from INSIDE the reduction, through EigenProgress with the tick interval
// shrunk to every column. Arming it anywhere earlier makes the KRYLOV phase checkpoint instead —
// correct behaviour, but the wrong half. There is no gap between the two phases to aim at: the
// Krylov loop visits its own checkpoint hook once more after its last Progress call, so a flag set
// from there is always consumed before the eigensolve begins. That is how an earlier version of
// this test "passed" while writing no reduction checkpoint at all.
func TestSolveLowMemSurfacesReductionInterrupt(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping heavy low-memory Lanczos test in -short mode")
	}
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Singlet)
	const blocks = 3

	var stop atomic.Bool
	dir := t.TempDir()
	saved := bandEigProgressEvery
	bandEigProgressEvery = 1
	t.Cleanup(func() { bandEigProgressEvery = saved })
	res := SolveLowMem(mx, be, Options{
		MaxBlocks: blocks,
		Checkpoint: &Checkpoint{
			Path: filepath.Join(dir, "h2o"),
			Stop: &stop,
		},
		EigenProgress: func(col, cols int, elapsed time.Duration) {
			// Not the completion call (col == cols): by then the reduction is over.
			if col > 0 && col < cols {
				stop.Store(true)
			}
		},
	})
	if !res.Interrupted {
		t.Fatal("a stop request during the band reduction did not surface as Result.Interrupted; " +
			"cmd/adcgo would analyze an empty spectrum instead of exiting 'resume needed'")
	}
	if _, err := os.Stat(filepath.Join(dir, "h2o.eig")); err != nil {
		t.Errorf("no reduction checkpoint written alongside the interrupt: %v", err)
	}

	// And the successor generation finishes from it, with the stop cleared.
	stop.Store(false)
	mx2 := buildH2O(t, dip.Singlet)
	res2 := SolveLowMem(mx2, be, Options{
		MaxBlocks:  blocks,
		Checkpoint: &Checkpoint{Path: filepath.Join(dir, "h2o"), Stop: &stop},
	})
	if res2.Interrupted {
		t.Fatal("the resumed generation reported itself interrupted with no stop request")
	}
	if len(res2.Values) == 0 {
		t.Fatal("the resumed generation returned no Ritz values")
	}

	// Against an uninterrupted solve of the same problem, bit for bit.
	mx3 := buildH2O(t, dip.Singlet)
	want := SolveLowMem(mx3, be, Options{MaxBlocks: blocks})
	if len(res2.Values) != len(want.Values) {
		t.Fatalf("resumed solve returned %d values, uninterrupted %d", len(res2.Values), len(want.Values))
	}
	for i := range want.Values {
		if res2.Values[i] != want.Values[i] {
			t.Errorf("value[%d] resumed %.17g, uninterrupted %.17g", i, res2.Values[i], want.Values[i])
			break
		}
	}
	for i := range want.PS {
		if res2.PS[i] != want.PS[i] {
			t.Errorf("PS[%d] resumed %.17g, uninterrupted %.17g", i, res2.PS[i], want.PS[i])
			break
		}
	}
}
