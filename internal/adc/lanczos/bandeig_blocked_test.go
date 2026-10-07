package lanczos

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestGivensBlockMatchesRotations is the correctness gate for the blocked back-transform: applying a
// run of QL rotations one at a time and applying the single dense block they accumulate to must give
// the same strip.
//
// Not bit-exact — the block product reassociates the arithmetic — so the comparison is to roundoff.
// That is the whole reason this needs its own test: every other strip guarantee in this package is
// exact, and this is the first place where "close" is the correct standard rather than a lowered bar.
func TestGivensBlockMatchesRotations(t *testing.T) {
	for _, w := range []int{1, 2, 7, 32, 33} {
		for _, rows := range []int{1, 8, 64, 257} {
			n := w + 8
			ld := rows
			rng := rand.New(rand.NewSource(int64(w*1000 + rows)))
			z := make([]float64, ld*n)
			for i := range z {
				z[i] = rng.NormFloat64()
			}
			// tddiag's pattern: rotations on (i, i+1) with i descending by one.
			cols := make([]int32, w)
			f1 := make([]float64, w)
			f2 := make([]float64, w)
			hiCol := n - 2 // 1-based; the top rotation also touches column hiCol+1 <= n
			for t := range w {
				cols[t] = int32(hiCol - t)
				th := rng.Float64() * 6.28
				f1[t], f2[t] = cosSin(th)
			}

			// Reference: one rotation at a time, exactly as zAccum.replay does.
			want := append([]float64(nil), z...)
			for t := range w {
				i := int(cols[t])
				ci := want[(i-1)*ld:][:rows]
				ci1 := want[i*ld:][:rows]
				for r := range ci {
					hh := ci1[r]
					ci1[r] = f2[t]*ci[r] + f1[t]*hh
					ci[r] = f1[t]*ci[r] - f2[t]*hh
				}
			}

			got := append([]float64(nil), z...)
			g, col0, span := givensBlock(cols, f1, f2)
			buf := make([]float64, rows*span)
			applyGivensBlock(got, ld, 0, rows, col0, span, g, buf)

			if e := blockedReplayErr(got, want); e > 1e-13 {
				t.Errorf("w=%d rows=%d: blocked vs one-at-a-time max|Δ| = %.3e, want <= 1e-13", w, rows, e)
			}
			// The accumulated block must itself be orthogonal, which is what makes the strip's
			// S*S^T = I invariant survive the substitution.
			for i := range span {
				for j := i; j < span; j++ {
					var acc float64
					for k := range span {
						acc += g[i*span+k] * g[j*span+k]
					}
					want := 0.0
					if i == j {
						want = 1.0
					}
					if e := acc - want; e > 1e-13 || e < -1e-13 {
						t.Fatalf("w=%d: accumulated block is not orthogonal at (%d,%d): %.3e", w, i, j, acc)
					}
				}
			}
		}
	}
}

// TestRunOfChainedRotTD pins the run detector: it must find exactly the descending chains tddiag
// emits, stop at any other op kind, stop at a break in the descent, and honour the cap. Getting this
// wrong does not corrupt anything — a short run is merely slower — but a run that is too LONG would
// build a block for columns the rotations do not actually span.
func TestRunOfChainedRotTD(t *testing.T) {
	kind := []uint8{zOpRotTD, zOpRotTD, zOpRotTD, zOpScale, zOpRotTD, zOpRotTD, zOpRotA}
	cols := []int32{10, 9, 8, 4, 6, 3, 2}
	n := len(kind)
	for _, tc := range []struct{ from, maxRun, want int }{
		{0, 32, 3}, // 10,9,8 then a scale ends it
		{0, 2, 2},  // the cap binds
		{3, 32, 0}, // a scale is not a chain
		{4, 32, 1}, // 6 then 3 is not a descent by one
		{5, 32, 1}, // 3 then a rotA ends it
		{6, 32, 0}, // rotA
		{7, 32, 0}, // past the end
	} {
		if got := runOfChainedRotTD(kind, cols, tc.from, n, tc.maxRun); got != tc.want {
			t.Errorf("runOfChainedRotTD(from=%d, maxRun=%d) = %d, want %d",
				tc.from, tc.maxRun, got, tc.want)
		}
	}
}

// TestBlockedReplayMatchesExact checks the blocked back-transform end to end through the solver: the
// eigenvalues and the strip must agree with the exact one-at-a-time path to roundoff, and the strip
// must still satisfy S*S^T = I.
//
// Two things make this the right pair of checks. Eigenvalues are untouched in principle — blocking
// only changes how the strip is updated, not the matrix reduction — so they should agree far more
// tightly than the strip does, and a discrepancy there would mean the block is leaking into `a`.
// The strip cannot be compared bit-for-bit because the block product reassociates, so the orthogonality
// invariant carries the weight: it holds for any orthogonal transform and would catch a block built
// over the wrong column window, which is the plausible bug here.
func TestBlockedReplayMatchesExact(t *testing.T) {
	for _, tc := range []struct{ dim, band int }{{40, 7}, {200, 2}, {260, 128}, {512, 5}} {
		main := (tc.band + 1) / 2
		if 2*main > tc.dim {
			continue
		}
		seed := int64(7700 + tc.dim*10 + tc.band)
		_, exact := buildBanded(tc.dim, tc.band, seed)
		wantD, wantZ, wantLD, _ := bandSymDiagFastOpts(exact, bandEigOpts{topRows: main, botRows: main})

		for _, run := range []int{8, 32, 64} {
			_, bs := buildBanded(tc.dim, tc.band, seed)
			gotD, gotZ, gotLD, _ := bandSymDiagFastOpts(bs, bandEigOpts{
				topRows: main, botRows: main, blockRun: run,
			})
			if gotLD != wantLD {
				t.Fatalf("dim=%d band=%d run=%d: ld %d, want %d", tc.dim, tc.band, run, gotLD, wantLD)
			}
			// The reduction is untouched, so the eigenvalues must be bit-identical, not merely close:
			// blocking changes only the strip. Anything else means the block wrote outside the strip.
			for k := range wantD {
				if gotD[k] != wantD[k] {
					t.Fatalf("dim=%d band=%d run=%d: eval[%d] = %.17g, exact %.17g — blocking must not "+
						"touch the matrix reduction", tc.dim, tc.band, run, k, gotD[k], wantD[k])
				}
			}
			if e := blockedReplayErr(gotZ, wantZ); e > 1e-11 {
				t.Errorf("dim=%d band=%d run=%d: strip max|Δ| vs exact = %.3e, want <= 1e-11",
					tc.dim, tc.band, run, e)
			}
			if e := stripOrthErr(gotZ, 2*main, gotLD, tc.dim); e > 1e-11 {
				t.Errorf("dim=%d band=%d run=%d: max|S·Sᵀ-I| = %.3e, want <= 1e-11",
					tc.dim, tc.band, run, e)
			}
		}
	}
}

// BenchmarkBlockedReplay compares the blocked strip update against the exact one at the tier-1 shape.
// It is the measurement that decides whether blocking is worth taking to the device: on the host it
// shows the flops/traffic trade, and the device case only improves on it, because the block product's
// parallelism is rows x window rather than rows alone — which is precisely what the one-at-a-time
// kernel lacked (18.3 G updates/s flat across every block size, 4% of an H200's bandwidth).
// The shape matters more than the size here. rows = 2*main is what sets the block product's
// arithmetic intensity (span/8 flop per byte, against the rotations' 0.19), so a shape with few
// accumulated rows measures the wrong regime entirely — tier-1's band 37 gives 38 rows against
// production's 3080. Both are run: "narrow" is the cheap tier-1 shape, "wide" has production's row
// count at a dim small enough to finish.
func BenchmarkBlockedReplay(b *testing.B) {
	for _, shape := range []struct {
		name      string
		dim, band int
	}{{"narrow", 3712, 37}, {"wide", 3200, 1599}} {
		b.Run(shape.name, func(b *testing.B) { benchBlockedReplay(b, shape.dim, shape.band) })
	}
}

func benchBlockedReplay(b *testing.B, dim, band int) {
	main := (band + 1) / 2
	_, proto := buildBanded(dim, band, 7)
	for _, run := range []int{0, 8, 32, 64} {
		name := "exact"
		if run > 0 {
			name = fmt.Sprintf("blockRun=%d", run)
		}
		b.Run(name, func(b *testing.B) {
			for b.Loop() {
				bs := bandStorage{data: append([]float64(nil), proto.data...), dim: proto.dim, band: proto.band}
				bandSymDiagFastOpts(bs, bandEigOpts{topRows: main, botRows: main, blockRun: run})
			}
		})
	}
}
