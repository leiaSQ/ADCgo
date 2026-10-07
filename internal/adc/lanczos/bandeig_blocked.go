package lanczos

// bandeig_blocked.go — blocked back-transform of the eigenvector strip: accumulate a run of
// consecutive Givens rotations into one small dense orthogonal block, then apply that block to the
// strip as a single matrix product instead of one rotation at a time.
//
// MEASURED RESULT FIRST, because it is negative and the idea is seductive: blocking is SLOWER, in
// both regimes tested, and the penalty grows with the window almost exactly as the flop count does.
//
//	tier-1 shape (3712x37, 38 accumulated rows)     exact 2.02 s | w=8 3.41 s | w=32 5.99 s | w=64 9.33 s
//	production row count (3200x1599, 3200 rows)     exact 32.7 s | w=8 60.0 s | w=32 125 s  | w=64 207 s
//
// So this path stays OFF (bandEigOpts.blockRun defaults to 0). It is kept because it is validated and
// is the reference for anyone attempting the device version — not because it is an improvement.
//
// WHY IT LOSES, and why the device would not obviously rescue it. Blocking raises arithmetic
// intensity from ~0.19 to span/8 flop per byte, which is the right medicine for a bandwidth-starved
// device — the one-at-a-time device kernel runs at 195 GB/s, 4% of an H200's bandwidth. But it buys
// that by doing ~w/3 times more arithmetic, and on a CPU with bandwidth to spare that is a straight
// loss. On a GPU the blocker is different and harder: the blocks are tiny (3080 x span by span x
// span), so they would have to be issued as BATCHED GEMMs to amortize launch cost — there are ~4e9
// of them at the production shape — and consecutive blocks in a chain overlap by one column, so a
// chain cannot be batched with itself. Batching would need independent chains, which the QL sweep
// does not hand out. Anyone reviving this should start there, not with the kernel.
//
// WHAT WAS SUPPOSED TO HAPPEN, in measured numbers. Replaying the rotations one at a time is what the strip costs, and after
// a two-stage reduction it is essentially ALL the strip costs — the rotation count is independent of
// the bandwidth, so narrowing the band does not touch it. On an H200 the one-at-a-time device kernel
// reaches 18.3e9 updates/s, which is 195 GB/s, **4% of the machine's bandwidth**, and it is flat
// across every block size from 32 to 1024 threads. That flatness is the diagnosis: the replay's only
// parallelism is the accumulated row count (~3080), so the kernel runs ~96 warps on a 132-SM GPU and
// is bound by memory latency it has no occupancy to hide. No amount of tuning moves it.
//
// Blocking fixes that by changing the shape of the work rather than its schedule:
//
//   - traffic halves. w consecutive rotations touch w+1 columns; replayed singly they read and write
//     2 columns each (4*w column-touches), blocked they read and write the window once (2*(w+1)).
//   - what remains becomes a dense matrix product, whose parallelism is rows x window rather than
//     rows alone — so it can actually fill a GPU, and on the host it is a cache-friendly kernel
//     instead of a strided one.
//
// The price is flops: 2*rows*span^2 for the product against 6*rows*w for the rotations, i.e. ~w/3
// times more arithmetic. That is the right trade on hardware with flops to spare and no bandwidth to
// spare, and it is why the window stays small (zBlockDefault) rather than spanning a whole chase.
//
// WHICH ROTATIONS THIS SUITS. tddiag's QL bulge chase emits rotations on column pairs (i, i+1) with
// i DESCENDING, so w consecutive ops span exactly w+1 adjacent columns — a compact window, which is
// what blocking needs. It is about two thirds of the strip work. bnd2td's rotations are a chase
// stepping by m1, so consecutive ops are far apart and touch disjoint column pairs: blocking them
// would build a mostly-zero block, but being disjoint they are already independent and want
// batching, not blocking. This file is therefore about the QL sweep.
//
// NOT BIT-EXACT, by construction: the block product reassociates the arithmetic. It is validated
// against the one-at-a-time replay to roundoff, and against the reference-free S*S^T = I invariant.

import "math"

// cosSin returns a unit (cosine, sine) pair for angle th, for tests that need a real rotation.
func cosSin(th float64) (float64, float64) { return math.Cos(th), math.Sin(th) }

// zBlockDefault is how many consecutive chained rotations are accumulated into one dense block.
//
// Traffic improves by ~2x for any window size, so this does not chase traffic — it trades flops
// (2*rows*span^2, growing with the window) against per-block overhead (fixed). 32 keeps the block at
// 33x33, which is 8 KB and stays in L1 while the product streams the strip window past it, and keeps
// the flop multiplier at ~11x rather than the ~100x a 128-wide window would cost.
const zBlockDefault = 32

// zBlockMin is the shortest run worth blocking. A block costs 2*rows*span^2 flops against the
// 6*rows*w the rotations would cost, so a short run pays the extra arithmetic without earning back
// enough of the halved traffic; below this the ordinary one-at-a-time path is used.
const zBlockMin = 8

// givensBlock accumulates w consecutive QL rotations into the dense orthogonal matrix they are
// equivalent to, and reports the column window it acts on.
//
// One rotation on columns (i, i+1) with cosine c and sine s sends
//
//	Z[:,i]   <- c*Z[:,i] - s*Z[:,i+1]
//	Z[:,i+1] <- s*Z[:,i] + c*Z[:,i+1]
//
// i.e. Z <- Z*R with R the identity except R[i][i]=c, R[i][i+1]=s, R[i+1][i]=-s, R[i+1][i+1]=c.
// A run of rotations is therefore Z <- Z*R_1*R_2*...*R_w, and g is that product, built by applying
// each R on the RIGHT of the running block — which touches only two of its columns, so the whole
// accumulation is O(w*span) and negligible beside the strip product it replaces.
//
// cols[t] is the LOWER column index of rotation t (1-based, as everywhere in this port). The run
// must be contiguous and descending, which is what tddiag emits; runOfChainedRotTD finds such runs.
// g is returned column-major, span x span, and col0 is the 1-based first column of the window.
func givensBlock(cols []int32, f1, f2 []float64) (g []float64, col0, span int) {
	if len(cols) == 0 {
		return nil, 0, 0
	}
	g = make([]float64, (len(cols)+1)*(len(cols)+1))
	col0, span = givensBlockInto(g, cols, f1, f2)
	return g[:span*span], col0, span
}

// givensBlockInto is givensBlock writing into a caller-owned buffer, which must hold at least
// (len(cols)+1)^2 entries. A flush blocks millions of runs, so the buffer is hoisted out of the loop
// for the same reason the deferred-op tape exists at all.
func givensBlockInto(g []float64, cols []int32, f1, f2 []float64) (col0, span int) {
	w := len(cols)
	if w == 0 {
		return 0, 0
	}
	lo, hi := int(cols[0]), int(cols[0])
	for _, c := range cols {
		lo = min(lo, int(c))
		hi = max(hi, int(c))
	}
	col0 = lo
	span = hi - lo + 2 // the run's highest rotation also touches its upper column
	g = g[:span*span]
	clear(g)
	for i := range span {
		g[i*span+i] = 1
	}
	for t := range w {
		c, s := f1[t], f2[t]
		i := int(cols[t]) - col0 // 0-based lower column within the window
		// g <- g*R: only columns i and i+1 change, and each new column is a combination of the two
		// old ones, so both must be read before either is written.
		ci := g[i*span : (i+1)*span]
		ci1 := g[(i+1)*span : (i+2)*span]
		for r := range span {
			a, b := ci[r], ci1[r]
			ci[r] = c*a - s*b
			ci1[r] = s*a + c*b
		}
	}
	return col0, span
}

// runOfChainedRotTD reports how many ops starting at `from` form one contiguous descending chain of
// QL rotations, capped at maxRun. A run is extendable while the next op is a rotTD whose column is
// exactly one below the previous one — tddiag's inner loop pattern — so the run's window stays
// compact. Any other op kind, or a break in the descent, ends the run.
//
// Returning 1 means "no run worth blocking here"; the caller replays that op the ordinary way.
func runOfChainedRotTD(kind []uint8, cols []int32, from, n, maxRun int) int {
	if from >= n || kind[from] != zOpRotTD {
		return 0
	}
	run := 1
	for from+run < n && run < maxRun {
		if kind[from+run] != zOpRotTD || cols[from+run] != cols[from+run-1]-1 {
			break
		}
		run++
	}
	return run
}

// applyGivensBlock right-multiplies rows [lo,hi) of the strip's column window by g, the span x span
// block givensBlock built.
//
// This is the operation that replaces the rotations: one pass over the window instead of one pass
// per rotation. The window is contiguous in the strip's column-major layout, so the whole thing is a
// dense matrix product and is what a device backend hands to GEMM.
//
// buf is scratch of at least (hi-lo)*span; it is passed in because a production flush calls this
// millions of times and allocating per call is what the deferred-op design exists to avoid.
func applyGivensBlock(z []float64, ld, lo, hi, col0, span int, g, buf []float64) {
	m := hi - lo
	if m <= 0 || span <= 0 {
		return
	}
	// new[:, j] = sum_k old[:, k] * g[k][j]
	for j := range span {
		out := buf[j*m : (j+1)*m]
		clear(out)
		gc := g[j*span : (j+1)*span]
		for k := range span {
			gkj := gc[k]
			if gkj == 0 {
				continue
			}
			src := z[(col0-1+k)*ld+lo:][:m]
			for r := range out {
				out[r] += src[r] * gkj
			}
		}
	}
	for j := range span {
		copy(z[(col0-1+j)*ld+lo:][:m], buf[j*m:(j+1)*m])
	}
}

// blockedReplayErr is a test helper: the largest absolute difference between two strips.
func blockedReplayErr(a, b []float64) float64 {
	worst := 0.0
	for i := range a {
		if e := math.Abs(a[i] - b[i]); e > worst {
			worst = e
		}
	}
	return worst
}
