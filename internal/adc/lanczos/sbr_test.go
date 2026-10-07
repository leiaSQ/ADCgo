package lanczos

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// sbrDenseRef is the DENSE reference for successive band reduction: it narrows a symmetric matrix's
// half-bandwidth from bw to b2 by blocked orthogonal similarity, carrying a strip of the accumulated
// transform, using a plain n x n array throughout.
//
// It exists to isolate the one thing that is genuinely hard here — WHERE the reflectors act — from
// everything else. There is no band storage, so none of the index arithmetic that silently corrupts a
// bandStorage (unchecked `set`, out-of-band writes landing in the next column) can be involved, and
// the similarity can be checked directly: ‖A2 − ZᵀAZ‖, which is impossible at production scale.
//
// The window schedule, which is the part to get right:
//
//	d   = bw − b2                 columns narrowed per QR
//	m   = bw − b2 + d             window height; equals bw when the sweep halves the bandwidth
//	R_0 = [j+b2, j+b2+m)          first window for block column C = [j, j+d)
//	R_s = R_{s−1} shifted by bw   the chase, until it runs off the matrix
//
// The chase stride is bw, NOT the window height m. After an update on a window of height mm the deepest
// entry in its columns sits at distance mm-1+bw, so the entries beyond the band start at distance bw+1
// and the next window must begin bw below the last one. m and bw coincide exactly when a sweep halves
// the bandwidth, which is why a schedule of halving sweeps hides the difference — and why a sweep like
// 9 -> 8 did not narrow at all until this was corrected.
//
// R_0's last row is (j+d−1) + bw — exactly the deepest band entry of C's last column, which is why the
// window has that height. A QR of the m x d panel can only triangularise it, leaving a d x d triangle
// whose diagonal sits at distance b2 from the main diagonal: that is what sets the resulting
// bandwidth, and it is why the window top is sub-diagonal b2 rather than b2+1.
//
// A is n x n row-major and is modified in place. Z is zr x n row-major and is right-multiplied.
func sbrDenseRef(A []float64, n, bw, b2 int, Z []float64, zr int) {
	d := bw - b2
	if d <= 0 {
		return
	}
	m := bw - b2 + d
	for j := 0; j+b2 < n; j += d {
		src0, srcN := j, min(d, n-j)
		for w0 := j + b2; w0 < n; w0 += bw {
			mm := min(m, n-w0)
			// Bottom-right clipping: the window runs out before the panel is exhausted, so the panel
			// can be WIDER than it is tall and cannot be triangularised. Narrow it to what the window
			// can hold — the columns dropped this way have no entries inside the window left to
			// annihilate — and let the bandwidth assertion in the gate judge whether that is true
			// rather than assuming it.
			sn := min(srcN, mm)
			if mm <= 1 || sn <= 0 {
				break
			}
			q := densePanelQ(A, n, w0, mm, src0, sn)
			denseSimilarity(A, n, w0, mm, q)
			denseApplyRight(Z, zr, n, w0, mm, q)
			src0, srcN = w0, mm
		}
	}
}

// densePanelQ returns the mm x mm orthogonal factor of the QR of A[w0:w0+mm, src0:src0+srcN].
func densePanelQ(A []float64, n, w0, mm, src0, srcN int) []float64 {
	p := mat.NewDense(mm, srcN, nil)
	for r := range mm {
		for c := range srcN {
			p.Set(r, c, A[(w0+r)*n+src0+c])
		}
	}
	var qr mat.QR
	qr.Factorize(p)
	var qd mat.Dense
	qr.QTo(&qd)
	q := make([]float64, mm*mm)
	for r := range mm {
		for c := range mm {
			q[r*mm+c] = qd.At(r, c)
		}
	}
	return q
}

// denseSimilarity applies A <- QᵀAQ, with Q the identity outside rows/columns [w0, w0+mm).
func denseSimilarity(A []float64, n, w0, mm int, q []float64) {
	// rows: A[R, :] <- Qᵀ A[R, :]
	tmp := make([]float64, mm*n)
	for r := range mm {
		for c := range n {
			var acc float64
			for k := range mm {
				acc += q[k*mm+r] * A[(w0+k)*n+c]
			}
			tmp[r*n+c] = acc
		}
	}
	for r := range mm {
		copy(A[(w0+r)*n:(w0+r)*n+n], tmp[r*n:(r+1)*n])
	}
	// columns: A[:, R] <- A[:, R] Q
	tmp2 := make([]float64, n*mm)
	for r := range n {
		for c := range mm {
			var acc float64
			for k := range mm {
				acc += A[r*n+w0+k] * q[k*mm+c]
			}
			tmp2[r*mm+c] = acc
		}
	}
	for r := range n {
		for c := range mm {
			A[r*n+w0+c] = tmp2[r*mm+c]
		}
	}
}

// denseApplyRight applies Z <- Z Q on columns [w0, w0+mm).
func denseApplyRight(Z []float64, zr, n, w0, mm int, q []float64) {
	if zr == 0 {
		return
	}
	tmp := make([]float64, zr*mm)
	for r := range zr {
		for c := range mm {
			var acc float64
			for k := range mm {
				acc += Z[r*n+w0+k] * q[k*mm+c]
			}
			tmp[r*mm+c] = acc
		}
	}
	for r := range zr {
		for c := range mm {
			Z[r*n+w0+c] = tmp[r*mm+c]
		}
	}
}

// denseProfile returns the largest row-column distance carrying a value above tol.
func denseProfile(A []float64, n int, tol float64) int {
	worst := 0
	for r := range n {
		for c := range n {
			if d := r - c; d > worst && math.Abs(A[r*n+c]) > tol {
				worst = d
			}
		}
	}
	return worst
}

// TestSBRDenseRefIsASimilarity is gate A0 for stage 1, and the only place the three properties can all
// be checked at once: the bandwidth really drops to the target, the accumulated transform is
// orthogonal, and A2 really equals ZᵀAZ. The last is the direct statement of correctness and is
// unavailable at any realistic size, which is the whole reason this gate exists before any band-storage
// or performance work.
func TestSBRDenseRefIsASimilarity(t *testing.T) {
	// Halving cases AND non-halving ones. The second group is not padding: the chase stride is bw
	// while the window height is m, and those coincide only when a sweep halves — so a halving-only
	// suite passes a wrong stride, which is exactly what happened here until 9->8 failed in the
	// production chain. sbrSweepTo produces such sweeps whenever the target is not a power-of-two
	// fraction of the bandwidth.
	for _, tc := range []struct{ n, bw, b2 int }{
		{40, 8, 4}, {41, 8, 4}, {60, 12, 6}, {97, 20, 10}, {128, 16, 8}, {300, 40, 20},
		{60, 9, 8}, {97, 5, 4}, {128, 7, 5}, {200, 12, 7}, {150, 20, 11},
	} {
		n := tc.n
		rng := rand.New(rand.NewSource(int64(8800 + n)))
		A0 := make([]float64, n*n)
		for c := range n {
			for r := c; r < n && r-c <= tc.bw; r++ {
				v := rng.NormFloat64()
				A0[r*n+c] = v
				A0[c*n+r] = v
			}
		}
		nrm := 0.0
		for _, v := range A0 {
			nrm = math.Max(nrm, math.Abs(v))
		}

		A := append([]float64(nil), A0...)
		Z := make([]float64, n*n)
		for i := range n {
			Z[i*n+i] = 1
		}
		sbrDenseRef(A, n, tc.bw, tc.b2, Z, n)

		if got := denseProfile(A, n, 1e-11*nrm); got > tc.b2 {
			t.Errorf("n=%d bw=%d b2=%d: half-bandwidth %d after the sweep, want <= %d",
				n, tc.bw, tc.b2, got, tc.b2)
		}
		// Z orthogonal.
		worstO := 0.0
		for i := range n {
			for j := i; j < n; j++ {
				var acc float64
				for k := range n {
					acc += Z[k*n+i] * Z[k*n+j]
				}
				want := 0.0
				if i == j {
					want = 1
				}
				worstO = math.Max(worstO, math.Abs(acc-want))
			}
		}
		if worstO > 1e-12 {
			t.Errorf("n=%d bw=%d b2=%d: max|ZᵀZ-I| = %.3e", n, tc.bw, tc.b2, worstO)
		}
		// A == Z A2 Zᵀ, i.e. A2 == Zᵀ A Z.
		worstS := 0.0
		for i := range n {
			for j := range n {
				var acc float64
				for k := range n {
					for l := range n {
						if Z[k*n+i] == 0 || Z[l*n+j] == 0 {
							continue
						}
						acc += Z[k*n+i] * A0[k*n+l] * Z[l*n+j]
					}
				}
				worstS = math.Max(worstS, math.Abs(acc-A[i*n+j]))
			}
		}
		if worstS > 1e-10*nrm {
			t.Errorf("n=%d bw=%d b2=%d: max|A2 - ZᵀAZ| = %.3e (‖A‖≈%.3g)", n, tc.bw, tc.b2, worstS, nrm)
		}
		// And the spectrum is preserved.
		be := backend.Gonum{}
		M0, M2 := backend.NewMat(n, n), backend.NewMat(n, n)
		copy(M0.Data, A0)
		copy(M2.Data, A)
		w0, _ := be.SymEig(M0)
		w2, _ := be.SymEig(M2)
		for k := range w0 {
			if e := math.Abs(w0[k] - w2[k]); e > 1e-10*nrm {
				t.Errorf("n=%d bw=%d b2=%d: eval[%d] moved by %.3e", n, tc.bw, tc.b2, k, e)
				break
			}
		}
	}
}

// sbrDenseSchedule narrows from bw to b2 by repeated halving sweeps, which is how stage 1 is actually
// used: a single sweep can at most halve the bandwidth, because the tiling constraint d = bw - b'
// combined with d <= b' forces b' >= bw/2. At the production shape 3079 -> 128 is five sweeps.
func sbrDenseSchedule(A []float64, n, bw, b2 int, Z []float64, zr int) int {
	for bw > b2 {
		next := max(b2, (bw+1)/2)
		sbrDenseRef(A, n, bw, next, Z, zr)
		bw = next
	}
	return bw
}

// TestSBRDenseScheduleComposes checks that repeated sweeps compose — that a sweep starting from an
// already-narrowed matrix behaves, which is not self-evident: each sweep's window schedule is derived
// from the bandwidth it is handed, so an off-by-one in the previous sweep's result would put the next
// sweep's windows in the wrong place and show up only here.
func TestSBRDenseScheduleComposes(t *testing.T) {
	for _, tc := range []struct{ n, bw, b2 int }{
		{128, 32, 4}, {200, 48, 6}, {300, 64, 8}, {301, 40, 5},
	} {
		n := tc.n
		rng := rand.New(rand.NewSource(int64(9100 + n)))
		A0 := make([]float64, n*n)
		for c := range n {
			for r := c; r < n && r-c <= tc.bw; r++ {
				v := rng.NormFloat64()
				A0[r*n+c] = v
				A0[c*n+r] = v
			}
		}
		nrm := 0.0
		for _, v := range A0 {
			nrm = math.Max(nrm, math.Abs(v))
		}
		A := append([]float64(nil), A0...)
		Z := make([]float64, n*n)
		for i := range n {
			Z[i*n+i] = 1
		}
		got := sbrDenseSchedule(A, n, tc.bw, tc.b2, Z, n)
		if got != tc.b2 {
			t.Errorf("n=%d %d->%d: schedule ended at %d", n, tc.bw, tc.b2, got)
		}
		if p := denseProfile(A, n, 1e-11*nrm); p > tc.b2 {
			t.Errorf("n=%d %d->%d: half-bandwidth %d after the schedule, want <= %d",
				n, tc.bw, tc.b2, p, tc.b2)
		}
		be := backend.Gonum{}
		M0, M2 := backend.NewMat(n, n), backend.NewMat(n, n)
		copy(M0.Data, A0)
		copy(M2.Data, A)
		w0, _ := be.SymEig(M0)
		w2, _ := be.SymEig(M2)
		worst := 0.0
		for k := range w0 {
			worst = math.Max(worst, math.Abs(w0[k]-w2[k]))
		}
		if worst > 1e-9*nrm {
			t.Errorf("n=%d %d->%d: spectrum moved by %.3e (‖A‖≈%.3g)", n, tc.bw, tc.b2, worst, nrm)
		}
		t.Logf("n=%d %d->%d: reached %d, spectrum moved %.2e", n, tc.bw, tc.b2, got, worst)
	}
}

// TestSBRBandMatchesDense is gate A1: the band-storage sweep must reproduce the dense reference
// ELEMENTWISE on the same input. That isolates the band indexing from the algebra completely — the
// algorithm is identical, so any difference is an index error and the bound can be tight.
//
// This is the split that caught the two band-storage bugs earlier in this work: an out-of-band write
// wrapping into the next column, and a shape assumption that deflation violates. Neither is visible
// in a dense formulation, and neither shows up as anything but a wrong number.
func TestSBRBandMatchesDense(t *testing.T) {
	for _, tc := range []struct{ n, bw, b2 int }{
		{40, 8, 4}, {41, 8, 4}, {60, 12, 6}, {97, 20, 10}, {128, 16, 8}, {200, 24, 12},
		{60, 9, 8}, {97, 5, 4}, {128, 7, 5}, {200, 12, 7},
	} {
		n := tc.n
		rng := rand.New(rand.NewSource(int64(9300 + n)))
		dense := make([]float64, n*n)
		bs := newBandStorage(n, tc.bw)
		for c := range n {
			for t := 0; t <= tc.bw && c+t < n; t++ {
				v := rng.NormFloat64()
				bs.set(c, t, v)
				dense[(c+t)*n+c] = v
				dense[c*n+(c+t)] = v
			}
		}
		nrm := 0.0
		for _, v := range dense {
			nrm = math.Max(nrm, math.Abs(v))
		}

		// Dense reference, one sweep.
		dz := make([]float64, n*n)
		for i := range n {
			dz[i*n+i] = 1
		}
		sbrDenseRef(dense, n, tc.bw, tc.b2, dz, n)

		// Band version, same sweep, with the strip seeded the same way for the top rows.
		main := tc.b2
		pool := parallel.NewFixedPool(1)
		za := newZAccum(n, bandEigOpts{topRows: main, botRows: main, workers: 1}, pool)
		// sbrSweep does not seed — only sbrReduce does, since a schedule seeds once and then sweeps
		// repeatedly. Seed here because this test drives a single sweep directly.
		za.seed()
		got := sbrSweep(bs, tc.b2, za)
		za.flush()
		pool.Close()

		if got.band != tc.b2 {
			t.Fatalf("n=%d %d->%d: band version ended at %d", n, tc.bw, tc.b2, got.band)
		}
		worst, at := 0.0, [2]int{}
		for c := range n {
			for t := 0; t <= tc.b2 && c+t < n; t++ {
				if e := math.Abs(got.at(c, t) - dense[(c+t)*n+c]); e > worst {
					worst, at = e, [2]int{c + t, c}
				}
			}
		}
		if worst > 1e-11*nrm {
			t.Errorf("n=%d %d->%d: band vs dense max|Δ| = %.3e at (%d,%d) (‖A‖≈%.3g)",
				n, tc.bw, tc.b2, worst, at[0], at[1], nrm)
		}
		// The strip the band version accumulated must match the corresponding rows of the dense Z.
		worstZ := 0.0
		for k := range n {
			for r := range main {
				worstZ = math.Max(worstZ, math.Abs(za.z[k*za.ld+r]-dz[r*n+k]))
			}
		}
		if worstZ > 1e-11 {
			t.Errorf("n=%d %d->%d: strip vs dense Z max|Δ| = %.3e", n, tc.bw, tc.b2, worstZ)
		}
	}
}

// TestSBRReduceComposition is the production-path gate: sbrReduce followed by the existing solver at
// the narrowed bandwidth must reproduce the single-stage answer. Eigenvalues against the frozen serial
// reference, and the strip against the reference-free S·Sᵀ = I, since a non-bit-exact path cannot be
// compared elementwise and near-degenerate eigenvalues reorder.
func TestSBRReduceComposition(t *testing.T) {
	for _, tc := range []struct{ n, bw, b2 int }{
		{60, 12, 3}, {97, 20, 5}, {128, 16, 4}, {200, 24, 6}, {260, 40, 10},
	} {
		n := tc.n
		main := (tc.bw + 1) / 2
		if 2*main > n {
			continue
		}
		seed := int64(9500 + n)
		_, ref := buildBanded(n, tc.bw, seed)
		wantD, _ := bandSymDiagFastRef(ref)
		nrm := 0.0
		for _, v := range wantD {
			nrm = math.Max(nrm, math.Abs(v))
		}

		_, bs := buildBanded(n, tc.bw, seed)
		pool := parallel.NewFixedPool(1)
		o := bandEigOpts{topRows: main, botRows: main, workers: 1}
		za := newZAccum(n, o, pool)
		narrow := sbrReduce(bs, tc.b2, za)
		za.flush()
		z1 := append([]float64(nil), za.z...)
		pool.Close()

		if narrow.band != tc.b2 {
			t.Fatalf("n=%d %d->%d: reached band %d", n, tc.bw, tc.b2, narrow.band)
		}
		gotD, z, ld, _ := bandSymDiagFastOpts(narrow, bandEigOpts{
			topRows: main, botRows: main, workers: 1, resumeZ: z1,
		})
		for k := range wantD {
			if e := math.Abs(gotD[k] - wantD[k]); e > 1e-9*nrm {
				t.Errorf("n=%d %d->%d: eval[%d] = %.12g, single-stage %.12g (Δ%.2e)",
					n, tc.bw, tc.b2, k, gotD[k], wantD[k], e)
				break
			}
		}
		if e := stripOrthErr(z, 2*main, ld, n); e > 1e-10 {
			t.Errorf("n=%d %d->%d: max|S·Sᵀ-I| = %.3e", n, tc.bw, tc.b2, e)
		}
	}
}
