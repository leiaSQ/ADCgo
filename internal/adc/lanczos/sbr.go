package lanczos

// sbr.go — stage 1 of the two-stage reduction on band storage: narrow the projected matrix's
// half-bandwidth by blocked orthogonal similarity, carrying the eigenvector strip, so the chase in
// bandeig.go then runs at a small bandwidth instead of 3079.
//
// WHY. The chase's band work is proportional to the bandwidth while its eigenvector work is not, so
// narrowing first is the one structural saving available. Measured at the production shape: ~63 h of
// band work at bandwidth 3079 against ~1.3 h at 64.
//
// THE SCHEDULE, validated against a dense reference before any of this was written (see
// TestSBRDenseRefIsASimilarity and TestSBRDenseScheduleComposes in sbr_test.go — those gates exist
// precisely because the window placement is the part that cannot be derived reliably from memory):
//
//	d   = bw − b'              columns narrowed per QR
//	m   = bw − b' + d          window height
//	R_0 = [j+b', j+b'+m)       first window for block column [j, j+d)
//	R_s = R_{s−1} + bw         the chase, until it runs off the matrix (stride bw, NOT m: after an
//	                           update the deepest entry in the window's columns is at distance
//	                           mm-1+bw, so the next window starts bw lower. m == bw only when a sweep
//	                           halves, which is why a halving-only test suite hides the difference)
//
// A QR of the m x d panel can only TRIANGULARISE it — a d x d triangle always survives — and that
// triangle's diagonal sits at distance b' from the main diagonal. That is what sets the resulting
// bandwidth, and it is why the window top is sub-diagonal b' and not b'+1. Combined with the tiling
// requirement d = bw − b' and the constraint d <= b' (so the panel does not straddle the diagonal),
// a single sweep can at most HALVE the bandwidth — hence sbrReduce runs a schedule of sweeps, five of
// them to take 3079 to 128.
//
// STORAGE. The two-sided update spreads the coupling below the band, out to distance bw + m − 1: the
// bulge. The working store is therefore allocated wider than the matrix needs and tightened once the
// schedule finishes. Every write goes through sbSet, which panics rather than let an out-of-band index
// wrap into the next column — bandStorage.set is unchecked, and that failure mode has already cost
// this package an afternoon (see fillBandNarrowed).

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"

	"github.com/leiaSQ/ADCgo/backend"
)

// sbTol is the largest magnitude that may be dropped as "outside the band". Anything above it is a
// real entry and a bug, not roundoff.
const sbTol = 1e-11

// sbAt reads element (r,c) of a symmetric band matrix stored lower-only, in either order, returning 0
// outside the band.
func sbAt(bs bandStorage, r, c int) float64 {
	if r < c {
		r, c = c, r
	}
	t := r - c
	if t > bs.band || c < 0 || r >= bs.dim {
		return 0
	}
	return bs.at(c, t)
}

// sbSet writes element (r,c), in either order. Out-of-band is tolerated only for a value that is
// numerically zero; anything else panics, because bandStorage.set would otherwise write it into the
// next column and corrupt a real entry.
func sbSet(bs bandStorage, r, c int, v float64) {
	if r < c {
		r, c = c, r
	}
	t := r - c
	if c < 0 || r >= bs.dim {
		return
	}
	if t > bs.band {
		if math.Abs(v) > sbTol {
			panic(fmt.Sprintf("lanczos: sbr would write %g at (%d,%d), distance %d beyond the "+
				"allocated band %d", v, r, c, t, bs.band))
		}
		return
	}
	bs.set(c, t, v)
}

// sbrReduce narrows bs to half-bandwidth b2 and returns the narrowed matrix, applying the same
// transform to za. za must be freshly constructed; this seeds it.
//
// Unlike blockQRReduce this reads only bandStorage, so it is immune to the deflation shape mismatch
// that blockQRReduceOK has to guard against — which makes it the better fallback when that guard
// fires: SBR from the full bandwidth costs one extra sweep rather than losing stage 1 entirely.
func sbrReduce(bs bandStorage, b2 int, za *zAccum) bandStorage {
	za.seed()
	if b2 < 1 {
		panic(fmt.Sprintf("lanczos: sbrReduce needs b2 >= 1, got %d", b2))
	}
	if b2 >= bs.band || bs.dim <= 2 {
		return bs
	}
	cur := bs
	for cur.band > b2 {
		next := max(b2, (cur.band+1)/2)
		cur = sbrSweep(cur, next, za)
	}
	return cur
}

// sbrSweep narrows cur's half-bandwidth to b2 (which must be at least half of it) in one pass, and
// returns the tightened result.
func sbrSweep(cur bandStorage, b2 int, za *zAccum) bandStorage {
	bw := cur.band
	d := bw - b2
	if d <= 0 {
		return cur
	}
	if d > b2 {
		panic(fmt.Sprintf("lanczos: sbrSweep from %d to %d would need d=%d > b2, so the panel would "+
			"straddle the diagonal; a sweep can at most halve the bandwidth", bw, b2, d))
	}
	m := bw - b2 + d
	n := cur.dim

	// Working store, wide enough for the bulge the two-sided update creates.
	work := newBandStorage(n, min(bw+m, max(n-1, 0)))
	for c := range n {
		for t := 0; t <= bw && c+t < n; t++ {
			if v := cur.at(c, t); v != 0 {
				work.set(c, t, v)
			}
		}
	}

	for j := 0; j+b2 < n; j += d {
		src0, srcN := j, min(d, n-j)
		for w0 := j + b2; w0 < n; w0 += bw {
			mm := min(m, n-w0)
			// Bottom-right clipping: the window can run out before the panel is exhausted, leaving a
			// panel wider than it is tall, which cannot be triangularised. Narrow it to the window.
			sn := min(srcN, mm)
			if mm <= 1 || sn <= 0 {
				break
			}
			q := sbrPanelQ(work, w0, mm, src0, sn)
			// reach is bw, the bandwidth the matrix has AWAY from the window, not work.band. Using the
			// allocated width instead makes the bottom panel far wider than the structure warrants, and
			// mixing its rows then plants entries at distance mm-1+work.band — so the bulge compounds
			// along the chase and runs off the allocation. sbSet caught exactly that
			// ("would write ... distance 12 beyond the allocated band 11").
			//
			// bw is sufficient: below the window nothing is further out than bw, because the only
			// sub-band mass is the bulge being annihilated, and that lies in the window's own rows
			// against the PREVIOUS window's columns — which the left panel covers, since m <= bw
			// whenever a sweep at most halves the bandwidth.
			sbrSimilarity(work, w0, mm, bw, q)
			stripApplyQ(za, w0, mm, backend.Mat{Rows: mm, Cols: mm, Data: q})
			src0, srcN = w0, mm
		}
	}

	// Tighten, asserting that nothing real is left outside the target band.
	out := newBandStorage(n, b2)
	for c := range n {
		for t := 0; t <= work.band && c+t < n; t++ {
			v := work.at(c, t)
			if v == 0 {
				continue
			}
			if t > b2 {
				if math.Abs(v) > sbTol {
					panic(fmt.Sprintf("lanczos: sbr sweep %d->%d left %g at distance %d",
						bw, b2, v, t))
				}
				continue
			}
			out.set(c, t, v)
		}
	}
	return out
}

// sbrPanelQ returns the mm x mm orthogonal factor of the QR of the window panel
// work[w0:w0+mm, src0:src0+sn].
func sbrPanelQ(work bandStorage, w0, mm, src0, sn int) []float64 {
	p := mat.NewDense(mm, sn, nil)
	for r := range mm {
		for c := range sn {
			p.Set(r, c, sbAt(work, w0+r, src0+c))
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

// sbrSimilarity applies A <- QᵀAQ with Q the identity outside [w0, w0+mm).
//
// Three non-overlapping regions, which is what makes this safe on a symmetric lower-only store: the
// diagonal block R x R, the left panel R x [w0-band, w0), and the bottom panel [w0+mm, w0+mm+band) x R.
// Nothing is touched twice, so there is no risk of applying the transform to the same datum from both
// the row side and the column side.
func sbrSimilarity(work bandStorage, w0, mm, reach int, q []float64) {
	n := work.dim
	lo := max(0, w0-reach)
	hiRow := min(n, w0+mm+reach)

	// Diagonal block: D <- Qᵀ D Q.
	dblk := make([]float64, mm*mm)
	for r := range mm {
		for c := range mm {
			dblk[r*mm+c] = sbAt(work, w0+r, w0+c)
		}
	}
	dblk = gemmTN(q, dblk, mm, mm, mm) // Qᵀ D
	dblk = gemmNN(dblk, q, mm, mm, mm) // (Qᵀ D) Q
	for r := range mm {
		for c := r; c < mm; c++ {
			sbSet(work, w0+r, w0+c, dblk[r*mm+c])
		}
	}

	// Left panel: L <- Qᵀ L, columns strictly left of the window.
	if nl := w0 - lo; nl > 0 {
		l := make([]float64, mm*nl)
		for r := range mm {
			for c := range nl {
				l[r*nl+c] = sbAt(work, w0+r, lo+c)
			}
		}
		l = gemmTN(q, l, mm, mm, nl)
		for r := range mm {
			for c := range nl {
				sbSet(work, w0+r, lo+c, l[r*nl+c])
			}
		}
	}

	// Bottom panel: B <- B Q, rows strictly below the window.
	if nb := hiRow - (w0 + mm); nb > 0 {
		b := make([]float64, nb*mm)
		for r := range nb {
			for c := range mm {
				b[r*mm+c] = sbAt(work, w0+mm+r, w0+c)
			}
		}
		b = gemmNN(b, q, nb, mm, mm)
		for r := range nb {
			for c := range mm {
				sbSet(work, w0+mm+r, w0+c, b[r*mm+c])
			}
		}
	}
}

// gemmNN returns a*b for row-major a (ar x ac) and b (ac x bc).
//
// blas64, not a hand-rolled nest: together with the strip product these carry stage 1's ~1.17e15
// flops, and the panels here are large (the left panel is mm x mm x bw, i.e. 1540 x 1540 x 3079 on the
// first production sweep). BenchmarkStage1Gemm measured gonum's pure-Go Gemm at 159 GFlop/s on that
// shape on 64 cores; the triple loop these replaced ran on one. `-tags openblas` does NOT link on this
// cluster (no -lopenblas/-llapacke), so the untagged rate is the rate that matters.
func gemmNN(a, b []float64, ar, ac, bc int) []float64 {
	out := make([]float64, ar*bc)
	if ar == 0 || ac == 0 || bc == 0 {
		return out
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: ar, Cols: ac, Stride: ac, Data: a},
		blas64.General{Rows: ac, Cols: bc, Stride: bc, Data: b},
		0,
		blas64.General{Rows: ar, Cols: bc, Stride: bc, Data: out})
	return out
}

// gemmTN returns aᵀ*b for row-major a (ar x ac) and b (ar x bc), so the result is ac x bc.
//
// It TRANSPOSES a explicitly rather than passing blas.Trans, because gonum's pure-Go Dgemm is 2.5x
// slower on a transposed operand: measured at the production strip shape (3080x1540x1540),
// blas.Trans gives 9.5 GFlop/s against 23.7 for the same product with the transpose materialized.
// The copy is ar*ac elements against 2*ar*ac*bc flops — 0.016% of the work at m = 1540 — so it is
// free at every shape stage 1 issues. (The strided operand, by contrast, costs ~5% and is left alone.)
func gemmTN(a, b []float64, ar, ac, bc int) []float64 {
	out := make([]float64, ac*bc)
	if ar == 0 || ac == 0 || bc == 0 {
		return out
	}
	return gemmNN(transposed(a, ar, ac), b, ac, ar, bc)
}

// transposed returns the transpose of row-major a (ar x ac) as a fresh ac x ar slice.
func transposed(a []float64, ar, ac int) []float64 {
	out := make([]float64, ac*ar)
	for r := range ar {
		row := a[r*ac : (r+1)*ac]
		for c, v := range row {
			out[c*ar+r] = v
		}
	}
	return out
}
