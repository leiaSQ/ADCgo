package lanczos

// bandreduce.go — stage 1 of a two-stage reduction: narrow a symmetric band matrix from
// half-bandwidth `band` to `b2`, carrying the eigenvector strip along, so that the chase in
// bandeig.go then runs at `b2` instead of `band`.
//
// WHY. The chase's two costs scale differently, and that asymmetry is the whole reason this file
// exists. Its outer loops give `~dim²/2` steps regardless of the bandwidth (the `r` loop has `m1`
// iterations and each chase has `(dim-k)/m1` steps, so `m1` cancels), and each step does
//
//	one eigenvector rotation over the retained strip — cost independent of the bandwidth, and
//	~2*band band-matrix updates                     — cost proportional to it.
//
// Measured at the production shape (dim 308000, band 3079): 5.84e14 band updates (~63 h) against
// 4.38e14 strip updates (~36 h). Narrowing the band 3079 -> 64 therefore cuts the first by 48x and
// leaves the second untouched, which is what turns a ~100 h stage into single-digit hours.
//
// WHY THE STRIP SURVIVES. Writing the stages as A = U1ᵀ·A1·U1, A1 = U2ᵀ·T·U2, T = U3·Λ·U3ᵀ, the full
// eigenvectors are Q = U1·U2·U3 and the requested strip is
//
//	S = Eᵀ·Q = (Eᵀ·U1)·U2·U3
//
// for the selector E of the retained rows. Every factor is applied FROM THE RIGHT to a short, fat
// matrix, so no dim x dim object is ever formed — which is exactly the contract zAccum already
// implements. Stage 1 hands its result over through zAccum.restore, the entry point the checkpoint
// resume path already uses, and bnd2td/tddiag then run unchanged.
//
// WHAT IS HERE. bandReduceRef is the REFERENCE stage 1: it reuses bnd2td's own chase, stopped early,
// so it introduces no new numerics at all. It is not faster than the chase it replaces — it does the
// same Givens work — but it makes the two-stage composition provable before a fast, GEMM-based
// stage 1 is written, and it is what that faster version will be validated against.

import (
	"fmt"
	"math"
	"os"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// bandReduceRef narrows bs from its own half-bandwidth to b2, applying the same transform to za, and
// returns the narrowed matrix. za must be freshly constructed: bnd2td seeds it, so anything already
// accumulated would be overwritten.
//
// This is the reference implementation. It reaches the target bandwidth by running bnd2td's chase and
// stopping it at sub-diagonal b2+1 rather than 1, then converting the result out of the chase's
// scaled representation. Correct by construction, and no faster than the work it replaces.
func bandReduceRef(bs bandStorage, b2 int, za *zAccum, pool *parallel.FixedPool) bandStorage {
	dim := bs.dim
	if b2 < 1 {
		panic(fmt.Sprintf("lanczos: bandReduceRef needs b2 >= 1, got %d", b2))
	}
	if b2 >= bs.band {
		// Already narrow enough. Seed the accumulator so the caller's contract still holds: the
		// transform is the identity, which is what an unrotated seed represents.
		za.seed()
		return bs
	}
	b1 := bs.band + 1
	af := packBandColumnMajor(bs)
	d := make([]float64, dim)
	e := make([]float64, dim)
	e2 := make([]float64, dim)
	bnd2td(za, dim, b1, af, d, e, e2, pool, 0, nil, nil, 0, nil, b2)
	return unscaleBand(af, d, dim, b1, b2, za)
}

// packBandColumnMajor lays bs out the way bnd2td wants it: the Fortran a(n,mb) indexing stored
// row-major, A(row,col) 1-based at af[(row-1)*mb + col-1]. Band entry i of column j is matrix row
// i+j+1, band column mb-i (lanczos_util.cpp:114-116).
//
// Shared with bandSymDiagFastOpts, which does the same repack; kept here as a function so the two
// cannot drift.
func packBandColumnMajor(bs bandStorage) []float64 {
	b1 := bs.band + 1
	af := make([]float64, b1*bs.dim)
	parallel.Chunks(bs.dim, parallel.ChunkWorkers(bs.dim), func(_, lo, hi int) {
		for r := lo; r < hi; r++ {
			row := af[r*b1 : (r+1)*b1]
			hiI := min(b1-1, r)
			for i := 0; i <= hiI; i++ {
				row[b1-1-i] = bs.at(r-i, i)
			}
		}
	})
	return af
}

// unscaleBand converts a chase stopped at half-bandwidth b2 out of its scaled representation into a
// plain bandStorage, and applies the matching column scaling to the accumulator.
//
// The chase works on a scaled matrix: the true element is A[i][j] = e_i*e_j*a(i,j) with
// e_j = sqrt(d_j), and the accumulator's column j carries a factor e_j. This is the same conversion
// label 800 of bnd2td performs on the tridiagonal (`e[j-1] = sqrt(d[j-1])`, then `za.scaleAll(e)`,
// then `A(j,m1) *= e_{j-1}*e_j` and `A(j,mb) *= d_j`) — generalized from two diagonals to b2+1.
//
// e_1 is 1, never sqrt(d_1): the chase never writes d[0] (its innermost index is d[j1-1] with
// j1 >= 2), and label 800 correspondingly leaves row 1 and column 1 unscaled. Applying sqrt(d_1)
// here would be a silent factor on the first row of the matrix and the first eigenvector.
func unscaleBand(af, d []float64, dim, mb, b2 int, za *zAccum) bandStorage {
	e := make([]float64, dim)
	e[0] = 1
	for j := 2; j <= dim; j++ {
		e[j-1] = math.Sqrt(d[j-1])
	}
	out := newBandStorage(dim, b2)
	// Matrix element (i, i-t) sits at band column mb-t of row i, and bandStorage.set(col, t, v)
	// stores element [col+t][col] — so column i-t, offset t.
	parallel.Chunks(dim, parallel.ChunkWorkers(dim), func(_, lo, hi int) {
		for i0 := lo; i0 < hi; i0++ {
			i := i0 + 1 // 1-based matrix row
			row := af[i0*mb : (i0+1)*mb]
			for t := 0; t <= b2 && t < i; t++ {
				j := i - t // 1-based matrix column
				out.set(j-1, t, e[i-1]*e[j-1]*row[mb-t-1])
			}
		}
	})
	// The accumulator's column scaling, exactly as label 800 applies it: column 1 is left alone.
	za.scaleAll(e)
	za.flush()
	return out
}

// blockQRReduce halves the projected matrix's half-bandwidth by triangularizing every off-diagonal
// block, and applies the same orthogonal similarity to the eigenvector strip. It is the GEMM-rich
// stage 1: ~6e12 flops of dense b x b products at the production shape, against the ~5.8e14 scalar
// band updates it removes from the chase.
//
// THE ALGEBRA, which is why this is exact rather than heuristic. The projected matrix is block
// tridiagonal: diagonal blocks alpha_i and off-diagonal blocks beta_i coupling block i to block i-1,
// with beta_i.At(q,p) = T[off_i+q][off_{i-1}+p] (fillBand). QR-factorize beta_i = Q_i * R_i and apply
// the similarity that is the identity except Q_i on block i's indices:
//
//	beta_i     <- Q_i^T * beta_i = R_i        (upper trapezoidal)
//	alpha_i    <- Q_i^T * alpha_i * Q_i
//	beta_{i+1} <- beta_{i+1} * Q_i           (its columns ARE block i)
//	Z[:, block i] <- Z[:, block i] * Q_i     (the strip, right-multiplied as always)
//
// R_i[q][p] = 0 for q > p, and element T[off_i+q][off_{i-1}+p] sits at row-column distance
// size_{i-1} + q - p, so a nonzero needs q <= p and the distance is at most size_{i-1}. The diagonal
// blocks contribute size_i - 1. The half-bandwidth therefore drops from 2b-1 to b — exactly a factor
// of two, and the band work in the chase is proportional to it.
//
// WHY IT STOPS AT b, and what would be needed to go below. The diagonal blocks stay dense, so their
// own size-1 bandwidth is a floor for any similarity that is block-diagonal like this one. Reaching a
// genuinely narrow band (b2 ~ 64) means reducing WITHIN the diagonal blocks too, which is the general
// successive-band-reduction problem with bulge chasing — deliberately not attempted here, because its
// index bookkeeping is exactly the kind whose errors are silent.
//
// SHAPE PRECONDITION. Every beta must be exactly (size_i x size_{i-1}). That is the documented
// layout, but it is not guaranteed: the driver appends `lmBlock{size: rank, beta: r}`, so when
// deflation drops the rank below the block width, beta's stored extent and the block's size can
// disagree — and sizing Q from the wrong one silently applies a transform on the wrong index set.
// blockQRReduceOK reports whether the shapes are the ones this reduction understands; callers must
// check it and skip stage 1 rather than proceed, because the failure is a wrong spectrum, not a
// crash. (It was in fact a crash here, a MatMul shape mismatch, only because the mismatch happened to
// be visible one step later.)
//
// blocks is modified in place, so the caller must pass a copy if it needs the original. za must be
// freshly constructed; this seeds it.
func blockQRReduce(blocks []lmBlock, dim int, za *zAccum) bandStorage {
	if !blockQRReduceOK(blocks) {
		panic("lanczos: blockQRReduce called on block shapes it cannot handle; check blockQRReduceOK")
	}
	za.seed()
	for bi := 1; bi < len(blocks); bi++ {
		b := blocks[bi]
		if b.beta.Rows == 0 {
			continue
		}
		q := qrQ(b.beta) // size_i x size_i, explicit
		m := b.size

		// beta_i <- Q^T beta_i, alpha_i <- Q^T alpha_i Q.
		//
		// The strictly-lower part of R is zero BY CONSTRUCTION but not numerically — QR leaves ~1e-17
		// there — and it has to be zeroed explicitly, because fillBand writes a beta entry at band
		// index prev.size+q-p, which for q > p exceeds the reduced bandwidth. bandStorage.set is
		// unchecked, so those residues land in the NEXT column's slots and silently overwrite real
		// entries. That showed up as eigenvalues wrong in the first significant figure.
		r := backend.MatMul(transpose(q), b.beta)
		for row := range r.Rows {
			for col := 0; col < row && col < r.Cols; col++ {
				r.Set(row, col, 0)
			}
		}
		blocks[bi].beta = r
		al := alphaMat(b)
		blocks[bi].alpha = matToColMajor(backend.MatMul(backend.MatMul(transpose(q), al), q))

		// beta_{i+1}'s COLUMNS are block i, so it is right-multiplied.
		if bi+1 < len(blocks) && blocks[bi+1].beta.Rows > 0 {
			blocks[bi+1].beta = backend.MatMul(blocks[bi+1].beta, q)
		}

		// The strip: columns [off, off+m) of the accumulator, right-multiplied by Q. za.z is
		// column-major with leading dimension ld, so a column range is contiguous.
		stripApplyQ(za, b.off, m, q)
	}

	// The reduced matrix, whose half-bandwidth is now bounded by the widest block coupling.
	band := 0
	for bi := range blocks {
		band = max(band, blocks[bi].size-1)
		if bi > 0 {
			band = max(band, blocks[bi-1].size)
		}
	}
	band = min(band, max(dim-1, 0))
	out := newBandStorage(dim, band)
	fillBandNarrowed(out, blocks)
	return out
}

// fillBandNarrowed is fillBand for a matrix whose off-diagonal blocks have been triangularized: it
// writes only the entries that fall inside the narrowed band, and refuses to discard anything that
// is not numerically zero.
//
// fillBand itself cannot be used here. It writes every (q,p) pair of a beta block at band index
// prev.size+q-p, which for q > p exceeds the narrowed bandwidth — and bandStorage.set is unchecked,
// so the write lands in the NEXT column's slots and destroys a real entry. Zeroing the triangle first
// does not help: it is the write that corrupts, not the value. That cost an hour and showed up as
// eigenvalues wrong in the first significant figure while a full-width rebuild of the very same
// blocks reproduced them to 4e-15, which is what localized it.
//
// The panic is the point. Silent truncation here is a wrong spectrum; a loud one is a bug report.
func fillBandNarrowed(bs bandStorage, blocks []lmBlock) {
	const tol = 1e-12
	for bi := range blocks {
		blk := blocks[bi]
		a := blk.alpha
		s := blk.size
		for q := range s {
			for p := q; p < s; p++ {
				if p-q > bs.band {
					if v := a[q*s+p]; math.Abs(v) > tol {
						panic(fmt.Sprintf("lanczos: narrowed band %d drops alpha[%d][%d] = %g in block %d",
							bs.band, p, q, v, bi))
					}
					continue
				}
				bs.set(blk.off+q, p-q, a[q*s+p])
			}
		}
		if blk.beta.Rows > 0 {
			prev := blocks[bi-1]
			for q := range blk.beta.Rows {
				for p := range blk.beta.Cols {
					i := prev.size + q - p
					if i > bs.band {
						if v := blk.beta.At(q, p); math.Abs(v) > tol {
							panic(fmt.Sprintf("lanczos: narrowed band %d drops beta[%d][%d] = %g in "+
								"block %d (band index %d)", bs.band, q, p, v, bi, i))
						}
						continue
					}
					bs.set(prev.off+p, i, blk.beta.At(q, p))
				}
			}
		}
	}
}

// blockQRReduceOK reports whether every block's alpha and beta have the extent blockQRReduce assumes:
// alpha is size x size and beta is exactly size_i x size_{i-1}. Deflation can break the second.
func blockQRReduceOK(blocks []lmBlock) bool {
	for bi := range blocks {
		b := blocks[bi]
		if len(b.alpha) != b.size*b.size {
			return false
		}
		if bi == 0 {
			continue
		}
		if b.beta.Rows == 0 {
			continue
		}
		if b.beta.Rows != b.size || b.beta.Cols != blocks[bi-1].size {
			return false
		}
	}
	return true
}

// qrQ returns the explicit m x m orthogonal factor of a's QR factorization.
//
// Explicit Q rather than the compact WY form: it is m x m = 19 MB at the production block size,
// which makes every subsequent update a plain dense product that backend.MatMul (and hence a device
// GEMM) handles directly, and it keeps the algebra above readable. The factorization itself is a
// rounding error of the cost it saves.
func qrQ(a backend.Mat) backend.Mat {
	d := mat.NewDense(a.Rows, a.Cols, append([]float64(nil), a.Data...))
	var qr mat.QR
	qr.Factorize(d)
	var q mat.Dense
	qr.QTo(&q)
	out := backend.NewMat(a.Rows, a.Rows)
	for i := range a.Rows {
		for j := range a.Rows {
			out.Set(i, j, q.At(i, j))
		}
	}
	return out
}

func transpose(a backend.Mat) backend.Mat {
	out := backend.NewMat(a.Cols, a.Rows)
	for i := range a.Rows {
		for j := range a.Cols {
			out.Set(j, i, a.At(i, j))
		}
	}
	return out
}

// alphaMat materializes a block's alpha as a full symmetric row-major Mat.
//
// It reads ONLY the lower triangle and mirrors it. fillBand also reads only p >= q
// (`for p := q; p < s; p++`), so nothing guarantees the stored upper triangle holds anything
// meaningful — reading it would feed whatever the solver happened to leave there into the reduction.
func alphaMat(b lmBlock) backend.Mat {
	s := b.size
	out := backend.NewMat(s, s)
	for q := range s {
		for p := q; p < s; p++ {
			v := b.alpha[q*s+p] // column-major: alpha[q*s+p] = T[off+p][off+q], p >= q
			out.Set(p, q, v)
			out.Set(q, p, v)
		}
	}
	return out
}

// matToColMajor is alphaMat's inverse: back into lmBlock.alpha's column-major layout.
func matToColMajor(m backend.Mat) []float64 {
	s := m.Rows
	out := make([]float64, s*s)
	for q := range s {
		for p := range s {
			out[q*s+p] = m.At(p, q)
		}
	}
	return out
}

// stripApplyQ right-multiplies columns [off, off+m) of the strip by the m x m matrix q.
//
// THIS MUST BE A GEMM, not an axpy loop. It carries half of stage 1's ~1.17e15 flops (the strip term
// is 2*rows*dim^2, independent of both the block width and the bandwidth), so the achieved rate is the
// difference between stage 1 disappearing and stage 1 costing more than the chase it exists to remove.
// blas64.Gemm was measured at 178 GFlop/s on 64 cores at this exact shape (3080x1540x1540,
// BenchmarkStage1Gemm, job 15048404) against the single-threaded axpy nest that used to be here.
//
// THE TRANSPOSE. The strip is COLUMN-major with leading dimension za.ld, so the block
// Z[:, off:off+m] viewed as a row-major blas64.General with Stride = za.ld is its TRANSPOSE: an
// m x rows matrix S with S[k][r] = Z[r][off+k]. The update Z <- Z*q is therefore S <- q^T * S, which
// is why q is the LEFT factor and why it is transposed. Getting this backwards applies q^T and is
// silently orthogonal, so it does not fail loudly — S*S^T = I still holds. What catches it is
// TestSBRBandMatchesDense, which compares the strip against explicitly formed rows of the dense
// reference's accumulated transform.
//
// q^T is MATERIALIZED rather than passed as blas.Trans. gonum's pure-Go Dgemm is 2.5x slower on a
// transposed operand (9.5 vs 23.7 GFlop/s at this exact shape), and passing Trans is what held the
// real call to 68.2 GFlop/s on 64 cores where the same product reaches 158.6 (job 15058672,
// BenchmarkStripApplyQ against BenchmarkStage1Gemm/strip-first-sweep). The transpose is m*m copies
// against 2*rows*m*m flops — 0.016% of the work at m = 1540. The STRIDED operand costs only ~5% and
// is left as it is.
func stripApplyQ(za *zAccum, off, m int, q backend.Mat) {
	if za.rows == 0 || m == 0 {
		return
	}
	za.flush()
	qt := make([]float64, m*m)
	for r := range m {
		row := q.Data[r*q.Cols:][:m]
		for c, v := range row {
			qt[c*m+r] = v
		}
	}
	// C aliases neither operand, which blas64.Gemm requires; the copy back is the price.
	buf := make([]float64, m*za.rows)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: m, Cols: m, Stride: m, Data: qt},
		blas64.General{Rows: m, Cols: za.rows, Stride: za.ld, Data: za.z[off*za.ld:]},
		0,
		blas64.General{Rows: m, Cols: za.rows, Stride: za.rows, Data: buf})
	for k := range m {
		copy(za.z[(off+k)*za.ld:][:za.rows], buf[k*za.rows:(k+1)*za.rows])
	}
}

// narrowProjected runs stage 1: it narrows the projected matrix's half-bandwidth toward b2 and
// returns the narrowed matrix together with the strip the reduction accumulated, ready to be handed
// to bandSymDiagFastOpts as bandEigOpts.resumeZ.
//
// It reduces as far as the block structure allows — about the Krylov block width, a factor of two —
// and REPORTS rather than silently accepts a target it cannot reach. Going below the block width
// requires reducing within the diagonal blocks, i.e. general successive band reduction with bulge
// chasing, which is not implemented here.
//
// blocks is copied, so the caller's own α/β (which are the projected matrix, and the answer in their
// own right) are untouched.
func narrowProjected(blocks []lmBlock, dim, b2 int, o bandEigOpts, bs bandStorage) (bandStorage, []float64) {
	// Stage 1 is dense GEMM work with no deferred-op replay, so it needs no worker pool of its own.
	pool := parallel.NewFixedPool(1)
	defer pool.Close()

	// accel deliberately cleared. Stage 1 is host GEMM work that mutates za.z directly, so a device
	// accumulator would be allocated by seed()/activate() and then never used — 7.6 GB per sector at
	// the production shape, leaked, with bandSymDiagFastOpts allocating a second one from resumeZ
	// immediately afterwards. Worse, while a device accumulator is live the host z is by contract
	// STALE, and stripApplyQ writes the host copy: that is correct today only because the tape happens
	// to be empty at every call, which nothing enforces. Clearing it removes both hazards.
	so := o
	so.accel = nil
	za := newZAccum(dim, so, pool)

	// Two reductions, chained, because they are good at different things.
	//
	// blockQRReduce exploits the BLOCK-tridiagonal structure: one QR per off-diagonal block halves the
	// bandwidth (2b-1 -> b) in a handful of large GEMMs, far cheaper per unit of narrowing than
	// anything general. It cannot go further, because the diagonal blocks stay dense.
	//
	// sbrSweepTo then narrows to the target by successive band reduction. It reads only bandStorage, so
	// unlike blockQRReduce it is immune to the deflation shape mismatch — which makes it the fallback
	// when that guard fires: one extra sweep from the full bandwidth, rather than losing stage 1
	// entirely and leaving the chase at 3079.
	narrow := bs
	if blockQRReduceOK(blocks) {
		narrow = blockQRReduce(copyBlocks(blocks), dim, za)
	} else {
		fmt.Fprintf(os.Stderr, "adcgo: banded eigensolve stage 1: the accepted blocks are not the shape "+
			"the block-tridiagonal reduction needs (deflation left a beta whose extent and block size "+
			"disagree); narrowing from the full bandwidth instead\n")
		za.seed()
	}
	if narrow.band > b2 {
		narrow = sbrSweepTo(narrow, b2, za)
	}
	za.flush()
	if narrow.band > b2 {
		fmt.Fprintf(os.Stderr, "adcgo: banded eigensolve stage 1 reached half-bandwidth %d, not the "+
			"requested %d\n", narrow.band, b2)
	}
	return narrow, za.z
}

// sbrSweepTo runs the successive-band-reduction schedule on an ALREADY-SEEDED accumulator, which is
// what distinguishes it from sbrReduce: the caller has just run blockQRReduce into the same
// accumulator and must not have it reseeded, or the transform accumulated so far is lost.
func sbrSweepTo(cur bandStorage, b2 int, za *zAccum) bandStorage {
	for cur.band > b2 && cur.band > 1 {
		next := max(b2, (cur.band+1)/2)
		if next >= cur.band {
			break
		}
		cur = sbrSweep(cur, next, za)
	}
	return cur
}

// copyBlocks deep-copies the per-block α and β so a reduction can work in place without destroying
// the caller's projected matrix.
func copyBlocks(in []lmBlock) []lmBlock {
	out := make([]lmBlock, len(in))
	for i, b := range in {
		c := b
		c.alpha = append([]float64(nil), b.alpha...)
		if b.beta.Rows > 0 {
			c.beta = backend.Mat{Rows: b.beta.Rows, Cols: b.beta.Cols,
				Data: append([]float64(nil), b.beta.Data...)}
		}
		out[i] = c
	}
	return out
}
