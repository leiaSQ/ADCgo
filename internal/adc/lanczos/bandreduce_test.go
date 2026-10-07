package lanczos

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// TestBandReduceRefComposition is the Phase 1 decision gate for the two-stage reduction: narrowing
// the band first and then running the existing chase at the smaller bandwidth must reproduce the
// single-stage answer.
//
// It checks the two things that can go wrong independently. Eigenvalues are self-checking against
// the frozen serial reference. The eigenvector strip is NOT — sign and degenerate-subspace rotation
// are both legitimate — so it is checked against the reference-free invariant S·Sᵀ = I, which holds
// for any orthogonal transform and catches a dropped rotation or a botched rescaling.
//
// Stage 1 here is bnd2td's own chase stopped at sub-diagonal b2+1, so this gate isolates the
// COMPOSITION and the scaled-representation conversion, not new arithmetic. A faster GEMM-based
// stage 1 is then validated against this.
func TestBandReduceRefComposition(t *testing.T) {
	for _, tc := range []struct{ dim, band, b2 int }{
		{40, 7, 2}, {40, 7, 4}, {60, 12, 3}, {80, 12, 6}, {200, 20, 4}, {260, 40, 8},
	} {
		main := (tc.band + 1) / 2
		if 2*main > tc.dim {
			continue
		}
		seed := int64(9000 + tc.dim*10 + tc.band)

		_, ref := buildBanded(tc.dim, tc.band, seed)
		wantD, _ := bandSymDiagFastRef(ref)

		// Stage 1, then the existing solver at the narrowed bandwidth. The accumulator has to be
		// shared: stage 1 seeds it and leaves Eᵀ·U1 in it, and the second stage must continue from
		// there rather than reseed, which is what zAccum.restore is for.
		_, bs := buildBanded(tc.dim, tc.band, seed)
		pool := parallel.NewFixedPool(1)
		defer pool.Close()
		o := bandEigOpts{topRows: main, botRows: main, workers: 1}
		za := newZAccum(tc.dim, o, pool)
		narrow := bandReduceRef(bs, tc.b2, za, pool)
		if narrow.band != tc.b2 {
			t.Fatalf("dim=%d band=%d b2=%d: stage 1 produced band %d", tc.dim, tc.band, tc.b2, narrow.band)
		}
		z1 := append([]float64(nil), za.z...)

		gotD, z, ld, _ := bandSymDiagFastOpts(narrow, bandEigOpts{
			topRows: main, botRows: main, workers: 1, resumeZ: z1,
		})

		// Eigenvalues: backward-stable, so an absolute bound scaled by the norm, not a relative one.
		nrm := 0.0
		for _, v := range wantD {
			if a := math.Abs(v); a > nrm {
				nrm = a
			}
		}
		for k := range wantD {
			if e := math.Abs(gotD[k] - wantD[k]); e > 1e-10*nrm {
				t.Errorf("dim=%d band=%d b2=%d: eval[%d] = %.15g, single-stage %.15g (Δ%.2e, ‖A‖≈%.3g)",
					tc.dim, tc.band, tc.b2, k, gotD[k], wantD[k], e, nrm)
				break
			}
		}
		if got := stripOrthErr(z, 2*main, ld, tc.dim); got > 1e-11 {
			t.Errorf("dim=%d band=%d b2=%d: max|S·Sᵀ-I| = %.3e, want <= 1e-11",
				tc.dim, tc.band, tc.b2, got)
		}
	}
}

// buildBlockTri makes a random symmetric block-tridiagonal projected matrix in the driver's own
// representation, plus the equivalent bandStorage, so the block-QR reduction can be compared against
// solving the same matrix directly.
func buildBlockTri(nb, b int, seed int64) ([]lmBlock, bandStorage) {
	rng := rand.New(rand.NewSource(seed))
	dim := nb * b
	blocks := make([]lmBlock, nb)
	for bi := range nb {
		al := make([]float64, b*b)
		for q := range b {
			for p := q; p < b; p++ {
				v := rng.NormFloat64()
				al[q*b+p] = v // column-major: alpha[q*b+p] = T[off+p][off+q]
				al[p*b+q] = v
			}
		}
		blk := lmBlock{off: bi * b, size: b, alpha: al, beta: backend.NewMat(0, 0)}
		if bi > 0 {
			be := backend.NewMat(b, b)
			for q := range b {
				for p := range b {
					be.Set(q, p, rng.NormFloat64())
				}
			}
			blk.beta = be
		}
		blocks[bi] = blk
	}
	bs := newBandStorage(dim, min(2*b-1, dim-1))
	fillBand(bs, blocks)
	return blocks, bs
}

// TestBlockQRReduceHalvesBandwidth is the gate on the GEMM stage 1. It checks the three things that
// can independently be wrong, on the same principle as the Phase 1 gate:
//
//	the bandwidth really drops to b (otherwise the chase gains nothing),
//	the eigenvalues are unchanged (it is a similarity, so they must be — to roundoff), and
//	the strip stays orthonormal (S*S^T = I), which is reference-free and is what catches a transform
//	  applied to the wrong block, the wrong side, or with Q instead of Q^T.
//
// The last one carries the weight: a mis-sided Q still produces a plausible spectrum, because
// Q^T*A*Q and Q*A*Q^T are both similarities — only the accumulated strip distinguishes them.
func TestBlockQRReduceHalvesBandwidth(t *testing.T) {
	for _, tc := range []struct{ nb, b int }{{4, 3}, {6, 5}, {5, 8}, {8, 4}, {3, 16}} {
		dim := tc.nb * tc.b
		main := tc.b
		if 2*main > dim {
			continue
		}
		seed := int64(5500 + tc.nb*100 + tc.b)

		_, direct := buildBlockTri(tc.nb, tc.b, seed)
		wantD, _ := bandSymDiagFastRef(direct)

		blocks, _ := buildBlockTri(tc.nb, tc.b, seed)
		pool := parallel.NewFixedPool(1)
		defer pool.Close()
		o := bandEigOpts{topRows: main, botRows: main, workers: 1}
		za := newZAccum(dim, o, pool)
		narrow := blockQRReduce(blocks, dim, za)

		if narrow.band > tc.b {
			t.Errorf("nb=%d b=%d: half-bandwidth %d after stage 1, want <= %d (was %d)",
				tc.nb, tc.b, narrow.band, tc.b, 2*tc.b-1)
		}
		gotD, z, ld, _ := bandSymDiagFastOpts(narrow, bandEigOpts{
			topRows: main, botRows: main, workers: 1, resumeZ: append([]float64(nil), za.z...),
		})
		nrm := 0.0
		for _, v := range wantD {
			nrm = math.Max(nrm, math.Abs(v))
		}
		for k := range wantD {
			if e := math.Abs(gotD[k] - wantD[k]); e > 1e-9*nrm {
				t.Errorf("nb=%d b=%d: eval[%d] = %.12g, direct %.12g (Δ%.2e, ‖A‖≈%.3g)",
					tc.nb, tc.b, k, gotD[k], wantD[k], e, nrm)
				break
			}
		}
		if e := stripOrthErr(z, 2*main, ld, dim); e > 1e-10 {
			t.Errorf("nb=%d b=%d: max|S·Sᵀ-I| = %.3e, want <= 1e-10", tc.nb, tc.b, e)
		}
	}
}

// BenchmarkStripApplyQ measures the real call rather than a synthetic GEMM of the same shape.
//
// BenchmarkStage1Gemm establishes what blas64.Gemm can do; this establishes what stripApplyQ actually
// gets, which is not the same question. The strip is COLUMN-major with leading dimension za.ld, so the
// product is issued with a transposed left operand and a strided right one, and it pays a copy back
// into the strip's columns because blas64 forbids C aliasing A or B. If gonum declines to thread a
// strided General, or the copy dominates, that shows up here and nowhere else.
//
// rows = 3080 is production's accumulated width. m = 1540 is the first sweep's window at the melanin
// shape (bandwidth 3079 halving to 1540); m = 192 is the last.
func BenchmarkStripApplyQ(b *testing.B) {
	for _, m := range []int{1540, 192} {
		b.Run(fmt.Sprintf("rows=3080/m=%d", m), func(b *testing.B) {
			const rows = 3080
			// The strip cannot be taller than the dimension (newZAccum enforces it), so n is driven by
			// rows, not by the window; the extra columns just keep off away from 0.
			n := max(m+64, rows+64)
			pool := parallel.NewFixedPool(1)
			defer pool.Close()
			za := newZAccum(n, bandEigOpts{topRows: rows / 2, botRows: rows / 2, workers: 1,
				batch: 1 << 16}, pool)
			rng := rand.New(rand.NewSource(11))
			for i := range za.z {
				za.z[i] = rng.NormFloat64()
			}
			q := backend.NewMat(m, m)
			for i := range q.Data {
				q.Data[i] = rng.NormFloat64()
			}
			b.SetBytes(int64(2 * rows * m * m)) // read the MB/s column as FLOP/s
			for b.Loop() {
				stripApplyQ(za, 32, m, q)
			}
		})
	}
}

// BenchmarkStage1Gemm is the gate on whether SBR stage 1 is viable at all in this package.
//
// `internal/adc/lanczos` builds with NO build tag, so `blas64.Gemm` resolves to gonum's pure-Go
// implementation unless `-tags openblas` swaps the engine in `backend/openblas.go`'s init. Stage 1 is
// ~1.17e15 flops, split evenly between the band update and the strip update, so the achieved GEMM rate
// decides between **1.9 min on a device, ~49 min threaded, and 6.5 h single-threaded** — i.e. between
// a stage that disappears and one that costs more than the chase it replaces.
//
// The shapes are the ones stage 1 actually issues: the large band and strip products of the first
// sweep, and the small ones of the last, where per-call overhead starts to dominate. Run it twice on a
// compute node, plain and with -tags openblas, and compare GFlop/s.
func BenchmarkStage1Gemm(b *testing.B) {
	for _, s := range []struct {
		name    string
		m, k, n int
	}{
		{"band-first-sweep/1540x1540x4620", 1540, 1540, 4620},
		{"strip-first-sweep/3080x1540x1540", 3080, 1540, 1540},
		{"band-last-sweep/192x192x576", 192, 192, 576},
		{"strip-last-sweep/3080x192x192", 3080, 192, 192},
	} {
		b.Run(s.name, func(b *testing.B) {
			A := backend.NewMat(s.m, s.k)
			B := backend.NewMat(s.k, s.n)
			for i := range A.Data {
				A.Data[i] = float64(i%97) * 0.01
			}
			for i := range B.Data {
				B.Data[i] = float64(i%89) * 0.01
			}
			C := backend.NewMat(s.m, s.n)
			ag, bg, cg := matGeneral(A), matGeneral(B), matGeneral(C)
			flops := 2 * float64(s.m) * float64(s.k) * float64(s.n)
			b.SetBytes(int64(flops)) // read the MB/s column as FLOP/s
			for b.Loop() {
				blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, ag, bg, 0, cg)
			}
		})
	}
}

// matGeneral views a backend.Mat (row-major, stride == Cols) as a blas64.General.
func matGeneral(m backend.Mat) blas64.General {
	return blas64.General{Rows: m.Rows, Cols: m.Cols, Stride: m.Cols, Data: m.Data}
}
