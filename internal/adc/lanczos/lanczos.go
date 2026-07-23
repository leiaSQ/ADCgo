// Package lanczos is the block-Lanczos eigensolver for the DIP-ADC(2) secular
// problem. It sweeps the whole photoionization band (not a few extremal roots),
// which is what a DIP spectrum needs, using the same start vectors as theADCcode
// (../ADC): the Cartesian unit vectors spanning the 2h "main" space.
//
// The implementation is a block-Krylov Rayleigh–Ritz with full reorthogonaliza-
// tion and deflation: it builds an orthonormal basis of
// span{Q0, M·Q0, M²·Q0, …} (Q0 = the main-space unit vectors), projects M onto
// it, and diagonalizes the small dense projection. Because Q0 is exactly the
// main space, the main-space-weighted states (the ones carrying pole strength)
// converge quickly. As the subspace grows to full dimension the Ritz pairs
// become exact, so the driver is validated against the dense path.
//
// The projected matrix is diagonalized densely rather than via the reference's
// Fortran banded solver (bnd2td.f/tddiag.f): the subspace is small, so a band
// solver buys nothing here (it is an M2/M3 concern for very large cases).
package lanczos

import (
	"math"
	"time"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// Timing accumulates the wall time of each phase of Solve. The phases are
// disjoint and sum to the solve time. It exists to keep the optimization work
// honest: the BLAS level of a phase, not its flop count, decides its cost.
type Timing struct {
	Apply time.Duration // M·q mat-vecs
	Orth  time.Duration // reorthogonalization of the new block against the basis
	Proj  time.Duration // building the projected matrix T
	Eig   time.Duration // SymEig(T)
	Back  time.Duration // Ritz back-transform to main-space components
}

// Total returns the summed phase time.
func (t Timing) Total() time.Duration { return t.Apply + t.Orth + t.Proj + t.Eig + t.Back }

// Operator is a real-symmetric matrix applied matrix-free on backend-resident
// vectors, with a distinguished "main" block (its leading MainBlockSize rows)
// whose squared eigenvector weight is the pole strength. *dip.Matrix satisfies
// this.
type Operator interface {
	ApplyFull(out, in backend.Vector)
	// ApplyBlock applies M to every column of in at once. Equivalent to calling
	// ApplyFull per column, but it streams the assembled operator once per block
	// rather than once per vector — the difference between a bandwidth/overhead-bound
	// GEMV and a compute-bound GEMM.
	ApplyBlock(out, in backend.BlockView)
	Size() int
	MainBlockSize() int
}

// Options tunes the block-Krylov build.
//
// MaxBlocks counts the blocks in the Krylov basis, start block included, so the subspace
// it spans is MaxBlocks × MainBlockSize() columns. This is exactly theADCcode's `iter`
// input keyword: there, `iter N` runs N+1 Iterate() calls (the first only registers the
// start block), reaching dim = (N+1)·block, and Diagonalize() then sets
// dimd = dim − block = N·block (../ADC/libLanczos/lanczos.h:226-238, :257). The trailing
// block exists only to supply the coupling for the Ritz residuals, which is what the
// discarded orthogonalization in Solve reproduces. So `-blocks N` == `iter N`, and the
// reference's printed "Size of Lanczos space" is N·block.
type Options struct {
	MaxBlocks int     // blocks in the basis, start block included (0 → until deflation/full)
	MaxDim    int     // cap on subspace dimension (0 → Size())
	DeflTol   float64 // deflation threshold for new basis vectors (0 → 1e-8)
	// WantFull retains the satellite rows of each Ritz vector too, not just the main
	// block, at the price of an n×dim back-transform and an n×dim host matrix. Only the
	// transition-moment machinery needs them; the spectrum does not. SolveDense ignores
	// this — the dense eigenvectors are already full, so it always returns them.
	WantFull bool

	// The following tune the block-Davidson driver (SolveDavidson) only; Solve ignores
	// them. MaxDim doubles as the maximum subspace dimension before a thick restart
	// (theADCcode's `maxdavsp`).
	NRoots   int     // Davidson: number of lowest roots to converge (≤0 → 1)
	ConvThr  float64 // Davidson: residual 2-norm threshold in a.u. (0 → 1e-3, theADCcode's convthr)
	MaxIters int     // Davidson: cap on iterations before giving up (0 → 200)

	// Checkpoint enables save/resume of the block-Krylov build across processes (see
	// checkpoint.go). nil (the default) leaves Solve bit-for-bit unchanged. Only Solve
	// honors it; SolveDense and SolveDavidson ignore it.
	Checkpoint *Checkpoint

	// LowMemBlock selects the block width for the limited-memory driver (SolveLowMem);
	// other drivers ignore it. It is the memory knob: the driver keeps only three n×block
	// panels resident instead of Solve's whole n×maxdim basis (lowmem.go). 0 (the default)
	// means block = MainBlockSize(), which reproduces theADCcode's short-recurrence DIP
	// solve exactly (Mode B: Tarantelli subspace-iteration gate + banded eigensolver, ~3·n·main
	// resident — a fat-memory CPU node). A smaller value 0<block<main selects Mode A: a
	// small start block that fits a single GPU, at the price of reaching only the states with
	// overlap in its Krylov space (the strongest lines, not a guaranteed-complete band).
	LowMemBlock int
}

// Result holds the Ritz spectrum, ascending in eigenvalue.
type Result struct {
	Values   []float64   // eigenvalues (a.u.)
	PS       []float64   // pole strength percent = 100·‖main part‖²
	MainVecs backend.Mat // main-space components of each Ritz vector (main × len(Values))
	// FullVecs holds every one of the Size() rows of each Ritz vector
	// (Size() × len(Values)), main block first. Solve fills it only under
	// Options.WantFull; SolveDense always does. Empty otherwise — test with HasFull.
	FullVecs backend.Mat
	Residual []float64 // ‖M y_k − θ_k y_k‖ (a.u.); nil for the dense path, which is exact
	Timing   Timing    // per-phase wall time
	// Interrupted is set when a checkpointing Solve stopped early on an external Stop
	// signal (Options.Checkpoint) rather than converging. The other fields are then
	// unpopulated — a checkpoint was written and the caller should arrange a resume.
	Interrupted bool
}

// HasFull reports whether FullVecs was retained.
func (r Result) HasFull() bool { return r.FullVecs.Rows > 0 }

// Solve runs the block-Lanczos driver.
func (o Options) normalize(n int) Options {
	if o.DeflTol == 0 {
		o.DeflTol = 1e-8
	}
	if o.MaxDim == 0 || o.MaxDim > n {
		o.MaxDim = n
	}
	if o.MaxBlocks <= 0 {
		o.MaxBlocks = n // effectively until deflation / MaxDim
	}
	if o.ConvThr == 0 {
		o.ConvThr = 1e-3 // theADCcode's default convthr, on the a.u. residual norm
	}
	if o.MaxIters <= 0 {
		o.MaxIters = 200
	}
	return o
}

// SubspaceDim reports the Krylov subspace dimension Solve will build for a sector of
// size n with a main block of `main` columns. Exported so the backend chooser can size a
// sector without running it; Solve uses the same expression, so the two cannot drift.
//
// The basis holds MaxBlocks blocks of at most `main` columns each (the start block is one
// of them), capped at MaxDim. Deflation can stop it earlier, so this is an upper bound.
// MaxBlocks·main is the reference's "Size of Lanczos space" for `iter MaxBlocks`.
func SubspaceDim(n, main int, opts Options) int {
	if n == 0 || main == 0 {
		return 0
	}
	o := opts.normalize(n)
	return min(o.MaxDim, o.MaxBlocks*main)
}

// DenseOperator additionally exposes the densely-built matrix for the exact
// validation path.
type DenseOperator interface {
	Operator
	BuildMatrix() backend.Mat
}

// SatelliteOperator is an Operator that can apply only its satellite↔satellite blocks —
// the sub-operator with the main block and the main↔satellite couplings removed. The
// limited-memory driver (SolveLowMem, Mode B) uses it for Tarantelli's subspace iteration:
// after the first two blocks it applies only this gated operator, which forces every later
// Lanczos vector to zero weight in the main space, so a projected eigenvector's top `main`
// rows equal its true main-space components (its pole strength) with the basis discarded.
// *dip.Matrix satisfies it.
type SatelliteOperator interface {
	Operator
	ApplyBlockSatellite(out, in backend.BlockView)
}

// LowMemSectorBytes estimates the resident footprint of SolveLowMem for a sector of size n
// at block width b. Unlike SubspaceDim/SectorBytes (the whole n×maxdim basis), the
// short-recurrence driver keeps only three n×b panels plus small scratch and the assembled
// operator, so this is the number the backend chooser should size a low-memory sector with.
// dim = MaxBlocks·b is the retained main-space slice / banded-T growth, which is off-device.
func LowMemSectorBytes(n, b int) uint64 {
	const opFrac = 0.5 // assembled block-sparse operator, ~like SectorBytes
	nf, bf := float64(n), float64(b)
	// three n×b panels (prev/cur/work) + one n×b candidate scratch + a b×b Gram.
	bytes := 4*nf*bf*8 + bf*bf*8 + opFrac*nf*nf*8
	return uint64(bytes)
}

// SolveDense diagonalizes the full matrix directly (the reference's DiagFull
// path). Exact; used as the correctness oracle and for small cases.
func SolveDense(op DenseOperator, be backend.Backend) Result {
	var tm Timing
	tApply := time.Now()
	M := op.BuildMatrix()
	tm.Apply = time.Since(tApply)

	tEig := time.Now()
	evals, evecs := be.SymEig(M)
	tm.Eig = time.Since(tEig)
	main := op.MainBlockSize()
	tBack := time.Now()
	mainVecs := backend.NewMat(main, len(evals))
	ps := make([]float64, len(evals))
	for k := range evals {
		var acc float64
		for c := range main {
			v := evecs.At(c, k)
			mainVecs.Set(c, k, v)
			acc += v * v
		}
		ps[k] = 100 * acc
	}
	tm.Back = time.Since(tBack)
	return Result{Values: evals, PS: ps, MainVecs: mainVecs, FullVecs: evecs, Timing: tm}
}

// blockOrth orthonormalizes the columns of v in place against nothing (v is assumed
// already projected out of the basis), returning the number of surviving columns and
// the R factor of the reduction, R (rank × cols) such that the original v equals
// q·R for the returned orthonormal q occupying v's leading `rank` columns.
//
// It works through the small cols×cols Gram matrix G = vᵀv and its symmetric
// eigendecomposition G = UΛUᵀ: q = v·U·Λ^(-1/2), R = Λ^(1/2)·Uᵀ. Directions whose
// singular value √λ falls below deflTol are dropped — the blocked analogue of the
// per-vector ‖v‖ > DeflTol test this replaces.
//
// Cholesky-QR would be cheaper but breaks down exactly when the block is rank
// deficient, which is the case deflation exists to handle; the Gram
// eigendecomposition is rank-revealing by construction and cols is small (≤ the 2h
// space, tens of columns).
//
// Forming G squares the condition number, so one pass loses orthogonality like
// cond(v)²·eps. Two guards make that safe: directions are dropped relative to λ_max as
// well as absolutely, bounding cond(v) at ~1e7; and orthBlock repeats this pass (the
// CholeskyQR2 pattern) whenever the block is ill conditioned or rank deficient, the
// second pass seeing an already well-conditioned block and restoring ‖qᵀq − I‖ to
// O(eps). Without both, the full-subspace exactness test fails by ~1e-2 once the
// candidate block is nearly dependent on the basis.
//
// cond2 is the returned λ_max/λ_min over the surviving directions — the squared
// condition number of v, which governs this pass's orthogonality loss. orthBlock uses it
// to decide whether the second pass is needed at all.
func blockOrth(be backend.Backend, v backend.BlockView, scratch backend.Vector, deflTol float64) (rank int, r backend.Mat, cond2 float64) {
	nrows, cols := v.Rows, v.Cols
	g := backend.BlockView{V: scratch, Rows: cols, Cols: cols, Ld: cols}
	be.Gemm(true, false, 1, v, v, 0, g) // G = vᵀ v

	gd := be.Download(g.V)
	gm := backend.NewMat(cols, cols)
	for i := range cols {
		for j := range cols {
			gm.Set(i, j, gd[j*cols+i]) // column-major → row-major
		}
	}
	lambda, u := be.SymEig(gm) // ascending

	// relFloor bounds cond(v) = sqrt(λ_max/λ_min) at ~1e7, keeping cond(G) = cond(v)²
	// well inside double precision.
	const relCond2 = 1e-14
	relFloor := lambda[cols-1] * relCond2
	keep := make([]int, 0, cols)
	for k := range cols {
		if lambda[k] > deflTol*deflTol && lambda[k] > relFloor {
			keep = append(keep, k)
		}
	}
	rank = len(keep)
	if rank == 0 {
		return 0, backend.NewMat(0, cols), math.Inf(1)
	}
	cond2 = lambda[keep[rank-1]] / lambda[keep[0]]

	// tm = U_keep·Λ_keep^(-1/2) (cols × rank), column-major for the device.
	tm := make([]float64, cols*rank)
	r = backend.NewMat(rank, cols)
	for ri, k := range keep {
		s := math.Sqrt(lambda[k])
		for i := range cols {
			tm[ri*cols+i] = u.At(i, k) / s
			r.Set(ri, i, s*u.At(i, k)) // R = Λ^(1/2) Uᵀ
		}
	}

	// q = v·tm, written over v's leading columns. v and the product overlap, so use a
	// scratch panel: cols is small, so this is nrows×rank, never the basis.
	tmv := be.Upload(tm)
	defer be.Free(tmv)
	qbuf := be.Alloc(nrows * rank)
	defer be.Free(qbuf)
	q := backend.BlockView{V: qbuf, Rows: nrows, Cols: rank, Ld: nrows}
	be.Gemm(false, false, 1, v, backend.BlockView{V: tmv, Rows: cols, Cols: rank, Ld: cols}, 0, q)
	be.Copy(v.ColRange(0, rank).V, qbuf)
	return rank, r, cond2
}

// maxGramCond2 is the largest λ_max/λ_min (= cond(v)²) for which a single
// orthogonalization pass is trusted. Two error terms bound the decision:
//
//   - within-block orthogonality after one Gram-QR: ‖qᵀq − I‖ ≈ cond(v)²·eps;
//   - orthogonality to the basis: Bᵀq = (Bᵀv)·U·Λ^(-1/2), and CGS2 leaves
//     ‖Bᵀv‖ ≈ eps·‖v‖, which Λ^(-1/2) amplifies by cond(v).
//
// At cond² = 1e4 (cond = 100) those are ~2e-12 and ~2e-14 — comfortably below the
// 1e-10 accuracy the solver is gated to. Anything worse conditioned gets the second
// pass. Rank deficiency always gets it: deflation means the block *was* ill conditioned.
const maxGramCond2 = 1e4

// orthBlock projects v out of the basis and orthonormalizes it: CGS2 + rank-revealing
// Gram-QR, repeated only when the first pass leaves the block ill conditioned or
// rank deficient (the classic "reorthogonalize if needed" criterion).
//
// The repeat is what makes Gram-based orthogonalization backward stable (CholeskyQR2),
// but it is not free: the second CGS2 is four GEMMs of O(n·dim·b) against the whole
// basis, which was 53% of a formic-acid sector on the GPU. The second Gram-QR is cheap
// (O(n·b²)); it is the re-projection that costs. In practice M·Q_last is well
// conditioned after projection and the second pass is skipped, while near the end of a
// run — where the candidate block becomes nearly dependent on the basis, and where a
// single pass provably fails — it fires.
//
// Returns the surviving column count and the composite R such that the incoming v
// equals q·R, with q in v's leading `rank` columns. Since v = q₁R₁ and (after the
// second projection, which barely moves q₁) q₁ = q₂R₂, the composite is R = R₂·R₁.
func orthBlock(be backend.Backend, basis, v backend.BlockView, pbuf backend.Vector, ld int,
	gbuf backend.Vector, deflTol float64) (int, backend.Mat) {
	cgs2(be, basis, v, pbuf, ld)
	rank1, r1, cond2 := blockOrth(be, v, gbuf, deflTol)
	if rank1 == 0 {
		return 0, r1
	}
	if rank1 == v.Cols && cond2 <= maxGramCond2 {
		return rank1, r1 // well conditioned and nothing deflated: one pass suffices
	}

	v2 := v.Cut(rank1)
	cgs2(be, basis, v2, pbuf, ld)
	rank2, r2, _ := blockOrth(be, v2, gbuf, deflTol)
	if rank2 == 0 {
		return 0, backend.NewMat(0, v.Cols)
	}
	return rank2, backend.MatMul(r2, r1)
}

// Solve builds the block-Krylov subspace and returns the Rayleigh–Ritz spectrum.
//
// The basis stays backend-resident as one contiguous column-major n×maxdim panel, so
// every phase is a GEMM:
//
//	W  = M·Q_j                       (one ApplyBlock, not b mat-vecs)
//	T[:,j-block] = Bᵀ·W              (one GEMM; M symmetric fills the transpose)
//	V  = W;  twice: V -= B·(Bᵀ·V)    (blocked CGS2, two GEMMs per pass)
//	Q_{j+1}, R_{j+1} = orth(V)       (rank-revealing, small)
//
// Two passes of classical Gram–Schmidt carry the same backward-stability guarantee as
// the two-pass modified Gram–Schmidt it replaces, at level 3 instead of level 1.
//
// T is accumulated a block-column at a time. Because M is symmetric, every entry
// (i,j) with i < j has i in an already-completed block, so processing blocks in order
// fills the whole upper triangle — which is why the M-images need not be retained.
// The old code held both the basis and its images (2·dim·n) for the whole run.
func Solve(op Operator, be backend.Backend, opts Options) Result {
	n := op.Size()
	main := op.MainBlockSize()
	opts = opts.normalize(n)
	var tm Timing
	if n == 0 || main == 0 {
		return Result{MainVecs: backend.NewMat(main, 0), Timing: tm}
	}

	maxdim := SubspaceDim(n, main, opts)

	bbuf := be.Alloc(n * maxdim)
	defer be.Free(bbuf)
	basis := backend.BlockView{V: bbuf, Rows: n, Cols: maxdim, Ld: n}

	wbuf := be.Alloc(n * main) // M·Q_j
	vbuf := be.Alloc(n * main) // the candidate next block
	pbuf := be.Alloc(maxdim * main)
	gbuf := be.Alloc(main * main)
	defer be.Free(wbuf)
	defer be.Free(vbuf)
	defer be.Free(pbuf)
	defer be.Free(gbuf)

	t := backend.NewMat(maxdim, maxdim)
	dim, blkStart, blkSize := main, 0, main
	iter0 := 0

	// Resume from a checkpoint if one is present and matches this problem; otherwise start
	// fresh. A resumed run reloads the basis and the projected matrix and re-enters the loop
	// at the saved block index (checkpoint.go).
	cp := opts.Checkpoint
	resumed := false
	if cp != nil && cp.Path != "" {
		if st := loadResumable(cp.Path, n, main, maxdim, opts.MaxBlocks); st != nil {
			up := be.Upload(st.Basis)
			be.Copy(basis.ColRange(0, st.Dim).V, up)
			be.Free(up)
			for i := 0; i < st.Dim; i++ {
				copy(t.Data[i*maxdim:i*maxdim+st.Dim], st.T[i*st.Dim:(i+1)*st.Dim])
			}
			dim, blkStart, blkSize, iter0 = st.Dim, st.BlkStart, st.BlkSize, st.Iter
			resumed = true
		}
	}
	if !resumed {
		// Start block: the main-space Cartesian unit vectors e_0..e_{main-1}, already
		// orthonormal. Same start vectors as theADCcode, so pole strengths converge first.
		start := make([]float64, n*main)
		for c := range main {
			start[c*n+c] = 1
		}
		tmp := be.Upload(start)
		be.Copy(basis.ColRange(0, main).V, tmp)
		be.Free(tmp)
	}

	var rNext backend.Mat // R factor of the block after the last accepted one

	// project accumulates T's block-column for the current block and returns the
	// candidate V = W projected out of the existing basis, plus its rank/R factor.
	for iter := iter0; ; iter++ {
		// Checkpoint hook. The state here (basis[:dim], T[:dim,:dim], the block scalars)
		// fully re-does iteration `iter`, so it is a consistent resume point. Save on an
		// external stop request (then return early) or every `Every` blocks for crash
		// resilience. Skip iter0 itself: it was just loaded.
		if cp != nil && cp.Path != "" {
			stop := cp.stopRequested()
			due := cp.Every > 0 && iter != iter0 && iter%cp.Every == 0
			if stop || due {
				_ = saveKrylov(be, cp.Path, basis, t, n, main, maxdim, opts.MaxBlocks,
					dim, blkStart, blkSize, iter)
				if stop {
					return Result{Interrupted: true, Timing: tm}
				}
			}
		}

		w := backend.BlockView{V: wbuf, Rows: n, Cols: blkSize, Ld: n}
		q := basis.ColRange(blkStart, blkStart+blkSize)

		t0 := time.Now()
		op.ApplyBlock(w, q)
		tm.Apply += time.Since(t0)

		// T[0:dim, blkStart:blkStart+blkSize] = Bᵀ·W, mirrored into the lower triangle.
		t0 = time.Now()
		pc := backend.BlockView{V: pbuf, Rows: dim, Cols: blkSize, Ld: maxdim}
		be.Gemm(true, false, 1, basis.Cut(dim), w, 0, pc)
		pd := be.Download(pc.V)
		for j := range blkSize {
			for i := range dim {
				v := pd[j*maxdim+i]
				t.Set(i, blkStart+j, v)
				t.Set(blkStart+j, i, v)
			}
		}
		tm.Proj += time.Since(t0)

		// iter is the 0-based index of the block just projected, so the basis now holds
		// iter+1 blocks. Stop once that reaches MaxBlocks — the basis spans exactly the
		// same MaxBlocks·main columns the reference diagonalizes for `iter MaxBlocks`.
		if dim >= opts.MaxDim || iter+1 >= opts.MaxBlocks {
			// One extra projection, discarded, purely to obtain R_{j+1} for the Ritz
			// residuals: a truncated run never forms the block after the last one. This
			// is the reference's trailing block (dim = (iter+1)·block, dimd = iter·block).
			t0 = time.Now()
			v := backend.BlockView{V: vbuf, Rows: n, Cols: blkSize, Ld: n}
			be.Copy(vbuf, wbuf)
			_, rNext = orthBlock(be, basis.Cut(dim), v, pbuf, maxdim, gbuf, opts.DeflTol)
			tm.Orth += time.Since(t0)
			break
		}

		// Candidate next block: V = W, projected out of the basis (two CGS passes),
		// then orthonormalized within itself with rank-revealing deflation.
		t0 = time.Now()
		v := backend.BlockView{V: vbuf, Rows: n, Cols: blkSize, Ld: n}
		be.Copy(vbuf, wbuf)
		rank, r := orthBlock(be, basis.Cut(dim), v, pbuf, maxdim, gbuf, opts.DeflTol)
		tm.Orth += time.Since(t0)

		if rank == 0 {
			rNext = r // zero rows: the subspace is M-invariant, residuals vanish
			break
		}
		if dim+rank > opts.MaxDim {
			rank = opts.MaxDim - dim
		}
		be.Copy(basis.ColRange(dim, dim+rank).V, v.ColRange(0, rank).V)
		rNext = r
		blkStart, blkSize = dim, rank
		dim += rank
	}

	// Rayleigh–Ritz on the projected matrix.
	tEig := time.Now()
	tt := backend.NewMat(dim, dim)
	for i := range dim {
		copy(tt.Data[i*dim:(i+1)*dim], t.Data[i*maxdim:i*maxdim+dim])
	}
	theta, s := be.SymEig(tt) // ascending
	tm.Eig = time.Since(tEig)

	// Ritz vectors' main-space part: mainVecs = Bmain·S, one host GEMM. Only the
	// leading `main` rows of each basis column come back from the device.
	tBack := time.Now()
	// The leading `main` rows of every basis column: `dim` short runs, basis.Ld apart. A
	// StridedDownloader fetches the whole rectangle in one transfer; without it this is one
	// blocking round-trip per column, and dim reaches the thousands on a large sector.
	bmain := backend.NewMat(main, dim)
	if sd, ok := be.(backend.StridedDownloader); ok {
		flat := sd.Download2D(basis.V, main, dim, basis.Ld) // main×dim, column-major
		for j := range dim {
			for c := range main {
				bmain.Set(c, j, flat[j*main+c])
			}
		}
	} else {
		for j := range dim {
			col := be.Download(basis.Col(j).Slice(0, main))
			for c := range main {
				bmain.Set(c, j, col[c])
			}
		}
	}
	mainVecs := backend.MatMul(bmain, s)
	ps := make([]float64, dim)
	for k := range dim {
		var acc float64
		for c := range main {
			v := mainVecs.At(c, k)
			acc += v * v
		}
		ps[k] = 100 * acc
	}

	// Full Ritz vectors, satellite rows included: fullVecs = B·S. Unlike the main-space
	// slice, this panel is n rows tall, so it stays on the device — bringing the basis
	// home to multiply it here would cost an n×dim transfer and an O(n·dim²) host GEMM.
	// It is computed in column chunks of `main` states so the output panel is wbuf, which
	// the Krylov build already paid for; no device memory is added.
	//
	// mainVecs is deliberately not re-derived from its leading rows: WantFull must leave
	// every existing field bit-for-bit as it was, and a device GEMM rounds differently
	// from the host MatMul above.
	var fullVecs backend.Mat
	if opts.WantFull {
		fullVecs = backend.NewMat(n, dim)
		for k0 := 0; k0 < dim; k0 += main {
			cols := min(main, dim-k0)
			sc := make([]float64, dim*cols) // column-major s[:, k0:k0+cols]
			for j := range cols {
				for i := range dim {
					sc[j*dim+i] = s.At(i, k0+j)
				}
			}
			sv := be.Upload(sc)
			sblk := backend.BlockView{V: sv, Rows: dim, Cols: cols, Ld: dim}
			wv := backend.BlockView{V: wbuf, Rows: n, Cols: cols, Ld: n}
			be.Gemm(false, false, 1, basis.Cut(dim), sblk, 0, wv)
			be.Free(sv)
			hd := be.Download(wbuf)
			for j := range cols {
				for r := range n {
					fullVecs.Set(r, k0+j, hd[j*n+r])
				}
			}
		}
	}

	// Ritz residual. With T = BᵀMB block-tridiagonal in exact arithmetic,
	// M·B = B·T + Q_{j+1}·R_{j+1}·Eᵀ_last, so for T·s_k = θ_k·s_k the first term
	// cancels and ‖r_k‖ = ‖R_{j+1}·s_k[last block]‖ — O(b) per Ritz vector.
	residual := make([]float64, dim)
	for k := range dim {
		var acc float64
		for r := range rNext.Rows {
			var v float64
			for c := range rNext.Cols {
				v += rNext.At(r, c) * s.At(blkStart+c, k)
			}
			acc += v * v
		}
		residual[k] = math.Sqrt(acc)
	}
	tm.Back = time.Since(tBack)

	// The solve completed; drop any checkpoint so a later rerun of the same job starts fresh
	// rather than resuming a finished computation.
	if cp != nil && cp.Path != "" {
		removeCheckpoint(cp.Path)
	}

	return Result{Values: theta, PS: ps, MainVecs: mainVecs, FullVecs: fullVecs, Residual: residual, Timing: tm}
}

// cgs2 applies two passes of classical Gram–Schmidt: v -= B·(Bᵀ·v), twice. Each pass
// is two GEMMs against the whole basis, replacing 2·dim BLAS-1 calls per column.
func cgs2(be backend.Backend, b, v backend.BlockView, pbuf backend.Vector, ld int) {
	p := backend.BlockView{V: pbuf, Rows: b.Cols, Cols: v.Cols, Ld: ld}
	for range 2 {
		be.Gemm(true, false, 1, b, v, 0, p)   // P = Bᵀ V
		be.Gemm(false, false, -1, b, p, 1, v) // V -= B P
	}
}

// Spurious reports whether a Ritz vector is a Lanczos ghost: essentially zero
// weight in the main space (the reference's spur_thresh = 1e-9 test on the
// main-block components). k indexes a column of MainVecs.
func (r Result) Spurious(k int, thresh float64) bool {
	for c := range r.MainVecs.Rows {
		if math.Abs(r.MainVecs.At(c, k)) > thresh {
			return false
		}
	}
	return true
}
