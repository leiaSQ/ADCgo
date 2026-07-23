// davidson.go — a block Davidson–Liu eigensolver for the same real-symmetric ADC
// secular problem the block-Lanczos driver (Solve) handles, but root-targeting: it
// iterates the algebraically lowest `NRoots` eigenpairs to a residual threshold rather
// than sweeping the whole spectral band. This is what reproduces the legacy
// adc4_diag.x Davidson positions directly (see examples/CVS_NSOB/lanczos_vs_davidson_N1s.md):
// where block-Lanczos at a fixed `-blocks` count leaves interior CVS core poles shifted by
// ~1 eV (each Ritz value is a pole-strength-weighted centroid of a cluster of true states),
// Davidson pins the specific lowest roots to ‖Mψ−θψ‖ ≤ ConvThr.
//
// The algorithm mirrors ../ADC/adc4core/adc4_diag/davidson.F:
//   - block size = NRoots, subspace grown by up to NRoots correction vectors per iteration;
//   - start block = the first NRoots main-block Cartesian unit vectors (davidson.F:188-194),
//     which is exactly Solve's start block (lanczos.go), so the main-space-weighted states
//     seed the search;
//   - diagonal Davidson–Liu preconditioner t = r / (θ − D) with the |θ − D| < 1e-3 → 1.0
//     proximity guard (davidson.F:361-374); D is the operator's exact matrix diagonal;
//   - Rayleigh–Ritz H = BᵀMB diagonalized densely (be.SymEig), lowest NRoots targeted;
//   - convergence on the residual 2-norm in a.u. against ConvThr (davidson.F:345-352);
//   - thick restart: when the subspace would exceed MaxDim (theADCcode's maxdavsp) it
//     collapses to the NRoots current Ritz vectors and continues (davidson.F:458-484).
//
// It reuses the package's orthogonalization suite (orthBlock: two-pass CGS2 +
// rank-revealing Gram-QR), which is strictly more robust than the reference's single-pass
// classical Gram–Schmidt; to the residual gate the converged roots are identical. It
// returns the same Result as Solve, so nothing downstream (analyze/spectrum/TDM) changes.

package lanczos

import (
	"math"
	"sort"
	"time"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// PreconOperator is an Operator that can also expose its matrix diagonal as a resident
// vector, which the Davidson (θ − D)⁻¹ preconditioner needs. *sip.Matrix and *dip.Matrix
// satisfy it. The diagonal is assembled directly from the per-block element functions,
// never from BuildMatrix (which is terabytes for a large matrix-free ADC(4) sector).
type PreconOperator interface {
	Operator
	Diagonal(be backend.Backend) backend.Vector
}

// DavidsonSubspaceDim reports the maximum subspace dimension SolveDavidson will allocate
// for a sector of size n (two n×dim resident panels: the basis and its M-images). Exported
// so the backend chooser can size a sector without running it; SolveDavidson uses the same
// expression, so the two cannot drift.
func DavidsonSubspaceDim(n int, opts Options) int {
	if n == 0 {
		return 0
	}
	o := opts.normalize(n)
	nw := workingRoots(clampRoots(o.NRoots, n), n)
	return min(max(o.MaxDim, 2*nw), n)
}

func clampRoots(nr, n int) int {
	if nr <= 0 {
		nr = 1
	}
	if nr > n {
		nr = n
	}
	return nr
}

// workingRoots is the block width the driver actually iterates: a few more than the nr
// requested. Block Davidson with a block of exactly nr can swap the nr-th root with the
// (nr+1)-th at the window boundary and converge the wrong one; carrying a small buffer of
// extra Ritz vectors protects the requested roots. nr is assumed already clamped.
func workingRoots(nr, n int) int {
	return min(nr+max(4, nr/4), n)
}

// SolveDavidson runs the block Davidson–Liu driver and returns the lowest NRoots Ritz
// pairs (ascending) in the same Result shape as Solve.
func SolveDavidson(op PreconOperator, be backend.Backend, opts Options) Result {
	n := op.Size()
	main := op.MainBlockSize()
	opts = opts.normalize(n)
	var tm Timing
	if n == 0 || main == 0 {
		return Result{MainVecs: backend.NewMat(main, 0), Timing: tm}
	}

	nr := clampRoots(opts.NRoots, n) // roots requested (gated + returned)
	nw := workingRoots(nr, n)        // roots actively driven (nr + boundary buffer)
	maxdim := DavidsonSubspaceDim(n, opts)

	// The exact matrix diagonal, on the host, for the preconditioner.
	dv := op.Diagonal(be)
	dHost := be.Download(dv)
	be.Free(dv)

	// Resident panels: the basis B and its images W = M·B, column-aligned.
	bbuf := be.Alloc(n * maxdim)
	wbuf := be.Alloc(n * maxdim)
	cbuf := be.Alloc(n * nw) // residual / Ritz-block scratch
	ybuf := be.Alloc(n * nw) // Ritz-vector block
	pbuf := be.Alloc(maxdim * nw)
	gbuf := be.Alloc(nw * nw)
	hbuf := be.Alloc(maxdim * maxdim)
	defer be.Free(bbuf)
	defer be.Free(wbuf)
	defer be.Free(cbuf)
	defer be.Free(ybuf)
	defer be.Free(pbuf)
	defer be.Free(gbuf)
	defer be.Free(hbuf)
	basis := backend.BlockView{V: bbuf, Rows: n, Cols: maxdim, Ld: n}
	wmat := backend.BlockView{V: wbuf, Rows: n, Cols: maxdim, Ld: n}

	// Start block: unit vectors on the nr smallest diagonal entries. theADCcode's guess is
	// the first nr Cartesian units (davidson.F:188-194), which is the special case where the
	// low (core) states are ordered first; seeding by smallest diagonal is the robust
	// generalization — it lands on the same core configs when the space is core-restricted,
	// but also spans every symmetry block that hosts a low root, which a fixed first-nr seed
	// misses on a block-structured matrix (leaving those roots orthogonal to the whole
	// Krylov space).
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return dHost[idx[a]] < dHost[idx[b]] })
	start := make([]float64, n*nw)
	for c := range nw {
		start[c*n+idx[c]] = 1
	}
	tmp := be.Upload(start)
	be.Copy(basis.ColRange(0, nw).V, tmp)
	be.Free(tmp)

	dim, applied := nw, 0
	var theta []float64
	var s backend.Mat // subspace eigenvectors of the last Rayleigh–Ritz (dim × dim)
	residNorm := make([]float64, nw)

	for iter := 0; iter < opts.MaxIters; iter++ {
		// Apply M to the columns added since the last apply (all of them after a restart).
		t0 := time.Now()
		op.ApplyBlock(wmat.ColRange(applied, dim), basis.ColRange(applied, dim))
		applied = dim
		tm.Apply += time.Since(t0)

		// Projected matrix H = BᵀW (symmetric because M is), diagonalized densely.
		t0 = time.Now()
		hview := backend.BlockView{V: hbuf, Rows: dim, Cols: dim, Ld: dim}
		be.Gemm(true, false, 1, basis.Cut(dim), wmat.Cut(dim), 0, hview)
		hd := be.Download(hbuf)
		hm := backend.NewMat(dim, dim)
		for i := range dim {
			for j := range dim {
				hm.Set(i, j, 0.5*(hd[j*dim+i]+hd[i*dim+j])) // symmetrize the round-off
			}
		}
		tm.Proj += time.Since(t0)

		t0 = time.Now()
		theta, s = be.SymEig(hm) // ascending
		tm.Eig += time.Since(t0)

		// Ritz block Y = B·S_w and its M-image WS = W·S_w over the lowest nw eigenvectors;
		// residual R = WS − Y·diag(θ).
		t0 = time.Now()
		snr := make([]float64, dim*nw) // column-major dim×nw
		for k := range nw {
			for i := range dim {
				snr[k*dim+i] = s.At(i, k)
			}
		}
		snrV := be.Upload(snr)
		snrBlk := backend.BlockView{V: snrV, Rows: dim, Cols: nw, Ld: dim}
		yblk := backend.BlockView{V: ybuf, Rows: n, Cols: nw, Ld: n}
		rblk := backend.BlockView{V: cbuf, Rows: n, Cols: nw, Ld: n}
		be.Gemm(false, false, 1, basis.Cut(dim), snrBlk, 0, yblk)
		be.Gemm(false, false, 1, wmat.Cut(dim), snrBlk, 0, rblk)
		be.Free(snrV)
		for k := range nw {
			be.Axpy(-theta[k], yblk.Col(k), rblk.Col(k)) // R_k = WS_k − θ_k·Y_k
		}
		nconv := 0
		for k := range nw {
			residNorm[k] = be.Nrm2(rblk.Col(k))
			if k < nr && residNorm[k] <= opts.ConvThr {
				nconv++ // convergence is gated on the lowest nr only
			}
		}
		tm.Back += time.Since(t0)
		if nconv == nr {
			break
		}

		// Precondition the unconverged residuals on the host: t = r/(θ−D), with the
		// |θ−D| < 1e-3 → 1.0 proximity guard (theADCcode davidson.F:361-374). The buffer
		// roots [nr,nw) are driven too, keeping the block full so the requested roots do not
		// swap out at the window boundary.
		t0 = time.Now()
		rHost := be.Download(cbuf)
		// Collect the unconverged column indices first so the correction block can be sized and
		// filled in one allocation: the previous form allocated a fresh n-vector per column and
		// appended it, i.e. nunc+1 allocations plus a copy on every Davidson iteration.
		unc := make([]int, 0, nw)
		for k := range nw {
			if residNorm[k] > opts.ConvThr {
				unc = append(unc, k)
			}
		}
		nunc := len(unc)
		cor := make([]float64, n*nunc) // n×nunc, column-major
		// Parallel over correction columns: column c writes only cor[c*n:(c+1)*n], so the work
		// items are disjoint (parallel.Rows' contract). Bit-identical to the serial form — every
		// element uses the same expression and nrm2 still accumulates in ascending j within a
		// column; only the order across independent columns changes.
		//
		// Note this runs mid-solve, unlike parallel's assemble-phase framing. That is safe here
		// because the device is idle at this point: the residual has just been downloaded and the
		// correction is not uploaded until below, so there is no concurrent BLAS to oversubscribe.
		parallel.Rows(nunc, func(c int) {
			k := unc[c]
			col := cor[c*n : (c+1)*n]
			var nrm2 float64
			for j := range n {
				a1 := theta[k] - dHost[j]
				if math.Abs(a1) < 1e-3 {
					col[j] = 1.0 // proximity guard (theADCcode davidson.F:361-374)
				} else {
					col[j] = rHost[k*n+j] / a1
				}
				nrm2 += col[j] * col[j]
			}
			// Normalize the correction to unit length. Its magnitude tracks the (shrinking)
			// residual, so without this the rank-revealing orthogonalization would deflate a
			// still-needed direction once ‖t‖ falls below DeflTol — stalling the subspace on
			// the wrong roots well before ConvThr. The direction is what matters.
			if nrm2 > 0 {
				s := 1 / math.Sqrt(nrm2)
				for j := range n {
					col[j] *= s
				}
			}
		})
		corUp := be.Upload(cor) // n×nunc, column-major
		corBlk := backend.BlockView{V: corUp, Rows: n, Cols: nunc, Ld: n}

		// Orthonormalize the correction block against the basis and within itself.
		rank, _ := orthBlock(be, basis.Cut(dim), corBlk, pbuf, maxdim, gbuf, opts.DeflTol)
		tm.Orth += time.Since(t0)
		if rank == 0 {
			be.Free(corUp)
			break // the subspace is M-invariant on the targeted roots; nothing to add
		}

		// Thick restart: if appending would exceed the subspace cap, collapse to the nw
		// current Ritz vectors (already orthonormal: B orthonormal, S_w orthonormal) and
		// begin a new sweep, re-imaging the collapsed basis on the next iteration.
		if dim+rank > maxdim {
			be.Copy(basis.ColRange(0, nw).V, ybuf)
			be.Free(corUp)
			dim, applied = nw, 0
			continue
		}
		be.Copy(basis.ColRange(dim, dim+rank).V, corBlk.ColRange(0, rank).V)
		be.Free(corUp)
		dim += rank
	}

	// Pack the Result (lowest nr) from ybuf, which holds the current best Ritz vectors
	// B·S_nr consistent with theta at whatever point the loop stopped (converged, stalled,
	// or iteration-capped) — no basis/eigenvector re-derivation, which would be stale after
	// a restart collapse.
	tBack := time.Now()
	yh := be.Download(ybuf) // column-major n×nr: Ritz vector k at [k*n:(k+1)*n]

	values := make([]float64, nr)
	copy(values, theta[:nr])
	mainVecs := backend.NewMat(main, nr)
	ps := make([]float64, nr)
	for k := range nr {
		var acc float64
		for c := range main {
			v := yh[k*n+c]
			mainVecs.Set(c, k, v)
			acc += v * v
		}
		ps[k] = 100 * acc
	}
	var fullVecs backend.Mat
	if opts.WantFull {
		fullVecs = backend.NewMat(n, nr)
		for k := range nr {
			for r := range n {
				fullVecs.Set(r, k, yh[k*n+r])
			}
		}
	}
	residual := make([]float64, nr)
	copy(residual, residNorm)
	tm.Back += time.Since(tBack)

	return Result{Values: values, PS: ps, MainVecs: mainVecs, FullVecs: fullVecs, Residual: residual, Timing: tm}
}
