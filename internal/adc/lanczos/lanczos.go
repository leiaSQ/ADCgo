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

	"adcgo/internal/adc/backend"
)

// Operator is a real-symmetric matrix applied matrix-free on backend-resident
// vectors, with a distinguished "main" block (its leading MainBlockSize rows)
// whose squared eigenvector weight is the pole strength. *dip.Matrix satisfies
// this.
type Operator interface {
	ApplyFull(out, in backend.Vector)
	Size() int
	MainBlockSize() int
}

// Options tunes the block-Krylov build.
type Options struct {
	MaxBlocks int     // cap on block iterations (0 → until deflation/full)
	MaxDim    int     // cap on subspace dimension (0 → Size())
	DeflTol   float64 // deflation threshold for new basis vectors (0 → 1e-8)
}

// Result holds the Ritz spectrum, ascending in eigenvalue.
type Result struct {
	Values   []float64   // eigenvalues (a.u.)
	PS       []float64   // pole strength percent = 100·‖main part‖²
	MainVecs backend.Mat // main-space components of each Ritz vector (main × len(Values))
}

// Solve runs the block-Lanczos driver.
func (o Options) normalize(n int) Options {
	if o.DeflTol == 0 {
		o.DeflTol = 1e-8
	}
	if o.MaxDim == 0 || o.MaxDim > n {
		o.MaxDim = n
	}
	if o.MaxBlocks == 0 {
		o.MaxBlocks = n // effectively until deflation / MaxDim
	}
	return o
}

// DenseOperator additionally exposes the densely-built matrix for the exact
// validation path.
type DenseOperator interface {
	Operator
	BuildMatrix() backend.Mat
}

// SolveDense diagonalizes the full matrix directly (the reference's DiagFull
// path). Exact; used as the correctness oracle and for small cases.
func SolveDense(op DenseOperator, be backend.Backend) Result {
	M := op.BuildMatrix()
	evals, evecs := be.SymEig(M)
	main := op.MainBlockSize()
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
	return Result{Values: evals, PS: ps, MainVecs: mainVecs}
}

// Solve builds the block-Krylov subspace and returns the Rayleigh–Ritz spectrum.
// The basis and its M-images stay backend-resident across iterations; only the
// small projected matrix and the main-space components of the basis (for pole
// strengths) are brought back to the host.
func Solve(op Operator, be backend.Backend, opts Options) Result {
	n := op.Size()
	main := op.MainBlockSize()
	opts = opts.normalize(n)

	apply := func(v backend.Vector) backend.Vector {
		w := be.Alloc(n)
		op.ApplyFull(w, v)
		return w
	}

	// Basis (columns) and their M-images. mImg[j] may be nil until needed.
	basis := make([]backend.Vector, 0, opts.MaxDim)
	mImg := make([]backend.Vector, 0, opts.MaxDim)

	// Start block: main-space Cartesian unit vectors e_0..e_{main-1}
	// (already orthonormal).
	e := make([]float64, n)
	for c := range main {
		e[c] = 1
		basis = append(basis, be.Upload(e))
		mImg = append(mImg, nil)
		e[c] = 0
	}
	last := make([]int, main)
	for i := range last {
		last[i] = i
	}

	orthonormalize := func(v backend.Vector) float64 {
		// Two passes of modified Gram–Schmidt against the whole basis.
		for range 2 {
			for _, q := range basis {
				be.Axpy(-be.Dot(q, v), q, v)
			}
		}
		nrm := be.Nrm2(v)
		if nrm > opts.DeflTol {
			be.Scal(1/nrm, v)
		}
		return nrm
	}

	for iter := 0; iter < opts.MaxBlocks && len(basis) < opts.MaxDim; iter++ {
		next := make([]int, 0, len(last))
		for _, idx := range last {
			w := apply(basis[idx])
			mImg[idx] = w
			cand := be.Alloc(n)
			be.Copy(cand, w)
			if nrm := orthonormalize(cand); nrm > opts.DeflTol {
				basis = append(basis, cand)
				mImg = append(mImg, nil)
				next = append(next, len(basis)-1)
				if len(basis) >= opts.MaxDim {
					break
				}
			} else {
				be.Free(cand)
			}
		}
		if len(next) == 0 {
			break // subspace is M-invariant: exact
		}
		last = next
	}

	// Ensure every basis vector has its M-image (last block may be unfilled).
	for j := range basis {
		if mImg[j] == nil {
			mImg[j] = apply(basis[j])
		}
	}

	// Projected matrix T[i][j] = q_i · (M q_j), symmetric.
	dim := len(basis)
	T := backend.NewMat(dim, dim)
	for i := range dim {
		for j := i; j < dim; j++ {
			v := be.Dot(basis[i], mImg[j])
			T.Set(i, j, v)
			T.Set(j, i, v)
		}
	}

	theta, s := be.SymEig(T) // ascending

	// Ritz vectors' main-space part: y_main[c,k] = Σ_j basis[j][c] · s[j,k].
	// Download only the leading `main` components of each basis vector.
	bmain := make([][]float64, dim)
	for j := range basis {
		bmain[j] = be.Download(basis[j].Slice(0, main))
	}
	mainVecs := backend.NewMat(main, dim)
	ps := make([]float64, dim)
	for k := range dim {
		var acc float64
		for c := range main {
			var y float64
			for j := range dim {
				y += bmain[j][c] * s.At(j, k)
			}
			mainVecs.Set(c, k, y)
			acc += y * y
		}
		ps[k] = 100 * acc
	}

	return Result{Values: theta, PS: ps, MainVecs: mainVecs}
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
