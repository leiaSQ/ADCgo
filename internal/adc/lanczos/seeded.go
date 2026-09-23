package lanczos

// seeded.go — single-vector Lanczos from a given start vector (width engine W-B,
// fano/gauss.go). Stieltjes imaging needs the spectral measure of the coupling vector
// g with respect to PHP, mu(E) = sum_j |<psi_j|g>|^2 delta(E - E_j). The m-step Lanczos
// run seeded by g produces the Jacobi matrix T_m of exactly that measure, and the
// eigen-decomposition of its leading k x k block is the k-point Gauss rule of mu for
// every k <= m: nodes the eigenvalues, weights ||g||^2 times the squared first
// eigenvector components (Golub & Meurant, "Matrices, Moments and Quadrature with
// Applications", 2010, ch. 6).
//
// Two things make this preferable to the eigen path for large P spaces. It needs no
// eigenvector of PHP, which a (k+2)h2p relaxation class makes 1e6-1e7 rows. And its usable
// orders are set by m, not by the moment recurrence's loss of orthogonality. Full
// reorthogonalization (classical Gram-Schmidt, twice) keeps T_m the Jacobi matrix of
// mu in floating point; without it Lanczos produces ghost copies of converged Ritz
// values, which would double-count weight in the rules.

import (
	"fmt"
	"math"

	"github.com/leiaSQ/ADCgo/backend"
)

// SeededResult is the Jacobi matrix of the measure of v0: Alpha (diagonal) and Beta
// (off-diagonal, len(Alpha)-1), and ||v0||^2.
type SeededResult struct {
	Alpha, Beta []float64
	Norm2       float64
	// Invariant reports that the Krylov space of v0 closed before m steps (a beta
	// vanished): the rules are then exact for the whole measure.
	Invariant bool
}

// SolveSeeded runs at most m Lanczos steps of the symmetric operator apply (dst = A src)
// from v0, with full reorthogonalization.
func SolveSeeded(apply func(dst, src []float64), v0 []float64, m int) (SeededResult, error) {
	n := len(v0)
	if m < 1 {
		return SeededResult{}, fmt.Errorf("lanczos: SolveSeeded needs m >= 1")
	}
	m = min(m, n)
	norm2 := dot(v0, v0)
	if norm2 == 0 {
		return SeededResult{}, fmt.Errorf("lanczos: SolveSeeded: the start vector is zero")
	}
	nv := math.Sqrt(norm2)
	Q := make([][]float64, 0, m)
	q := make([]float64, n)
	for i := range q {
		q[i] = v0[i] / nv
	}
	out := SeededResult{Norm2: norm2}
	w := make([]float64, n)
	for k := 0; k < m; k++ {
		Q = append(Q, q)
		apply(w, q)
		a := dot(q, w)
		out.Alpha = append(out.Alpha, a)
		for range 2 { // CGS2 against every basis vector
			for _, p := range Q {
				c := dot(p, w)
				for i := range w {
					w[i] -= c * p[i]
				}
			}
		}
		if k == m-1 {
			break
		}
		b := math.Sqrt(dot(w, w))
		// the breakdown test is relative to the operator's scale on this vector
		if b <= 1e-12*math.Max(math.Abs(a), 1) {
			out.Invariant = true
			break
		}
		out.Beta = append(out.Beta, b)
		q = make([]float64, n)
		for i := range q {
			q[i] = w[i] / b
		}
	}
	return out, nil
}

// Steps is the number of Lanczos steps taken (the largest available rule order).
func (r SeededResult) Steps() int { return len(r.Alpha) }

// GaussRule is the k-point Gauss rule of the seeded measure, nodes ascending: the
// eigenvalues of the leading k x k block of the Jacobi matrix and Norm2 times their
// squared first eigenvector components.
func (r SeededResult) GaussRule(k int) (nodes, weights []float64, err error) {
	if k < 1 || k > len(r.Alpha) {
		return nil, nil, fmt.Errorf("lanczos: Gauss rule of order %d from %d Lanczos steps", k, len(r.Alpha))
	}
	T := backend.NewMat(k, k)
	for i := range k {
		T.Set(i, i, r.Alpha[i])
		if i > 0 {
			T.Set(i, i-1, r.Beta[i-1])
			T.Set(i-1, i, r.Beta[i-1])
		}
	}
	lam, vec := backend.Gonum{}.SymEig(T)
	nodes = make([]float64, k)
	weights = make([]float64, k)
	for i := range k {
		z := vec.At(0, i)
		nodes[i] = lam[i]
		weights[i] = r.Norm2 * z * z
	}
	return nodes, weights, nil
}
