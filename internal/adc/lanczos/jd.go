package lanczos

// jd.go — an interior eigenpair selected by its weight on given rows, converged to a tight
// residual. The decaying state of a multiply ionized cluster can sit tens of eV above the
// bottom of QHQ, inside a dense manifold of charge-transfer and Rydberg pseudo-states, so
// neither block-Lanczos (moments, not eigenvectors) nor Davidson (the lowest roots)
// reaches it, and a small width needs its residual near 1e-10 (the noise floor of the
// coupling vector scales with the residual squared).
//
// The algorithm (first proven on pyscf FCI of such states):
//
//  1. Jacobi-Davidson (Sleijpen & van der Vorst, SIAM J. Matrix Anal. Appl. 17, 401
//     (1996)): in the search space V take the Ritz value theta whose Ritz vector has the
//     largest weight on the target rows, extract the REFINED Ritz vector u minimizing
//     ||(A - theta) V y|| (Jia, Linear Algebra Appl. 259, 1 (1997)), and extend V by an
//     approximate solution of the correction
//     equation (I - uu^T)(A - theta)(I - uu^T) t = -r, t _|_ u (MINRES, preconditioned by
//     the SPD |diag(A) - theta| + 1e-2). Restarts are thick (GD+k, Stathopoulos & Saad,
//     ETNA 7, 163 (1998)): the selected Ritz vector, the previous iterate, and the Ritz
//     vectors nearest theta and heaviest on the targets, with A v rebuilt by the same
//     linear combination so a restart costs no operator application.
//     The refined vector is what makes the interior target converge: with the plain Ritz
//     vector the residual wandered between 1e-7 and 1e-5 for hundreds of iterations while
//     theta was exact to 1e-14 (TestJacobiDavidsonFindsTargetedInteriorState's model; on a
//     4e6-determinant FCI it stalled at 1e-2), because a new Ritz value
//     near theta contaminates the Ritz vector; the refined residual cannot grow as the
//     space grows.
//  2. Shifted inverse iteration (PolishInverse): x <- (A - sigma)^-1 x with sigma a hair
//     below theta, each solve by MINRES; exact RQI (sigma = theta) makes the system
//     singular to working precision and stalls, the 1e-6 offset keeps it solvable. An
//     overlap guard refuses a polish that lands on a different state.

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// JDOptions controls JacobiDavidson.
type JDOptions struct {
	Targets  []int     // rows whose squared weight selects the Ritz vector (required)
	Diag     []float64 // diag(A), for the preconditioner (required)
	Tol      float64   // residual 2-norm at which JD stops (default 1e-3: hand over to PolishInverse)
	MaxIter  int       // outer iterations (default 300)
	Inner    int       // MINRES steps per correction equation (default 40)
	InnerTol float64   // MINRES relative tolerance (default 1e-3)
	MaxSpace int       // search-space size that triggers a restart (default 40)
	Keep     int       // vectors kept at a restart, previous iterate included (default 12)
	// Deflate holds orthonormal vectors the search stays orthogonal to (converged roots).
	Deflate [][]float64
	Log     func(string)
}

func (o JDOptions) withDefaults() JDOptions {
	if o.Tol <= 0 {
		o.Tol = 1e-3
	}
	if o.MaxIter <= 0 {
		o.MaxIter = 300
	}
	if o.Inner <= 0 {
		o.Inner = 40
	}
	if o.InnerTol <= 0 {
		o.InnerTol = 1e-3
	}
	if o.MaxSpace <= 0 {
		o.MaxSpace = 40
	}
	if o.Keep <= 0 {
		o.Keep = 12
	}
	o.Keep = min(o.Keep, o.MaxSpace-1)
	return o
}

// RitzPair is one Ritz pair of the final search space: an approximation to an eigenpair
// of A near the selected one, with its residual so a reader can tell how good it is.
type RitzPair struct {
	Theta        float64
	Residual     float64
	TargetWeight float64
	X            []float64
}

// JDResult is the selected eigenpair and the search space's other Ritz pairs.
type JDResult struct {
	Theta        float64
	X            []float64 // normalized
	Residual     float64   // ||A x - theta x||
	TargetWeight float64
	Iter         int
	Ritz         []RitzPair // every Ritz pair of the final search space, ascending
}

// ErrNotConverged is returned (wrapped) when the iteration budget runs out; the result
// still carries the best approximation.
var ErrNotConverged = errors.New("not converged")

// JacobiDavidson finds the eigenpair of the symmetric operator apply (dst = A src, n rows)
// with the largest weight on o.Targets, starting from the vectors x0.
func JacobiDavidson(apply func(dst, src []float64), n int, x0 [][]float64, o JDOptions) (JDResult, error) {
	o = o.withDefaults()
	if len(o.Targets) == 0 || len(o.Diag) != n {
		return JDResult{}, fmt.Errorf("lanczos: JacobiDavidson needs target rows and the diagonal (%d rows)", n)
	}
	// V is the orthonormal search space, AV its image, G the projected matrix V^T A V and
	// P = (AV)^T (AV), both grown by one row per added vector (O(m n)) rather than rebuilt
	// (O(m^2 n)). P is what the refined Ritz vector needs:
	// ||(A - theta) V y||^2 = y^T (P - 2 theta G + theta^2) y.
	var V, AV, G, P [][]float64
	orth := func(v, av []float64) ([]float64, []float64, bool) {
		for range 2 {
			for _, q := range o.Deflate {
				c := dot(q, v)
				for i := range v {
					v[i] -= c * q[i]
				}
			}
			for k, q := range V {
				c := dot(q, v)
				for i := range v {
					v[i] -= c * q[i]
				}
				for i := range av { // av == nil: nothing to update
					av[i] -= c * AV[k][i]
				}
			}
		}
		nv := math.Sqrt(dot(v, v))
		if nv < 1e-10 {
			return nil, nil, false
		}
		for i := range v {
			v[i] /= nv
		}
		for i := range av {
			av[i] /= nv
		}
		return v, av, true
	}
	add := func(v, av []float64) bool {
		v = append([]float64(nil), v...)
		if av != nil {
			av = append([]float64(nil), av...)
		}
		v, av, ok := orth(v, av)
		if !ok {
			return false
		}
		if av == nil {
			av = make([]float64, n)
			apply(av, v)
		}
		row := make([]float64, len(V)+1)
		prow := make([]float64, len(V)+1)
		for k := range V {
			row[k] = 0.5 * (dot(V[k], av) + dot(v, AV[k]))
			prow[k] = dot(AV[k], av)
		}
		row[len(V)] = dot(v, av)
		prow[len(V)] = dot(av, av)
		V = append(V, v)
		AV = append(AV, av)
		G = append(G, row)
		P = append(P, prow)
		return true
	}
	for _, x := range x0 {
		add(x, nil)
	}
	if len(V) == 0 {
		return JDResult{}, fmt.Errorf("lanczos: JacobiDavidson has no usable start vector")
	}

	var (
		res           JDResult
		uPrev, auPrev []float64
		u             = make([]float64, n)
		au            = make([]float64, n)
		r             = make([]float64, n)
	)
	for it := 0; it < o.MaxIter; it++ {
		m := len(V)
		H := mat.NewSymDense(m, nil)
		for b := range m {
			for a := 0; a <= b; a++ {
				H.SetSym(a, b, G[b][a])
			}
		}
		var eig mat.EigenSym
		if !eig.Factorize(H, true) {
			return res, fmt.Errorf("lanczos: JacobiDavidson projected eigenproblem failed")
		}
		w := eig.Values(nil)
		var U mat.Dense
		eig.VectorsTo(&U)
		// target weight of every Ritz vector: sum over target rows of (V^T U)_row^2
		ov := make([]float64, m)
		for j := range m {
			for _, t := range o.Targets {
				var c float64
				for a := range m {
					c += V[a][t] * U.At(a, j)
				}
				ov[j] += c * c
			}
		}
		k := 0
		for j := range m {
			if ov[j] > ov[k] {
				k = j
			}
		}
		combine := func(dst []float64, src [][]float64, j int) {
			clear(dst)
			for a := range m {
				c := U.At(a, j)
				for i := range dst {
					dst[i] += c * src[a][i]
				}
			}
		}
		// refined Ritz vector at the selected Ritz value, then its own Rayleigh quotient
		thetaK := w[k]
		C := mat.NewSymDense(m, nil)
		for b := range m {
			for a := 0; a <= b; a++ {
				c := P[b][a] - 2*thetaK*G[b][a]
				if a == b {
					c += thetaK * thetaK
				}
				C.SetSym(a, b, c)
			}
		}
		var ceig mat.EigenSym
		if !ceig.Factorize(C, true) {
			return res, fmt.Errorf("lanczos: JacobiDavidson refined extraction failed")
		}
		var Y mat.Dense
		ceig.VectorsTo(&Y)
		clear(u)
		clear(au)
		for a := range m {
			c := Y.At(a, 0) // smallest singular direction
			for i := range u {
				u[i] += c * V[a][i]
				au[i] += c * AV[a][i]
			}
		}
		nu := math.Sqrt(dot(u, u))
		for i := range u {
			u[i] /= nu
			au[i] /= nu
		}
		theta := dot(u, au)
		var uw float64
		for _, t := range o.Targets {
			uw += u[t] * u[t]
		}
		for i := range r {
			r[i] = au[i] - theta*u[i]
		}
		for _, q := range o.Deflate {
			c := dot(q, r)
			for i := range r {
				r[i] -= c * q[i]
			}
		}
		rn := math.Sqrt(dot(r, r))
		res = JDResult{Theta: theta, X: append([]float64(nil), u...), Residual: rn,
			TargetWeight: uw, Iter: it}
		if o.Log != nil {
			o.Log(fmt.Sprintf("jd it=%d m=%d theta=%.12f res=%.2e weight=%.3f", it, m, theta, rn, uw))
		}
		if rn < o.Tol || it == o.MaxIter-1 {
			res.Ritz = make([]RitzPair, m)
			x := make([]float64, n)
			ax := make([]float64, n)
			for j := range m {
				combine(x, V, j)
				combine(ax, AV, j)
				var s float64
				for i := range x {
					d := ax[i] - w[j]*x[i]
					s += d * d
				}
				res.Ritz[j] = RitzPair{Theta: w[j], Residual: math.Sqrt(s), TargetWeight: ov[j],
					X: append([]float64(nil), x...)}
			}
			if rn < o.Tol {
				return res, nil
			}
			break
		}
		if m >= o.MaxSpace {
			// thick restart: the selected vector, then alternately the next nearest in
			// energy and the next heaviest on the targets, then the previous iterate
			idx := []int{k}
			near := argsortBy(m, func(j int) float64 { return math.Abs(w[j] - theta) })
			heavy := argsortBy(m, func(j int) float64 { return -ov[j] })
			seen := map[int]bool{k: true}
			for p := 0; p < m && len(idx) < o.Keep-1; p++ {
				for _, j := range []int{near[p], heavy[p]} {
					if !seen[j] && len(idx) < o.Keep-1 {
						seen[j] = true
						idx = append(idx, j)
					}
				}
			}
			oldV, oldAV := V, AV
			V, AV, G, P = nil, nil, nil, nil
			add(u, au) // the refined vector leads
			x := make([]float64, n)
			ax := make([]float64, n)
			for _, j := range idx {
				clear(x)
				clear(ax)
				for a := range m {
					c := U.At(a, j)
					for i := range x {
						x[i] += c * oldV[a][i]
						ax[i] += c * oldAV[a][i]
					}
				}
				add(x, ax)
			}
			if uPrev != nil {
				add(uPrev, auPrev)
			}
			if o.Log != nil {
				o.Log(fmt.Sprintf("jd restart it=%d kept=%d", it, len(V)))
			}
		}
		uPrev = append(uPrev[:0], u...)
		auPrev = append(auPrev[:0], au...)

		// correction equation, projected against u and the deflated vectors
		Q := append(append([][]float64(nil), o.Deflate...), append([]float64(nil), u...))
		proj := func(v []float64) {
			for _, q := range Q {
				c := dot(q, v)
				for i := range v {
					v[i] -= c * q[i]
				}
			}
		}
		th := theta
		tmp := make([]float64, n)
		op := func(dst, src []float64) {
			copy(tmp, src)
			proj(tmp)
			apply(dst, tmp)
			for i := range dst {
				dst[i] -= th * tmp[i]
			}
			proj(dst)
		}
		pre := func(dst, src []float64) {
			copy(dst, src)
			proj(dst)
			for i := range dst {
				dst[i] /= math.Abs(o.Diag[i]-th) + 1e-2
			}
			proj(dst)
		}
		b := make([]float64, n)
		for i := range b {
			b[i] = -r[i]
		}
		t, _ := Minres(op, b, nil, MinresOptions{RTol: o.InnerTol, MaxIter: o.Inner, Precond: pre})
		proj(t)
		if !add(t, nil) {
			// the correction lies in the space already: fall back to the preconditioned
			// residual, which cannot
			for i := range t {
				t[i] = r[i] / (math.Abs(o.Diag[i]-th) + 1e-2)
			}
			add(t, nil)
		}
	}
	return res, fmt.Errorf("lanczos: JacobiDavidson residual %.2e after %d iterations (theta %.10f): %w",
		res.Residual, o.MaxIter, res.Theta, ErrNotConverged)
}

// argsortBy returns 0..m-1 ordered by key ascending (stable).
func argsortBy(m int, key func(int) float64) []int {
	idx := make([]int, m)
	for i := range idx {
		idx[i] = i
	}
	for i := 1; i < m; i++ {
		for j := i; j > 0 && key(idx[j]) < key(idx[j-1]); j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}
	return idx
}

// PolishOptions controls PolishInverse.
type PolishOptions struct {
	Diag       []float64 // diag(A), for the preconditioner (required)
	Tol        float64   // residual 2-norm goal (default 1e-10)
	Offset     float64   // sigma = theta - Offset (default 1e-6 hartree)
	MaxSteps   int       // inverse-iteration steps (default 8)
	InnerTol   float64   // MINRES relative tolerance per step (default 1e-10)
	Inner      int       // MINRES iteration cap per step (default 800)
	MinOverlap float64   // |<x|x0>| below this is a state swap, refused (default 0.9)
	Log        func(string)
}

// PolishResult is the polished eigenpair and the residual after every step.
type PolishResult struct {
	Theta    float64
	X        []float64
	Residual float64
	Overlap  float64   // |<x|x0>| with the start vector
	Ladder   []float64 // residual before the first and after every step
}

// PolishInverse drives x0 (an approximate eigenvector of the symmetric operator apply) to
// residual o.Tol by shifted inverse iteration, and refuses a result that is no longer the
// same state. It returns the best pair reached with a wrapped ErrNotConverged when the
// residual stops short of the goal — the float64 floor of MINRES at the conditioning of
// (A - sigma) was ~1e-8 on a 1e6-determinant FCI, so the caller decides whether that is
// good enough and reports it.
func PolishInverse(apply func(dst, src []float64), x0 []float64, o PolishOptions) (PolishResult, error) {
	n := len(x0)
	if len(o.Diag) != n {
		return PolishResult{}, fmt.Errorf("lanczos: PolishInverse needs the diagonal (%d rows)", n)
	}
	if o.Tol <= 0 {
		o.Tol = 1e-10
	}
	if o.Offset <= 0 {
		o.Offset = 1e-6
	}
	if o.MaxSteps <= 0 {
		o.MaxSteps = 8
	}
	if o.InnerTol <= 0 {
		o.InnerTol = 1e-10
	}
	if o.Inner <= 0 {
		o.Inner = 800
	}
	if o.MinOverlap <= 0 {
		o.MinOverlap = 0.9
	}
	start := append([]float64(nil), x0...)
	ns := math.Sqrt(dot(start, start))
	for i := range start {
		start[i] /= ns
	}
	x := append([]float64(nil), start...)
	ax := make([]float64, n)
	eval := func() (float64, float64) {
		apply(ax, x)
		th := dot(x, ax)
		var s float64
		for i := range x {
			d := ax[i] - th*x[i]
			s += d * d
		}
		return th, math.Sqrt(s)
	}
	th, rn := eval()
	out := PolishResult{Theta: th, X: append([]float64(nil), x...), Residual: rn, Overlap: 1,
		Ladder: []float64{rn}}
	for step := 1; step <= o.MaxSteps && rn > o.Tol; step++ {
		sigma := th - o.Offset
		op := func(dst, src []float64) {
			apply(dst, src)
			for i := range dst {
				dst[i] -= sigma * src[i]
			}
		}
		pre := func(dst, src []float64) {
			for i := range dst {
				dst[i] = src[i] / (math.Abs(o.Diag[i]-sigma) + 1e-3)
			}
		}
		y, info := Minres(op, x, nil, MinresOptions{RTol: o.InnerTol, MaxIter: o.Inner, Precond: pre})
		ny := math.Sqrt(dot(y, y))
		for i := range x {
			x[i] = y[i] / ny
		}
		th, rn = eval()
		out.Ladder = append(out.Ladder, rn)
		if o.Log != nil {
			o.Log(fmt.Sprintf("polish step=%d theta=%.12f res=%.2e minres it=%d rel=%.1e",
				step, th, rn, info.Iter, info.RelResid))
		}
		if rn < out.Residual {
			out.Theta, out.Residual = th, rn
			out.X = append(out.X[:0], x...)
		}
	}
	out.Overlap = math.Abs(dot(out.X, start))
	if out.Overlap < o.MinOverlap {
		return out, fmt.Errorf("lanczos: the polish moved to a different state (overlap %.3f with "+
			"the start vector, theta %.10f); the target is not isolated at this residual", out.Overlap, out.Theta)
	}
	if out.Residual > o.Tol {
		return out, fmt.Errorf("lanczos: polish reached residual %.2e, goal %.1e: %w",
			out.Residual, o.Tol, ErrNotConverged)
	}
	return out, nil
}
