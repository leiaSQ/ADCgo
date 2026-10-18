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
//     The refined vector is blind to the target, though: with a poorly converged target
//     and an accurately converged neighbour in V, the neighbour has the smaller residual at
//     theta and the refined vector jumps to it (on a 1e6-row He5 Q space the target weight
//     went 0.29 -> 0.000 for 25 iterations). When the refined vector keeps less than half
//     the selected Ritz vector's target weight, the Ritz vector is used instead.
//     The correction equation's MINRES budget doubles (up to InnerMax) whenever the outer
//     residual has not fallen by 30% over three iterations.
//  2. Shifted inverse iteration (PolishInverse): x <- (A - sigma)^-1 x with sigma a hair
//     below theta, each solve by MINRES; exact RQI (sigma = theta) makes the system
//     singular to working precision and stalls, the 1e-6 offset keeps it solvable. An
//     overlap guard refuses a polish that lands on a different state. A step whose MINRES
//     solve stops above BailRel is noise, not an inverse-iteration step: the polish stops
//     there and keeps the best pair (on the He5 point, 16 such steps of 800 applies each
//     were most of a 53 h solve and left the residual where JD had handed it over).
//
// The O(m n) vector work (Gram-Schmidt, Ritz combinations, projections) is split over
// rows with parallel.Chunks; per-worker partial dot products are reduced in worker order,
// so results do not depend on scheduling.

import (
	"errors"
	"fmt"
	"math"

	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
	"gonum.org/v1/gonum/mat"
)

// parMinRows is the row count per worker below which the vector kernels stay serial: a
// Chunks round costs ~80 us, a 16k-row axpy ~10 us.
const parMinRows = 1 << 14

func vecWorkers(n int) int { return min(parallel.ChunkWorkers(n), max(1, n/parMinRows)) }

// rowsDo runs body over [0, n) split into contiguous row ranges.
func rowsDo(n int, body func(lo, hi int)) {
	parallel.Chunks(n, vecWorkers(n), func(_, lo, hi int) { body(lo, hi) })
}

// dotsInto sets out[k] = <qs[k], v>.
func dotsInto(out []float64, qs [][]float64, v []float64) {
	m, n := len(qs), len(v)
	if m == 0 {
		return
	}
	w := vecWorkers(n)
	part := make([]float64, w*m)
	parallel.Chunks(n, w, func(wk, lo, hi int) {
		acc := part[wk*m : (wk+1)*m]
		for k, q := range qs {
			var s float64
			for i := lo; i < hi; i++ {
				s += q[i] * v[i]
			}
			acc[k] = s
		}
	})
	for k := range m {
		var s float64
		for wk := range w {
			s += part[wk*m+k]
		}
		out[k] = s
	}
}

func pdot(a, b []float64) float64 {
	var out [1]float64
	dotsInto(out[:], [][]float64{a}, b)
	return out[0]
}

// subCombo sets v -= sum_k c[k] qs[k].
func subCombo(v []float64, qs [][]float64, c []float64) {
	if len(qs) == 0 {
		return
	}
	rowsDo(len(v), func(lo, hi int) {
		for k, q := range qs {
			ck := c[k]
			for i := lo; i < hi; i++ {
				v[i] -= ck * q[i]
			}
		}
	})
}

// combo sets dst = sum_a c(a) src[a].
func combo(dst []float64, src [][]float64, c func(a int) float64) {
	cs := make([]float64, len(src))
	for a := range cs {
		cs[a] = c(a)
	}
	rowsDo(len(dst), func(lo, hi int) {
		d := dst[lo:hi]
		clear(d)
		for a, v := range src {
			ca := cs[a]
			for i := lo; i < hi; i++ {
				dst[i] += ca * v[i]
			}
		}
	})
}

// project removes from v its components along the orthonormal vectors qs (twice, CGS2).
func project(v []float64, qs [][]float64) {
	if len(qs) == 0 {
		return
	}
	c := make([]float64, len(qs))
	for range 2 {
		dotsInto(c, qs, v)
		subCombo(v, qs, c)
	}
}

// JDOptions controls JacobiDavidson.
type JDOptions struct {
	Targets  []int     // rows whose squared weight selects the Ritz vector (required)
	Diag     []float64 // diag(A), for the preconditioner (required)
	Tol      float64   // residual 2-norm at which JD stops (default 1e-3: hand over to PolishInverse)
	MaxIter  int       // outer iterations (default 300)
	Inner    int       // MINRES steps per correction equation at the start (default 40)
	InnerMax int       // cap of the adaptive MINRES budget (default 8 Inner)
	InnerTol float64   // MINRES relative tolerance (default 1e-3)
	MaxSpace int       // search-space size that triggers a restart (default 40)
	Keep     int       // vectors kept at a restart, previous iterate included (default 12)
	// Deflate holds orthonormal vectors the search stays orthogonal to (converged roots).
	Deflate [][]float64
	// Follow, when set, selects the Ritz vector with the largest squared overlap with it
	// instead of the largest target weight: a known state (the same eigenvector of a
	// nearby operator, say at another transfer scaling) is tracked by its shape, which
	// target weight cannot do when charge resonance spreads that weight over several
	// states. It need not be normalized.
	Follow []float64
	Log    func(string)
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
	if o.InnerMax < o.Inner {
		o.InnerMax = 8 * o.Inner
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
	// InnerRel is the relative residual the last correction-equation solve stopped at: a
	// value near 1 says MINRES cannot solve shifted systems of A at this budget, which is
	// what PolishInverse would need.
	InnerRel float64
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
		project(v, o.Deflate)
		c := make([]float64, len(V))
		for range 2 { // CGS2; av follows v by the same coefficients
			dotsInto(c, V, v)
			subCombo(v, V, c)
			if av != nil { // av == nil: nothing to update
				subCombo(av, AV, c)
			}
		}
		nv := math.Sqrt(pdot(v, v))
		if nv < 1e-10 {
			return nil, nil, false
		}
		rowsDo(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				v[i] /= nv
			}
			if av != nil {
				for i := lo; i < hi; i++ {
					av[i] /= nv
				}
			}
		})
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
		va := make([]float64, len(V))
		dotsInto(row[:len(V)], V, av)
		dotsInto(va, AV, v)
		dotsInto(prow[:len(V)], AV, av)
		for k := range V {
			row[k] = 0.5 * (row[k] + va[k])
		}
		row[len(V)] = pdot(v, av)
		prow[len(V)] = pdot(av, av)
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
		inner         = o.Inner
		innerInfo     MinresInfo
		hist          []float64
		followNorm2   = 1.0
	)
	if o.Follow != nil {
		if len(o.Follow) != n {
			return JDResult{}, fmt.Errorf("lanczos: JacobiDavidson Follow has %d rows, the operator %d", len(o.Follow), n)
		}
		if followNorm2 = pdot(o.Follow, o.Follow); followNorm2 == 0 {
			return JDResult{}, fmt.Errorf("lanczos: JacobiDavidson Follow is the zero vector")
		}
	}
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
		score := ov
		if o.Follow != nil {
			fv := make([]float64, m)
			dotsInto(fv, V, o.Follow)
			score = make([]float64, m)
			for j := range m {
				var c float64
				for a := range m {
					c += fv[a] * U.At(a, j)
				}
				score[j] = c * c / followNorm2
			}
		}
		k := 0
		for j := range m {
			if score[j] > score[k] {
				k = j
			}
		}
		combine := func(dst []float64, src [][]float64, j int) {
			combo(dst, src, func(a int) float64 { return U.At(a, j) })
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
		combo(u, V, func(a int) float64 { return Y.At(a, 0) }) // smallest singular direction
		combo(au, AV, func(a int) float64 { return Y.At(a, 0) })
		nu := math.Sqrt(pdot(u, u))
		rowsDo(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				u[i] /= nu
				au[i] /= nu
			}
		})
		targetWeight := func(x []float64) float64 {
			var s float64
			for _, t := range o.Targets {
				s += x[t] * x[t]
			}
			return s
		}
		uw := targetWeight(u)
		selOf := func(x []float64) float64 {
			if o.Follow == nil {
				return targetWeight(x)
			}
			c := pdot(x, o.Follow)
			return c * c / followNorm2
		}
		vec := "refined"
		if selOf(u) < 0.5*score[k] {
			// the refined vector left the target for a better-converged neighbour
			combine(u, V, k)
			combine(au, AV, k)
			uw, vec = targetWeight(u), "ritz"
		}
		theta := pdot(u, au)
		rowsDo(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				r[i] = au[i] - theta*u[i]
			}
		})
		project(r, o.Deflate)
		rn := math.Sqrt(pdot(r, r))
		res = JDResult{Theta: theta, X: append([]float64(nil), u...), Residual: rn,
			TargetWeight: uw, Iter: it, InnerRel: innerInfo.RelResid}
		if o.Log != nil {
			follow := ""
			if o.Follow != nil {
				follow = fmt.Sprintf(" follow=%.4f", selOf(u))
			}
			o.Log(fmt.Sprintf("jd it=%d m=%d theta=%.12f res=%.2e weight=%.3f%s (%s; ritz %.12f weight %.3f) inner=%d/%d rel=%.1e",
				it, m, theta, rn, uw, follow, vec, w[k], ov[k], innerInfo.Iter, inner, innerInfo.RelResid))
		}
		hist = append(hist, rn)
		if h := len(hist); h > 3 && rn > 0.7*hist[h-4] && inner < o.InnerMax {
			inner = min(2*inner, o.InnerMax)
			if o.Log != nil {
				o.Log(fmt.Sprintf("jd it=%d residual stagnating (%.2e three iterations ago): MINRES budget %d",
					it, hist[h-4], inner))
			}
			hist = hist[:0]
		}
		if rn < o.Tol || it == o.MaxIter-1 {
			res.Ritz = make([]RitzPair, m)
			x := make([]float64, n)
			ax := make([]float64, n)
			for j := range m {
				combine(x, V, j)
				combine(ax, AV, j)
				wj := w[j]
				rowsDo(n, func(lo, hi int) {
					for i := lo; i < hi; i++ {
						ax[i] -= wj * x[i]
					}
				})
				res.Ritz[j] = RitzPair{Theta: w[j], Residual: math.Sqrt(pdot(ax, ax)), TargetWeight: ov[j],
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
				combo(x, oldV, func(a int) float64 { return U.At(a, j) })
				combo(ax, oldAV, func(a int) float64 { return U.At(a, j) })
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
		proj := func(v []float64) { project(v, Q) }
		th := theta
		tmp := make([]float64, n)
		op := func(dst, src []float64) {
			copy(tmp, src)
			proj(tmp)
			apply(dst, tmp)
			rowsDo(n, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					dst[i] -= th * tmp[i]
				}
			})
			proj(dst)
		}
		pre := func(dst, src []float64) {
			copy(dst, src)
			proj(dst)
			rowsDo(n, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					dst[i] /= math.Abs(o.Diag[i]-th) + 1e-2
				}
			})
			proj(dst)
		}
		b := make([]float64, n)
		for i := range b {
			b[i] = -r[i]
		}
		var t []float64
		t, innerInfo = Minres(op, b, nil, MinresOptions{RTol: o.InnerTol, MaxIter: inner, Precond: pre})
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
	// BailRel stops the polish at the first step whose MINRES solve ends above this relative
	// residual: that step's direction is not (A - sigma)^-1 x and is discarded (default 1e-2).
	BailRel float64
	Log     func(string)
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
	if o.BailRel <= 0 {
		o.BailRel = 1e-2
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
		if info.RelResid > o.BailRel && !info.Breakdown {
			if o.Log != nil {
				o.Log(fmt.Sprintf("polish step=%d: MINRES stopped at rel %.1e after %d steps (bail-out %.0e); "+
					"the shifted system is not solved at this budget, stopping with residual %.2e",
					step, info.RelResid, info.Iter, o.BailRel, out.Residual))
			}
			break
		}
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
