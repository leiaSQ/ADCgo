package lanczos

// minres.go — preconditioned MINRES for symmetric (possibly indefinite) systems, in
// FP64 on the host. The interior eigensolver (jd.go) needs it twice: the Jacobi-Davidson
// correction equation (I - uu^T)(A - theta)(I - uu^T) t = -r, and the shifted inverse
// iteration that polishes the decaying state to a residual ~1e-10, where (A - sigma) is
// indefinite by construction (sigma sits inside the spectrum). Conjugate gradients
// needs a definite operator; MINRES does not.
//
// The recurrence is Paige & Saunders (SIAM J. Numer. Anal. 12, 617 (1975)) in the
// preconditioned form of scipy.sparse.linalg.minres: the Lanczos process on M^-1 A with an SPD preconditioner M, a QR
// factorization updated by Givens rotations, and x updated from the three-term
// direction recurrence. phibar is the M^-1-norm of the residual of the current iterate.

import (
	"math"
)

// MinresOptions controls a MINRES solve.
type MinresOptions struct {
	// RTol stops the solve once the preconditioned residual norm has dropped by this
	// factor relative to the right-hand side's.
	RTol float64
	// MaxIter caps the number of operator applications.
	MaxIter int
	// Precond applies M^-1 (dst = M^-1 src); nil means the identity. It must be
	// symmetric positive definite on the space the operator acts on.
	Precond func(dst, src []float64)
}

// MinresInfo reports how a solve ended.
type MinresInfo struct {
	Iter      int
	RelResid  float64 // final preconditioned residual norm / initial
	Converged bool    // RelResid <= RTol
	Breakdown bool    // the Krylov space became invariant (beta = 0): x is exact there
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Minres solves A x = b for symmetric A (apply computes dst = A src). x0 may be nil
// (zero start). The returned x is freshly allocated.
func Minres(apply func(dst, src []float64), b, x0 []float64, o MinresOptions) ([]float64, MinresInfo) {
	n := len(b)
	pre := o.Precond
	if pre == nil {
		pre = func(dst, src []float64) { copy(dst, src) }
	}
	x := make([]float64, n)
	if x0 != nil {
		copy(x, x0)
	}
	r1 := make([]float64, n)
	tmp := make([]float64, n)
	if x0 != nil {
		apply(tmp, x)
		for i := range r1 {
			r1[i] = b[i] - tmp[i]
		}
	} else {
		copy(r1, b)
	}
	y := make([]float64, n)
	pre(y, r1)
	beta1 := dot(r1, y)
	if beta1 < 0 {
		panic("lanczos: Minres preconditioner is not positive definite")
	}
	beta1 = math.Sqrt(beta1)
	info := MinresInfo{}
	if beta1 == 0 {
		info.Converged = true
		return x, info
	}
	r2 := append([]float64(nil), r1...)
	v := make([]float64, n)
	w := make([]float64, n)
	w1 := make([]float64, n)
	w2 := make([]float64, n)

	var (
		oldb, beta  = 0.0, beta1
		dbar, epsln float64
		phibar      = beta1
		cs, sn      = -1.0, 0.0
		maxit       = max(1, o.MaxIter)
		rtol        = o.RTol
		epsMach     = math.Nextafter(1, 2) - 1
	)
	for itn := 1; itn <= maxit; itn++ {
		info.Iter = itn
		s := 1 / beta
		for i := range v {
			v[i] = s * y[i]
		}
		apply(y, v)
		if itn >= 2 {
			f := beta / oldb
			for i := range y {
				y[i] -= f * r1[i]
			}
		}
		alfa := dot(v, y)
		f := alfa / beta
		for i := range y {
			y[i] -= f * r2[i]
		}
		r1, r2 = r2, r1
		copy(r2, y)
		pre(y, r2)
		oldb = beta
		bb := dot(r2, y)
		if bb < 0 {
			// A projected preconditioner (P D^-1 P) is only semidefinite, and on a vector
			// that is already in its null space rounding can leave -1e-30; only a clearly
			// negative value means an indefinite M.
			if bb < -1e-10*oldb*oldb {
				panic("lanczos: Minres preconditioner is not positive definite")
			}
			bb = 0
		}
		beta = math.Sqrt(bb)

		oldeps := epsln
		delta := cs*dbar + sn*alfa
		gbar := sn*dbar - cs*alfa
		epsln = sn * beta
		dbar = -cs * beta
		gamma := math.Max(math.Hypot(gbar, beta), epsMach)
		cs, sn = gbar/gamma, beta/gamma
		phi := cs * phibar
		phibar = sn * phibar

		denom := 1 / gamma
		w1, w2, w = w2, w, w1
		for i := range w {
			w[i] = (v[i] - oldeps*w1[i] - delta*w2[i]) * denom
			x[i] += phi * w[i]
		}
		info.RelResid = phibar / beta1
		if info.RelResid <= rtol {
			info.Converged = true
			return x, info
		}
		if beta == 0 {
			info.Breakdown = true
			info.Converged = info.RelResid <= rtol
			return x, info
		}
	}
	return x, info
}
