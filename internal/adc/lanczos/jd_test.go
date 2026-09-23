package lanczos

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// interiorModel is a symmetric matrix with a known dense interior spectrum: a random
// orthogonal rotation of diag(e) that is "diagonally dominant" in the sense Davidson-type
// preconditioners need (rotation confined to small angles), with one eigenvector
// concentrated on a chosen target row set. It mimics an interior QHQ problem: the wanted
// state sits in the middle of the spectrum with neighbours closer than its gap to the
// bottom.
type interiorModel struct {
	n       int
	A       *mat.SymDense
	diag    []float64
	targets []int
}

func newInteriorModel(n int, seed uint64, coupling, density float64) interiorModel {
	rng := rand.New(rand.NewPCG(seed, 7))
	A := mat.NewSymDense(n, nil)
	for i := range n {
		A.SetSym(i, i, float64(i)*0.05+0.2*rng.Float64())
		for j := range i {
			if rng.Float64() < density {
				A.SetSym(i, j, coupling*rng.NormFloat64())
			}
		}
	}
	d := make([]float64, n)
	for i := range n {
		d[i] = A.At(i, i)
	}
	return interiorModel{n: n, A: A, diag: d, targets: []int{n / 2}}
}

func (m interiorModel) apply(dst, src []float64) {
	for i := range m.n {
		var s float64
		for j := range m.n {
			s += m.A.At(i, j) * src[j]
		}
		dst[i] = s
	}
}

// exact returns the eigenpair with the largest weight on the targets.
func (m interiorModel) exact(t *testing.T) (float64, []float64) {
	var eig mat.EigenSym
	if !eig.Factorize(m.A, true) {
		t.Fatal("eig")
	}
	w := eig.Values(nil)
	var V mat.Dense
	eig.VectorsTo(&V)
	best, bw := 0, -1.0
	for j := range m.n {
		var s float64
		for _, r := range m.targets {
			s += V.At(r, j) * V.At(r, j)
		}
		if s > bw {
			best, bw = j, s
		}
	}
	x := make([]float64, m.n)
	for i := range x {
		x[i] = V.At(i, best)
	}
	return w[best], x
}

// TestMinresIndefinite: MINRES solves a symmetric indefinite system (A - sigma with sigma
// inside the spectrum) to the requested tolerance, with and without a preconditioner.
func TestMinresIndefinite(t *testing.T) {
	m := newInteriorModel(120, 1, 0.02, 0.2)
	sigma := m.diag[60] + 0.013
	op := func(dst, src []float64) {
		m.apply(dst, src)
		for i := range dst {
			dst[i] -= sigma * src[i]
		}
	}
	rng := rand.New(rand.NewPCG(3, 3))
	b := make([]float64, m.n)
	for i := range b {
		b[i] = rng.NormFloat64()
	}
	for _, pre := range []func(dst, src []float64){nil, func(dst, src []float64) {
		for i := range dst {
			dst[i] = src[i] / (math.Abs(m.diag[i]-sigma) + 1e-3)
		}
	}} {
		x, info := Minres(op, b, nil, MinresOptions{RTol: 1e-12, MaxIter: 2000, Precond: pre})
		ax := make([]float64, m.n)
		op(ax, x)
		var rn, bn float64
		for i := range ax {
			rn += (ax[i] - b[i]) * (ax[i] - b[i])
			bn += b[i] * b[i]
		}
		rel := math.Sqrt(rn / bn)
		t.Logf("preconditioned=%v: %d iterations, true relative residual %.1e", pre != nil, info.Iter, rel)
		if !info.Converged || rel > 1e-9 {
			t.Errorf("MINRES: converged=%v, relative residual %.2e", info.Converged, rel)
		}
	}
}

// TestJacobiDavidsonFindsTargetedInteriorState: JD, started from the unit vector on the
// target row, converges to the eigenpair with the largest target weight (not the lowest),
// through several thick restarts; PolishInverse then takes it to 1e-10 without swapping
// states, and the result agrees with the dense eigensolver.
func TestJacobiDavidsonFindsTargetedInteriorState(t *testing.T) {
	m := newInteriorModel(300, 2, 0.08, 0.5)
	e, xe := m.exact(t)
	x0 := make([]float64, m.n)
	x0[m.targets[0]] = 1
	restarts := 0
	jd, err := JacobiDavidson(m.apply, m.n, [][]float64{x0}, JDOptions{
		Targets: m.targets, Diag: m.diag, Tol: 1e-9, MaxSpace: 8, Keep: 4, Inner: 10, MaxIter: 500,
		Log: func(s string) {
			if len(s) > 10 && s[:10] == "jd restart" {
				restarts++
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if restarts == 0 {
		t.Error("the test did not exercise a thick restart")
	}
	pol, err := PolishInverse(m.apply, jd.X, PolishOptions{Diag: m.diag, Tol: 1e-10})
	if err != nil {
		t.Fatal(err)
	}
	var ov float64
	for i := range xe {
		ov += xe[i] * pol.X[i]
	}
	t.Logf("JD %d iterations, %d restarts, residual %.1e; polish ladder %.1e; |theta - exact| %.1e, |<x|exact>| %.12f",
		jd.Iter, restarts, jd.Residual, pol.Ladder, math.Abs(pol.Theta-e), math.Abs(ov))
	if math.Abs(pol.Theta-e) > 1e-12 || math.Abs(math.Abs(ov)-1) > 1e-10 {
		t.Errorf("polished pair theta %.14f vs exact %.14f, overlap %.12f", pol.Theta, e, ov)
	}
	if len(jd.Ritz) == 0 {
		t.Error("no Ritz pairs reported for the audit")
	}
}

// TestPolishRefusesStateSwap: a start vector 0.6 v40 + 0.8 v41 converges under inverse
// iteration onto v41, with overlap 0.8 with the start: the polish found A state, but not
// the one handed over, and the overlap guard must say so rather than return it.
func TestPolishRefusesStateSwap(t *testing.T) {
	m := newInteriorModel(80, 4, 0.02, 0.2)
	var eig mat.EigenSym
	eig.Factorize(m.A, true)
	var V mat.Dense
	eig.VectorsTo(&V)
	x0 := make([]float64, m.n)
	for i := range x0 {
		x0[i] = 0.6*V.At(i, 40) + 0.8*V.At(i, 41)
	}
	pol, err := PolishInverse(m.apply, x0, PolishOptions{Diag: m.diag, Tol: 1e-10, MinOverlap: 0.9})
	t.Logf("polish of a 0.6/0.8 mixture: residual %.1e, overlap %.3f: %v", pol.Residual, pol.Overlap, err)
	if err == nil || errors.Is(err, ErrNotConverged) {
		t.Fatalf("a state swap was accepted (err %v)", err)
	}
}
