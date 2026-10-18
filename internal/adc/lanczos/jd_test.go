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

// TestPolishStopsOnFailedSolve: when MINRES cannot solve the shifted system within its cap
// (the He5 production point: 800 steps left relative residuals of 0.2-0.6), an inverse-
// iteration step on that solution is noise. The polish must stop at the first failed solve,
// keep the best pair it had, and say it did not converge, instead of spending its whole
// step budget on garbage directions.
func TestPolishStopsOnFailedSolve(t *testing.T) {
	m := newInteriorModel(300, 2, 0.08, 0.5)
	_, xe := m.exact(t)
	rng := rand.New(rand.NewPCG(5, 5))
	x0 := make([]float64, m.n)
	for i := range x0 {
		x0[i] = xe[i] + 1e-3*rng.NormFloat64()
	}
	applies := 0
	apply := func(dst, src []float64) { applies++; m.apply(dst, src) }
	const inner = 2
	var trace []string
	pol, err := PolishInverse(apply, x0, PolishOptions{Diag: m.diag, Tol: 1e-12, Inner: inner,
		Log: func(s string) { trace = append(trace, s) }})
	t.Logf("polish with a %d-step MINRES cap: %d applies, ladder %.2e, err %v, trace %q",
		inner, applies, pol.Ladder, err, trace)
	if !errors.Is(err, ErrNotConverged) {
		t.Fatalf("err = %v, want ErrNotConverged", err)
	}
	if applies > 2*(inner+2) {
		t.Errorf("%d applies: the polish kept stepping on failed solves (one step costs %d)", applies, inner+1)
	}
	if pol.Residual > pol.Ladder[0] {
		t.Errorf("returned residual %.2e is worse than the start %.2e", pol.Residual, pol.Ladder[0])
	}
}

// TestJacobiDavidsonRefinedKeepsTarget: the refined Ritz vector minimizes ||(A - theta) V y||
// at the selected Ritz value, which is blind to the target. With a poorly converged target
// and an accurately converged neighbour in the space, the neighbour has the smaller residual
// at theta and the refined vector jumps to it (the He5 production run lost its target for
// 25 iterations this way, weight 0.29 -> 0.000). JD must keep a vector that still carries
// the target, and from the same start converge to the targeted eigenpair.
func TestJacobiDavidsonRefinedKeepsTarget(t *testing.T) {
	m := newInteriorModel(300, 2, 0.08, 0.5)
	var eig mat.EigenSym
	if !eig.Factorize(m.A, true) {
		t.Fatal("eig")
	}
	w := eig.Values(nil)
	var V mat.Dense
	eig.VectorsTo(&V)
	jt, bw := 0, -1.0
	for j := range m.n {
		c := V.At(m.targets[0], j)
		if c*c > bw {
			jt, bw = j, c*c
		}
	}
	jn := jt - 1
	if jt+1 < m.n && math.Abs(w[jt+1]-w[jt]) < math.Abs(w[jt-1]-w[jt]) {
		jn = jt + 1
	}
	rng := rand.New(rand.NewPCG(9, 9))
	neigh := make([]float64, m.n)
	rough := make([]float64, m.n)
	for i := range rough {
		neigh[i] = V.At(i, jn)
		rough[i] = V.At(i, jt) + 0.05*rng.NormFloat64()
	}
	var rw, rn float64
	for i := range rough {
		rn += rough[i] * rough[i]
	}
	for _, r := range m.targets {
		rw += rough[r] * rough[r] / rn
	}
	one, _ := JacobiDavidson(m.apply, m.n, [][]float64{neigh, rough}, JDOptions{
		Targets: m.targets, Diag: m.diag, MaxIter: 1})
	t.Logf("target eigenpair %d at %.6f (weight %.3f), neighbour %d at %.6f; rough start weight %.3f; "+
		"first JD vector weight %.3f", jt, w[jt], bw, jn, w[jn], rw, one.TargetWeight)
	if one.TargetWeight < 0.5*rw {
		t.Errorf("the first JD vector carries target weight %.3f, the start had %.3f: the refined "+
			"vector left the target", one.TargetWeight, rw)
	}
	jd, err := JacobiDavidson(m.apply, m.n, [][]float64{neigh, rough}, JDOptions{
		Targets: m.targets, Diag: m.diag, Tol: 1e-9, MaxIter: 300})
	if err != nil {
		t.Fatal(err)
	}
	var ov float64
	for i := range jd.X {
		ov += jd.X[i] * V.At(i, jt)
	}
	t.Logf("converged: %d iterations, theta %.12f (exact %.12f), |overlap| %.12f", jd.Iter, jd.Theta, w[jt], math.Abs(ov))
	if math.Abs(jd.Theta-w[jt]) > 1e-10 || 1-math.Abs(ov) > 1e-8 {
		t.Errorf("JD converged to theta %.12f (overlap %.6f with the target), want %.12f", jd.Theta, ov, w[jt])
	}
}

// TestJacobiDavidsonFollowsShape: with Follow set to a noisy copy of a neighbour of the
// target-heaviest eigenpair, JD started on the target row must converge to that NEIGHBOUR
// (the state is tracked by shape, as a charge-resonance branch is across transfer scalings),
// and without Follow to the target-heaviest one.
func TestJacobiDavidsonFollowsShape(t *testing.T) {
	m := newInteriorModel(300, 2, 0.08, 0.5)
	var eig mat.EigenSym
	if !eig.Factorize(m.A, true) {
		t.Fatal("eig")
	}
	w := eig.Values(nil)
	var V mat.Dense
	eig.VectorsTo(&V)
	jt, bw := 0, -1.0
	for j := range m.n {
		c := V.At(m.targets[0], j)
		if c*c > bw {
			jt, bw = j, c*c
		}
	}
	jn, nw := -1, -1.0 // the next heaviest on the target: a state the start vector also overlaps
	for j := range m.n {
		c := V.At(m.targets[0], j)
		if j != jt && c*c > nw {
			jn, nw = j, c*c
		}
	}
	rng := rand.New(rand.NewPCG(11, 11))
	follow := make([]float64, m.n)
	for i := range follow {
		follow[i] = V.At(i, jn) + 0.02*rng.NormFloat64()
	}
	x0 := make([]float64, m.n)
	x0[m.targets[0]] = 1
	for _, c := range []struct {
		name   string
		follow []float64
		want   int
	}{{"by weight", nil, jt}, {"by shape", follow, jn}} {
		jd, err := JacobiDavidson(m.apply, m.n, [][]float64{x0}, JDOptions{
			Targets: m.targets, Diag: m.diag, Tol: 1e-9, MaxIter: 400, Follow: c.follow})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		var ov float64
		for i := range jd.X {
			ov += jd.X[i] * V.At(i, c.want)
		}
		t.Logf("%s: %d iterations, theta %.10f, want eigenpair %d at %.10f (target weights %.3f / %.3f), |overlap| %.10f",
			c.name, jd.Iter, jd.Theta, c.want, w[c.want], bw, nw, math.Abs(ov))
		if math.Abs(jd.Theta-w[c.want]) > 1e-10 || 1-math.Abs(ov) > 1e-8 {
			t.Errorf("%s: converged to theta %.12f (overlap %.6f), want %.12f", c.name, jd.Theta, ov, w[c.want])
		}
	}
}
