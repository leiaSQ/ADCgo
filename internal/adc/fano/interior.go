package fano

// interior.go — the decaying state as an INTERIOR eigenvector of QHQ, selected by its
// weight on given configurations and converged to a tight residual.
//
// SelectDiscrete assumes |Phi> is among the lowest QMQ roots, which holds for a single
// inner-valence vacancy (every other Q configuration carries an additional hole and lies
// higher). It fails for multiply ionized initial states (two He2+ sites, say), which sit
// tens of eV above the charge-transfer states of Q, inside a dense manifold of Rydberg and
// charge-resonance states, and a small width needs a residual near 1e-10 because the
// noise floor of the coupling vector grows with the residual squared. The pipeline is
//
//  1. a start vector from the dense eigenproblem of the main class (plus the target
//     rows), taking the eigenvector heaviest on the targets;
//  2. lanczos.JacobiDavidson from each candidate start to a SCREENING residual (1e-3):
//     enough to tell which converged state carries the most target weight;
//  3. Jacobi-Davidson again on the winner only, from its vector and its neighbours in the
//     final search space, down to the hand-over residual (1e-6);
//  4. lanczos.PolishInverse, shifted inverse iteration to the goal, with an overlap guard
//     against landing on a neighbour.
// Inverse iteration is handed only a vector JD has taken to 1e-6: from 1e-3 inside a dense
// manifold (the He5 production Q, 1e6 rows, three states within 0.1 eV) MINRES could not
// solve the shifted system in 800 steps, every step was noise, and polishing two candidates
// that way took most of a 53 h solve without lowering the residual. The losing candidates
// are reported at their screening residual and never polished.
//
// The final Jacobi-Davidson search space also yields the C4 audit: its Ritz pairs within
// a window of E_Phi, with residual, target weight and class weights. A Ritz pair of a
// ~40-vector space is an approximation, and its residual says how good; the audit lists
// the Q states the discrete state could be confused or mixed with (charge-resonance
// partners of a He2+ site can hold a large share of the weight).

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"gonum.org/v1/gonum/mat"
)

// InteriorOptions controls SelectInterior.
type InteriorOptions struct {
	// Targets are the Q rows the state is selected by (required).
	Targets []int
	// GuessRows are diagonalized densely for the start vector; nil means the main class.
	// The targets are always added.
	GuessRows []int
	// Tol is the residual goal (default 1e-10 hartree).
	Tol float64
	// AuditWindow is the half-width of the C4 audit around E_Phi (default 0.5 eV).
	AuditWindow float64
	// ScreenTol is the JD residual at which candidates are compared (default 1e-3);
	// HandOver the residual the winner is taken to before the polish (default 1e-6, or Tol
	// if that is looser).
	ScreenTol, HandOver float64
	// Candidates is how many dense-start eigenvectors (those with at least half the
	// largest target weight, heaviest first) are each followed to convergence; the
	// converged state with the largest target weight wins (default 3). One is not
	// enough when charge resonance splits the target character nearly evenly: the
	// Jacobi-Davidson iteration follows the state it started near, which need not be
	// the heaviest one.
	Candidates int
	// Branch picks the converged candidate by rank of target weight, 0 = the heaviest.
	// When charge resonance splits the target character (Ambiguous), each branch is a
	// different decaying state and has its own width.
	Branch int
	// Guess, when set, replaces the dense start: Jacobi-Davidson starts from it and
	// follows it by overlap (lanczos.JDOptions.Follow) instead of by target weight, so a
	// branch converged at one transfer scaling is tracked to another. There are then no
	// alternatives and no ambiguity check.
	Guess []float64
	// JD and Polish tune the two stages; Targets and Diag are filled in here.
	JD     lanczos.JDOptions
	Polish lanczos.PolishOptions
	Log    func(string)
}

// AuditEntry is one Q Ritz pair near E_Phi.
type AuditEntry struct {
	Energy       float64            // hartree
	Residual     float64            // of the Ritz pair; large means only indicative
	TargetWeight float64            // weight on the selecting configurations
	Classes      map[string]float64 // weight per excitation class
	Selected     bool               // this is |Phi> itself (before polishing)
}

// InteriorResult is the decaying state and how it was reached.
type InteriorResult struct {
	Discrete  Discrete
	Residual  float64   // ||(QHQ - E) Phi|| after the polish
	Ladder    []float64 // residual before and after each polish step
	Converged bool      // Residual <= Tol; otherwise the floor reached is reported
	JDIter    int
	Audit     []AuditEntry
	// Alternatives are the other candidates' converged states (energy, residual, target
	// weight). Ambiguous is set when one of them carries at least 80% of the selected
	// state's target weight: "the state with He2+ on I" is then not a sharp notion, and a
	// width that follows it (a lock-in lambda grid) can switch branch between runs.
	Alternatives []AuditEntry
	Ambiguous    bool
	// Branch is the rank of the returned state among the distinct candidates; GuessOverlap
	// is |<Phi|Guess>| / ||Guess|| when a guess was given.
	Branch       int
	GuessOverlap float64
}

// Operator is what SelectInterior needs of QHQ: a symmetric operator on the Q space.
type Operator interface {
	Size() int
	ApplyFull(out, in backend.Vector)
}

// ClassWeights is v's squared weight per excitation class of sp, named by hole and
// particle counts ("2h", "3h1p", ...). A row's class is its hole count (a doubly emptied
// orbital counts twice); its particles are the holes beyond the main class.
func ClassWeights(sp Space, v []float64) map[string]float64 {
	if sp.Size() == 0 || sp.MainBlockSize() == 0 {
		return nil
	}
	k := len(sp.Holes(0, nil))
	out := map[string]float64{}
	buf := make([]int, 0, 8)
	for r, x := range v {
		n := len(sp.Holes(r, buf[:0]))
		name := fmt.Sprintf("%dh", n)
		if n > k {
			name = fmt.Sprintf("%dh%dp", n, n-k)
		}
		out[name] += x * x
	}
	return out
}

// SelectInterior finds the eigenvector of op (QHQ over qsp) with the largest weight on
// o.Targets (or the o.Branch-th heaviest, or the state o.Guess leads to). diag is op's
// diagonal. It returns an error when the state cannot be followed (the screening budget
// runs out for every candidate, or the requested branch does not exist); a residual that
// stops above o.Tol is not an error — Converged is false and Residual/Ladder say where it
// stopped.
func SelectInterior(qsp Space, op Operator, diag []float64, be backend.Backend, o InteriorOptions) (InteriorResult, error) {
	n := qsp.Size()
	if op.Size() != n || len(diag) != n {
		return InteriorResult{}, fmt.Errorf("fano: SelectInterior: space %d, operator %d, diagonal %d",
			n, op.Size(), len(diag))
	}
	if len(o.Targets) == 0 {
		return InteriorResult{}, fmt.Errorf("fano: SelectInterior needs target configurations")
	}
	for _, t := range o.Targets {
		if t < 0 || t >= n {
			return InteriorResult{}, fmt.Errorf("fano: target row %d outside the Q space (%d)", t, n)
		}
	}
	if o.Tol <= 0 {
		o.Tol = 1e-10
	}
	if o.AuditWindow <= 0 {
		o.AuditWindow = 0.5 / (HartreeToMeV / 1000)
	}
	if o.ScreenTol <= 0 {
		o.ScreenTol = 1e-3
	}
	if o.HandOver <= 0 {
		o.HandOver = 1e-6
	}
	logf := func(format string, a ...any) {
		if o.Log != nil {
			o.Log(fmt.Sprintf(format, a...))
		}
	}
	yout := be.Alloc(n)
	defer be.Free(yout)
	apply := func(dst, src []float64) {
		v := be.Upload(src)
		op.ApplyFull(yout, v)
		be.Free(v)
		copy(dst, be.Download(yout))
	}

	// 1. start vectors: the guess, or the dense start
	var (
		startVecs [][]float64
		startIDs  []int
	)
	if o.Guess != nil {
		if len(o.Guess) != n {
			return InteriorResult{}, fmt.Errorf("fano: SelectInterior: the guess has %d rows, the Q space %d",
				len(o.Guess), n)
		}
		if o.Branch != 0 {
			return InteriorResult{}, fmt.Errorf("fano: SelectInterior: a guess selects its own state; branch %d is meaningless with it", o.Branch)
		}
		startVecs, startIDs = [][]float64{o.Guess}, []int{-1}
		logf("interior: starting from the given guess, followed by overlap")
	} else {
		var err error
		if startVecs, startIDs, err = denseStarts(qsp, n, apply, op, o, logf); err != nil {
			return InteriorResult{}, err
		}
	}

	jo := o.JD
	jo.Targets, jo.Diag, jo.Follow = o.Targets, diag, o.Guess
	if jo.Log == nil && o.Log != nil {
		jo.Log = o.Log
	}
	po := o.Polish
	po.Diag, po.Tol = diag, o.Tol
	if po.Log == nil && o.Log != nil {
		po.Log = o.Log
	}
	bail := po.BailRel
	if bail <= 0 {
		bail = 1e-2
	}

	// 2. screening: every candidate to ScreenTol
	type screened struct {
		start int
		jd    lanczos.JDResult
		err   error
	}
	var outs []screened
	so := jo
	so.Tol = math.Max(o.ScreenTol, o.Tol)
	for c, x0 := range startVecs {
		jd, err := lanczos.JacobiDavidson(apply, n, [][]float64{x0}, so)
		if err == nil {
			logf("interior: candidate %d screened at %.10f Eh, residual %.1e, target weight %.4f",
				startIDs[c], jd.Theta, jd.Residual, jd.TargetWeight)
		}
		outs = append(outs, screened{startIDs[c], jd, err})
	}
	// distinct converged candidates, heaviest first; Branch picks one
	var ranked []int
	for k, oc := range outs {
		if oc.err != nil {
			continue
		}
		dup := false
		for _, q := range ranked {
			dup = dup || math.Abs(outs[q].jd.Theta-oc.jd.Theta) < 1e-8 // the same state from another start
		}
		if !dup {
			ranked = append(ranked, k)
		}
	}
	if len(ranked) == 0 {
		return InteriorResult{JDIter: outs[0].jd.Iter}, outs[0].err
	}
	slices.SortStableFunc(ranked, func(a, b int) int {
		return cmpFloat(outs[b].jd.TargetWeight, outs[a].jd.TargetWeight)
	})
	if o.Branch < 0 || o.Branch >= len(ranked) {
		return InteriorResult{}, fmt.Errorf("fano: SelectInterior: branch %d requested, %d distinct candidates converged",
			o.Branch, len(ranked))
	}
	best := ranked[o.Branch]
	win := outs[best]
	if o.Branch > 0 {
		logf("interior: branch %d: candidate %d (target weight %.4f; the heaviest has %.4f)",
			o.Branch, win.start, win.jd.TargetWeight, outs[ranked[0]].jd.TargetWeight)
	}

	// 3. the selected candidate to the hand-over residual, from its vector and its neighbours
	to := jo
	to.Tol = math.Max(o.HandOver, o.Tol)
	x0s := [][]float64{win.jd.X}
	near := slices.Clone(win.jd.Ritz)
	slices.SortFunc(near, func(a, b lanczos.RitzPair) int {
		return cmpFloat(math.Abs(a.Theta-win.jd.Theta), math.Abs(b.Theta-win.jd.Theta))
	})
	for _, rp := range near {
		if len(x0s) == 6 {
			break
		}
		if math.Abs(rp.Theta-win.jd.Theta) > 1e-12 {
			x0s = append(x0s, rp.X)
		}
	}
	logf("interior: tightening candidate %d to residual %.0e (%d start vectors)", win.start, to.Tol, len(x0s))
	jd, err := lanczos.JacobiDavidson(apply, n, x0s, to)
	switch {
	case err == nil:
	case errors.Is(err, lanczos.ErrNotConverged):
		logf("interior: WARNING Jacobi-Davidson stopped at residual %.2e above the hand-over %.0e; "+
			"the polish is skipped and the residual is reported", jd.Residual, to.Tol)
	default:
		return InteriorResult{JDIter: win.jd.Iter + jd.Iter}, err
	}
	res := InteriorResult{JDIter: win.jd.Iter + jd.Iter, Branch: o.Branch}
	for _, rp := range jd.Ritz {
		if math.Abs(rp.Theta-jd.Theta) > o.AuditWindow {
			continue
		}
		res.Audit = append(res.Audit, AuditEntry{Energy: rp.Theta, Residual: rp.Residual,
			TargetWeight: rp.TargetWeight, Classes: ClassWeights(qsp, rp.X),
			Selected: math.Abs(rp.Theta-jd.Theta) < 1e-12})
	}

	// 4. polish, only from the hand-over residual, and only when MINRES can solve shifted
	// systems of this operator at all: JD's own correction solves say so
	theta, x, rn := jd.Theta, jd.X, jd.Residual
	res.Ladder = []float64{rn}
	switch {
	case err != nil || rn <= o.Tol:
	case jd.InnerRel > bail:
		logf("interior: polish skipped: Jacobi-Davidson's last correction solve stopped at rel %.1e "+
			"(bail-out %.0e), so inverse iteration's solves would not converge either", jd.InnerRel, bail)
	default:
		pol, perr := lanczos.PolishInverse(apply, jd.X, po)
		switch {
		case perr == nil, errors.Is(perr, lanczos.ErrNotConverged):
			res.Ladder = pol.Ladder
			if pol.Residual < rn {
				theta, x, rn = pol.Theta, pol.X, pol.Residual
			}
		default:
			// the polish left the state: JD's hand-over vector is the best estimate there is
			logf("interior: WARNING %v; keeping the Jacobi-Davidson vector", perr)
		}
	}
	res.Residual, res.Converged = rn, rn <= o.Tol
	if !res.Converged {
		logf("interior: WARNING the state stopped at residual %.2e (goal %.1e); reported, not assumed", rn, o.Tol)
	}
	var tw float64
	for _, t := range o.Targets {
		tw += x[t] * x[t]
	}
	res.Discrete = Discrete{Energy: theta, Vec: x, Weight: tw, Root: -1, Rows: slices.Clone(o.Targets)}
	logf("interior: candidate %d converged at %.10f Eh, residual %.1e, target weight %.4f", win.start, theta, rn, tw)
	if o.Guess != nil {
		var c, gg float64
		for i := range x {
			c += x[i] * o.Guess[i]
			gg += o.Guess[i] * o.Guess[i]
		}
		res.GuessOverlap = math.Abs(c) / math.Sqrt(gg)
		logf("interior: overlap with the guess %.6f", res.GuessOverlap)
	}

	for _, k := range ranked {
		if k == best {
			continue
		}
		oc := outs[k]
		res.Alternatives = append(res.Alternatives, AuditEntry{Energy: oc.jd.Theta, Residual: oc.jd.Residual,
			TargetWeight: oc.jd.TargetWeight, Classes: ClassWeights(qsp, oc.jd.X)})
		if oc.jd.TargetWeight >= 0.8*res.Discrete.Weight {
			res.Ambiguous = true
		}
	}
	if res.Ambiguous {
		logf("interior: WARNING the target character is split: another state carries >= 80%% of "+
			"the selected state's weight %.4f (see the alternatives)", res.Discrete.Weight)
	}
	return res, nil
}

// denseStarts diagonalizes the guess rows' block densely (one apply per unit vector, or
// op's Element) and returns up to o.Candidates eigenvectors (default 3) with at least half
// the largest target weight, heaviest first, and their indices in the block.
func denseStarts(qsp Space, n int, apply func(dst, src []float64), op Operator, o InteriorOptions,
	logf func(string, ...any)) ([][]float64, []int, error) {
	rows := o.GuessRows
	if rows == nil {
		for r := range qsp.MainBlockSize() {
			rows = append(rows, r)
		}
	}
	rows = append(slices.Clone(rows), o.Targets...)
	slices.Sort(rows)
	rows = slices.Compact(rows)
	g := len(rows)
	B := make([]float64, g*g) // B[a*g+b] = <rows[a]|op|rows[b]>
	if el, ok := op.(interface{ Element(r, c int) float64 }); ok {
		for a, r := range rows {
			for b, c := range rows {
				B[a*g+b] = el.Element(r, c)
			}
		}
	} else {
		e := make([]float64, n)
		col := make([]float64, n)
		for b, c := range rows {
			clear(e)
			e[c] = 1
			apply(col, e)
			for a, r := range rows {
				B[a*g+b] = col[r]
			}
		}
	}
	G := mat.NewSymDense(g, nil)
	for a := range g {
		for b := a; b < g; b++ {
			G.SetSym(a, b, 0.5*(B[a*g+b]+B[b*g+a]))
		}
	}
	var eig mat.EigenSym
	if !eig.Factorize(G, true) {
		return nil, nil, fmt.Errorf("fano: SelectInterior: dense start failed")
	}
	var U mat.Dense
	eig.VectorsTo(&U)
	w := eig.Values(nil)
	pos := map[int]int{}
	for a, r := range rows {
		pos[r] = a
	}
	cw := make([]float64, g)
	for j := range g {
		for _, t := range o.Targets {
			c := U.At(pos[t], j)
			cw[j] += c * c
		}
	}
	order := argsortDesc(cw)
	ncand := o.Candidates
	if ncand <= 0 {
		ncand = 3
	}
	var vecs [][]float64
	var ids []int
	for _, j := range order {
		if len(vecs) == ncand || cw[j] < 0.5*cw[order[0]] {
			break
		}
		x0 := make([]float64, n)
		for a, r := range rows {
			x0[r] = U.At(a, j)
		}
		logf("interior: dense start over %d rows: root %d at %.8f Eh, target weight %.3f", g, j, w[j], cw[j])
		vecs, ids = append(vecs, x0), append(ids, j)
	}
	return vecs, ids, nil
}

func cmpFloat(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// argsortDesc returns the indices of v ordered by value, largest first (stable).
func argsortDesc(v []float64) []int {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(a, b int) int {
		switch {
		case v[a] > v[b]:
			return -1
		case v[a] < v[b]:
			return 1
		}
		return 0
	})
	return idx
}

// TargetRows returns the main-class rows of sp whose holes, as a multiset of occupied
// spatial orbitals (a doubly emptied orbital twice), equal holes.
func TargetRows(sp Space, holes []int) []int {
	want := slices.Clone(holes)
	slices.Sort(want)
	var out []int
	buf := make([]int, 0, 8)
	for r := range sp.MainBlockSize() {
		h := sp.Holes(r, buf[:0])
		slices.Sort(h)
		if slices.Equal(h, want) {
			out = append(out, r)
		}
	}
	return out
}
