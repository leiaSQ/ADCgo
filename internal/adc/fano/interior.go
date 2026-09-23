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
//  2. lanczos.JacobiDavidson, which follows that weight with refined Ritz vectors and
//     thick restarts, down to the hand-over residual;
//  3. lanczos.PolishInverse, shifted inverse iteration to the goal, with an overlap guard
//     against landing on a neighbour.
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
	// Candidates is how many dense-start eigenvectors (those with at least half the
	// largest target weight, heaviest first) are each followed to convergence; the
	// converged state with the largest target weight wins (default 3). One is not
	// enough when charge resonance splits the target character nearly evenly: the
	// Jacobi-Davidson iteration follows the state it started near, which need not be
	// the heaviest one.
	Candidates int
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
// o.Targets. diag is op's diagonal. It returns an error when the state cannot be
// followed (the solver's budget runs out above the hand-over residual, or the polish
// swaps states); a polish that stops above o.Tol at the float64 floor is not an error —
// Converged is false and Residual/Ladder say where it stopped.
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

	// 1. dense start: the guess rows' block, one apply per unit vector
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
		return InteriorResult{}, fmt.Errorf("fano: SelectInterior: dense start failed")
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
	var starts []int
	for _, j := range order {
		if len(starts) == ncand || cw[j] < 0.5*cw[order[0]] {
			break
		}
		starts = append(starts, j)
	}

	jo := o.JD
	jo.Targets, jo.Diag = o.Targets, diag
	if jo.Log == nil && o.Log != nil {
		jo.Log = o.Log
	}
	if jo.Tol <= 0 {
		jo.Tol = math.Max(o.Tol, 1e-3)
	}
	po := o.Polish
	po.Diag, po.Tol = diag, o.Tol
	if po.Log == nil && o.Log != nil {
		po.Log = o.Log
	}
	type outcome struct {
		res InteriorResult
		err error
	}
	var outs []outcome
	for _, j := range starts {
		x0 := make([]float64, n)
		for a, r := range rows {
			x0[r] = U.At(a, j)
		}
		logf("interior: dense start over %d rows: root %d at %.8f Eh, target weight %.3f", g, j, w[j], cw[j])
		// 2. Jacobi-Davidson to the hand-over residual
		jd, err := lanczos.JacobiDavidson(apply, n, [][]float64{x0}, jo)
		if err != nil {
			outs = append(outs, outcome{InteriorResult{JDIter: jd.Iter}, err})
			continue
		}
		res := InteriorResult{JDIter: jd.Iter}
		for _, rp := range jd.Ritz {
			if math.Abs(rp.Theta-jd.Theta) > o.AuditWindow {
				continue
			}
			res.Audit = append(res.Audit, AuditEntry{Energy: rp.Theta, Residual: rp.Residual,
				TargetWeight: rp.TargetWeight, Classes: ClassWeights(qsp, rp.X),
				Selected: math.Abs(rp.Theta-jd.Theta) < 1e-12})
		}
		// 3. polish
		pol, err := lanczos.PolishInverse(apply, jd.X, po)
		res.Residual, res.Ladder = pol.Residual, pol.Ladder
		switch {
		case err == nil:
			res.Converged = true
		case errors.Is(err, lanczos.ErrNotConverged):
			logf("interior: WARNING the polish stopped at residual %.2e (goal %.1e); reported, not assumed",
				pol.Residual, o.Tol)
		default:
			outs = append(outs, outcome{res, err})
			continue
		}
		var tw float64
		for _, t := range o.Targets {
			tw += pol.X[t] * pol.X[t]
		}
		res.Discrete = Discrete{Energy: pol.Theta, Vec: pol.X, Weight: tw, Root: -1,
			Rows: slices.Clone(o.Targets)}
		logf("interior: candidate %d converged at %.10f Eh, residual %.1e, target weight %.4f",
			j, pol.Theta, pol.Residual, tw)
		outs = append(outs, outcome{res, nil})
	}
	best := -1
	for k, oc := range outs {
		if oc.err == nil && (best < 0 || oc.res.Discrete.Weight > outs[best].res.Discrete.Weight) {
			best = k
		}
	}
	if best < 0 {
		return outs[0].res, outs[0].err
	}
	res := outs[best].res
	for k, oc := range outs {
		if k == best || oc.err != nil {
			continue
		}
		d := oc.res.Discrete
		if math.Abs(d.Energy-res.Discrete.Energy) < 1e-8 {
			continue // the same state reached from another start
		}
		res.Alternatives = append(res.Alternatives, AuditEntry{Energy: d.Energy, Residual: oc.res.Residual,
			TargetWeight: d.Weight, Classes: ClassWeights(qsp, d.Vec)})
		if d.Weight >= 0.8*res.Discrete.Weight {
			res.Ambiguous = true
		}
	}
	if res.Ambiguous {
		logf("interior: WARNING the target character is split: another state carries >= 80%% of "+
			"the selected state's weight %.4f (see the alternatives)", res.Discrete.Weight)
	}
	return res, nil
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
