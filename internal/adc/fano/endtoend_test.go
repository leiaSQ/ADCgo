package fano

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// endtoend_test.go — the Fano scheme A pipeline over a real ADC(2,2) matrix, as far as
// F3 goes: partition, restrict, solve QMQ, select |Phi>. It is here rather than in sip
// because it is the fano package's contract that is under test; sip does not import
// fano, so there is no cycle.

const hartreeToEV = 27.211386245988

// h2o22 builds a reduced H2O ADC(2,2) sector: the parent space, its matrix, and the
// integral machinery needed to build restricted matrices over the same orbitals.
func h2o22(t *testing.T, norb int, v sip.Variant) (*sip.Space, *sip.Matrix, func(*sip.Space) *sip.Matrix) {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.fcidump"))
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	build := func(sp *sip.Space) *sip.Matrix {
		mx := sip.New(sp, ints, eps, sip.Order22, backend.Gonum{})
		mx.SetVariant(v)
		return mx
	}
	sp := sip.NewSpace22(nocc, norb, nil, 0)
	return sp, build(sp), build
}

// TestSchemeAPipeline runs the F3 pipeline end to end and checks what it is supposed to
// produce: a Q subspace holding the initial vacancy, a P subspace holding decay
// channels, and a discrete state |Phi> whose energy is the decaying state's position.
//
// The physics check is the one that matters. E_Phi comes from QMQ, which has the
// Q-to-P couplings removed; those couplings are what produce the width, and a width is
// small (tens to hundreds of meV) compared with an ionization potential (tens of eV).
// So E_Phi must lie close to the full-matrix eigenvalue of the same character — close
// on the scale of the IP, not exactly equal. A partition that mixed the subspaces up
// would move it by electronvolts.
func TestSchemeAPipeline(t *testing.T) {
	parent, pmx, build := h2o22(t, 8, sip.VariantF)

	// Scheme A, "retains the initial vacancy": Q is every configuration still carrying
	// a hole in the deepest occupied orbital. That is the Auger-type criterion — a
	// configuration that has filled the vacancy has decayed.
	const vacancy = 0
	sel, err := NewHoleLocalization(AnyHole, []int{vacancy}, "initial vacancy")
	if err != nil {
		t.Fatal(err)
	}
	part := NewPartition(parent, sel)
	if err := part.Validate(); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s", part)

	qsp := parent.Restrict(part.Q)
	psp := parent.Restrict(part.P)
	if qsp.Size()+psp.Size() != parent.Size() {
		t.Fatalf("the two sub-spaces hold %d+%d rows, want %d",
			qsp.Size(), psp.Size(), parent.Size())
	}

	qres := lanczos.SolveDense(build(qsp), backend.Gonum{})
	phi, err := SelectDiscrete(qsp, qres, vacancy, 0, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("|Phi>: QMQ root %d, E_Phi = %.6f a.u. = %.4f eV, weight %.4f on 1h row(s) %v",
		phi.Root, phi.Energy, phi.Energy*hartreeToEV, phi.Weight, phi.Rows)

	if len(phi.Vec) != qsp.Size() {
		t.Errorf("|Phi> has %d components, want QSize() = %d", len(phi.Vec), qsp.Size())
	}
	var norm float64
	for _, c := range phi.Vec {
		norm += c * c
	}
	if math.Abs(norm-1) > 1e-10 {
		t.Errorf("|Phi> is not normalized: <Phi|Phi> = %.12g", norm)
	}

	// The full-matrix state of the same character, for comparison.
	full := lanczos.SolveDense(pmx, backend.Gonum{})
	vacRow := -1
	holes := make([]int, 0, 3)
	for r := range parent.MainBlockSize() {
		if h := parent.Holes(r, holes[:0]); len(h) == 1 && h[0] == vacancy {
			vacRow = r
		}
	}
	if vacRow < 0 {
		t.Fatal("the vacancy orbital has no 1h configuration in the parent space")
	}
	bestFull, bestW := -1, 0.0
	for j := range full.Values {
		c := full.FullVecs.At(vacRow, j)
		if w := c * c; w > bestW {
			bestFull, bestW = j, w
		}
	}
	dEV := math.Abs(phi.Energy-full.Values[bestFull]) * hartreeToEV
	t.Logf("full matrix: the state with the largest weight on the same 1h configuration is "+
		"root %d at %.4f eV (weight %.4f); E_Phi differs by %.4f eV",
		bestFull, full.Values[bestFull]*hartreeToEV, bestW, dEV)
	if dEV > 2.0 {
		t.Errorf("E_Phi is %.3f eV from the full-matrix state of the same character; "+
			"removing the Q-P coupling should shift it by far less than that", dEV)
	}
}

// TestCouplingIdentity gates the architectural claim this package is built on: that the
// Q-to-P coupling needs no projected operator and no new element evaluation, because
// applying the PARENT matrix to an embedded Q vector and reading off the P rows gives
//
//	(M Phi)_j = sum_{i in Q} M_ji Phi_i   for every j in P
//
// exactly. It compares that against the explicit cross-block product formed from the
// parent's dense matrix. If this did not hold, every width would be wrong.
//
// It also checks the consequence the plan singles out: under scheme A the subspaces are
// disjoint, so the -E_Phi delta term of g = P(M - E_Phi)Phi vanishes identically rather
// than approximately, and no overlap correction is needed anywhere.
func TestCouplingIdentity(t *testing.T) {
	parent, pmx, build := h2o22(t, 8, sip.VariantF)
	const vacancy = 0
	sel, err := NewHoleLocalization(AnyHole, []int{vacancy}, "")
	if err != nil {
		t.Fatal(err)
	}
	part := NewPartition(parent, sel)
	if err := part.Validate(); err != nil {
		t.Fatal(err)
	}
	qsp := parent.Restrict(part.Q)
	qres := lanczos.SolveDense(build(qsp), backend.Gonum{})
	phi, err := SelectDiscrete(qsp, qres, vacancy, 0, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	// The operator route: embed, apply the parent matrix, gather P.
	be := backend.Gonum{}
	n := parent.Size()
	emb := part.EmbedQ(phi.Vec, nil)
	xv, yv := be.Upload(emb), be.Upload(make([]float64, n))
	pmx.ApplyFull(yv, xv)
	gotFull := be.Download(yv)
	got := part.GatherP(gotFull, nil)

	// The explicit route: the dense cross block, row by row.
	M := pmx.BuildMatrix()
	want := make([]float64, part.PSize())
	for a, j := range part.P {
		var s float64
		for b, i := range part.Q {
			s += M.At(j, i) * phi.Vec[b]
		}
		// The -E_Phi delta term: zero because j is in P and Phi is zero there.
		s -= phi.Energy * emb[j]
		want[a] = s
	}

	var maxDiff, maxAbs float64
	for i := range want {
		maxAbs = math.Max(maxAbs, math.Abs(want[i]))
		maxDiff = math.Max(maxDiff, math.Abs(got[i]-want[i]))
	}
	var g2 float64
	for _, x := range want {
		g2 += x * x
	}
	t.Logf("coupling vector over %d P rows: max |g| = %.6g, ||g||^2 = %.6g "+
		"(the sum rule's 2 pi ||g||^2 = %.6g a.u. = %.4g meV)",
		len(want), maxAbs, g2, 2*math.Pi*g2, 2*math.Pi*g2*hartreeToEV*1000)
	t.Logf("operator route vs explicit cross block: max deviation %.3g", maxDiff)
	if maxDiff > 1e-12*math.Max(1, maxAbs) {
		t.Errorf("the operator route deviates from the explicit cross block by %.3g", maxDiff)
	}

	// Disjointness: the embedded vector really is zero on P.
	for _, j := range part.P {
		if emb[j] != 0 {
			t.Fatalf("the embedded |Phi> is nonzero on P row %d", j)
		}
	}
}

// TestSelectDiscreteErrors requires each way the selection can fail to be reported,
// rather than returning a wrong |Phi> that would silently produce a wrong rate.
func TestSelectDiscreteErrors(t *testing.T) {
	parent, _, build := h2o22(t, 8, sip.VariantF)
	sel, err := NewHoleLocalization(AnyHole, []int{0}, "")
	if err != nil {
		t.Fatal(err)
	}
	part := NewPartition(parent, sel)
	qsp := parent.Restrict(part.Q)
	res := lanczos.SolveDense(build(qsp), backend.Gonum{})

	if _, err := SelectDiscrete(qsp, res, 1, 0, 0.1); err == nil {
		t.Error("a vacancy orbital with no main-class configuration in Q was accepted")
	} else {
		t.Logf("vacancy not in Q: %v", err)
	}
	if _, err := SelectDiscrete(qsp, res, 0, 0, 1.01); err == nil {
		t.Error("an unreachable weight threshold was accepted")
	} else {
		t.Logf("threshold above 1: %v", err)
	}
	if _, err := SelectDiscrete(qsp, res, 0, 1<<20, 0.1); err == nil {
		t.Error("an out-of-range root index was accepted")
	}
	if _, err := SelectDiscrete(qsp, res, 0, -1, 0.1); err == nil {
		t.Error("a negative root index was accepted")
	}
	if _, err := SelectDiscrete(qsp, lanczos.Result{}, 0, 0, 0.1); err == nil {
		t.Error("a solve without full Ritz vectors was accepted")
	}
}
