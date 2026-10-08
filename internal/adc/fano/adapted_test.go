package fano

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// particleBlocks groups the 2h1p rows of sp by particle orbital and diagonalizes each
// group's sub-block of the dense parent M, keeping only rows for which keep is true —
// the scheme B construction, written against the dense matrix so the operator can be
// checked against it.
func particleBlocks(sp *sip.Space, M backend.Mat, keep func(row int) bool) []AdaptedBlock {
	groups := map[int][]int{}
	for r := sp.BeginSat; r < len(sp.Configs); r++ {
		if keep(r) {
			p := sp.Configs[r].Vir
			groups[p] = append(groups[p], r)
		}
	}
	var ps []int
	for p := range groups {
		ps = append(ps, p)
	}
	slices.Sort(ps)
	var out []AdaptedBlock
	for _, p := range ps {
		rows := groups[p]
		sub := backend.NewMat(len(rows), len(rows))
		for i, ri := range rows {
			for j, rj := range rows {
				sub.Set(i, j, M.At(ri, rj))
			}
		}
		vals, vecs := backend.Gonum{}.SymEig(sub)
		out = append(out, AdaptedBlock{Rows: rows, U: vecs, Values: vals})
	}
	return out
}

func diagOf(M backend.Mat) []float64 {
	d := make([]float64, M.Rows)
	for i := range d {
		d[i] = M.At(i, i)
	}
	return d
}

func TestAdaptedRoundTripAndValidation(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	sym := backend.NewMat(4, 4)
	for i := range 4 {
		for j := 0; j <= i; j++ {
			v := rng.NormFloat64()
			sym.Set(i, j, v)
			sym.Set(j, i, v)
		}
	}
	vals, vecs := backend.Gonum{}.SymEig(sym)
	ad, err := NewAdapted(9, []AdaptedBlock{{Rows: []int{2, 3, 5, 7}, U: vecs, Values: vals}})
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float64, 9)
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	y, z := make([]float64, 9), make([]float64, 9)
	ad.ToConfig(x, y)
	ad.ToAdapted(y, z)
	for i := range x {
		if math.Abs(z[i]-x[i]) > 1e-13 {
			t.Fatalf("T^T T x differs from x at %d: %g vs %g", i, z[i], x[i])
		}
	}
	for _, i := range []int{0, 1, 4, 6, 8} {
		if y[i] != x[i] {
			t.Errorf("row %d outside every block was changed", i)
		}
	}
	// Overlapping blocks and a non-orthogonal U are refused.
	if _, err := NewAdapted(9, []AdaptedBlock{{Rows: []int{2, 3, 5, 7}, U: vecs, Values: vals},
		{Rows: []int{7}, U: backend.NewMat(1, 1), Values: []float64{0}}}); err == nil {
		t.Error("a row in two blocks was accepted")
	}
	bad := backend.NewMat(1, 1)
	bad.Set(0, 0, 2)
	if _, err := NewAdapted(9, []AdaptedBlock{{Rows: []int{1}, U: bad, Values: []float64{0}}}); err == nil {
		t.Error("a non-orthogonal block was accepted")
	}
}

// TestAdaptedOperatorIsTMT: the matrix-free T^T M T equals the dense product, its
// diagonal on each block is that block's eigenvalues, and the rotated blocks are
// diagonal — the defining property of the adapted states.
func TestAdaptedOperatorIsTMT(t *testing.T) {
	parent, pmx, _ := h2o22(t, 8, sip.VariantF)
	M := pmx.BuildMatrix()
	n := M.Rows
	blocks := particleBlocks(parent, M, func(int) bool { return true })
	ad, err := NewAdapted(n, blocks)
	if err != nil {
		t.Fatal(err)
	}
	op, err := NewAdaptedOperator(pmx, backend.Gonum{}, ad, diagOf(M))
	if err != nil {
		t.Fatal(err)
	}
	sub := NewSubOperator(op, backend.Gonum{}, seq(n), parent.MainBlockSize(), op.DiagonalHost())
	A := sub.BuildMatrix()
	// Dense T^T M T, column by column.
	var maxDiff float64
	e, y, tm := make([]float64, n), make([]float64, n), make([]float64, n)
	for j := range n {
		clear(e)
		e[j] = 1
		ad.ToConfig(e, y)
		mv := M.MulVec(y)
		ad.ToAdapted(mv, tm)
		for i := range n {
			maxDiff = math.Max(maxDiff, math.Abs(A.At(i, j)-tm[i]))
		}
	}
	if maxDiff > 1e-12 {
		t.Errorf("matrix-free T^T M T deviates from the dense product by %.3g", maxDiff)
	}
	for _, bl := range blocks {
		for a, ra := range bl.Rows {
			if math.Abs(A.At(ra, ra)-bl.Values[a]) > 1e-12 || math.Abs(op.DiagonalHost()[ra]-bl.Values[a]) > 1e-12 {
				t.Fatalf("diagonal at row %d is %g, block eigenvalue %g", ra, A.At(ra, ra), bl.Values[a])
			}
			for b, rb := range bl.Rows {
				if a != b && math.Abs(A.At(ra, rb)) > 1e-12 {
					t.Fatalf("rotated block is not diagonal: (%d,%d) = %g", ra, rb, A.At(ra, rb))
				}
			}
		}
	}
	t.Logf("%d rows, %d particle blocks; max |T^T M T - dense| = %.3g", n, len(blocks), maxDiff)
}

func seq(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

// classesFrom turns a scheme A partition into per-row classes.
func classesFrom(part *Partition, n int) []Class {
	c := make([]Class, n)
	for r := range n {
		switch {
		case part.QIndex(r) >= 0:
			c[r] = ClassQ
		case part.PIndex(r) >= 0:
			c[r] = ClassP
		default:
			c[r] = ClassX
		}
	}
	return c
}

// TestSchemeBIdentityReproducesSchemeA: with every block's U the identity, the adapted
// basis is the configuration basis, so the same classes must give scheme A's QMQ, PMP and
// coupling to rounding.
func TestSchemeBIdentityReproducesSchemeA(t *testing.T) {
	parent, pmx, build := h2o22(t, 8, sip.VariantF)
	M := pmx.BuildMatrix()
	n := M.Rows
	sel, err := ParseClassRule("q:0:1;x:2/2-4:2", "")
	if err != nil {
		t.Fatal(err)
	}
	partA := NewPartition(parent, sel)

	var blocks []AdaptedBlock
	for _, bl := range particleBlocks(parent, M, func(int) bool { return true }) {
		id := backend.NewMat(len(bl.Rows), len(bl.Rows))
		vals := make([]float64, len(bl.Rows))
		for i, r := range bl.Rows {
			id.Set(i, i, 1)
			vals[i] = M.At(r, r)
		}
		blocks = append(blocks, AdaptedBlock{Rows: bl.Rows, U: id, Values: vals})
	}
	ad, err := NewAdapted(n, blocks)
	if err != nil {
		t.Fatal(err)
	}
	op, err := NewAdaptedOperator(pmx, backend.Gonum{}, ad, diagOf(M))
	if err != nil {
		t.Fatal(err)
	}
	holes := func(r int) int { return len(parent.Holes(r, nil)) }
	partB := NewPartitionFromClasses(classesFrom(partA, n), parent.MainBlockSize(), holes, "identity")
	if !slices.Equal(partA.Q, partB.Q) || !slices.Equal(partA.P, partB.P) || partA.XSize() != partB.XSize() {
		t.Fatal("per-row classes did not reproduce the scheme A partition")
	}

	// QMQ: scheme B's restricted operator against scheme A's restricted ADC matrix.
	qB := NewSubOperator(op, backend.Gonum{}, partB.Q, partB.QMain, op.DiagonalHost()).BuildMatrix()
	qA := build(parent.Restrict(partA.Q)).BuildMatrix()
	if d := maxAbsDiff(qA, qB); d > 1e-12 {
		t.Errorf("QMQ differs from scheme A by %.3g", d)
	}
	qsp := parent.Restrict(partA.Q)
	phi, err := SelectDiscrete(qsp, lanczos.SolveDense(build(qsp), backend.Gonum{}), 0, 0, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	gA, err := Coupling(pmx, partA, phi, backend.Gonum{})
	if err != nil {
		t.Fatal(err)
	}
	gB, err := Coupling(op, partB, phi, backend.Gonum{})
	if err != nil {
		t.Fatal(err)
	}
	var d float64
	for i := range gA {
		d = math.Max(d, math.Abs(gA[i]-gB[i]))
	}
	if d > 1e-12 {
		t.Errorf("coupling differs from scheme A by %.3g", d)
	}
}

// TestRotationInsidePKeepsTheWidth: rotating only rows that are all in P leaves P as a
// subspace unchanged, so PMP's spectrum and every gamma_i = 2 pi <g|chi_i>^2 — the whole
// input of Stieltjes imaging — must be invariant. This is the property that makes scheme B
// a partition and not a different Hamiltonian.
func TestRotationInsidePKeepsTheWidth(t *testing.T) {
	parent, pmx, build := h2o22(t, 8, sip.VariantF)
	M := pmx.BuildMatrix()
	n := M.Rows
	sel, err := ParseClassRule("q:0:1", "")
	if err != nil {
		t.Fatal(err)
	}
	partA := NewPartition(parent, sel)
	inP := func(r int) bool { return partA.PIndex(r) >= 0 }
	blocks := particleBlocks(parent, M, inP)
	ad, err := NewAdapted(n, blocks)
	if err != nil {
		t.Fatal(err)
	}
	op, err := NewAdaptedOperator(pmx, backend.Gonum{}, ad, diagOf(M))
	if err != nil {
		t.Fatal(err)
	}
	holes := func(r int) int { return len(parent.Holes(r, nil)) }
	partB := NewPartitionFromClasses(classesFrom(partA, n), parent.MainBlockSize(), holes, "rotated P")

	qsp := parent.Restrict(partA.Q)
	phi, err := SelectDiscrete(qsp, lanczos.SolveDense(build(qsp), backend.Gonum{}), 0, 0, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	gA, _ := Coupling(pmx, partA, phi, backend.Gonum{})
	gB, _ := Coupling(op, partB, phi, backend.Gonum{})

	pA := lanczos.SolveDense(build(parent.Restrict(partA.P)), backend.Gonum{})
	pB := lanczos.SolveDense(denseOp{NewSubOperator(op, backend.Gonum{}, partB.P, 0, op.DiagonalHost())}, backend.Gonum{})
	if len(pA.Values) != len(pB.Values) {
		t.Fatalf("PMP sizes %d vs %d", len(pA.Values), len(pB.Values))
	}
	gamma := func(res lanczos.Result, g []float64) []float64 {
		out := make([]float64, len(res.Values))
		for i := range res.Values {
			var s float64
			for r := range g {
				s += g[r] * res.FullVecs.At(r, i)
			}
			out[i] = 2 * math.Pi * s * s
		}
		return out
	}
	gaA, gaB := gamma(pA, gA), gamma(pB, gB)
	var dE, dG, gmax float64
	for i := range pA.Values {
		dE = math.Max(dE, math.Abs(pA.Values[i]-pB.Values[i]))
		dG = math.Max(dG, math.Abs(gaA[i]-gaB[i]))
		gmax = math.Max(gmax, gaA[i])
	}
	t.Logf("%d P rows in %d rotated blocks: max |dE| %.3g, max |d gamma| %.3g (max gamma %.3g)",
		len(partA.P), len(blocks), dE, dG, gmax)
	if dE > 1e-10 || dG > 1e-10*math.Max(1, gmax) {
		t.Errorf("rotating inside P changed the pseudo-continuum: dE %.3g, d gamma %.3g", dE, dG)
	}
}

// denseOp adapts a SubOperator to lanczos.DenseOperator for SolveDense.
type denseOp struct{ *SubOperator }

func maxAbsDiff(a, b backend.Mat) float64 {
	var d float64
	for i := range a.Rows {
		for j := range a.Cols {
			d = math.Max(d, math.Abs(a.At(i, j)-b.At(i, j)))
		}
	}
	return d
}
