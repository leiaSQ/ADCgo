package fano

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
)

// TestSelectInteriorHe2Plus: on the labelled He3 fixture (1.30 A chain,
// localized orbitals), the Q space of the net-charge partition ("every atom +1 and one free
// electron" is P) holds He2+(He1) He He as an INTERIOR root: the He+He+ charge-transfer
// states lie below it. SelectInterior, targeting the configuration with He1's orbital
// doubly emptied, must reproduce the dense QMQ eigenpair with the largest weight there:
// energy to 1e-10, the vector to an overlap of 1 - 1e-10, the residual at or below the
// goal, and an audit that lists it.
func TestSelectInteriorHe2Plus(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/khci/he3_ghost.fcidump")
	if err != nil {
		t.Fatal(err)
	}
	md, err := mo.ReadFile("../../../testdata/khci/he3_ghost.mo.json")
	if err != nil {
		t.Fatal(err)
	}
	const nocc = 3
	kinds := make([]khci.VirKind, d.NORB-nocc)
	for a := range kinds {
		if md.OrbKind[nocc+a] == mo.OrbFree {
			kinds[a] = khci.Free
		}
	}
	sp, err := khci.NewSpace(khci.Options{K: 2, NOcc: nocc, NVir: d.NORB - nocc, TwoMs: 0,
		VirKind: kinds, LimitFree: true, MaxFreeTop: 1})
	if err != nil {
		t.Fatal(err)
	}
	ci, err := khci.NewCI(sp, d, backend.Gonum{})
	if err != nil {
		t.Fatal(err)
	}
	am, err := AtomMapFromMO(md, nocc)
	if err != nil {
		t.Fatal(err)
	}
	rule, err := ParseConfigRule("p: charge He1=1,He2=1,He3=1 & free=1", "He2+(He1)", am)
	if err != nil {
		t.Fatal(err)
	}
	part := NewPartition(sp, rule)
	if err := part.Validate(); err != nil {
		t.Fatal(err)
	}
	qop, err := ci.Restrict(part.Q)
	if err != nil {
		t.Fatal(err)
	}
	qsp := qop.Space()
	he1 := am.OccAtom
	var orb int
	for i, a := range he1 {
		if am.Names[a] == "He1" {
			orb = i
		}
	}
	targets := TargetRows(qsp, []int{orb, orb})
	if len(targets) != 1 {
		t.Fatalf("expected one He1^-2 configuration in Q, got %v", targets)
	}

	// the reference: dense QMQ, the eigenvector heaviest on the target
	w, V := backend.Gonum{}.SymEig(qop.BuildMatrix())
	n := qsp.Size()
	best, bw, below := 0, -1.0, 0
	for j := range n {
		c := V.At(targets[0], j)
		if c*c > bw {
			best, bw = j, c*c
		}
	}
	for j := range n {
		if w[j] < w[best] {
			below++
		}
	}
	if below == 0 {
		t.Fatal("the He2+ state is the lowest Q root: the test would not exercise an interior target")
	}

	var trace []string
	res, err := SelectInterior(qsp, qop, qop.DiagonalHost(), backend.Gonum{}, InteriorOptions{
		Targets: targets, Tol: 1e-10,
		Log: func(s string) { trace = append(trace, s) },
	})
	if err != nil {
		for _, s := range trace {
			t.Log(s)
		}
		t.Fatal(err)
	}
	var ov float64
	for i := range n {
		ov += V.At(i, best) * res.Discrete.Vec[i]
	}
	t.Logf("Q %d rows, He2+(He1) is root %d (%d below it) at %.10f Eh, target weight %.3f; "+
		"interior: %.10f Eh, residual %.1e (ladder %.1e), JD %d iterations, |overlap| %.12f, audit %d entries",
		n, best, below, w[best], bw, res.Discrete.Energy, res.Residual, res.Ladder, res.JDIter, math.Abs(ov), len(res.Audit))
	if math.Abs(res.Discrete.Energy-w[best]) > 1e-10 || 1-math.Abs(ov) > 1e-10 {
		t.Errorf("interior state %.12f (overlap %.12f), dense %.12f", res.Discrete.Energy, ov, w[best])
	}
	if !res.Converged || res.Residual > 1e-10 {
		t.Errorf("residual %.2e, converged %v", res.Residual, res.Converged)
	}
	if len(res.Audit) == 0 {
		t.Error("empty C4 audit: the selected state itself should be listed")
	}
}
