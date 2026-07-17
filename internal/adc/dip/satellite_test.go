package dip

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestApplyBlockSatellite checks that the gated apply equals the full operator with the 2h
// main block and the 2h↔3h1p couplings zeroed — i.e. only the 3h1p↔3h1p satellite blocks
// act. This is the operator half of the lanczos.SolveLowMem Mode B Tarantelli gate.
func TestApplyBlockSatellite(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)
	be := backend.Gonum{}
	rng := rand.New(rand.NewSource(7))

	tested := 0
	for _, spin := range []Spin{Singlet, Triplet} {
		for sym := range 4 {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			n := sp.Size()
			main := sp.MainBlockSize()
			if n == 0 || n == main { // need a satellite space to test
				continue
			}
			mx := New(sp, ints, eps, be)

			// Dense reference: M with rows/cols in the main space zeroed out.
			M := mx.BuildMatrix()
			for i := range n {
				for j := range n {
					if i < main || j < main {
						M.Set(i, j, 0)
					}
				}
			}

			// Random n-vector; compare M_sat·x (dense) to ApplyBlockSatellite (1 column).
			x := make([]float64, n)
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			want := M.MulVec(x)

			in := backend.BlockView{V: be.Upload(x), Rows: n, Cols: 1, Ld: n}
			out := backend.BlockView{V: be.Alloc(n), Rows: n, Cols: 1, Ld: n}
			mx.ApplyBlockSatellite(out, in)
			got := be.Download(out.V)

			var maxDiff, scale float64
			for i := range n {
				scale = math.Max(scale, math.Abs(want[i]))
				maxDiff = math.Max(maxDiff, math.Abs(want[i]-got[i]))
			}
			// Also assert the main-space rows of the output are exactly zero (the gate must
			// not touch the main space).
			for i := range main {
				if got[i] != 0 {
					t.Errorf("spin=%v sym=%d: satellite apply wrote main row %d = %g (want 0)", spin, sym, i, got[i])
				}
			}
			if rel := maxDiff / math.Max(scale, 1e-300); rel > 1e-12 {
				t.Errorf("spin=%v sym=%d n=%d main=%d: satellite apply relative diff %.3e", spin, sym, n, main, rel)
			}
			tested++
		}
	}
	if tested == 0 {
		t.Fatal("no sectors with a satellite space exercised")
	}
	t.Logf("%d sectors: ApplyBlockSatellite == masked dense", tested)
}
