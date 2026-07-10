package sip

import (
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// adc4_a1_gate_test.go — the A1-sector matched-integral gate. A1 contains the core
// hole, so its matrix has the 1h main block absent from B2, exercising the 1h
// couplings KOPP1/KOPP2 (1h↔2h1p), KOPP4 (1h↔3h2p) and the 1h diagonal. Reference
// tape: testdata/reference/adc4_a1_tape (see README). Complements TestADC4MatchedGate.
//
// Bit-exact (asserted): 2h1p/2h1p (WERT1), 2h1p↔3h2p (WERT2, multiset), 1h↔3h2p
// (KOPP4, multiset). Not yet bit-exact (pinned as documented bounds, will tighten):
//   - 1h↔2h1p carries a 4th-order piece KOPP3 (K2P2H+K1P3H+K3P1H) not yet ported;
//     kopp1+kopp2 reproduce it through 3rd order (residual ~6.7e-3).
//   - the 1h diagonal is −ε_core − Σ with Σ the external static self-energy
//     (&self-energy infinite, computed outside the ADC4 matrix); ADCgo uses −ε_core
//     only, leaving the fixed offset Σ(1,1) ≈ −0.0116 a.u.
func TestADC4MatchedGateA1(t *testing.T) {
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	dir := filepath.Join("..", "..", "..", "testdata", "reference", "adc4_a1_tape")
	if _, err := os.Stat(filepath.Join(dir, "FT21F001.ADC")); err != nil {
		t.Skipf("A1 reference tape unavailable: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace4(nocc, d.NORB, d.OrbSym, 0, []int{0}) // A1, core = orb 0
	if sp.Size() != 1712 || sp.BeginSat != 1 || sp.Begin3h2p-sp.BeginSat != 46 {
		t.Fatalf("A1 space = (size %d, 1h %d, 2h1p %d), want (1712, 1, 46)",
			sp.Size(), sp.BeginSat, sp.Begin3h2p-sp.BeginSat)
	}
	mx := New(sp, integrals.New(d, nocc, d.OrbSym), eps, 4, backend.Gonum{})
	M := mx.BuildMatrix()
	n := sp.Size()
	b1, b2 := sp.BeginSat, sp.Begin3h2p

	Ref := make([][]float64, n)
	for i := range Ref {
		Ref[i] = make([]float64, n)
	}
	rows, cols, vals := readTapeOff(t, filepath.Join(dir, "FT21F001.ADC"))
	for k := range rows {
		i, j := rows[k]-1, cols[k]-1
		Ref[i][j], Ref[j][i] = vals[k], vals[k]
	}
	for i, dv := range readTapeDiag(t, filepath.Join(dir, "FT18F001.ADC")) {
		Ref[i][i] = dv
	}

	// bit-exact blocks.
	if md := blockMaxDiff(Ref, M, b1, b2, b1, b2); md > 1e-12 {
		t.Errorf("2h1p/2h1p block maxdiff %g exceeds 1e-12", md)
	}
	if md := multisetMaxDiff(Ref, M, b1, b2, b2, n); md > 1e-12 {
		t.Errorf("2h1p↔3h2p multiset maxdiff %g exceeds 1e-12", md)
	}
	if md := multisetMaxDiff(Ref, M, 0, b1, b2, n); md > 1e-12 {
		t.Errorf("1h↔3h2p (KOPP4) multiset maxdiff %g exceeds 1e-12", md)
	}
	// 1h↔2h1p: kopp1+kopp2 (2nd+3rd) + kopp3 (4th, K2P2H+K1P3H+K3P1H) — bit-exact.
	if md := blockMaxDiff(Ref, M, 0, b1, b1, b2); md > 1e-12 {
		t.Errorf("1h↔2h1p (KOPP1/2/3) maxdiff %g exceeds 1e-12", md)
	}
	// 1h diagonal. Bare block is −ε_core (no static self-energy).
	p0 := sp.Configs[0].Occ[0]
	if md := math.Abs(M.At(0, 0) - (-eps[p0])); md > 1e-12 {
		t.Errorf("bare 1h diagonal %g != −ε_core %g", M.At(0, 0), -eps[p0])
	}
	// theADCcode folds in an external static self-energy Σ (&self-energy infinite):
	// diag = −ε_core − Σ. Verify the pluggable SetStaticSelfEnergy wiring reproduces the
	// tape diagonal given that Σ (the value itself is the self-energy module's output).
	sig := -eps[p0] - Ref[0][0] // ≈ −0.0116 a.u.
	mx.SetStaticSelfEnergy(func(i, j int) float64 {
		if i == p0 && j == p0 {
			return sig
		}
		return 0
	})
	if d := mx.BuildMatrix().At(0, 0) - Ref[0][0]; math.Abs(d) > 1e-12 {
		t.Errorf("with static Σ, 1h diagonal off by %g", d)
	}
	t.Logf("A1 gate: KOPP1/2/3 + KOPP4 bit-exact; external static Σ(core)=%.5f a.u. reproduces the 1h diagonal", sig)
}

func blockMaxDiff(Ref [][]float64, M backend.Mat, r0, r1, c0, c1 int) float64 {
	var md float64
	for i := r0; i < r1; i++ {
		for j := c0; j < c1; j++ {
			if dd := math.Abs(Ref[i][j] - M.At(i, j)); dd > md {
				md = dd
			}
		}
	}
	return md
}

func multisetMaxDiff(Ref [][]float64, M backend.Mat, r0, r1, c0, c1 int) float64 {
	var md float64
	for i := r0; i < r1; i++ {
		var rv, mv []float64
		for j := c0; j < c1; j++ {
			if Ref[i][j] != 0 {
				rv = append(rv, Ref[i][j])
			}
			if v := M.At(i, j); v != 0 {
				mv = append(mv, v)
			}
		}
		if len(rv) != len(mv) {
			return math.Inf(1)
		}
		sort.Float64s(rv)
		sort.Float64s(mv)
		for k := range rv {
			if dd := math.Abs(rv[k] - mv[k]); dd > md {
				md = dd
			}
		}
	}
	return md
}
