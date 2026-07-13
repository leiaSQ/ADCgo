package sip

import (
	"math"
	"testing"
)

// WERT3 (the 5th-order 3h2p-diagonal correction, elements4.go wert3elem) is value-gated
// against theADCcode's own EIGAB diagonal in TestADC4EigabGate — ../ADC now dumps it to
// FT19F001.ADC before RSCRT1 truncates the diagonal tape. These remain as the
// theADCcode-free structural checks, which cover ground the value gate cannot: theADCcode
// only ever evaluates WERT3 on the diagonal, so the off-diagonal CI branch cascade has no
// reference at all and is pinned by Hermiticity alone.

// TestWert3DiagSpinSymmetric: for every 3h2p config the diagonal intermediate-spin block
// W[s][u] must be symmetric. If coeff2 columns were misaligned to the VINT slots, this
// almost certainly breaks.
func TestWert3DiagSpinSymmetric(t *testing.T) {
	mx := buildH2O4(t, 0, []int{0}) // A1: has the 3h2p space
	sp := mx.Space()
	if len(sp.Sat3) == 0 {
		t.Fatal("no 3h2p configs")
	}
	var maxAsym float64
	var checked, multi int
	for _, c := range sp.Sat3 {
		maxs := maxS3(c.L == c.M, c.I == c.J)
		if maxs > 1 {
			multi++
		}
		w := mx.el.wert3elem(c, c)
		for s := 0; s < maxs; s++ {
			for u := s + 1; u < maxs; u++ {
				if d := math.Abs(w[s][u] - w[u][s]); d > maxAsym {
					maxAsym = d
				}
				checked++
			}
		}
	}
	if multi == 0 {
		t.Fatal("no multi-spin configs — test is vacuous")
	}
	if maxAsym > 1e-12 {
		t.Fatalf("WERT3 diagonal spin block not symmetric: max |W[s][u]-W[u][s]| = %g", maxAsym)
	}
	t.Logf("checked %d spin pairs across %d multi-spin configs; max asymmetry %g", checked, multi, maxAsym)
}

// TestWert3MatrixHermitian: the (unused-in-production) off-diagonal 3h2p↔3h2p CI matrix
// must satisfy W_AB[s][u] == W_BA[u][s]. This exercises the full ported branch cascade
// (theADCcode only ever ran the diagonal), so it is a strong port-fidelity check.
func TestWert3MatrixHermitian(t *testing.T) {
	mx := buildH2O4(t, 0, []int{0})
	sp := mx.Space()
	n := len(sp.Sat3)
	step := max(n/60, 1)
	var maxAsym float64
	var checked int
	for a := 0; a < n; a += step {
		for b := 0; b < n; b += step {
			A, B := sp.Sat3[a], sp.Sat3[b]
			wab := mx.el.wert3elem(A, B)
			wba := mx.el.wert3elem(B, A)
			maxsA := maxS3(A.L == A.M, A.I == A.J)
			maxsB := maxS3(B.L == B.M, B.I == B.J)
			for s := 0; s < maxsA; s++ {
				for u := 0; u < maxsB; u++ {
					if d := math.Abs(wab[s][u] - wba[u][s]); d > maxAsym {
						maxAsym = d
					}
					checked++
				}
			}
		}
	}
	if maxAsym > 1e-12 {
		t.Fatalf("WERT3 CI matrix not Hermitian: max |W_AB[s][u]-W_BA[u][s]| = %g (checked %d)", maxAsym, checked)
	}
	t.Logf("checked %d bra/ket spin pairs; max asymmetry %g", checked, maxAsym)
}

// TestApplyEqualsBuild4Wert3 is the dense==Lanczos oracle with WERT3 enabled: ApplyFull
// (the iterative apply) must reproduce BuildMatrix column-by-column when the 3h2p diagonal
// carries the full EIGAB correction.
func TestApplyEqualsBuild4Wert3(t *testing.T) {
	mx := buildH2O4(t, 2, []int{0})
	mx.SetWert3(true)
	checkApplyEqualsBuild4(t, mx)
}
