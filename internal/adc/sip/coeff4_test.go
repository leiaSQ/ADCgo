package sip

import (
	"math"
	"testing"
)

// TestCoeff4Shapes guards the generated coefficient-table shapes against drift.
func TestCoeff4Shapes(t *testing.T) {
	if len(coeff0) != 13 || len(coeff0[0]) != 36 {
		t.Errorf("coeff0 shape [%d][%d], want [13][36]", len(coeff0), len(coeff0[0]))
	}
	if len(coeff1) != 3 || len(coeff1[0]) != 13 || len(coeff1[0][0]) != 30 {
		t.Errorf("coeff1 shape [%d][%d][%d], want [3][13][30]",
			len(coeff1), len(coeff1[0]), len(coeff1[0][0]))
	}
	if len(coeff2) != 13 || len(coeff2[0]) != 13 || len(coeff2[0][0]) != 52 {
		t.Errorf("coeff2 shape [%d][%d][%d], want [13][13][52]",
			len(coeff2), len(coeff2[0]), len(coeff2[0][0]))
	}
}

// TestCoeff4SpotValues checks entries hand-verified against the F77 DATA blocks
// (init0/1/2.F) so a bad regeneration is caught.
func TestCoeff4SpotValues(t *testing.T) {
	const eps = 1e-12
	s2, s3 := math.Sqrt(2), math.Sqrt(3)
	cases := []struct {
		name string
		got  float64
		want float64
	}{
		// coeff0[I-1][K-1] = SPIN{map(K)}(I) * VORFAK(I).
		{"coeff0[0][0]", coeff0[0][0], 2 * 0.5},            // SPIN1(1)=2, VORFAK(1)=1/2
		{"coeff0[5][0]", coeff0[5][0], 1 * (1 / s2)},       // SPIN1(6)=1, VORFAK(6)=1/√2
		{"coeff0[4][0]", coeff0[4][0], 6 * (1 / (3 * s2))}, // SPIN1(5)=6, VORFAK(5)=1/(3√2)=√2
		{"coeff0[1][2]", coeff0[1][2], 3 * (1 / (2 * s3))}, // SPIN3(2)=3, VORFAK(2)=1/(2√3)
		// coeff2[I-1][J-1][0] = SPIN1(IZ) * VORFAK(I)*VORFAK(J), IZ=(I-1)*13+J.
		{"coeff2[0][0][0]", coeff2[0][0][0], 4 * 0.5 * 0.5}, // SPIN1(1)=4
	}
	for _, c := range cases {
		if math.Abs(c.got-c.want) > eps {
			t.Errorf("%s = %.15g, want %.15g", c.name, c.got, c.want)
		}
	}
}
