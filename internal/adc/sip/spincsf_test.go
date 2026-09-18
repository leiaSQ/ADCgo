package sip

import (
	"math"
	"testing"
)

// s2Apply applies S^2 to a spin string of n open shells, returning the resulting
// vector over the determinant basis dets. For n spin-1/2 particles,
//
//	S^2 = 3n/4 + sum_{i<j} ( 2 m_i m_j + (S+_i S-_j + S-_i S+_j) )
//
// so the diagonal is 3n/4 + sum_{i<j} 2 m_i m_j and the off-diagonal flips each
// unlike pair (i,j) with coefficient 1. This is the independent oracle for the
// generator: it is written from the operator, not from the branching diagram.
func s2Apply(dets [][]int, d int) map[string]float64 {
	det := dets[d]
	n := len(det)
	m := func(sp int) float64 {
		if sp == spinUp {
			return 0.5
		}
		return -0.5
	}
	out := map[string]float64{}
	diag := 0.75 * float64(n)
	for i := range n {
		for j := i + 1; j < n; j++ {
			diag += 2 * m(det[i]) * m(det[j])
		}
	}
	out[detKey(det)] += diag
	for i := range n {
		for j := i + 1; j < n; j++ {
			if det[i] == det[j] {
				continue
			}
			sw := append([]int(nil), det...)
			sw[i], sw[j] = sw[j], sw[i]
			out[detKey(sw)]++
		}
	}
	return out
}

func detKey(det []int) string {
	b := make([]byte, len(det))
	for i, s := range det {
		b[i] = byte('0' + s)
	}
	return string(b)
}

// TestCSFCountsMatchMaxS3 pins the number of doublet spin functions against the
// multiplicities the ADC(4) 3h2p enumeration already uses (maxS3), and against the
// 2h1p counts in config.go: 1 open shell -> 1 function, 3 -> 2, 5 -> 5.
func TestCSFCountsMatchMaxS3(t *testing.T) {
	for _, tc := range []struct{ open, want int }{{1, 1}, {3, 2}, {5, 5}} {
		if got := doubletCSF(tc.open).NumFuncs(); got != tc.want {
			t.Errorf("open=%d: %d spin functions, want %d", tc.open, got, tc.want)
		}
	}
	// The 3h2p multiplicities of maxS3 are the open-shell counts of the four
	// coincidence cases: L==M empties that orbital, I==J doubly occupies it.
	cases := []struct{ lEqM, iEqJ bool }{{true, true}, {true, false}, {false, true}, {false, false}}
	for _, c := range cases {
		open := 5
		if c.lEqM {
			open -= 2
		}
		if c.iEqJ {
			open -= 2
		}
		if got, want := doubletCSF(open).NumFuncs(), maxS3(c.lEqM, c.iEqJ); got != want {
			t.Errorf("lEqM=%v iEqJ=%v: csf gives %d, maxS3 gives %d", c.lEqM, c.iEqJ, got, want)
		}
	}
}

// TestCSFOrthonormal checks the coefficient rows are an orthonormal basis.
func TestCSFOrthonormal(t *testing.T) {
	for _, n := range []int{1, 3, 5, 7} {
		b := doubletCSF(n)
		for p := range b.Coef {
			for q := range b.Coef {
				var dot float64
				for d := range b.Dets {
					dot += b.Coef[p][d] * b.Coef[q][d]
				}
				want := 0.0
				if p == q {
					want = 1.0
				}
				if math.Abs(dot-want) > 1e-12 {
					t.Errorf("n=%d <%d|%d> = %.15g, want %g", n, p, q, dot, want)
				}
			}
		}
	}
}

// TestCSFAreSpinEigenfunctions applies the S^2 operator, built independently in
// s2Apply, to every generated spin function and requires the eigenvalue 3/4.
func TestCSFAreSpinEigenfunctions(t *testing.T) {
	for _, n := range []int{1, 3, 5, 7} {
		b := doubletCSF(n)
		idx := map[string]int{}
		for d, det := range b.Dets {
			idx[detKey(det)] = d
		}
		for p, row := range b.Coef {
			out := make([]float64, len(b.Dets))
			for d, c := range row {
				if c == 0 {
					continue
				}
				for k, v := range s2Apply(b.Dets, d) {
					out[idx[k]] += c * v
				}
			}
			for d := range out {
				if math.Abs(out[d]-0.75*row[d]) > 1e-12 {
					t.Fatalf("n=%d func %d: S^2 not diagonal at det %v: got %.15g want %.15g",
						n, p, b.Dets[d], out[d], 0.75*row[d])
				}
			}
		}
	}
}
