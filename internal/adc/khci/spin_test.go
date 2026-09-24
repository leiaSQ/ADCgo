package khci

import (
	"math"
	"testing"
)

// TestMainSpinSquare: S² on the main class is symmetric, its spectrum is S(S+1), and the
// multiplicity counts are the spin-coupling counts of k holes in a closed shell.
func TestMainSpinSquare(t *testing.T) {
	for _, tc := range []struct {
		K, nocc, twoMs int
		want           map[int]int // 2S -> number of states
	}{
		// two holes in 3 orbitals, Ms = 0: 6 singlets (3 closed + 3 open pairs), 3 triplets
		{2, 3, 0, map[int]int{0: 6, 2: 3}},
		// Ms = 1: only the 3 triplets
		{2, 3, 2, map[int]int{0: 0, 2: 3}},
		// one hole: 3 doublets
		{1, 3, 1, map[int]int{1: 3}},
		// three holes in 3 orbitals, Ms = 1/2: 3·2 (one open orbital) + 2 (three open) doublets, 1 quartet
		{3, 3, 1, map[int]int{1: 8, 3: 1}},
		// four holes in 4 orbitals, Ms = 0: singlets 6 + 12 + 2 = 20, triplets 12 + 3 = 15, 1 quintet
		{4, 4, 0, map[int]int{0: 20, 2: 15, 4: 1}},
	} {
		sp, err := NewSpace(Options{K: tc.K, NOcc: tc.nocc, NVir: 2, TwoMs: tc.twoMs, MaxClass: tc.K})
		if err != nil {
			t.Fatal(err)
		}
		m, err := sp.MainSpinSquare()
		if err != nil {
			t.Fatal(err)
		}
		n := m.Rows
		for r := range n {
			for c := range n {
				if math.Abs(m.Data[r*n+c]-m.Data[c*n+r]) > 1e-14 {
					t.Fatalf("K=%d: S² not symmetric at (%d,%d)", tc.K, r, c)
				}
			}
		}
		total := 0
		for twoS, cnt := range tc.want {
			v, b, err := sp.MainSpinVectors(twoS)
			if err != nil {
				t.Fatal(err)
			}
			if b != cnt {
				t.Errorf("K=%d 2Ms=%d: %d states of 2S=%d, want %d", tc.K, tc.twoMs, b, twoS, cnt)
			}
			if len(v) != b*sp.Size() {
				t.Errorf("K=%d: panel holds %d values for %d columns", tc.K, len(v), b)
			}
			total += b
		}
		if total != n {
			t.Errorf("K=%d 2Ms=%d: multiplicities cover %d of %d main rows", tc.K, tc.twoMs, total, n)
		}
	}
}

// TestSpinProjector: over a space with particles, the projectors of every reachable
// multiplicity are idempotent, mutually orthogonal and sum to the identity, and S² acts
// on each range as S(S+1).
func TestSpinProjector(t *testing.T) {
	for _, tc := range []struct{ K, twoMs, maxS int }{{2, 0, 6}, {1, 1, 5}, {3, 1, 7}} {
		sp, err := NewSpace(Options{K: tc.K, NOcc: 3, NVir: 3, TwoMs: tc.twoMs, MaxClass: tc.K + 2})
		if err != nil {
			t.Fatal(err)
		}
		n := sp.Size()
		m, err := sp.spinSquare()
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float64, n)
		for i := range x {
			x[i] = math.Sin(float64(3*i + 1))
		}
		sum := make([]float64, n)
		var parts [][]float64
		for twoS := tc.twoMs; twoS <= tc.maxS; twoS += 2 {
			p, err := sp.SpinProjector(twoS)
			if err != nil {
				t.Fatal(err)
			}
			y := append([]float64(nil), x...)
			p.Apply(y, 1, n)
			z := append([]float64(nil), y...)
			p.Apply(z, 1, n)
			s2y := make([]float64, n)
			m.apply(s2y, y)
			for i := range y {
				if math.Abs(z[i]-y[i]) > 1e-10 {
					t.Fatalf("K=%d 2S=%d: P not idempotent at %d", tc.K, twoS, i)
				}
				if math.Abs(s2y[i]-ss(twoS)*y[i]) > 1e-9 {
					t.Fatalf("K=%d 2S=%d: S²P != S(S+1)P at %d", tc.K, twoS, i)
				}
				sum[i] += y[i]
			}
			parts = append(parts, y)
		}
		for i := range x {
			if math.Abs(sum[i]-x[i]) > 1e-9 {
				t.Fatalf("K=%d: Σ_S P_S x differs from x at %d by %g", tc.K, i, sum[i]-x[i])
			}
		}
		for a := range parts {
			for b := a + 1; b < len(parts); b++ {
				var d float64
				for i := range x {
					d += parts[a][i] * parts[b][i]
				}
				if math.Abs(d) > 1e-9 {
					t.Fatalf("K=%d: projections %d and %d overlap by %g", tc.K, a, b, d)
				}
			}
		}
	}
}
