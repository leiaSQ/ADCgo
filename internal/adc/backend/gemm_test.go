package backend

import (
	"math"
	"math/rand"
	"testing"
)

// colMajor builds a BlockView over freshly-allocated storage and fills it from a
// row-major reference, so tests can state matrices in the natural reading order.
// ldPad exercises a leading dimension strictly greater than Rows, which is what a
// column panel of a larger basis buffer looks like.
func colMajor(be Backend, rows, cols, ldPad int, rowMajor []float64) BlockView {
	ld := rows + ldPad
	buf := make([]float64, ld*cols)
	for i := range rows {
		for j := range cols {
			buf[j*ld+i] = rowMajor[i*cols+j]
		}
	}
	return BlockView{V: be.Upload(buf), Rows: rows, Cols: cols, Ld: ld}
}

// readBack returns c as a row-major slice.
func readBack(be Backend, c BlockView) []float64 {
	buf := be.Download(c.V)
	out := make([]float64, c.Rows*c.Cols)
	for i := range c.Rows {
		for j := range c.Cols {
			out[i*c.Cols+j] = buf[j*c.Ld+i]
		}
	}
	return out
}

// refGemm is the obvious triple loop over row-major operands, the oracle.
func refGemm(transA, transB bool, alpha float64, a []float64, ar, ac int,
	b []float64, bc int, beta float64, c []float64, cr, cc int) {
	at := func(i, j int) float64 {
		if transA {
			return a[j*ac+i]
		}
		return a[i*ac+j]
	}
	bt := func(i, j int) float64 {
		if transB {
			return b[j*bc+i]
		}
		return b[i*bc+j]
	}
	k := ac
	if transA {
		k = ar
	}
	for i := range cr {
		for j := range cc {
			var acc float64
			for p := range k {
				acc += at(i, p) * bt(p, j)
			}
			c[i*cc+j] = alpha*acc + beta*c[i*cc+j]
		}
	}
}

// TestGemmAgainstReference sweeps shapes, transpose flags, alpha/beta, and a padded
// leading dimension. Non-square, transposed shapes are the point: a Gemm that
// silently transposes an operand still passes on square symmetric inputs.
func TestGemmAgainstReference(t *testing.T) {
	be := Gonum{}
	rng := rand.New(rand.NewSource(7))
	fill := func(n int) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		return v
	}

	shapes := []struct{ m, n, k int }{
		{1, 1, 1}, {3, 4, 5}, {5, 3, 4}, {8, 8, 8}, {17, 5, 11}, {2, 9, 3},
	}
	for _, s := range shapes {
		for _, transA := range []bool{false, true} {
			for _, transB := range []bool{false, true} {
				for _, pad := range []int{0, 3} {
					// op(A) is m×k, op(B) is k×n, C is m×n.
					ar, ac := s.m, s.k
					if transA {
						ar, ac = s.k, s.m
					}
					br, bc := s.k, s.n
					if transB {
						br, bc = s.n, s.k
					}
					aRM, bRM, cRM := fill(ar*ac), fill(br*bc), fill(s.m*s.n)
					want := append([]float64(nil), cRM...)
					const alpha, beta = 0.75, -1.25
					refGemm(transA, transB, alpha, aRM, ar, ac, bRM, bc, beta, want, s.m, s.n)

					A := colMajor(be, ar, ac, pad, aRM)
					B := colMajor(be, br, bc, pad, bRM)
					C := colMajor(be, s.m, s.n, pad, cRM)
					be.Gemm(transA, transB, alpha, A, B, beta, C)
					got := readBack(be, C)

					var maxDiff float64
					for i := range want {
						maxDiff = math.Max(maxDiff, math.Abs(got[i]-want[i]))
					}
					if maxDiff > 1e-12 {
						t.Errorf("m=%d n=%d k=%d tA=%v tB=%v pad=%d: max diff %.3e\n got  %v\n want %v",
							s.m, s.n, s.k, transA, transB, pad, maxDiff, got, want)
					}
				}
			}
		}
	}
}

// TestGemmGramMatrix covers the exact shape block-Lanczos uses: P = Bᵀ·V, where B is
// a tall n×dim basis and V an n×b block. Bᵀ·B must come out symmetric positive
// semi-definite with the right trace.
func TestGemmGramMatrix(t *testing.T) {
	be := Gonum{}
	const n, b = 40, 6
	rng := rand.New(rand.NewSource(11))
	rm := make([]float64, n*b)
	for i := range rm {
		rm[i] = rng.NormFloat64()
	}
	V := colMajor(be, n, b, 0, rm)
	G := BlockView{V: be.Alloc(b * b), Rows: b, Cols: b, Ld: b}
	be.Gemm(true, false, 1, V, V, 0, G) // G = Vᵀ V

	g := readBack(be, G)
	var trace float64
	for i := range b {
		trace += g[i*b+i]
		for j := range b {
			if d := math.Abs(g[i*b+j] - g[j*b+i]); d > 1e-12 {
				t.Fatalf("Gram not symmetric at (%d,%d): %.3e", i, j, d)
			}
		}
	}
	// trace(VᵀV) == ‖V‖_F², computable straight from the source data.
	var frob float64
	for _, x := range rm {
		frob += x * x
	}
	if math.Abs(trace-frob) > 1e-10 {
		t.Errorf("trace(VᵀV)=%.6f, want ‖V‖_F²=%.6f", trace, frob)
	}
}
