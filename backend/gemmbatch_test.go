package backend

import (
	"math"
	"math/rand"
	"testing"
)

// TestGemmBatchAgainstReference is the gate for the GemmBatch capability: the batched entry
// point must agree with the same triple-loop oracle TestGemmAgainstReference uses, for every
// transpose combination, for beta = 0 (overwrite) and beta != 0 (accumulate), and for a leading
// dimension strictly greater than Rows — which is what a column panel of a larger buffer looks
// like, and the case a naive batched implementation gets wrong.
func TestGemmBatchAgainstReference(t *testing.T) {
	be := Gonum{}
	rng := rand.New(rand.NewSource(20260826))
	fill := func(n int) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		return v
	}

	shapes := []struct{ m, n, k int }{
		{1, 1, 1}, {3, 4, 5}, {5, 3, 2}, {8, 8, 8}, {2, 7, 3},
	}
	const batch = 4
	const alpha = 0.75

	for _, s := range shapes {
		for _, transA := range []bool{false, true} {
			for _, transB := range []bool{false, true} {
				for _, pad := range []int{0, 3} {
					for _, beta := range []float64{0, 1, -0.5} {
						ar, ac := s.m, s.k
						if transA {
							ar, ac = s.k, s.m
						}
						br, bc := s.k, s.n
						if transB {
							br, bc = s.n, s.k
						}

						av := make([]BlockView, batch)
						bv := make([]BlockView, batch)
						cv := make([]BlockView, batch)
						want := make([][]float64, batch)
						for i := range batch {
							a, b, c := fill(ar*ac), fill(br*bc), fill(s.m*s.n)
							av[i] = colMajor(be, ar, ac, pad, a)
							bv[i] = colMajor(be, br, bc, pad, b)
							cv[i] = colMajor(be, s.m, s.n, pad, c)

							want[i] = make([]float64, len(c))
							copy(want[i], c)
							refGemm(transA, transB, alpha, a, ar, ac, b, bc, beta, want[i], s.m, s.n)
						}

						GemmBatch(be, transA, transB, alpha, av, bv, beta, cv)

						for i := range batch {
							got := readBack(be, cv[i])
							for idx := range want[i] {
								if math.Abs(got[idx]-want[i][idx]) > 1e-12 {
									t.Fatalf("m=%d n=%d k=%d transA=%v transB=%v pad=%d beta=%g member %d elem %d: got %g want %g",
										s.m, s.n, s.k, transA, transB, pad, beta, i, idx, got[idx], want[i][idx])
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestGemmBatchMatchesLoopedGemm states the property tensorop depends on directly: batching
// changes nothing but the number of calls. It is the ADCgo-side half of MCTDHgo's D4 gate.
func TestGemmBatchMatchesLoopedGemm(t *testing.T) {
	be := Gonum{}
	rng := rand.New(rand.NewSource(11))
	fill := func(n int) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		return v
	}

	const batch, m, n, k = 5, 6, 4, 3
	av := make([]BlockView, batch)
	bv := make([]BlockView, batch)
	batched := make([]BlockView, batch)
	looped := make([]BlockView, batch)
	for i := range batch {
		a, b, c := fill(m*k), fill(k*n), fill(m*n)
		av[i] = colMajor(be, m, k, 2, a)
		bv[i] = colMajor(be, k, n, 2, b)
		batched[i] = colMajor(be, m, n, 2, c)
		looped[i] = colMajor(be, m, n, 2, c)
	}

	GemmBatch(be, false, false, 1, av, bv, 1, batched)
	for i := range batch {
		be.Gemm(false, false, 1, av[i], bv[i], 1, looped[i])
	}

	for i := range batch {
		got, want := readBack(be, batched[i]), readBack(be, looped[i])
		for idx := range want {
			if got[idx] != want[idx] {
				t.Fatalf("member %d elem %d: batched %g != looped %g", i, idx, got[idx], want[idx])
			}
		}
	}
}

// TestGemmBatchLengthMismatch pins the panic contract: a caller whose operand slices drift out
// of step is told, rather than silently getting the shortest prefix computed.
func TestGemmBatchLengthMismatch(t *testing.T) {
	be := Gonum{}
	v := colMajor(be, 2, 2, 0, []float64{1, 2, 3, 4})
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic on mismatched operand slice lengths")
		}
	}()
	GemmBatch(be, false, false, 1, []BlockView{v, v}, []BlockView{v}, 0, []BlockView{v, v})
}
