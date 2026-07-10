package lanczos

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// orthHarness builds an orthonormal basis B (n×d) and scratch buffers, then runs
// orthBlock on a caller-supplied candidate block and reports the two quantities the
// conditional second pass is supposed to protect: within-block orthonormality
// ‖qᵀq − I‖_max, and orthogonality to the basis ‖Bᵀq‖_max.
func orthHarness(t *testing.T, n, d int, vcols [][]float64) (rank int, orthErr, basisErr float64) {
	t.Helper()
	be := backend.Gonum{}
	rng := rand.New(rand.NewSource(17))

	// Orthonormal basis via Gram-Schmidt on random columns.
	bcols := make([][]float64, d)
	for j := range d {
		c := make([]float64, n)
		for i := range n {
			c[i] = rng.NormFloat64()
		}
		for k := range j {
			var dot float64
			for i := range n {
				dot += bcols[k][i] * c[i]
			}
			for i := range n {
				c[i] -= dot * bcols[k][i]
			}
		}
		var nrm float64
		for i := range n {
			nrm += c[i] * c[i]
		}
		nrm = math.Sqrt(nrm)
		for i := range n {
			c[i] /= nrm
		}
		bcols[j] = c
	}
	bdata := make([]float64, n*d)
	for j := range d {
		copy(bdata[j*n:(j+1)*n], bcols[j])
	}
	basis := backend.BlockView{V: be.Upload(bdata), Rows: n, Cols: d, Ld: n}

	nc := len(vcols)
	vdata := make([]float64, n*nc)
	for j, c := range vcols {
		copy(vdata[j*n:(j+1)*n], c)
	}
	v := backend.BlockView{V: be.Upload(vdata), Rows: n, Cols: nc, Ld: n}

	pbuf := be.Alloc(d * nc)
	gbuf := be.Alloc(nc * nc)
	rank, _ = orthBlock(be, basis, v, pbuf, d, gbuf, 1e-8)
	if rank == 0 {
		return 0, 0, 0
	}

	q := be.Download(v.V)
	for a := range rank {
		for b := range rank {
			var dot float64
			for i := range n {
				dot += q[a*n+i] * q[b*n+i]
			}
			want := 0.0
			if a == b {
				want = 1
			}
			orthErr = math.Max(orthErr, math.Abs(dot-want))
		}
		for j := range d {
			var dot float64
			for i := range n {
				dot += bcols[j][i] * q[a*n+i]
			}
			basisErr = math.Max(basisErr, math.Abs(dot))
		}
	}
	return rank, orthErr, basisErr
}

// TestOrthBlockWellConditioned: a healthy block keeps full rank and, after the single
// pass the criterion permits, is orthonormal and orthogonal to the basis to ~eps.
func TestOrthBlockWellConditioned(t *testing.T) {
	const n, d, b = 300, 40, 6
	rng := rand.New(rand.NewSource(23))
	vcols := make([][]float64, b)
	for j := range b {
		c := make([]float64, n)
		for i := range n {
			c[i] = rng.NormFloat64()
		}
		vcols[j] = c
	}
	rank, orthErr, basisErr := orthHarness(t, n, d, vcols)
	if rank != b {
		t.Fatalf("rank %d, want %d (nothing should deflate)", rank, b)
	}
	if orthErr > 1e-13 || basisErr > 1e-13 {
		t.Errorf("well-conditioned: ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", orthErr, basisErr)
	}
	t.Logf("well-conditioned: rank=%d ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", rank, orthErr, basisErr)
}

// illConditionedBlock builds a full-rank block with cond(v) ≈ 1/scale. The perturbation
// must stay well above the relative deflation floor (relCond2 = 1e-14 on λ, i.e. 1e-7 on
// singular values), or the offending direction is simply *deflated* and the survivors
// come out well conditioned — in which case the `rank < cols` branch, not the cond²
// branch, is what triggers the second pass.
func illConditionedBlock(n, b int, scale float64, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	cols := make([][]float64, b)
	for j := range b {
		c := make([]float64, n)
		for i := range n {
			c[i] = rng.NormFloat64()
		}
		cols[j] = c
	}
	for i := range n {
		cols[1][i] = cols[0][i] + scale*cols[1][i]
	}
	return cols
}

// TestOrthBlockIllConditionedStaysFullRank exercises the cond² branch: the block is
// ill conditioned (cond ≈ 1e5, so cond² ≈ 1e10 ≫ maxGramCond2) but nothing deflates.
// A single Gram-QR pass would lose orthogonality at ~cond²·eps ≈ 1e-6; the criterion
// must catch it on cond² alone and reorthogonalize.
func TestOrthBlockIllConditionedStaysFullRank(t *testing.T) {
	const n, d, b = 300, 40, 5
	vcols := illConditionedBlock(n, b, 1e-5, 29)

	rank, orthErr, basisErr := orthHarness(t, n, d, vcols)
	if rank != b {
		t.Fatalf("rank %d, want %d: this block should NOT deflate, so only cond² can trigger the second pass", rank, b)
	}
	if orthErr > 1e-12 {
		t.Errorf("‖qᵀq−I‖=%.3e — the second pass did not recover orthogonality", orthErr)
	}
	if basisErr > 1e-12 {
		t.Errorf("‖Bᵀq‖=%.3e — q is not orthogonal to the basis", basisErr)
	}
	t.Logf("cond²-triggered: rank=%d ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", rank, orthErr, basisErr)
}

// TestOrthBlockDeflates exercises the rank branch: a duplicated column and a column
// perturbed below the deflation floor must both be dropped, and the survivors must come
// out orthonormal.
func TestOrthBlockDeflates(t *testing.T) {
	const n, d, b = 300, 40, 6
	vcols := illConditionedBlock(n, b, 1e-9, 41) // 1e-9 is below the 1e-7 singular-value floor
	copy(vcols[3], vcols[2])                     // exact duplicate

	rank, orthErr, basisErr := orthHarness(t, n, d, vcols)
	if rank == 0 {
		t.Fatal("everything deflated")
	}
	if rank >= b {
		t.Errorf("rank %d of %d: the duplicate and the sub-floor direction should have deflated", rank, b)
	}
	if orthErr > 1e-12 || basisErr > 1e-12 {
		t.Errorf("after deflation: ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", orthErr, basisErr)
	}
	t.Logf("rank-triggered: rank=%d (of %d) ‖qᵀq−I‖=%.3e ‖Bᵀq‖=%.3e", rank, b, orthErr, basisErr)
}

// TestSinglePassLosesOrthogonality documents *why* the criterion exists. On a full-rank
// block with cond² ≫ maxGramCond2, one Gram-QR pass leaves an orthogonality error many
// orders of magnitude above the solver's 1e-10 gate. This is the failure the conditional
// must never skip: if it ever stops holding, maxGramCond2 can be revisited.
func TestSinglePassLosesOrthogonality(t *testing.T) {
	be := backend.Gonum{}
	const n, b = 300, 3
	vcols := illConditionedBlock(n, b, 1e-6, 31) // cond ≈ 1e6 → cond² ≈ 1e12
	vdata := make([]float64, n*b)
	for j, c := range vcols {
		copy(vdata[j*n:(j+1)*n], c)
	}
	v := backend.BlockView{V: be.Upload(vdata), Rows: n, Cols: b, Ld: n}
	gbuf := be.Alloc(b * b)

	rank, _, cond2 := blockOrth(be, v, gbuf, 1e-8) // ONE pass, deliberately
	if rank != b {
		t.Fatalf("rank %d: block deflated, so this does not test the cond² path", rank)
	}
	if cond2 <= maxGramCond2 {
		t.Fatalf("cond²=%.3e <= maxGramCond2=%.3e: the criterion would skip the second pass here", cond2, maxGramCond2)
	}

	q := be.Download(v.V)
	var orthErr float64
	for a := range rank {
		for c := range rank {
			var dot float64
			for i := range n {
				dot += q[a*n+i] * q[c*n+i]
			}
			want := 0.0
			if a == c {
				want = 1
			}
			orthErr = math.Max(orthErr, math.Abs(dot-want))
		}
	}
	if orthErr < 1e-11 {
		t.Errorf("single pass at cond²=%.3e gave ‖qᵀq−I‖=%.3e — unexpectedly accurate; "+
			"the second pass may be unnecessary and maxGramCond2 should be re-derived", cond2, orthErr)
	}
	t.Logf("single pass at cond²=%.3e: ‖qᵀq−I‖=%.3e (%.0fx the 1e-12 target) — second pass required",
		cond2, orthErr, orthErr/1e-12)
}
