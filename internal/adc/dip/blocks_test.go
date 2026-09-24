package dip

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
)

// TestSecondOrderTablesMatchDirectSums pins the W/U tables against the per-element double sums
// they replace — blocks.go wTerm and uTerm, which are kept precisely as this oracle.
//
// W is a straight memoization and must agree BIT-EXACTLY: the table sweeps (r,s) in the same
// order the element builders did, so only the number of times it runs changed.
//
// U is the reassociated GEMM form (see buildSecondOrder). Its energy factor is rewritten via
// [x − ½(e_p+e_q)]/[(x−e_p)(x−e_q)] = ½[1/(x−e_p) + 1/(x−e_q)], which is an algebraic identity
// but not a bitwise one, so it is checked to a tolerance. Anything above rounding here means the
// factorization or the symmetry channel is wrong, not that floating point drifted.
func TestSecondOrderTablesMatchDirectSums(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		mx := New(sp, ints, eps, be)
		b := mx.blk.(interface {
			ensureSecondOrder()
			wAt(i, k int) float64
			uAt(i, j, k, l int) float64
			wTerm(i, k, s, r int) float64
			uTerm(i, j, k, l, s, r int, plus bool) float64
		})
		b.ensureSecondOrder()

		nocc, norb := sp.Nocc, sp.Norb
		plus := spin != Triplet

		// W: bit-exact.
		for i := range nocc {
			for k := range nocc {
				var want float64
				for r := nocc; r < norb; r++ {
					for s := nocc; s <= r; s++ {
						want += b.wTerm(i, k, s, r)
					}
				}
				if got := b.wAt(i, k); got != want {
					t.Fatalf("spin=%v sym=%d: W(%d,%d) = %v, direct sum = %v (must be bit-exact)",
						spin, sym, i, k, got, want)
				}
			}
		}

		// U: over the real 2h config pairs, using each element builder's own (r,s) guard.
		var maxRel float64
		main := sp.MainBlockSize()
		for row := range main {
			for col := 0; col <= row; col++ {
				rc, cc := sp.Configs[row], sp.Configs[col]
				i, j := rc.Occ[0], rc.Occ[1]
				k, l := cc.Occ[0], cc.Occ[1]

				// The guard the builders apply: σ_rs == 0 when either side is a |ii⟩ closed
				// shell (iiJJ/ijKK), σ_rs == σ_ij otherwise (ijKL).
				gate := symProduct(sp.irrep(i), sp.irrep(j))
				if row < sp.BeginIJ || col < sp.BeginIJ {
					gate = 0
				}
				var want float64
				for r := nocc; r < norb; r++ {
					for s := nocc; s <= r; s++ {
						if symProduct(sp.irrep(r), sp.irrep(s)) != gate {
							continue
						}
						want += b.uTerm(i, j, k, l, s, r, plus)
					}
				}
				got := b.uAt(i, j, k, l)
				rel := math.Abs(got-want) / math.Max(math.Abs(want), 1)
				if rel > maxRel {
					maxRel = rel
				}
				if rel > 1e-12 {
					t.Fatalf("spin=%v sym=%d: U(%d%d,%d%d) = %v, direct sum = %v (rel %.3e)",
						spin, sym, i, j, k, l, got, want, rel)
				}
			}
		}
		t.Logf("spin=%v sym=%d: max relative U deviation %.3e over %d pairs", spin, sym, maxRel, main*(main+1)/2)
	})
}
