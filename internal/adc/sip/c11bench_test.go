package sip

import (
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// BenchmarkC11_3sums times the order-3 main-block element, whose C^(A) term is the
// O(nvir^4 * nocc) five-deep loop that dominates sip.mainBlock(). H2O/DZP is far smaller than
// a production system (nvir 24 vs the production system's 154) and its ERI fits in L3, so this understates
// the gain from moving the l-invariant reads into compact tables — it is a lower bound.
func BenchmarkC11_3sums(b *testing.B) {
	fc := filepath.Join("..", "..", "..", "testdata", "h2o_dzp.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		b.Skipf("fcidump unavailable: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace(nocc, d.NORB, d.OrbSym, 0)
	el := newElements(sp, integrals.New(d, nocc, d.OrbSym), eps, 3)

	b.ReportAllocs()
	for b.Loop() {
		var acc float64
		for i := range nocc {
			for j := 0; j <= i; j++ {
				c, f1, f2 := el.c11_3sums(i, j)
				acc += c + f1 + f2
			}
		}
		_ = acc
	}
}
