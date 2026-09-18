package fano

import (
	"fmt"
	"math"
	"sort"

	"github.com/leiaSQ/ADCgo/backend"
)

// coupling.go — the coupling vector g = P(M - E_Phi)|Phi>, and the start block the
// pseudo-continuum solve is seeded with.

// Applier is the PARENT ADC matrix, applied to a backend-resident vector.
// *sip.Matrix and *dip.Matrix satisfy it. Only the parent is needed: the coupling
// lives in the Q-to-P cross block, which is in neither sub-block.
type Applier interface {
	ApplyFull(out, in backend.Vector)
	Size() int
}

// Coupling returns g = P(M - E_Phi)|Phi>, in P-subspace coordinates.
//
// The whole computation is one parent mat-vec and a gather, because embedding |Phi>
// into the parent index space makes it zero off Q, so for every parent row j
//
//	(M Phi_embedded)_j = sum_{i in Q} M_ji Phi_i,
//
// which on the P rows is exactly the cross-block product. No projected operator, no
// second element evaluation, and the parent's matrix-free and GPU paths are used as
// they stand. TestCouplingIdentity gates this against the explicit dense cross block.
//
// The -E_Phi delta term vanishes identically under scheme A, since Q and P are disjoint
// so the embedded |Phi> is zero on every P row. That is asserted rather than assumed:
// the reference has to carry a non-orthogonal-subspace warning
// (../ADC/adc2_pol/partgammas.f90:648) precisely because its two selections are not
// complementary, and the term becomes live again under scheme B.
//
// This mirrors getf_HIJ_adc2 / get_mgamma_adc2e (partgammas.f90:544-650), which builds
// mgammavec[J] = sum_I (M_JI - E_Phi delta_JI) Phi_I element by element. Note that
// reference's case 21 (type 3,3) assigns eb and then passes an unset ek
// (partgammas.f90:613-617, repeated at :731-735); nothing here can reproduce that,
// since no element is transcribed at all.
func Coupling(mx Applier, part *Partition, phi Discrete, be backend.Backend) ([]float64, error) {
	n := mx.Size()
	if n != len(part.qOf) {
		return nil, fmt.Errorf("fano: the matrix has %d rows but the partition covers %d; "+
			"Coupling must be given the PARENT matrix, not a restricted one", n, len(part.qOf))
	}
	if len(phi.Vec) != part.QSize() {
		return nil, fmt.Errorf("fano: |Phi> has %d components but Q has %d configurations",
			len(phi.Vec), part.QSize())
	}

	emb := part.EmbedQ(phi.Vec, nil)
	for _, j := range part.P {
		if emb[j] != 0 {
			return nil, fmt.Errorf("fano: the embedded |Phi> is nonzero (%g) on P row %d; "+
				"Q and P are not disjoint, so the -E_Phi term of g does not vanish", emb[j], j)
		}
	}

	in := be.Upload(emb)
	out := be.Alloc(n)
	defer be.Free(in)
	defer be.Free(out)
	mx.ApplyFull(out, in)
	return part.GatherP(be.Download(out), nil), nil
}

// SumRule is 2*pi*||g||^2, the exact zeroth moment of the pseudo-spectrum.
//
// It is a moment, NOT a width: the chi_i are unit-normalized rather than
// energy-normalized, so every gamma_i carries energy squared. The width is the density of
// that distribution, which only Stieltjes imaging produces.
//
// It is the completeness statement behind the whole method: for a complete basis of the
// P subspace, sum_i gamma_i = 2 pi sum_i <g|chi_i>^2 = 2 pi ||g||^2, since the chi_i
// are an orthonormal basis of P. Comparing the discrete sum against this is therefore a
// direct measure of how much of the coupling the Krylov space actually captured, and it
// is the internal check that stands in for the reference tape this layer cannot have —
// the Fano path is commented out of adc2_pol's own build, so there is no matched
// matrix-tape gate available. The reference never sums its widths at all.
func SumRule(g []float64) float64 {
	var s float64
	for _, x := range g {
		s += x * x
	}
	return 2 * math.Pi * s
}

// CoupledChannels counts the P configurations with a coupling above frac times the largest
// |g|, and returns that largest value. It is the number that says whether a basis supports
// the calculation at all, and it belongs in any run log.
//
// Stieltjes imaging reconstructs a density from a sampling of it, so what matters is not the
// size of P but how many of its configurations actually carry coupling — the rest contribute
// gamma_i = 0 and sample nothing. For an atomic Auger decay that count is roughly the number
// of final dication configurations times the number of discretized continuum orbitals per
// channel, and the latter is what a bigger basis buys. A run with a few tens of coupled
// channels will produce a width with a large Stieltjes spread however well the rest of the
// pipeline works; the paper's few-meV error bars need hundreds.
func CoupledChannels(g []float64, frac float64) (n int, max float64) {
	for _, x := range g {
		if a := math.Abs(x); a > max {
			max = a
		}
	}
	if max == 0 {
		return 0, 0
	}
	for _, x := range g {
		if math.Abs(x) > frac*max {
			n++
		}
	}
	return n, max
}

// StartRows returns the k rows of g with the largest g^2, ascending, for
// lanczos.Options.StartRows. Ties are broken by row index so the choice is
// reproducible.
//
// This is fill_stvc (../ADC/adc2_pol/fspace.f90:714-734), which sorts |g|^2 descending
// and takes the top lmain indices as the Lanczos start block. Seeding with the most
// strongly coupled configurations is what makes a truncated Krylov space capture most
// of the width: those rows are where <g|chi> comes from, so the sum rule closes fastest.
//
// k is the solver's block width, i.e. the P space's MainBlockSize(). Passing k >= len(g)
// returns every row, which is the complete-basis case.
func StartRows(g []float64, k int) []int {
	if k <= 0 {
		return nil
	}
	if k > len(g) {
		k = len(g)
	}
	idx := make([]int, len(g))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		ga, gb := g[idx[a]]*g[idx[a]], g[idx[b]]*g[idx[b]]
		if ga != gb {
			return ga > gb
		}
		return idx[a] < idx[b]
	})
	out := idx[:k:k]
	sort.Ints(out)
	return out
}
