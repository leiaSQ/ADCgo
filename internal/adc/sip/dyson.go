package sip

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// The Dyson amplitude of a cationic state — the overlap of the (N−1)-electron state
// with the neutral ground state after one electron has been removed from orbital p:
//
//	d_p = ⟨Ψ_n^{N−1}| c_pβ |Ψ_0^N⟩
//
// (β, because the sip configurations are the M_S = +1/2 doublet functions). Over the
// ISR space d_p = Σ_I f_{I,p} X_{I,n}, and the block anatomy of f is
// (docs/adc4_rassi_plan.md, element 2):
//
//	1h × occ         O(0)   1 + F⁽²⁾ + F⁽³⁾ — the existing FMatrix (amplitudes.go)
//	2h1p × virt      O(1)   f⁽¹⁾, below — the only zeroth/first-order route into the
//	                        virtual orbitals, which stand in for the ICD continuum
//	1h × virt        O(2)   second-order singles t₁⁽²⁾ — not implemented
//	2h1p × occ       O(2)   t⁽¹⁾ products — not implemented
//	3h2p × any       O(2)+  triples — not implemented
//
// The occupied projection of d is therefore exactly the spectroscopic amplitude
// a = F·Y that analyze.BuildSIPSector already reports, and ‖d_occ‖² its pole strength.
// The virtual components are new: they are the leading-order amplitude for finding the
// ejected electron in virtual orbital a, and tdmoment.go contracts them with the dipole
// integrals into an L² photoionization moment.
//
// The sign. d is defined with an overall minus relative to the literal matrix element
// above, because the 1h configuration is |i⟩ = −c_iβ|0⟩ (isrdipole.go) and the
// convention that fixes F's leading term to +1 is d_i⁽⁰⁾ = +δ_ip. The same minus then
// applies to the virtual block, where it is *not* free: μ(ε) mixes d_occ and d_virt, so
// their relative phase is observable. TestDysonVirtualMatchesDeterminants pins it.

// dysonVirtCoeff is f⁽¹⁾_{c,a}: the amplitude of the external virtual orbital a
// (absolute index) carried by the 2h1p configuration c = (holes k,l; particle b).
//
// It is ⟨Φ_c| c_aβ |Ψ₀⁽¹⁾⟩ up to the overall sign above. Annihilating one β electron
// from the first-order doubles Σ t^{ab}_{kl} lands exactly on the 2h1p configurations,
// and the spin algebra that contracts ⟨ab|kl⟩ and ⟨ab|lk⟩ against c's spin function is
// the very contraction c12_1 performs on ⟨ja|kl⟩ — c12_1 never assumes its first
// argument is occupied, it only reads integrals. So the coupling element evaluated with
// a virtual in the "external 1h" slot, divided by the MP1 denominator, *is* f⁽¹⁾.
//
// For a CVS ADC(4) space the 2h1p spin functions are the reference's Dyson-convention
// ones (type II negated), so kopp1 replaces c12_1; the two agree on type I.
func (mx *Matrix) dysonVirtCoeff(a int, c Config) float64 {
	e := mx.el
	k, l := c.Occ[0], c.Occ[1]
	b := e.nocc + c.Vir
	den := e.eps[k] + e.eps[l] - e.eps[a] - e.eps[b]
	var v float64
	if mx.isADC4() {
		v = e.kopp1(a, c)
	} else {
		v = e.c12_1(a, c)
	}
	return -v / den
}

// DysonOrbitals returns the Norb × len(states) matrix of Dyson amplitudes, one column
// per requested state, in the MO basis (row p = orbital p).
//
// vecs must carry every row of each Ritz vector — lanczos.Result.FullVecs, i.e.
// Options.WantFull or SolveDense. The satellite rows are what produce the virtual
// components; main-block-only vectors would silently yield a Dyson orbital with no
// continuum part at all, which is why they are rejected rather than tolerated.
//
// Only orbitals of the sector's irrep can be nonzero; the rest come back zero, as the
// integrals themselves enforce.
func (mx *Matrix) DysonOrbitals(vecs backend.Mat, states []int) (backend.Mat, error) {
	sp := mx.sp
	if vecs.Rows != sp.Size() {
		return backend.Mat{}, fmt.Errorf("sip: eigenvectors have %d rows, want %d (Space.Size); "+
			"lanczos.Options.WantFull retains the satellite rows", vecs.Rows, sp.Size())
	}
	for _, k := range states {
		if k < 0 || k >= vecs.Cols {
			return backend.Mat{}, fmt.Errorf("sip: state %d out of range [0,%d)", k, vecs.Cols)
		}
	}

	out := backend.NewMat(sp.Norb, len(states))
	main := sp.BeginSat

	// Occupied block: a = F·Y over the 1h main space.
	f := mx.FMatrix()
	y := make([]float64, main)
	for si, k := range states {
		for c := range main {
			y[c] = vecs.At(c, k)
		}
		a := f.MulVec(y)
		for c := range main {
			out.Set(sp.Configs[c].Occ[0], si, a[c])
		}
	}

	// Virtual block: one sweep over the 2h1p configurations, accumulating f⁽¹⁾·X into
	// every requested state at once. The 3h2p configurations of an ADC(4) space are
	// skipped — they first contribute at O(2).
	for ci := main; ci < len(sp.Configs); ci++ {
		cfg := sp.Configs[ci]
		for av := range sp.Nvir {
			a := sp.Nocc + av
			if sp.irrep(a) != sp.Sym {
				continue
			}
			fc := mx.dysonVirtCoeff(a, cfg)
			if fc == 0 {
				continue
			}
			for si, k := range states {
				out.Set(a, si, out.At(a, si)+fc*vecs.At(ci, k))
			}
		}
	}
	return out, nil
}

// SpectroscopicFactor is Σ_{i occupied} d_i², the pole strength carried by the Dyson
// orbital's occupied block. By construction it equals the ‖F·Y‖² that
// analyze.BuildSIPSector reports; the virtual components add to ‖d‖² but not to this.
func SpectroscopicFactor(sp *Space, d []float64) float64 {
	var acc float64
	for i := range sp.Nocc {
		acc += d[i] * d[i]
	}
	return acc
}

// matColumn copies column k of a row-major matrix.
func matColumn(m backend.Mat, k int) []float64 {
	v := make([]float64, m.Rows)
	for r := range m.Rows {
		v[r] = m.At(r, k)
	}
	return v
}
