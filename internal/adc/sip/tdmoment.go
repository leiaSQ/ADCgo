package sip

import "github.com/leiaSQ/ADCgo/internal/adc/backend"

// The bound→continuum half of the decay chain (docs/adc4_rassi_plan.md, element 2):
// the neighbour's photoionization, which ejects the ICD electron.
//
// There is no L² representation of a continuum orbital. The standard discretized
// surrogate is to let each virtual MO φ_a stand in for the outgoing electron at its own
// orbital energy ε_a, and to take the photoionization dipole of channel n as the
// one-electron matrix element between that surrogate and the channel's Dyson orbital,
//
//	μ_a(n) = ⟨φ_a| r |d_n⟩ = Σ_p D^MO_{ap} d_p ,   ω_a(n) = E_n + ε_a .
//
// The resulting (ω, f) pairs are a *pseudo-spectrum*, not a cross-section: the discrete
// strengths must be smoothed by Stieltjes imaging before σ_ion(ω) means anything. That
// smoothing is Track W's Fano–Stieltjes machinery; this file emits the discrete moments
// and records ε_a so a caller can discard the ε_a < 0 (bound-like) virtuals first.

// ChannelMoment is one discretized photoionization channel: cationic state n left
// behind, ejected electron parked in virtual orbital Vir.
type ChannelMoment struct {
	State int // index into the state list handed to PhotoionizationMoments
	Vir   int // 0-based virtual position; absolute orbital = Space.Nocc + Vir

	Eps   float64 // ε_a, hartree — the L² proxy for the photoelectron energy
	Omega float64 // photon energy E_n + ε_a, hartree

	// Mu is ⟨φ_a|r|d_n⟩ in a.u.: the one-electron dipole, without the electron's
	// charge — that is an overall sign, and it squares out of Osc.
	Mu  [3]float64
	Osc float64 // oscillator strength (2/3)·ω·|μ|²
}

// PhotoionizationMoments evaluates μ_a and its oscillator strength for every virtual
// orbital, given one state's Dyson orbital d (length Space.Norb, from DysonOrbitals)
// and that state's ionization energy eIP in hartree. dipMO is mo.Data.DipMO.
//
// state only labels the output rows; it is not read.
func PhotoionizationMoments(dipMO [3]backend.Mat, sp *Space, eps []float64, d []float64, eIP float64, state int) []ChannelMoment {
	out := make([]ChannelMoment, 0, sp.Nvir)
	for av := range sp.Nvir {
		a := sp.Nocc + av
		cm := ChannelMoment{State: state, Vir: av, Eps: eps[a]}
		cm.Omega = eIP + eps[a]
		for c := range 3 {
			var acc float64
			for p := range sp.Norb {
				acc += dipMO[c].At(a, p) * d[p]
			}
			cm.Mu[c] = acc
		}
		cm.Osc = OscillatorStrength(cm.Omega, cm.Mu)
		out = append(out, cm)
	}
	return out
}
