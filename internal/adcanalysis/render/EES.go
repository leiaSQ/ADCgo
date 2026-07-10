// EES.go computes the electron-emission (EES) kinetic-energy spectrum from a
// single-ionization (SIP) and a double-ionization (DIP) stick spectrum, using
// the equilibrium-geometry envelope-convolution approximation. It covers the
// secondary electron emitted by ICD/ETMD/Auger-like decay, not ICD alone.
//
// An intermediate singly-ionized state at single-ionization energy E_in decays
// to a final dicationic state at double-ionization energy E_fin, emitting an
// electron of kinetic energy ε = E_in − E_fin (open only when ε > 0). The
// electron spectrum is
//
//	σ(ε) = ∫ dE · S_in(E) · S_fin_num(E−ε) / N(E),   N(E) = ∫₀^E S_fin_tot(E′) dE′
//
// where S_in is the Gaussian-broadened SIP envelope, S_fin the broadened DIP
// envelope, and N(E) the energy-dependent normalization to the open final-state
// population (the cumulative *total* DIP envelope). N uses the total envelope so
// always-open channels are not over-weighted; the numerator S_fin_num may be a
// channel subset to extract that channel's correctly weighted partial spectrum.

package render

import "math"

// fwhmToSigma is FWHM = 2*sqrt(2 ln2) * sigma.
const fwhmToSigma = 2.3548200450309493 // 2*sqrt(2*ln2)

// SigmaFromFWHM converts a Gaussian FWHM to its standard deviation.
func SigmaFromFWHM(fwhm float64) float64 { return fwhm / fwhmToSigma }

// EnvelopeOptions configures Envelope.
type EnvelopeOptions struct {
	Sigma      float64         // Gaussian standard deviation (eV)
	Channels   map[string]bool // nil/empty = include all channels
	SpinWeight bool            // scale singlet (Spin == 1) sticks by Ratio
	Ratio      float64         // singlet:triplet ratio used when SpinWeight
}

// Envelope Gaussian-broadens the selected channels' lines, summed, onto grid.
// Lines whose channel is not selected (when Channels is non-empty) are skipped.
func Envelope(lines []Line, grid []float64, opt EnvelopeOptions) []float64 {
	out := make([]float64, len(grid))
	if opt.Sigma <= 0 {
		return out
	}
	twoSigma2 := 2 * opt.Sigma * opt.Sigma
	for _, l := range lines {
		if len(opt.Channels) > 0 && !opt.Channels[l.Channel] {
			continue
		}
		w := l.Intensity
		if opt.SpinWeight && l.Spin == 1 {
			w *= opt.Ratio
		}
		for i, e := range grid {
			d := e - l.Energy
			out[i] += w * math.Exp(-d*d/twoSigma2)
		}
	}
	return out
}

// ElectronSpectrum evaluates the ICD-electron spectrum on eGrid. grid is the
// uniform energy grid (spacing dE) on which sIn, sFinNum and sFinTot are
// sampled. N(E) is the cumulative integral of sFinTot; terms where N(E) is
// non-positive (no open final channel yet) are skipped. sFinNum is linearly
// interpolated at grid[k]−ε, contributing zero outside grid.
//
// dipThreshold is the double-ionization onset (the lowest final-state energy):
// only intermediate energies E ≥ dipThreshold can undergo ICD, since below the
// onset there is no open dicationic final state to decay into. Intermediate
// energies below it are skipped. This is essential because the Gaussian-broadened
// sFinTot has a sub-threshold tail where N(E) → 0⁺; without the gate, single-
// ionization intensity sitting in that tail (outer-valence states that cannot
// ICD-decay at all) is divided by a near-zero N and produces a spurious spike
// pinned at ε ≈ 0. A non-positive dipThreshold disables the gate.
func ElectronSpectrum(grid, sIn, sFinNum, sFinTot, eGrid []float64, dipThreshold float64) []float64 {
	out := make([]float64, len(eGrid))
	n := len(grid)
	if n < 2 {
		return out
	}
	dE := grid[1] - grid[0]
	if dE <= 0 {
		return out
	}

	// Cumulative open-channel population N(E) = ∫₀^E sFinTot dE′.
	cum := make([]float64, n)
	run := 0.0
	for k := range grid {
		run += sFinTot[k] * dE
		cum[k] = run
	}

	const eps = 1e-300
	lo := grid[0]
	for j, e := range eGrid {
		var acc float64
		for k := range grid {
			// No open final channel below the double-ionization onset: an
			// intermediate at E < dipThreshold cannot ICD-decay.
			if dipThreshold > 0 && grid[k] < dipThreshold {
				continue
			}
			if cum[k] <= eps {
				continue
			}
			// Linear interpolation of sFinNum at x = grid[k] - e.
			x := grid[k] - e
			pos := (x - lo) / dE
			if pos < 0 || pos > float64(n-1) {
				continue
			}
			i := int(pos)
			var sfin float64
			if i >= n-1 {
				sfin = sFinNum[n-1]
			} else {
				frac := pos - float64(i)
				sfin = sFinNum[i]*(1-frac) + sFinNum[i+1]*frac
			}
			acc += sIn[k] * sfin / cum[k] * dE
		}
		out[j] = acc
	}
	return out
}
