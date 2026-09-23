package fano

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/internal/adc/stieltjes"
)

// image.go — the last step: Stieltjes imaging of the pseudo-continuum into a width.
//
// Kept here rather than in the stieltjes package because it is Fano-specific bookkeeping —
// which spectrum to image, what to do with a channel too thin to image, how to convert to
// the units a rate is quoted in. The imaging itself is entirely in internal/adc/stieltjes.

// Atomic-unit conversions for reporting a rate.
const (
	// HartreeToMeV converts a width in hartree to meV.
	HartreeToMeV = 27211.386245988
	// LifetimeFsMeV is hbar in fs*meV: tau[fs] = LifetimeFsMeV / Gamma[meV].
	LifetimeFsMeV = 658.2119569
)

// Width is a decay width and the lifetime it implies.
type Width struct {
	Gamma    float64 // hartree
	Sigma    float64 // hartree, the Stieltjes order-averaging standard deviation
	MeV      float64 // Gamma in meV
	SigmaMeV float64
	Tau      float64 // lifetime in fs
	// Stieltjes carries the imaging diagnostics: the orders used, the maximum usable
	// order, whether only a low-order approximation was available.
	Stieltjes *stieltjes.Result
}

func newWidth(r *stieltjes.Result) Width {
	w := Width{Gamma: r.Gamma, Sigma: r.Sigma, Stieltjes: r}
	w.MeV = r.Gamma * HartreeToMeV
	w.SigmaMeV = r.Sigma * HartreeToMeV
	if w.MeV > 0 {
		w.Tau = LifetimeFsMeV / w.MeV
	}
	return w
}

// NewWidth wraps a width computed without Stieltjes imaging (KPMWidth): Gamma and its
// uncertainty in hartree.
func NewWidth(gamma, sigma float64) Width {
	w := Width{Gamma: gamma, Sigma: sigma, MeV: gamma * HartreeToMeV, SigmaMeV: sigma * HartreeToMeV}
	if w.MeV > 0 {
		w.Tau = LifetimeFsMeV / w.MeV
	}
	return w
}

// String reports the width and lifetime the way a paper does.
func (w Width) String() string {
	s := fmt.Sprintf("Gamma = %.4g +/- %.3g meV", w.MeV, w.SigmaMeV)
	if w.Tau > 0 {
		s += fmt.Sprintf(", tau = %.4g fs", w.Tau)
	}
	if w.Stieltjes != nil && w.Stieltjes.LowOrder {
		s += " [LOW ORDER]"
	}
	return s
}

// ImageWidth images the total pseudo-continuum at the discrete state's energy.
//
// This is where a Fano calculation finally produces a width: the discrete gamma_i are a
// sampling of a distribution whose DENSITY at E_Phi is the decay width, and nothing before
// this step has the units of one (Pseudo.SumGamma is that distribution's zeroth moment, in
// hartree squared).
//
// The requested energy has to lie inside the pseudo-continuum's range, and it is worth
// knowing what it means when it does not: the basis does not represent the decay channels
// at the energy the state decays at. For a core vacancy that is the usual case in a small
// basis — an O 1s hole at 554 eV needs final states with a 500 eV free electron, and a
// handful of valence virtuals cannot supply one. The error says so rather than
// extrapolating; the reference prints a warning and returns zero.
func ImageWidth(ps *Pseudo, at float64, opts stieltjes.Options) (Width, error) {
	if ps == nil || len(ps.Energy) == 0 {
		return Width{}, fmt.Errorf("fano: the pseudo-continuum is empty; the final-state cuts " +
			"may have removed everything")
	}
	r, err := stieltjes.Image(ps.Energy, ps.Gamma, at, opts)
	if err != nil {
		return Width{}, err
	}
	return newWidth(r), nil
}

// ChannelWidth is one decay channel's imaged partial width.
type ChannelWidth struct {
	Name  string
	Width Width
	// Ratio is this channel's branching ratio, its imaged width over the sum of the
	// successfully imaged channel widths. It is zero when Err is set.
	Ratio float64
	// Err is non-nil when this channel could not be imaged on its own — most often because
	// too few pseudo-continuum states carry any of its character, which is exactly the
	// case for a weak channel. The channel is then reported with its moment share intact
	// and no width, rather than being dropped or silently given zero.
	Err error
	// Share is the channel's fraction of the total zeroth moment, which is available even
	// when imaging is not.
	Share float64
}

// ImagePartial images every channel's pseudo-spectrum separately, which is the paper's
// prescription for partial widths (ADC22.pdf §III D: channel projectors applied to the L2
// states, then "Stieltjes repeated").
//
// APPROXIMATE, and the paper says so explicitly, deferring the rigorous treatment to a
// follow-up. Two reasons beyond the channel-character approximation of PartialWidths: a
// channel's spectrum is the total one with most of its weight zeroed, so it samples its own
// density more thinly than the total does and images less accurately; and a channel whose
// support is too thin cannot be imaged at all, which is reported per channel rather than
// folded into the total.
//
// The returned ratios are over the channels that COULD be imaged, so they sum to 1 across
// those; compare them against Share, which is over all channels, to see how much was lost.
func ImagePartial(p *Partial, at float64, opts stieltjes.Options) ([]ChannelWidth, error) {
	if p == nil || len(p.Names) == 0 {
		return nil, fmt.Errorf("fano: no decay channels to image")
	}
	out := make([]ChannelWidth, len(p.Names))
	var total float64
	for b, name := range p.Names {
		cw := ChannelWidth{Name: name, Share: p.Sum[b] / nonZero(p.Total)}
		r, err := stieltjes.Image(p.Energy, p.Gamma[b], at, opts)
		if err != nil {
			cw.Err = fmt.Errorf("channel %q: %w", name, err)
		} else {
			cw.Width = newWidth(r)
			total += r.Gamma
		}
		out[b] = cw
	}
	if total > 0 {
		for b := range out {
			if out[b].Err == nil {
				out[b].Ratio = out[b].Width.Gamma / total
			}
		}
	}
	return out, nil
}
