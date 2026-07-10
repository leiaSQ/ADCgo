package render

import (
	"math"
	"testing"
)

func grid(lo, hi float64, n int) []float64 {
	g := make([]float64, n)
	for i := range g {
		g[i] = lo + (hi-lo)*float64(i)/float64(n-1)
	}
	return g
}

func argmax(xs []float64) int {
	bi, bv := 0, math.Inf(-1)
	for i, x := range xs {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

// A single line broadens to a Gaussian peaking at its own energy.
func TestEnvelopePeaksAtLine(t *testing.T) {
	g := grid(0, 20, 2001)
	env := Envelope([]Line{{Energy: 12.0, Intensity: 1, Channel: "a"}}, g,
		EnvelopeOptions{Sigma: SigmaFromFWHM(1.0)})
	if got := g[argmax(env)]; math.Abs(got-12.0) > 0.05 {
		t.Errorf("peak at %v, want ~12.0", got)
	}
}

// Channel filtering excludes non-selected lines; spin weighting scales singlets.
func TestEnvelopeChannelAndSpin(t *testing.T) {
	g := grid(0, 20, 401)
	lines := []Line{
		{Energy: 10, Intensity: 1, Channel: "keep", Spin: 1},
		{Energy: 10, Intensity: 1, Channel: "drop", Spin: 1},
	}
	only := Envelope(lines, g, EnvelopeOptions{Sigma: SigmaFromFWHM(1.0), Channels: map[string]bool{"keep": true}})
	all := Envelope(lines, g, EnvelopeOptions{Sigma: SigmaFromFWHM(1.0)})
	if maxOf(all) <= maxOf(only) {
		t.Errorf("all (%.3f) should exceed filtered (%.3f)", maxOf(all), maxOf(only))
	}

	weighted := Envelope([]Line{{Energy: 10, Intensity: 1, Channel: "keep", Spin: 1}}, g,
		EnvelopeOptions{Sigma: SigmaFromFWHM(1.0), SpinWeight: true, Ratio: 3})
	plain := Envelope([]Line{{Energy: 10, Intensity: 1, Channel: "keep", Spin: 1}}, g,
		EnvelopeOptions{Sigma: SigmaFromFWHM(1.0)})
	if math.Abs(maxOf(weighted)-3*maxOf(plain)) > 1e-9 {
		t.Errorf("singlet weighting: got %.4f, want 3x %.4f", maxOf(weighted), maxOf(plain))
	}
}

// With S_in a peak at E_i and S_fin a peak at E_f (<E_i), σ peaks at ε≈E_i−E_f,
// and there is no signal below threshold (ε where no open channel exists).
func TestElectronSpectrumPeakAndThreshold(t *testing.T) {
	g := grid(0, 40, 4001)
	sIn := Envelope([]Line{{Energy: 30, Intensity: 1, Channel: "in"}}, g, EnvelopeOptions{Sigma: SigmaFromFWHM(1.0)})
	sFin := Envelope([]Line{{Energy: 20, Intensity: 1, Channel: "fin"}}, g, EnvelopeOptions{Sigma: SigmaFromFWHM(1.0)})

	eGrid := grid(0, 30, 3001)
	sig := ElectronSpectrum(g, sIn, sFin, sFin, eGrid, 20)

	if got := eGrid[argmax(sig)]; math.Abs(got-10.0) > 0.1 {
		t.Errorf("ICD peak at ε=%v, want ~10.0", got)
	}
	// ε just above E_i (30) implies E_fin<0: no contribution.
	for i, e := range eGrid {
		if e > 31 && sig[i] > 1e-6*maxOf(sig) {
			t.Errorf("unexpected signal at ε=%v above threshold", e)
			break
		}
	}
}

// Restricting the numerator while normalizing by the total reduces the height by
// the branching fraction.
func TestElectronSpectrumPartialBranching(t *testing.T) {
	g := grid(0, 40, 4001)
	sIn := Envelope([]Line{{Energy: 30, Intensity: 1, Channel: "in"}}, g, EnvelopeOptions{Sigma: SigmaFromFWHM(1.0)})

	// Two final channels of equal weight at the same energy: numerator = one of
	// them, total = both. The partial spectrum should be ~half the total.
	finLines := []Line{
		{Energy: 20, Intensity: 1, Channel: "ICD"},
		{Energy: 20, Intensity: 1, Channel: "Auger"},
	}
	sFinTot := Envelope(finLines, g, EnvelopeOptions{Sigma: SigmaFromFWHM(1.0)})
	sFinNum := Envelope(finLines, g, EnvelopeOptions{Sigma: SigmaFromFWHM(1.0), Channels: map[string]bool{"ICD": true}})

	eGrid := grid(0, 30, 3001)
	total := ElectronSpectrum(g, sIn, sFinTot, sFinTot, eGrid, 20)
	partial := ElectronSpectrum(g, sIn, sFinNum, sFinTot, eGrid, 20)

	if r := maxOf(partial) / maxOf(total); math.Abs(r-0.5) > 0.01 {
		t.Errorf("partial/total = %.3f, want ~0.5", r)
	}
}

// Single-ionization intensity below the double-ionization onset must not produce
// a spurious spike at ε≈0: such states have no open final channel and the gate
// must exclude them, leaving the genuine E_i=30→E_f=20 decay peaking at ε≈10.
func TestElectronSpectrumSubThresholdGate(t *testing.T) {
	g := grid(0, 40, 4001)
	// A strong outer-valence SIP line at 12 (below the 20 eV onset) plus the
	// real inner-valence line at 30 that can decay.
	sIn := Envelope([]Line{
		{Energy: 12, Intensity: 10, Channel: "in"},
		{Energy: 30, Intensity: 1, Channel: "in"},
	}, g, EnvelopeOptions{Sigma: SigmaFromFWHM(1.0)})
	sFin := Envelope([]Line{{Energy: 20, Intensity: 1, Channel: "fin"}}, g, EnvelopeOptions{Sigma: SigmaFromFWHM(1.0)})

	eGrid := grid(0, 30, 3001)
	sig := ElectronSpectrum(g, sIn, sFin, sFin, eGrid, 20)

	if got := eGrid[argmax(sig)]; math.Abs(got-10.0) > 0.2 {
		t.Errorf("peak at ε=%v, want ~10.0 (sub-threshold spike not gated)", got)
	}
	// The ε≈0 bin must be small relative to the real peak.
	if r := sig[0] / maxOf(sig); r > 0.05 {
		t.Errorf("ε=0 intensity = %.3f of peak, want ≪1 (sub-threshold leakage)", r)
	}
}

func maxOf(xs []float64) float64 {
	m := math.Inf(-1)
	for _, x := range xs {
		if x > m {
			m = x
		}
	}
	return m
}
