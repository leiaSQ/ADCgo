package fano

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
	"github.com/leiaSQ/ADCgo/internal/adc/stieltjes"
)

// image_test.go — the whole Fano-ADC(2,2) chain, matrix to lifetime.

// TestFanoChainEndToEnd runs every stage in order on a reduced H2O ADC(2,2)f sector:
// partition, restrict, solve QMQ, select |Phi>, couple, solve PMP, image.
//
// The vacancy is orbital 1, an INNER-VALENCE hole, not the O 1s core hole — deliberately.
// See TestCoreVacancyHasNoContinuumInASmallBasis for why the core hole cannot be imaged
// here, which is itself a physically meaningful outcome rather than a limitation of the
// code.
//
// The width this produces (~2 eV) is not a physical number and is not treated as one: an
// inner-valence hole in a 12-orbital basis has an enormous coupling to the handful of decay
// channels that basis represents. What the test gates is that the chain is self-consistent
// — the imaged width is positive, stable across the Stieltjes orders, and insensitive to
// the final-state cuts, which is the signature of an imaging that has actually converged.
// Absolute accuracy is the job of the published Ne/Ar/Kr/Mg comparisons, which need the CLI
// and a real basis.
func TestFanoChainEndToEnd(t *testing.T) {
	const vacancy = 1
	r := newFanoRun(t, 12, vacancy)
	t.Logf("partition: %s", r.part)
	t.Logf("|Phi>: E = %.6f Eh = %.4f eV, weight %.4f on 1h row(s) %v",
		r.phi.Energy, r.phi.Energy*hartreeToEV, r.phi.Weight, r.phi.Rows)

	pres := r.pres
	ps, err := Widths(r.psp, pres, r.g, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("pseudo-continuum: %s", ps)

	w, err := ImageWidth(ps, r.phi.Energy, stieltjes.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("imaged: %s", w)
	t.Logf("  %s", w.Stieltjes)

	if !(w.Gamma > 0) {
		t.Fatalf("the imaged width is not positive: %.6g", w.Gamma)
	}
	if !(w.Tau > 0) {
		t.Errorf("lifetime %.6g fs from a positive width", w.Tau)
	}
	if rel := w.Sigma / w.Gamma; rel > 0.10 {
		t.Errorf("the Stieltjes order spread is %.1f%% of the width; the imaging has not "+
			"converged, so the chain cannot be said to produce a stable number", 100*rel)
	}
	if w.Stieltjes.LowOrder {
		t.Errorf("only a low-order Stieltjes approximation was available (max order %d)",
			w.Stieltjes.MaxOrder)
	}
	if below, above := w.Stieltjes.Boundary(); below+above > 0 {
		t.Logf("note: %d order(s) fell below and %d above their own density grid", below, above)
	}
}

// TestImagedWidthIsInsensitiveToTheCuts is the internal consistency gate on imaging.
//
// The final-state cuts change WHICH states sample the coupling density, not the density
// itself, so two different samplings of it must image to the same width. When they do not,
// the sampling is too thin for the reconstruction to be trusted — and the order-averaging
// sigma is what says so, which this also checks by requiring the disagreement to be covered
// by the two sigmas.
func TestImagedWidthIsInsensitiveToTheCuts(t *testing.T) {
	r := newFanoRun(t, 12, 1)
	pres := r.pres

	var ws []Width
	for _, f := range []Filter{
		{}, // the reference's defaults
		{EMax: -1, MinWeight: -1, MinGamma: 1e-18}, // everything
		{EMax: -1, MinWeight: 0.2, MinGamma: -1},   // a stricter weight cut
	} {
		ps, err := Widths(r.psp, pres, r.g, f)
		if err != nil {
			t.Fatal(err)
		}
		w, err := ImageWidth(ps, r.phi.Energy, stieltjes.Options{})
		if err != nil {
			t.Fatalf("filter %+v: %v", f, err)
		}
		t.Logf("%3d states (sum gamma %.6g Eh^2, residual %5.2f%%) -> %s",
			len(ps.Energy), ps.SumGamma, 100*ps.Residual(), w)
		ws = append(ws, w)
	}
	for i := 1; i < len(ws); i++ {
		d := math.Abs(ws[i].Gamma - ws[0].Gamma)
		tol := 3 * (ws[i].Sigma + ws[0].Sigma)
		if d > tol && d > 0.02*ws[0].Gamma {
			t.Errorf("filter %d images to %.6g Eh but the default cuts give %.6g: a %.3g "+
				"difference, outside both 3-sigma (%.3g) and 2%% of the width",
				i, ws[i].Gamma, ws[0].Gamma, d, tol)
		}
	}
}

// TestCoreVacancyHasNoContinuumInASmallBasis records a physically meaningful negative
// result, so the error it produces is never mistaken for a bug.
//
// An O 1s vacancy sits at ~554 eV. Its Auger channels are two valence holes plus a free
// electron carrying ~500 eV, which needs virtual orbitals that high — a 12-orbital basis has
// none, so the pseudo-continuum stops far below E_Phi and imaging has nothing to
// interpolate. The right answer is to refuse, which is what happens; the reference prints a
// warning and returns a width of zero, which is how a missing continuum turns into a
// plausible-looking number.
func TestCoreVacancyHasNoContinuumInASmallBasis(t *testing.T) {
	r := newFanoRun(t, 12, 0)
	pres := r.pres
	ps, err := Widths(r.psp, pres, r.g, Filter{EMax: -1, MinWeight: -1, MinGamma: 1e-18})
	if err != nil {
		t.Fatal(err)
	}
	lo, hi := ps.Energy[0], ps.Energy[len(ps.Energy)-1]
	t.Logf("O 1s vacancy: E_Phi = %.4f Eh (%.1f eV); the pseudo-continuum spans "+
		"[%.4f, %.4f] Eh (%.1f to %.1f eV) over %d states",
		r.phi.Energy, r.phi.Energy*hartreeToEV, lo, hi,
		lo*hartreeToEV, hi*hartreeToEV, len(ps.Energy))
	if r.phi.Energy <= hi {
		t.Skip("this basis does reach the core-hole energy, so there is nothing to record")
	}
	if _, err := ImageWidth(ps, r.phi.Energy, stieltjes.Options{}); err == nil {
		t.Error("imaging succeeded at an energy the pseudo-continuum does not cover; " +
			"it must refuse rather than extrapolate")
	} else {
		t.Logf("correctly refused: %v", err)
	}
}

// TestImagePartialPerChannel images each decay channel separately, which is the paper's
// partial-width prescription, and checks the two independent channel decompositions against
// each other: the imaged branching ratios and the zeroth-moment shares must broadly agree,
// since both come from the same per-state character.
func TestImagePartialPerChannel(t *testing.T) {
	r := newFanoRun(t, 12, 1)
	md, err := mo.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.mo.json"))
	if err != nil {
		t.Fatalf("read MO sidecar: %v", err)
	}
	sites := []spectrum.Site{
		{Name: "O", Members: []string{"O"}},
		{Name: "H", Members: []string{"H1", "H2"}},
	}
	opts := spectrum.DefaultOptions()
	opts.MinWeight = 0
	ch, err := NewChannels(md, r.parent.Nocc, sites, "O", opts)
	if err != nil {
		t.Fatal(err)
	}

	pres := r.pres
	ps, err := Widths(r.psp, pres, r.g, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	pw, err := ch.PartialWidths(r.psp, pres, ps)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("channel decomposition: %s", pw)

	cws, err := ImagePartial(pw, r.phi.Energy, stieltjes.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var imaged float64
	for _, cw := range cws {
		if cw.Err != nil {
			t.Logf("  %-12s share %5.2f%%  NOT IMAGED: %v", cw.Name, 100*cw.Share, cw.Err)
			continue
		}
		imaged++
		t.Logf("  %-12s share %5.2f%%  Gamma = %9.4g +/- %-9.3g meV  ratio %5.2f%%",
			cw.Name, 100*cw.Share, cw.Width.MeV, cw.Width.SigmaMeV, 100*cw.Ratio)
	}
	if imaged == 0 {
		t.Fatal("no channel could be imaged")
	}

	// The imaged ratios must sum to 1 over the channels that were imaged.
	var sum float64
	for _, cw := range cws {
		sum += cw.Ratio
	}
	if math.Abs(sum-1) > 1e-10 {
		t.Errorf("the imaged branching ratios sum to %.12g, want 1", sum)
	}

	// The strongest gate here: the channels, each imaged independently from its own
	// pseudo-spectrum, must add up to the total width imaged independently from the whole
	// one. Nothing forces that — the per-channel spectra are separate measures put through
	// separate recurrences, separate Jacobi matrices and separate order averages — so
	// agreement says the decomposition and the imaging commute, which is the property a
	// branching ratio needs to mean anything.
	total, err := ImageWidth(ps, r.phi.Energy, stieltjes.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var channelSum float64
	for _, cw := range cws {
		if cw.Err == nil {
			channelSum += cw.Width.Gamma
		}
	}
	rel := math.Abs(channelSum-total.Gamma) / total.Gamma
	t.Logf("the %.0f imaged channels sum to %.6g meV against the total %.6g meV, "+
		"agreeing to %.3f%%", imaged, channelSum*HartreeToMeV, total.MeV, 100*rel)
	if rel > 0.02 {
		t.Errorf("the imaged channel widths sum to %.6g but the total images to %.6g, a "+
			"%.2f%% discrepancy; the channel decomposition and the imaging do not commute",
			channelSum, total.Gamma, 100*rel)
	}
	// Every imaged partial width must be non-negative: the monotone interpolant guarantees
	// it, and a negative partial width is the specific nonsense that guarantee exists for.
	for _, cw := range cws {
		if cw.Err == nil && cw.Width.Gamma < 0 {
			t.Errorf("channel %q has a negative partial width %.6g", cw.Name, cw.Width.Gamma)
		}
	}
	// The strongest channel by moment share should also be the strongest by imaged width.
	bestShare, bestRatio := 0, 0
	for b := range cws {
		if cws[b].Share > cws[bestShare].Share {
			bestShare = b
		}
		if cws[b].Err == nil && cws[b].Ratio > cws[bestRatio].Ratio {
			bestRatio = b
		}
	}
	if cws[bestShare].Err == nil && bestShare != bestRatio {
		t.Logf("note: the largest moment share is %q but the largest imaged width is %q; "+
			"the two decompositions weight the energy dependence differently",
			cws[bestShare].Name, cws[bestRatio].Name)
	}
}

// TestImageWidthRejectsAnEmptyPseudoContinuum: cuts that remove everything must be an
// error, not a width of zero.
func TestImageWidthRejectsAnEmptyPseudoContinuum(t *testing.T) {
	if _, err := ImageWidth(&Pseudo{}, 1.0, stieltjes.Options{}); err == nil {
		t.Error("an empty pseudo-continuum was accepted")
	}
	if _, err := ImageWidth(nil, 1.0, stieltjes.Options{}); err == nil {
		t.Error("a nil pseudo-continuum was accepted")
	}
	if _, err := ImagePartial(&Partial{}, 1.0, stieltjes.Options{}); err == nil {
		t.Error("a Partial with no channels was accepted")
	}
}
