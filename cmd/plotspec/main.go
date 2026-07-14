// Command plotspec renders an ADCanalysis spectrum JSON into a PNG using Gonum
// Plot. It reads the schema written by cmd/ADCanalysis (decay_analyzer_sketch.md
// §6), Gaussian-broadens each channel's sticks onto a shared energy grid, and
// overlays one curve per channel.
//
// Broadening and (optional) singlet:triplet weighting live here, not in the Go
// analyzer, so they can be re-tuned without re-running the pipeline.
//
// An optional experimental/reference spectrum (-exp) in the same JSON schema is
// overlaid as dotted lines, drawn in the same per-channel colours as the theory
// curves so each measured channel sits directly on top of its calculated
// counterpart. Reference rates already embed the true singlet:triplet intensity
// ratio, so -spin-weight is applied only to the theoretical sticks, never to the
// experimental overlay (that would double-count the multiplicity weighting).
//
// Usage:
//
//	plotspec -in spec.json -out spectrum.png
//	plotspec -in spec.json -out spectrum.png -fwhm 1.2 -spin-weight
//	plotspec -in spec.json -exp h2o_auger_experimental.json -out spectrum.png
//	plotspec -in spec.json -out spectrum.png -xrange 30-100
//	plotspec -in sip.json  -out sip.png            # SIP: one curve per orbital
//	plotspec -mode tdm -in tdm.json -out tdm.png -stick -overlay-broadened -stick-height 0.6
//
// The same renderer handles DIP, SIP and TDM JSON (meta.kind): SIP channels are
// orbitals, and the axis/title switch to single-ionization wording.
//
// -stick draws one bar per state instead of the broadened curves; adding
// -overlay-broadened draws the curves on top of the sticks as well, and
// -stick-height scales the sticks (sticks and curves are normalised separately,
// so -stick-height 0.6 keeps the bars under the envelope). Both work in every
// single-spectrum mode (spectrum/SIP/DIP/tdm) as well as in panel mode.
//
// The Y axis defaults to relative intensity (tallest displayed peak = 1); pass
// -absolute to plot the raw broadened intensity instead.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"image/color"
	"image/png"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/text"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
	"gonum.org/v1/plot/vg/vgpdf"
	"gonum.org/v1/plot/vg/vgsvg"

	ees "github.com/leiaSQ/ADCgo/internal/adcanalysis/render"
)

// specLine is one (state, channel) stick. Extra JSON fields the analyzer or a
// reference file may carry (kinetic_energy, config, relative_intensity, ...) are
// ignored on unmarshal.
type specLine struct {
	Energy    float64 `json:"energy"`
	Intensity float64 `json:"intensity"`
	Channel   string  `json:"channel"`
	Spin      int     `json:"spin"`
}

// spectrum mirrors the subset of the JSON schema this script needs.
type spectrum struct {
	Meta struct {
		Kind                string                `json:"kind"`
		Molecule            string                `json:"molecule"`
		InitialIonization   struct{ Atom string } `json:"initial_ionization"`
		EnergyUnit          string                `json:"energy_unit"`
		SingletTripletRatio float64               `json:"singlet_triplet_ratio"`
	} `json:"meta"`
	Channels []string   `json:"channels"`
	Lines    []specLine `json:"lines"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "plotspec:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		mode        = flag.String("mode", "spectrum", "plot mode: \"spectrum\" (single DIP/SIP JSON via -in), \"tdm\" (transition-dipole spectrum from an adcgo -tdm JSON via -in), \"ees\" (electron-emission spectrum from -sip + -dip), or \"panel\"")
		sipPath     = flag.String("sip", "", "ees mode: single-ionization spectrum JSON (S_in)")
		dipPath     = flag.String("dip", "", "ees mode: double-ionization spectrum JSON (S_fin)")
		fwhmSIP     = flag.Float64("fwhm-sip", 0, "ees mode: Gaussian FWHM (eV) for the SIP envelope; 0 = use -fwhm")
		fwhmDIP     = flag.Float64("fwhm-dip", 0, "ees mode: Gaussian FWHM (eV) for the DIP envelope; 0 = use -fwhm")
		fwhmEES     = flag.Float64("fwhm-ees", 0, "ees/panel mode: Gaussian FWHM (eV) for both envelopes inside the EES convolution, decoupling it from the panel (a)/(b) overlays; 0 = use -fwhm-sip/-fwhm-dip")
		finChans    = flag.String("fin-channels", "", "ees mode: restrict the final (numerator) DIP channels, comma-separated names or prefixes (e.g. \"ICD\"); empty = all. N(E) always uses all channels")
		inPath      = flag.String("in", "spec.json", "input spectrum JSON (from ADCanalysis)")
		expPath     = flag.String("exp", "", "optional experimental/reference spectrum JSON (same schema); overlaid as dotted lines")
		expScale    = flag.Float64("exp-scale", 0, "scale factor for the experimental overlay; 0 = auto (match its tallest peak to the theory's)")
		outPath     = flag.String("out", "spectrum.png", "output image (.png/.svg/.pdf by extension)")
		fwhm        = flag.Float64("fwhm", 1, "Gaussian FWHM for broadening, in eV")
		points      = flag.Int("points", 1000, "number of points on the energy grid")
		pad         = flag.Float64("pad", 5.0, "energy padding (eV) added on each side of the data range")
		xRange      = flag.String("xrange", "", "override X-axis range, e.g. \"30-100\" (eV); default is data range ± pad")
		absolute    = flag.Bool("absolute", false, "plot absolute broadened intensity instead of relative (tallest peak = 1)")
		spinWeight  = flag.Bool("spin-weight", false, "weight singlet sticks by meta.singlet_triplet_ratio (triplet = 1); theory only, never applied to -exp")
		width       = flag.Float64("width", 8, "figure width in inches")
		height      = flag.Float64("height", 5, "figure height in inches")
		dpi         = flag.Int("dpi", 192, "resolution for raster output (PNG); ignored for SVG/PDF")
		stick       = flag.Bool("stick", false, "plot sticks instead of curve (default is false)")
		colorblind  = flag.Bool("colorblind", false, "use the Okabe–Ito colour-blind-safe palette")
		xRangeAB    = flag.String("xrange-ab", "0-40", "panel mode: shared X window (eV) for panels (a) SIP and (b) DIP")
		xRangeC     = flag.String("xrange-c", "0-14", "panel mode: X window (eV) for panel (c) electron-emission spectrum")
		sipGroups   = multiFlag{}
		sipGroupI   bool
		overlay     = flag.Bool("overlay-broadened", false, "overlay the Gaussian-broadened curves on the -stick spectrum (spectrum/tdm mode: one curve per channel; panel mode: one total envelope on the (a)/(b) sticks)")
		stickHeight = flag.Float64("stick-height", 1, "scale factor for the normalised stick heights (the broadened curves stay at 1); use <1 so the sticks sit below the -overlay-broadened curve")
	)
	flag.Var(sipGroupFlag{specs: &sipGroups, interactive: &sipGroupI}, "sip-group", "panel mode: group panel (a) MOs. Bare \"-sip-group\" opens an interactive dialogue listing every MO; or pass a spec \"-sip-group=core=1,2\" (MO numbers, or symN for a whole symmetry), append #RRGGBB to the label for a custom colour, e.g. \"-sip-group=core#e41a1c=1,2\" (repeatable)")
	flag.Parse()

	// -colorblind applies to every mode: swap the palette before any plotting.
	if *colorblind {
		activePalette = okabeItoColors
	}

	switch *mode {
	case "ees":
		return runEES(eesParams{
			sip: *sipPath, dip: *dipPath, out: *outPath,
			fwhm: *fwhm, fwhmSIP: *fwhmSIP, fwhmDIP: *fwhmDIP, fwhmEES: *fwhmEES,
			finChannels: *finChans, points: *points, pad: *pad, xRange: *xRange,
			absolute: *absolute, spinWeight: *spinWeight,
			width: *width, height: *height, dpi: *dpi, stick: *stick,
		})
	case "panel":
		return runPanel(panelParams{
			sip: *sipPath, dip: *dipPath, out: *outPath,
			fwhm: *fwhm, fwhmSIP: *fwhmSIP, fwhmDIP: *fwhmDIP, fwhmEES: *fwhmEES,
			finChannels: *finChans, points: *points, pad: *pad,
			xRangeAB: *xRangeAB, xRangeC: *xRangeC,
			sipGroups: sipGroups, sipGroupI: sipGroupI, overlay: *overlay, stickHeight: *stickHeight,
			absolute: *absolute, spinWeight: *spinWeight,
			width: *width, height: *height, dpi: *dpi,
		})
	case "spectrum":
		// fall through to the single-spectrum renderer below
	case "tdm":
		// fall through: the transition-dipole document is flattened into the same
		// (energy, intensity, channel) spectrum below, then rendered identically.
	default:
		return fmt.Errorf("-mode must be \"spectrum\", \"tdm\", \"ees\" or \"panel\", got %q", *mode)
	}

	var spec *spectrum
	var err error
	if *mode == "tdm" {
		spec, err = readTDM(*inPath)
	} else {
		spec, err = readSpectrum(*inPath)
	}
	if err != nil {
		return err
	}
	if len(spec.Lines) == 0 {
		return fmt.Errorf("%s contains no lines", *inPath)
	}

	// Optional experimental/reference overlay.
	var exp *spectrum
	if *expPath != "" {
		exp, err = readSpectrum(*expPath)
		if err != nil {
			return err
		}
		if len(exp.Lines) == 0 {
			return fmt.Errorf("%s contains no lines", *expPath)
		}
	}

	// sigma from FWHM: FWHM = 2*sqrt(2 ln2) * sigma.
	sigma := *fwhm / (2 * math.Sqrt(2*math.Ln2))
	if sigma <= 0 {
		return fmt.Errorf("fwhm must be > 0")
	}

	// Energy grid spanning all sticks (theory + overlay) plus padding, so the
	// overlay is never clipped when it extends past the theory range. An
	// explicit -xrange overrides this and fixes the grid (and axis) to the
	// requested window instead.
	lo, hi := spec.Lines[0].Energy, spec.Lines[0].Energy
	if *xRange != "" {
		lo, hi, err = parseRange(*xRange)
		if err != nil {
			return err
		}
	} else {
		expand := func(lines []specLine) {
			for _, l := range lines {
				lo = math.Min(lo, l.Energy)
				hi = math.Max(hi, l.Energy)
			}
		}
		expand(spec.Lines)
		if exp != nil {
			expand(exp.Lines)
		}
		lo -= *pad
		hi += *pad
	}
	grid := make([]float64, *points)
	for i := range grid {
		grid[i] = lo + (hi-lo)*float64(i)/float64(*points-1)
	}

	ratio := spec.Meta.SingletTripletRatio
	if ratio == 0 {
		ratio = 1
	}

	// Channel orders (canonical: meta.channels first, then any extras sorted).
	order := channelOrder(*spec)
	var expOrder []string
	if exp != nil {
		expOrder = channelOrder(*exp)
	}

	// Theory/overlay intensities are Gaussian-broadened curves on the grid, raw
	// sticks (one vertical bar per state) when -stick is set, or both when -stick
	// is combined with -overlay-broadened. Each geometry applies the same optional
	// spin weighting, experimental scaling and relative normalisation, but they are
	// normalised independently (sticks and broadened intensities carry different
	// units), so -stick-height can scale the sticks down under the curves.
	var curves, expCurves map[string][]float64   // curves: broadened
	var sticks, expSticks map[string]plotter.XYs // sticks: one bar per state
	drawSticks := *stick
	drawCurves := !*stick || *overlay

	if drawSticks {
		sticks = stickHeights(spec.Lines, order, *spinWeight, ratio)
		if exp != nil {
			// Measured rates already embed the singlet:triplet ratio, so spin
			// weighting is never applied to the overlay (see package doc).
			expSticks = stickHeights(exp.Lines, expOrder, false, 1)

			// Scale the overlay so it is visually comparable to the theory; a
			// single global factor preserves relative heights within the overlay.
			scale := *expScale
			if scale == 0 {
				tMax, eMax := maxXYIn(sticks, lo, hi), maxXYIn(expSticks, lo, hi)
				if tMax > 0 && eMax > 0 {
					scale = tMax / eMax
				} else {
					scale = 1
				}
			}
			for _, xys := range expSticks {
				for i := range xys {
					xys[i].Y *= scale
				}
			}
			fmt.Fprintf(os.Stderr, "Plotspec: experimental overlay scaled by %.4g\n", scale)
		}

		// Normalise so the tallest displayed stick = 1 (unless -absolute), then
		// apply -stick-height. The overlay shares both factors so its calibrated
		// relative height is kept. Only sticks inside the plotted window count:
		// under -xrange a taller line just outside it would otherwise squash the
		// whole visible spectrum (the broadened curves, evaluated only on the
		// in-window grid, are normalised the same way).
		scale := 1.0
		if !*absolute {
			norm := maxXYIn(sticks, lo, hi)
			if exp != nil {
				if m := maxXYIn(expSticks, lo, hi); m > norm {
					norm = m
				}
			}
			if norm > 0 {
				scale = 1 / norm
			}
		}
		if *stickHeight > 0 {
			scale *= *stickHeight
		}
		if scale != 1 {
			for _, xys := range sticks {
				for i := range xys {
					xys[i].Y *= scale
				}
			}
			for _, xys := range expSticks {
				for i := range xys {
					xys[i].Y *= scale
				}
			}
		}
	}

	if drawCurves {
		// Broaden theory channels (canonical order; any stray channel appended).
		curves = broaden(spec.Lines, order, grid, sigma, *spinWeight, ratio)

		// Broaden the experimental overlay. Spin weighting is deliberately NOT
		// applied: measured rates already contain the real singlet:triplet ratio.
		if exp != nil {
			expCurves = broaden(exp.Lines, expOrder, grid, sigma, false, 1)

			// Scale the overlay so it is visually comparable to the theory (their
			// intensity units differ: populations vs. transition rates). A single
			// global factor preserves the relative heights within the overlay.
			scale := *expScale
			if scale == 0 {
				tMax, eMax := maxCurve(curves), maxCurve(expCurves)
				if tMax > 0 && eMax > 0 {
					scale = tMax / eMax
				} else {
					scale = 1
				}
			}
			for _, ys := range expCurves {
				for i := range ys {
					ys[i] *= scale
				}
			}
			fmt.Fprintf(os.Stderr, "Plotspec: experimental overlay scaled by %.4g\n", scale)
		}

		// Normalise to relative intensity (tallest displayed peak = 1) unless the
		// caller asks for absolute broadened intensity. The overlay shares the same
		// divisor so its calibrated height relative to the theory is preserved.
		if !*absolute {
			norm := maxCurve(curves)
			if exp != nil {
				if m := maxCurve(expCurves); m > norm {
					norm = m
				}
			}
			if norm > 0 {
				for _, ys := range curves {
					for i := range ys {
						ys[i] /= norm
					}
				}
				for _, ys := range expCurves {
					for i := range ys {
						ys[i] /= norm
					}
				}
			}
		}
	}

	// Stable colour per channel: theory channels take palette slots in order;
	// any experimental-only channel gets the next free slot. Matching channels
	// (e.g. Auger@O in both) thus share a colour — dotted exp over solid theory.
	colorIdx := map[string]int{}
	for i, ch := range order {
		colorIdx[ch] = i
	}
	next := len(order)
	for _, ch := range expOrder {
		if _, ok := colorIdx[ch]; !ok {
			colorIdx[ch] = next
			next++
		}
	}

	// Assemble the plot. SIP is a per-orbital single-ionization (binding-energy)
	// spectrum; DIP is a decay-channel double-ionization spectrum.
	p := plot.New()
	kind := spec.Meta.Kind
	sip := kind == "sip"
	tdm := kind == "tdm"
	unit := spec.Meta.EnergyUnit
	if unit == "" {
		unit = "eV"
	}
	var title string
	switch {
	case tdm:
		title = "Transition-dipole spectrum"
	case sip:
		title = "Single-ionization spectrum"
	default:
		title = "Decay-channel spectrum"
	}
	if spec.Meta.Molecule != "" {
		title = spec.Meta.Molecule + " " + title
	}
	if a := spec.Meta.InitialIonization.Atom; a != "" && !sip && !tdm {
		title += fmt.Sprintf(" (initial: %s)", a)
	}
	p.Title.Text = title
	switch {
	case tdm:
		p.X.Label.Text = "Transition energy (" + unit + ")"
	case sip:
		p.X.Label.Text = "Ionization energy (" + unit + ")"
	default:
		p.X.Label.Text = "Double-ionization energy (" + unit + ")"
	}
	// The transition-dipole intensity is oscillator strength, not a population, so
	// its Y axis is labelled accordingly; every other kind is a generic intensity.
	yQty := "intensity"
	yQtyCap := "Intensity"
	if tdm {
		yQty = "oscillator strength"
		yQtyCap = "Oscillator strength"
	}
	geom := fmt.Sprintf("Gaussian-broadened, FWHM %.2g %s", *fwhm, unit)
	switch {
	case drawSticks && drawCurves:
		geom = fmt.Sprintf("sticks + Gaussian FWHM %.2g %s", *fwhm, unit)
	case drawSticks:
		geom = "sticks"
	}
	if *absolute {
		p.Y.Label.Text = fmt.Sprintf("%s (%s)", yQtyCap, geom)
	} else {
		p.Y.Label.Text = fmt.Sprintf("Relative %s (%s)", yQty, geom)
	}
	p.Legend.Top = true

	if drawSticks {
		// All channels' sticks share one plotter so they can be drawn tallest
		// first: a state shared by several channels has one stick per channel at
		// the same energy and width, and drawing the shortest last keeps it on
		// top so no taller stick of another channel hides it. Theory sticks are
		// solid; the experimental overlay is dashed in the matching colour.
		stem := newStemPlot(vg.Points(2), lo, hi)
		for _, ch := range order {
			col := palette(colorIdx[ch])
			for _, pt := range sticks[ch] {
				stem.sticks = append(stem.sticks, stem3{x: pt.X, y: pt.Y, color: col})
			}
			if len(sticks[ch]) > 0 {
				p.Legend.Add(ch, stemThumb{color: col, width: vg.Points(2)})
			}
		}
		dash := []vg.Length{vg.Points(2), vg.Points(2)}
		for _, ch := range expOrder {
			col := palette(colorIdx[ch])
			for _, pt := range expSticks[ch] {
				stem.sticks = append(stem.sticks, stem3{x: pt.X, y: pt.Y, color: col, dashes: dash})
			}
			if len(expSticks[ch]) > 0 {
				p.Legend.Add(ch+" (exp)", stemThumb{color: col, width: vg.Points(2), dashes: dash})
			}
		}
		p.Add(stem)
	}

	if drawCurves {
		// Theory curves: solid. When they overlay sticks the channels already have
		// a legend entry, so only the curve-only plot adds one here.
		for _, ch := range order {
			line, err := plotter.NewLine(asXYs(grid, curves[ch]))
			if err != nil {
				return err
			}
			line.Color = palette(colorIdx[ch])
			line.Width = vg.Points(1.5)
			p.Add(line)
			if !drawSticks {
				p.Legend.Add(ch, line)
			}
		}

		// Experimental curves: dotted, same colour as the matching theory channel.
		for _, ch := range expOrder {
			line, err := plotter.NewLine(asXYs(grid, expCurves[ch]))
			if err != nil {
				return err
			}
			line.Color = palette(colorIdx[ch])
			line.Width = vg.Points(1.5)
			line.Dashes = []vg.Length{vg.Points(2), vg.Points(2)}
			p.Add(line)
			if !drawSticks {
				p.Legend.Add(ch+" (exp)", line)
			}
		}
	}

	// Apply the explicit X window last: p.Add expands the axes to fit every
	// plotter's DataRange, so setting it earlier would be undone (the stick
	// plotter holds all sticks, not just those in range).
	if *xRange != "" {
		p.X.Min, p.X.Max = lo, hi
	}

	if err := savePlot(p, *outPath, *width, *height, *dpi); err != nil {
		return err
	}
	nLines := len(spec.Lines)
	nChans := len(order)
	if exp != nil {
		nLines += len(exp.Lines)
		nChans += len(expOrder)
	}
	fmt.Fprintf(os.Stderr, "plotspec: wrote %s (%d channels, %d sticks)\n",
		*outPath, nChans, nLines)
	return nil
}

// tdmTransition is one radiative transition (emission or cross-emission) of the
// -tdm document written by `adcgo -sip -tdm` (cmd/adcgo/sip_tdm.go). Only the
// fields the spectrum needs are decoded; the rest (mu, rate, indices) are ignored.
type tdmTransition struct {
	OmegaEV float64 `json:"omega_ev"` // photon energy E_init − E_mid, eV
	Osc     float64 `json:"osc"`      // oscillator strength f
}

// tdmPhotChannel is one virtual-orbital channel of a state's Dyson
// photoionization pseudo-spectrum.
type tdmPhotChannel struct {
	OmegaEV float64 `json:"omega_ev"` // photon energy E_state + ε_a, eV
	Osc     float64 `json:"osc"`      // oscillator strength (2/3)·ω·|μ|²
}

type tdmPhotoionization struct {
	Channels []tdmPhotChannel `json:"channels"`
}

// tdmDocument mirrors the subset of cmd/adcgo's SIPTDMDocument that carries
// transition intensities. Unknown fields (sectors, norb, …) are ignored.
type tdmDocument struct {
	Emissions       []tdmTransition      `json:"emissions"`
	CrossEmissions  []tdmTransition      `json:"cross_emissions"`
	Photoionization []tdmPhotoionization `json:"photoionization"`
}

// tdmChannelOrder is the canonical channel ordering for a transition-dipole plot.
var tdmChannelOrder = []string{"emission", "cross-emission", "photoionization"}

// readTDM loads a -tdm document and flattens its transitions into the standard
// (energy, intensity, channel) spectrum so the shared renderer draws it: the
// x-position is the photon energy omega_ev, the height is the oscillator strength
// osc, and the three transition families become the plotted channels. Transitions
// with non-positive oscillator strength are dropped (dipole-forbidden / numerical
// zeros contribute no visible stick).
func readTDM(path string) (*spectrum, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d tdmDocument
	if err := json.Unmarshal(b, &d); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	s := &spectrum{}
	s.Meta.Kind = "tdm"
	s.Meta.EnergyUnit = "eV"
	add := func(ch string, omega, osc float64) {
		if osc <= 0 {
			return
		}
		s.Lines = append(s.Lines, specLine{Energy: omega, Intensity: osc, Channel: ch})
	}
	for _, e := range d.Emissions {
		add("emission", e.OmegaEV, e.Osc)
	}
	for _, e := range d.CrossEmissions {
		add("cross-emission", e.OmegaEV, e.Osc)
	}
	for _, ph := range d.Photoionization {
		for _, c := range ph.Channels {
			add("photoionization", c.OmegaEV, c.Osc)
		}
	}
	// Emit only the channels actually present, in canonical order, so the legend
	// and colour assignment are deterministic across runs.
	present := map[string]bool{}
	for _, l := range s.Lines {
		present[l.Channel] = true
	}
	for _, ch := range tdmChannelOrder {
		if present[ch] {
			s.Channels = append(s.Channels, ch)
		}
	}
	return s, nil
}

// readSpectrum loads and decodes one spectrum JSON file.
func readSpectrum(path string) (*spectrum, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s spectrum
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return &s, nil
}

// savePlot writes p to outPath, rasterizing for .png/.jpg/.jpeg (honouring dpi)
// and using gonum's vector writer (.svg/.pdf) otherwise.
func savePlot(p *plot.Plot, outPath string, width, height float64, dpi int) error {
	w, h := vg.Length(width)*vg.Inch, vg.Length(height)*vg.Inch
	ext := strings.ToLower(outPath)
	if strings.HasSuffix(ext, ".png") || strings.HasSuffix(ext, ".jpg") || strings.HasSuffix(ext, ".jpeg") {
		c := vgimg.NewWith(vgimg.UseDPI(dpi), vgimg.UseWH(w, h))
		p.Draw(draw.New(c))
		f, err := os.Create(outPath)
		if err != nil {
			return fmt.Errorf("create %s: %w", outPath, err)
		}
		defer f.Close()
		if err := png.Encode(f, c.Image()); err != nil {
			return fmt.Errorf("write %s: %w", outPath, err)
		}
		return nil
	}
	if err := p.Save(w, h, outPath); err != nil {
		return fmt.Errorf("save %s: %w", outPath, err)
	}
	return nil
}

// eesParams bundles the flags driving the electron-emission-spectrum mode.
type eesParams struct {
	sip, dip, out                   string
	fwhm, fwhmSIP, fwhmDIP, fwhmEES float64
	finChannels                     string
	points                          int
	pad                             float64
	xRange                          string
	absolute, spinWeight            bool
	width, height                   float64
	dpi                             int
	stick                           bool
}

// runEES builds the electron-emission spectrum σ(ε)=∫ S_in(E) S_fin(E−ε)/N(E) dE
// from a single- and a double-ionization spectrum, then plots it against electron
// kinetic energy. S_in is the broadened SIP envelope; S_fin the broadened DIP
// envelope; N(E) the cumulative *total* DIP envelope (open final population).
// -fin-channels restricts only the numerator so a channel's partial electron
// spectrum stays correctly weighted by the full branching.
func runEES(p eesParams) error {
	if p.sip == "" || p.dip == "" {
		return fmt.Errorf("-mode ees requires both -sip and -dip")
	}
	sipSpec, err := readSpectrum(p.sip)
	if err != nil {
		return err
	}
	dipSpec, err := readSpectrum(p.dip)
	if err != nil {
		return err
	}
	if len(sipSpec.Lines) == 0 {
		return fmt.Errorf("%s contains no lines", p.sip)
	}
	if len(dipSpec.Lines) == 0 {
		return fmt.Errorf("%s contains no lines", p.dip)
	}

	res, err := eesElectronCurve(sipSpec, dipSpec, eesCurveParams{
		fwhm: p.fwhm, fwhmSIP: p.fwhmSIP, fwhmDIP: p.fwhmDIP, fwhmEES: p.fwhmEES,
		finChannels: p.finChannels, points: p.points, pad: p.pad,
		xRange: p.xRange, absolute: p.absolute, spinWeight: p.spinWeight,
	})
	if err != nil {
		return err
	}
	eGrid := res.eGrid
	eLo, eHi := res.eLo, res.eHi
	fwhmSIP, fwhmDIP := res.fwhmSIP, res.fwhmDIP

	plt := plot.New()
	title := "Electron-emission spectrum"
	if sipSpec.Meta.Molecule != "" {
		title = sipSpec.Meta.Molecule + " " + title
	}
	if p.finChannels != "" {
		title += fmt.Sprintf(" (%s)", p.finChannels)
	}
	plt.Title.Text = title
	plt.X.Label.Text = "Emitted electron kinetic energy (eV)"
	if p.absolute {
		plt.Y.Label.Text = "Intensity (envelope convolution)"
	} else {
		plt.Y.Label.Text = "Relative intensity"
	}
	if p.xRange != "" {
		plt.X.Min, plt.X.Max = eLo, eHi
	}
	plt.Legend.Top = true

	// One curve per decay channel, in canonical order, sharing the per-channel
	// palette so a channel keeps the same colour as in the DIP spectrum.
	for i, ch := range res.order {
		line, err := plotter.NewLine(asXYs(eGrid, res.sigmas[ch]))
		if err != nil {
			return err
		}
		line.Color = palette(i)
		line.Width = vg.Points(1.5)
		plt.Add(line)
		plt.Legend.Add(ch, line)
	}

	if err := savePlot(plt, p.out, p.width, p.height, p.dpi); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "plotspec: wrote %s (electron-emission spectrum, %d channels; SIP fwhm %.2g, DIP fwhm %.2g)\n",
		p.out, len(res.order), fwhmSIP, fwhmDIP)
	return nil
}

// multiFlag collects a repeatable string flag (e.g. -sip-group used several times).
type multiFlag []string

func (m *multiFlag) String() string { return strings.Join(*m, ", ") }
func (m *multiFlag) Set(s string) error {
	*m = append(*m, s)
	return nil
}

// panelParams bundles the flags driving the three-panel (Kuleff-style) figure.
type panelParams struct {
	sip, dip, out                   string
	fwhm, fwhmSIP, fwhmDIP, fwhmEES float64
	finChannels                     string
	points                          int
	pad                             float64
	xRangeAB, xRangeC               string
	sipGroups                       multiFlag
	sipGroupI                       bool
	overlay                         bool
	stickHeight                     float64
	absolute, spinWeight            bool
	width, height                   float64
	dpi                             int
}

// runPanel renders the composite figure: panel (a) SIP sticks (grouped by
// symmetry), panel (b) DIP sticks, and panel (c) the Gaussian-broadened
// electron-emission curve. Panels (a)/(b) are full-width, stacked, and share one
// x-axis with no gap (only (b) is labelled); panel (c) sits below with a gap,
// narrower and horizontally inset, on its own independent axis.
func runPanel(p panelParams) error {
	if p.sip == "" || p.dip == "" {
		return fmt.Errorf("-mode panel requires both -sip and -dip")
	}
	sipSpec, err := readSpectrum(p.sip)
	if err != nil {
		return err
	}
	dipSpec, err := readSpectrum(p.dip)
	if err != nil {
		return err
	}
	if len(sipSpec.Lines) == 0 {
		return fmt.Errorf("%s contains no lines", p.sip)
	}
	if len(dipSpec.Lines) == 0 {
		return fmt.Errorf("%s contains no lines", p.dip)
	}

	// Shared (a)/(b) window.
	abLo, abHi, err := parseRange(p.xRangeAB)
	if err != nil {
		return fmt.Errorf("-xrange-ab: %w", err)
	}
	abGrid := linspace(abLo, abHi, p.points)

	// Resolved FWHMs for the optional broadened overlay (default to -fwhm).
	fwhmSIP, fwhmDIP := p.fwhmSIP, p.fwhmDIP
	if fwhmSIP <= 0 {
		fwhmSIP = p.fwhm
	}
	if fwhmDIP <= 0 {
		fwhmDIP = p.fwhm
	}
	if fwhmSIP <= 0 || fwhmDIP <= 0 {
		return fmt.Errorf("fwhm must be > 0")
	}
	dipRatio := dipSpec.Meta.SingletTripletRatio
	if dipRatio == 0 {
		dipRatio = 1
	}

	// Panel (a): SIP sticks regrouped by symmetry. SIP states are doublets, so
	// singlet:triplet weighting does not apply (ratio 1, spinWeight off).
	// -sip-group (bare) opens an interactive dialogue that lists every MO and
	// lets the user assign them to named groups, appending the chosen specs to
	// any passed non-interactively. Needs a terminal; otherwise it is skipped.
	sipSpecs := p.sipGroups
	if p.sipGroupI {
		if isInteractive(os.Stdin) {
			sipSpecs = append(sipSpecs, promptSIPGrouping(os.Stderr, bufio.NewScanner(os.Stdin), distinctSIPMOs(sipSpec.Lines))...)
		} else {
			fmt.Fprintln(os.Stderr, "plotspec: -sip-group dialogue needs a terminal; skipping (use -sip-group=SPEC instead)")
		}
	}
	groupOrder, moToLabel, symToLabel, sipColors, err := parseSIPGroups(sipSpecs)
	if err != nil {
		return err
	}
	aLines, aOrder := groupSIPByGroup(sipSpec.Lines, groupOrder, moToLabel, symToLabel)
	// Panel (a)'s symmetry groups are unrelated to the decay channels, so colour
	// them from the generated colours (past the curated palette) to avoid implying
	// a shared meaning with (b)/(c), unless a group requested a custom colour via
	// -sip-group. It keeps its own legend.
	pa, err := buildStickPanel(aLines, aOrder, abGrid,
		ees.SigmaFromFWHM(fwhmSIP), false, 1, p.overlay, p.stickHeight, p.absolute, len(activePalette), true, sipColors)
	if err != nil {
		return err
	}
	pa.X.Min, pa.X.Max = abLo, abHi
	pa.Y.Label.Text = "(a) SIP"
	pa.HideX() // no ticks/label here so (a) and (b) share one axis with no gap

	// Panel (b): DIP sticks with the decay-channel colouring. Its legend is drawn
	// once as a shared key in the gap above (c), so this panel adds none itself.
	bOrder := channelOrder(*dipSpec)
	// Channel→colour map shared by (b) and (c): the channel's slot in the full DIP
	// order fixes its colour, so a channel keeps the same hue in both panels even
	// when (c) draws only a -fin-channels subset.
	dipColorIdx := map[string]int{}
	for i, ch := range bOrder {
		dipColorIdx[ch] = i
	}
	pb, err := buildStickPanel(dipSpec.Lines, bOrder, abGrid,
		ees.SigmaFromFWHM(fwhmDIP), p.spinWeight, dipRatio, p.overlay, p.stickHeight, p.absolute, 0, false, nil)
	if err != nil {
		return err
	}
	pb.X.Min, pb.X.Max = abLo, abHi
	pb.X.Label.Text = "E (eV)"
	pb.Y.Label.Text = "(b) DIP"

	// Panel (c): electron-emission spectrum on its own window (-xrange-c).
	res, err := eesElectronCurve(sipSpec, dipSpec, eesCurveParams{
		fwhm: p.fwhm, fwhmSIP: p.fwhmSIP, fwhmDIP: p.fwhmDIP, fwhmEES: p.fwhmEES,
		finChannels: p.finChannels, points: p.points, pad: p.pad,
		xRange: p.xRangeC, absolute: p.absolute, spinWeight: p.spinWeight,
	})
	if err != nil {
		return err
	}
	pc := plot.New()
	pc.X.Min, pc.X.Max = res.eLo, res.eHi
	pc.X.Label.Text = "E (eV)"
	pc.Y.Label.Text = "(c) Secondary EES"
	for _, ch := range res.order {
		cline, err := plotter.NewLine(asXYs(res.eGrid, res.sigmas[ch]))
		if err != nil {
			return err
		}
		cline.Color = palette(dipColorIdx[ch])
		cline.Width = vg.Points(1.5)
		pc.Add(cline)
	}

	// Compose the three panels onto one canvas.
	W := vg.Length(p.width) * vg.Inch
	H := vg.Length(p.height) * vg.Inch
	dc, save, err := newCanvas(p.out, W, H, p.dpi)
	if err != nil {
		return err
	}
	rect := dc.Rectangle
	height := rect.Max.Y - rect.Min.Y
	gap := 0.06 * height // vertical gap between the (a)/(b) block and (c)
	cTop := rect.Min.Y + 0.34*height

	// Top region: panels (a) and (b), aligned and abutting (PadY 0).
	abCanvas := draw.Canvas{Canvas: dc.Canvas, Rectangle: vg.Rectangle{
		Min: vg.Point{X: rect.Min.X, Y: cTop + gap},
		Max: vg.Point{X: rect.Max.X, Y: rect.Max.Y},
	}}
	tiles := draw.Tiles{Rows: 2, Cols: 1, PadY: 0}
	canvases := plot.Align([][]*plot.Plot{{pa}, {pb}}, tiles, abCanvas)
	pa.Draw(canvases[0][0])
	pb.Draw(canvases[1][0])

	// Shared (b)/(c) key in the gap between the two blocks: one entry per DIP
	// channel in the colour both panels use.
	keyCanvas := draw.Canvas{Canvas: dc.Canvas, Rectangle: vg.Rectangle{
		Min: vg.Point{X: rect.Min.X, Y: cTop},
		Max: vg.Point{X: rect.Max.X, Y: cTop + gap},
	}}
	drawChannelKey(keyCanvas, bOrder, func(ch string) color.Color { return palette(dipColorIdx[ch]) })

	// Bottom region: panel (c), narrower and horizontally inset.
	cCanvas := draw.Canvas{Canvas: dc.Canvas, Rectangle: vg.Rectangle{
		Min: vg.Point{X: rect.Min.X, Y: rect.Min.Y},
		Max: vg.Point{X: rect.Max.X, Y: cTop},
	}}
	pc.Draw(cCanvas)

	if err := save(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "plotspec: wrote %s (3-panel figure; SIP fwhm %.2g, DIP fwhm %.2g)\n",
		p.out, fwhmSIP, fwhmDIP)
	return nil
}

// buildStickPanel constructs a stick-spectrum plot for lines grouped by the
// channel labels in order, each in its palette colour. Heights are normalised so
// the tallest stick is 1 (unless absolute), then scaled by stickHeight so the
// caller can keep the sticks below the overlay envelope. When overlay is set, a
// single thin Gaussian-broadened envelope of the *total* spectrum (summed over
// every channel/group, normalised to peak 1) is drawn over the sticks — one
// curve, not one per grouping. colorBase offsets the per-channel palette index,
// so a panel whose channels are unrelated to the decay-channel colouring (e.g.
// panel (a)'s symmetry groups) can skip past the curated palette slots into the
// generated colours. customColors overrides the palette colour for any channel
// present in it (e.g. -sip-group's #RRGGBB). legend controls whether this panel
// draws its own legend (panels (b)/(c) share one key instead).
func buildStickPanel(lines []specLine, order []string, grid []float64, sigma float64,
	spinWeight bool, ratio float64, overlay bool, stickHeight float64, absolute bool, colorBase int, legend bool,
	customColors map[string]color.Color) (*plot.Plot, error) {
	p := plot.New()
	p.Legend.Top = true

	colorIdx := map[string]int{}
	for i, ch := range order {
		colorIdx[ch] = i
	}
	colorOf := func(ch string) color.Color {
		if c, ok := customColors[ch]; ok {
			return c
		}
		return palette(colorBase + colorIdx[ch])
	}

	if stickHeight <= 0 {
		stickHeight = 1
	}
	sticks := stickHeights(lines, order, spinWeight, ratio)
	norm := 1.0
	if !absolute {
		// Only the sticks inside the panel's window (the grid's span) set the
		// scale, so a taller line outside it cannot squash the visible ones.
		if m := maxXYIn(sticks, grid[0], grid[len(grid)-1]); m > 0 {
			norm = m
		}
	}
	stem := newStemPlot(vg.Points(2), grid[0], grid[len(grid)-1])
	for _, ch := range order {
		col := colorOf(ch)
		for _, pt := range sticks[ch] {
			stem.sticks = append(stem.sticks, stem3{x: pt.X, y: pt.Y / norm * stickHeight, color: col})
		}
		if legend && len(sticks[ch]) > 0 {
			p.Legend.Add(ch, stemThumb{color: col, width: vg.Points(2)})
		}
	}
	p.Add(stem)

	if overlay {
		curves := broaden(lines, order, grid, sigma, spinWeight, ratio)
		// Total envelope: sum every channel/group so the overlay shows the whole
		// spectrum as one curve rather than one per grouping.
		total := make([]float64, len(grid))
		for _, ch := range order {
			for i, y := range curves[ch] {
				total[i] += y
			}
		}
		if !absolute {
			if m := maxSlice(total); m > 0 {
				for i := range total {
					total[i] /= m
				}
			}
		}
		line, err := plotter.NewLine(asXYs(grid, total))
		if err != nil {
			return nil, err
		}
		line.Color = overlayColor
		line.Width = vg.Points(1.5)
		p.Add(line)
	}
	return p, nil
}

// drawChannelKey draws a single horizontal legend, shared by panels (b) and (c),
// centred in canvas c: one colour swatch plus channel label per entry, laid out
// left-to-right. colorFor maps each channel to the hue both panels use, so the one
// key in the gap between them describes both.
func drawChannelKey(c draw.Canvas, order []string, colorFor func(string) color.Color) {
	if len(order) == 0 {
		return
	}
	sty := plot.NewLegend().TextStyle
	sty.YAlign = text.YCenter
	const (
		thumb vg.Length = 18 // colour swatch width
		tgap  vg.Length = 4  // swatch→label spacing
		egap  vg.Length = 16 // entry→entry spacing
	)
	// Total row width, so the key can be centred horizontally.
	var total vg.Length
	for i, ch := range order {
		total += thumb + tgap + sty.Width(ch)
		if i != 0 {
			total += egap
		}
	}
	y := (c.Min.Y + c.Max.Y) / 2
	x := c.Min.X + (c.Max.X-c.Min.X-total)/2
	for _, ch := range order {
		ls := draw.LineStyle{Color: colorFor(ch), Width: vg.Points(2)}
		c.StrokeLine2(ls, x, y, x+thumb, y)
		x += thumb + tgap
		c.FillText(sty, vg.Point{X: x, Y: y}, ch)
		x += sty.Width(ch) + egap
	}
}

// sipGroupFlag is the -sip-group flag. It is bool-like (IsBoolFlag), so a bare
// "-sip-group" (Set called with "true") requests the interactive grouping
// dialogue, while "-sip-group=SPEC" appends a group spec. Repeatable.
type sipGroupFlag struct {
	specs       *multiFlag
	interactive *bool
}

func (f sipGroupFlag) String() string   { return "" }
func (f sipGroupFlag) IsBoolFlag() bool { return true }
func (f sipGroupFlag) Set(v string) error {
	if v == "true" { // bare -sip-group
		*f.interactive = true
		return nil
	}
	return f.specs.Set(v)
}

// moInfo is one SIP molecular orbital: its MO index and symmetry.
type moInfo struct{ mo, sym int }

// sipMOSym extracts a SIP line's MO index and symmetry from its "MO N (sym S)"
// channel label. The symmetry is the SCF MO-table value carried in that label —
// the same value the stick-spectrum legend ("the plotting key") displays — so
// grouping stays consistent with it. Note this MO-table sym can differ from the
// ADC `irrep` block number for the same line; the label's value is the one used
// for display and grouping. ok is false when the line is not a "MO N (sym S)"
// SIP channel.
func sipMOSym(l specLine) (mo, sym int, ok bool) {
	if _, err := fmt.Sscanf(l.Channel, "MO %d (sym %d)", &mo, &sym); err != nil {
		return 0, 0, false
	}
	return mo, sym, true
}

// distinctSIPMOs returns the distinct MOs found in the SIP lines, sorted by MO
// number, each tagged with its authoritative symmetry (see sipMOSym). Lines that
// are not "MO N (sym S)" SIP channels are ignored.
func distinctSIPMOs(lines []specLine) []moInfo {
	seen := map[int]moInfo{}
	for _, l := range lines {
		if mo, sym, ok := sipMOSym(l); ok {
			seen[mo] = moInfo{mo: mo, sym: sym}
		}
	}
	out := make([]moInfo, 0, len(seen))
	for _, mi := range seen {
		out = append(out, mi)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].mo < out[j].mo })
	return out
}

// parseSIPGroups parses -sip-group specs ("label=tok[,tok]") into the ordered
// list of group labels, an MO→label lookup and a sym→label lookup. Each token is
// either a bare MO number ("12") or a whole symmetry ("sym3"). An MO or a
// symmetry may appear in only one group. A label may carry an optional "#RRGGBB"
// (or "#RGB") suffix requesting a custom colour for that group; those are
// returned in customColors keyed by the (colour-stripped) label. A group without
// a suffix keeps its auto-generated colour.
func parseSIPGroups(specs []string) (order []string, moToLabel, symToLabel map[int]string, customColors map[string]color.Color, err error) {
	moToLabel = map[int]string{}
	symToLabel = map[int]string{}
	customColors = map[string]color.Color{}
	for _, spec := range specs {
		eq := strings.IndexByte(spec, '=')
		if eq <= 0 {
			return nil, nil, nil, nil, fmt.Errorf("invalid -sip-group %q: want LABEL[#RRGGBB]=MO[,MO,symN...]", spec)
		}
		label := strings.TrimSpace(spec[:eq])
		// Optional "#RRGGBB" custom-colour suffix on the label.
		if h := strings.IndexByte(label, '#'); h >= 0 {
			col, cerr := parseHexColor(label[h+1:])
			if cerr != nil {
				return nil, nil, nil, nil, fmt.Errorf("invalid -sip-group %q: %w", spec, cerr)
			}
			label = strings.TrimSpace(label[:h])
			customColors[label] = col
		}
		if label == "" {
			return nil, nil, nil, nil, fmt.Errorf("invalid -sip-group %q: empty label", spec)
		}
		order = append(order, label)
		for _, tok := range strings.Split(spec[eq+1:], ",") {
			tok = strings.TrimSpace(tok)
			if tok == "" {
				continue
			}
			if rest, isSym := strings.CutPrefix(tok, "sym"); isSym {
				s, cerr := strconv.Atoi(strings.TrimSpace(rest))
				if cerr != nil {
					return nil, nil, nil, nil, fmt.Errorf("invalid -sip-group %q: bad symmetry %q", spec, tok)
				}
				if prev, ok := symToLabel[s]; ok {
					return nil, nil, nil, nil, fmt.Errorf("-sip-group: symmetry %d assigned to both %q and %q", s, prev, label)
				}
				symToLabel[s] = label
				continue
			}
			m, cerr := strconv.Atoi(tok)
			if cerr != nil {
				return nil, nil, nil, nil, fmt.Errorf("invalid -sip-group %q: bad MO %q (want a number or symN)", spec, tok)
			}
			if prev, ok := moToLabel[m]; ok {
				return nil, nil, nil, nil, fmt.Errorf("-sip-group: MO %d assigned to both %q and %q", m, prev, label)
			}
			moToLabel[m] = label
		}
	}
	return order, moToLabel, symToLabel, customColors, nil
}

// promptSIPGrouping interactively collects panel (a) MO groupings from the user,
// mirroring ADCanalysis's decay-site prompt. It lists every MO (with its
// symmetry) and reads "NAME = MO, MO, symN ..." lines until an empty line. It
// returns the accepted definitions as -sip-group spec strings (so the caller
// parses them through the same path as the flag). Ungrouped MOs are left for
// groupSIPByGroup to fold into their default "sym S" group.
func promptSIPGrouping(w io.Writer, in *bufio.Scanner, mos []moInfo) []string {
	if len(mos) == 0 {
		return nil
	}
	syms := map[int]bool{}
	valid := map[int]bool{}
	var list []string
	for _, mi := range mos {
		valid[mi.mo] = true
		syms[mi.sym] = true
		list = append(list, fmt.Sprintf("MO %d (sym %d)", mi.mo, mi.sym))
	}
	fmt.Fprintf(w, "\nAvailable MOs (panel a): %s\n", strings.Join(list, ", "))
	fmt.Fprintln(w, "Group them as  NAME = MO, MO, ...  (or symN for a whole symmetry)")
	fmt.Fprintln(w, "Append #RRGGBB to NAME for a custom colour, e.g.  core#e41a1c = 1, 2")
	fmt.Fprintln(w, "Press Enter on an empty line to finish; ungrouped MOs stay in their symmetry group.")

	used := map[int]bool{}     // MOs already assigned
	usedSym := map[int]bool{}  // whole symmetries already assigned
	names := map[string]bool{} // labels already used (colour-stripped)
	var specs []string
	for {
		fmt.Fprint(w, "group> ")
		if !in.Scan() {
			break
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			break
		}
		rawName, mem, ok := strings.Cut(line, "=")
		rawName = strings.TrimSpace(rawName)
		name := rawName
		if h := strings.IndexByte(name, '#'); h >= 0 {
			name = strings.TrimSpace(name[:h])
		}
		if !ok || name == "" {
			fmt.Fprintln(w, "  format: NAME = MO, MO, ...")
			continue
		}
		if names[name] {
			fmt.Fprintf(w, "  group %q already defined\n", name)
			continue
		}
		var toks []string
		bad := false
		for _, t := range strings.Split(mem, ",") {
			t = strings.TrimSpace(t)
			if t == "" {
				continue
			}
			if rest, isSym := strings.CutPrefix(t, "sym"); isSym {
				s, err := strconv.Atoi(strings.TrimSpace(rest))
				switch {
				case err != nil || !syms[s]:
					fmt.Fprintf(w, "  unknown symmetry %q\n", t)
					bad = true
				case usedSym[s]:
					fmt.Fprintf(w, "  symmetry %d already grouped\n", s)
					bad = true
				default:
					usedSym[s] = true
					toks = append(toks, fmt.Sprintf("sym%d", s))
				}
			} else {
				m, err := strconv.Atoi(t)
				switch {
				case err != nil || !valid[m]:
					fmt.Fprintf(w, "  unknown MO %q\n", t)
					bad = true
				case used[m]:
					fmt.Fprintf(w, "  MO %d already grouped\n", m)
					bad = true
				default:
					used[m] = true
					toks = append(toks, strconv.Itoa(m))
				}
			}
			if bad {
				break
			}
		}
		if bad || len(toks) == 0 {
			continue
		}
		names[name] = true
		specs = append(specs, rawName+"="+strings.Join(toks, ","))
		fmt.Fprintf(w, "  group %q = %s\n", name, strings.Join(toks, ", "))
	}
	return specs
}

// parseHexColor parses a hex colour — "RRGGBB" or shorthand "RGB", with an
// optional leading '#' — into an opaque RGBA.
func parseHexColor(s string) (color.RGBA, error) {
	s = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(s), "#"))
	v, err := strconv.ParseUint(s, 16, 32)
	if err != nil {
		return color.RGBA{}, fmt.Errorf("bad hex colour %q", s)
	}
	switch len(s) {
	case 3: // RGB shorthand: each nibble is doubled (f -> ff).
		return color.RGBA{
			R: uint8((v>>8)&0xf) * 0x11,
			G: uint8((v>>4)&0xf) * 0x11,
			B: uint8(v&0xf) * 0x11,
			A: 0xff,
		}, nil
	case 6:
		return color.RGBA{
			R: uint8(v >> 16),
			G: uint8(v >> 8),
			B: uint8(v),
			A: 0xff,
		}, nil
	default:
		return color.RGBA{}, fmt.Errorf("bad hex colour %q: want RGB or RRGGBB", s)
	}
}

// groupSIPByGroup rewrites each SIP line's channel ("MO N (sym S)") to its group
// label: an explicit MO group (moToLabel[N]) wins, else an explicit symmetry
// group (symToLabel[S]), else the MO falls into a default "sym S" group. The
// returned order lists the explicit groups first (in spec order), then the
// default "sym S" groups that actually received a line, by ascending symmetry.
func groupSIPByGroup(lines []specLine, groupOrder []string, moToLabel, symToLabel map[int]string) ([]specLine, []string) {
	out := make([]specLine, len(lines))
	seen := map[string]bool{}
	var order []string
	for _, l := range groupOrder {
		if !seen[l] {
			seen[l] = true
			order = append(order, l)
		}
	}
	var defaultSyms []int
	defaultSeen := map[int]bool{}
	for i, l := range lines {
		label := l.Channel
		if mo, sym, ok := sipMOSym(l); ok {
			switch {
			case moToLabel[mo] != "":
				label = moToLabel[mo]
			case symToLabel[sym] != "":
				label = symToLabel[sym]
			default:
				label = fmt.Sprintf("sym %d", sym)
				if !defaultSeen[sym] {
					defaultSeen[sym] = true
					defaultSyms = append(defaultSyms, sym)
				}
			}
		}
		out[i] = l
		out[i].Channel = label
	}
	sort.Ints(defaultSyms)
	for _, s := range defaultSyms {
		label := fmt.Sprintf("sym %d", s)
		if !seen[label] {
			seen[label] = true
			order = append(order, label)
		}
	}
	return out, order
}

// isInteractive reports whether f is a terminal, so the -sip-group dialogue only
// prompts when there is a user to answer it.
func isInteractive(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

// newCanvas creates a drawing canvas for out (by extension) and returns it with a
// save closure that writes the rendered canvas to out. PNG/JPEG honour dpi;
// SVG/PDF are vector. It is the multi-plot analogue of savePlot.
func newCanvas(out string, w, h vg.Length, dpi int) (draw.Canvas, func() error, error) {
	ext := strings.ToLower(out)
	switch {
	case strings.HasSuffix(ext, ".png") || strings.HasSuffix(ext, ".jpg") || strings.HasSuffix(ext, ".jpeg"):
		c := vgimg.NewWith(vgimg.UseDPI(dpi), vgimg.UseWH(w, h))
		save := func() error {
			f, err := os.Create(out)
			if err != nil {
				return fmt.Errorf("create %s: %w", out, err)
			}
			defer f.Close()
			if err := png.Encode(f, c.Image()); err != nil {
				return fmt.Errorf("write %s: %w", out, err)
			}
			return nil
		}
		return draw.New(c), save, nil
	case strings.HasSuffix(ext, ".svg"):
		c := vgsvg.New(w, h)
		save := func() error {
			f, err := os.Create(out)
			if err != nil {
				return fmt.Errorf("create %s: %w", out, err)
			}
			defer f.Close()
			if _, err := c.WriteTo(f); err != nil {
				return fmt.Errorf("write %s: %w", out, err)
			}
			return nil
		}
		return draw.New(c), save, nil
	case strings.HasSuffix(ext, ".pdf"):
		c := vgpdf.New(w, h)
		save := func() error {
			f, err := os.Create(out)
			if err != nil {
				return fmt.Errorf("create %s: %w", out, err)
			}
			defer f.Close()
			if _, err := c.WriteTo(f); err != nil {
				return fmt.Errorf("write %s: %w", out, err)
			}
			return nil
		}
		return draw.New(c), save, nil
	default:
		return draw.Canvas{}, nil, fmt.Errorf("unsupported output extension for %s (use .png/.svg/.pdf)", out)
	}
}

// eesCurveParams bundles the inputs to eesElectronCurve.
type eesCurveParams struct {
	fwhm, fwhmSIP, fwhmDIP, fwhmEES float64
	finChannels                     string
	points                          int
	pad                             float64
	xRange                          string
	absolute, spinWeight            bool
}

// eesResult is the output of eesElectronCurve: the electron kinetic-energy grid,
// one (optionally normalised) partial-intensity curve per decay channel (sigmas,
// keyed by channel; order gives the canonical drawing order), the axis bounds,
// and the resolved SIP/DIP FWHMs (after defaulting to -fwhm).
type eesResult struct {
	eGrid            []float64
	sigmas           map[string][]float64
	order            []string
	eLo, eHi         float64
	fwhmSIP, fwhmDIP float64
}

// eesElectronCurve computes the electron-emission spectrum σ(ε)=∫ S_in(E) S_fin(E−ε)/N(E) dE
// from a SIP and DIP spectrum. It is the shared numeric core of -mode ees and the
// panel mode's panel (c). See runEES for the physics commentary.
func eesElectronCurve(sipSpec, dipSpec *spectrum, p eesCurveParams) (eesResult, error) {
	var res eesResult
	if len(sipSpec.Lines) == 0 || len(dipSpec.Lines) == 0 {
		return res, fmt.Errorf("electron-emission spectrum needs non-empty SIP and DIP spectra")
	}

	fwhmSIP, fwhmDIP := p.fwhmSIP, p.fwhmDIP
	if fwhmSIP <= 0 {
		fwhmSIP = p.fwhm
	}
	if fwhmDIP <= 0 {
		fwhmDIP = p.fwhm
	}
	// -fwhm-ees, when set, overrides both envelope widths used inside the
	// electron-spectrum convolution. This decouples the EES broadening from the
	// panel (a)/(b) total-envelope overlays, which keep -fwhm-sip/-fwhm-dip.
	if p.fwhmEES > 0 {
		fwhmSIP, fwhmDIP = p.fwhmEES, p.fwhmEES
	}
	if fwhmSIP <= 0 || fwhmDIP <= 0 {
		return res, fmt.Errorf("fwhm must be > 0")
	}

	sipLines := toEESLines(sipSpec.Lines)
	dipLines := toEESLines(dipSpec.Lines)

	// Shared energy grid spanning both spectra ±pad; starting below the lowest
	// DIP state so the cumulative N(E) effectively integrates from threshold.
	lo, hi := sipLines[0].Energy, sipLines[0].Energy
	for _, l := range append(append([]ees.Line{}, sipLines...), dipLines...) {
		lo = math.Min(lo, l.Energy)
		hi = math.Max(hi, l.Energy)
	}
	lo -= p.pad
	hi += p.pad
	grid := linspace(lo, hi, p.points)

	ratio := dipSpec.Meta.SingletTripletRatio
	if ratio == 0 {
		ratio = 1
	}

	sIn := ees.Envelope(sipLines, grid, ees.EnvelopeOptions{Sigma: ees.SigmaFromFWHM(fwhmSIP)})
	dipOpt := ees.EnvelopeOptions{Sigma: ees.SigmaFromFWHM(fwhmDIP), SpinWeight: p.spinWeight, Ratio: ratio}
	sFinTot := ees.Envelope(dipLines, grid, dipOpt)

	// Final decay channels to split the electron spectrum into, one curve each
	// (canonical DIP order). -fin-channels restricts which channels are drawn
	// (comma-separated names or prefixes); empty = all.
	order := channelOrder(*dipSpec)
	if sel := matchChannels(p.finChannels, dipLines); len(sel) > 0 {
		var kept []string
		for _, ch := range order {
			if sel[ch] {
				kept = append(kept, ch)
			}
		}
		order = kept
	}
	if len(order) == 0 {
		return res, fmt.Errorf("no DIP channels to plot (check -fin-channels)")
	}

	// Double-ionization onset: the lowest final-state energy. Only intermediate
	// SIP states above this can decay by electron emission (open dicationic channel);
	// lower-lying outer-valence ionizations have none and must not contribute.
	dipThreshold := minEnergy(dipLines)

	// Electron kinetic-energy grid: 0 .. (max E_in − min E_fin), or -xrange.
	eLo, eHi := 0.0, maxEnergy(sipLines)-minEnergy(dipLines)
	if eHi <= 0 {
		eHi = maxEnergy(sipLines)
	}
	if p.xRange != "" {
		var err error
		eLo, eHi, err = parseRange(p.xRange)
		if err != nil {
			return res, err
		}
	}
	eGrid := linspace(eLo, eHi, p.points)

	// One partial electron spectrum per channel: its own broadened envelope is
	// the numerator, while N(E) stays the *total* DIP envelope so each channel
	// keeps its correct branching weight (the -fin-channels computation, run once
	// per channel).
	sigmas := make(map[string][]float64, len(order))
	for _, ch := range order {
		numOpt := dipOpt
		numOpt.Channels = map[string]bool{ch: true}
		sFinNum := ees.Envelope(dipLines, grid, numOpt)
		sigmas[ch] = ees.ElectronSpectrum(grid, sIn, sFinNum, sFinTot, eGrid, dipThreshold)
	}

	// Relative normalisation shares one divisor across channels (tallest peak of
	// any channel = 1) so the channels keep their correct heights relative to one
	// another, unless -absolute is requested.
	if !p.absolute {
		norm := 0.0
		for _, s := range sigmas {
			if m := maxSlice(s); m > norm {
				norm = m
			}
		}
		if norm > 0 {
			for _, s := range sigmas {
				for i := range s {
					s[i] /= norm
				}
			}
		}
	}

	res = eesResult{eGrid: eGrid, sigmas: sigmas, order: order, eLo: eLo, eHi: eHi, fwhmSIP: fwhmSIP, fwhmDIP: fwhmDIP}
	return res, nil
}

// toEESLines converts the decoded JSON lines into the spectrum package's Line type.
func toEESLines(lines []specLine) []ees.Line {
	out := make([]ees.Line, len(lines))
	for i, l := range lines {
		out[i] = ees.Line{Energy: l.Energy, Intensity: l.Intensity, Channel: l.Channel, Spin: l.Spin}
	}
	return out
}

// matchChannels turns the -fin-channels spec (comma-separated names or prefixes)
// into the set of DIP channel names to keep. An empty spec returns nil (all).
func matchChannels(spec string, lines []ees.Line) map[string]bool {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil
	}
	var tokens []string
	for _, t := range strings.Split(spec, ",") {
		if t = strings.TrimSpace(t); t != "" {
			tokens = append(tokens, t)
		}
	}
	sel := map[string]bool{}
	for _, l := range lines {
		for _, t := range tokens {
			if l.Channel == t || strings.HasPrefix(l.Channel, t) {
				sel[l.Channel] = true
				break
			}
		}
	}
	return sel
}

// linspace returns n points evenly spaced over [lo, hi].
func linspace(lo, hi float64, n int) []float64 {
	if n < 2 {
		return []float64{lo}
	}
	g := make([]float64, n)
	for i := range g {
		g[i] = lo + (hi-lo)*float64(i)/float64(n-1)
	}
	return g
}

func maxSlice(xs []float64) float64 {
	m := 0.0
	for _, x := range xs {
		if x > m {
			m = x
		}
	}
	return m
}

func maxEnergy(lines []ees.Line) float64 {
	m := math.Inf(-1)
	for _, l := range lines {
		m = math.Max(m, l.Energy)
	}
	return m
}

func minEnergy(lines []ees.Line) float64 {
	m := math.Inf(1)
	for _, l := range lines {
		m = math.Min(m, l.Energy)
	}
	return m
}

// broaden sums Gaussian-broadened sticks for each channel in order onto grid.
// When spinWeight is true, singlet sticks (Spin == 1) are scaled by ratio.
// Lines whose channel is not in order are skipped (channelOrder guarantees all
// present channels are listed, so this only guards against malformed input).
func broaden(lines []specLine, order []string, grid []float64, sigma float64,
	spinWeight bool, ratio float64) map[string][]float64 {
	curves := make(map[string][]float64, len(order))
	for _, ch := range order {
		curves[ch] = make([]float64, len(grid))
	}
	twoSigma2 := 2 * sigma * sigma
	for _, l := range lines {
		dst, ok := curves[l.Channel]
		if !ok {
			continue
		}
		w := l.Intensity
		if spinWeight && l.Spin == 1 {
			w *= ratio
		}
		for i, e := range grid {
			d := e - l.Energy
			dst[i] += w * math.Exp(-d*d/twoSigma2)
		}
	}
	return curves
}

// maxCurve returns the largest value across all channel curves (0 if empty).
func maxCurve(curves map[string][]float64) float64 {
	max := 0.0
	for _, ys := range curves {
		for _, y := range ys {
			if y > max {
				max = y
			}
		}
	}
	return max
}

// channelOrder returns the channels to plot: meta.channels first (canonical
// order), then any extra channel names found only in the lines, sorted.
func channelOrder(spec spectrum) []string {
	seen := map[string]bool{}
	var order []string
	for _, ch := range spec.Channels {
		if !seen[ch] {
			seen[ch] = true
			order = append(order, ch)
		}
	}
	var extra []string
	for _, l := range spec.Lines {
		if !seen[l.Channel] {
			seen[l.Channel] = true
			extra = append(extra, l.Channel)
		}
	}
	sort.Strings(extra)
	return append(order, extra...)
}

// parseRange parses an X-axis window like "30-100" or "30.5-100" into (lo, hi).
// A leading minus on the low bound (e.g. "-5-10") is supported by splitting on
// the last '-' that is not at the start.
func parseRange(s string) (float64, float64, error) {
	s = strings.TrimSpace(s)
	sep := strings.LastIndex(s, "-")
	if sep <= 0 {
		return 0, 0, fmt.Errorf("invalid -xrange %q: want LO-HI, e.g. 30-100", s)
	}
	lo, err := strconv.ParseFloat(strings.TrimSpace(s[:sep]), 64)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid -xrange %q: %w", s, err)
	}
	hi, err := strconv.ParseFloat(strings.TrimSpace(s[sep+1:]), 64)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid -xrange %q: %w", s, err)
	}
	if lo >= hi {
		return 0, 0, fmt.Errorf("invalid -xrange %q: LO must be < HI", s)
	}
	return lo, hi, nil
}

// stem3 is one stick: a vertical bar at energy x rising to height y, in its
// channel's colour (dashes set only for the experimental overlay).
type stem3 struct {
	x, y   float64
	color  color.Color
	dashes []vg.Length
}

// stemPlot draws a stick (stem) spectrum: a vertical line from the baseline
// (y=0) up to each stick's height, all at one fixed width. It implements
// plot.Plotter and plot.DataRanger; legend entries are added separately via
// stemThumb so each channel appears once.
//
// Sticks are drawn tallest first, so where several channels carry a stick at the
// same eigenvalue (same x and width) the shorter ones are painted last and stay
// visible instead of being hidden behind a taller stick of another channel.
//
// xlo/xhi bound the plotted window: sticks outside it are clipped away when
// drawn, and DataRange ignores them, so a tall line just off-screen cannot
// stretch the Y axis and flatten everything the window does show.
type stemPlot struct {
	sticks   []stem3
	width    vg.Length
	xlo, xhi float64
}

// newStemPlot returns an empty stick plotter drawing at the given width over the
// window [xlo, xhi].
func newStemPlot(width vg.Length, xlo, xhi float64) *stemPlot {
	return &stemPlot{width: width, xlo: xlo, xhi: xhi}
}

// inWindow reports whether stick st falls inside the plotted window.
func (s *stemPlot) inWindow(st stem3) bool { return st.x >= s.xlo && st.x <= s.xhi }

func (s *stemPlot) Plot(c draw.Canvas, plt *plot.Plot) {
	trX, trY := plt.Transforms(&c)
	y0 := trY(0)
	order := make([]int, len(s.sticks))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return s.sticks[order[a]].y > s.sticks[order[b]].y
	})
	for _, idx := range order {
		st := s.sticks[idx]
		if !s.inWindow(st) {
			continue
		}
		x := trX(st.x)
		sty := draw.LineStyle{Color: st.color, Width: s.width, Dashes: st.dashes}
		c.StrokeLine2(sty, x, y0, x, trY(st.y))
	}
}

// DataRange reports the span of the sticks inside the plotted window, pinning the
// lower Y bound to the baseline so the stems are always drawn from zero. With no
// stick in the window it falls back to the window itself, leaving the axes to any
// other plotter (e.g. an -overlay-broadened curve).
func (s *stemPlot) DataRange() (xmin, xmax, ymin, ymax float64) {
	xmin, ymin = math.Inf(1), 0
	xmax, ymax = math.Inf(-1), 0
	for _, st := range s.sticks {
		if !s.inWindow(st) {
			continue
		}
		xmin = math.Min(xmin, st.x)
		xmax = math.Max(xmax, st.x)
		ymax = math.Max(ymax, st.y)
	}
	if xmin > xmax {
		xmin, xmax = s.xlo, s.xhi
	}
	return
}

// stemThumb is a one-channel legend swatch: a short horizontal line in the
// channel's colour and stick width (dashed for the experimental overlay).
type stemThumb struct {
	color  color.Color
	width  vg.Length
	dashes []vg.Length
}

func (t stemThumb) Thumbnail(c *draw.Canvas) {
	sty := draw.LineStyle{Color: t.color, Width: t.width, Dashes: t.dashes}
	y := c.Center().Y
	c.StrokeLine2(sty, c.Min.X, y, c.Max.X, y)
}

// stickHeights groups stick intensities by channel as (energy, height) points,
// scaling singlet sticks by ratio when spinWeight is set. Lines whose channel is
// not in order are skipped (channelOrder lists every present channel).
func stickHeights(lines []specLine, order []string, spinWeight bool, ratio float64) map[string]plotter.XYs {
	want := make(map[string]bool, len(order))
	for _, ch := range order {
		want[ch] = true
	}
	m := make(map[string]plotter.XYs, len(order))
	for _, l := range lines {
		if !want[l.Channel] {
			continue
		}
		w := l.Intensity
		if spinWeight && l.Spin == 1 {
			w *= ratio
		}
		m[l.Channel] = append(m[l.Channel], plotter.XY{X: l.Energy, Y: w})
	}
	return m
}

// maxXYIn returns the largest Y across all channels' sticks whose energy lies in
// [lo, hi], i.e. the tallest stick actually drawn in the plotted window (0 if
// none).
func maxXYIn(sticks map[string]plotter.XYs, lo, hi float64) float64 {
	max := 0.0
	for _, xys := range sticks {
		for _, pt := range xys {
			if pt.X >= lo && pt.X <= hi && pt.Y > max {
				max = pt.Y
			}
		}
	}
	return max
}

func asXYs(x, y []float64) plotter.XYs {
	pts := make(plotter.XYs, len(x))
	for i := range x {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}
	return pts
}

// tableauColors is the default 6-colour palette (Tableau/matplotlib-inspired).
var tableauColors = []color.Color{
	color.RGBA{R: 0xff, G: 0x7f, B: 0x0e, A: 0xff}, // orange
	color.RGBA{R: 0x1f, G: 0x77, B: 0xb4, A: 0xff}, // blue
	color.RGBA{R: 0x2c, G: 0xa0, B: 0x2c, A: 0xff}, // green
	color.RGBA{R: 0x94, G: 0x67, B: 0xbd, A: 0xff}, // purple
	color.RGBA{R: 0x8c, G: 0x56, B: 0x4b, A: 0xff}, // brown
	color.RGBA{R: 0xd6, G: 0x27, B: 0x28, A: 0xff}, // red
}

// okabeItoColors is the Okabe–Ito colour-blind-safe palette (the black entry is
// dropped so every channel keeps a distinct hue against the white background).
var okabeItoColors = []color.Color{
	color.RGBA{R: 0xe6, G: 0x9f, B: 0x00, A: 0xff}, // orange
	color.RGBA{R: 0x56, G: 0xb4, B: 0xe9, A: 0xff}, // sky blue
	color.RGBA{R: 0x00, G: 0x9e, B: 0x73, A: 0xff}, // bluish green
	color.RGBA{R: 0xf0, G: 0xe4, B: 0x42, A: 0xff}, // yellow
	color.RGBA{R: 0x00, G: 0x72, B: 0xb2, A: 0xff}, // blue
	color.RGBA{R: 0xd5, G: 0x5e, B: 0x00, A: 0xff}, // vermillion
	color.RGBA{R: 0xcc, G: 0x79, B: 0xa7, A: 0xff}, // reddish purple
}

// activePalette is the palette palette() indexes into; -colorblind swaps it for
// okabeItoColors at the start of each run path.
var activePalette = tableauColors

// overlayColor is the neutral hue for the -overlay-broadened total envelope. It
// is deliberately channel-agnostic (a dark grey) so the single summed curve does
// not read as belonging to any one channel/group.
var overlayColor color.Color = color.RGBA{R: 0x80, G: 0x80, B: 0x80, A: 0xff}

// palette returns a distinguishable colour for channel i. The first
// len(activePalette) channels take the curated palette (Tableau, or Okabe–Ito
// under -colorblind); any further channel gets an algorithmically generated
// colour (genColor) so the palette never runs out and never repeats a curated
// hue.
func palette(i int) color.Color {
	if i < len(activePalette) {
		return activePalette[i]
	}
	return genColor(i - len(activePalette))
}

// genColor synthesises distinct colours for index n = 0, 1, 2, … by walking the
// hue circle in golden-angle (~137.5°) steps so successive colours are spread as
// far apart as possible, while cycling saturation/value so even same-hue repeats
// (after many channels) stay distinguishable.
func genColor(n int) color.Color {
	const golden = 0.61803398875 // golden-ratio conjugate, in turns
	hue := math.Mod(float64(n)*golden, 1)
	sat := 0.65 + 0.20*float64(n%2)     // alternate 0.65 / 0.85
	val := 0.95 - 0.15*float64((n/2)%2) // alternate 0.95 / 0.80
	r, g, b := hsvToRGB(hue, sat, val)
	return color.RGBA{R: r, G: g, B: b, A: 0xff}
}

// hsvToRGB converts an HSV colour (h, s, v each in [0,1]) to 8-bit RGB.
func hsvToRGB(h, s, v float64) (uint8, uint8, uint8) {
	i := math.Floor(h * 6)
	f := h*6 - i
	p := v * (1 - s)
	q := v * (1 - f*s)
	t := v * (1 - (1-f)*s)
	var r, g, b float64
	switch int(i) % 6 {
	case 0:
		r, g, b = v, t, p
	case 1:
		r, g, b = q, v, p
	case 2:
		r, g, b = p, v, t
	case 3:
		r, g, b = p, q, v
	case 4:
		r, g, b = t, p, v
	case 5:
		r, g, b = v, p, q
	}
	return uint8(r * 255), uint8(g * 255), uint8(b * 255)
}
