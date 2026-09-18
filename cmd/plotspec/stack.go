// Energy-offset waterfall ("stack") mode: one broadened spectrum per conformer,
// each drawn on a baseline set by that conformer's relative electronic energy.
//
// A structure search returns a basin per starting guess, not a spectrum per
// paper figure: twenty hydrogen-bonded minima cannot each have a three-panel
// figure, and a table of them is not read. Offsetting each conformer's envelope
// by its relative energy -- the stacked convention of an NMR/IR series, where
// the offset is usually time or temperature -- puts the whole search on one
// axis pair: horizontal is the spectrum, vertical is stability.
//
// Each trace is named on the offset axis, under its energy, and carries a small
// skeletal drawing of its own geometry in the corner (see geom.go) -- a name
// like "O2-N1H" only means something to a reader holding the site-numbering
// convention in their head.
//
// Only the total envelope is drawn per conformer. Twenty channel-resolved
// spectra overlaid would be unreadable at this scale, and the channel
// decomposition is already carried by the single-conformer panels; the
// waterfall's job is conformer-to-conformer shape variability.
package main

import (
	"encoding/json"
	"fmt"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Energy conversions from hartree, keyed by the unit names the flags accept.
var stackUnits = map[string]struct {
	perHartree float64
	label      string
}{
	"hartree": {1, "hartree"},
	"au":      {1, "a.u."},
	"ev":      {27.211386245988, "eV"},
	"kcal":    {627.5094740631, "kcal/mol"},
	"kj":      {2625.4996394799, "kJ/mol"},
}

// Sequential single-hue ramp, dark = most stable. The light end stops short of
// the surface so the least-stable trace still reads on white paper; a
// sequential encoding is used rather than the categorical per-channel palette
// because the traces differ by a magnitude (relative energy), not by identity,
// and twenty categorical hues would be a legend explosion.
var stackRamp = []color.RGBA{
	{0x86, 0xb6, 0xef, 0xff}, // least stable
	{0x55, 0x98, 0xe7, 0xff},
	{0x39, 0x87, 0xe5, 0xff},
	{0x25, 0x6a, 0xbf, 0xff},
	{0x18, 0x4f, 0x95, 0xff},
	{0x0d, 0x36, 0x6b, 0xff}, // global minimum
}

var (
	stackInk   = color.RGBA{0x1a, 0x1a, 0x18, 0xff}
	stackMuted = color.RGBA{0x6b, 0x6b, 0x66, 0xff}
	stackRule  = color.RGBA{0xdc, 0xdc, 0xd6, 0xff}
	stackPaper = color.RGBA{0xff, 0xff, 0xff, 0xff}
)

// blendOnPaper returns col laid over the paper colour at the given opacity, as
// an opaque colour. Used instead of an alpha channel so every output backend
// renders the same fill.
func blendOnPaper(col color.RGBA, alpha float64) color.RGBA {
	mix := func(c, p uint8) uint8 {
		return uint8(math.Round(alpha*float64(c) + (1-alpha)*float64(p)))
	}
	return color.RGBA{
		mix(col.R, stackPaper.R),
		mix(col.G, stackPaper.G),
		mix(col.B, stackPaper.B),
		0xff,
	}
}

// stackColor samples the ramp at t in [0,1], 0 = most stable (darkest).
func stackColor(t float64) color.RGBA {
	t = math.Max(0, math.Min(1, t))
	// Walk the ramp from its dark end so t=0 lands on the global minimum.
	x := (1 - t) * float64(len(stackRamp)-1)
	i := int(x)
	if i >= len(stackRamp)-1 {
		return stackRamp[len(stackRamp)-1]
	}
	f := x - float64(i)
	a, b := stackRamp[i], stackRamp[i+1]
	mix := func(p, q uint8) uint8 { return uint8(math.Round(float64(p) + f*(float64(q)-float64(p)))) }
	return color.RGBA{mix(a.R, b.R), mix(a.G, b.G), mix(a.B, b.B), 0xff}
}

// stackTrim drops the longest prefix shared by every label and ending at a "_",
// so run names like UW1_O2-N1H / UW1_O4-N3H label their traces as O2-N1H /
// O4-N3H -- the hydration level is already the figure's title. A label is left
// alone if trimming would empty it.
func stackTrim(traces []stackTrace) {
	if len(traces) < 2 {
		return
	}
	pre := traces[0].label
	for _, t := range traces[1:] {
		for !strings.HasPrefix(t.label, pre) {
			pre = pre[:len(pre)-1]
			if pre == "" {
				return
			}
		}
	}
	i := strings.LastIndex(pre, "_")
	if i < 0 {
		return
	}
	pre = pre[:i+1]
	// All or nothing: trimming only the labels that survive it would leave the
	// set inconsistent, one trimmed and one not.
	for _, t := range traces {
		if strings.TrimPrefix(t.label, pre) == "" {
			return
		}
	}
	for i := range traces {
		traces[i].label = strings.TrimPrefix(traces[i].label, pre)
	}
}

// stackTrace is one conformer's contribution to the waterfall.
type stackTrace struct {
	label   string
	path    string
	offset  float64 // relative energy, display units
	aliases []string
	curve   []float64
	base    float64 // drawing baseline: offset, nudged clear of its neighbour
	depict  *depiction
}

// stackParams bundles the flags driving the waterfall mode.
type stackParams struct {
	entries      multiFlag // LABEL=PATH[@OFFSET], repeatable
	manifest     string
	specDir      string
	specSuffix   string
	energyKey    string
	groupKey     string
	group        string
	manifestUnit string
	offsetUnit   string
	out          string
	title        string
	fwhm         float64
	points       int
	pad          float64
	xRange       string
	norm         string
	traceScale   float64
	minSep       float64
	dedup        float64
	geomSuffix   string
	insetWidth   float64
	insetHeight  float64
	insetLift    float64
	width        float64
	height       float64
	dpi          int
}

// runStack renders the energy-offset waterfall.
func runStack(p stackParams) error {
	traces, err := stackCollect(p)
	if err != nil {
		return err
	}
	if len(traces) == 0 {
		return fmt.Errorf("-mode stack: no spectra selected")
	}
	// Ascending relative energy: index 0 is the global minimum.
	sort.SliceStable(traces, func(i, j int) bool { return traces[i].offset < traces[j].offset })

	sigma := p.fwhm / (2 * math.Sqrt(2*math.Ln2))
	if sigma <= 0 {
		return fmt.Errorf("fwhm must be > 0")
	}

	// Read every spectrum once: an automatic window needs all the sticks before
	// the grid exists, and these files run to tens of thousands of lines each.
	specs := make([]*spectrum, len(traces))
	for i := range traces {
		spec, err := readSpectrum(traces[i].path)
		if err != nil {
			return err
		}
		if len(spec.Lines) == 0 {
			return fmt.Errorf("%s contains no lines", traces[i].path)
		}
		// One series per conformer: the channel decomposition is deliberately
		// dropped here (see the package comment).
		collapseChannels(spec, "total")
		specs[i] = spec
	}

	lo, hi, err := stackWindow(p, specs)
	if err != nil {
		return err
	}
	grid := linspace(lo, hi, p.points)

	for i := range traces {
		traces[i].curve = broaden(specs[i].Lines, []string{"total"}, grid, sigma, false, 1)["total"]
	}

	switch p.norm {
	case "each":
		for i := range traces {
			if m := maxSlice(traces[i].curve); m > 0 {
				for j := range traces[i].curve {
					traces[i].curve[j] /= m
				}
			}
		}
	case "common":
		// Against the global minimum's tallest in-window peak, so trace height
		// still carries relative total yield.
		ref := 0.0
		for i := range traces {
			ref = math.Max(ref, maxSlice(traces[i].curve))
		}
		if ref > 0 {
			for i := range traces {
				for j := range traces[i].curve {
					traces[i].curve[j] /= ref
				}
			}
		}
	default:
		return fmt.Errorf("-stack-norm must be \"each\" or \"common\", got %q", p.norm)
	}

	if p.dedup > 0 {
		var dropped int
		traces, dropped = stackDedup(traces, p.dedup)
		if dropped > 0 {
			fmt.Fprintf(os.Stderr,
				"plotspec: folded %d duplicate basin(s) into their representative (max |Δ| ≤ %.3g)\n",
				dropped, p.dedup)
		}
	}

	if p.geomSuffix != "" && p.insetWidth > 0 {
		stackLoadGeometry(traces, p.specSuffix, p.geomSuffix)
	}
	stackTrim(traces)
	stackBaselines(traces, p.minSep)
	return stackDraw(traces, grid, lo, hi, p)
}

// stackCollect resolves the traces to plot from the -stack entries and, when
// given, the run manifest.
func stackCollect(p stackParams) ([]stackTrace, error) {
	disp, ok := stackUnits[strings.ToLower(p.offsetUnit)]
	if !ok {
		return nil, fmt.Errorf("-offset-unit: unknown unit %q", p.offsetUnit)
	}

	var traces []stackTrace
	for _, e := range p.entries {
		label, rest, found := strings.Cut(e, "=")
		if !found {
			return nil, fmt.Errorf("-stack %q: want LABEL=PATH[@OFFSET]", e)
		}
		path, offStr, hasOff := strings.Cut(rest, "@")
		t := stackTrace{label: label, path: path}
		if hasOff {
			off, err := strconv.ParseFloat(offStr, 64)
			if err != nil {
				return nil, fmt.Errorf("-stack %q: offset: %w", e, err)
			}
			t.offset = off // already in display units
		}
		traces = append(traces, t)
	}

	if p.manifest != "" {
		mTraces, err := stackFromManifest(p, disp.perHartree)
		if err != nil {
			return nil, err
		}
		traces = append(traces, mTraces...)
	}
	return traces, nil
}

// stackFromManifest builds traces from a run manifest: a JSON object mapping a
// run name to its metadata. The energy key holds a total energy (hartree by
// default) which is re-referenced to the most stable run in the selected group;
// the group key, when set, restricts the manifest to one subset (one hydration
// level, say). A run whose spectrum JSON is absent -- its calculation has not
// landed yet -- is reported and skipped rather than failing the figure.
func stackFromManifest(p stackParams, perHartree float64) ([]stackTrace, error) {
	raw, err := os.ReadFile(p.manifest)
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var entries map[string]map[string]any
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("parse manifest %s: %w", p.manifest, err)
	}

	src, ok := stackUnits[strings.ToLower(p.manifestUnit)]
	if !ok {
		return nil, fmt.Errorf("-manifest-unit: unknown unit %q", p.manifestUnit)
	}
	// Manifest energies are in src units; convert to hartree, then to display.
	scale := perHartree / src.perHartree

	dir := p.specDir
	if dir == "" {
		dir = filepath.Dir(p.manifest)
	}

	type cand struct {
		name   string
		energy float64
		path   string
	}
	var got []cand
	var pending, noEnergy []string
	for name, meta := range entries {
		if p.group != "" {
			v, ok := meta[p.groupKey]
			if !ok || stackScalarString(v) != p.group {
				continue
			}
		}
		ev, ok := meta[p.energyKey].(float64)
		if !ok {
			noEnergy = append(noEnergy, name)
			continue
		}
		path := filepath.Join(dir, name+p.specSuffix)
		if _, err := os.Stat(path); err != nil {
			pending = append(pending, name)
			continue
		}
		got = append(got, cand{name: name, energy: ev, path: path})
	}
	sort.Strings(pending)
	sort.Strings(noEnergy)
	if len(noEnergy) > 0 {
		return nil, fmt.Errorf("manifest %s: no %q for %s",
			p.manifest, p.energyKey, strings.Join(noEnergy, ", "))
	}
	if len(got) == 0 {
		return nil, fmt.Errorf("manifest %s: no spectra found in %s (%d run(s) pending: %s)",
			p.manifest, dir, len(pending), strings.Join(pending, ", "))
	}
	if len(pending) > 0 {
		fmt.Fprintf(os.Stderr, "plotspec: %d run(s) have no spectrum yet, omitted: %s\n",
			len(pending), strings.Join(pending, ", "))
	}

	// Re-reference to the most stable run that actually has a spectrum. When a
	// more stable basin is still pending the whole ladder shifts once it lands,
	// so say which run the zero is.
	ref := got[0]
	for _, c := range got[1:] {
		if c.energy < ref.energy {
			ref = c
		}
	}
	fmt.Fprintf(os.Stderr, "plotspec: offsets referenced to %s (%.9g %s)\n",
		ref.name, ref.energy, p.manifestUnit)

	traces := make([]stackTrace, 0, len(got))
	for _, c := range got {
		traces = append(traces, stackTrace{
			label:  c.name,
			path:   c.path,
			offset: (c.energy - ref.energy) * scale,
		})
	}
	return traces, nil
}

// stackScalarString renders a manifest scalar for comparison against -stack-group,
// so "1" matches the JSON number 1 as well as the string "1".
func stackScalarString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	case bool:
		return strconv.FormatBool(t)
	default:
		return fmt.Sprint(v)
	}
}

// stackWindow resolves the plotted energy window: -xrange when given, else the
// range spanned by every trace's sticks plus -pad.
func stackWindow(p stackParams, specs []*spectrum) (float64, float64, error) {
	if p.xRange != "" {
		return parseRange(p.xRange)
	}
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, spec := range specs {
		for _, l := range spec.Lines {
			lo = math.Min(lo, l.Energy)
			hi = math.Max(hi, l.Energy)
		}
	}
	if math.IsInf(lo, 1) {
		return 0, 0, fmt.Errorf("-mode stack: no sticks in any input")
	}
	return lo - p.pad, hi + p.pad, nil
}

// stackDedup folds traces whose normalised curves agree everywhere to within
// tol into the most stable member, which keeps the others as aliases. A
// structure search reaches the same minimum from several starting guesses, so
// the raw basin list holds duplicates that would otherwise draw exactly on top
// of one another and misrepresent how many distinct basins were found.
func stackDedup(traces []stackTrace, tol float64) ([]stackTrace, int) {
	if tol <= 0 {
		return traces, 0
	}
	var kept []stackTrace
	dropped := 0
	for _, t := range traces { // ascending energy: the representative comes first
		merged := false
		for i := range kept {
			if stackMaxDiff(kept[i].curve, t.curve) <= tol {
				kept[i].aliases = append(kept[i].aliases, t.label)
				merged, dropped = true, dropped+1
				break
			}
		}
		if !merged {
			kept = append(kept, t)
		}
	}
	return kept, dropped
}

// stackMaxDiff is the largest pointwise difference between two curves.
func stackMaxDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	m := 0.0
	for i := range a {
		m = math.Max(m, math.Abs(a[i]-b[i]))
	}
	return m
}

// stackBaselines places each trace's baseline at its relative energy, nudged up
// only as far as needed to clear the trace below by minSep (a fraction of the
// mean spacing). Isoenergetic basins would otherwise share a baseline exactly;
// the y ticks stay on the true energies, so the nudge is a drawing device only.
func stackBaselines(traces []stackTrace, minSep float64) {
	n := len(traces)
	if n == 0 {
		return
	}
	span := traces[n-1].offset - traces[0].offset
	unit := 1.0
	if span > 0 && n > 1 {
		unit = span / float64(n-1)
	}
	gap := minSep * unit
	traces[0].base = traces[0].offset
	for i := 1; i < n; i++ {
		traces[i].base = math.Max(traces[i].offset, traces[i-1].base+gap)
	}
}

// stackDraw renders the waterfall.
func stackDraw(traces []stackTrace, grid []float64, lo, hi float64, p stackParams) error {
	n := len(traces)
	disp := stackUnits[strings.ToLower(p.offsetUnit)]

	baseLo, baseHi := traces[0].base, traces[n-1].base
	row := 1.0
	if n > 1 && baseHi > baseLo {
		row = (baseHi - baseLo) / float64(n-1)
	}
	amp := p.traceScale * row

	plt := plot.New()
	plt.Title.Text = p.title
	plt.Title.TextStyle.Color = stackInk
	plt.X.Label.Text = "Ionization energy (eV)"
	plt.Y.Label.Text = "Relative conformer energy (" + disp.label + ")"
	for _, ax := range []*plot.Axis{&plt.X, &plt.Y} {
		ax.Label.TextStyle.Color = stackInk
		ax.Color = stackRule
		ax.LineStyle.Width = vg.Points(0.7)
		ax.Tick.Color = stackRule
		ax.Tick.Label.Color = stackMuted
	}
	plt.X.Min, plt.X.Max = lo, hi
	plt.Y.Min = baseLo - 0.10*amp
	plt.Y.Max = baseHi + 1.15*amp

	// One tick per trace, at the drawing baseline but labelled with the *true*
	// relative energy (stackBaselines may have nudged the two apart), and with
	// the conformer's name on a second line under that number, so the axis
	// names each trace where its energy is read rather than over the curve.
	ticks := make([]plot.Tick, 0, n)
	for _, t := range traces {
		name := t.label
		if len(t.aliases) > 0 {
			name = fmt.Sprintf("%s (x%d)", name, len(t.aliases)+1)
		}
		ticks = append(ticks, plot.Tick{
			Value: t.base,
			Label: strconv.FormatFloat(t.offset, 'f', 2, 64) + "\n" + name,
		})
	}
	plt.Y.Tick.Marker = plot.ConstantTicks(ticks)
	plt.Y.Tick.Label.XAlign = draw.XRight

	spanOff := traces[n-1].offset - traces[0].offset

	// Least stable first, so the global minimum is drawn last and is never
	// occluded; reading downward is then stability increasing.
	for k := n - 1; k >= 0; k-- {
		t := traces[k]
		tint := 0.0
		if spanOff > 0 {
			tint = t.offset / spanOff
		}
		col := stackColor(tint)

		top := make(plotter.XYs, len(grid))
		for i, x := range grid {
			top[i] = plotter.XY{X: x, Y: t.base + amp*t.curve[i]}
		}

		// Baseline rule, then an opaque fill that hides the traces below, then
		// a wash of the trace's own colour so the overlap is still legible.
		rule, err := plotter.NewLine(plotter.XYs{{X: lo, Y: t.base}, {X: hi, Y: t.base}})
		if err != nil {
			return err
		}
		rule.Color = stackRule
		rule.Width = vg.Points(0.5)
		plt.Add(rule)

		ring := make(plotter.XYs, 0, len(grid)+2)
		ring = append(ring, plotter.XY{X: grid[0], Y: t.base})
		ring = append(ring, top...)
		ring = append(ring, plotter.XY{X: grid[len(grid)-1], Y: t.base})

		// Two opaque passes rather than one translucent one: the PDF backend
		// does not honour alpha, so a wash left as NRGBA prints grey while the
		// PNG comes out tinted. Blending against the paper here keeps vector
		// and raster output identical.
		for _, fill := range []color.Color{
			stackPaper,
			blendOnPaper(col, 0.17),
		} {
			poly, err := plotter.NewPolygon(ring)
			if err != nil {
				return err
			}
			poly.Color = fill
			poly.LineStyle.Width = 0
			plt.Add(poly)
		}

		line, err := plotter.NewLine(top)
		if err != nil {
			return err
		}
		line.Color = col
		line.Width = vg.Points(1.1)
		plt.Add(line)

		// The structure inset goes in the corner the spectrum leaves empty:
		// the high-energy end of this trace. It floats just clear of the
		// trace's own curve there rather than sitting at a fixed height, so it
		// hugs the spectrum it belongs to however far that tail has decayed.
		// Drawn in the trace's own colour, tying picture to curve without a
		// legend.
		if t.depict != nil && p.insetWidth > 0 {
			ix1 := hi - 0.015*(hi-lo)
			ix0 := ix1 - p.insetWidth*(hi-lo)
			y0 := t.base + p.insetLift*amp
			plt.Add(&moleculeInset{
				d: t.depict, x0: ix0, x1: ix1,
				y0: y0, y1: y0 + p.insetHeight*amp, color: col,
			})
		}
	}

	if err := savePlot(plt, p.out, p.width, p.height, p.dpi); err != nil {
		return err
	}
	norm := "each trace normalised to its own maximum"
	if p.norm == "common" {
		norm = "common scale"
	}
	fmt.Fprintf(os.Stderr, "plotspec: wrote %s (waterfall, %d trace(s), fwhm %.2g eV, %s)\n",
		p.out, n, p.fwhm, norm)
	for _, t := range traces {
		extra := ""
		if len(t.aliases) > 0 {
			extra = " [= " + strings.Join(t.aliases, ", ") + "]"
		}
		fmt.Fprintf(os.Stderr, "    %+7.3f %s  %s%s\n", t.offset, disp.label, t.label, extra)
	}
	return nil
}

// stackLoadGeometry attaches each trace's structure drawing, looking for the
// geometry beside its spectrum: the spectrum's suffix swapped for geomSuffix.
// A trace whose geometry is missing or unreadable simply goes without a
// picture -- a figure that still plots is worth more than one that refuses to.
func stackLoadGeometry(traces []stackTrace, specSuffix, geomSuffix string) {
	for i := range traces {
		path := strings.TrimSuffix(traces[i].path, specSuffix) + geomSuffix
		g, err := readGeometry(path)
		if err != nil {
			if !os.IsNotExist(err) {
				fmt.Fprintf(os.Stderr, "plotspec: %s: no structure inset: %v\n", traces[i].label, err)
			}
			continue
		}
		d, err := depict(g)
		if err != nil {
			fmt.Fprintf(os.Stderr, "plotspec: %s: no structure inset: %v\n", traces[i].label, err)
			continue
		}
		traces[i].depict = d
	}
}

// moleculeInset draws a skeletal structure inside a data-space box. It is a
// plot.Plotter rather than a pre-rendered image so the drawing stays vector in
// a PDF and scales with the figure.
type moleculeInset struct {
	d              *depiction
	x0, y0, x1, y1 float64 // anchor box, data space
	color          color.Color
}

// insetPad is the fraction of the drawing's box kept clear on each side, so the
// outermost atom labels do not touch (or clip at) the edge.
const insetPad = 0.12

func (m *moleculeInset) Plot(c draw.Canvas, plt *plot.Plot) {
	trX, trY := plt.Transforms(&c)
	bx0, bx1 := trX(m.x0), trX(m.x1)
	by0, by1 := trY(m.y0), trY(m.y1)
	if bx1 < bx0 {
		bx0, bx1 = bx1, bx0
	}
	if by1 < by0 {
		by0, by1 = by1, by0
	}

	// The depiction is normalised into a unit square with its short axis
	// centred, so mapping that square onto the largest square the box holds
	// keeps the molecule undistorted whatever the axis units are.
	side := bx1 - bx0
	if h := by1 - by0; h < side {
		side = h
	}
	if side <= 0 {
		return
	}
	pad := side * insetPad
	side -= 2 * pad
	// Anchored to the top-right corner of the box.
	ox := bx1 - pad - side
	oy := by1 - pad - side

	at := func(i int) vg.Point {
		p := m.d.pts[i]
		return vg.Point{X: ox + side*vg.Length(p[0]), Y: oy + side*vg.Length(p[1])}
	}

	bondSty := draw.LineStyle{Color: m.color, Width: side * 0.011}
	if bondSty.Width < vg.Points(0.3) {
		bondSty.Width = vg.Points(0.3)
	}
	hbondSty := bondSty
	hbondSty.Dashes = []vg.Length{bondSty.Width * 2, bondSty.Width * 2}

	// Sticks first, then balls over them, so a bond stops at the atom it
	// reaches instead of striking through it.
	for _, b := range m.d.bonds {
		c.StrokeLines(bondSty, []vg.Point{at(b[0]), at(b[1])})
	}
	for _, b := range m.d.hbonds {
		c.StrokeLines(hbondSty, []vg.Point{at(b[0]), at(b[1])})
	}

	ballR := side * 0.042
	if ballR < vg.Points(0.9) {
		ballR = vg.Points(0.9)
	}
	for i, el := range m.d.elems {
		col, rel := cpkStyle(el)
		disc := discPoints(at(i), ballR*vg.Length(rel))
		c.FillPolygon(col, disc)
		// A darker rim keeps the pale atoms (hydrogen especially) from
		// disappearing into the paper at this size.
		c.StrokeLines(draw.LineStyle{Color: darken(col, 0.55), Width: bondSty.Width * 0.8},
			append(disc, disc[0]))
	}
}

// darken scales a colour towards black by f, for the rim around each ball.
func darken(col color.RGBA, f float64) color.RGBA {
	return color.RGBA{
		uint8(float64(col.R) * f),
		uint8(float64(col.G) * f),
		uint8(float64(col.B) * f),
		0xff,
	}
}

// DataRange keeps the inset from widening the axes: the box is placed inside a
// range the traces have already set.
func (m *moleculeInset) DataRange() (xmin, xmax, ymin, ymax float64) {
	return math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)
}

// discPoints approximates a filled circle; vg has no fill-circle primitive and
// a 20-gon is indistinguishable at these sizes.
func discPoints(centre vg.Point, r vg.Length) []vg.Point {
	const n = 20
	pts := make([]vg.Point, n)
	for i := range pts {
		th := 2 * math.Pi * float64(i) / n
		pts[i] = vg.Point{
			X: centre.X + r*vg.Length(math.Cos(th)),
			Y: centre.Y + r*vg.Length(math.Sin(th)),
		}
	}
	return pts
}
