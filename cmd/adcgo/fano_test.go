package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fano"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
	"github.com/leiaSQ/ADCgo/internal/adc/stieltjes"
)

func testFCIDUMP(t *testing.T) *fcidump.Data {
	t.Helper()
	d, err := fcidump.ReadFile("../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	return d
}

func TestParseOrbitalList(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []int
		bad  bool
	}{
		{"", nil, false},
		{"   ", nil, false},
		{"0", []int{0}, false},
		{"0,3, 7 ", []int{0, 3, 7}, false},
		{"1,,2", []int{1, 2}, false},
		{"-1", nil, true},
		{"a", nil, true},
		{"1.5", nil, true},
	} {
		got, err := parseOrbitalList("-fano-q", c.in)
		if c.bad {
			if err == nil {
				t.Errorf("parseOrbitalList(%q) accepted", c.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseOrbitalList(%q): %v", c.in, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("parseOrbitalList(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestParseHoleRule(t *testing.T) {
	for _, c := range []struct {
		in   string
		want fano.HoleRule
		bad  bool
	}{
		{"", fano.AnyHole, false},
		{"any", fano.AnyHole, false},
		{"all", fano.AllHoles, false},
		{"ANY", fano.AnyHole, true},
		{"either", fano.AnyHole, true},
	} {
		got, err := parseHoleRule(c.in)
		if c.bad != (err != nil) {
			t.Errorf("parseHoleRule(%q): err = %v, want bad = %v", c.in, err, c.bad)
		}
		if err == nil && got != c.want {
			t.Errorf("parseHoleRule(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestParseStieltjesOrders(t *testing.T) {
	for _, c := range []struct {
		in     string
		lo, hi int
		bad    bool
	}{
		{"", 0, 0, false},
		{"5-40", 5, 40, false},
		{"8-", 8, 0, false},
		{"-40", 0, 40, false},
		{"7-7", 7, 7, false},
		{"40-5", 0, 0, true},
		{"0-10", 0, 0, true},
		{"x-10", 0, 0, true},
	} {
		lo, hi, err := parseStieltjesOrders(c.in)
		if c.bad != (err != nil) {
			t.Errorf("parseStieltjesOrders(%q): err = %v, want bad = %v", c.in, err, c.bad)
			continue
		}
		if err == nil && (lo != c.lo || hi != c.hi) {
			t.Errorf("parseStieltjesOrders(%q) = (%d,%d), want (%d,%d)", c.in, lo, hi, c.lo, c.hi)
		}
	}
}

func TestParseAverageMode(t *testing.T) {
	if m, err := parseAverageMode(""); err != nil || m != stieltjes.AveragePaper {
		t.Errorf("the default averaging mode is %v (err %v), want the paper's", m, err)
	}
	if m, err := parseAverageMode("reference"); err != nil || m != stieltjes.AverageReference {
		t.Errorf("parseAverageMode(reference) = %v, err %v", m, err)
	}
	if _, err := parseAverageMode("mean"); err == nil {
		t.Error("an unknown averaging mode was accepted")
	}
}

// TestFanoRunRejectsBadConfig covers the guards that would otherwise produce a confident
// wrong number: an order with no Fano path, and a vacancy that is not occupied.
func TestFanoRunRejectsBadConfig(t *testing.T) {
	d := testFCIDUMP(t)
	for _, c := range []struct {
		name string
		cfg  fanoConfig
		want string
	}{
		{"order 4", fanoConfig{sip: sipConfig{order: 4, solver: "dense"}, vacancy: 0}, "-order"},
		{"unoccupied vacancy", fanoConfig{sip: sipConfig{order: 2, solver: "dense"}, vacancy: 999}, "occupied"},
		{"bad solver", fanoConfig{sip: sipConfig{order: 2, solver: "magic"}, vacancy: 0}, "solver"},
		{"dip, several sectors", fanoConfig{sip: sipConfig{solver: "dense", sym: "all"}, dip: true,
			spin: dip.Singlet, vacancy: 0}, "-sym"},
		{"dip partial widths", fanoConfig{sip: sipConfig{solver: "dense", sym: "none", moPath: "x.json"},
			dip: true, spin: dip.Singlet, vacancy: 0}, "partial widths"},
		{"lowmem keeps no Ritz vectors", fanoConfig{sip: sipConfig{solver: "lanczos-lowmem", sym: "none"},
			dip: true, spin: dip.Singlet, vacancy: 0}, "Ritz"},
	} {
		err := runFano(d, c.cfg)
		if err == nil {
			t.Errorf("%s: accepted", c.name)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %q does not mention %q", c.name, err, c.want)
		}
	}
}

// TestFanoDocumentSchema is the schema guard: the JSON keys a downstream parser or plotting
// script reads must not drift silently. Fields that are only sometimes present carry
// omitempty and are checked to be absent from a minimal document.
func TestFanoDocumentSchema(t *testing.T) {
	b, err := json.Marshal(FanoDocument{})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"norb", "nelec", "family", "order", "scheme", "vacancy", "irrep", "criterion",
		"size", "q_size", "p_size", "q_main", "p_main", "p_decay",
		"e_phi_hartree", "e_phi_ev", "phi_weight", "phi_root",
		"pseudo_states", "sum_gamma_hartree2", "sum_rule_hartree2", "sum_rule_residual",
		"dropped_energy", "dropped_weight", "dropped_gamma", "coupled_channels",
		"width_mev", "width_sigma_mev", "width_ev", "lifetime_fs",
		"stieltjes_max_order", "stieltjes_orders_used", "pseudo_emin_ev", "pseudo_emax_ev", "phi_position",
	}
	for _, k := range want {
		if _, ok := got[k]; !ok {
			t.Errorf("the Fano document is missing the required key %q", k)
		}
	}
	for _, k := range []string{
		"variant", "spin", "class_3h2p", "stieltjes_low_order", "partial_widths", "eval_ev",
		"partial_widths_note", "timing", "stieltjes_below", "stieltjes_above",
	} {
		if _, ok := got[k]; ok {
			t.Errorf("key %q should carry omitempty: a minimal document must not emit it", k)
		}
	}
	if len(got) != len(want) {
		t.Errorf("the Fano document has %d always-present keys, want %d — a new field needs "+
			"adding to this test's list (or omitempty)", len(got), len(want))
	}

	// The channel record's own schema.
	cb, err := json.Marshal(FanoChannel{Name: "Auger@O", Share: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	var cm map[string]any
	if err := json.Unmarshal(cb, &cm); err != nil {
		t.Fatal(err)
	}
	if _, ok := cm["name"]; !ok {
		t.Error("a channel record must always carry its name")
	}
	if _, ok := cm["strength_share"]; !ok {
		t.Error("a channel record must always carry its strength share")
	}
	if _, ok := cm["width_mev"]; ok {
		t.Error("width_mev should carry omitempty: a channel that could not be imaged has none")
	}
}

// TestFanoDIPEndToEnd runs -dip -fano on water (DIP-ADC(2) singlet, symmetry off, dense
// solves) with the vacancy in the O 2s orbital (an O 1s pair lies at ~21 Eh, far above the
// ~7.7 Eh top of cc-pVDZ's 3h1p pseudo-continuum, so imaging rightly refuses to
// extrapolate there). The dense PMP solve samples the whole
// pseudo-continuum, so the imaged strength must close the sum rule 2 pi ||g||^2; the
// document must name the family and sector, and the width must be finite.
func TestFanoDIPEndToEnd(t *testing.T) {
	d := testFCIDUMP(t)
	out := filepath.Join(t.TempDir(), "dip_fano.json")
	cfg := fanoConfig{
		sip: sipConfig{solver: "dense", sym: "none", backend: "gonum", order: 2, blocks: 50,
			out: out},
		dip: true, spin: dip.Singlet, vacancy: 1, rule: fano.AnyHole, qMin: 0.1,
		emax: -1, wmin: -1, gmin: -1,
	}
	if err := runFano(d, cfg); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var doc FanoDocument
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s %s: parent %d = Q %d + P %d; E_Phi %.4f eV (weight %.3f); Gamma %.4g +/- %.2g meV; "+
		"sum-rule residual %.2e; %d coupled", doc.Family, doc.Spin, doc.Size, doc.QSize, doc.PSize,
		doc.EPhiEV, doc.PhiWeight, doc.WidthMeV, doc.WidthSigmaMeV, doc.SumResidual, doc.CoupledChannels)
	if doc.Family != "dip" || doc.Spin != "singlet" || doc.Order != 2 {
		t.Errorf("document names family %q spin %q order %d", doc.Family, doc.Spin, doc.Order)
	}
	if doc.QSize+doc.PSize != doc.Size || doc.QSize == 0 || doc.PSize == 0 {
		t.Errorf("partition %d + %d of %d", doc.QSize, doc.PSize, doc.Size)
	}
	if math.Abs(doc.SumResidual) > 1e-8 {
		t.Errorf("dense PMP leaves sum-rule residual %.2e", doc.SumResidual)
	}
	if math.IsNaN(doc.WidthMeV) || math.IsInf(doc.WidthMeV, 0) || doc.WidthMeV < 0 {
		t.Errorf("width %g meV", doc.WidthMeV)
	}
}

// TestFanoKhciEndToEnd runs the k-hole CI family on water (K = 2, canonical orbitals, the
// vacancy rule of the DIP test, dense): the same pipeline as -dip -fano over CI instead of
// DIP-ADC(2). The dense PMP solve must close the sum rule, Phi's class weights must add up
// to one, and the census must be in the document.
func TestFanoKhciEndToEnd(t *testing.T) {
	d := testFCIDUMP(t)
	out := filepath.Join(t.TempDir(), "khci_fano.json")
	cfg := fanoConfig{
		sip:  sipConfig{solver: "dense", sym: "none", backend: "gonum", blocks: 50, out: out},
		khci: 2, khciMaxClass: 3, khciTwoMs: 0, khciMaxFree: -1, khciCSR: 1 << 30,
		vacancy: 1, rule: fano.AnyHole, qMin: 0.1, emax: -1, wmin: -1, gmin: -1,
	}
	if err := runFano(d, cfg); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var doc FanoDocument
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	var wsum float64
	for _, w := range doc.PhiClassWeights {
		wsum += w
	}
	t.Logf("%s K=%d: parent %d = Q %d + P %d; E_Phi %.4f eV (weight %.3f, classes %v); Gamma %.4g +/- %.2g meV; "+
		"sum-rule residual %.2e; census %s", doc.Family, doc.K, doc.Size, doc.QSize, doc.PSize, doc.EPhiEV,
		doc.PhiWeight, doc.PhiClassWeights, doc.WidthMeV, doc.WidthSigmaMeV, doc.SumResidual, doc.Census)
	if doc.Family != "khci" || doc.K != 2 || doc.Census == "" {
		t.Errorf("document names family %q K %d census %q", doc.Family, doc.K, doc.Census)
	}
	if math.Abs(wsum-1) > 1e-10 {
		t.Errorf("Phi class weights add up to %.12f", wsum)
	}
	if math.Abs(doc.SumResidual) > 1e-8 {
		t.Errorf("dense PMP leaves sum-rule residual %.2e", doc.SumResidual)
	}
	if math.IsNaN(doc.WidthMeV) || doc.WidthMeV < 0 {
		t.Errorf("width %g meV", doc.WidthMeV)
	}
}

// TestFanoKhciNullSystem runs the net-charge partition on the labelled He3 fixture (1.30 A
// chain, localized orbitals, ghost centres) with P = "every real atom +1 and one free
// electron", the open channel of single ETMD from He1. SelectDiscrete takes the lowest Q
// root carrying the He1 hole, a He+He+-type state near 49 eV; the He2+(He1) state lies
// higher and is what -fano-phi-holes selects. Both lie below the whole open-channel
// pseudo-continuum (three He+ ions and an electron, from ~85 eV here; pyscf FCI has the
// channel closed by several eV), so imaging must refuse to
// extrapolate rather than report a width. Reaching that refusal runs the labelled-sidecar
// path end to end: the net-charge rule, the free-particle limit, the CI over
// non-canonical orbitals, QMQ, the coupling and PMP.
func TestFanoKhciNullSystem(t *testing.T) {
	d, err := fcidump.ReadFile("../../testdata/khci/he3_ghost.fcidump")
	if err != nil {
		t.Fatal(err)
	}
	cfg := fanoConfig{
		sip: sipConfig{solver: "dense", sym: "none", backend: "gonum", blocks: 50,
			moPath: "../../testdata/khci/he3_ghost.mo.json", out: filepath.Join(t.TempDir(), "null.json")},
		khci: 2, khciTwoMs: 0, khciMaxFree: 1, khciCSR: 1 << 30,
		vacancy: 0, qpSpec: "p: charge He1=1,He2=1,He3=1 & free=1", qMin: 0.1,
		emax: -1, wmin: -1, gmin: -1,
	}
	err = runFano(d, cfg)
	if err == nil {
		t.Fatal("a width was reported for the closed null system")
	}
	if !strings.Contains(err.Error(), "outside the pseudo-continuum") {
		t.Fatalf("expected imaging to refuse below the open continuum, got: %v", err)
	}
	t.Logf("null system: %v", err)
}

// TestFanoKhciNullSystemInterior is TestFanoKhciNullSystem with |Phi> selected as the
// interior QMQ root on He1's doubly emptied orbital (-fano-phi-holes 0,0): the
// He2+(He1) He He state itself, at 2.8445 Eh (77.4 eV, internal/adc/fano
// TestSelectInteriorHe2Plus), not the lowest Q root carrying the He1 hole. The single-ETMD
// channel is still closed, so imaging must refuse at exactly that energy.
func TestFanoKhciNullSystemInterior(t *testing.T) {
	d, err := fcidump.ReadFile("../../testdata/khci/he3_ghost.fcidump")
	if err != nil {
		t.Fatal(err)
	}
	cfg := fanoConfig{
		sip: sipConfig{solver: "dense", sym: "none", backend: "gonum", blocks: 50,
			moPath: "../../testdata/khci/he3_ghost.mo.json", out: filepath.Join(t.TempDir(), "null.json")},
		khci: 2, khciTwoMs: 0, khciMaxFree: 1, khciCSR: 1 << 30,
		vacancy: 0, qpSpec: "p: charge He1=1,He2=1,He3=1 & free=1",
		phiHoles: []int{0, 0}, phiTol: 1e-10,
		emax: -1, wmin: -1, gmin: -1,
	}
	err = runFano(d, cfg)
	if err == nil {
		t.Fatal("a width was reported for the closed null system")
	}
	if !strings.Contains(err.Error(), "requested energy 2.8445") || !strings.Contains(err.Error(), "outside") {
		t.Fatalf("expected imaging to refuse at E(He2+) = 2.8445 Eh, got: %v", err)
	}
	t.Logf("null system, interior He2+(He1): %v", err)
}

// TestFanoEnginesRun runs -dip -fano on water (the TestFanoDIPEndToEnd case) through the
// three eigenvector-free width engines. Agreement between engines is gated in
// internal/adc/fano, where the continuum is well sampled (TestWidthEnginesRecoverLorentzian,
// TestGaussRuleIsEigenMeasure). This case is not one: the O 2s^-2 state sits at the
// bottom edge of its coupled pseudo-continuum in cc-pVDZ, so Gauss and shift-invert both
// return the below-the-first-node fallback and report it, and KPM finds almost no density
// there. What is checked is that each engine runs, records itself, returns a finite
// width, and flags the edge.
func TestFanoEnginesRun(t *testing.T) {
	d := testFCIDUMP(t)
	for _, c := range []struct {
		engine string
		order  int
	}{{"gauss", 390}, {"shiftinvert", 60}, {"kpm", 2000}} {
		out := filepath.Join(t.TempDir(), c.engine+".json")
		cfg := fanoConfig{
			sip: sipConfig{solver: "dense", sym: "none", backend: "gonum", order: 2, blocks: 50, out: out},
			dip: true, spin: dip.Singlet, vacancy: 1, rule: fano.AnyHole, qMin: 0.1,
			emax: -1, wmin: -1, gmin: -1, engine: c.engine, order: c.order, siShift: 1e-3,
		}
		if err := runFano(d, cfg); err != nil {
			t.Fatalf("%s: %v", c.engine, err)
		}
		raw, err := os.ReadFile(out)
		if err != nil {
			t.Fatal(err)
		}
		var doc FanoDocument
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Fatal(err)
		}
		t.Logf("%-11s Gamma %.4g +/- %.2g meV; orders below/above their grid %d/%d",
			c.engine, doc.WidthMeV, doc.WidthSigmaMeV, doc.StieltjesBelow, doc.StieltjesAbove)
		if doc.Engine != c.engine {
			t.Errorf("document names engine %q", doc.Engine)
		}
		if math.IsNaN(doc.WidthMeV) || math.IsInf(doc.WidthMeV, 0) || doc.WidthMeV < 0 {
			t.Errorf("%s: width %g", c.engine, doc.WidthMeV)
		}
		if c.engine != "kpm" && doc.StieltjesBelow == 0 {
			t.Errorf("%s: the edge case was not flagged (no order below its density grid)", c.engine)
		}
	}
}

// TestFanoLockInPlumbing exercises the lock-in driver pieces on the labelled He3 fixture. Its
// channel is closed at E_Phi, so Gamma is evaluated at a fixed energy inside the open
// continuum (-fano-at 3.5 Eh), which is enough for plumbing:
//   - a coupling vector written by -fano-save-g and imaged by -fano-image-g gives the same
//     width to the last digit;
//   - -fano-lambda is applied (the document records it and the width changes);
//   - -fano-decompose reports every Q class and every pair's interference.
func TestFanoLockInPlumbing(t *testing.T) {
	d, err := fcidump.ReadFile("../../testdata/khci/he3_ghost.fcidump")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	base := func(out string) fanoConfig {
		return fanoConfig{
			sip: sipConfig{solver: "dense", sym: "none", backend: "gonum", blocks: 50,
				moPath: "../../testdata/khci/he3_ghost.mo.json", out: filepath.Join(dir, out)},
			khci: 2, khciTwoMs: 0, khciMaxFree: 1, khciCSR: 1 << 30,
			vacancy: 0, qpSpec: "p: charge He1=1,He2=1,He3=1 & free=1",
			phiHoles: []int{0, 0}, phiTol: 1e-10, emax: -1, wmin: -1, gmin: -1,
			engine: "gauss", order: 100, atEV: 3.5 * hartreeToEV,
		}
	}
	read := func(out string) FanoDocument {
		raw, err := os.ReadFile(filepath.Join(dir, out))
		if err != nil {
			t.Fatal(err)
		}
		var doc FanoDocument
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Fatal(err)
		}
		return doc
	}
	c := base("full.json")
	c.saveG, c.decompose = filepath.Join(dir, "g.json"), true
	if err := runFano(d, c); err != nil {
		t.Fatal(err)
	}
	full := read("full.json")
	c = base("imaged.json")
	c.imageG = filepath.Join(dir, "g.json")
	if err := runFano(d, c); err != nil {
		t.Fatal(err)
	}
	imaged := read("imaged.json")
	c = base("lam.json")
	c.lambdaSpec = "He1:He2=0.5"
	if err := runFano(d, c); err != nil {
		t.Fatal(err)
	}
	lam := read("lam.json")
	t.Logf("Gamma at 3.5 Eh: %.6g meV; re-imaged %.6g meV; He1:He2 at 0.5: %.6g meV", full.WidthMeV,
		imaged.WidthMeV, lam.WidthMeV)
	for _, p := range full.Decomposition {
		t.Logf("  %-12s %.6g +/- %.2g meV (interference %v)", p.Name, p.WidthMeV, p.SigmaMeV, p.Interference)
	}
	if imaged.WidthMeV != full.WidthMeV {
		t.Errorf("-fano-image-g reproduced %.10g meV, the saving run had %.10g", imaged.WidthMeV, full.WidthMeV)
	}
	if lam.Lambda != "He1:He2=0.5" || lam.WidthMeV == full.WidthMeV {
		t.Errorf("-fano-lambda not applied (document %q, width %.6g vs %.6g)", lam.Lambda, lam.WidthMeV, full.WidthMeV)
	}
	var classes, pairs int
	for _, p := range full.Decomposition {
		if p.Interference {
			pairs++
		} else {
			classes++
		}
	}
	if classes != 3 || pairs != 3 {
		t.Errorf("decomposition has %d classes and %d pairs, want 3 and 3", classes, pairs)
	}
}

// toyResolverMO is a four-AO, four-orbital sidecar with S = 1 and one AO per atom, so
// each orbital's Mulliken population is the square of its column: orbital 0 sits on O2,
// 1 on O1, 2 is 0.8 O1 / 0.2 O2, and 3 is 0.6 H1 / 0.4 H2.
func toyResolverMO() *mo.Data {
	c := backend.NewMat(4, 4)
	s := backend.NewMat(4, 4)
	for p := range 4 {
		s.Set(p, p, 1)
	}
	c.Set(2, 0, 1)
	c.Set(0, 1, 1)
	c.Set(0, 2, math.Sqrt(0.8))
	c.Set(2, 2, math.Sqrt(0.2))
	c.Set(1, 3, math.Sqrt(0.6))
	c.Set(3, 3, math.Sqrt(0.4))
	return &mo.Data{NAO: 4, NMO: 4, C: c, S: s, AOAtom: []int{0, 1, 2, 3},
		AtomNames: []string{"O1", "H1", "O2", "H2"}}
}

// TestOrbitalResolver: the symbolic orbital forms, on a sidecar whose populations are
// known by construction.
func TestOrbitalResolver(t *testing.T) {
	eps := []float64{-1.40, -1.35, -0.70, -0.50}
	sites := []spectrum.Site{{Name: "W1", Members: []string{"O1", "H1"}},
		{Name: "W2", Members: []string{"O2", "H2"}}}
	r, err := newOrbitalResolver(eps, toyResolverMO(), sites)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in   string
		want []int
		bad  bool
	}{
		{"@W1", []int{1, 2, 3}, false},
		{"@W1.0", []int{1}, false},
		{"@W1.2", []int{3}, false},
		{"@W2", []int{0}, false},
		{"@W2.1", nil, true}, // W2 owns one orbital
		{"@W3", nil, true},   // no such site
		{"@W1.x", nil, true}, // bad rank
		{"e<-1.0", []int{0, 1}, false},
		{"e>-0.6", []int{3}, false},
		{"e<-2", nil, true},                      // empty window
		{"@W2.0,@W1.0,1", []int{0, 1, 1}, false}, // order and repeats kept
		{"4", nil, true},                         // not occupied
	} {
		got, err := r.list(c.in)
		if c.bad {
			if err == nil {
				t.Errorf("%q: accepted, got %v", c.in, got)
			}
			continue
		}
		if err != nil || !reflect.DeepEqual(got, c.want) {
			t.Errorf("%q = %v, %v; want %v", c.in, got, err, c.want)
		}
	}
	if got := r.pop[2][0]; math.Abs(got-0.8) > 1e-12 {
		t.Errorf("orbital 2 population on W1 = %g, want 0.8", got)
	}

	for _, c := range []struct{ in, want string }{
		{"q:1/0-3:1;q:e<-1.0:1", "q:1/0-3:1;q:0,1:1"},
		{"p:2/@W1.0:1&2/@W2:1", "p:2/1:1&2/0:1"},
		{"q:0:1;p:2/4:1;p:3/4:2", "q:0:1;p:2/4:1;p:3/4:2"},                     // nothing symbolic: unchanged
		{"q:1/e<0:1;x:2/@W1:2;x:2/@W2:2", "q:1/0,1,2,3:1;x:2/1,2,3:2;x:2/0:2"}, // exclusions expand too
	} {
		got, err := r.expandQP(c.in)
		if err != nil || got != c.want {
			t.Errorf("expandQP(%q) = %q, %v; want %q", c.in, got, err, c.want)
		}
	}

	// Without a sidecar the energy windows still resolve and @SITE says what it needs.
	bare, err := newOrbitalResolver(eps, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := bare.list("e<-1.0"); err != nil || !reflect.DeepEqual(got, []int{0, 1}) {
		t.Errorf("bare e<-1.0 = %v, %v", got, err)
	}
	if _, err := bare.list("@W1.0"); err == nil || !strings.Contains(err.Error(), "-mo") {
		t.Errorf("bare @W1.0: err = %v, want a pointer to -mo", err)
	}

	// A site naming a column the sidecar does not have is refused, as is a column in two sites.
	if _, err := newOrbitalResolver(eps, toyResolverMO(),
		[]spectrum.Site{{Name: "W1", Members: []string{"O9"}}}); err == nil {
		t.Error("unknown column accepted")
	}
	if _, err := newOrbitalResolver(eps, toyResolverMO(), []spectrum.Site{
		{Name: "A", Members: []string{"O1"}}, {Name: "B", Members: []string{"O1"}}}); err == nil {
		t.Error("column in two sites accepted")
	}
}

// TestResolveOrbitalsWater: -fano-init @O.0 on water, without -group, is the O 1s, and
// the document fields record where it sits.
func TestResolveOrbitalsWater(t *testing.T) {
	d := testFCIDUMP(t)
	nocc := mp.NOcc(d)
	cfg := fanoConfig{vacancy: -1, vacancySpec: "@O.0", qpSpec: "q:e<-1.0:1",
		sip: sipConfig{moPath: "../../testdata/h2o.mo.json"}}
	if err := cfg.resolveOrbitals(mp.OrbitalEnergies(d, nocc)[:nocc]); err != nil {
		t.Fatal(err)
	}
	if cfg.vacancy != 0 || cfg.vacancySite != "O" || cfg.vacancyPop < 0.99 {
		t.Errorf("vacancy %d on %s (%.4f), want orbital 0 on O above 0.99",
			cfg.vacancy, cfg.vacancySite, cfg.vacancyPop)
	}
	// water: 1a1 at -20.5 Eh and 2a1 at -1.3 Eh are the only orbitals below -1 Eh
	if cfg.qpSpec != "q:0,1:1" {
		t.Errorf("qpSpec = %q, want q:0,1:1", cfg.qpSpec)
	}
}
