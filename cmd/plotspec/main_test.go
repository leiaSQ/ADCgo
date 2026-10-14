package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestReadTDM checks that a -tdm document is flattened into the (energy,
// intensity, channel) spectrum the renderer expects: omega_ev → energy,
// osc → intensity, the three transition families → channels in canonical order,
// and non-positive oscillator strengths dropped.
func TestReadTDM(t *testing.T) {
	const doc = `{
	  "norb": 10, "order": 3,
	  "emissions": [
	    {"omega_ev": 12.5, "osc": 0.30},
	    {"omega_ev": 9.0,  "osc": 0.0}
	  ],
	  "photoionization": [
	    {"channels": [
	      {"omega_ev": 540.1, "osc": 0.80},
	      {"omega_ev": 545.6, "osc": 0.05},
	      {"omega_ev": 550.0, "osc": -1e-9}
	    ]}
	  ]
	}`
	path := filepath.Join(t.TempDir(), "tdm.json")
	if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}

	s, err := readTDM(path)
	if err != nil {
		t.Fatalf("readTDM: %v", err)
	}
	if s.Meta.Kind != "tdm" || s.Meta.EnergyUnit != "eV" {
		t.Errorf("meta = %+v, want kind=tdm unit=eV", s.Meta)
	}

	// Two positive emission/photoionization lines survive; the osc=0 emission and
	// the negative photoionization channel are dropped.
	if len(s.Lines) != 3 {
		t.Fatalf("got %d lines, want 3: %+v", len(s.Lines), s.Lines)
	}
	got := map[string]int{}
	for _, l := range s.Lines {
		got[l.Channel]++
		if l.Intensity <= 0 {
			t.Errorf("line with non-positive intensity leaked: %+v", l)
		}
	}
	if got["emission"] != 1 || got["photoionization"] != 2 {
		t.Errorf("channel counts = %v, want emission:1 photoionization:2", got)
	}

	// Channels are only the present families, in canonical order.
	want := []string{"emission", "photoionization"}
	if len(s.Channels) != len(want) {
		t.Fatalf("channels = %v, want %v", s.Channels, want)
	}
	for i := range want {
		if s.Channels[i] != want[i] {
			t.Errorf("channels[%d] = %q, want %q", i, s.Channels[i], want[i])
		}
	}

	// The first emission maps omega_ev→energy and osc→intensity verbatim.
	var em *specLine
	for i := range s.Lines {
		if s.Lines[i].Channel == "emission" {
			em = &s.Lines[i]
			break
		}
	}
	if em == nil || em.Energy != 12.5 || em.Intensity != 0.30 {
		t.Errorf("emission line = %+v, want energy=12.5 intensity=0.30", em)
	}
}

// TestEESDiffSubtractsExactly pins the arithmetic of -mode ees-diff, which is the Auger-Meitner
// difference spectrum sigma_sys - scale*sigma_ref (dimer minus 2x counterpoise monomer).
//
// Using one spectrum as BOTH system and reference turns the physics into an identity the test can
// assert exactly, with no reference data: at scale 1 the difference must be identically zero, and at
// scale 2 it must be exactly the negated system curve. Both hold bit-for-bit because -xrange forces
// the two curves onto the same linspace grid, so the subtraction is element-wise with no
// interpolation — which is the property the mode's -xrange requirement exists to guarantee.
func TestEESDiffSubtractsExactly(t *testing.T) {
	dir := t.TempDir()
	spec := func(name, kind string, lines string) string {
		doc := `{"meta":{"kind":"` + kind + `","energy_unit":"eV","singlet_triplet_ratio":1,` +
			`"initial_ionization":{"atom":""},"irreps":["1"]},"channels":["states"],"lines":[` + lines + `]}`
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}
	line := func(e, i float64) string {
		return fmt.Sprintf(`{"energy":%g,"intensity":%g,"channel":"states","spin":2,"irrep":1}`, e, i)
	}
	sip := spec("sip.json", "sip", line(20, 1.0)+","+line(26, 0.5))
	dip := spec("dip.json", "dip", line(32, 1.0)+","+line(38, 0.4))

	read := func(path string) (eps, sys, scaled, diff []float64) {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		for _, ln := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			if strings.HasPrefix(ln, "#") {
				continue
			}
			f := strings.Split(ln, "\t")
			if len(f) != 4 {
				t.Fatalf("data row %q has %d fields, want 4", ln, len(f))
			}
			v := make([]float64, 4)
			for i := range f {
				if v[i], err = strconv.ParseFloat(f[i], 64); err != nil {
					t.Fatal(err)
				}
			}
			eps, sys, scaled, diff = append(eps, v[0]), append(sys, v[1]), append(scaled, v[2]), append(diff, v[3])
		}
		return
	}

	run := func(scale float64, tag string) (eps, sys, scaled, diff []float64) {
		data := filepath.Join(dir, tag+".tsv")
		err := runEESDiff(eesDiffParams{
			base: eesParams{
				sip: sip, dip: dip, out: filepath.Join(dir, tag+".png"),
				fwhm: 1.5, points: 128, pad: 5, xRange: "0-20",
				width: 6, height: 4, dpi: 72,
			},
			refSIP: sip, refDIP: dip, scale: scale, dataOut: data,
		})
		if err != nil {
			t.Fatalf("runEESDiff(scale=%g): %v", scale, err)
		}
		return read(data)
	}

	eps, sys, _, diff := run(1, "s1")
	if len(eps) != 128 {
		t.Fatalf("got %d grid points, want 128", len(eps))
	}
	if sum := 0.0; true {
		for _, v := range sys {
			sum += v
		}
		if sum <= 0 {
			t.Fatalf("system curve is all zero (sum %g); the test asserts nothing", sum)
		}
	}
	for i, v := range diff {
		if v != 0 {
			t.Fatalf("scale 1: diff[%d] = %g, want exactly 0", i, v)
		}
	}

	_, sys2, scaled2, diff2 := run(2, "s2")
	for i := range diff2 {
		if diff2[i] != -sys2[i] {
			t.Fatalf("scale 2: diff[%d] = %g, want exactly -system = %g", i, diff2[i], -sys2[i])
		}
		if scaled2[i] != 2*sys2[i] {
			t.Fatalf("scale 2: scaled[%d] = %g, want 2*system = %g", i, scaled2[i], 2*sys2[i])
		}
	}

	// The guards. Each would otherwise fail silently into a wrong difference.
	for _, tc := range []struct {
		name string
		p    eesDiffParams
		want string
	}{
		{"no xrange", eesDiffParams{base: eesParams{sip: sip, dip: dip, fwhm: 1, points: 8}, refSIP: sip, refDIP: dip, scale: 2}, "-xrange"},
		{"no reference", eesDiffParams{base: eesParams{sip: sip, dip: dip, fwhm: 1, points: 8, xRange: "0-20"}, scale: 2}, "-ref-sip"},
		{"no system", eesDiffParams{base: eesParams{fwhm: 1, points: 8, xRange: "0-20"}, refSIP: sip, refDIP: dip, scale: 2}, "-sip"},
		{"zero scale", eesDiffParams{base: eesParams{sip: sip, dip: dip, fwhm: 1, points: 8, xRange: "0-20"}, refSIP: sip, refDIP: dip}, "-ref-scale"},
	} {
		err := runEESDiff(tc.p)
		if err == nil {
			t.Errorf("%s: want an error mentioning %q, got nil", tc.name, tc.want)
		} else if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %q does not mention %q", tc.name, err, tc.want)
		}
	}
}

// TestTwoSiteLines pins the independent-hole construction of the two-site double-ionization family,
// which is the term a monomer DIP calculation structurally cannot contain and therefore the reason a
// plain dimer - 2*monomer residual is large even with no interaction.
//
// The three properties are exact, so they are asserted exactly: ORDERED pairs (the two monomers are
// distinguishable, so i-on-A/j-on-B and its swap are different states) give N^2 lines and total
// weight w*(sum s)^2; energies are E_i + E_j + shift; and non-positive intensities are dropped
// rather than contributing a negative or zero-weight state.
func TestTwoSiteLines(t *testing.T) {
	sip := []specLine{
		{Energy: 10, Intensity: 0.5},
		{Energy: 12, Intensity: 0.25},
		{Energy: 20, Intensity: 0}, // dropped: no weight
		{Energy: 22, Intensity: -1},
	}
	const w, shift = 0.5, 4.2
	got := twoSiteLines(sip, w, shift)

	if len(got) != 4 { // 2 positive lines -> 2^2 ordered pairs
		t.Fatalf("got %d lines, want 4 (ordered pairs of the 2 positive SIP lines)", len(got))
	}
	// Total weight must be w*(sum of positive intensities)^2 = 0.5*(0.75)^2.
	sum := 0.0
	for _, l := range got {
		sum += l.Intensity
		if l.Channel != "two-site" {
			t.Errorf("channel = %q, want \"two-site\"", l.Channel)
		}
	}
	if want := w * 0.75 * 0.75; math.Abs(sum-want) > 1e-15 {
		t.Errorf("total two-site weight = %.17g, want %.17g", sum, want)
	}
	// The energies, including the shift, with their independent-hole weights.
	want := map[float64]float64{
		10 + 10 + shift: w * 0.5 * 0.5,
		10 + 12 + shift: w * 0.5 * 0.25, // appears twice (both orderings)
		12 + 12 + shift: w * 0.25 * 0.25,
	}
	seen := map[float64]float64{}
	for _, l := range got {
		seen[l.Energy] += l.Intensity
	}
	if len(seen) != 3 {
		t.Fatalf("got %d distinct energies, want 3: %v", len(seen), seen)
	}
	for e, wv := range want {
		mult := 1.0
		if e == 10+12+shift {
			mult = 2 // i<->j swap: two distinct states at the same energy
		}
		if math.Abs(seen[e]-mult*wv) > 1e-15 {
			t.Errorf("energy %.3g: weight %.17g, want %.17g", e, seen[e], mult*wv)
		}
	}

	// A zero/negative weight must not survive, and no line may sit at a dropped line's energy.
	for _, l := range got {
		if l.Intensity <= 0 {
			t.Errorf("line %+v has non-positive weight", l)
		}
		if l.Energy >= 20+20+shift {
			t.Errorf("line %+v is built from a dropped SIP line", l)
		}
	}
}

// TestEESDiffTwoSiteChangesReference checks the wiring: -ref-twosite must actually alter the
// reference curve (and so the residual), and its guard must fire. Without this the flag could be
// silently inert, which is the failure mode that matters -- the plot would look plausible and the
// physics conclusion would be wrong.
func TestEESDiffTwoSiteChangesReference(t *testing.T) {
	dir := t.TempDir()
	write := func(name, kind, lines string) string {
		doc := `{"meta":{"kind":"` + kind + `","energy_unit":"eV","singlet_triplet_ratio":1,` +
			`"initial_ionization":{"atom":""},"irreps":["1"]},"channels":["states"],"lines":[` + lines + `]}`
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}
	line := func(e, i float64) string {
		return fmt.Sprintf(`{"energy":%g,"intensity":%g,"channel":"states","spin":2,"irrep":1}`, e, i)
	}
	sip := write("sip.json", "sip", line(10, 1.0)+","+line(14, 0.6))
	dip := write("dip.json", "dip", line(26, 1.0)+","+line(31, 0.5))

	run := func(twoSite bool, tag string) []float64 {
		data := filepath.Join(dir, tag+".tsv")
		err := runEESDiff(eesDiffParams{
			base: eesParams{
				sip: sip, dip: dip, out: filepath.Join(dir, tag+".png"),
				fwhm: 1.5, points: 64, pad: 5, xRange: "0-12",
				width: 6, height: 4, dpi: 72,
			},
			refSIP: sip, refDIP: dip, scale: 2, dataOut: data,
			twoSite: twoSite, twoSiteWeight: 0.5, twoSiteShift: 0,
		})
		if err != nil {
			t.Fatalf("runEESDiff(twoSite=%v): %v", twoSite, err)
		}
		b, err := os.ReadFile(data)
		if err != nil {
			t.Fatal(err)
		}
		var ref []float64
		for _, ln := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			if strings.HasPrefix(ln, "#") {
				continue
			}
			f := strings.Split(ln, "\t")
			v, err := strconv.ParseFloat(f[2], 64) // the scaled reference column
			if err != nil {
				t.Fatal(err)
			}
			ref = append(ref, v)
		}
		return ref
	}

	without, with := run(false, "off"), run(true, "on")
	if len(without) != len(with) {
		t.Fatalf("grid lengths differ: %d vs %d", len(without), len(with))
	}
	same := true
	for i := range with {
		if with[i] != without[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("-ref-twosite left the reference curve bit-identical; the flag is inert")
	}

	// Guard: the weight must be positive, or the flag would quietly contribute nothing.
	err := runEESDiff(eesDiffParams{
		base:   eesParams{sip: sip, dip: dip, fwhm: 1, points: 8, xRange: "0-12"},
		refSIP: sip, refDIP: dip, scale: 2, twoSite: true, twoSiteWeight: 0,
	})
	if err == nil || !strings.Contains(err.Error(), "-twosite-weight") {
		t.Errorf("want a -twosite-weight error, got %v", err)
	}
}
