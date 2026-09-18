package analyze

import (
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/refout"
)

func h2oSinglet(t *testing.T) *dip.Matrix {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := dip.NewSpace(nocc, d.NORB, nil, 0, dip.Singlet)
	return dip.New(sp, integrals.New(d, nocc, nil), eps, backend.Gonum{})
}

func TestBuildSector(t *testing.T) {
	mx := h2oSinglet(t)
	res := lanczos.SolveDense(mx, backend.Gonum{})
	sec := BuildSector(mx.Space(), res, Options{PSThresh: 5, CoeffThresh: 0.1}, nil)

	if sec.Irrep != 1 || sec.Spin != 1 {
		t.Fatalf("sector irrep/spin = %d/%d, want 1/1", sec.Irrep, sec.Spin)
	}
	if len(sec.States) == 0 {
		t.Fatal("no states")
	}

	// Ground dication state: ~39.17 eV, ps ~83%, dominated by |5,5> (HOMO²).
	g := sec.States[0]
	if math.Abs(g.EnergyEV-39.172) > 0.01 {
		t.Errorf("ground DIP energy = %.3f eV, want ~39.17", g.EnergyEV)
	}
	if math.Abs(g.PSPercent-83.35) > 0.5 {
		t.Errorf("ground pole strength = %.2f%%, want ~83.3", g.PSPercent)
	}
	if len(g.Leading) == 0 || g.Leading[0].I != 5 || g.Leading[0].J != 5 {
		t.Errorf("ground leading config = %+v, want first {5,5}", g.Leading)
	}

	// States must be energy-ordered with sequential 1-based indices, and every
	// leading list sorted by descending |coeff|.
	for i, s := range sec.States {
		if s.Index != i+1 {
			t.Errorf("state %d has index %d", i, s.Index)
		}
		if i > 0 && s.EnergyEV < sec.States[i-1].EnergyEV {
			t.Errorf("states not energy-ordered at %d", i)
		}
		for j := 1; j < len(s.Leading); j++ {
			if math.Abs(s.Leading[j].Coeff) > math.Abs(s.Leading[j-1].Coeff)+1e-12 {
				t.Errorf("state %d leading not sorted by |coeff|", s.Index)
			}
		}
	}
}

// refBlock extracts theADCcode's first "Eigenvalue (eV), ps (%), residue" block from a
// reference .out: the title line through the blank line after the last state, which is
// exactly what WriteRef is supposed to produce.
func refBlock(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("reference output not present: %v", err)
	}
	lines := strings.Split(string(raw), "\n")
	start := -1
	for i, l := range lines {
		if l == refTitle {
			start = i
			break
		}
	}
	if start < 0 {
		t.Fatalf("%s: no %q line", path, refTitle)
	}
	// The block ends at the first line after a state that is neither blank, a state line,
	// an overlap heading nor an overlap list — in adcdip1.out, " ADC two-hole population
	// analysis".
	end := len(lines)
	for i := start + 4; i < len(lines); i++ {
		tr := strings.TrimSpace(lines[i])
		switch {
		case tr == "", strings.HasPrefix(tr, "<"),
			strings.HasPrefix(tr, "Overlaps with"),
			reStateLine.MatchString(lines[i]):
			continue
		default:
			end = i
		}
		break
	}
	// Trim back to the single blank line that closes the last state.
	for end > start && strings.TrimSpace(lines[end-1]) == "" {
		end--
	}
	return strings.Join(lines[start:end], "\n") + "\n\n"
}

var reStateLine = regexp.MustCompile(`(?m)^\s*\d+:\s`)

// TestWriteRefReproducesReferenceBytes is the whole point of WriteRef: emitting a sector
// must give back theADCcode's own block, byte for byte. It parses adcdip1.out with the
// refout reader, feeds those states straight back through the writer, and diffs. Any drift
// in a field width, a separator or the wrap column fails here rather than at the far end of
// a validation run, where it would look like a physics disagreement.
func TestWriteRefReproducesReferenceBytes(t *testing.T) {
	path := filepath.Join("..", "..", "..", "testdata", "reference", "adcdip1.out")
	want := refBlock(t, path)

	f, err := refout.ParseFile(path)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	// adcdip1.out prints symmetry 1 twice (once per task), and refout merges both passes
	// into one spin-1 list, so take exactly as many states as the extracted block holds —
	// they are in file order, so those are the first block's.
	nWant := len(reStateLine.FindAllString(want, -1))
	if nWant == 0 {
		t.Fatalf("extracted block has no state lines")
	}
	var sec Sector
	for _, rs := range f.States {
		if rs.Spin != 1 || len(sec.States) == nWant {
			continue
		}
		st := State{
			Root:      rs.Index,
			EnergyEV:  rs.EnergyEV,
			PSPercent: rs.PSPercent,
			Residue:   rs.Residue * au2eV, // WriteRef converts back to a.u.
		}
		for _, c := range rs.Leading {
			st.Leading = append(st.Leading, Leading{I: c.I, J: c.J, Coeff: c.Coeff})
		}
		sec.States = append(sec.States, st)
	}
	if len(sec.States) != nWant {
		t.Fatalf("parsed %d singlet states, block has %d", len(sec.States), nWant)
	}
	t.Logf("round-tripped %d states of %s byte for byte", nWant, filepath.Base(path))

	var got strings.Builder
	if err := sec.WriteRef(&got); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}
	if got.String() != want {
		gl, wl := strings.Split(got.String(), "\n"), strings.Split(want, "\n")
		for i := 0; i < len(gl) || i < len(wl); i++ {
			var g, w string
			if i < len(gl) {
				g = gl[i]
			}
			if i < len(wl) {
				w = wl[i]
			}
			if g != w {
				t.Fatalf("line %d differs:\n got %q\nwant %q", i+1, g, w)
			}
		}
		t.Fatalf("output differs in length: got %d lines, want %d", len(gl), len(wl))
	}
}

// TestWriteRefWrapsAtSixEntries pins the wrap column with a state wide enough to need it —
// adcdip1.out's main block is only 5 configurations, so it never wraps and cannot prove
// this. The fixture is a real theADCcode state (21 overlaps over four lines), which also
// exercises the right-aligned label: "<21,19|" fills the 7 columns exactly, " <21,9|" and
// "  <5,5|" are padded to it.
func TestWriteRefWrapsAtSixEntries(t *testing.T) {
	const want = ` 3: 24.098894, 81.65, 25.938925
 Overlaps with main-space configurations:
<21,18|:-0.774312<20,19|:-0.287488<20,18|: 0.263260<21,19|: 0.146328<21,15|: 0.110736<19,17|:-0.099694
<18,16|: 0.098024<21,14|: 0.054628<20,14|:-0.040668 <21,9|: 0.039465 <20,9|:-0.036240<21,13|:-0.033225
 <21,6|:-0.024544<21,11|:-0.022208 <20,7|: 0.019923<16,15|:-0.013608<20,15|:-0.012809<16,14|:-0.012561
<18,17|: 0.012052<18,12|:-0.010500 <21,7|:-0.010438

`
	pairs := []struct {
		i, j int
		c    float64
	}{
		{21, 18, -0.774312}, {20, 19, -0.287488}, {20, 18, 0.263260}, {21, 19, 0.146328},
		{21, 15, 0.110736}, {19, 17, -0.099694}, {18, 16, 0.098024}, {21, 14, 0.054628},
		{20, 14, -0.040668}, {21, 9, 0.039465}, {20, 9, -0.036240}, {21, 13, -0.033225},
		{21, 6, -0.024544}, {21, 11, -0.022208}, {20, 7, 0.019923}, {16, 15, -0.013608},
		{20, 15, -0.012809}, {16, 14, -0.012561}, {18, 17, 0.012052}, {18, 12, -0.010500},
		{21, 7, -0.010438},
	}
	st := State{Root: 3, EnergyEV: 24.098894, PSPercent: 81.65, Residue: 25.938925 * au2eV}
	for _, p := range pairs {
		st.Leading = append(st.Leading, Leading{I: p.i, J: p.j, Coeff: p.c})
	}

	var got strings.Builder
	if err := (Sector{States: []State{st}}).WriteRef(&got); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}
	// Drop the four header lines (title, rule, two blanks); this is about the state body.
	body := strings.SplitN(got.String(), "\n", 5)[4]
	if body != want {
		t.Errorf("state body differs:\n got:\n%s\nwant:\n%s", body, want)
	}
}

// TestSIPWriteRefLabelsSingleHole checks the one-hole label and that the shared 17-column
// entry layout is unchanged by it: "<3|" padded to 7 columns is theADCcode's "    <3|",
// as printed in testdata/reference/h2o_dzp.sip.ADC.out.
func TestSIPWriteRefLabelsSingleHole(t *testing.T) {
	sec := SIPSector{States: []SIPState{{
		Root: 1, EnergyEV: 14.882182, PSPercent: 94.10,
		Main: []OrbWeight{{Orbital: 3, Coeff: 0.969693}},
	}}}
	var got strings.Builder
	if err := sec.WriteRef(&got); err != nil {
		t.Fatalf("WriteRef: %v", err)
	}
	const want = " 1: 14.882182, 94.10, 0.000000\n" +
		" Overlaps with main-space configurations:\n" +
		"    <3|: 0.969693\n\n"
	if body := strings.SplitN(got.String(), "\n", 5)[4]; body != want {
		t.Errorf("got:\n%q\nwant:\n%q", body, want)
	}
}
