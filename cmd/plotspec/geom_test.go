package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// formamideW1 is formamide with one water bridging its cis N-H and its carbonyl,
// written as GAMESS-UK echoes a Z-matrix into its output. The ">>>>>" prefixes are
// kept deliberately -- reading a Z-matrix straight out of a run's output is the
// path the figures take. The free water hydrogen sits out of the molecular plane
// so the round trip below exercises a dihedral that is neither 0 nor 180 degrees.
const formamideW1 = `>>>>> zmat angstroms
>>>>> C
>>>>> O 1 1.220000
>>>>> N 1 1.360349 2 124.729783
>>>>> H 1 1.100172 2 121.999863 3 179.750000
>>>>> H 3 1.010046 1 119.948325 2 0.000000
>>>>> H 3 1.010415 1 120.017942 2 180.000000
>>>>> O 5 1.999634 3 139.092368 1 0.000000
>>>>> H 7 0.970277 5 60.847977 3 0.000000
>>>>> H 7 0.969153 8 104.464819 5 -165.000000
>>>>> variables
>>>>> constants
>>>>> end
`

func writeTemp(t *testing.T, name, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestZMatRoundTrip is the gate on the Cartesian placement: every distance,
// angle and dihedral re-measured from the built geometry must reproduce the
// Z-matrix it came from. A sign slip in the dihedral convention would mirror
// the molecule and silently draw the wrong enantiomeric view.
func TestZMatRoundTrip(t *testing.T) {
	g, err := readGeometry(writeTemp(t, "g.zmat", formamideW1))
	if err != nil {
		t.Fatal(err)
	}
	if len(g.pos) != 9 {
		t.Fatalf("got %d atoms, want 9", len(g.pos))
	}

	rows := []struct {
		i, a, b, c       int // 0-based
		dist, ang, dihed float64
	}{
		{1, 0, -1, -1, 1.220000, 0, 0},
		{2, 0, 1, -1, 1.360349, 124.729783, 0},
		{3, 0, 1, 2, 1.100172, 121.999863, 179.750000},
		{4, 2, 0, 1, 1.010046, 119.948325, 0},
		{6, 4, 2, 0, 1.999634, 139.092368, 0},
		{8, 6, 7, 4, 0.969153, 104.464819, -165.000000},
	}
	for _, r := range rows {
		if got := dist(g.pos[r.i], g.pos[r.a]); math.Abs(got-r.dist) > 1e-6 {
			t.Errorf("atom %d distance = %.6f, want %.6f", r.i+1, got, r.dist)
		}
		if r.b < 0 {
			continue
		}
		if got := angleDeg(g.pos[r.i], g.pos[r.a], g.pos[r.b]); math.Abs(got-r.ang) > 1e-4 {
			t.Errorf("atom %d angle = %.6f, want %.6f", r.i+1, got, r.ang)
		}
		if r.c < 0 {
			continue
		}
		got := dihedralDeg(g.pos[r.i], g.pos[r.a], g.pos[r.b], g.pos[r.c])
		if d := math.Abs(math.Mod(got-r.dihed+540, 360) - 180); d > 1e-4 {
			t.Errorf("atom %d dihedral = %.6f, want %.6f", r.i+1, got, r.dihed)
		}
	}
}

// TestTopology checks the inferred chemistry: the formamide skeleton, the single
// water, and the two hydrogen bonds that close the cyclic complex.
func TestTopology(t *testing.T) {
	g, err := readGeometry(writeTemp(t, "g.zmat", formamideW1))
	if err != nil {
		t.Fatal(err)
	}
	bonds := g.bonds()

	// Formamide: C=O + C-N + C-H + 2 N-H = 5; water: 2 O-H.
	if len(bonds) != 7 {
		t.Errorf("got %d covalent bonds, want 7 (5 formamide, 2 water O-H)", len(bonds))
	}
	comp := components(len(g.pos), bonds)
	nSolvent := 0
	for _, c := range comp {
		if c != 0 {
			nSolvent++
		}
	}
	if nSolvent != 3 {
		t.Errorf("solvent fragment has %d atoms, want 3 (one water)", nSolvent)
	}

	// The water bridges: it accepts from the N-H cis to the carbonyl and donates
	// back to the carbonyl oxygen.
	hb := g.hydrogenBonds(bonds)
	if len(hb) != 2 {
		t.Fatalf("got %d hydrogen bonds, want 2: %v", len(hb), hb)
	}
	want := [][2]int{{4, 6}, {7, 1}} // N-H···Ow, and Hw···O=C; 0-based
	for i, w := range want {
		if hb[i] != w {
			t.Errorf("hydrogen bond %d = %v, want %v", i, hb[i], w)
		}
		if g.labels[hb[i][0]] != "H" || g.labels[hb[i][1]] != "O" {
			t.Errorf("hydrogen bond %d is %s···%s, want H···O",
				i, g.labels[hb[i][0]], g.labels[hb[i][1]])
		}
	}
}

// TestDepict checks the ball-and-stick reduction: every atom is drawn, carries
// an element style, and lands in the unit box.
func TestDepict(t *testing.T) {
	g, err := readGeometry(writeTemp(t, "g.zmat", formamideW1))
	if err != nil {
		t.Fatal(err)
	}
	d, err := depict(g)
	if err != nil {
		t.Fatal(err)
	}
	if len(d.pts) != 9 {
		t.Errorf("drew %d atoms, want 9", len(d.pts))
	}
	if len(d.elems) != len(d.pts) {
		t.Fatalf("%d elements for %d points", len(d.elems), len(d.pts))
	}
	counts := map[string]int{}
	for _, el := range d.elems {
		counts[el]++
	}
	for el, want := range map[string]int{"C": 1, "N": 1, "O": 2, "H": 5} {
		if counts[el] != want {
			t.Errorf("%d %s atoms, want %d", counts[el], el, want)
		}
	}
	// Every element must resolve to a style; an unknown one would silently
	// come out in the fallback colour.
	for _, el := range d.elems {
		if _, ok := cpk[el]; !ok {
			t.Errorf("element %q has no CPK style", el)
		}
	}
	if len(d.bonds) != 7 {
		t.Errorf("%d bonds drawn, want 7", len(d.bonds))
	}
	if len(d.hbonds) != 2 {
		t.Errorf("%d hydrogen bonds drawn, want 2", len(d.hbonds))
	}
	for i, p := range d.pts {
		if p[0] < -1e-9 || p[0] > 1+1e-9 || p[1] < -1e-9 || p[1] > 1+1e-9 {
			t.Errorf("point %d = %v, outside the unit box", i, p)
		}
	}
}

// TestPlanarProjection checks that the best-fit plane really is the molecular
// plane, by confirming the out-of-plane spread is tiny next to the in-plane one.
func TestPlanarProjection(t *testing.T) {
	g, err := readGeometry(writeTemp(t, "g.zmat", formamideW1))
	if err != nil {
		t.Fatal(err)
	}
	kept := make([]int, len(g.pos))
	for i := range kept {
		kept[i] = i
	}
	xy := projectPlane(g, kept)

	// In-plane extent from the projection; total extent from the raw geometry.
	var maxIn float64
	for _, p := range xy {
		maxIn = math.Max(maxIn, math.Hypot(p[0], p[1]))
	}
	var c vec3
	for _, p := range g.pos {
		c = add(c, p)
	}
	c = scale(c, 1/float64(len(g.pos)))
	var lost float64
	for i, p := range g.pos {
		r := norm(sub(p, c))
		inPlane := math.Hypot(xy[i][0], xy[i][1])
		lost = math.Max(lost, math.Sqrt(math.Max(0, r*r-inPlane*inPlane)))
	}
	if lost > 0.1*maxIn {
		t.Errorf("out-of-plane spread %.3f Å is %.0f%% of the in-plane extent %.3f Å; "+
			"the projection is not finding the molecular plane", lost, 100*lost/maxIn, maxIn)
	}
}

// TestNormalizeElement covers the site-numbered and all-caps spellings that
// turn up in run geometries.
func TestNormalizeElement(t *testing.T) {
	for in, want := range map[string]string{
		"N": "N", "n": "N", "O2": "O", "CL": "Cl", "cl3": "Cl",
		"C": "C", "H12": "H", "CA": "Ca", // a two-letter element wins over an all-caps one-letter reading
	} {
		if got := normalizeElement(in); got != want {
			t.Errorf("normalizeElement(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestReadXYZ checks the other accepted geometry format.
func TestReadXYZ(t *testing.T) {
	const xyz = `3
water
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
`
	g, err := readGeometry(writeTemp(t, "w.xyz", xyz))
	if err != nil {
		t.Fatal(err)
	}
	if len(g.pos) != 3 || g.labels[0] != "O" {
		t.Fatalf("got %d atoms starting %q, want 3 starting O", len(g.pos), g.labels[0])
	}
	if b := g.bonds(); len(b) != 2 {
		t.Errorf("got %d bonds, want 2", len(b))
	}
}
