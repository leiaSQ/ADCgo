package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// uracilW1 is the UW1_N1H basin as GAMESS-UK echoes it into sip.out: uracil
// with one water hydrogen-bonded to the N1-H site. The ">>>>>" prefixes are
// kept deliberately -- reading a Z-matrix straight out of a run's output is the
// path the figures take.
const uracilW1 = `>>>>> zmat angstroms
>>>>> N
>>>>> C 1 1.375755
>>>>> N 2 1.370544 1 114.279419
>>>>> C 3 1.405248 1 94.732670 2 179.758427
>>>>> C 4 1.452074 2 87.650786 1 -0.235971
>>>>> C 5 1.350162 3 86.921053 2 0.112842
>>>>> O 2 1.228024 3 122.856338 5 179.769304
>>>>> O 4 1.218063 3 120.031629 1 179.904843
>>>>> H 1 1.023713 7 87.947141 8 -179.807073
>>>>> H 3 1.012333 7 89.074207 9 179.973201
>>>>> H 5 1.081068 8 94.129275 10 -179.728839
>>>>> H 6 1.084675 9 90.917638 7 179.555628
>>>>> O 9 1.875941 7 76.999576 8 176.783361
>>>>> H 13 0.976455 9 85.330123 7 1.398664
>>>>> H 13 0.960256 14 107.377315 11 -138.001223
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
	g, err := readGeometry(writeTemp(t, "g.zmat", uracilW1))
	if err != nil {
		t.Fatal(err)
	}
	if len(g.pos) != 15 {
		t.Fatalf("got %d atoms, want 15", len(g.pos))
	}

	rows := []struct {
		i, a, b, c       int // 0-based
		dist, ang, dihed float64
	}{
		{1, 0, -1, -1, 1.375755, 0, 0},
		{2, 1, 0, -1, 1.370544, 114.279419, 0},
		{3, 2, 0, 1, 1.405248, 94.732670, 179.758427},
		{6, 1, 2, 4, 1.228024, 122.856338, 179.769304},
		{8, 0, 6, 7, 1.023713, 87.947141, -179.807073},
		{12, 8, 6, 7, 1.875941, 76.999576, 176.783361},
		{14, 12, 13, 10, 0.960256, 107.377315, -138.001223},
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

// TestTopology checks the inferred chemistry: the uracil ring, its substituents
// and the single water, plus the N-H···OH2 contact that names this basin.
func TestTopology(t *testing.T) {
	g, err := readGeometry(writeTemp(t, "g.zmat", uracilW1))
	if err != nil {
		t.Fatal(err)
	}
	bonds := g.bonds()

	// Uracil: 6 ring bonds + 2 C=O + 2 N-H + 2 C-H = 12; water: 2 O-H.
	if len(bonds) != 14 {
		t.Errorf("got %d covalent bonds, want 14 (12 uracil, 2 water O-H)", len(bonds))
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

	// The water bridges: it accepts from N1-H and donates back to the C2
	// carbonyl. That bidentate motif is why the N1H, O2 and O2-N1H starting
	// guesses all relax into one basin.
	hb := g.hydrogenBonds(bonds)
	if len(hb) != 2 {
		t.Fatalf("got %d hydrogen bonds, want 2: %v", len(hb), hb)
	}
	want := [][2]int{{8, 12}, {13, 6}} // N1-H···Ow, and Hw···O(C2); 0-based
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
	g, err := readGeometry(writeTemp(t, "g.zmat", uracilW1))
	if err != nil {
		t.Fatal(err)
	}
	d, err := depict(g)
	if err != nil {
		t.Fatal(err)
	}
	if len(d.pts) != 15 {
		t.Errorf("drew %d atoms, want 15", len(d.pts))
	}
	if len(d.elems) != len(d.pts) {
		t.Fatalf("%d elements for %d points", len(d.elems), len(d.pts))
	}
	counts := map[string]int{}
	for _, el := range d.elems {
		counts[el]++
	}
	for el, want := range map[string]int{"C": 4, "N": 2, "O": 3, "H": 6} {
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
	if len(d.bonds) != 14 {
		t.Errorf("%d bonds drawn, want 14", len(d.bonds))
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
	g, err := readGeometry(writeTemp(t, "g.zmat", uracilW1))
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
