// Geometry for the stack-mode structure insets: read a conformer's geometry,
// infer its bond topology, and flatten it to the 2D ball-and-stick depiction
// drawn in the corner of that conformer's trace.
//
// The waterfall says how a basin's spectrum differs; it cannot say which basin.
// Twenty names like "O2-N1H" only mean something to someone holding the
// numbering convention in their head, so each trace carries a small picture of
// the hydrogen-bonding motif it belongs to, generated from the same geometry
// the calculation ran on -- never drawn by hand, so it cannot disagree with it.
//
// The covalent radii, the 1.3 bond tolerance and the 1.6-2.2 Å hydrogen-bond
// window are HeidelBIRDS' (mol.CovRadii, coord.bondScale, mol.DefaultHBondDmin
// /Dmax), so the topology drawn here is the topology the structure search used.
package main

import (
	"bufio"
	"fmt"
	"image/color"
	"math"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

const (
	// bondScale multiplies the sum of covalent radii to decide a bond.
	bondScale = 1.3
	// Hydrogen-bond donor-H···acceptor window, Å.
	hbondMin = 1.6
	hbondMax = 2.2
)

// covRadii maps element symbol to covalent radius (Å), H through Kr; anything
// heavier falls back to covRadiiDefault, which is only ever used for a bond
// guess in a picture.
var covRadii = map[string]float64{
	"H": 0.32, "He": 0.37, "Li": 1.3, "Be": 0.99, "B": 0.84, "C": 0.75,
	"N": 0.71, "O": 0.64, "F": 0.6, "Ne": 0.62, "Na": 1.6, "Mg": 1.4,
	"Al": 1.24, "Si": 1.14, "P": 1.09, "S": 1.04, "Cl": 1.0, "Ar": 1.01,
	"K": 2.0, "Ca": 1.74, "Sc": 1.59, "Ti": 1.48, "V": 1.44, "Cr": 1.3,
	"Mn": 1.29, "Fe": 1.24, "Co": 1.18, "Ni": 1.17, "Cu": 1.22, "Zn": 1.2,
	"Ga": 1.23, "Ge": 1.2, "As": 1.2, "Se": 1.18, "Br": 1.17, "Kr": 1.16,
}

const covRadiiDefault = 0.77

// polar is the set of elements whose bonded hydrogens are drawn (and that can
// accept a hydrogen bond).
var polar = map[string]bool{"N": true, "O": true, "F": true}

const bohrToAngstrom = 0.529177210903

type vec3 = [3]float64

// geometry is a molecular geometry in Cartesian coordinates (Å).
type geometry struct {
	labels []string
	pos    []vec3
}

// readGeometry reads a geometry file, sniffing XYZ (a leading atom count)
// from a GAMESS-UK Z-matrix. Lines echoed by the GAMESS input parser keep a
// ">>>>>" prefix, which is stripped either way, so a Z-matrix lifted straight
// out of a run's sip.out parses as-is.
func readGeometry(path string) (*geometry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		l := strings.TrimSpace(sc.Text())
		l = strings.TrimSpace(strings.TrimPrefix(l, ">>>>>"))
		lines = append(lines, l)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}

	for _, l := range lines {
		if l == "" {
			continue
		}
		if _, err := strconv.Atoi(strings.Fields(l)[0]); err == nil {
			g, err := parseXYZ(lines)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", path, err)
			}
			return g, nil
		}
		break
	}
	g, err := parseZMat(lines)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return g, nil
}

// parseXYZ reads a standard XYZ block: count, comment, then "El x y z" rows.
func parseXYZ(lines []string) (*geometry, error) {
	var body []string
	for _, l := range lines {
		if l != "" {
			body = append(body, l)
		}
	}
	if len(body) < 2 {
		return nil, fmt.Errorf("xyz: too short")
	}
	n, err := strconv.Atoi(strings.Fields(body[0])[0])
	if err != nil {
		return nil, fmt.Errorf("xyz: atom count: %w", err)
	}
	body = body[2:] // count + comment
	if len(body) < n {
		return nil, fmt.Errorf("xyz: want %d atoms, found %d", n, len(body))
	}
	g := &geometry{}
	for _, l := range body[:n] {
		f := strings.Fields(l)
		if len(f) < 4 {
			return nil, fmt.Errorf("xyz: short row %q", l)
		}
		var p vec3
		for k := 0; k < 3; k++ {
			if p[k], err = strconv.ParseFloat(f[k+1], 64); err != nil {
				return nil, fmt.Errorf("xyz: row %q: %w", l, err)
			}
		}
		g.labels = append(g.labels, normalizeElement(f[0]))
		g.pos = append(g.pos, p)
	}
	return g, nil
}

// zRow is one Z-matrix row; refs are 0-based, -1 when absent.
type zRow struct {
	label            string
	ra, rb, rc       int
	dist, ang, dihed float64
}

// parseZMat reads a GAMESS-UK Z-matrix and places it in Cartesian space. The
// header line ("zmat angstroms" / "zmat bohr") sets the length unit; the
// "variables" / "constants" / "end" keywords close the block. Every geometry
// here is fully numeric -- the searches write explicit values, not symbols --
// so a symbolic Z-matrix is rejected rather than half-read.
func parseZMat(lines []string) (*geometry, error) {
	scale := 1.0
	var rows []zRow
	for _, l := range lines {
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		f := strings.Fields(l)
		switch strings.ToLower(f[0]) {
		case "zmat":
			for _, tok := range f[1:] {
				if strings.HasPrefix(strings.ToLower(tok), "bohr") ||
					strings.HasPrefix(strings.ToLower(tok), "au") {
					scale = bohrToAngstrom
				}
			}
			continue
		case "variables", "constants", "end":
			// "end" closes the block; the keyword sections that may precede it
			// carry no geometry for a fully numeric Z-matrix.
			if strings.ToLower(f[0]) == "end" {
				goto done
			}
			continue
		}

		r := zRow{label: normalizeElement(f[0]), ra: -1, rb: -1, rc: -1}
		// Rows are "El", "El i r", "El i r j a", "El i r j a k d".
		var err error
		get := func(idx int) (int, float64, error) {
			ref, err := strconv.Atoi(f[idx])
			if err != nil {
				return 0, 0, fmt.Errorf("row %q: reference %q: %w", l, f[idx], err)
			}
			v, err := strconv.ParseFloat(f[idx+1], 64)
			if err != nil {
				return 0, 0, fmt.Errorf("row %q: value %q (symbolic Z-matrices are not supported): %w", l, f[idx+1], err)
			}
			return ref - 1, v, nil
		}
		if len(f) >= 3 {
			if r.ra, r.dist, err = get(1); err != nil {
				return nil, err
			}
			r.dist *= scale
		}
		if len(f) >= 5 {
			if r.rb, r.ang, err = get(3); err != nil {
				return nil, err
			}
		}
		if len(f) >= 7 {
			if r.rc, r.dihed, err = get(5); err != nil {
				return nil, err
			}
		}
		rows = append(rows, r)
	}
done:
	if len(rows) == 0 {
		return nil, fmt.Errorf("zmat: no atoms")
	}
	return zmatToCartesian(rows)
}

// zmatToCartesian places Z-matrix rows by natural extension (NeRF): each atom
// sits at distance r from its bond reference, at angle θ to the angle
// reference, and at dihedral φ about the bond-angle axis.
func zmatToCartesian(rows []zRow) (*geometry, error) {
	g := &geometry{
		labels: make([]string, len(rows)),
		pos:    make([]vec3, len(rows)),
	}
	for i, r := range rows {
		g.labels[i] = r.label
		switch {
		case i == 0:
			g.pos[i] = vec3{0, 0, 0}
		case i == 1:
			if r.ra != 0 {
				return nil, fmt.Errorf("zmat: row 2 must reference atom 1")
			}
			g.pos[i] = vec3{r.dist, 0, 0}
		case i == 2:
			if r.ra < 0 || r.rb < 0 {
				return nil, fmt.Errorf("zmat: row 3 needs a bond and an angle reference")
			}
			a, b := g.pos[r.ra], g.pos[r.rb]
			u := normalize(sub(a, b))
			// Any axis not parallel to u serves to fix the (arbitrary) plane of
			// the first three atoms.
			t := vec3{0, 0, 1}
			if math.Abs(dot(u, t)) > 0.9 {
				t = vec3{0, 1, 0}
			}
			n := normalize(cross(u, t))
			p := cross(n, u)
			th := r.ang * math.Pi / 180
			g.pos[i] = add(a, add(scale(u, -r.dist*math.Cos(th)), scale(p, r.dist*math.Sin(th))))
		default:
			if r.ra < 0 || r.rb < 0 || r.rc < 0 {
				return nil, fmt.Errorf("zmat: row %d needs three references", i+1)
			}
			a, b, c := g.pos[r.ra], g.pos[r.rb], g.pos[r.rc]
			u := normalize(sub(a, b)) // angle ref -> bond ref
			n := cross(sub(b, c), u)
			if norm(n) < 1e-9 {
				return nil, fmt.Errorf("zmat: row %d references three collinear atoms", i+1)
			}
			n = normalize(n)
			p := cross(n, u)
			// The dihedral is negated because the frame below is built from
			// u = A-B (pointing *towards* the bond reference); measuring the
			// placed atom back with dihedralDeg without it mirrors the
			// molecule, which TestZMatRoundTrip pins down.
			th, ph := r.ang*math.Pi/180, -r.dihed*math.Pi/180
			d := add(
				scale(u, -r.dist*math.Cos(th)),
				add(
					scale(p, r.dist*math.Sin(th)*math.Cos(ph)),
					scale(n, r.dist*math.Sin(th)*math.Sin(ph)),
				),
			)
			g.pos[i] = add(a, d)
		}
	}
	return g, nil
}

// normalizeElement turns "N1", "o", "CL2" into "N", "O", "Cl": run geometries
// often tag an element with its site number.
func normalizeElement(s string) string {
	var letters []rune
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' {
			letters = append(letters, r)
			continue
		}
		break
	}
	if len(letters) == 0 {
		return s
	}
	out := strings.ToUpper(string(letters[:1]))
	if len(letters) > 1 {
		out += strings.ToLower(string(letters[1:]))
	}
	return out
}

func radius(el string) float64 {
	if r, ok := covRadii[el]; ok {
		return r
	}
	return covRadiiDefault
}

// bonds infers the covalent bond list from covalent radii.
func (g *geometry) bonds() [][2]int {
	var out [][2]int
	for i := range g.pos {
		for j := i + 1; j < len(g.pos); j++ {
			cut := bondScale * (radius(g.labels[i]) + radius(g.labels[j]))
			if dist(g.pos[i], g.pos[j]) < cut {
				out = append(out, [2]int{i, j})
			}
		}
	}
	return out
}

// hydrogenBonds finds donor-H···acceptor contacts: a hydrogen covalently bound
// to a polar element, and a polar acceptor it is *not* bound to, inside the
// HeidelBIRDS contact window.
func (g *geometry) hydrogenBonds(bonds [][2]int) [][2]int {
	adj := adjacency(len(g.pos), bonds)
	bonded := map[[2]int]bool{}
	for _, b := range bonds {
		bonded[b] = true
		bonded[[2]int{b[1], b[0]}] = true
	}
	var out [][2]int
	for h := range g.pos {
		if g.labels[h] != "H" {
			continue
		}
		donor := false
		for _, nb := range adj[h] {
			if polar[g.labels[nb]] {
				donor = true
			}
		}
		if !donor {
			continue
		}
		for a := range g.pos {
			if a == h || !polar[g.labels[a]] || bonded[[2]int{h, a}] {
				continue
			}
			if d := dist(g.pos[h], g.pos[a]); d >= hbondMin && d <= hbondMax {
				out = append(out, [2]int{h, a})
			}
		}
	}
	return out
}

func adjacency(n int, bonds [][2]int) [][]int {
	adj := make([][]int, n)
	for _, b := range bonds {
		adj[b[0]] = append(adj[b[0]], b[1])
		adj[b[1]] = append(adj[b[1]], b[0])
	}
	return adj
}

// components labels each atom with the index of its covalently connected
// fragment, 0 being the largest (the solute).
func components(n int, bonds [][2]int) []int {
	adj := adjacency(n, bonds)
	comp := make([]int, n)
	for i := range comp {
		comp[i] = -1
	}
	var sizes []int
	next := 0
	for i := range comp {
		if comp[i] >= 0 {
			continue
		}
		stack, size := []int{i}, 0
		comp[i] = next
		for len(stack) > 0 {
			v := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			size++
			for _, w := range adj[v] {
				if comp[w] < 0 {
					comp[w] = next
					stack = append(stack, w)
				}
			}
		}
		sizes = append(sizes, size)
		next++
	}
	// Renumber so 0 is the largest fragment.
	big := 0
	for i, s := range sizes {
		if s > sizes[big] {
			big = i
		}
	}
	for i := range comp {
		switch comp[i] {
		case big:
			comp[i] = 0
		case 0:
			comp[i] = big
		}
	}
	return comp
}

// depiction is a flattened, unit-scaled 2D ball-and-stick drawing.
type depiction struct {
	pts    [][2]float64 // one per atom, in [0,1]² preserving aspect
	elems  []string     // element symbol per point, for colour and radius
	bonds  [][2]int     // indices into pts
	hbonds [][2]int     // indices into pts
}

// depict flattens a geometry into the 2D ball-and-stick drawing: every atom
// becomes a ball, coloured and sized by element, joined by its covalent bonds,
// with hydrogen bonds carried alongside for the renderer to dash.
func depict(g *geometry) (*depiction, error) {
	if len(g.pos) < 2 {
		return nil, fmt.Errorf("depict: fewer than two atoms to draw")
	}
	bonds := g.bonds()

	all := make([]int, len(g.pos))
	for i := range all {
		all[i] = i
	}
	d := &depiction{
		elems:  append([]string(nil), g.labels...),
		bonds:  bonds,
		hbonds: g.hydrogenBonds(bonds),
	}

	xy := projectPlane(g, all)
	orient(xy, g, all, bonds)
	d.pts = normalizeBox(xy)
	return d, nil
}

// cpk is the element style of the structure insets: the conventional CPK
// colouring, muted a little for paper, with the ball radius following the
// covalent radius so hydrogens read as the small atoms they are.
var cpk = map[string]struct {
	col    color.RGBA
	radius float64 // relative to carbon
}{
	"H": {color.RGBA{0xd4, 0xd4, 0xd0, 0xff}, 0.55},
	"C": {color.RGBA{0x3a, 0x3a, 0x38, 0xff}, 1.00},
	"N": {color.RGBA{0x24, 0x5b, 0xc8, 0xff}, 0.95},
	"O": {color.RGBA{0xcf, 0x2f, 0x2a, 0xff}, 0.92},
	"S": {color.RGBA{0xd8, 0xb4, 0x27, 0xff}, 1.10},
	"F": {color.RGBA{0x5f, 0xb0, 0x4f, 0xff}, 0.85},
}

// cpkDefault styles an element the table does not name.
var cpkDefault = struct {
	col    color.RGBA
	radius float64
}{color.RGBA{0x8a, 0x6f, 0xb0, 0xff}, 1.05}

// cpkStyle returns the ball colour and relative radius for an element.
func cpkStyle(el string) (color.RGBA, float64) {
	if s, ok := cpk[el]; ok {
		return s.col, s.radius
	}
	return cpkDefault.col, cpkDefault.radius
}

// projectPlane drops the geometry onto its own best-fit plane: the two leading
// principal axes of the drawn atoms. These systems are near-planar, so this is
// the view that shows the ring and its hydrogen bonds without foreshortening,
// and it needs no per-conformer hand-tuning.
func projectPlane(g *geometry, kept []int) [][2]float64 {
	var c vec3
	for _, i := range kept {
		c = add(c, g.pos[i])
	}
	c = scale(c, 1/float64(len(kept)))

	cov := mat.NewSymDense(3, nil)
	for _, i := range kept {
		d := sub(g.pos[i], c)
		for a := 0; a < 3; a++ {
			for b := a; b < 3; b++ {
				cov.SetSym(a, b, cov.At(a, b)+d[a]*d[b])
			}
		}
	}

	var eig mat.EigenSym
	axes := [2]vec3{{1, 0, 0}, {0, 1, 0}}
	if eig.Factorize(cov, true) {
		var vecs mat.Dense
		eig.VectorsTo(&vecs)
		// EigenSym returns ascending eigenvalues: the last two columns are the
		// in-plane axes, the first is the plane normal.
		for k, col := range [2]int{2, 1} {
			axes[k] = vec3{vecs.At(0, col), vecs.At(1, col), vecs.At(2, col)}
		}
	}

	out := make([][2]float64, len(kept))
	for n, i := range kept {
		d := sub(g.pos[i], c)
		out[n] = [2]float64{dot(d, axes[0]), dot(d, axes[1])}
	}
	return out
}

// orient fixes the two sign ambiguities the principal axes leave, so the same
// motif is drawn the same way up in every panel: solvent fragments are put to
// the right of the solute and above it. A geometry with no separate fragment
// (bare solute) is left as the projection found it.
func orient(xy [][2]float64, g *geometry, kept []int, bonds [][2]int) {
	comp := components(len(g.pos), bonds)
	var solute, solvent [2]float64
	var ns, nv int
	for n, i := range kept {
		if comp[i] == 0 {
			solute[0] += xy[n][0]
			solute[1] += xy[n][1]
			ns++
		} else {
			solvent[0] += xy[n][0]
			solvent[1] += xy[n][1]
			nv++
		}
	}
	if ns == 0 || nv == 0 {
		return
	}
	for k := 0; k < 2; k++ {
		if solvent[k]/float64(nv) < solute[k]/float64(ns) {
			for n := range xy {
				xy[n][k] = -xy[n][k]
			}
		}
	}
}

// normalizeBox rescales into [0,1]² with the aspect ratio preserved: the long
// axis fills the box and the short one is centred in it, so the renderer can
// map the unit square onto any square region and get an undistorted molecule.
func normalizeBox(xy [][2]float64) [][2]float64 {
	lo := [2]float64{math.Inf(1), math.Inf(1)}
	hi := [2]float64{math.Inf(-1), math.Inf(-1)}
	for _, p := range xy {
		for k := 0; k < 2; k++ {
			lo[k] = math.Min(lo[k], p[k])
			hi[k] = math.Max(hi[k], p[k])
		}
	}
	w, h := hi[0]-lo[0], hi[1]-lo[1]
	s := math.Max(w, h)
	if s <= 0 {
		s = 1
	}
	out := make([][2]float64, len(xy))
	for i, p := range xy {
		// Centre the short axis so the drawing sits in the middle of its box.
		out[i] = [2]float64{
			(p[0] - lo[0] + (s-w)/2) / s,
			(p[1] - lo[1] + (s-h)/2) / s,
		}
	}
	return out
}

// Small vector helpers; vec3 is a value type, so these stay allocation-free.
func add(a, b vec3) vec3     { return vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]} }
func addv(a, b vec3) vec3    { return add(a, b) }
func sub(a, b vec3) vec3     { return vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]} }
func dot(a, b vec3) float64  { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
func norm(a vec3) float64    { return math.Sqrt(dot(a, a)) }
func dist(a, b vec3) float64 { return norm(sub(a, b)) }

func scale(a vec3, s float64) vec3 { return vec3{a[0] * s, a[1] * s, a[2] * s} }

func cross(a, b vec3) vec3 {
	return vec3{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

func normalize(a vec3) vec3 {
	n := norm(a)
	if n == 0 {
		return a
	}
	return scale(a, 1/n)
}

// angleDeg and dihedralDeg re-measure internal coordinates from Cartesians;
// the Z-matrix round-trip test uses them to check the placement above.
func angleDeg(a, b, c vec3) float64 {
	u, v := normalize(sub(a, b)), normalize(sub(c, b))
	return math.Acos(math.Max(-1, math.Min(1, dot(u, v)))) * 180 / math.Pi
}

func dihedralDeg(a, b, c, d vec3) float64 {
	b1, b2, b3 := sub(b, a), sub(c, b), sub(d, c)
	n1, n2 := cross(b1, b2), cross(b2, b3)
	m := cross(n1, normalize(b2))
	return math.Atan2(dot(m, n2), dot(n1, n2)) * 180 / math.Pi
}
