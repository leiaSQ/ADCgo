package fcidump

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func loadData(t *testing.T) *Data {
	t.Helper()
	d, err := ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	return d
}

func TestHeader(t *testing.T) {
	d := loadData(t)
	if d.NORB != 24 {
		t.Errorf("NORB = %d, want 24", d.NORB)
	}
	if d.NELEC != 10 {
		t.Errorf("NELEC = %d, want 10", d.NELEC)
	}
	if d.MS2 != 0 {
		t.Errorf("MS2 = %d, want 0", d.MS2)
	}
	if len(d.OrbSym) != 24 {
		t.Errorf("len(OrbSym) = %d, want 24", len(d.OrbSym))
	}
}

// TestEcore: with no frozen core the FCIDUMP core energy is the nuclear
// repulsion, which the reference records as e_nuc.
func TestEcore(t *testing.T) {
	d := loadData(t)
	b, err := os.ReadFile("../../../testdata/h2o.ref.json")
	if err != nil {
		t.Fatalf("read reference: %v", err)
	}
	var r struct {
		ENuc float64 `json:"e_nuc"`
	}
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	if diff := math.Abs(d.Ecore - r.ENuc); diff > 1e-9 {
		t.Errorf("Ecore = %.12f, want e_nuc %.12f (|Δ|=%.2e)", d.Ecore, r.ENuc, diff)
	}
}

func TestPermutationSymmetry(t *testing.T) {
	d := loadData(t)
	// One-electron integrals are symmetric.
	for p := 0; p < d.NORB; p++ {
		for q := 0; q < d.NORB; q++ {
			if d.OneE(p, q) != d.OneE(q, p) {
				t.Fatalf("h not symmetric at (%d,%d)", p, q)
			}
		}
	}
	// Two-electron 8-fold symmetry on a representative index tuple.
	p, q, r, s := 0, 3, 5, 2
	v := d.TwoE(p, q, r, s)
	perms := [][4]int{
		{q, p, r, s}, {p, q, s, r}, {q, p, s, r},
		{r, s, p, q}, {s, r, p, q}, {r, s, q, p}, {s, r, q, p},
	}
	for _, pm := range perms {
		if got := d.TwoE(pm[0], pm[1], pm[2], pm[3]); got != v {
			t.Errorf("(%d%d|%d%d)=%.10f but permutation %v=%.10f", p, q, r, s, v, pm, got)
		}
	}
}

// readSerial is a deliberately naive, line-at-a-time reference parser: the exact
// algorithm Read used before the body was split into concurrently parsed blocks.
// It exists only so TestConcurrentParseMatchesSerial can pin the new parser's
// output to the old semantics byte for byte.
func readSerial(t *testing.T, path string) *Data {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)

	var header strings.Builder
	var body []string
	inHeader, headerDone := false, false
	for sc.Scan() {
		tl := strings.TrimSpace(sc.Text())
		if headerDone {
			if tl != "" {
				body = append(body, tl)
			}
			continue
		}
		up := strings.ToUpper(tl)
		if !inHeader {
			if strings.HasPrefix(up, "&FCI") || strings.HasPrefix(up, "$FCI") {
				inHeader = true
				header.WriteString(" " + tl)
			}
			continue
		}
		if up == "&END" || up == "$END" || up == "/" || strings.HasSuffix(up, "&END") {
			headerDone = true
			continue
		}
		header.WriteString(" " + tl)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan %s: %v", path, err)
	}
	if !inHeader {
		t.Fatalf("%s: no &FCI header", path)
	}

	hstr := header.String()
	d := &Data{}
	d.NORB, _ = parseInt(hstr, "NORB")
	d.NELEC, _ = parseInt(hstr, "NELEC")
	d.MS2, _ = parseInt(hstr, "MS2")
	d.ISYM, _ = parseInt(hstr, "ISYM")
	d.OrbSym = parseOrbSym(hstr, d.NORB)
	n := d.NORB
	d.h = make([]float64, n*n)
	d.eri = make([]float64, n*n*n*n)

	for _, ln := range body {
		fields := strings.Fields(ln)
		if len(fields) < 5 {
			continue
		}
		val, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			t.Fatalf("%s: bad value %q", path, fields[0])
		}
		idx := make([]int, 4)
		for k := 0; k < 4; k++ {
			if idx[k], err = strconv.Atoi(fields[k+1]); err != nil {
				t.Fatalf("%s: bad index %q", path, fields[k+1])
			}
		}
		i, j, kk, l := idx[0], idx[1], idx[2], idx[3]
		switch {
		case i == 0 && j == 0 && kk == 0 && l == 0:
			d.Ecore = val
		case kk == 0 && l == 0:
			d.setOne(i-1, j-1, val)
		default:
			d.setTwo(i-1, j-1, kk-1, l-1, val)
		}
	}
	return d
}

// assertSameData compares two parses element by element with ==, not a tolerance:
// the FCIDUMP feeds every downstream number, so the concurrent parse has to be
// bit-identical, not merely close.
func assertSameData(t *testing.T, what string, got, want *Data) {
	t.Helper()
	if got.NORB != want.NORB || got.NELEC != want.NELEC || got.MS2 != want.MS2 || got.ISYM != want.ISYM {
		t.Fatalf("%s: header mismatch: got %+v want %+v",
			what, [4]int{got.NORB, got.NELEC, got.MS2, got.ISYM},
			[4]int{want.NORB, want.NELEC, want.MS2, want.ISYM})
	}
	if len(got.OrbSym) != len(want.OrbSym) {
		t.Fatalf("%s: %d OrbSym entries, want %d", what, len(got.OrbSym), len(want.OrbSym))
	}
	for i := range want.OrbSym {
		if got.OrbSym[i] != want.OrbSym[i] {
			t.Fatalf("%s: OrbSym[%d] = %d, want %d", what, i, got.OrbSym[i], want.OrbSym[i])
		}
	}
	if got.Ecore != want.Ecore {
		t.Fatalf("%s: Ecore = %.17g, want %.17g", what, got.Ecore, want.Ecore)
	}
	if len(got.h) != len(want.h) || len(got.eri) != len(want.eri) {
		t.Fatalf("%s: sizes h=%d/%d eri=%d/%d", what,
			len(got.h), len(want.h), len(got.eri), len(want.eri))
	}
	for i := range want.h {
		if got.h[i] != want.h[i] {
			t.Fatalf("%s: h[%d] = %.17g, want %.17g", what, i, got.h[i], want.h[i])
		}
	}
	for i := range want.eri {
		if got.eri[i] != want.eri[i] {
			t.Fatalf("%s: eri[%d] = %.17g, want %.17g", what, i, got.eri[i], want.eri[i])
		}
	}
}

// TestConcurrentParseMatchesSerial is the gate on the block-parallel body parse.
// It is not a formality: h2o.fcidump lists 21783 of its 23731 distinct two-electron
// integrals twice, as (pq|rs) and again as (rs|pq), and every one of those pairs
// prints two values that differ in the last ulp. Those lines expand to the same
// eight slots, so the stored bits depend on which line is applied last — this test
// is what pins the store to file order. Both dumps span several blocks, so the parse
// really is split across goroutines here.
func TestConcurrentParseMatchesSerial(t *testing.T) {
	for _, name := range []string{"h2o.fcidump", "h2o_dzp.fcidump"} {
		path := "../../../testdata/" + name
		st, err := os.Stat(path)
		if err != nil {
			t.Fatalf("stat %s: %v", name, err)
		}
		if st.Size() <= blockBytes {
			t.Fatalf("%s is %d bytes, at most one block (%d): the parse would not "+
				"be split and this test would prove nothing", name, st.Size(), blockBytes)
		}
		want := readSerial(t, path)
		got, err := ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		assertSameData(t, name, got, want)

		// Same input twice: block completion order varies run to run, so this
		// catches any residual dependence on scheduling.
		again, err := ReadFile(path)
		if err != nil {
			t.Fatalf("re-read %s: %v", name, err)
		}
		assertSameData(t, name+" (repeat)", again, got)
	}
}

// TestReadErrors pins the messages and the header handling the parser is expected
// to produce; the concurrent body parse must still report the FIRST malformed line
// in file order, not whichever goroutine happened to fail first.
func TestReadErrors(t *testing.T) {
	const hdr = "&FCI NORB=2,NELEC=2\n&END\n"
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"no header", "1.0 1 1 0 0\n", "fcidump: no &FCI header found"},
		{"no norb", "&FCI NELEC=2\n/\n1.0 1 1 0 0\n", "fcidump: NORB not found in header"},
		{"no nelec", "&FCI NORB=2\n/\n1.0 1 1 0 0\n", "fcidump: NELEC not found in header"},
		{"bad value", hdr + "1.0 1 1 0 0\nxx 2 2 0 0\n", `fcidump: bad value "xx"`},
		{"bad index", hdr + "1.0 1 1 0 0\n2.0 2 y 0 0\n", `fcidump: bad index "y"`},
		{"first bad line wins", hdr + "1.0 a 1 0 0\n2.0 b 2 0 0\n",
			`fcidump: bad index "a"`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Read(strings.NewReader(c.in))
			if err == nil {
				t.Fatalf("no error, want %q", c.want)
			}
			if !strings.HasPrefix(err.Error(), c.want) {
				t.Fatalf("error %q, want prefix %q", err.Error(), c.want)
			}
		})
	}
}

// TestReadSemantics covers the body-splitting edge cases the block reader has to
// get right and that the whole-file line buffer got for free: a final line with no
// newline, blank and short lines, CRLF, and last-wins for the core energy.
func TestReadSemantics(t *testing.T) {
	in := "&FCI NORB=2,NELEC=2,MS2=0,ISYM=1\r\n ORBSYM=1,1\r\n&END\r\n" +
		"\r\n" + // blank line
		"9.0 0 0 0 0\r\n" + // Ecore, later overwritten
		"short line\r\n" + // <5 fields: skipped
		"0.5 1 1 2 2\r\n" +
		"0.25 2 1 0 0\r\n" +
		"7.5 0 0 0 0" // final line, no trailing newline: last Ecore wins
	d, err := Read(strings.NewReader(in))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if d.NORB != 2 || d.NELEC != 2 || d.ISYM != 1 {
		t.Fatalf("header = %+v", *d)
	}
	if len(d.OrbSym) != 2 {
		t.Fatalf("OrbSym = %v", d.OrbSym)
	}
	if d.Ecore != 7.5 {
		t.Errorf("Ecore = %v, want 7.5 (last (0,0,0,0) line wins)", d.Ecore)
	}
	if d.OneE(1, 0) != 0.25 || d.OneE(0, 1) != 0.25 {
		t.Errorf("h = %v, want 0.25 symmetric", d.h)
	}
	if d.TwoE(0, 0, 1, 1) != 0.5 || d.TwoE(1, 1, 0, 0) != 0.5 {
		t.Errorf("(11|22) = %v / %v, want 0.5", d.TwoE(0, 0, 1, 1), d.TwoE(1, 1, 0, 0))
	}
}

// TestReadLongLine: the block reader carries a partial line across block boundaries
// and must grow its buffer for a line longer than a block rather than stalling.
func TestReadLongLine(t *testing.T) {
	var b strings.Builder
	b.WriteString("&FCI NORB=2,NELEC=2\n&END\n")
	b.WriteString(strings.Repeat(" ", 3*blockBytes))
	b.WriteString("0.5 1 1 2 2\n")
	d, err := Read(strings.NewReader(b.String()))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if d.TwoE(0, 0, 1, 1) != 0.5 {
		t.Errorf("(11|22) = %v, want 0.5", d.TwoE(0, 0, 1, 1))
	}
}

// TestReadErrorStopsMultiBlockStream exercises the pipeline's early-shutdown path,
// which the single-block error cases above never reach: a malformed line makes the
// applier set stop while the reader is still streaming, so the reader returns from
// the middle of the file and readBody has to close and drain cleanly.
//
// The invariant that keeps that from deadlocking is that a block only ever reaches
// the applier's `order` channel together with the parsers' `work` channel — the two
// sends are adjacent and the stop check sits at the top of the read loop, never
// between them. A block on `order` that no parser ever took would hang the applier
// forever on <-b.done. Run this under -race and it also covers the shutdown
// handshake between the reader, the parsers and the applier.
func TestReadErrorStopsMultiBlockStream(t *testing.T) {
	body := func(badAt string) string {
		var b strings.Builder
		b.WriteString("&FCI NORB=2,NELEC=2\n&END\n")
		if badAt == "first" {
			b.WriteString("1.0 x 1 0 0\n")
		}
		for b.Len() < 20*blockBytes {
			b.WriteString("0.5 1 1 2 2\n")
		}
		if badAt == "last" {
			b.WriteString("1.0 x 1 0 0\n")
		}
		return b.String()
	}
	for _, where := range []string{"first", "last"} {
		t.Run(where, func(t *testing.T) {
			_, err := Read(strings.NewReader(body(where)))
			if err == nil {
				t.Fatal("malformed line in a 20-block body parsed without error")
			}
			if want := `fcidump: bad index "x"`; !strings.HasPrefix(err.Error(), want) {
				t.Fatalf("error %q, want prefix %q", err.Error(), want)
			}
		})
	}
}

// TestReadSingleWorker pins the degenerate GOMAXPROCS=1 shape of the pipeline (one
// parser, nbuf=5), which is a different set of channel-capacity arguments from the
// many-core case and is what a cgroup-limited batch job can actually get.
func TestReadSingleWorker(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	path := "../../../testdata/h2o.fcidump"
	got, err := ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	assertSameData(t, "h2o.fcidump (GOMAXPROCS=1)", got, readSerial(t, path))
}

// TestScaleTransfer: the lock-in scaling multiplies h_pq by lambda for a pair of
// groups, (pq|rs) by lambda per such distribution (lambda^2 for two), leaves everything
// else bit for bit, keeps the 8-fold symmetry, and does not touch the original through
// a Clone.
func TestScaleTransfer(t *testing.T) {
	d, err := ReadFile("../../../testdata/khci/he3_ghost.fcidump")
	if err != nil {
		t.Fatal(err)
	}
	n := d.NORB
	// He3 labels (he3_ghost.mo.json orb_atom): occupied 0,1,2 and compact 3,4,5 on atoms
	// 0,1,2; free 6,7,8 on none
	group := []int{0, 1, 2, 0, 1, 2, -1, -1, -1}
	if len(group) != n {
		t.Fatalf("fixture has %d orbitals", n)
	}
	const lam = 0.37
	s := d.Clone()
	if err := s.ScaleTransfer(group, 0, 1, lam); err != nil {
		t.Fatal(err)
	}
	pair := func(p, q int) bool {
		return (group[p] == 0 && group[q] == 1) || (group[p] == 1 && group[q] == 0)
	}
	fac := func(p, q int) float64 {
		if pair(p, q) {
			return lam
		}
		return 1
	}
	for p := range n {
		for q := range n {
			if want := d.OneE(p, q) * fac(p, q); s.OneE(p, q) != want {
				t.Fatalf("h_%d%d = %g, want %g", p, q, s.OneE(p, q), want)
			}
			for r := range n {
				for u := range n {
					want := d.TwoE(p, q, r, u) * (fac(p, q) * fac(r, u))
					got := s.TwoE(p, q, r, u)
					if got != want {
						t.Fatalf("(%d%d|%d%d) = %g, want %g", p, q, r, u, got, want)
					}
					if got != s.TwoE(q, p, r, u) || got != s.TwoE(r, u, p, q) || got != s.TwoE(p, q, u, r) {
						t.Fatalf("(%d%d|%d%d): permutational symmetry broken", p, q, r, u)
					}
				}
			}
		}
	}
	if d.TwoE(0, 1, 0, 1) == s.TwoE(0, 1, 0, 1) {
		t.Error("the transfer exchange (01|01) was not scaled (or the Clone shares storage)")
	}
	one := d.Clone()
	if err := one.ScaleTransfer(group, 0, 1, 1); err != nil {
		t.Fatal(err)
	}
	for p := range n {
		for q := range n {
			if one.OneE(p, q) != d.OneE(p, q) {
				t.Fatal("lambda = 1 changed an integral")
			}
		}
	}
	if err := s.ScaleTransfer(group, 1, 1, 0); err == nil {
		t.Error("a pair of identical groups was accepted")
	}
}
