// Package fcidump reads MO-basis integrals from a standard FCIDUMP file.
//
// FCIDUMP is the integral-ingestion contract for ADCgo: SCF and integral
// evaluation are delegated to an external code (e.g. pyscf's
// pyscf.tools.fcidump.from_scf), which writes the one-electron integrals
// h_pq, the two-electron integrals (pq|rs) in chemist notation, the core
// energy, and the electron/orbital counts. Orbital energies are NOT stored in
// FCIDUMP; for a canonical HF reference they are reconstructed from the Fock
// diagonal (see package mp).
package fcidump

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Data holds the parsed MO-basis integrals and metadata.
//
// Two-electron integrals are stored fully expanded (8-fold permutation
// symmetry applied) in a dense NORB^4 array using chemist notation (pq|rs);
// this is memory-cheap at the small NORB of the M0 spike and keeps access O(1).
// A blocked, symmetry-aware store replaces this for larger cases (see plan).
type Data struct {
	NORB   int
	NELEC  int
	MS2    int
	ISYM   int
	OrbSym []int // point-group label per orbital (all 1 when symmetry is off)

	Ecore float64   // core energy: nuclear repulsion (+ frozen-core, if any)
	h     []float64 // NORB*NORB one-electron integrals h_pq (symmetric)
	eri   []float64 // NORB^4 two-electron integrals (pq|rs), chemist notation
}

// OneE returns the one-electron integral h_pq (0-based indices).
func (d *Data) OneE(p, q int) float64 { return d.h[p*d.NORB+q] }

// TwoE returns the two-electron integral (pq|rs) in chemist notation (0-based).
func (d *Data) TwoE(p, q, r, s int) float64 {
	n := d.NORB
	return d.eri[((p*n+q)*n+r)*n+s]
}

var headerIntRe = func(key string) *regexp.Regexp {
	return regexp.MustCompile(`(?i)\b` + key + `\s*=\s*(-?\d+)`)
}

// ReadFile parses the FCIDUMP at path.
func ReadFile(path string) (*Data, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

// Read parses a FCIDUMP from r.
func Read(r io.Reader) (*Data, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)

	var header strings.Builder
	var body []string
	inHeader := false
	headerDone := false
	for sc.Scan() {
		line := sc.Text()
		t := strings.TrimSpace(line)
		if headerDone {
			if t != "" {
				body = append(body, t)
			}
			continue
		}
		up := strings.ToUpper(t)
		if !inHeader {
			if strings.HasPrefix(up, "&FCI") || strings.HasPrefix(up, "$FCI") {
				inHeader = true
				header.WriteString(" " + t)
			}
			continue
		}
		if up == "&END" || up == "$END" || up == "/" || strings.HasSuffix(up, "&END") {
			headerDone = true
			continue
		}
		header.WriteString(" " + t)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if !inHeader {
		return nil, fmt.Errorf("fcidump: no &FCI header found")
	}

	d := &Data{}
	hstr := header.String()
	norb, ok := parseInt(hstr, "NORB")
	if !ok {
		return nil, fmt.Errorf("fcidump: NORB not found in header")
	}
	nelec, ok := parseInt(hstr, "NELEC")
	if !ok {
		return nil, fmt.Errorf("fcidump: NELEC not found in header")
	}
	d.NORB = norb
	d.NELEC = nelec
	d.MS2, _ = parseInt(hstr, "MS2")
	d.ISYM, _ = parseInt(hstr, "ISYM")
	d.OrbSym = parseOrbSym(hstr, norb)

	n := norb
	d.h = make([]float64, n*n)
	d.eri = make([]float64, n*n*n*n)

	for _, ln := range body {
		fields := strings.Fields(ln)
		if len(fields) < 5 {
			continue
		}
		val, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			return nil, fmt.Errorf("fcidump: bad value %q: %w", fields[0], err)
		}
		idx := make([]int, 4)
		for k := 0; k < 4; k++ {
			idx[k], err = strconv.Atoi(fields[k+1])
			if err != nil {
				return nil, fmt.Errorf("fcidump: bad index %q: %w", fields[k+1], err)
			}
		}
		i, j, kk, l := idx[0], idx[1], idx[2], idx[3]
		switch {
		case i == 0 && j == 0 && kk == 0 && l == 0:
			d.Ecore = val
		case kk == 0 && l == 0:
			// one-electron h_ij (indices 1-based, i>=j); symmetric.
			d.setOne(i-1, j-1, val)
		default:
			d.setTwo(i-1, j-1, kk-1, l-1, val)
		}
	}
	return d, nil
}

func (d *Data) setOne(p, q int, v float64) {
	n := d.NORB
	d.h[p*n+q] = v
	d.h[q*n+p] = v
}

// setTwo stores (pq|rs) into all 8 permutation-equivalent slots.
func (d *Data) setTwo(p, q, r, s int, v float64) {
	n := d.NORB
	set := func(a, b, c, e int) { d.eri[((a*n+b)*n+c)*n+e] = v }
	set(p, q, r, s)
	set(q, p, r, s)
	set(p, q, s, r)
	set(q, p, s, r)
	set(r, s, p, q)
	set(s, r, p, q)
	set(r, s, q, p)
	set(s, r, q, p)
}

func parseInt(header, key string) (int, bool) {
	m := headerIntRe(key).FindStringSubmatch(header)
	if m == nil {
		return 0, false
	}
	v, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, false
	}
	return v, true
}

// parseOrbSym reads the ORBSYM=... comma list following the key, up to the next
// KEY= token or the end of the header. Returns nil if absent.
func parseOrbSym(header string, norb int) []int {
	loc := regexp.MustCompile(`(?i)\bORBSYM\s*=`).FindStringIndex(header)
	if loc == nil {
		return nil
	}
	rest := header[loc[1]:]
	// Stop at the next "WORD=" (another namelist key) or the &END terminator.
	if stop := regexp.MustCompile(`(?i)[A-Z_]+\s*=|&END|\$END`).FindStringIndex(rest); stop != nil {
		rest = rest[:stop[0]]
	}
	syms := make([]int, 0, norb)
	for _, tok := range strings.FieldsFunc(rest, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' }) {
		if v, err := strconv.Atoi(strings.TrimSpace(tok)); err == nil {
			syms = append(syms, v)
		}
	}
	if len(syms) == 0 {
		return nil
	}
	return syms
}
