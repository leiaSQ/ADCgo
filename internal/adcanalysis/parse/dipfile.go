// Package parse turns theADCcode adcdip{i}.out files into model structures.
package parse

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/leiaSQ/ADCgo/internal/adcanalysis/model"
)

// EnergyJoinTolEV is the maximum energy difference (eV) allowed when joining a
// popana row to a state. State energies are printed to 6 dp, popana eigenvalues
// to 4 dp, so the rounding gap is ~5e-5 eV; 1e-3 leaves ample margin while
// still being far tighter than the spacing between distinct roots.
const EnergyJoinTolEV = 1e-3

var (
	// "  1\t1\t-1.35904"  (index, sym, energy) — whitespace/tab separated.
	reMORow = regexp.MustCompile(`^\s*(\d+)\s+(\d+)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$`)

	// " 1: 39.660357, 83.39, 0.000000"  (index: energy, ps, residue)
	reStateHdr = regexp.MustCompile(`^\s*(\d+):\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(\S+)\s*$`)

	// "<4,4|:-0.889428"  — one overlap component. Coeff may be preceded by a
	// space or a sign; both are handled by \s* and the optional - in the number.
	reOverlap = regexp.MustCompile(`<(\d+),(\d+)\|:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)`)

	// " Computing the spectrum for symmetry 1, spin 3"
	// NOTE the literal "the": this marks a *real* diagonalization pass. The
	// eigenvector-saving pass prints "Computing spectrum ..." (no "the") and
	// must be skipped — it emits sentinel states (0.0 eV, 100%, nan) and no
	// population table.
	reRealHdr = regexp.MustCompile(`^\s*Computing the spectrum for symmetry\s+(\d+),\s*spin\s+(\d+)`)

	// Either kind of "Computing ... spectrum ..." line; used to bound a block.
	reAnyHdr = regexp.MustCompile(`^\s*Computing (?:the )?spectrum for symmetry`)
)

// ParseDIPFile reads a single adcdip{i}.out file and returns its MO table plus
// every real state joined to its two-hole population row.
//
// Eigenvector-saving passes ("Computing spectrum ...", no "the") are ignored,
// as is all the per-iteration Lanczos chatter.
func ParseDIPFile(path string) (*model.OutFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 8*1024*1024) // long lines just in case
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%s: read: %w", path, err)
	}

	out := &model.OutFile{}

	mos, err := parseMOTable(lines)
	if err != nil {
		return nil, fmt.Errorf("%s: MO table: %w", path, err)
	}
	out.MOTable = mos

	// Locate the start of every real diagonalization block.
	type blk struct {
		start, end, irrep, spin int
	}
	var blocks []blk
	for i, l := range lines {
		if m := reRealHdr.FindStringSubmatch(l); m != nil {
			irrep, _ := strconv.Atoi(m[1])
			spin, _ := strconv.Atoi(m[2])
			blocks = append(blocks, blk{start: i, irrep: irrep, spin: spin})
		}
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("%s: no real spectrum block found", path)
	}
	// A block ends at the next "Computing ... spectrum" line of any kind (the
	// eigenvector-save pass that follows it), or at EOF.
	for bi := range blocks {
		end := len(lines)
		for j := blocks[bi].start + 1; j < len(lines); j++ {
			if reAnyHdr.MatchString(lines[j]) {
				end = j
				break
			}
		}
		blocks[bi].end = end
	}

	seenGroups := false
	for _, b := range blocks {
		seg := lines[b.start:b.end]

		states, err := parseStates(seg, b.irrep, b.spin)
		if err != nil {
			return nil, fmt.Errorf("%s: symmetry %d spin %d: states: %w",
				path, b.irrep, b.spin, err)
		}

		rows, groups, err := parsePopTable(seg)
		if err != nil {
			return nil, fmt.Errorf("%s: symmetry %d spin %d: popana: %w",
				path, b.irrep, b.spin, err)
		}

		if !seenGroups && (len(groups.OneSite) > 0 || len(groups.TwoSite) > 0) {
			out.Groups = groups
			seenGroups = true
		}

		if err := joinPop(states, rows); err != nil {
			return nil, fmt.Errorf("%s: symmetry %d spin %d: join: %w",
				path, b.irrep, b.spin, err)
		}

		out.States = append(out.States, states...)
		if out.Symmetry == 0 {
			out.Symmetry = b.irrep
			out.Path = b.irrep
		}
	}

	return out, nil
}

// parseMOTable reads the "m.o. / sym / energy(a.u.)" block at the top of the
// file. It starts after the "m.o." header line and stops at the first line
// that is not an MO row (blank line or the "Integral table loaded:" banner).
func parseMOTable(lines []string) ([]model.MO, error) {
	start := -1
	for i, l := range lines {
		t := strings.TrimSpace(l)
		if strings.HasPrefix(t, "m.o.") && strings.Contains(t, "energy") {
			start = i + 1
			break
		}
	}
	if start < 0 {
		return nil, fmt.Errorf("MO table header not found")
	}
	// Skip the dashed separator line if present.
	if start < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[start]), "---") {
		start++
	}

	var mos []model.MO
	for i := start; i < len(lines); i++ {
		m := reMORow.FindStringSubmatch(lines[i])
		if m == nil {
			if strings.TrimSpace(lines[i]) == "" && len(mos) == 0 {
				continue // tolerate blank lines before the first row
			}
			break
		}
		idx, _ := strconv.Atoi(m[1])
		sym, _ := strconv.Atoi(m[2])
		en, err := strconv.ParseFloat(m[3], 64)
		if err != nil {
			return nil, fmt.Errorf("orbital %d energy %q: %w", idx, m[3], err)
		}
		mos = append(mos, model.MO{Index: idx, Sym: sym, EnergyAU: en})
	}
	if len(mos) == 0 {
		return nil, fmt.Errorf("MO table is empty")
	}
	return mos, nil
}

// parseStates reads the "Eigenvalue (eV), ps (%), residue" state list within a
// single real block. Index gaps are preserved (states are not assumed
// contiguous). Overlap components are read from the line(s) following the
// "Overlaps with main-space configurations:" marker until a non-overlap line.
func parseStates(seg []string, irrep, spin int) ([]model.State, error) {
	// Find the eigenvalue list header; the popana table that follows reuses the
	// word "Eigenvalue", so we anchor on the distinctive "ps (%)" header line.
	hdr := -1
	for i, l := range seg {
		if strings.Contains(l, "Eigenvalue") && strings.Contains(l, "ps (%)") {
			hdr = i
			break
		}
	}
	if hdr < 0 {
		return nil, fmt.Errorf("eigenvalue header not found")
	}

	var states []model.State
	for i := hdr + 1; i < len(seg); i++ {
		// Stop at the population analysis table.
		if strings.Contains(seg[i], "two-hole population analysis") {
			break
		}
		m := reStateHdr.FindStringSubmatch(seg[i])
		if m == nil {
			continue
		}
		idx, _ := strconv.Atoi(m[1])
		en, err := strconv.ParseFloat(m[2], 64)
		if err != nil {
			return nil, fmt.Errorf("state %d energy %q: %w", idx, m[2], err)
		}
		ps, err := strconv.ParseFloat(m[3], 64)
		if err != nil {
			return nil, fmt.Errorf("state %d ps %q: %w", idx, m[3], err)
		}
		// Residue may legitimately be "nan" in save-pass sentinels, but those
		// blocks are excluded; here parse leniently and tolerate non-numeric.
		res, _ := strconv.ParseFloat(m[4], 64)

		st := model.State{
			Irrep: irrep, Spin: spin, Index: idx,
			EnergyEV: en, PSPercent: ps, Residue: res,
		}

		// Collect overlaps: the next non-blank line should be the marker,
		// after which one or more lines carry <i,j|: coeff tokens.
		j := i + 1
		if j < len(seg) && strings.Contains(seg[j], "Overlaps with main-space") {
			j++
			for j < len(seg) {
				if !strings.Contains(seg[j], "<") {
					break
				}
				for _, mm := range reOverlap.FindAllStringSubmatch(seg[j], -1) {
					a, _ := strconv.Atoi(mm[1])
					b, _ := strconv.Atoi(mm[2])
					c, _ := strconv.ParseFloat(mm[3], 64)
					st.Leading = append(st.Leading, model.Config{I: a, J: b, Coeff: c})
				}
				j++
			}
			i = j - 1
		}
		states = append(states, st)
	}
	return states, nil
}

// parsePopTable reads the "ADC two-hole population analysis" table within a
// block. The header line (e.g. "O   O/H1   O/H2   H1   H1/H2   H2") drives
// column naming: a name with "/" is a two-site pair, otherwise one-site.
func parsePopTable(seg []string) ([]model.PopRow, model.PopGroups, error) {
	var groups model.PopGroups
	hdr := -1
	for i, l := range seg {
		if strings.Contains(l, "Orbital contributions") {
			hdr = i
			break
		}
	}
	if hdr < 0 {
		// Not every block necessarily has one; treat as empty rather than error.
		return nil, groups, nil
	}
	// header line of dashes, then the column-name line.
	colLine := -1
	for i := hdr + 1; i < len(seg); i++ {
		t := strings.TrimSpace(seg[i])
		if t == "" || strings.HasPrefix(t, "---") {
			continue
		}
		colLine = i
		break
	}
	if colLine < 0 {
		return nil, groups, fmt.Errorf("popana column header not found")
	}
	cols := strings.Fields(seg[colLine])
	for _, c := range cols {
		if strings.Contains(c, "/") {
			groups.TwoSite = append(groups.TwoSite, c)
		} else {
			groups.OneSite = append(groups.OneSite, c)
		}
	}

	var rows []model.PopRow
	for i := colLine + 1; i < len(seg); i++ {
		t := strings.TrimSpace(seg[i])
		if t == "" {
			break
		}
		fields := strings.Fields(t)
		// Expect: eigenvalue + one value per column.
		if len(fields) != len(cols)+1 {
			break
		}
		vals := make([]float64, len(fields))
		ok := true
		for k, fld := range fields {
			v, err := strconv.ParseFloat(fld, 64)
			if err != nil {
				ok = false
				break
			}
			vals[k] = v
		}
		if !ok {
			break
		}
		row := model.PopRow{
			EnergyEV: vals[0],
			OneSite:  map[string]float64{},
			TwoSite:  map[string]float64{},
		}
		for k, name := range cols {
			v := vals[k+1]
			if strings.Contains(name, "/") {
				row.TwoSite[name] = v
			} else {
				row.OneSite[name] = v
			}
		}
		rows = append(rows, row)
	}
	return rows, groups, nil
}

// joinPop attaches each popana row to the state whose energy is closest within
// EnergyJoinTolEV. Each row is consumed at most once. A row that matches no
// state is an error (it signals a parsing or ordering problem); a state with no
// row is left with Pop == nil (tolerated — e.g. a state below the popana cut).
func joinPop(states []model.State, rows []model.PopRow) error {
	used := make([]bool, len(rows))
	for si := range states {
		best, bestDiff := -1, math.MaxFloat64
		for ri := range rows {
			if used[ri] {
				continue
			}
			d := math.Abs(rows[ri].EnergyEV - states[si].EnergyEV)
			if d < bestDiff {
				bestDiff, best = d, ri
			}
		}
		if best >= 0 && bestDiff <= EnergyJoinTolEV {
			r := rows[best]
			states[si].Pop = &r
			used[best] = true
		}
	}
	for ri := range rows {
		if !used[ri] {
			return fmt.Errorf("popana row at %.4f eV matched no state", rows[ri].EnergyEV)
		}
	}
	return nil
}
