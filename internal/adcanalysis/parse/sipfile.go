// SIP parsing: theADCcode single-ionization ADC.out files (ndadc3ip propagator).
//
// Unlike a DIP run (one adcdip{i}.out per irrep, two-hole configurations and a
// two-hole population table), a single-ionization run writes one ADC.out holding
// every symmetry block. Each final cationic state carries one-hole main-space
// overlaps "<i|: coeff" whose squares decompose its pole strength per molecular
// orbital; the satellite-space (2h1p shake-up) overlaps are not part of the
// per-orbital main spectrum and are skipped.
package parse

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/leiaSQ/ADCgo/internal/adcanalysis/model"
)

var (
	// "<3|:-0.969806" — one one-hole main-space overlap. A single MO index
	// immediately followed by "|" distinguishes it from a satellite-space
	// component like "<4,4,10|" (which carries commas) or "<4,3,7,II|".
	reOneHole = regexp.MustCompile(`<(\d+)\|:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)`)

	// " Computing spectrum for symmetry 1, spin 2"
	// Single ionization does a full diagonalization (one pass per symmetry), so
	// there is no "the"/no-"the" save-pass distinction as in the DIP output.
	reSIPHdr = regexp.MustCompile(`^\s*Computing spectrum for symmetry\s+(\d+),\s*spin\s+(\d+)`)
)

// ParseSIPFile reads a single-ionization ADC.out and returns its MO table plus
// every state across all symmetry blocks, each carrying its one-hole main-space
// overlaps. Satellite-space overlaps are ignored.
func ParseSIPFile(path string) (*model.SIPOutFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 8*1024*1024)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%s: read: %w", path, err)
	}

	out := &model.SIPOutFile{}

	mos, err := parseMOTable(lines)
	if err != nil {
		return nil, fmt.Errorf("%s: MO table: %w", path, err)
	}
	out.MOTable = mos

	// Locate every symmetry block; each spans from its header to the next.
	type blk struct{ start, end, irrep, spin int }
	var blocks []blk
	for i, l := range lines {
		if m := reSIPHdr.FindStringSubmatch(l); m != nil {
			irrep, _ := strconv.Atoi(m[1])
			spin, _ := strconv.Atoi(m[2])
			blocks = append(blocks, blk{start: i, irrep: irrep, spin: spin})
		}
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("%s: no spectrum block found", path)
	}
	for bi := range blocks {
		end := len(lines)
		for j := blocks[bi].start + 1; j < len(lines); j++ {
			if reSIPHdr.MatchString(lines[j]) {
				end = j
				break
			}
		}
		blocks[bi].end = end
	}

	for _, b := range blocks {
		states, err := parseSIPStates(lines[b.start:b.end], b.irrep, b.spin)
		if err != nil {
			return nil, fmt.Errorf("%s: symmetry %d spin %d: states: %w",
				path, b.irrep, b.spin, err)
		}
		out.States = append(out.States, states...)
	}
	if len(out.States) == 0 {
		return nil, fmt.Errorf("%s: no states parsed", path)
	}

	return out, nil
}

// parseSIPStates reads the "Eigenvalue (eV), ps (%), residue" state list within
// one symmetry block, collecting each state's one-hole main-space overlaps.
// Index gaps are preserved. A state may have no main-space overlaps (pure
// satellite); its Main list is then empty.
func parseSIPStates(seg []string, irrep, spin int) ([]model.SIPState, error) {
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

	var states []model.SIPState
	for i := hdr + 1; i < len(seg); i++ {
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
		res, _ := strconv.ParseFloat(m[4], 64)

		st := model.SIPState{
			Irrep: irrep, Spin: spin, Index: idx,
			EnergyEV: en, PSPercent: ps, Residue: res,
		}

		// Collect one-hole overlaps following a "main-space" marker. Stop at the
		// "satellite-space" marker (no "<" tokens) or a non-overlap line; the
		// satellite lines that follow are skipped by the outer loop.
		j := i + 1
		if j < len(seg) && strings.Contains(seg[j], "Overlaps with main-space") {
			j++
			for j < len(seg) && strings.Contains(seg[j], "<") {
				for _, mm := range reOneHole.FindAllStringSubmatch(seg[j], -1) {
					orb, _ := strconv.Atoi(mm[1])
					c, _ := strconv.ParseFloat(mm[2], 64)
					st.Main = append(st.Main, model.OrbWeight{Orbital: orb, Coeff: c})
				}
				j++
			}
			i = j - 1
		}
		states = append(states, st)
	}
	return states, nil
}
