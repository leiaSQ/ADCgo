// Package refout parses theADCcode's DIP output files (adcdip{i}.out) into
// structured states, so ADCgo can be cross-validated against that independent
// DIP-ADC(2) reference (M4). It vendors just the slice of ADCanalysis's parser
// that the validation needs; ADCgo is expected to absorb the analysis half over
// time.
//
// One file holds one point-group symmetry block, with a spin-1 (singlet) and a
// spin-3 (triplet) sub-block. Each sub-block prints an "Eigenvalue (eV), ps (%),
// residue" state list (with leading two-hole overlaps) and an "ADC two-hole
// population analysis" table (atom-resolved one-/two-site weights), joined here
// by energy.
package refout

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Config is one leading two-hole component "<i,j|: coeff" (I >= J, 1-based MO
// indices in the reference's active-space numbering).
type Config struct {
	I, J  int
	Coeff float64
}

// PopRow is one row of the two-hole population table: atom one-site (A⁻²) and
// two-site (A⁻¹B⁻¹) weights keyed by group / group-pair name.
type PopRow struct {
	EnergyEV float64
	OneSite  map[string]float64
	TwoSite  map[string]float64
}

// Sum is the total two-hole population (one-site + two-site), which equals the
// state's ps/100 to table rounding.
func (p PopRow) Sum() float64 {
	var s float64
	for _, v := range p.OneSite {
		s += v
	}
	for _, v := range p.TwoSite {
		s += v
	}
	return s
}

// MO is one row of the MO table at the top of the file.
type MO struct {
	Index, Sym int
	EnergyAU   float64
}

// State is one final dicationic state. Residue is NaN for triplet blocks.
type State struct {
	Symmetry, Spin, Index int
	EnergyEV, PSPercent   float64
	Residue               float64
	Leading               []Config
	Pop                   *PopRow
}

// File is one parsed adcdip{i}.out.
type File struct {
	Symmetry      int
	MOs           []MO
	OneSiteGroups []string
	TwoSiteGroups []string
	States        []State
}

var (
	reMO     = regexp.MustCompile(`^\s*(\d+)\s+(\d+)\s+(-?[\d.]+)\s*$`)
	reBlock  = regexp.MustCompile(`symmetry\s+(\d+),\s*spin\s+(\d+)`)
	reState  = regexp.MustCompile(`^\s*(\d+):\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(nan|-?[\d.]+)`)
	reConfig = regexp.MustCompile(`<(\d+),\s*(\d+)\|:\s*(-?[\d.]+)`)
	reFloat  = regexp.MustCompile(`^-?[\d.]+$`)
)

// ParseFile reads one adcdip{i}.out.
func ParseFile(path string) (*File, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()

	var lines []string
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}

	f := &File{}
	f.MOs = parseMOTable(lines)

	// State lists and population tables are collected per spin, then joined.
	type block struct {
		states []State
		pops   []PopRow
	}
	blocks := map[int]*block{} // spin -> block
	getBlock := func(spin int) *block {
		b := blocks[spin]
		if b == nil {
			b = &block{}
			blocks[spin] = b
		}
		return b
	}

	curSpin := 0
	for i := 0; i < len(lines); i++ {
		line := lines[i]

		if m := reBlock.FindStringSubmatch(line); m != nil {
			f.Symmetry, _ = strconv.Atoi(m[1])
			curSpin, _ = strconv.Atoi(m[2])
			continue
		}

		if m := reState.FindStringSubmatch(line); m != nil {
			s := State{Symmetry: f.Symmetry, Spin: curSpin}
			s.Index, _ = strconv.Atoi(m[1])
			s.EnergyEV, _ = strconv.ParseFloat(m[2], 64)
			s.PSPercent, _ = strconv.ParseFloat(m[3], 64)
			if m[4] == "nan" {
				s.Residue = math.NaN()
			} else {
				s.Residue, _ = strconv.ParseFloat(m[4], 64)
			}
			// Leading overlaps are on the next line(s) beginning with "<".
			for j := i + 1; j < len(lines) && j <= i+3; j++ {
				if strings.Contains(lines[j], "<") {
					for _, c := range reConfig.FindAllStringSubmatch(lines[j], -1) {
						cfg := Config{}
						cfg.I, _ = strconv.Atoi(c[1])
						cfg.J, _ = strconv.Atoi(c[2])
						cfg.Coeff, _ = strconv.ParseFloat(c[3], 64)
						s.Leading = append(s.Leading, cfg)
					}
					break
				}
			}
			getBlock(curSpin).states = append(getBlock(curSpin).states, s)
			continue
		}

		if strings.Contains(line, "Orbital contributions") {
			rows, groups := parsePopana(lines, i)
			if f.OneSiteGroups == nil {
				for _, g := range groups {
					if strings.Contains(g, "/") {
						f.TwoSiteGroups = append(f.TwoSiteGroups, g)
					} else {
						f.OneSiteGroups = append(f.OneSiteGroups, g)
					}
				}
			}
			b := getBlock(curSpin)
			b.pops = append(b.pops, rows...)
		}
	}

	// Join each spin's population rows to its states by rounded energy, and flatten.
	for _, spin := range []int{1, 3} {
		b := blocks[spin]
		if b == nil {
			continue
		}
		byEnergy := map[string]*PopRow{}
		for k := range b.pops {
			byEnergy[energyKey(b.pops[k].EnergyEV)] = &b.pops[k]
		}
		for k := range b.states {
			if p := byEnergy[energyKey(b.states[k].EnergyEV)]; p != nil {
				b.states[k].Pop = p
			}
			f.States = append(f.States, b.states[k])
		}
	}
	return f, nil
}

// energyKey rounds an energy to the 4 decimal places the popana table prints, so
// a state's full-precision energy joins to its rounded population row.
func energyKey(e float64) string { return fmt.Sprintf("%.4f", e) }

func parseMOTable(lines []string) []MO {
	var mos []MO
	inTable := false
	for _, line := range lines {
		if strings.Contains(line, "m.o.") && strings.Contains(line, "sym") {
			inTable = true
			continue
		}
		if !inTable {
			continue
		}
		if strings.Contains(line, "---") {
			continue
		}
		m := reMO.FindStringSubmatch(line)
		if m == nil {
			if len(mos) > 0 {
				break // table ended
			}
			continue
		}
		var mo MO
		mo.Index, _ = strconv.Atoi(m[1])
		mo.Sym, _ = strconv.Atoi(m[2])
		mo.EnergyAU, _ = strconv.ParseFloat(m[3], 64)
		mos = append(mos, mo)
	}
	return mos
}

// parsePopana reads the population table beginning at the "Orbital contributions"
// line at index start: the group-name header follows a dashes line, then the data
// rows (energy + one value per group).
func parsePopana(lines []string, start int) (rows []PopRow, groups []string) {
	i := start + 1
	for i < len(lines) && !strings.Contains(lines[i], "---") {
		i++
	}
	i++ // dashes
	if i >= len(lines) {
		return nil, nil
	}
	groups = strings.Fields(lines[i]) // group-name header
	i++
	for ; i < len(lines); i++ {
		fields := strings.Fields(lines[i])
		if len(fields) != len(groups)+1 || !reFloat.MatchString(fields[0]) {
			break
		}
		row := PopRow{OneSite: map[string]float64{}, TwoSite: map[string]float64{}}
		row.EnergyEV, _ = strconv.ParseFloat(fields[0], 64)
		for g, name := range groups {
			v, _ := strconv.ParseFloat(fields[g+1], 64)
			if strings.Contains(name, "/") {
				row.TwoSite[name] = v
			} else {
				row.OneSite[name] = v
			}
		}
		rows = append(rows, row)
	}
	return rows, groups
}
