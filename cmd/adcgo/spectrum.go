package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"

	"adcgo/internal/adc/analyze"
	"adcgo/internal/adc/mo"
	"adcgo/internal/adc/spectrum"
)

// specConfig carries the -spectrum-mode options resolved from the CLI flags.
type specConfig struct {
	enabled     bool
	initAtom    string
	initOrbital string
	stRatio     float64
	groups      []spectrum.Site // explicit -group sites (before completeSites)
	interactive bool            // a bare -group requested the interactive dialogue
	classify    spectrum.Options
}

// groupFlag is the repeatable -group flag. A value "NAME=col1,~col2" defines one
// decay site (with ~col marking a passive column); a bare -group (parsed as the
// boolean "true") requests the interactive grouping dialogue instead. Both forms
// may not be mixed meaningfully — a bare -group takes precedence at build time.
type groupFlag struct {
	sites       []spectrum.Site
	interactive bool
}

func (g *groupFlag) String() string {
	parts := make([]string, len(g.sites))
	for i, s := range g.sites {
		passive := make(map[string]bool, len(s.Passive))
		for _, p := range s.Passive {
			passive[p] = true
		}
		ms := make([]string, len(s.Members))
		for j, m := range s.Members {
			if passive[m] {
				ms[j] = "~" + m
			} else {
				ms[j] = m
			}
		}
		parts[i] = s.Name + "=" + strings.Join(ms, ",")
	}
	return strings.Join(parts, " ")
}

// IsBoolFlag lets a bare -group (no "=value") be accepted; the flag package then
// calls Set("true"), which we interpret as "prompt interactively".
func (g *groupFlag) IsBoolFlag() bool { return true }

func (g *groupFlag) Set(v string) error {
	if v == "true" { // bare -group
		g.interactive = true
		return nil
	}
	name, members, ok := strings.Cut(v, "=")
	name = strings.TrimSpace(name)
	if !ok || name == "" {
		return fmt.Errorf("group %q must be NAME=col1[,col2...]", v)
	}
	var ms, passive []string
	for m := range strings.SplitSeq(members, ",") {
		m = strings.TrimSpace(m)
		if m == "" {
			continue
		}
		if col, isPassive := strings.CutPrefix(m, "~"); isPassive {
			if col = strings.TrimSpace(col); col == "" {
				return fmt.Errorf("group %q has a bare %q with no column", v, "~")
			}
			ms = append(ms, col)
			passive = append(passive, col)
			continue
		}
		ms = append(ms, m)
	}
	if len(ms) == 0 {
		return fmt.Errorf("group %q has no members", v)
	}
	g.sites = append(g.sites, spectrum.Site{Name: name, Members: ms, Passive: passive})
	return nil
}

// buildDIPSpectrum resolves the decay sites (from -group flags or the interactive
// dialogue) against the MO sidecar's atom names and flattens the solved sectors
// into the decay-channel stick spectrum.
func buildDIPSpectrum(secs []analyze.Sector, md *mo.Data, cfg specConfig) (*spectrum.Spectrum, error) {
	columns := md.AtomNames
	sites, initAtom, err := resolveSites(cfg, columns)
	if err != nil {
		return nil, err
	}
	spec, skipped, err := spectrum.BuildDIP(secs, sites, spectrum.DIPOptions{
		InitialAtom:         initAtom,
		InitialOrbital:      cfg.initOrbital,
		Classify:            cfg.classify,
		SingletTripletRatio: cfg.stRatio,
	})
	if err != nil {
		return nil, err
	}
	if skipped > 0 {
		fmt.Fprintf(os.Stderr, "adcgo: %d state(s) without atom-resolved populations were skipped\n", skipped)
	}
	return spec, nil
}

// resolveSites turns the CLI config + the discovered population columns into a
// complete site partition and the initial-ionization site. A bare -group opens
// the interactive dialogue (only on a terminal); otherwise the explicit -group
// sites are completed with singleton sites for every ungrouped column.
func resolveSites(cfg specConfig, columns []string) (sites []spectrum.Site, initAtom string, err error) {
	initAtom = cfg.initAtom
	if cfg.interactive {
		if !isInteractive(os.Stdin) {
			return nil, "", fmt.Errorf("a bare -group needs a terminal; pass -group NAME=col1,col2 instead")
		}
		in := bufio.NewScanner(os.Stdin)
		explicit := promptGrouping(os.Stderr, in, columns)
		if sites, err = completeSites(explicit, columns); err != nil {
			return nil, "", err
		}
		initAtom = promptInitial(os.Stderr, in, sites, initAtom)
		return sites, initAtom, nil
	}
	if sites, err = completeSites(cfg.groups, columns); err != nil {
		return nil, "", err
	}
	return sites, initAtom, nil
}

// isInteractive reports whether f is a terminal (so prompting makes sense).
func isInteractive(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

// completeSites validates the user sites against the discovered columns and
// appends a singleton site for every column not already covered, so the result
// partitions all population columns. It errors on an unknown column, a column
// placed in two sites, or a duplicate/colliding site name.
func completeSites(sites []spectrum.Site, columns []string) ([]spectrum.Site, error) {
	colset := make(map[string]bool, len(columns))
	for _, c := range columns {
		colset[c] = true
	}
	covered := make(map[string]bool)
	names := make(map[string]bool)
	for _, s := range sites {
		if names[s.Name] {
			return nil, fmt.Errorf("duplicate site name %q", s.Name)
		}
		names[s.Name] = true
		for _, m := range s.Members {
			if !colset[m] {
				return nil, fmt.Errorf("site %q references unknown column %q (have: %s)",
					s.Name, m, strings.Join(columns, ", "))
			}
			if covered[m] {
				return nil, fmt.Errorf("column %q is in more than one site", m)
			}
			covered[m] = true
		}
	}
	out := append([]spectrum.Site(nil), sites...)
	for _, c := range columns {
		if covered[c] {
			continue
		}
		if names[c] {
			return nil, fmt.Errorf("site name %q collides with the ungrouped column %q", c, c)
		}
		out = append(out, spectrum.Site{Name: c, Members: []string{c}})
		names[c] = true
	}
	return out, nil
}

// promptGrouping interactively collects decay-site definitions. Each line is
// "NAME = col1, col2 ..."; an empty line ends input. Columns left ungrouped
// become their own sites later (completeSites). Returns the explicitly grouped
// sites only.
func promptGrouping(w io.Writer, in *bufio.Scanner, columns []string) []spectrum.Site {
	fmt.Fprintf(w, "\nPopulation columns: %s\n", strings.Join(columns, ", "))
	fmt.Fprintln(w, "Group them into decay sites as  NAME = col1, col2 ...")
	fmt.Fprintln(w, "Prefix a column with ~ to make it passive (its holes are discounted), e.g.  wat = O, ~H1, ~H2")
	fmt.Fprintln(w, "Press Enter on an empty line to finish; ungrouped columns each become their own site.")

	colset := make(map[string]bool, len(columns))
	for _, c := range columns {
		colset[c] = true
	}
	used := make(map[string]bool)
	names := make(map[string]bool)
	var sites []spectrum.Site
	for {
		fmt.Fprint(w, "site> ")
		if !in.Scan() {
			break
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			break
		}
		name, mem, ok := strings.Cut(line, "=")
		name = strings.TrimSpace(name)
		if !ok || name == "" {
			fmt.Fprintln(w, "  format: NAME = col1, col2")
			continue
		}
		if names[name] {
			fmt.Fprintf(w, "  site %q already defined\n", name)
			continue
		}
		var ms, passive []string
		bad := false
		for m := range strings.SplitSeq(mem, ",") {
			m = strings.TrimSpace(m)
			if m == "" {
				continue
			}
			col, isPassive := strings.CutPrefix(m, "~")
			col = strings.TrimSpace(col)
			switch {
			case col == "":
				fmt.Fprintf(w, "  bare %q with no column\n", "~")
				bad = true
			case !colset[col]:
				fmt.Fprintf(w, "  unknown column %q (have: %s)\n", col, strings.Join(columns, ", "))
				bad = true
			case used[col]:
				fmt.Fprintf(w, "  column %q already grouped\n", col)
				bad = true
			default:
				ms = append(ms, col)
				if isPassive {
					passive = append(passive, col)
				}
			}
			if bad {
				break
			}
		}
		if bad || len(ms) == 0 {
			continue
		}
		for _, m := range ms {
			used[m] = true
		}
		names[name] = true
		sites = append(sites, spectrum.Site{Name: name, Members: ms, Passive: passive})
		fmt.Fprintf(w, "  site %q = %s\n", name, strings.Join(ms, ", "))
	}
	return sites
}

// promptInitial asks which site is the initial ionization site, defaulting to
// def (falling back to the first site if def is not one of them).
func promptInitial(w io.Writer, in *bufio.Scanner, sites []spectrum.Site, def string) string {
	names := make([]string, len(sites))
	has := make(map[string]bool, len(sites))
	for i, s := range sites {
		names[i] = s.Name
		has[s.Name] = true
	}
	if !has[def] && len(names) > 0 {
		def = names[0]
	}
	for {
		fmt.Fprintf(w, "\nSites: %s\nInitial ionization site? [%s]: ", strings.Join(names, ", "), def)
		if !in.Scan() {
			return def
		}
		ans := strings.TrimSpace(in.Text())
		if ans == "" {
			return def
		}
		if has[ans] {
			return ans
		}
		fmt.Fprintf(w, "  %q is not a site\n", ans)
	}
}
