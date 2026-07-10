// Command ADCanalysis is the v1 decay-channel analyzer CLI.
//
// It reads the adcdip{i}.out files of an ADC(2) double-ionization run, maps
// each final dicationic state's two-hole population onto decay channels
// (Auger / ICD / ETMD) relative to a chosen initial core-ionized atom, and
// writes a channel-resolved stick spectrum as JSON (schema: decay_analyzer_sketch.md §6).
//
// Usage:
//
//	ADCanalysis -in examples/h2o -init-atom O -out spec.json
//	ADCanalysis -in examples/h2o -group "wat=O,H1,H2" -init-atom wat -out -
//	ADCanalysis -in examples/h2o -group "wat=O,~H1,~H2" -init-atom wat   # discount the H holes
//	ADCanalysis -in examples/h2o                              # prompt for grouping + initial site
//	ADCanalysis -mode sip -in examples/SIP -out sip.json      # single-ionization (per-orbital) spectrum
//
// With -mode sip the tool instead reads a single-ionization ADC.out (ndadc3ip)
// and emits a per-orbital stick spectrum in the same JSON schema (channel =
// orbital), bypassing the decay-site/initial-atom machinery; see runSIP.
//
// The DIP pipeline is parse -> classify -> spectrum -> JSON. The population columns
// are grouped into decay sites (via -group, or interactively on a terminal),
// the initial ionization site is chosen (via -init-atom or interactively), and
// channels — Auger, ICD, ETMD(2), ETMD(3) — are resolved against it. Channel
// thresholds and the singlet:triplet ratio are configurable via flags.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/leiaSQ/ADCgo/internal/adcanalysis/classify"
	"github.com/leiaSQ/ADCgo/internal/adcanalysis/model"
	"github.com/leiaSQ/ADCgo/internal/adcanalysis/parse"
	spectrum "github.com/leiaSQ/ADCgo/internal/adcanalysis/render"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "ADCanalysis:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		mode        = flag.String("mode", "dip", "analysis mode: \"dip\" (double ionization, adcdip*.out) or \"sip\" (single ionization, ADC.out)")
		inDir       = flag.String("in", "", "input directory containing adcdip*.out + dip.in (dip), or ADC.out + scf_adc.in (sip)")
		initAtom    = flag.String("init-atom", "O", "initial core-ionized site name; prompted interactively if omitted on a terminal")
		initOrbital = flag.String("init-orbital", "", "optional initial orbital label, e.g. \"1s\" (metadata only)")
		outPath     = flag.String("out", "spec.json", "output JSON path, or \"-\" for stdout")
		minWeight   = flag.Float64("min-weight", 0, "drop channels with weight <= this (default removes zero/rounding-noise)")
		minFraction = flag.Float64("min-fraction", 0, "drop channels below this fraction of a state's total 2h population (0..1)")
		includeZero = flag.Bool("include-zero", false, "emit the full canonical channel set per state, even at zero weight")
		stRatio     = flag.Float64("st-ratio", 3.0, "singlet:triplet ratio recorded in meta for the plotting layer")
		molecule    = flag.String("molecule", "", "molecule label for meta (auto-read from dip.in if empty)")
		basis       = flag.String("basis", "", "basis label for meta (auto-read from dip.in if empty)")
		pointGroup  = flag.String("point-group", "", "point-group label for meta, e.g. C2v")
		pretty      = flag.Bool("pretty", true, "indent the JSON output")
	)
	var groupFlag groupList
	flag.Var(&groupFlag, "group", "decay-site grouping NAME=col1,col2 (repeatable); prefix a column with ~ to make it passive (its holes are discounted); default each population column is its own site")
	flag.Parse()

	if *inDir == "" {
		flag.Usage()
		return fmt.Errorf("-in is required")
	}

	switch *mode {
	case "sip":
		return runSIP(*inDir, *outPath, *pretty, *molecule, *basis, *pointGroup)
	case "dip":
		// fall through to the DIP pipeline below
	default:
		return fmt.Errorf("-mode must be \"dip\" or \"sip\", got %q", *mode)
	}

	// 1. Discover and parse the adcdip{i}.out files.
	paths, err := filepath.Glob(filepath.Join(*inDir, "adcdip*.out"))
	if err != nil {
		return fmt.Errorf("globbing inputs: %w", err)
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return fmt.Errorf("no adcdip*.out files found in %q", *inDir)
	}

	var files []*model.OutFile
	var columns []string
	var sources []string
	for _, p := range paths {
		out, err := parse.ParseDIPFile(p)
		if err != nil {
			return fmt.Errorf("parse %s: %w", p, err)
		}
		files = append(files, out)
		sources = append(sources, filepath.Base(p))
		if columns == nil && len(out.Groups.OneSite) > 0 {
			columns = out.Groups.OneSite
		}
	}
	if columns == nil {
		return fmt.Errorf("no &popana population columns found in any input file")
	}

	// 2. Resolve the site grouping and the initial ionization site. Flags win;
	// when they are absent and stdin is a terminal, prompt interactively so the
	// tool is usable without memorizing the column names. Prompts go to stderr,
	// keeping stdout clean for "-out -".
	wasSet := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { wasSet[f.Name] = true })
	var in *bufio.Scanner
	if isInteractive(os.Stdin) {
		in = bufio.NewScanner(os.Stdin)
	}

	sites := []model.Site(groupFlag)
	if len(sites) == 0 && in != nil {
		sites = promptGrouping(os.Stderr, in, columns)
	}
	sites, err = completeSites(sites, columns)
	if err != nil {
		return err
	}

	initSite := *initAtom
	if !wasSet["init-atom"] && in != nil {
		initSite = promptInitial(os.Stderr, in, sites, initSite)
	}
	if err := classify.ValidateInitialAtom(initSite, sites); err != nil {
		return err
	}

	// 3. Best-effort metadata from dip.in (flags win when set).
	mol, bas := *molecule, *basis
	if mol == "" || bas == "" {
		m, b := readRunMeta(filepath.Join(*inDir, "dip.in"))
		if mol == "" {
			mol = m
		}
		if bas == "" {
			bas = b
		}
	}

	// 4. Build the spectrum (parse -> regroup -> classify -> flatten).
	spec, skipped, err := spectrum.BuildDIP(files, sites, spectrum.DIPOptions{
		InitialAtom:    initSite,
		InitialOrbital: *initOrbital,
		Classify: classify.Options{
			MinWeight:   *minWeight,
			MinFraction: *minFraction,
			IncludeZero: *includeZero,
		},
		SingletTripletRatio: *stRatio,
		Molecule:            mol,
		Basis:               bas,
		PointGroup:          *pointGroup,
		SourceFiles:         sources,
	})
	if err != nil {
		return err
	}
	if skipped > 0 {
		fmt.Fprintf(os.Stderr, "ADCanalysis: note: %d state(s) had no population row and were skipped\n", skipped)
	}

	// 5. Marshal and write.
	return writeSpectrum(spec, *outPath, *pretty)
}

// runSIP is the single-ionization pipeline: parse one ADC.out (ndadc3ip output)
// and build a per-orbital stick spectrum in the same JSON schema as DIP. The
// decay-site grouping / initial-atom machinery does not apply and is skipped.
func runSIP(inDir, outPath string, pretty bool, molecule, basis, pointGroup string) error {
	paths, err := filepath.Glob(filepath.Join(inDir, "ADC.out"))
	if err != nil {
		return fmt.Errorf("globbing inputs: %w", err)
	}
	if len(paths) == 0 {
		return fmt.Errorf("no ADC.out found in %q", inDir)
	}
	sort.Strings(paths)
	path := paths[0]

	f, err := parse.ParseSIPFile(path)
	if err != nil {
		return fmt.Errorf("parse %s: %w", path, err)
	}

	// Best-effort metadata from scf_adc.in (flags win when set).
	mol, bas := molecule, basis
	if mol == "" || bas == "" {
		m, b := readRunMeta(filepath.Join(inDir, "scf_adc.in"))
		if mol == "" {
			mol = m
		}
		if bas == "" {
			bas = b
		}
	}

	spec, err := spectrum.BuildSIP(f, spectrum.SIPOptions{
		Molecule:    mol,
		Basis:       bas,
		PointGroup:  pointGroup,
		SourceFiles: []string{filepath.Base(path)},
	})
	if err != nil {
		return err
	}
	return writeSpectrum(spec, outPath, pretty)
}

// writeSpectrum marshals a spectrum to JSON and writes it to outPath ("-" for
// stdout). HTML escaping is disabled so channel labels keep their literal "->"
// (the schema's form) rather than ">".
func writeSpectrum(spec *spectrum.Spectrum, outPath string, pretty bool) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if pretty {
		enc.SetIndent("", "  ")
	}
	if err := enc.Encode(spec); err != nil { // Encode appends a trailing newline
		return fmt.Errorf("marshal JSON: %w", err)
	}
	data := buf.Bytes()

	if outPath == "-" {
		_, err := os.Stdout.Write(data)
		return err
	}
	if err := os.WriteFile(outPath, data, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", outPath, err)
	}
	fmt.Fprintf(os.Stderr, "ADCanalysis: wrote %d lines across %d channels to %s\n",
		len(spec.Lines), len(spec.Channels), outPath)
	return nil
}

// groupList is a repeatable -group flag. Each value is "NAME=col1,col2", one
// decay site grouping one or more population columns.
type groupList []model.Site

func (g *groupList) String() string {
	parts := make([]string, len(*g))
	for i, s := range *g {
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

func (g *groupList) Set(v string) error {
	name, members, ok := strings.Cut(v, "=")
	name = strings.TrimSpace(name)
	if !ok || name == "" {
		return fmt.Errorf("group %q must be NAME=col1[,col2...]", v)
	}
	var ms, passive []string
	for _, m := range strings.Split(members, ",") {
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
	*g = append(*g, model.Site{Name: name, Members: ms, Passive: passive})
	return nil
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
func completeSites(sites []model.Site, columns []string) ([]model.Site, error) {
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
	out := append([]model.Site(nil), sites...)
	for _, c := range columns {
		if covered[c] {
			continue
		}
		if names[c] {
			return nil, fmt.Errorf("site name %q collides with the ungrouped column %q", c, c)
		}
		out = append(out, model.Site{Name: c, Members: []string{c}})
		names[c] = true
	}
	return out, nil
}

// promptGrouping interactively collects decay-site definitions from the user.
// Each line is "NAME = col1, col2 ..."; an empty line ends input. Columns left
// ungrouped become their own sites later (completeSites). Returns the explicitly
// grouped sites only.
func promptGrouping(w io.Writer, in *bufio.Scanner, columns []string) []model.Site {
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
	var sites []model.Site
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
		for _, m := range strings.Split(mem, ",") {
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
		sites = append(sites, model.Site{Name: name, Members: ms, Passive: passive})
		fmt.Fprintf(w, "  site %q = %s\n", name, strings.Join(ms, ", "))
	}
	return sites
}

// promptInitial asks which site is the initial ionization site, defaulting to
// def (falling back to the first site if def is not one of them).
func promptInitial(w io.Writer, in *bufio.Scanner, sites []model.Site, def string) string {
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

var (
	reMolec = regexp.MustCompile(`(?m)^\s*molec\s*=\s*"?([^"\s#]+)"?`)
	reBasis = regexp.MustCompile(`(?m)^\s*basis\s*=\s*"?([^"\s#]+)"?`)
)

// readRunMeta does a light scan of dip.in for the molecule and basis labels
// (the `molec="..."` / `basis="..."` shell assignments). A full dip.in parser
// is a separate concern (sketch: parse/infile.go); this just enriches metadata
// when available and silently returns empties otherwise.
func readRunMeta(path string) (molec, basis string) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", ""
	}
	s := string(b)
	if m := reMolec.FindStringSubmatch(s); m != nil {
		molec = strings.TrimSpace(m[1])
	}
	if m := reBasis.FindStringSubmatch(s); m != nil {
		basis = strings.TrimSpace(m[1])
	}
	return molec, basis
}
