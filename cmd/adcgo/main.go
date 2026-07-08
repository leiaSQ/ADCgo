// Command adcgo is the ADCgo CLI.
//
// Without -dip it ingests an FCIDUMP and reports the reference energy and the
// RHF-MP2 correlation energy (the M0 integral-ingestion check). With -dip it
// solves the DIP-ADC(2) double-ionization problem and writes the dication states
// (energies, pole strengths, leading two-hole configurations) as JSON.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"time"

	"adcgo/internal/adc/analyze"
	"adcgo/internal/adc/backend"
	"adcgo/internal/adc/dip"
	"adcgo/internal/adc/fcidump"
	"adcgo/internal/adc/integrals"
	"adcgo/internal/adc/lanczos"
	"adcgo/internal/adc/mo"
	"adcgo/internal/adc/mp"
	"adcgo/internal/adc/sip"
	"adcgo/internal/adc/spectrum"
)

func main() {
	path := flag.String("fcidump", "", "path to an FCIDUMP file (MO integrals)")
	doDIP := flag.Bool("dip", false, "solve DIP-ADC(2) and emit dication states as JSON")
	doSIP := flag.Bool("sip", false, "solve IP-ADC(n) (non-Dyson) and emit cation states as JSON")
	order := flag.Int("order", 3, "SIP ADC order: 2 or 3")
	solver := flag.String("solver", "lanczos", "eigensolver: lanczos | dense")
	spinSel := flag.String("spin", "both", "spin sector: both | singlet | triplet")
	psThresh := flag.Float64("ps-thresh", 1.0, "drop states with pole strength below this (percent)")
	coeffThresh := flag.Float64("coeff-thresh", 0.1, "drop leading components with |coeff| below this")
	blocks := flag.Int("blocks", 30, "max block-Lanczos iterations")
	moPath := flag.String("mo", "", "MO-coefficient/overlap sidecar for atom-resolved 2h populations")
	sym := flag.String("sym", "all", "target dication irrep: all | none | <0-based index>")
	backendName := flag.String("backend", "gonum", "linear-algebra backend: gonum | hip | cuda | auto (auto calibrates and picks per sector; build-tag gated)")
	out := flag.String("out", "", "write JSON to this file (default stdout)")
	profile := flag.Bool("profile", false, "print per-sector solver phase timings to stderr")

	doSpectrum := flag.Bool("spectrum", false, "emit the decay-channel stick spectrum instead of the solver document (needs -dip or -sip; DIP needs -mo)")
	initAtom := flag.String("init-atom", "O", "initial core-ionized site for DIP decay channels (overridden by the interactive prompt)")
	initOrbital := flag.String("init-orbital", "", "optional initial-orbital label recorded in the spectrum meta")
	stRatio := flag.Float64("st-ratio", 3.0, "singlet:triplet ratio recorded in the spectrum meta for the plotting layer")
	minWeight := flag.Float64("min-weight", 0, "drop decay channels with weight <= this")
	minFraction := flag.Float64("min-fraction", 0, "drop decay channels below this fraction of a state's 2h population (0..1)")
	includeZero := flag.Bool("include-zero", false, "emit the full canonical channel set per state, even at zero weight")
	var groups groupFlag
	flag.Var(&groups, "group", "decay-site grouping NAME=col1,col2 (repeatable; ~col makes a column passive); a bare -group prompts interactively; default each population column is its own site")
	flag.Parse()

	if *path == "" {
		fmt.Fprintln(os.Stderr, "usage: adcgo -fcidump <file> [-dip ...]")
		os.Exit(2)
	}

	d, err := fcidump.ReadFile(*path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "adcgo:", err)
		os.Exit(1)
	}

	if *doDIP && *doSIP {
		fmt.Fprintln(os.Stderr, "adcgo: -dip and -sip are mutually exclusive")
		os.Exit(2)
	}

	if *doSpectrum && !*doDIP && !*doSIP {
		fmt.Fprintln(os.Stderr, "adcgo: -spectrum needs -dip or -sip")
		os.Exit(2)
	}

	specCfg := specConfig{
		enabled:     *doSpectrum,
		initAtom:    *initAtom,
		initOrbital: *initOrbital,
		stRatio:     *stRatio,
		groups:      groups.sites,
		interactive: groups.interactive,
		classify: spectrum.Options{
			MinWeight: *minWeight, MinFraction: *minFraction, IncludeZero: *includeZero,
		},
	}

	if *doDIP {
		cfg := dipConfig{
			solver: *solver, spinSel: *spinSel, moPath: *moPath, out: *out, sym: *sym,
			backend:  *backendName,
			psThresh: *psThresh, coeffThresh: *coeffThresh, blocks: *blocks,
			profile: *profile,
			spec:    specCfg,
		}
		if err := runDIP(d, cfg); err != nil {
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(1)
		}
		return
	}

	if *doSIP {
		cfg := sipConfig{
			solver: *solver, out: *out, sym: *sym, backend: *backendName, order: *order,
			psThresh: *psThresh, coeffThresh: *coeffThresh, blocks: *blocks,
			profile: *profile,
			spec:    specCfg,
		}
		if err := runSIP(d, cfg); err != nil {
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(1)
		}
		return
	}

	reportMP2(*path, d)
}

func reportMP2(path string, d *fcidump.Data) {
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	eref := referenceEnergy(d, nocc)
	ecorr := mp.MP2Corr(d, nocc, eps)

	fmt.Printf("FCIDUMP        : %s\n", path)
	fmt.Printf("NORB / NELEC   : %d / %d  (nocc=%d, nvir=%d)\n", d.NORB, d.NELEC, nocc, d.NORB-nocc)
	fmt.Printf("E(core)        : % .10f Ha\n", d.Ecore)
	fmt.Printf("E(HF, recon.)  : % .10f Ha\n", eref)
	fmt.Printf("E(MP2 corr.)   : % .10f Ha\n", ecorr)
	fmt.Printf("E(MP2 total)   : % .10f Ha\n", eref+ecorr)
	fmt.Printf("HOMO / LUMO    : % .6f / % .6f Ha\n", eps[nocc-1], eps[nocc])
}

// Document is the native ADCgo DIP output.
type Document struct {
	NORB    int              `json:"norb"`
	NELEC   int              `json:"nelec"`
	Solver  string           `json:"solver"`
	Sectors []analyze.Sector `json:"sectors"`
}

type dipConfig struct {
	solver, spinSel, moPath, out, sym, backend string
	psThresh, coeffThresh                      float64
	blocks                                     int
	profile                                    bool
	spec                                       specConfig
}

// reportTiming prints one solver's phase breakdown to stderr. The percentages are
// what matter: a phase that dominates because it runs at the wrong BLAS level looks
// identical, in flop count, to one that does not.
func reportTiming(label string, n, main int, tm lanczos.Timing) {
	tot := tm.Total()
	if tot == 0 {
		return
	}
	pct := func(d time.Duration) float64 { return 100 * float64(d) / float64(tot) }
	fmt.Fprintf(os.Stderr,
		"profile %-22s n=%-6d b=%-3d total=%8.2fs | apply %6.2fs (%4.1f%%)  orth %7.2fs (%4.1f%%)  proj %7.2fs (%4.1f%%)  eig %6.2fs (%4.1f%%)  back %6.2fs (%4.1f%%)\n",
		label, n, main, tot.Seconds(),
		tm.Apply.Seconds(), pct(tm.Apply),
		tm.Orth.Seconds(), pct(tm.Orth),
		tm.Proj.Seconds(), pct(tm.Proj),
		tm.Eig.Seconds(), pct(tm.Eig),
		tm.Back.Seconds(), pct(tm.Back))
}

func runDIP(d *fcidump.Data, cfg dipConfig) error {
	spins, err := selectSpins(cfg.spinSel)
	if err != nil {
		return err
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ch, err := newChooser(cfg.backend, cfg.profile)
	if err != nil {
		return err
	}
	opts := analyze.Options{PSThresh: cfg.psThresh, CoeffThresh: cfg.coeffThresh}

	// Symmetry mode: "none" collapses every orbital into one totally-symmetric
	// group (the full spectrum in one sector); "all"/an index blocks the problem
	// per point-group irrep. The integral store must share the chosen grouping.
	orbSym, syms, err := selectSymmetry(cfg.sym, d)
	if err != nil {
		return err
	}
	ints := integrals.New(d, nocc, orbSym)

	var moData *mo.Data
	if cfg.moPath != "" {
		if moData, err = mo.ReadFile(cfg.moPath); err != nil {
			return err
		}
	}

	doc := Document{NORB: d.NORB, NELEC: d.NELEC, Solver: cfg.solver}
	for _, spin := range spins {
		for _, targetSym := range syms {
			sp := dip.NewSpace(nocc, d.NORB, orbSym, targetSym, spin)
			if sp.Size() == 0 {
				continue // no configurations in this (irrep, spin) sector
			}
			label := fmt.Sprintf("dip spin=%d irrep=%d", spin, targetSym+1)
			lopts := lanczos.Options{MaxBlocks: cfg.blocks}
			n, b := sp.Size(), sp.MainBlockSize()

			var be backend.Backend
			if cfg.solver == "dense" {
				be = ch.pickDense(label, n)
			} else {
				be = ch.pickLanczos(label, n, b, lanczos.SubspaceDim(n, b, lopts),
					func(cand backend.Backend) time.Duration {
						m := dip.New(sp, ints, eps, cand)
						defer m.Release()
						return timeApplyBlock(cand, n, b, m.ApplyBlock)
					})
			}
			mx := dip.New(sp, ints, eps, be)

			var res lanczos.Result
			switch cfg.solver {
			case "dense":
				res = lanczos.SolveDense(mx, be)
			case "lanczos":
				res = lanczos.Solve(mx, be, lopts)
			default:
				return fmt.Errorf("unknown solver %q (want lanczos or dense)", cfg.solver)
			}
			// Reclaim the sector's resident operator before the next one is assembled;
			// on a device this is up to 0.5 GB, and the memory check depends on it.
			mx.Release()
			if cfg.profile {
				reportTiming(label, sp.Size(), sp.MainBlockSize(), res.Timing)
			}

			var pe *analyze.PopEngine
			if moData != nil {
				pe = analyze.NewPopEngine(sp, moData)
			}
			doc.Sectors = append(doc.Sectors, analyze.BuildSector(sp, res, opts, pe))
		}
	}

	if cfg.spec.enabled {
		if moData == nil {
			return fmt.Errorf("-spectrum -dip needs -mo (decay channels require atom-resolved populations)")
		}
		spec, err := buildDIPSpectrum(doc.Sectors, moData, cfg.spec)
		if err != nil {
			return err
		}
		return emitJSON(spec, cfg.out)
	}

	return emitJSON(doc, cfg.out)
}

// emitJSON writes v as indented JSON to out (stdout when out == "").
func emitJSON(v any, out string) error {
	enc, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	enc = append(enc, '\n')
	if out == "" {
		_, err = os.Stdout.Write(enc)
		return err
	}
	return os.WriteFile(out, enc, 0o644)
}

// SIPDocument is the native ADCgo single-ionization output.
type SIPDocument struct {
	NORB    int                 `json:"norb"`
	NELEC   int                 `json:"nelec"`
	Order   int                 `json:"order"`
	Solver  string              `json:"solver"`
	Sectors []analyze.SIPSector `json:"sectors"`
}

type sipConfig struct {
	solver, out, sym, backend string
	order                     int
	psThresh, coeffThresh     float64
	blocks                    int
	profile                   bool
	spec                      specConfig
}

func runSIP(d *fcidump.Data, cfg sipConfig) error {
	if cfg.order != 2 && cfg.order != 3 {
		return fmt.Errorf("unknown -order %d (want 2 or 3)", cfg.order)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ch, err := newChooser(cfg.backend, cfg.profile)
	if err != nil {
		return err
	}
	opts := analyze.Options{PSThresh: cfg.psThresh, CoeffThresh: cfg.coeffThresh}

	orbSym, syms, err := selectSymmetry(cfg.sym, d)
	if err != nil {
		return err
	}
	ints := integrals.New(d, nocc, orbSym)

	doc := SIPDocument{NORB: d.NORB, NELEC: d.NELEC, Order: cfg.order, Solver: cfg.solver}
	for _, targetSym := range syms {
		sp := sip.NewSpace(nocc, d.NORB, orbSym, targetSym)
		if sp.MainBlockSize() == 0 {
			continue // no 1h configurations in this irrep
		}
		label := fmt.Sprintf("sip irrep=%d", targetSym+1)
		lopts := lanczos.Options{MaxBlocks: cfg.blocks}
		n, b := sp.Size(), sp.MainBlockSize()

		var be backend.Backend
		if cfg.solver == "dense" {
			be = ch.pickDense(label, n)
		} else {
			be = ch.pickLanczos(label, n, b, lanczos.SubspaceDim(n, b, lopts),
				func(cand backend.Backend) time.Duration {
					m := sip.New(sp, ints, eps, cfg.order, cand)
					defer m.Release()
					return timeApplyBlock(cand, n, b, m.ApplyBlock)
				})
		}
		mx := sip.New(sp, ints, eps, cfg.order, be)

		var res lanczos.Result
		switch cfg.solver {
		case "dense":
			res = lanczos.SolveDense(mx, be)
		case "lanczos":
			res = lanczos.Solve(mx, be, lopts)
		default:
			return fmt.Errorf("unknown solver %q (want lanczos or dense)", cfg.solver)
		}
		if cfg.profile {
			reportTiming(label, sp.Size(), sp.MainBlockSize(), res.Timing)
		}
		sector := analyze.BuildSIPSector(sp, res, mx.FMatrix(), opts)
		mx.Release()
		doc.Sectors = append(doc.Sectors, sector)
	}

	if cfg.spec.enabled {
		spec, err := spectrum.BuildSIP(doc.Sectors, d.OrbSym, spectrum.SIPOptions{})
		if err != nil {
			return err
		}
		return emitJSON(spec, cfg.out)
	}

	return emitJSON(doc, cfg.out)
}

// selectSymmetry resolves the -sym flag into the orbital-symmetry labels to hand
// the solver (nil disables symmetry) and the list of target dication irreps to
// loop over.
func selectSymmetry(sel string, d *fcidump.Data) (orbSym []int, syms []int, err error) {
	switch sel {
	case "none":
		return nil, []int{0}, nil
	case "all":
		nsym := numIrreps(d.OrbSym, d.NORB)
		syms = make([]int, nsym)
		for i := range nsym {
			syms[i] = i
		}
		return d.OrbSym, syms, nil
	default:
		idx, e := strconv.Atoi(sel)
		if e != nil || idx < 0 {
			return nil, nil, fmt.Errorf("unknown -sym %q (want all, none, or a 0-based irrep index)", sel)
		}
		return d.OrbSym, []int{idx}, nil
	}
}

// numIrreps is the number of symmetry groups implied by the ORBSYM labels (the
// smallest power of two spanning them), matching integrals.Store's grouping.
func numIrreps(orbSym []int, norb int) int {
	if orbSym == nil {
		return 1
	}
	max0 := 0
	for o := range norb {
		if lab := orbSym[o] - 1; lab > max0 {
			max0 = lab
		}
	}
	n := 1
	for n < max0+1 {
		n <<= 1
	}
	return n
}

func selectSpins(sel string) ([]dip.Spin, error) {
	switch sel {
	case "both":
		return []dip.Spin{dip.Singlet, dip.Triplet}, nil
	case "singlet":
		return []dip.Spin{dip.Singlet}, nil
	case "triplet":
		return []dip.Spin{dip.Triplet}, nil
	default:
		return nil, fmt.Errorf("unknown spin %q (want both, singlet, or triplet)", sel)
	}
}

// referenceEnergy reconstructs the closed-shell RHF energy from the MO
// integrals: E = Ecore + Σ_{i∈occ} [ 2 h_ii + Σ_{j∈occ} (2(ii|jj) − (ij|ji)) ].
func referenceEnergy(d *fcidump.Data, nocc int) float64 {
	e := d.Ecore
	for i := range nocc {
		e += 2 * d.OneE(i, i)
		for j := range nocc {
			e += 2*d.TwoE(i, i, j, j) - d.TwoE(i, j, j, i)
		}
	}
	return e
}
