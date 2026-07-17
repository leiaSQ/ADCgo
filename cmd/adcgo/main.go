// Command adcgo is the ADCgo CLI.
//
// Without -dip it ingests an FCIDUMP and reports the reference energy and the
// RHF-MP2 correlation energy (the M0 integral-ingestion check). With -dip it
// solves the DIP-ADC(2) double-ionization problem and writes the dication states
// (energies, pole strengths, leading two-hole configurations) as JSON.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
)

// errInterrupted signals that a checkpointing solve stopped early on a SIGUSR1 (walltime
// warning) after writing its checkpoint. main turns it into exit code 64 so a daisychain
// wrapper knows to resume rather than treat it as a hard failure (see runADCgo_helix_melanin
// family). exit 0 means the solve completed.
var errInterrupted = errors.New("solve interrupted; checkpoint written")

const exitResumeNeeded = 64

// installStopSignal arranges for SIGUSR1 to flip the returned flag, which a checkpointing
// lanczos loop polls at each block boundary to checkpoint-and-exit cleanly (exit 64). It is
// installed unconditionally at the very start of main — before any file I/O — so a SIGUSR1
// (SLURM --signal=B:USR1@<grace>) is never lost to the default "terminate" disposition, even
// if it arrives during startup. Only SIGUSR1 is trapped: SIGTERM keeps its default behavior
// so `scancel` and the walltime hard-kill still terminate the process (the periodic
// checkpoint covers an ungraceful death). When checkpointing is off nothing polls the flag,
// so a stray SIGUSR1 is simply ignored.
func installStopSignal() *atomic.Bool {
	stop := new(atomic.Bool)
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGUSR1)
	go func() {
		for range ch {
			stop.Store(true)
			fmt.Fprintln(os.Stderr, "adcgo: SIGUSR1 received; checkpointing at the next block boundary")
		}
	}()
	return stop
}

func main() {
	stopSig := installStopSignal()

	path := flag.String("fcidump", "", "path to an FCIDUMP file (MO integrals)")
	doDIP := flag.Bool("dip", false, "solve DIP-ADC(2) and emit dication states as JSON")
	doSIP := flag.Bool("sip", false, "solve IP-ADC(n) (non-Dyson) and emit cation states as JSON")
	order := flag.Int("order", 3, "SIP ADC order: 2, 3, or 4 (4 = CVS Dyson ADC(4), needs -core)")
	solver := flag.String("solver", "lanczos", "eigensolver: lanczos | lanczos-lowmem | davidson | dense")
	lowmemBlock := flag.Int("lowmem-block", 0, "-solver lanczos-lowmem: block width. 0 = the 2h main-space size, the faithful theADCcode short-recurrence solve (Tarantelli subspace-iteration gate + banded eigensolver, ~3×(n×main) resident — a fat-memory CPU node); a value below main selects the device-frugal full-reorthogonalization mode (only 3 blocks on the GPU, full basis in host RAM), which is exact on the states it reaches but a block smaller than main cannot span every pole-carrying direction")
	spinSel := flag.String("spin", "both", "spin sector: both | singlet | triplet")
	psThresh := flag.Float64("ps-thresh", 1.0, "drop states with pole strength below this (percent)")
	coeffThresh := flag.Float64("coeff-thresh", 0.1, "drop leading components with |coeff| below this")
	blocks := flag.Int("blocks", 100, "block-Lanczos iterations; Krylov subspace = blocks × 2h-space size (theADCcode's 'iter', whose reference DIP runs used 100)")
	nroots := flag.Int("nroots", 20, "-solver davidson: number of lowest roots to converge (theADCcode's 'nroots')")
	convthr := flag.Float64("convthr", 1e-3, "-solver davidson: residual 2-norm convergence threshold in a.u. (theADCcode's 'convthr')")
	maxdavsp := flag.Int("maxdavsp", 100, "-solver davidson: maximum subspace dimension before a thick restart (theADCcode's 'maxdavsp')")
	maxdavit := flag.Int("maxdavit", 200, "-solver davidson: iteration cap before giving up unconverged")
	moPath := flag.String("mo", "", "MO-coefficient/overlap sidecar for atom-resolved 2h populations")
	sym := flag.String("sym", "all", "target dication irrep: all | none | <0-based index>")
	coreOrb := flag.String("core", "", "CVS core orbitals for -order 4: comma-separated 0-based occupied indices (e.g. 0)")
	backendName := flag.String("backend", "gonum", "linear-algebra backend: gonum | hip | cuda | auto (auto calibrates and picks per sector; build-tag gated)")
	gpus := flag.Int("gpus", 0, "-backend cuda|hip only: max GPUs for concurrent per-sector solves (0 = all visible). Independent sectors (DIP spin×irrep, SIP irrep) run one per GPU")
	matfree := flag.String("matfree", "off", "matrix-free apply of large CVS-ADC(4) coupling blocks (recompute vs store): off | auto | on. Trades resident memory for per-mat-vec recompute; auto switches per block using -maxmem")
	maxMemGB := flag.Float64("maxmem", 4.0, "matrix-free -matfree=auto threshold: a coupling block whose dense size exceeds this many GB is applied matrix-free")
	wert3 := flag.Bool("wert3", true, "include the WERT3 5th-order 3h2p-diagonal correction in CVS-ADC(4) (the full EIGAB effective diagonal theADCcode itself uses; bit-exact vs its FT19 tape). -wert3=false for the bare 0th-order 3h2p diagonal.")
	sigma := flag.String("sigma", "auto", "static self-energy added to the SIP main block: auto | off | three | four | fplus | infinite. The ADC matrix code does not build Σ (theADCcode keeps it in a separate &self-energy module and subtracts it); omitting it shifts every main line by ~0.2-0.35 eV. auto = infinite, the all-order resolvent resummation, bit-exact vs theADCcode.")
	sigmaAkrit := flag.Float64("sigma-akrit", 0, "Σ(∞) resolvent convergence threshold on Σ(Δx)² (0 = converge tightly; theADCcode's own default is 1e-9)")
	sigmaMaxIt := flag.Int("sigma-maxit", 0, "Σ(∞) resolvent iteration cap (0 = 200; theADCcode's own default is 30)")
	out := flag.String("out", "", "write JSON to this file (default stdout)")
	profile := flag.Bool("profile", false, "print per-sector solver phase timings to stderr")
	checkpoint := flag.String("checkpoint", "", "-solver lanczos only: base path for block-Krylov checkpoints, so a solve can resume in a later process after a walltime kill. Each sector appends a suffix (SIP: .i<irrep>). A SIGUSR1 (SLURM --signal=B:USR1@<grace>) makes the run checkpoint and exit 64 (\"resume needed\"); exit 0 means done. Empty = no checkpointing")
	checkpointEvery := flag.Int("checkpoint-every", 25, "-checkpoint only: also save every N blocks for crash resilience (<=0 = save only on the stop signal)")

	doTDM := flag.Bool("tdm", false, "emit RASSI-like transition dipole moments instead of the solver document: ion→ion emission (element 1), Dyson photoionization (element 2), and — for -order 4 — core→valence X-ray emission; needs -sip -mo (with dipole integrals)")
	flag.BoolVar(doTDM, "rassi", false, "alias for -tdm")
	tdmOsc := flag.Float64("tdm-osc-thresh", 1e-6, "drop photoionization channels with oscillator strength below this")
	tdmISR := flag.Int("tdm-isr", -1, "order of the ISR property matrix behind -tdm: 0 = zeroth (uncorrelated), 2 = correlation-corrected. Default: 2 for -order 2/3, where it is order-consistent with the secular matrix; 0 for -order 4, where it is not (an ADC(4)-consistent property matrix needs 3rd/4th-order terms that do not exist yet) — pass -tdm-isr 2 to opt in anyway")

	doSpectrum := flag.Bool("spectrum", false, "emit a stick spectrum instead of the solver document (needs -dip or -sip). DIP with -mo gives decay channels; without -mo it falls back to the bare per-state spectrum. SIP decomposes per orbital")
	doBare := flag.Bool("bare", false, "emit a bare per-state stick spectrum (energy + pole strength, one \"states\" channel) instead of decay-channel/per-orbital classification; implies -spectrum")
	initAtom := flag.String("init-atom", "O", "initial core-ionized site for DIP decay channels (overridden by the interactive prompt)")
	initOrbital := flag.String("init-orbital", "", "optional initial-orbital label recorded in the spectrum meta")
	stRatio := flag.Float64("st-ratio", 3.0, "singlet:triplet ratio recorded in the spectrum meta for the plotting layer")
	minWeight := flag.Float64("min-weight", 0, "drop decay channels with weight <= this")
	minFraction := flag.Float64("min-fraction", 0, "drop decay channels below this fraction of a state's 2h population (0..1)")
	includeZero := flag.Bool("include-zero", false, "emit the full canonical channel set per state, even at zero weight")
	var groups groupFlag
	flag.Var(&groups, "group", "decay-site grouping NAME=col1,col2 (repeatable; ~col makes a column passive); a bare -group prompts interactively; default each population column is its own site")
	convert := flag.String("convert", "", "read a previously emitted solver document JSON (the default -dip/-sip output) and emit its bare stick spectrum without re-solving; needs -dip or -sip to say which kind")
	flag.Parse()

	// -convert post-processes an existing output file; it re-solves nothing and so
	// needs no FCIDUMP.
	if *convert != "" {
		if *doDIP == *doSIP { // both set, or neither
			fmt.Fprintln(os.Stderr, "adcgo: -convert needs exactly one of -dip or -sip")
			os.Exit(2)
		}
		// A DIP solver document written with -mo already carries per-atom two-hole
		// populations, so -group/-spectrum can regroup it into a decay-channel spectrum
		// with no re-solve (the populations, not the eigenvectors, are what the classifier
		// needs). Without -mo, or with -bare, fall through to the bare per-state spectrum.
		if *doDIP && *moPath != "" && !*doBare && (*doSpectrum || len(groups.sites) > 0 || groups.interactive) {
			md, err := mo.ReadFile(*moPath)
			if err != nil {
				fmt.Fprintln(os.Stderr, "adcgo:", err)
				os.Exit(1)
			}
			cfg := specConfig{
				enabled: true, initAtom: *initAtom, initOrbital: *initOrbital, stRatio: *stRatio,
				groups: groups.sites, interactive: groups.interactive,
				classify: spectrum.Options{MinWeight: *minWeight, MinFraction: *minFraction, IncludeZero: *includeZero},
			}
			if err := runDIPGroupedConvert(*convert, md, cfg, *out); err != nil {
				fmt.Fprintln(os.Stderr, "adcgo:", err)
				os.Exit(1)
			}
			return
		}
		if err := runBareConvert(*convert, *doDIP, *out); err != nil {
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(1)
		}
		return
	}

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

	// -bare is shorthand for a spectrum without decay-channel/per-orbital classification.
	doSpec := *doSpectrum || *doBare

	if doSpec && !*doDIP && !*doSIP {
		fmt.Fprintln(os.Stderr, "adcgo: -spectrum/-bare needs -dip or -sip")
		os.Exit(2)
	}

	if *doTDM {
		if !*doSIP {
			fmt.Fprintln(os.Stderr, "adcgo: -tdm needs -sip")
			os.Exit(2)
		}
		if doSpec {
			fmt.Fprintln(os.Stderr, "adcgo: -tdm and -spectrum/-bare are mutually exclusive")
			os.Exit(2)
		}
		if *moPath == "" {
			fmt.Fprintln(os.Stderr, "adcgo: -tdm needs -mo (a sidecar with dipole integrals)")
			os.Exit(2)
		}
		if *tdmISR >= 0 && *tdmISR != 0 && *tdmISR != 2 {
			fmt.Fprintf(os.Stderr, "adcgo: -tdm-isr %d is not available (want 0 or 2)\n", *tdmISR)
			os.Exit(2)
		}
	}

	specCfg := specConfig{
		enabled:     doSpec,
		bare:        *doBare,
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
			backend: *backendName, gpus: *gpus,
			psThresh: *psThresh, coeffThresh: *coeffThresh, blocks: *blocks,
			nroots: *nroots, maxdavsp: *maxdavsp, maxdavit: *maxdavit, convthr: *convthr,
			lowmemBlock: *lowmemBlock,
			profile:     *profile,
			spec:        specCfg,
		}
		if err := runDIP(d, cfg); err != nil {
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(1)
		}
		return
	}

	if *doSIP {
		core, err := parseCoreOrbitals(*coreOrb)
		if err != nil {
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(1)
		}
		cfg := sipConfig{
			solver: *solver, out: *out, sym: *sym, backend: *backendName, gpus: *gpus, order: *order,
			psThresh: *psThresh, coeffThresh: *coeffThresh, blocks: *blocks,
			nroots: *nroots, maxdavsp: *maxdavsp, maxdavit: *maxdavit, convthr: *convthr,
			profile: *profile,
			spec:    specCfg,
			core:    core,
			moPath:  *moPath, tdm: *doTDM, tdmOsc: *tdmOsc, tdmISR: *tdmISR,
		}
		mfMode, err := parseMatFree(*matfree)
		if err != nil {
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(2)
		}
		cfg.matFree = mfMode
		cfg.matFreeBudget = int64(*maxMemGB * (1 << 30))
		cfg.wert3 = *wert3
		cfg.sigma = *sigma
		cfg.sigmaAkrit = *sigmaAkrit
		cfg.sigmaMaxIt = *sigmaMaxIt
		cfg.ckpt = *checkpoint
		cfg.ckptEvery = *checkpointEvery
		if cfg.ckpt != "" {
			cfg.stop = stopSig
		}
		if err := runSIP(d, cfg); err != nil {
			if errors.Is(err, errInterrupted) {
				fmt.Fprintln(os.Stderr, "adcgo: checkpoint written; resume needed")
				os.Exit(exitResumeNeeded)
			}
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
	gpus                                       int // -gpus: cap on concurrent per-sector GPUs (0 = all)
	psThresh, coeffThresh                      float64
	blocks                                     int
	nroots, maxdavsp, maxdavit                 int     // -solver davidson
	convthr                                    float64 // -solver davidson
	lowmemBlock                                int     // -solver lanczos-lowmem block width (0 = main)
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

// davidsonOpts assembles the block-Davidson options from the -nroots/-maxdavsp/-maxdavit/
// -convthr flags. wantFull retains the full Ritz vectors (the SIP-TDM path needs them).
func davidsonOpts(nroots, maxdavsp, maxdavit int, convthr float64, wantFull bool) lanczos.Options {
	return lanczos.Options{
		NRoots: nroots, MaxDim: maxdavsp, MaxIters: maxdavit, ConvThr: convthr, WantFull: wantFull,
	}
}

// validateSolver rejects an unknown -solver up front, so the per-sector code (which may
// run concurrently across GPUs) can assume a valid solver and needs no error return.
func validateSolver(solver string) error {
	switch solver {
	case "dense", "lanczos", "lanczos-lowmem", "davidson":
		return nil
	default:
		return fmt.Errorf("unknown solver %q (want lanczos, lanczos-lowmem, davidson or dense)", solver)
	}
}

// solveDIPSector solves one (spin, irrep) DIP sector on the backend chosen by ch and
// returns its analyzed sector. It is self-contained (no shared mutable state beyond the
// read-only ints/eps/moData), so multiple sectors can run concurrently, each on its own
// GPU, via a per-worker single-backend chooser. cfg.solver is validated by the caller.
func solveDIPSector(ch *chooser, cfg dipConfig, sp *dip.Space, ints *integrals.Store, eps []float64, spin dip.Spin, targetSym int, moData *mo.Data, opts analyze.Options) analyze.Sector {
	label := fmt.Sprintf("dip spin=%d irrep=%d", spin, targetSym+1)
	lopts := lanczos.Options{MaxBlocks: cfg.blocks, LowMemBlock: cfg.lowmemBlock}
	davOpts := davidsonOpts(cfg.nroots, cfg.maxdavsp, cfg.maxdavit, cfg.convthr, false)
	n, b := sp.Size(), sp.MainBlockSize()
	subspace := lanczos.SubspaceDim(n, b, lopts)
	probeB := b
	switch cfg.solver {
	case "davidson":
		subspace = lanczos.DavidsonSubspaceDim(n, davOpts)
	case "lanczos-lowmem":
		// The short-recurrence driver keeps only a few n×block panels resident, not the whole
		// basis. Size the sector by that footprint (LowMemSectorBytes) so the chooser's device
		// fit check sees the real, much smaller memory — expressed as an equivalent "subspace"
		// dim the existing SectorBytes(n, dim, b) formula reproduces (its basis term is n·dim·8).
		if cfg.lowmemBlock > 0 && cfg.lowmemBlock < b {
			probeB = cfg.lowmemBlock
		}
		subspace = int(lanczos.LowMemSectorBytes(n, probeB) / uint64(8*n))
	}

	var be backend.Backend
	if cfg.solver == "dense" {
		be = ch.pickDense(label, n)
	} else {
		be = ch.pickLanczos(label, n, probeB, subspace,
			func(cand backend.Backend) time.Duration {
				m := dip.New(sp, ints, eps, cand)
				defer m.Release()
				return timeApplyBlock(cand, n, probeB, m.ApplyBlock)
			})
	}
	mx := dip.New(sp, ints, eps, be)

	var res lanczos.Result
	switch cfg.solver {
	case "dense":
		res = lanczos.SolveDense(mx, be)
	case "lanczos":
		res = lanczos.Solve(mx, be, lopts)
	case "lanczos-lowmem":
		res = lanczos.SolveLowMem(mx, be, lopts)
	case "davidson":
		res = lanczos.SolveDavidson(mx, be, davOpts)
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
	return analyze.BuildSector(sp, res, opts, pe)
}

func runDIP(d *fcidump.Data, cfg dipConfig) error {
	spins, err := selectSpins(cfg.spinSel)
	if err != nil {
		return err
	}
	if err := validateSolver(cfg.solver); err != nil {
		return err
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ch, err := newChooser(cfg.backend, cfg.profile, cfg.gpus)
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

	// Enumerate the non-empty (spin, irrep) sectors. These are independent solves;
	// with a multi-GPU pool they run concurrently (one GPU per sector), otherwise
	// serially. Either way results are emitted in this deterministic order.
	type dipItem struct {
		spin      dip.Spin
		targetSym int
		sp        *dip.Space
	}
	var items []dipItem
	for _, spin := range spins {
		for _, targetSym := range syms {
			sp := dip.NewSpace(nocc, d.NORB, orbSym, targetSym, spin)
			if sp.Size() == 0 {
				continue // no configurations in this (irrep, spin) sector
			}
			items = append(items, dipItem{spin, targetSym, sp})
		}
	}

	results := make([]analyze.Sector, len(items))
	solve := func(w *chooser, i int) error {
		it := items[i]
		results[i] = solveDIPSector(w, cfg, it.sp, ints, eps, it.spin, it.targetSym, moData, opts)
		return nil
	}
	if len(ch.pool) >= 2 {
		if err := ch.runConcurrent(len(items), solve); err != nil {
			return err
		}
	} else {
		for i := range items {
			_ = solve(ch, i)
		}
	}
	doc.Sectors = append(doc.Sectors, results...)

	if cfg.spec.enabled {
		// Without -mo (or with an explicit -bare) there are no atom-resolved
		// populations to classify into decay channels, so fall back to the bare
		// per-state eigenvalue spectrum.
		if cfg.spec.bare || moData == nil {
			return emitJSON(spectrum.BuildBareDIP(doc.Sectors, spectrum.BareOptions{}), cfg.out)
		}
		spec, err := buildDIPSpectrum(doc.Sectors, moData, cfg.spec)
		if err != nil {
			return err
		}
		return emitJSON(spec, cfg.out)
	}

	return emitJSON(doc, cfg.out)
}

// runBareConvert reads a previously emitted solver document JSON (a DIP Document or
// SIP SIPDocument) and re-emits it as a bare per-state stick spectrum, reusing the
// same spectrum.BuildBare* builders the -bare solve path uses — the document already
// serializes the analyze.Sector / analyze.SIPSector slices they consume, so no
// re-solving is needed. The caller passes dip=true for a DIP document, false for SIP.
func runBareConvert(path string, dip bool, out string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var spec *spectrum.Spectrum
	if dip {
		var doc Document
		if err := json.Unmarshal(b, &doc); err != nil {
			return fmt.Errorf("parse %s as a DIP solver document: %w", path, err)
		}
		if len(doc.Sectors) == 0 {
			return fmt.Errorf("%s has no sectors (is it a -dip solver document?)", path)
		}
		spec = spectrum.BuildBareDIP(doc.Sectors, spectrum.BareOptions{SourceFiles: []string{path}})
	} else {
		var doc SIPDocument
		if err := json.Unmarshal(b, &doc); err != nil {
			return fmt.Errorf("parse %s as a SIP solver document: %w", path, err)
		}
		if len(doc.Sectors) == 0 {
			return fmt.Errorf("%s has no sectors (is it a -sip solver document?)", path)
		}
		spec = spectrum.BuildBareSIP(doc.Sectors, spectrum.BareOptions{SourceFiles: []string{path}})
	}
	return emitJSON(spec, out)
}

// runDIPGroupedConvert reads a saved DIP solver document and re-emits it as a decay-channel
// stick spectrum grouped into the -group sites — the theADCcode &popana equivalent — without
// re-solving. It reuses buildDIPSpectrum, so the classification is identical to the solve-time
// -spectrum path; only the (expensive) eigensolve is skipped, since the document already
// stores the per-atom two-hole populations the classifier consumes.
func runDIPGroupedConvert(path string, md *mo.Data, cfg specConfig, out string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var doc Document
	if err := json.Unmarshal(b, &doc); err != nil {
		return fmt.Errorf("parse %s as a DIP solver document: %w", path, err)
	}
	if len(doc.Sectors) == 0 {
		return fmt.Errorf("%s has no sectors (is it a -dip solver document?)", path)
	}
	spec, err := buildDIPSpectrum(doc.Sectors, md, cfg)
	if err != nil {
		return err
	}
	return emitJSON(spec, out)
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
	solver, out, sym, backend  string
	gpus                       int // -gpus: cap on concurrent per-sector GPUs (0 = all)
	order                      int
	psThresh, coeffThresh      float64
	blocks                     int
	nroots, maxdavsp, maxdavit int     // -solver davidson
	convthr                    float64 // -solver davidson
	profile                    bool
	spec                       specConfig
	core                       []int // CVS core orbitals (order 4)

	moPath string  // MO/dipole sidecar (required by -tdm)
	tdm    bool    // emit transition dipole moments instead of the solver document
	tdmOsc float64 // photoionization channel oscillator-strength cutoff
	tdmISR int     // ISR property-matrix order: 0 or 2; -1 = pick from -order

	matFree       sip.MatFreeMode        // dense (default) vs matrix-free large ADC(4) blocks
	matFreeBudget int64                  // -matfree=auto per-block dense-size threshold (bytes)
	wert3         bool                   // include the WERT3 5th-order 3h2p-diagonal correction
	sigma         string                 // static self-energy scheme: auto | off | three | four | fplus | infinite
	sigmaAkrit    float64                // Σ(∞) resolvent convergence threshold (0 = converge tightly)
	sigmaMaxIt    int                    // Σ(∞) resolvent iteration cap
	sig           func(i, j int) float64 // resolved Σ, built once per run (nil = off)

	ckpt      string       // -checkpoint base path (lanczos only; "" = off)
	ckptEvery int          // -checkpoint-every: blocks between crash-resilience saves
	stop      *atomic.Bool // set by a stop signal; polled by the checkpointing lanczos loop
}

// parseMatFree maps the -matfree flag to a sip.MatFreeMode.
func parseMatFree(s string) (sip.MatFreeMode, error) {
	switch s {
	case "off", "":
		return sip.MatFreeOff, nil
	case "auto":
		return sip.MatFreeAuto, nil
	case "on":
		return sip.MatFreeOn, nil
	default:
		return sip.MatFreeOff, fmt.Errorf("bad -matfree %q (want off, auto, or on)", s)
	}
}

// parseCoreOrbitals parses the -core flag: comma-separated 0-based occupied indices.
func parseCoreOrbitals(s string) ([]int, error) {
	if s == "" {
		return nil, nil
	}
	var out []int
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		v, err := strconv.Atoi(f)
		if err != nil || v < 0 {
			return nil, fmt.Errorf("bad -core orbital %q (want 0-based occupied index)", f)
		}
		out = append(out, v)
	}
	return out, nil
}

func runSIP(d *fcidump.Data, cfg sipConfig) error {
	if cfg.order < 2 || cfg.order > 4 {
		return fmt.Errorf("unknown -order %d (want 2, 3, or 4)", cfg.order)
	}
	if cfg.order == 4 && len(cfg.core) == 0 {
		return fmt.Errorf("-order 4 is CVS Dyson ADC(4) and requires -core (e.g. -core 0)")
	}
	if err := validateSolver(cfg.solver); err != nil {
		return err
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ch, err := newChooser(cfg.backend, cfg.profile, cfg.gpus)
	if err != nil {
		return err
	}
	opts := analyze.Options{PSThresh: cfg.psThresh, CoeffThresh: cfg.coeffThresh}

	orbSym, syms, err := selectSymmetry(cfg.sym, d)
	if err != nil {
		return err
	}
	ints := integrals.New(d, nocc, orbSym)

	// The static self-energy is a property of the orbital space, not of a sector, so build it
	// once and let every sector's main block subtract the same Σ.
	if cfg.sig, err = buildSigma(cfg, ints, eps, nocc, d.NORB); err != nil {
		return err
	}

	var md *mo.Data
	if cfg.moPath != "" {
		if md, err = mo.ReadFile(cfg.moPath); err != nil {
			return err
		}
	}
	if cfg.tdm && !md.HasDipole {
		return fmt.Errorf("-tdm needs an MO sidecar with dipole integrals (dip_ao); regenerate it with fcidump_common.py")
	}
	if cfg.tdm && len(syms) == 1 && orbSym != nil {
		fmt.Fprintf(os.Stderr, "note: -tdm with a single -sym sector sees only the totally symmetric "+
			"dipole component; the x/y-polarized lines are cross-irrep. Use -sym all for a spectrum.\n")
	}

	doc := SIPDocument{NORB: d.NORB, NELEC: d.NELEC, Order: cfg.order, Solver: cfg.solver}
	var solved []solvedSIP // retained (with live operators) only in -tdm mode

	// Enumerate the non-empty irrep sectors (independent solves).
	type sipItem struct {
		targetSym int
		sp        *sip.Space
	}
	var items []sipItem
	for _, targetSym := range syms {
		var sp *sip.Space
		if cfg.order == 4 {
			sp = sip.NewSpace4(nocc, d.NORB, orbSym, targetSym, cfg.core)
		} else {
			sp = sip.NewSpace(nocc, d.NORB, orbSym, targetSym)
		}
		if sp.MainBlockSize() == 0 {
			continue // no 1h configurations in this irrep (e.g. CVS: no core hole here)
		}
		items = append(items, sipItem{targetSym, sp})
	}

	// -tdm retains every sector's live operator and cross-solves them afterwards, so it
	// stays serial. Otherwise, with a multi-GPU pool, run one sector per GPU concurrently.
	if !cfg.tdm && len(ch.pool) >= 2 {
		sectors := make([]analyze.SIPSector, len(items))
		err := ch.runConcurrent(len(items), func(w *chooser, i int) error {
			it := items[i]
			res, mx, err := solveSIPSpace(w, fmt.Sprintf("sip irrep=%d", it.targetSym+1),
				it.sp, ints, eps, cfg.order, cfg, false)
			if err != nil {
				return err
			}
			sectors[i] = analyze.BuildSIPSector(it.sp, res, mx.FMatrix(), opts)
			mx.Release()
			return nil
		})
		if err != nil {
			return err
		}
		doc.Sectors = append(doc.Sectors, sectors...)
	} else {
		for _, it := range items {
			label := fmt.Sprintf("sip irrep=%d", it.targetSym+1)
			res, mx, err := solveSIPSpace(ch, label, it.sp, ints, eps, cfg.order, cfg, cfg.tdm)
			if err != nil {
				return err
			}
			sector := analyze.BuildSIPSector(it.sp, res, mx.FMatrix(), opts)
			doc.Sectors = append(doc.Sectors, sector)
			if cfg.tdm {
				solved = append(solved, solvedSIP{sp: it.sp, mx: mx, res: res, sector: sector})
			} else {
				mx.Release()
			}
		}
	}

	if cfg.tdm {
		td, err := buildSIPTDMDoc(ch, d, ints, orbSym, eps, solved, md, opts, cfg)
		for _, s := range solved {
			s.mx.Release()
		}
		if err != nil {
			return err
		}
		return emitJSON(td, cfg.out)
	}

	if cfg.spec.enabled {
		if cfg.spec.bare {
			return emitJSON(spectrum.BuildBareSIP(doc.Sectors, spectrum.BareOptions{}), cfg.out)
		}
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
