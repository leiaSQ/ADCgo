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
	"io"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
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
// wrapper knows to resume rather than treat it as a hard failure (see the daisychain
// wrappers under scripts/). exit 0 means the solve completed.
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
	order := flag.Int("order", 3, "SIP ADC order: 2, 3, 4 or 22 (2 = extended ADC(2); 4 = CVS Dyson ADC(4), needs -core; 22 = ADC(2,2), see -adc22). Detail: adcgo -h order")
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
	mgpu := flag.Int("mgpu", 0, "-dip -solver lanczos-lowmem -lowmem-block 0 only: row-partition ONE sector across this many GPUs (0 = off), so a whole-band Mode B Krylov block that dwarfs a single GPU fits across a node. Sectors run serially, each spanning the pool; needs a fast inter-GPU link (NVLink)")
	satChunk := flag.Int("satchunk", dip.SatChunkCols, "-mgpu matrix-free DIP only: column-chunk width of the per-device satellite gather. Each device stages a full-height n×w slab (n·w·8 bytes), and the applier fences twice per chunk — so raising this cuts both the ceil(b/w) element recompute (~15% at 64, ~8% at 128) and the barrier count, at proportionally more slab VRAM")
	satTrace := flag.Bool("sat-trace", false, "-mgpu matrix-free DIP only: print a per-column-chunk timing breakdown of the satellite apply to stderr (gather / fences / operator fill / batched GEMM). A whole-band production mat-vec is 40 h and lanczos reports it as one number only once the block ends; the apply is ceil(b/-satchunk) chunks, so tracing per chunk gives the same split in ~1/27th of the time")
	matfree := flag.String("matfree", "off", "matrix-free apply of large blocks — CVS-ADC(4) 2h1p×3h2p/2h1p² coupling and the order-3 SIP 2h1p×2h1p satellite (recompute vs store): off | auto | on. Trades resident memory for per-mat-vec recompute; auto switches per block using -maxmem. Required for large SIP-ADC(3) sectors whose dense satellite block is TB-scale (e.g. the production system)")
	maxMemGB := flag.Float64("maxmem", 4.0, "matrix-free -matfree=auto threshold: a coupling block whose dense size exceeds this many GB is applied matrix-free")
	wert3 := flag.Bool("wert3", true, "include the WERT3 5th-order 3h2p-diagonal correction in CVS-ADC(4) (the full EIGAB effective diagonal theADCcode itself uses; bit-exact vs its FT19 tape). -wert3=false for the bare 0th-order 3h2p diagonal.")
	sigma := flag.String("sigma", "auto", "static self-energy added to the SIP main block: auto | off | three | four | fplus | infinite. The ADC matrix code does not build Σ (theADCcode keeps it in a separate &self-energy module and subtracts it); omitting it shifts every main line by ~0.2-0.35 eV. auto = infinite, the all-order resolvent resummation, bit-exact vs theADCcode.")
	sigmaAkrit := flag.Float64("sigma-akrit", 0, "Σ(∞) resolvent convergence threshold on Σ(Δx)² (0 = converge tightly; theADCcode's own default is 1e-9)")
	sigmaMaxIt := flag.Int("sigma-maxit", 0, "Σ(∞) resolvent iteration cap (0 = 200; theADCcode's own default is 30)")
	mgpuDevSymEig := flag.Bool("mgpu-device-symeig", false, "run the -mgpu Rayleigh-Ritz eigensolve on a GPU instead of the host. distBackend embeds the host backend and inherits its SymEig, so by default the O(dim^3) projected eigensolve runs on one CPU while all 8 GPUs idle (dim reaches 11,600 for production SIP). OFF by default because cuSOLVER's dsyevd and the host LAPACK path agree only to rounding, not bit for bit: turning this on moves every line in the last digits, so re-validate against the reference spectra before trusting a run that used it")
	mainCache := flag.String("mainblock-cache", "auto", "where to cache the assembled SIP 1h/1h main block so a later run skips rebuilding it: auto = <fcidump>.mainblock.o<order>.i<irrep>.cache | off | an explicit path prefix. The block is 26 KB at production scale but took 8h16m to build (job 14551670) because every element is an O(nvir^4*nocc) sum, and -checkpoint covers only the Krylov state, so each daisychain generation rebuilt it. The cached copy is rejected unless the ADC order, sector irrep and multiplicity, orbital-space dimensions, WERT3 flag, a hash of the orbital energies, a hash of the static self-energy, and the FCIDUMP size/mtime all match")
	sigmaCache := flag.String("sigma-cache", "auto", "where to cache the static self-energy so a later run skips rebuilding it: auto = <fcidump>.sigma-<scheme>.cache | off | an explicit path. Σ(∞) dominates a large SIP run (78 h for the production system) and is only n² floats (351 KB at norb=212), so a daisychain that is walltime-killed before its solver checkpoints would otherwise pay those hours again every generation. The cached copy is rejected unless the scheme, its tuning, the orbital-space dimensions, a hash of the orbital energies, and the FCIDUMP size/mtime all match")
	out := flag.String("out", "", "write the output to this file (default stdout)")
	format := flag.String("format", "json", "output format for the -dip/-sip solver document: json = ADCgo's native document | ref = theADCcode's own \"Eigenvalue (eV), ps (%), residue\" state list, byte-compatible with adcdip*.out so a run can be diffed straight against the reference implementation (main-space overlaps only, and the residue column in a.u. as the reference prints it). ref covers the solver document alone — -spectrum, -tdm and -convert have no reference format and reject it")
	profile := flag.Bool("profile", false, "print per-sector solver phase timings to stderr")
	checkpoint := flag.String("checkpoint", "", "base path for Krylov checkpoints, so a solve can resume in a later process after a walltime kill or a crash. Supported by -solver lanczos (SIP and DIP) and by -solver lanczos-lowmem with -lowmem-block 0 (DIP Mode B only — Mode A retains the whole basis on the host and is not resumable). Each sector appends a suffix: SIP .i<irrep>, DIP .s<spin>.i<irrep>. A SIGUSR1 (SLURM --signal=B:USR1@<grace>) makes the run checkpoint and exit 64 (\"resume needed\"); exit 0 means done. Empty = no checkpointing")
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
	molecule := flag.String("molecule", "", "molecule label recorded in the spectrum meta (used in the plot title)")
	basisLabel := flag.String("basis", "", "basis-set label recorded in the spectrum meta")
	pointGroup := flag.String("point-group", "", "point-group label recorded in the spectrum meta, e.g. C2v")
	minWeight := flag.Float64("min-weight", 0, "drop decay channels with weight <= this")
	minFraction := flag.Float64("min-fraction", 0, "drop decay channels below this fraction of a state's 2h population (0..1)")
	includeZero := flag.Bool("include-zero", false, "emit the full canonical channel set per state, even at zero weight")
	var groups groupFlag
	flag.Var(&groups, "group", "decay-site grouping NAME=col1,col2 (repeatable; ~col makes a column passive); a bare -group prompts interactively; default each population column is its own site")
	convert := flag.String("convert", "", "read a previously emitted solver document JSON (the default -dip/-sip output) and emit its bare stick spectrum without re-solving; needs -dip or -sip to say which kind")

	adc22 := flag.String("adc22", "f", "-order 22 variant (Kolorenc & Averbukh, JCP 152, 214107 (2020), Table I): f = full, the paper's recommendation and the only variant that gets double Auger right; x = drops the second-order 1h/2h1p coupling; m = also drops the first-order 3h2p/3h2p block, leaving it diagonal. m and x are documented to overshoot decay widths by ~14% and ~24%, so they are diagnostics rather than production settings")
	doFano := flag.Bool("fano", false, "compute an electronic decay width (Auger, ICD, ETMD and their double counterparts) by the Fano/Feshbach method with Stieltjes imaging, instead of a spectrum. Needs -sip, -fano-init, and an -order of 2 (Fano-ADC(2)x), 3, or 22 (Fano-ADC(2,2)). The vacancy fixes the target irrep, so -sym is determined rather than chosen")
	fanoInit := flag.Int("fano-init", -1, "-fano: the initially ionized orbital, a 0-based occupied index. It defines both the discrete state |Phi> (selected from the QMQ spectrum by its weight on this orbital's 1h configuration) and, by default, the Q subspace")
	fanoQ := flag.String("fano-q", "", "-fano: the Q (bound) orbital set as comma-separated 0-based occupied indices. Empty = just -fano-init, which with the default -fano-rule any is the Auger criterion: Q is every configuration still carrying the initial hole, P every one that has filled it. For interatomic decay name the whole donor subunit's orbitals and use -fano-rule all")
	fanoRule := flag.String("fano-rule", "any", "-fano: which reading of the scheme A predicate puts a configuration in Q. any = at least one hole in the Q set (retains the initial vacancy), right for local decay such as atomic Auger. all = every hole in the set (all holes localized on subunit A), right for ICD/ETMD between subunits, where it is a hole OUTSIDE the donor that marks a decay channel. The two are not interchangeable")
	fanoNth := flag.Int("fano-nth", 0, "-fano: which qualifying QMQ root to take as |Phi>, 0-based in ascending energy (the reference's ninista-1). Only roots whose weight on the vacancy configuration reaches -fano-qmin are counted")
	fanoQP := flag.String("fano-qp", "", "-fano: a Q/P partition stated PER EXCITATION CLASS, which -fano-q cannot express. Grammar: clauses separated by ';', each prefixed q: or p:, each a conjunction of terms joined by '&', each term [class/]orbitals:min[:max] with orbitals a comma-separated list of 0-based occupied indices and a-b ranges, class an excitation class as a hole count (omitted = every class), and max omitted = unbounded. A configuration is bound if some q clause matches, or if p clauses were given and none matches. Two of the four atoms in the paper's Table V need this: Mg(2s^-1) is 'q:0:1;p:2/4:1;p:3/4:2' — 2s vacancies bound, continuum is 2h1p with a 3s hole and 3h2p with TWO of them, which is what keeps the CLOSED 2p^-2 channel out of P — and Kr(3d^-1) is 'q:0-4:1;q:3/5:2:2&3/6-8:1', a 3d any-hole rule plus the 4s^-2 4p^-1 shake-up family in Q. Supersedes -fano-q and -fano-rule")
	fanoQMin := flag.Float64("fano-qmin", 0.1, "-fano: minimum weight of |Phi> on the vacancy's 1h configuration (the reference's mspacewi). A root below this is not the state that was ionized")
	fanoQSolver := flag.String("fano-qsolver", "", "-fano: eigensolver for the QMQ (bound) half, which wants a different one from the PMP half that -solver governs. Empty = davidson, or dense when -solver is dense. Only a few of QMQ's LOWEST roots are wanted — |Phi> is the bottom of that spectrum, since every Q configuration past the 1h class carries an extra hole — and under this partition Q's main block is often a single configuration, so a block-Lanczos seeded from it would be one column wide")
	fanoQRoots := flag.Int("fano-qroots", 8, "-fano -solver davidson: QMQ roots to converge. The lowest are the right ones: every Q configuration carries the initial vacancy and every one past the 1h class carries an extra hole, so |Phi> is the bottom of the QMQ spectrum even for a deep core hole")
	fanoEMax := flag.Float64("fano-emax", 0, "-fano: drop pseudo-continuum states above this energy in hartree. 0 (the default) = no ceiling. The reference hard-codes 4 hartree, and that constant does NOT transfer: it is a polarization-propagator scale, where the energies are neutral excitations of a few tenths of a hartree. An ionization potential is an order of magnitude larger and a core vacancy two (Ne 1s sits at 32 hartree), so 4 hartree would discard the entire decay continuum")
	fanoEMaxRel := flag.Float64("fano-emax-rel", 3.0, "-fano: energy ceiling as a MULTIPLE of E_Phi, used when -fano-emax is not given absolutely. This is the transferable form of the reference's hard-coded 4 hartree: the cut exists to drop states far ABOVE the decaying state, which contribute nothing to Gamma(E_Phi) but dominate the moments the Stieltjes reconstruction is built from, and that only means anything relative to E_Phi. Without it a basis with tight augmentation functions spans tens of keV and puts E_Phi in the bottom 1% of the sampled range, where imaging is extrapolating: for Ar 2p that alone moved the width from 108 to 82 meV against a published 114. Scanning it is the honest convergence check - Gamma plateaus over roughly 3-10x with the order spread minimized near 3x. 0 disables the ceiling")
	fanoWMin := flag.Float64("fano-wmin", 0.05, "-fano: drop pseudo-continuum states whose weight on P's decay class (2h1p for single ionization) is at or below this (the reference's cntr > 0.05; its ADC(2)-extended variant used 0.2). NOTE this is the 2h1p weight, not the 1h weight — the polarization-propagator reference tests its own lowest class, which is 1h1p. Negative disables the cut")
	fanoGMin := flag.Float64("fano-gmin", 1e-16, "-fano: drop pseudo-continuum states with gamma_i at or below this, in hartree squared (the reference's 1e-16 floor). Negative disables the cut")
	fanoAt := flag.Float64("fano-at", 0, "-fano: evaluate Gamma at this energy in eV instead of at E_Phi. 0 = at E_Phi, which is the physical answer. Pinning it is how two schemes are compared at FIXED energy: a scheme that moves E_Phi changes Gamma twice over, once through the coupling density it produces and once by evaluating that density somewhere else, and only separating the two attributes a difference to the physics")
	fanoBlock := flag.Int("fano-block", 0, "-fano -solver lanczos: Krylov block width for the PMP pseudo-continuum solve, seeded with the most strongly coupled P configurations (the reference's fill_stvc). 0 = min(64, |P|). P's own main block is a poor width here — under this partition it holds only the few 1h configurations Q did not claim — and the width is nearly free with -matfree, since one element recompute serves every column of the block")
	fanoBlocks := flag.Int("fano-blocks", 0, "-fano -solver lanczos: block count for the PMP solve (0 = -blocks). The pseudo-continuum needs only enough states to sample the coupling density; a few hundred is ample, and Stieltjes imaging reports its own convergence")
	stOrders := flag.String("stieltjes-orders", "", "-fano: Stieltjes order range as lo-hi (e.g. 5-40, or 8- for no upper bound). Empty = 5 to the largest order whose orthogonal polynomials survive, which is discovered rather than fixed")
	stPrec := flag.Uint("stieltjes-prec", 256, "-fano: mantissa bits for the Stieltjes moment recurrence. The maximum usable order is set purely by this: on a Lorentzian model, 53 bits (float64) reaches order 17 with a 21% density error, 113 bits (the reference's REAL*16) order 34 with 2.9%, and 256 bits order 72 with 0.37%")
	stAverage := flag.String("stieltjes-average", "paper", "-fano: how the per-order widths are combined. paper = the mean over -stieltjes-window consecutive orders in the region of best convergence, with that window's standard deviation as the error bar (Kolorenc & Averbukh's own protocol, and what their tabulated uncertainties are). reference = stieltjes_phi1.f's mean of the three highest orders with its relaxing convergence search")
	stWindow := flag.Int("stieltjes-window", 9, "-fano -stieltjes-average paper: consecutive orders per averaging window (the paper uses nine)")

	// Tiered help (help.go): `adcgo -h` prints the grouped overview, `adcgo -h <topic>`
	// one topic page, `adcgo -h all` the old unabridged dump. Intercepted before
	// flag.Parse because the flag package rejects the topic word as a stray positional
	// and its own -h prints exactly the dump this replaces. The flags above are already
	// registered on flag.CommandLine, so a topic page renders their real usage text.
	if topic, ok := helpRequested(os.Args[1:]); ok {
		if !printHelp(os.Stdout, topic) {
			os.Exit(2)
		}
		return
	}
	if len(os.Args) == 1 {
		printHelp(os.Stdout, "")
		return
	}
	flag.Usage = func() { printHelp(os.Stderr, "") }

	applyCgroupMemLimit() // bound RSS under the SLURM --mem cap (see memlimit.go)

	flag.Parse()

	println("\n ADCgo: a modern implementation of ADC \n Authors: Leia Wertebach, Alexander Kuleff \n\n Derived from: \n TheADCcode: A collection of ADC/ISR source codes.\n Contributors: Nikolay Golubev,\n Yasen Velkov (developer of the original version),\n Alexander Kuleff,\n Anthony Dutoi, Nicolas Sisourat, Tsveta Miteva,\n Joerg Breidbach, Imke Mueller, Nayana Vaval,\n Francesco Tarantelli, Soeren Kopelke,\n Sajeev Yesodharan, Kirill Gokhberg, Robin Santra\n\n")

	// Applied before any Matrix is built: the per-device satellite appliers latch the chunk
	// width when they are CONSTRUCTED, because it sizes their slab allocation.
	if *satChunk < 1 {
		fmt.Fprintf(os.Stderr, "adcgo: -satchunk must be >= 1 (got %d)\n", *satChunk)
		os.Exit(2)
	}
	dip.SatChunkCols = *satChunk

	// Same "before any Matrix is built" reason: the chooser hands out backends immediately after
	// this, and distBackend reads the switch on every SymEig call thereafter.
	backend.DistDeviceSymEig = *mgpuDevSymEig
	if *mgpuDevSymEig {
		fmt.Fprintf(os.Stderr, "adcgo: -mgpu-device-symeig: the projected eigensolve runs on a GPU; "+
			"cuSOLVER and host LAPACK agree only to rounding, so this run's lines will differ from "+
			"a host solve in the last digits\n")
	}
	dip.SatTrace = *satTrace

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
				molecule: *molecule, basis: *basisLabel, pointGroup: *pointGroup,
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
		fmt.Fprintln(os.Stderr, "adcgo: -fcidump is required (try `adcgo -h`)")
		fmt.Fprintln(os.Stderr, "usage: adcgo -fcidump <file> [-dip | -sip] [...]")
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
		molecule:    *molecule,
		basis:       *basisLabel,
		pointGroup:  *pointGroup,
		groups:      groups.sites,
		interactive: groups.interactive,
		classify: spectrum.Options{
			MinWeight: *minWeight, MinFraction: *minFraction, IncludeZero: *includeZero,
		},
	}

	if *doDIP {
		cfg := dipConfig{
			solver: *solver, spinSel: *spinSel, moPath: *moPath, out: *out, sym: *sym,
			backend: *backendName, gpus: *gpus, mgpu: *mgpu,
			psThresh: *psThresh, coeffThresh: *coeffThresh, blocks: *blocks, format: *format,
			nroots: *nroots, maxdavsp: *maxdavsp, maxdavit: *maxdavit, convthr: *convthr,
			lowmemBlock: *lowmemBlock,
			profile:     *profile,
			spec:        specCfg,
		}
		mfMode, err := parseMatFree(*matfree)
		if err != nil {
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(2)
		}
		cfg.matFree = mfMode
		cfg.matFreeBudget = int64(*maxMemGB * (1 << 30))
		cfg.ckpt = *checkpoint
		cfg.ckptEvery = *checkpointEvery
		if cfg.ckpt != "" {
			cfg.stop = stopSig
		}
		if err := runDIP(d, cfg); err != nil {
			if errors.Is(err, errInterrupted) {
				fmt.Fprintln(os.Stderr, "adcgo: checkpoint written; resume needed")
				os.Exit(exitResumeNeeded)
			}
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(1)
		}
		return
	}

	if *doFano {
		if !*doSIP {
			fmt.Fprintln(os.Stderr, "adcgo: -fano needs -sip")
			os.Exit(2)
		}
		if doSpec || *doTDM {
			fmt.Fprintln(os.Stderr, "adcgo: -fano is exclusive with -spectrum/-bare and -tdm")
			os.Exit(2)
		}
		if *fanoInit < 0 {
			fmt.Fprintln(os.Stderr, "adcgo: -fano needs -fano-init <0-based occupied orbital>")
			os.Exit(2)
		}
	}

	if *doSIP {
		core, err := parseCoreOrbitals(*coreOrb)
		if err != nil {
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(1)
		}
		cfg := sipConfig{
			solver: *solver, out: *out, sym: *sym, backend: *backendName, gpus: *gpus, order: *order,
			psThresh: *psThresh, coeffThresh: *coeffThresh, blocks: *blocks, format: *format,
			nroots: *nroots, maxdavsp: *maxdavsp, maxdavit: *maxdavit, convthr: *convthr,
			profile: *profile,
			spec:    specCfg,
			core:    core,
			moPath:  *moPath, tdm: *doTDM, tdmOsc: *tdmOsc, tdmISR: *tdmISR,
			sigmaCache: *sigmaCache, mainCache: *mainCache, fcidumpPath: *path,
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
		variant, ok := sip.ParseVariant(*adc22)
		if !ok {
			fmt.Fprintf(os.Stderr, "adcgo: bad -adc22 %q (want m, x or f)\n", *adc22)
			os.Exit(2)
		}
		cfg.variant = variant

		if *doFano {
			rule, err := parseHoleRule(*fanoRule)
			if err != nil {
				fmt.Fprintln(os.Stderr, "adcgo:", err)
				os.Exit(2)
			}
			qOrbs, err := parseOrbitalList("-fano-q", *fanoQ)
			if err != nil {
				fmt.Fprintln(os.Stderr, "adcgo:", err)
				os.Exit(2)
			}
			lo, hi, err := parseStieltjesOrders(*stOrders)
			if err != nil {
				fmt.Fprintln(os.Stderr, "adcgo:", err)
				os.Exit(2)
			}
			avg, err := parseAverageMode(*stAverage)
			if err != nil {
				fmt.Fprintln(os.Stderr, "adcgo:", err)
				os.Exit(2)
			}
			fcfg := fanoConfig{
				sip: cfg, variant: variant, vacancy: *fanoInit, qOrbs: qOrbs, rule: rule,
				nth: *fanoNth, qMin: *fanoQMin, qRoots: *fanoQRoots, qSolver: *fanoQSolver,
				qpSpec: *fanoQP,
				emax:   *fanoEMax, emaxRel: *fanoEMaxRel, wmin: *fanoWMin, gmin: *fanoGMin,
				pBlock: *fanoBlock, pBlocks: *fanoBlocks, atEV: *fanoAt,
				stOrderLo: lo, stOrderHi: hi, stPrec: *stPrec, stAverage: avg, stWindow: *stWindow,
				initSite: *initAtom, sites: groups.sites, specOpts: specCfg.classify,
			}
			if err := runFano(d, fcfg); err != nil {
				fmt.Fprintln(os.Stderr, "adcgo:", err)
				os.Exit(1)
			}
			return
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
	mgpu                                       int // -mgpu: row-partition ONE sector across this many GPUs (0 = off)
	psThresh, coeffThresh                      float64
	format                                     string // -format: json | ref
	blocks                                     int
	nroots, maxdavsp, maxdavit                 int     // -solver davidson
	convthr                                    float64 // -solver davidson
	lowmemBlock                                int     // -solver lanczos-lowmem block width (0 = main)
	profile                                    bool
	spec                                       specConfig
	matFree                                    sip.MatFreeMode // dense (default) vs matrix-free 3h1p↔3h1p satellite region
	matFreeBudget                              int64           // -matfree=auto per-block dense-size threshold (bytes)
	ckpt                                       string          // -checkpoint base path (lanczos / lanczos-lowmem Mode B; "" = off)
	ckptEvery                                  int             // -checkpoint-every: blocks between crash-resilience saves
	stop                                       *atomic.Bool    // set by a stop signal; polled by the checkpointing solver loop
}

// dipCkptPath keys a DIP checkpoint by BOTH spin and irrep.
//
// SIP keys by irrep alone (sip_tdm.go), which cannot work here: DIP sectors are spin × irrep, so
// the singlet and triplet of the same irrep would share one file — and in the non-mgpu path they
// run concurrently, so two goroutines would write it. Their sector sizes differ, so the guard
// would reject one of them at best and mis-resume at worst.
func dipCkptPath(base string, spin dip.Spin, sym int) string {
	return fmt.Sprintf("%s.s%d.i%d", base, int(spin), sym)
}

// dipLanczosOpts builds one DIP sector's solver options, including its checkpoint and progress
// reporting. Both solve paths route through it so the two cannot drift — solveDIPSector and
// solveDIPSectorMGPU previously each constructed Options independently.
func dipLanczosOpts(cfg dipConfig, spin dip.Spin, targetSym int) lanczos.Options {
	o := lanczos.Options{MaxBlocks: cfg.blocks, LowMemBlock: cfg.lowmemBlock}
	if cfg.ckpt != "" && (cfg.solver == "lanczos" || cfg.solver == "lanczos-lowmem") {
		o.Checkpoint = &lanczos.Checkpoint{
			Path:  dipCkptPath(cfg.ckpt, spin, targetSym),
			Every: cfg.ckptEvery,
			Stop:  cfg.stop,
		}
	}
	// A Mode B block at production scale costs hours and the driver is otherwise silent:
	// job 14040960 ran 1 d 15 h and emitted nothing, and only its panic frame (lowmem.go's
	// pre-gate ApplyBlock arm) revealed it had not finished two blocks. Without this there is no
	// way to know whether a checkpoint interval is ever reached.
	if cfg.profile || cfg.ckpt != "" {
		o.Progress = progressReporter(fmt.Sprintf("dip spin=%d irrep=%d", spin, targetSym+1))
	}
	return o
}

// progressReporter builds the per-block Progress callback both solve families install. It
// prints one line per block to stderr: the cumulative phase times first — the DIP line's
// original fields, in their original order, so scripts/uracil2W_dip_measure.sbatch's
// `grep -c "^progress dip"` and anything else reading them still match — then this block's
// own deltas.
//
// The deltas are the half that diagnoses a production run. Cumulative times answer "where has
// the solve spent itself"; only the per-block difference shows a block getting slower, which is
// what distinguishes a solve that is merely long from one whose apply cost is growing. Progress
// is handed lanczos.Timing, which is cumulative, so the previous value is kept here.
//
// The cumulative fields keep the original second resolution; the deltas are milliseconds,
// because a per-block difference rounded to the second reads as a column of "0s" on every
// sector small enough to debug on.
func progressReporter(prefix string) func(iter, dim, blockSize int, tm lanczos.Timing) {
	var prev lanczos.Timing
	return func(iter, dim, blockSize int, tm lanczos.Timing) {
		fmt.Fprintf(os.Stderr,
			"progress %s block=%d dim=%d size=%d apply=%s orth=%s | block apply=%s orth=%s proj=%s\n",
			prefix, iter, dim, blockSize,
			tm.Apply.Round(time.Second), tm.Orth.Round(time.Second),
			(tm.Apply - prev.Apply).Round(time.Millisecond),
			(tm.Orth - prev.Orth).Round(time.Millisecond),
			(tm.Proj - prev.Proj).Round(time.Millisecond))
		prev = tm
	}
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
func solveDIPSector(ch *chooser, cfg dipConfig, sp *dip.Space, ints *integrals.Store, eps []float64, spin dip.Spin, targetSym int, moData *mo.Data, opts analyze.Options) (analyze.Sector, error) {
	label := fmt.Sprintf("dip spin=%d irrep=%d", spin, targetSym+1)
	lopts := dipLanczosOpts(cfg, spin, targetSym)
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
				m.SetMatFree(cfg.matFree, cfg.matFreeBudget)
				defer m.Release()
				return timeApplyBlock(cand, n, probeB, m.ApplyBlock)
			})
	}
	mx := dip.New(sp, ints, eps, be)
	mx.SetMatFree(cfg.matFree, cfg.matFreeBudget)

	// Pre-flight device-memory guard for the short-recurrence path: SolveLowMem keeps three
	// n×block panels resident (4·n·b·8) and, on the first apply, uploads the whole block-
	// sparse operator — the dominant term (tens of GB for a large satellite space), whose
	// true size is only knowable here, not from the dense-n² cost model. If it will not fit
	// the chosen GPU, refuse cleanly rather than let a mid-assembly cudaMalloc panic tear
	// down the run. Only lanczos-lowmem has this simple, exact panel footprint.
	if cfg.solver == "lanczos-lowmem" {
		need := 4*uint64(n)*uint64(probeB)*8 + mx.OperatorResidentBytes()
		if err := ch.checkDeviceFit(label, be, need); err != nil {
			mx.Release()
			return analyze.Sector{}, err
		}
	}

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
	if res.Interrupted {
		// A stop signal checkpointed and bailed out mid-build; propagate so main exits with the
		// "resume needed" code instead of analyzing an unpopulated Result into an empty sector
		// that would then be emitted as if it had converged.
		return analyze.Sector{}, errInterrupted
	}
	return analyzeDIPSector(label, cfg, sp, res, moData, opts), nil
}

// analyzeDIPSector turns a solved sector's Result into an analyze.Sector (pole-strength
// filtering, atom-resolved populations), the shared tail of the single-backend and
// multi-GPU solve paths.
func analyzeDIPSector(label string, cfg dipConfig, sp *dip.Space, res lanczos.Result, moData *mo.Data, opts analyze.Options) analyze.Sector {
	if cfg.profile {
		reportTiming(label, sp.Size(), sp.MainBlockSize(), res.Timing)
	}
	var pe *analyze.PopEngine
	if moData != nil {
		pe = analyze.NewPopEngine(sp, moData)
	}
	return analyze.BuildSector(sp, res, opts, pe)
}

// solveDIPSectorMGPU solves one Mode B (whole-band, lowmem-block 0) sector row-partitioned
// across the sub-backends in subs, so a Krylov block too large for one GPU fits when spread
// over the pool. It builds a distributed backend from the sector's group-aligned partition
// boundaries (dip.Space.PartitionBounds) and runs SolveLowMem on it. subs is consumed for
// this sector only; the caller owns the underlying device backends.
func solveDIPSectorMGPU(subs []backend.Backend, cfg dipConfig, sp *dip.Space, ints *integrals.Store, eps []float64, spin dip.Spin, targetSym int, moData *mo.Data, opts analyze.Options) (analyze.Sector, error) {
	label := fmt.Sprintf("dip spin=%d irrep=%d", spin, targetSym+1)
	n, main := sp.Size(), sp.MainBlockSize()
	bounds := sp.PartitionBounds(len(subs))

	// A sector too small to partition (n ≤ 2·main², or only one group band) runs on a single
	// sub-backend — the row-partition buys nothing there and would trip the shape invariant.
	var be backend.Backend = subs[0]
	npart := 1
	if len(bounds)-1 >= 2 && n > 2*main*main {
		npart = len(bounds) - 1
		d, err := backend.NewDistributed(subs[:npart], n, main, bounds)
		if err != nil {
			return analyze.Sector{}, fmt.Errorf("%s: %w", label, err)
		}
		be = d
	}
	mx := dip.New(sp, ints, eps, be)
	mx.SetMatFree(cfg.matFree, cfg.matFreeBudget)

	// Per-GPU footprint = the ~4 live n×main Krylov panels + the block-sparse operator, split
	// over the partitions. NOT lanczos.LowMemSectorBytes: its opFrac·n²·8 operator term is a
	// dense estimate (≈870 TB at the production system's n) — meaningless at this scale. OperatorResidentBytes
	// sums the real block sizes (and collapses to ~0 when the satellite region is matrix-free);
	// the operator is replicated across the partitions it touches, so /npart charges each GPU
	// its share of the panels and an approximate share of the operator.
	panels := 4 * uint64(n) * uint64(main) * 8 // float64 panels
	perGPU := (panels + mx.OperatorResidentBytes()) / uint64(npart)
	if cfg.profile {
		fmt.Fprintf(os.Stderr, "dispatch %-18s mgpu n=%d main=%d over %d partition(s), ~%.1f GB/GPU\n",
			label, n, main, npart, float64(perGPU)/(1<<30))
	}
	// Pre-flight guard: the -mgpu path uploads each partition's row-band of the operator plus
	// its panels on the first apply. If a sub-backend GPU is too small, refuse cleanly here
	// rather than let a mid-assembly cudaMalloc panic tear the run down (the single-GPU path's
	// checkDeviceFit had no multi-GPU equivalent until now).
	if err := checkSubsFit(label, subs[:npart], perGPU); err != nil {
		mx.Release()
		return analyze.Sector{}, err
	}
	res := lanczos.SolveLowMem(mx, be, dipLanczosOpts(cfg, spin, targetSym))
	mx.Release()
	if res.Interrupted {
		// A stop signal checkpointed and bailed out mid-build; propagate so main exits with the
		// "resume needed" code instead of analyzing an unpopulated Result into an empty sector.
		return analyze.Sector{}, errInterrupted
	}
	return analyzeDIPSector(label, cfg, sp, res, moData, opts), nil
}

// mgpuSubs returns the sub-backends the multi-GPU path row-partitions a sector across. It
// reuses the chooser's already-built device pool (-backend cuda|hip), capped at cfg.mgpu;
// otherwise it builds cfg.mgpu independent instances of the chosen backend (e.g. gonum, the
// host validation / CPU path, where an instance is a stateless value and cloning is exactly
// what simulating partitions means).
//
// The pool is the authoritative device list, so cfg.mgpu is capped by it rather than taken
// on trust: cloning a GPU backend does not reach a second card, it binds device 0 again
// (backend.New -> ctor(0)), which would put every partition's resident operator on one card
// with none of the parallelism. Cloning is therefore refused outright for a device backend.
func mgpuSubs(ch *chooser, cfg dipConfig) ([]backend.Backend, error) {
	if len(ch.pool) >= 1 {
		g := min(cfg.mgpu, len(ch.pool))
		if g < cfg.mgpu {
			fmt.Fprintf(os.Stderr, "mgpu: -mgpu %d requested but %d device(s) visible; "+
				"partitioning across %d\n", cfg.mgpu, len(ch.pool), g)
		}
		return ch.pool[:g], nil
	}
	if backend.MultiDevice(cfg.backend) {
		return nil, fmt.Errorf("-mgpu %d: backend %q binds physical devices but none are pooled; "+
			"pass -backend %s explicitly (not -backend auto) so every visible device is bound",
			cfg.mgpu, cfg.backend, cfg.backend)
	}
	subs := make([]backend.Backend, cfg.mgpu)
	for i := range subs {
		be, err := backend.New(cfg.backend)
		if err != nil {
			return nil, err
		}
		subs[i] = be
	}
	return subs, nil
}

func runDIP(d *fcidump.Data, cfg dipConfig) error {
	spins, err := selectSpins(cfg.spinSel)
	if err != nil {
		return err
	}
	if err := validateFormat(cfg.format, !cfg.spec.enabled, "a stick spectrum"); err != nil {
		return err
	}
	if err := validateSolver(cfg.solver); err != nil {
		return err
	}
	// Reject an unresumable checkpoint request up front, with a message, rather than let it reach
	// SolveLowMem's panic backstop days into a run. Mode A keeps the whole basis on the host, which
	// is the object the short-recurrence checkpoint format exists to avoid writing.
	if cfg.ckpt != "" && cfg.solver == "lanczos-lowmem" && cfg.lowmemBlock != 0 {
		return fmt.Errorf("-checkpoint with -solver lanczos-lowmem requires -lowmem-block 0 (Mode B); "+
			"Mode A retains the full basis on the host and is not resumable (got -lowmem-block %d)",
			cfg.lowmemBlock)
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

	if cfg.mgpu > 0 {
		// Multi-GPU Mode B: row-partition ONE sector across the pool, sectors serial. This
		// is the whole-band path for sectors whose Krylov block dwarfs a single GPU (the production system).
		if cfg.solver != "lanczos-lowmem" {
			return fmt.Errorf("-mgpu requires -solver lanczos-lowmem (got %q)", cfg.solver)
		}
		subs, err := mgpuSubs(ch, cfg)
		if err != nil {
			return err
		}
		for i, it := range items {
			sec, err := solveDIPSectorMGPU(subs, cfg, it.sp, ints, eps, it.spin, it.targetSym, moData, opts)
			if err != nil {
				return err
			}
			results[i] = sec
		}
	} else {
		solve := func(w *chooser, i int) error {
			it := items[i]
			sec, err := solveDIPSector(w, cfg, it.sp, ints, eps, it.spin, it.targetSym, moData, opts)
			if err != nil {
				return err
			}
			results[i] = sec
			return nil
		}
		if len(ch.pool) >= 2 {
			if err := ch.runConcurrent(len(items), solve); err != nil {
				return err
			}
		} else {
			for i := range items {
				if err := solve(ch, i); err != nil {
					return err
				}
			}
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

	if cfg.format == formatRef {
		secs := make([]refSector, len(doc.Sectors))
		for i := range doc.Sectors {
			secs[i] = doc.Sectors[i]
		}
		return emitRef(secs, cfg.out)
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
// -format values. json is ADCgo's native document; ref is theADCcode's own state list.
const (
	formatJSON = "json"
	formatRef  = "ref"
)

// refSector is the shared behaviour of analyze.Sector and analyze.SIPSector: writing itself
// as one of theADCcode's "Eigenvalue (eV), ps (%), residue" blocks.
type refSector interface{ WriteRef(io.Writer) error }

// emitRef writes each sector as one reference block, concatenated in solve order. The
// reference puts one symmetry per file; a multi-symmetry ADCgo run writes them in sequence,
// which is what concatenating those files would give.
func emitRef(secs []refSector, out string) error {
	write := func(w io.Writer) error {
		for _, s := range secs {
			if err := s.WriteRef(w); err != nil {
				return err
			}
		}
		return nil
	}
	if out == "" {
		return write(os.Stdout)
	}
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	// Report the write error in preference to the close error, but never skip the close.
	werr := write(f)
	cerr := f.Close()
	if werr != nil {
		return werr
	}
	return cerr
}

// validateFormat rejects an unknown -format, and rejects "ref" for the outputs that have no
// reference counterpart. It is called BEFORE the solve, not at the emit site: a production
// run is hours long and discovering a bad output flag at the end of it would throw the whole
// thing away. refOK says whether this invocation ends in a solver document.
func validateFormat(format string, refOK bool, what string) error {
	switch format {
	case formatJSON:
		return nil
	case formatRef:
		if !refOK {
			return fmt.Errorf("-format ref is defined only for the -dip/-sip solver document, "+
				"and this run emits %s; drop -format or drop %s", what, what)
		}
		return nil
	default:
		return fmt.Errorf("unknown -format %q (want %q or %q)", format, formatJSON, formatRef)
	}
}

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
	variant                    sip.Variant // -adc22 m|x|f; only used when order == sip.Order22
	psThresh, coeffThresh      float64
	format                     string // -format: json | ref
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
	sigmaCache    string                 // -sigma-cache: auto (beside the FCIDUMP) | off | explicit path
	mainCache     string                 // -mainblock-cache: auto (beside the FCIDUMP) | off | explicit path
	fcidumpPath   string                 // the -fcidump argument, for the Σ/main-block cache keys and paths
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
	if (cfg.order < 2 || cfg.order > 4) && cfg.order != sip.Order22 {
		return fmt.Errorf("unknown -order %d (want 2, 3, 4, or %d for ADC(2,2))", cfg.order, sip.Order22)
	}
	if cfg.order == 4 && len(cfg.core) == 0 {
		return fmt.Errorf("-order 4 is CVS Dyson ADC(4) and requires -core (e.g. -core 0)")
	}
	refOK := !cfg.spec.enabled && !cfg.tdm
	what := "a stick spectrum"
	if cfg.tdm {
		what = "transition dipole moments"
	}
	if err := validateFormat(cfg.format, refOK, what); err != nil {
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
		switch cfg.order {
		case 4:
			sp = sip.NewSpace4(nocc, d.NORB, orbSym, targetSym, cfg.core)
		case sip.Order22:
			sp = sip.NewSpace22(nocc, d.NORB, orbSym, targetSym)
		default:
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
		spec, err := spectrum.BuildSIP(doc.Sectors, d.OrbSym, spectrum.SIPOptions{
			Molecule:   cfg.spec.molecule,
			Basis:      cfg.spec.basis,
			PointGroup: cfg.spec.pointGroup,
		})
		if err != nil {
			return err
		}
		return emitJSON(spec, cfg.out)
	}

	if cfg.format == formatRef {
		secs := make([]refSector, len(doc.Sectors))
		for i := range doc.Sectors {
			secs[i] = doc.Sectors[i]
		}
		return emitRef(secs, cfg.out)
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
