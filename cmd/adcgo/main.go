// Command adcgo is the ADCgo CLI.
//
// Without -dip it ingests an FCIDUMP and reports the reference energy and the
// RHF-MP2 correlation energy (the M0 integral-ingestion check). With -dip it
// solves the DIP-ADC(2) double-ionization problem and writes the dication states
// (energies, pole strengths, leading two-hole configurations) as JSON.
package main

import (
	"bufio"
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

// version is the release this binary was built from.
const version = "0.1.0"

// signOff closes the log of a solve that ran to completion. It goes to stderr, never
// stdout, because stdout carries the JSON/ref document when -out is not given.
func signOff() {
	fmt.Fprintln(os.Stderr, "adcgo: Without calculations you are lost! - Lorenz Cederbaum")
}

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
	showVersion := flag.Bool("version", false, "print the ADCgo version and exit")
	doDIP := flag.Bool("dip", false, "solve DIP-ADC(2) and emit dication states as JSON")
	doSIP := flag.Bool("sip", false, "solve IP-ADC(n) (non-Dyson) and emit cation states as JSON")
	order := flag.Int("order", 3, "ADC order: 3 for -sip (non-Dyson IP-ADC(3)); -dip is always ADC(2) and ignores it. Detail: adcgo -h order")
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
	backendName := flag.String("backend", "gonum", "linear-algebra backend: gonum | hip | cuda | auto (auto calibrates and picks per sector; build-tag gated)")
	gpus := flag.Int("gpus", 0, "-backend cuda|hip only: max GPUs for concurrent per-sector solves (0 = all visible). Independent sectors (DIP spin×irrep, SIP irrep) run one per GPU")
	eigWorkers := flag.Int("eig-workers", 0, "-dip -solver lanczos-lowmem -lowmem-block 0: workers for the projected banded eigensolve's deferred rotations (0 = the measured default, which is serial). The parallel path is tested bit-exact but measured SLOWER at the production half-bandwidth — 3.32x serial against 2.35x at 16 workers, band 3079 — so it is off unless asked for. What is still unmeasured is the band-matrix side at production chase length, and this is the knob for settling that on a real run")
	eigB2 := flag.Int("eig-b2", 0, "-dip -solver lanczos-lowmem -lowmem-block 0: narrow the projected matrix's half-bandwidth to at most N before the band->tridiagonal reduction. That reduction's band work is proportional to the bandwidth while its eigenvector work is independent of it, so narrowing first is the one structural saving available — at production scale the band half is ~63 h of a ~100 h stage. The reduction exploits the block-tridiagonal structure and reaches about the Krylov block width (a factor of two) in seconds of GEMM; a smaller N is reported but not delivered, since going below the block width needs successive band reduction. 0 = off")
	eigDevice := flag.Bool("eig-device", false, "-dip -solver lanczos-lowmem -lowmem-block 0 on a CUDA/HIP backend: replay the projected banded eigensolve's eigenvector rotations on the GPU. That half of the stage is bandwidth-bound streaming over a multi-gigabyte accumulator, which is what HBM is for; the band reduction driving it stays on the CPU because it synchronizes far too often for kernel launches. Off by default until the device path is confirmed bit-identical on the hardware in question")
	mgpu := flag.Int("mgpu", 0, "-dip -solver lanczos-lowmem -lowmem-block 0: row-partition ONE sector across this many GPUs (0 = off), so a whole-band Mode B Krylov block that dwarfs a single GPU fits across a node. Sectors run serially, each spanning the pool; needs a fast inter-GPU link (NVLink)")
	satChunk := flag.Int("satchunk", dip.SatChunkCols, "-mgpu matrix-free DIP only: column-chunk width of the per-device satellite gather. Each device stages a full-height n×w slab (n·w·8 bytes), and the applier fences twice per chunk — so raising this cuts both the ceil(b/w) element recompute (~15% at 64, ~8% at 128) and the barrier count, at proportionally more slab VRAM")
	satTrace := flag.Bool("sat-trace", false, "-mgpu matrix-free DIP only: print a per-column-chunk timing breakdown of the satellite apply to stderr (gather / fences / operator fill / batched GEMM). A whole-band production mat-vec is 40 h and lanczos reports it as one number only once the block ends; the apply is ceil(b/-satchunk) chunks, so tracing per chunk gives the same split in ~1/27th of the time")
	matfree := flag.String("matfree", "off", "matrix-free apply of large blocks — the DIP 3h1p×3h1p satellite region and the SIP-ADC(3) 2h1p×2h1p satellite (recompute vs store): off | auto | on. Trades resident memory for per-mat-vec recompute; auto switches per block using -maxmem. Required for large SIP-ADC(3) sectors whose dense satellite block is TB-scale (e.g. the production system)")
	maxMemGB := flag.Float64("maxmem", 4.0, "matrix-free -matfree=auto threshold: a coupling block whose dense size exceeds this many GB is applied matrix-free")
	sigma := flag.String("sigma", "auto", "static self-energy added to the SIP main block: auto | off | three | four | fplus | infinite. The ADC matrix code does not build Σ (theADCcode keeps it in a separate &self-energy module and subtracts it); omitting it shifts every main line by ~0.2-0.35 eV. auto = infinite, the all-order resolvent resummation, bit-exact vs theADCcode.")
	sigmaAkrit := flag.Float64("sigma-akrit", 0, "Σ(∞) resolvent convergence threshold on Σ(Δx)² (0 = converge tightly; theADCcode's own default is 1e-9)")
	sigmaMaxIt := flag.Int("sigma-maxit", 0, "Σ(∞) resolvent iteration cap (0 = 200; theADCcode's own default is 30)")
	mgpuDevSymEig := flag.Bool("mgpu-device-symeig", false, "run the -mgpu Rayleigh-Ritz eigensolve on a GPU instead of the host. distBackend embeds the host backend and inherits its SymEig, so by default the O(dim^3) projected eigensolve runs on one CPU while all 8 GPUs idle (dim reaches 11,600 for production SIP). OFF by default because cuSOLVER's dsyevd and the host LAPACK path agree only to rounding, not bit for bit: turning this on moves every line in the last digits, so re-validate against the reference spectra before trusting a run that used it")
	mainCache := flag.String("mainblock-cache", "auto", "where to cache the assembled SIP 1h/1h main block so a later run skips rebuilding it: auto = <fcidump>.mainblock.o<order>.i<irrep>.cache | off | an explicit path prefix. The block is 26 KB at production scale but took 8h16m to build (job 14551670) because every element is an O(nvir^4*nocc) sum, and -checkpoint covers only the Krylov state, so each daisychain generation rebuilt it. The cached copy is rejected unless the ADC order, sector irrep and multiplicity, orbital-space dimensions, a hash of the orbital energies, a hash of the static self-energy, and the FCIDUMP size/mtime all match")
	sigmaCache := flag.String("sigma-cache", "auto", "where to cache the static self-energy so a later run skips rebuilding it: auto = <fcidump>.sigma-<scheme>.cache | off | an explicit path. Σ(∞) dominates a large SIP run (78 h for the production system) and is only n² floats (351 KB at norb=212), so a daisychain that is walltime-killed before its solver checkpoints would otherwise pay those hours again every generation. The cached copy is rejected unless the scheme, its tuning, the orbital-space dimensions, a hash of the orbital energies, and the FCIDUMP size/mtime all match")
	out := flag.String("out", "", "write the output to this file (default stdout)")
	format := flag.String("format", "json", "output format for the -dip/-sip solver document: json = ADCgo's native document | ref = theADCcode's own \"Eigenvalue (eV), ps (%), residue\" state list, byte-compatible with adcdip*.out so a run can be diffed straight against the reference implementation (main-space overlaps only, and the residue column in a.u. as the reference prints it). ref covers the solver document alone — -spectrum, -tdm and -convert have no reference format and reject it")
	profile := flag.Bool("profile", false, "print per-sector solver phase timings to stderr")
	checkpoint := flag.String("checkpoint", "", "base path for Krylov checkpoints, so a solve can resume in a later process after a walltime kill or a crash. Supported by -solver lanczos (SIP and DIP) and by -solver lanczos-lowmem with -lowmem-block 0 (DIP Mode B only — Mode A retains the whole basis on the host and is not resumable). Each sector appends a suffix: SIP .i<irrep>, DIP .s<spin>.i<irrep>. A SIGUSR1 (SLURM --signal=B:USR1@<grace>) makes the run checkpoint and exit 64 (\"resume needed\"); exit 0 means done. Empty = no checkpointing. Mode B additionally checkpoints the projected banded eigensolve, in <path>.eig, every 30 minutes and on a stop signal: at production width that reduction is longer than a 120 h walltime, so without its own restart point the daisychain restored the Krylov state, ground most of a generation into the reduction and was killed, forever")
	checkpointEvery := flag.Int("checkpoint-every", 25, "-checkpoint only: also save every N blocks for crash resilience (<=0 = save only on the stop signal)")

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
	convert := flag.String("convert", "", "read a previously emitted solver document JSON (the default -dip/-sip output) and emit its bare stick spectrum without re-solving; needs -dip or -sip to say which kind. With -format ref it instead re-emits theADCcode's state listing and takes a COMMA-SEPARATED list, routing each sector by its own contents so SIP and DIP documents combine into one listing and -dip/-sip are not needed")

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

	flag.CommandLine.Parse(joinGroupArgs(os.Args[1:]))
	if flag.NArg() > 0 {
		// adcgo takes no positional arguments, and the flag package stops at the first
		// one: everything after it would otherwise be dropped without a word.
		fmt.Fprintf(os.Stderr, "adcgo: unexpected argument %q: every flag after it would be ignored "+
			"(a flag value with spaces needs quoting)\n", flag.Arg(0))
		os.Exit(2)
	}
	if *showVersion {
		fmt.Println("ADCgo", version)
		return
	}

	println("\n ADCgo v" + version + ": a modern implementation of ADC \n Authors: Leia Wertebach, Alexander Kuleff \n\n Derived from: \n TheADCcode: A collection of ADC/ISR source codes.\n Contributors: Nikolay Golubev,\n Yasen Velkov (developer of the original version),\n Alexander Kuleff,\n Anthony Dutoi, Nicolas Sisourat, Tsveta Miteva,\n Joerg Breidbach, Imke Mueller, Nayana Vaval,\n Francesco Tarantelli, Soeren Kopelke,\n Sajeev Yesodharan, Kirill Gokhberg, Robin Santra\n\n")

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
		// -format ref re-emits saved documents as theADCcode's state listing. It takes a
		// comma-separated list and routes each sector by its own contents, so SIP and DIP
		// come out in one listing and neither -dip nor -sip is needed.
		if *format == formatRef {
			if err := runRefConvert(splitPaths(*convert), *psThresh, *out); err != nil {
				fmt.Fprintln(os.Stderr, "adcgo:", err)
				os.Exit(1)
			}
			return
		}
		if *doDIP == *doSIP { // both set, or neither
			fmt.Fprintln(os.Stderr, "adcgo: -convert needs exactly one of -dip or -sip")
			os.Exit(2)
		}
		// A DIP solver document written with -mo already carries per-atom two-hole
		// populations, so -group/-spectrum can regroup it into a decay-channel spectrum
		// with no re-solve (the populations, not the eigenvectors, are what the classifier
		// needs). Without -mo, or with -bare, fall through to the bare per-state spectrum.
		if *doDIP && *moPath != "" && !*doBare && (*doSpectrum || len(groups.sites) > 0 || groups.interactive) {
			md, err := mo.ReadCanonical(*moPath)
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
			eigWorkers:  *eigWorkers,
			eigDevice:   *eigDevice,
			eigB2:       *eigB2,
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
		signOff()
		return
	}

	if *doSIP {
		cfg := sipConfig{
			solver: *solver, out: *out, sym: *sym, backend: *backendName, gpus: *gpus, order: *order,
			psThresh: *psThresh, coeffThresh: *coeffThresh, blocks: *blocks, format: *format,
			nroots: *nroots, maxdavsp: *maxdavsp, maxdavit: *maxdavit, convthr: *convthr,
			profile:    *profile,
			spec:       specCfg,
			sigmaCache: *sigmaCache, mainCache: *mainCache, fcidumpPath: *path,
		}
		mfMode, err := parseMatFree(*matfree)
		if err != nil {
			fmt.Fprintln(os.Stderr, "adcgo:", err)
			os.Exit(2)
		}
		cfg.matFree = mfMode
		cfg.matFreeBudget = int64(*maxMemGB * (1 << 30))
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
		signOff()
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

// writeSectorJSON persists one finished sector beside the final -out document.
//
// WHY THIS EXISTS. -out is written only after EVERY sector finishes, and a sector's checkpoints are
// deleted the moment it completes (removeLowMemCheckpoint, removeBandEigCheckpoint — both on the
// normal completion path, by design, so a rerun does not resume a finished computation). The
// combination means a failure in a LATER sector discards every earlier sector's result and leaves no
// restart point for it. Production job 15195647 lost a finished whole-band eigensolve (15h58m at
// dim 307998) plus ~43 h of apply exactly that way, to a GPU OOM in the next sector.
//
// The sidecar is a plain one-sector Document carrying the same .s<spin>.i<irrep> suffix the
// checkpoints use, written atomically. It is a safety net, so a failure to write it is reported but
// never fatal: losing the net must not also lose the run.
func writeSectorJSON(cfg dipConfig, norb, nelec int, sec analyze.Sector, spin dip.Spin, irrep int) {
	if cfg.out == "" {
		return // stdout: there is nothing to put a sidecar beside
	}
	path := fmt.Sprintf("%s.s%d.i%d.json", strings.TrimSuffix(cfg.out, ".json"), int(spin), irrep)
	b, err := json.MarshalIndent(Document{
		NORB: norb, NELEC: nelec, Solver: cfg.solver, Sectors: []analyze.Sector{sec},
	}, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "adcgo: cannot encode sector result for %s: %v\n", path, err)
		return
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "adcgo: cannot write %s: %v\n", tmp, err)
		return
	}
	if err := os.Rename(tmp, path); err != nil {
		fmt.Fprintf(os.Stderr, "adcgo: cannot rename %s -> %s: %v\n", tmp, path, err)
		os.Remove(tmp)
		return
	}
	fmt.Fprintf(os.Stderr, "sector result saved: %s (%d states)\n", path, len(sec.States))
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
	eigWorkers                                 int     // -eig-workers: banded eigensolve replay workers (0 = default)
	eigDevice                                  bool    // -eig-device: replay the banded eigensolve's rotations on the GPU
	eigB2                                      int     // -eig-b2: narrow the projected bandwidth before the reduction
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
	o := lanczos.Options{MaxBlocks: cfg.blocks, LowMemBlock: cfg.lowmemBlock,
		EigenWorkers: cfg.eigWorkers, EigenDevice: cfg.eigDevice, EigenB2: cfg.eigB2}
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
		prefix := fmt.Sprintf("dip spin=%d irrep=%d", spin, targetSym+1)
		o.Progress = progressReporter(prefix)
		o.EigenProgress = eigenProgressReporter(prefix)
	}
	return o
}

// progressReporter builds the per-block Progress callback both solve families install. It
// prints one line per block to stderr: the cumulative phase times first — the DIP line's
// original fields, in their original order, so a
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
// eigenProgressReporter builds the Mode B banded-eigensolve callback. It prints one line every
// lanczos.EigenProgress tick with the column reached, the percentage, and a projected finish from
// the rate so far — the reduction's cost is linear in the column index, so the extrapolation is
// meaningful rather than decorative.
//
// Without it the run goes silent between its last block line and its results. That is what made
// production DIP job 14834817 unreadable: it had in fact finished all 200 blocks and was grinding
// through the reduction, but nothing distinguished that from a hang.
func eigenProgressReporter(prefix string) func(col, cols int, elapsed time.Duration) {
	return func(col, cols int, elapsed time.Duration) {
		pct, eta := 0.0, time.Duration(0)
		if col > 0 && cols > 0 {
			pct = 100 * float64(col) / float64(cols)
			eta = time.Duration(float64(elapsed) * float64(cols-col) / float64(col))
		}
		fmt.Fprintf(os.Stderr, "progress %s eigensolve col=%d/%d (%.1f%%) elapsed=%s eta=%s\n",
			prefix, col, cols, pct, elapsed.Round(time.Second), eta.Round(time.Second))
	}
}

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
	// FOUR panels: pcBuf (n*2b, holding [prev|cur]) + workBuf + tmpBuf, all allocated up front in
	// SolveLowMem and held for the whole block loop.
	//
	// It was briefly five. blockOrth used to allocate one MORE n x main panel per partition while
	// orthogonalizing, and that transient was the real peak — two large triplet DIP runs both
	// died on exactly that allocation (20259985680 bytes = one partition's n*main*8). It no longer
	// allocates: SolveLowMem's two call sites now lend it whichever of their scratch panels is dead at
	// that point. Keep this at 4 only while that remains true.
	panels := 4 * uint64(n) * uint64(main) * 8 // float64 panels
	// The matrix-free ERI is added OUTSIDE the /npart: it is replicated on every device, not
	// row-partitioned like the panels and the assembled blocks. Leaving it out is what let a large
	// triplet sector pass this guard at 96.3 GiB and then die in cudaMalloc at ~126 GiB of a 140 GiB
	// card — the estimate was missing 18.10 GiB per GPU. See dip.Matrix.MatFreeDeviceBytes.
	perGPU := (panels+mx.OperatorResidentBytes())/uint64(npart) + mx.MatFreeDeviceBytes()
	if cfg.profile {
		fmt.Fprintf(os.Stderr, "dispatch %-18s mgpu n=%d main=%d over %d partition(s), ~%.1f GB/GPU "+
			"(panels %.1f + matfree %.1f)\n",
			label, n, main, npart, float64(perGPU)/(1<<30),
			float64((panels+mx.OperatorResidentBytes())/uint64(npart))/(1<<30),
			float64(mx.MatFreeDeviceBytes())/(1<<30))
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
		if moData, err = mo.ReadCanonical(cfg.moPath); err != nil {
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
			// Persist before starting the next sector: this one's checkpoints are already gone.
			writeSectorJSON(cfg, d.NORB, d.NELEC, sec, it.spin, it.targetSym+1)
		}
	} else {
		solve := func(w *chooser, i int) error {
			it := items[i]
			sec, err := solveDIPSector(w, cfg, it.sp, ints, eps, it.spin, it.targetSym, moData, opts)
			if err != nil {
				return err
			}
			results[i] = sec
			// Distinct path per sector, so this is safe from the concurrent dispatch too.
			writeSectorJSON(cfg, d.NORB, d.NELEC, sec, it.spin, it.targetSym+1)
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

// refDoc is the subset of a solver document that theADCcode's reference listing needs. It
// deliberately omits `pop`, the atom-resolved populations: those are the bulk of a production
// DIP document (1.45 GB at production scale) and play no part in the listing, so decoding them would
// cost gigabytes for nothing. Both component lists are present because one document is SIP and
// another DIP, and a sector is routed by whichever it carries.
type refDoc struct {
	Sectors []struct {
		Irrep  int `json:"irrep"`
		Spin   int `json:"spin"`
		States []struct {
			Index     int                 `json:"index"`
			EnergyEV  float64             `json:"energy_ev"`
			PSPercent float64             `json:"ps_percent"`
			Residue   float64             `json:"residue"`
			Root      int                 `json:"root"`
			Leading   []analyze.Leading   `json:"leading"` // DIP: two-hole configurations
			Main      []analyze.OrbWeight `json:"main"`    // SIP: one-hole overlaps
		} `json:"states"`
	} `json:"sectors"`
}

// splitPaths splits a comma-separated path list, dropping empty entries.
func splitPaths(spec string) []string {
	var out []string
	for _, p := range strings.Split(spec, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// runRefConvert re-emits saved solver documents in theADCcode's "Eigenvalue (eV), ps (%),
// residue" format without re-solving, concatenating them in the order given — which is how the
// reference's own per-symmetry files would read if catted together, and what emitRef already
// does for a live run.
//
// A sector is SIP or DIP by what its states carry: `main` is the one-hole overlap list, `leading`
// the two-hole configurations. Routing on the data rather than on a flag is what lets one command
// combine a SIP document with a DIP one; nothing else distinguishes them at this level.
//
// The listing reproduces the document, so the -ps-thresh/-coeff-thresh of the ORIGINAL solve are
// baked in: states already dropped cannot be recovered here, only by re-solving. -ps-thresh can
// however be RAISED at this point, which is how a production listing is cut down to the states that
// carry real intensity — a whole-band DIP is a quasi-continuum whose median pole strength is a couple
// of percent, so most of its bulk is states no experiment resolves. Dropping them keeps each survivor
// Root unchanged, so the gaps in the printed numbering still say which roots went, exactly as they do
// when theADCcode's own threshold drops them.
func runRefConvert(paths []string, psThresh float64, out string) error {
	if len(paths) == 0 {
		return errors.New("-convert needs at least one path")
	}
	var secs []refSector
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		var doc refDoc
		// Stream it: json.Unmarshal would hold the raw bytes and the decoded tree at once.
		err = json.NewDecoder(bufio.NewReaderSize(f, 1<<20)).Decode(&doc)
		f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if len(doc.Sectors) == 0 {
			return fmt.Errorf("%s has no sectors (is it a solver document?)", path)
		}
		for _, sec := range doc.Sectors {
			isSIP := false
			for _, st := range sec.States {
				if len(st.Main) > 0 {
					isSIP = true
					break
				}
				if len(st.Leading) > 0 {
					break
				}
			}
			if isSIP {
				s := analyze.SIPSector{Irrep: sec.Irrep, Spin: sec.Spin}
				for _, st := range sec.States {
					if st.PSPercent < psThresh {
						continue
					}
					s.States = append(s.States, analyze.SIPState{
						Index: st.Index, EnergyEV: st.EnergyEV, PSPercent: st.PSPercent,
						Residue: st.Residue, Main: st.Main, Root: st.Root,
					})
				}
				secs = append(secs, s)
				continue
			}
			s := analyze.Sector{Irrep: sec.Irrep, Spin: sec.Spin}
			for _, st := range sec.States {
				if st.PSPercent < psThresh {
					continue
				}
				s.States = append(s.States, analyze.State{
					Index: st.Index, EnergyEV: st.EnergyEV, PSPercent: st.PSPercent,
					Residue: st.Residue, Leading: st.Leading, Root: st.Root,
				})
			}
			secs = append(secs, s)
		}
	}
	return emitRef(secs, out)
}

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
	psThresh, coeffThresh      float64
	format                     string // -format: json | ref
	blocks                     int
	nroots, maxdavsp, maxdavit int     // -solver davidson
	convthr                    float64 // -solver davidson
	profile                    bool
	spec                       specConfig

	matFree       sip.MatFreeMode        // dense (default) vs matrix-free 2h1p×2h1p satellite block
	matFreeBudget int64                  // -matfree=auto per-block dense-size threshold (bytes)
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

func runSIP(d *fcidump.Data, cfg sipConfig) error {
	if cfg.order != 3 {
		return fmt.Errorf("-sip -order %d is not available in this release (want 3: IP-ADC(3))", cfg.order)
	}
	if err := validateFormat(cfg.format, !cfg.spec.enabled, "a stick spectrum"); err != nil {
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

	doc := SIPDocument{NORB: d.NORB, NELEC: d.NELEC, Order: cfg.order, Solver: cfg.solver}

	// Enumerate the non-empty irrep sectors (independent solves).
	type sipItem struct {
		targetSym int
		sp        *sip.Space
	}
	var items []sipItem
	for _, targetSym := range syms {
		sp := sip.NewSpace(nocc, d.NORB, orbSym, targetSym)
		if sp.MainBlockSize() == 0 {
			continue // no 1h configurations in this irrep
		}
		items = append(items, sipItem{targetSym, sp})
	}

	// With a multi-GPU pool, run one sector per GPU concurrently.
	if len(ch.pool) >= 2 {
		sectors := make([]analyze.SIPSector, len(items))
		err := ch.runConcurrent(len(items), func(w *chooser, i int) error {
			it := items[i]
			res, mx, err := solveSIPSpace(w, fmt.Sprintf("sip irrep=%d", it.targetSym+1),
				it.sp, ints, eps, cfg.order, cfg)
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
			res, mx, err := solveSIPSpace(ch, label, it.sp, ints, eps, cfg.order, cfg)
			if err != nil {
				return err
			}
			doc.Sectors = append(doc.Sectors, analyze.BuildSIPSector(it.sp, res, mx.FMatrix(), opts))
			mx.Release()
		}
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

// flagGiven reports whether the named flag was set on the command line (as opposed to
// holding its default).
func flagGiven(name string) bool {
	given := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			given = true
		}
	})
	return given
}
