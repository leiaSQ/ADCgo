package main

// Tiered command-line help. `adcgo -h` prints the overview — the methods, the ADC
// orders, the solvers, and the handful of knobs that decide whether a large run fits
// — one line each, in the same order the README introduces them. Everything else is
// one level down: `adcgo -h <topic>` prints a topic page whose FLAGS section renders
// the real flag.CommandLine entries, so the detail can never drift from the flags the
// binary actually accepts. `adcgo -h all` is the unabridged dump.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"unicode"
)

// helpWidth is the column prose is wrapped at.
const helpWidth = 88

// helpTopic is one page of the second tier. body lines are wrapped unless they start
// with two spaces (verbatim: examples and tables) or "- " (bullet, hanging indent).
// flags names the flag.CommandLine entries rendered in full underneath.
type helpTopic struct {
	key     string
	aliases []string
	title   string
	usage   []string
	body    []string
	flags   []string
	see     []string
}

// helpTopics is the registry. Every flag the binary defines appears in exactly one
// topic's flags list at minimum (TestHelpTopicsCoverEveryFlag enforces both
// directions), so no flag becomes unreachable by hiding it from the overview.
var helpTopics = []helpTopic{
	// ---------------------------------------------------------------- input
	{
		key:     "input",
		aliases: []string{"fcidump", "mo", "integrals"},
		title:   "input — FCIDUMP and the MO sidecar",
		usage:   []string{"adcgo -fcidump FILE [-mo FILE] ..."},
		body: []string{
			"SCF and molecular integrals are delegated: ADCgo ingests a standard FCIDUMP (e.g. from pyscf). Without -dip/-sip it just reports the reference energy and the RHF-MP2 correlation energy — the integral-ingestion check.",
			"",
			"The -mo sidecar carries what FCIDUMP does not: MO coefficients, the AO overlap, and dipole integrals. It is required for",
			"- atom-resolved two-hole populations (DIP, Tarantelli U-transform),",
			"- DIP decay-channel spectra (-spectrum without it falls back to -bare).",
			"",
			"    python scripts/fixtures/gen_fcidump.py                 # testdata/h2o.fcidump + h2o.mo.json",
			"    adcgo -fcidump testdata/h2o.fcidump           # HF + MP2 ingestion check",
		},
		flags: []string{"fcidump", "mo", "version"},
		see:   []string{"dip", "sip"},
	},

	// --------------------------------------------------------------- methods
	{
		key:     "dip",
		aliases: []string{"doubleionization", "dipadc2"},
		title:   "double ionization — DIP-ADC(2)  (-dip)",
		usage:   []string{"adcgo -fcidump FILE -dip [-mo FILE] [-spin SEL] [-sym SEL] [-solver S]"},
		body: []string{
			"Dication states: energies, pole strengths, and leading two-hole configurations. With -mo, also atom-resolved two-hole populations (Tarantelli U-transform), which are what -spectrum builds Auger/ICD/ETMD decay channels from.",
			"",
			"One sector per point-group irrep and spin. Sectors are independent, so -gpus runs them concurrently, one GPU each; -mgpu instead spreads a single sector over a whole pool.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -dip -mo testdata/h2o.mo.json \\",
			"        -solver lanczos -spin both -sym all",
		},
		flags: []string{"dip", "spin", "sym", "blocks"},
		see:   []string{"lanczos", "lowmem", "spectrum", "mgpu"},
	},
	{
		key:     "sip",
		aliases: []string{"singleionization", "ip"},
		title:   "single ionization — IP-ADC(3)  (-sip)",
		usage:   []string{"adcgo -fcidump FILE -sip [-sym SEL] [-sigma S]"},
		body: []string{
			"Cation (doublet) states: ionization energies, spectroscopic factors, per-orbital one-hole overlaps. One sector per irrep.",
			"",
			"The scheme is non-Dyson IP-ADC(3) (`adcgo -h order`). The static self-energy is added to the main block by default (`adcgo -h sigma`); omitting it shifts every main line by ~0.2-0.35 eV.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -sip -sym all",
		},
		flags: []string{"sip", "order", "sym"},
		see:   []string{"order", "sigma", "matfree"},
	},
	{
		key:     "order",
		aliases: []string{"orders", "adc", "order3", "adc3"},
		title:   "ADC order — non-Dyson IP-ADC(3) for -sip, ADC(2) for -dip",
		usage:   []string{"adcgo -sip [-order 3] ..."},
		body: []string{
			"-sip is non-Dyson IP-ADC(3): the 3rd-order hole/hole self-energy and the 2nd-order 1h/2h1p coupling over a 1h main space and a 2h1p satellite. It is validated against pyscf's ip_adc on matched integrals. -order 3 is the default and the only value this release accepts.",
			"",
			"-dip is DIP-ADC(2) and takes no order of its own (-order is ignored).",
			"",
			"The 2h1p x 2h1p satellite is the memory ceiling at scale — TB-class for large systems — so a big -sip sector needs -matfree (`adcgo -h matfree`).",
		},
		flags: []string{"order"},
		see:   []string{"sip", "dip", "matfree", "sigma"},
	},

	// --------------------------------------------------------------- solvers
	{
		key:     "solver",
		aliases: []string{"solvers", "diag"},
		title:   "solvers — -solver S",
		usage:   []string{"adcgo ... -solver lanczos | lanczos-lowmem | davidson | dense", "adcgo -h lanczos | lowmem | davidson | dense"},
		body: []string{
			"Every method builds the same real-symmetric secular matrix; -solver only chooses how it is diagonalized. All four return identical energies and pole strengths on the states they resolve — pick by problem size and how much of the spectrum you need.",
			"",
			"                                                                          adcgo -h ...",
			"  lanczos         whole ionization band at once, matrix-free.             lanczos",
			"                  Default. -blocks N sets the resolution.",
			"  lanczos-lowmem  the same band with only a few Krylov panels             lowmem",
			"                  resident — the mode that puts a full band on a GPU.",
			"  davidson        root-targeting: the lowest -nroots, each iterated       davidson",
			"                  to a residual threshold. For a handful of exact roots.",
			"  dense           forms the matrix and calls LAPACK. Every state,         dense",
			"                  O(N^3)/O(N^2). The correctness oracle for the rest.",
		},
		flags: []string{"solver"},
		see:   []string{"matfree", "mgpu", "checkpoint"},
	},
	{
		key:     "lanczos",
		aliases: []string{"blocklanczos"},
		title:   "-solver lanczos — whole-band block-Lanczos  (default)",
		body: []string{
			"Matrix-free block-Lanczos: builds a Krylov subspace from the main-block start vectors and Rayleigh-Ritz-projects onto it, never storing the matrix. It sweeps the whole ionization band at once, so it is the right tool for a broad spectrum (Auger/ICD, the full DIP band).",
			"",
			"-blocks N sets the subspace size: Krylov dim = N x main-block size (theADCcode's `iter N`, whose reference DIP runs used 100). More blocks = finer resolution. Because it matches spectral MOMENTS rather than individual eigenvalues, an interior pole at a fixed -blocks can sit at the pole-strength centroid of a cluster rather than on any one true root.",
			"",
			"Supports -checkpoint for both SIP and DIP, so a walltime-killed solve resumes in a later process.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -dip -solver lanczos -blocks 100",
		},
		flags: []string{"solver", "blocks"},
		see:   []string{"lowmem", "checkpoint", "matfree"},
	},
	{
		key:     "lowmem",
		aliases: []string{"lanczoslowmem"},
		title:   "-solver lanczos-lowmem — the memory-frugal band",
		usage:   []string{"adcgo ... -solver lanczos-lowmem [-lowmem-block N]"},
		body: []string{
			"The same block-Lanczos band, re-cast to keep only a handful of Krylov panels resident instead of the whole basis — the memory mode that puts the full DIP band of a large system within reach of a GPU.",
			"",
			"- -lowmem-block 0 (default) is the faithful theADCcode short recurrence: block width == the 2h main-space size, a Tarantelli subspace-iteration gate plus a banded eigensolver, ~4 n x main panels live at once. This is Mode B, the only mode -mgpu and -checkpoint support.",
			"- -lowmem-block N below main selects the device-frugal full-reorthogonalization variant: 3 blocks on the GPU, the full basis staged in host RAM. Exact on the states it reaches, but a block narrower than main cannot span every pole-carrying direction, and it retains the basis on the host so it is not resumable.",
			"",
			"Mode B ends in a banded eigensolve of the projected matrix, and at production width that stage costs days rather than minutes: it is O(dim^2 x band) and runs on the CPU (no cuSOLVER band routine exists). -checkpoint or -profile makes it report the column it has reached, so a run is no longer silent between its last block and its results. -eig-workers turns on the parallel replay of that stage on the CPU, which is bit-exact but measured SLOWER than serial at the production bandwidth, so it is off by default. -eig-device instead replays the eigenvector rotations on a CUDA/HIP GPU, which is the one part of the stage a device suits: bandwidth-bound streaming over a multi-gigabyte accumulator, independent per row. -eig-b2 narrows the projected matrix's half-bandwidth first, which is the one structural saving: the reduction's band work is proportional to the bandwidth while its eigenvector work is not, so halving the bandwidth halves the larger half. The band reduction itself stays on the CPU, and no solver library helps — neither cuSOLVER nor hipSOLVER has a banded or tridiagonal eigensolver, and their dense syevd would want all dim x dim eigenvectors where this needs about 3000 rows of them.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -dip -solver lanczos-lowmem -sym all",
		},
		flags: []string{"solver", "lowmem-block", "eig-workers", "eig-device", "eig-b2"},
		see:   []string{"mgpu", "matfree", "checkpoint"},
	},
	{
		key:     "davidson",
		aliases: []string{"davidsonliu"},
		title:   "-solver davidson — root-targeting block Davidson-Liu",
		body: []string{
			"Matrix-free block Davidson-Liu: targets the algebraically lowest -nroots eigenpairs and iterates each to a residual threshold with a diagonal (theta - D)^-1 preconditioner. When a handful of converged low-lying eigenvalues is what is wanted rather than a broad envelope, it hits the exact positions at a fraction of the Lanczos subspace size.",
			"",
			"Works for -sip and -dip.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -dip -sym all \\",
			"        -solver davidson -nroots 8 -convthr 1e-3",
		},
		flags: []string{"solver", "nroots", "convthr", "maxdavsp", "maxdavit"},
		see:   []string{"lanczos", "solver"},
	},
	{
		key:     "dense",
		aliases: []string{"lapack", "fulldiag"},
		title:   "-solver dense — full diagonalization",
		body: []string{
			"Forms the full matrix and diagonalizes it directly (LAPACK dsyev). Exact and returns every state, but O(N^3) time and O(N^2) memory — use it for small sectors, for validation, and as the correctness oracle for the other three.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -dip -solver dense -sym all",
		},
		flags: []string{"solver"},
		see:   []string{"solver"},
	},

	// ---------------------------------------------------------------- output
	{
		key:     "output",
		aliases: []string{"out", "format", "json", "thresholds"},
		title:   "output — document, formats and thresholds",
		body: []string{
			"The default -dip/-sip output is the solver document: one JSON object with each state's energy, pole strength and leading configurations, on stdout unless -out names a file.",
			"",
			"-format ref emits theADCcode's own \"Eigenvalue (eV), ps (%), residue\" state list instead, byte-compatible with adcdip*.out so a run can be diffed straight against the reference implementation (main-space overlaps only; residue in a.u. as the reference prints it). It covers the solver document alone — -spectrum, -tdm and -convert reject it.",
			"",
			"-ps-thresh and -coeff-thresh prune the document. theADCcode's own reference thresholds are 0.5 (%) and 0.01.",
		},
		flags: []string{"out", "format", "ps-thresh", "coeff-thresh", "profile"},
		see:   []string{"spectrum", "convert"},
	},
	{
		key:     "spectrum",
		aliases: []string{"bare", "channels", "auger", "icd", "etmd"},
		title:   "stick spectra — -spectrum / -bare",
		usage:   []string{"adcgo ... (-dip|-sip) -spectrum [-mo FILE] [-init-atom A] [-group SPEC]", "adcgo ... (-dip|-sip) -bare"},
		body: []string{
			"Solve and classify in one pass, emitting a stick-spectrum JSON that cmd/plotspec renders.",
			"",
			"- -spectrum on DIP with -mo gives decay channels (Auger/ICD/ETMD), built from the atom-resolved two-hole populations; -init-atom picks the core-ionized site and -group defines composite or passive sites. Without -mo it falls back to the bare spectrum.",
			"- -spectrum on SIP decomposes per orbital.",
			"- -bare skips classification entirely: one stick per state, energy = ionization energy, intensity = pole strength/100, all on a single `states` channel. Works for both -dip and -sip and needs no sidecar.",
			"",
			"-group NAME=col,~col is repeatable and a bare -group opens an interactive prompt; a ~column is passive (part of the site, but a hole there is not a decay channel).",
			"",
			"    adcgo -fcidump testdata/h2o_dzp.fcidump -dip -mo testdata/h2o_dzp.mo.json \\",
			"        -solver dense -sym all -spectrum -init-atom O",
			"",
			"    # both H as a passive \"water\" site: only Auger@wat survives",
			"    adcgo ... -spectrum -group \"wat=O,~H1,~H2\" -init-atom wat",
			"",
			"    adcgo ... -bare -out bare.json && plotspec -in bare.json -out bare.png -fwhm 1.0",
		},
		flags: []string{
			"spectrum", "bare", "init-atom", "init-orbital", "group",
			"min-weight", "min-fraction", "include-zero",
			"st-ratio", "molecule", "basis", "point-group",
		},
		see: []string{"convert", "dip"},
	},
	{
		key:     "convert",
		aliases: []string{"reconvert"},
		title:   "-convert — re-derive a spectrum without re-solving",
		usage:   []string{"adcgo -convert FILE (-dip|-sip) [-spectrum -mo FILE -group ...] [-out FILE]"},
		body: []string{
			"Reads a previously emitted solver document (the default -dip/-sip JSON) and emits its stick spectrum. No FCIDUMP and no re-solve: the document already carries every state's energy and pole strength. Pass -dip or -sip to say which kind of document it is.",
			"",
			"A DIP document written with -mo also carries the per-atom two-hole populations, so -group/-spectrum can regroup it into a full decay-channel spectrum after the fact. Otherwise the result is the bare spectrum, byte-identical to running -bare on the original problem.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -dip -solver dense -out dip.json",
			"    adcgo -convert dip.json -dip -out bare.json",
		},
		flags: []string{"convert"},
		see:   []string{"spectrum", "output"},
	},
	// ----------------------------------------------------------------- scale
	{
		key:     "backend",
		aliases: []string{"backends", "gpu", "gpus", "cuda", "hip"},
		title:   "backends and GPUs — -backend, -gpus",
		body: []string{
			"The default backend is pure Go (gonum); the accelerated ones are build-tag gated and selected at runtime with -backend.",
			"",
			"    go build -tags openblas ./cmd/adcgo     # multicore CPU BLAS",
			"    go build -tags cuda     ./cmd/adcgo     # cuBLAS",
			"    go build -tags hip      ./cmd/adcgo     # hipBLAS",
			"",
			"-backend auto calibrates each available backend once and picks the predicted-fastest per sector, measuring the real mat-vec cost rather than a flop estimate.",
			"",
			"On a multi-GPU node there are two independent parallelism axes:",
			"- -gpus N runs INDEPENDENT sectors concurrently, one GPU each (DIP spin x irrep, SIP irrep).",
			"- -mgpu N row-partitions a SINGLE sector across N GPUs — see `adcgo -h mgpu`.",
		},
		flags: []string{"backend", "gpus"},
		see:   []string{"mgpu", "matfree"},
	},
	{
		key:     "matfree",
		aliases: []string{"matrixfree", "maxmem"},
		title:   "-matfree — recompute instead of store",
		usage:   []string{"adcgo ... -matfree off|auto|on [-maxmem GB]"},
		body: []string{
			"By default the block-sparse ADC operator is materialized: every nonzero block is assembled once and kept resident, so each mat-vec is a batched GEMM. For a large sector the dominant blocks — the DIP 3h1p<->3h1p satellite region and the SIP-ADC(3) 2h1p x 2h1p satellite — are the resident-memory ceiling: hundreds of GB to several TB, larger than a whole 8-GPU node.",
			"",
			"-matfree on recomputes those blocks from the MO integrals on every mat-vec and never stores them (the direct-sigma approach theADCcode uses), collapsing the footprint to the Krylov panels plus the small main/coupling blocks. -matfree auto decides per block by dense size against -maxmem; off keeps everything materialized.",
			"",
			"It runs everywhere the dense path does — host, GPU (a custom recompute kernel over a device-resident ERI tensor), and composed with -mgpu — and is what puts a whole-band run of a large system on one node without dropping polarization or freezing extra orbitals.",
			"",
			"    adcgo-cuda -fcidump system.fcidump -dip -solver lanczos-lowmem -lowmem-block 0 \\",
			"        -mgpu 8 -matfree on -backend cuda -spin both -sym all -blocks 200",
		},
		flags: []string{"matfree", "maxmem"},
		see:   []string{"mgpu", "lowmem", "order3"},
	},
	{
		key:     "mgpu",
		aliases: []string{"multigpu", "distributed", "nvlink"},
		title:   "-mgpu — one sector row-partitioned across a node's GPUs",
		usage:   []string{"adcgo -dip -solver lanczos-lowmem -lowmem-block 0 -mgpu N -backend cuda ..."},
		body: []string{
			"At production scale one whole-band Mode B Krylov block can dwarf a single GPU (~137 GB for the production system, past a 141 GB H200). -mgpu N row-partitions ONE sector across N GPUs: the live n x main Krylov panels and the block-sparse operator are split along the config (row) dimension, so a block that fits nowhere alone fits spread over the pool.",
			"",
			"Every solver reduction (the alpha coefficients, the CGS2 projection, the Gram, dots and norms) contracts the row dimension, so each becomes a device-local partial plus a tiny all-reduce; only the mat-vec crosses devices, gathering the remote input band per apply — over NVLink peer-to-peer when the backend supports it, else staged through the host. Tested to 8xH200.",
			"",
			"Requires -dip -solver lanczos-lowmem -lowmem-block 0 and a fast inter-GPU link. Sectors run SERIALLY, each spanning the whole pool — in contrast to -gpus, which runs independent sectors concurrently. Row-partitioning only divides the operator by the pool size, so pair it with -matfree on for the multi-TB satellite region.",
			"",
			"-satchunk and -sat-trace tune and instrument the matrix-free satellite apply. See scripts/slurm/dip.sbatch for a complete SLURM job.",
			"",
			"    adcgo-cuda -fcidump system.fcidump -dip -order 2 \\",
			"        -solver lanczos-lowmem -lowmem-block 0 -mgpu 8 -matfree on \\",
			"        -backend cuda -spin both -sym all -blocks 200",
		},
		flags: []string{"mgpu", "satchunk", "sat-trace", "mgpu-device-symeig"},
		see:   []string{"lowmem", "matfree", "backend"},
	},
	{
		key:     "checkpoint",
		aliases: []string{"cache", "caches", "resume", "daisychain"},
		title:   "checkpointing and caches — surviving a walltime kill",
		usage:   []string{"adcgo ... -checkpoint PATH [-checkpoint-every N]"},
		body: []string{
			"-checkpoint makes a solve resumable in a later process. Each sector appends a suffix (SIP .i<irrep>, DIP .s<spin>.i<irrep>). A SIGUSR1 — SLURM --signal=B:USR1@<grace> — makes the run checkpoint and exit 64 (\"resume needed\"); exit 0 means done. -checkpoint-every also saves every N blocks for crash resilience.",
			"",
			"Supported by -solver lanczos (SIP and DIP) and by -solver lanczos-lowmem with -lowmem-block 0 (DIP Mode B only — the narrow-block mode retains the whole basis on the host and is not resumable).",
			"",
			"The two caches cover what -checkpoint does not, because they are built BEFORE the Krylov loop starts and would otherwise be rebuilt by every generation of a daisychain:",
			"- -mainblock-cache — the assembled SIP 1h/1h main block. Only 26 KB at production scale, but every element is an O(nvir^4 nocc) sum: 8h16m to build.",
			"- -sigma-cache — the static self-energy. Sigma(inf) dominates a large SIP run (78 h for the production system) and is only n^2 floats.",
			"Both default to auto (a file beside the FCIDUMP) and both reject a stale copy: the ADC order, sector, space dimensions, orbital-energy hash and FCIDUMP size/mtime must all match.",
		},
		flags: []string{"checkpoint", "checkpoint-every", "mainblock-cache", "sigma-cache"},
		see:   []string{"lanczos", "lowmem", "sigma"},
	},
	{
		key:     "sigma",
		aliases: []string{"selfenergy", "static"},
		title:   "-sigma — the static self-energy on the SIP main block",
		body: []string{
			"The ADC matrix code does not build Sigma: theADCcode keeps it in a separate &self-energy module and subtracts it. Omitting it shifts every main line by ~0.2-0.35 eV, so ADCgo adds it by default.",
			"",
			"-sigma auto (the default) means `infinite` — the all-order resolvent resummation, bit-exact against theADCcode. off | three | four | fplus select the lower-order schemes instead. -sigma-akrit and -sigma-maxit tune the resolvent iteration (the reference's own defaults are 1e-9 and 30).",
			"",
			"Sigma(inf) is hours of work on a large system and is cached — see `adcgo -h checkpoint`.",
		},
		flags: []string{"sigma", "sigma-akrit", "sigma-maxit"},
		see:   []string{"sip", "checkpoint"},
	},
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// helpRequested scans the raw arguments for -h/-help/--help before flag.Parse sees
// them, and returns everything after it as the normalized topic. The flag package
// would reject "sip" in `adcgo -h sip` as a stray positional, and its own -h handling
// prints the unabridged dump this package exists to replace.
func helpRequested(args []string) (string, bool) {
	for i, a := range args {
		switch strings.ToLower(strings.TrimLeft(a, "-")) {
		case "h", "help", "?":
			return normalizeTopic(strings.Join(args[i+1:], " ")), true
		}
	}
	return "", false
}

// normalizeTopic folds a topic argument to its lookup key: case, separators and
// leading dashes are ignored, so "-h lanczos-lowmem", "-h LANCZOS LOWMEM" and
// "-h -solver lanczos" all land on the same page.
func normalizeTopic(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		}
	}
	key := b.String()
	// "-h solver lanczos": the index name plus a member selects the member's page.
	// "-h order 3" folds to the "order3" alias of the order page.
	if rest := strings.TrimPrefix(key, "solver"); rest != "" && rest != key {
		return rest
	}
	return key
}

// lookupTopic resolves a normalized key against the registry keys and aliases.
func lookupTopic(key string) *helpTopic {
	for i := range helpTopics {
		t := &helpTopics[i]
		if normalizeTopic(t.key) == key {
			return t
		}
		for _, a := range t.aliases {
			if normalizeTopic(a) == key {
				return t
			}
		}
	}
	return nil
}

// printHelp writes the overview (empty topic), one topic page, or the unabridged flag
// dump. It reports whether the topic was known; an unknown one lists the index.
func printHelp(w io.Writer, topic string) bool {
	switch topic {
	case "":
		printOverview(w)
		return true
	case "topics", "index", "list":
		printTopicIndex(w)
		return true
	case "all", "flags", "everything", "full":
		fmt.Fprintf(w, "ADCgo — every flag, unabridged. `adcgo -h` is the grouped overview.\n\n")
		flag.CommandLine.SetOutput(w)
		flag.PrintDefaults()
		return true
	}
	if t := lookupTopic(topic); t != nil {
		printTopic(w, t)
		return true
	}
	fmt.Fprintf(os.Stderr, "adcgo: no help topic %q\n\n", topic)
	printTopicIndex(w)
	return false
}

// ---------------------------------------------------------------------------
// Tier 1: the overview
// ---------------------------------------------------------------------------

func printOverview(w io.Writer) {
	fmt.Fprint(w, `ADCgo — exact, hardware-accelerated ADC(n) ionization

USAGE
  adcgo -fcidump FILE (-dip | -sip) [-solver S] [options] [-out FILE]
  adcgo -h TOPIC                    detail on one method, solver or knob
  adcgo -h all                      every flag, unabridged
  adcgo -h topics                   list every help topic

METHODS                                                        detail: adcgo -h ...
  -dip                    double ionization, DIP-ADC(2)                    dip
  -sip                    single ionization, IP-ADC(3)                     sip

SOLVERS  -solver S        same matrix, same answers — pick by size and band
  lanczos       (default) whole-band block-Lanczos, -blocks N              lanczos
  lanczos-lowmem          same band, few panels resident — the GPU mode    lowmem
  davidson                converge the lowest -nroots exactly              davidson
  dense                   full LAPACK diagonalization, every state         dense

OUTPUT                    default: solver document JSON (E, ps, configs)
  -spectrum               decay channels (DIP + -mo) or per orbital (SIP)  spectrum
  -bare                   one stick per state: energy + pole strength      spectrum
  -convert FILE           re-derive a spectrum, no re-solve                convert
  -format json|ref        ref = theADCcode's state list, diffable          output

SCALE                     what makes a large system fit
  -backend B              gonum | hip | cuda | auto                        backend
  -matfree off|auto|on    recompute the memory-dominant blocks             matfree
  -mgpu N                 row-partition ONE sector across N GPUs           mgpu
  -checkpoint PATH        resume after a walltime kill; block caches       checkpoint

COMMON
  -fcidump PATH           FCIDUMP with MO integrals (required)             input
  -mo PATH                MO/overlap/dipole sidecar                        input
  -sym all|none|N         target irrep                        (default all)
  -spin both|singlet|triplet   DIP spin sector                (default both)
  -ps-thresh P            drop states below P % pole strength  (default 1)
  -profile                per-sector phase timings to stderr
`)
}

// printTopicIndex lists every topic key, for an unknown-topic miss.
func printTopicIndex(w io.Writer) {
	fmt.Fprintln(w, "Help topics:")
	keys := make([]string, 0, len(helpTopics))
	for _, t := range helpTopics {
		keys = append(keys, t.key)
	}
	sort.Strings(keys)
	for i := 0; i < len(keys); i += 4 {
		end := min(i+4, len(keys))
		var line strings.Builder
		for _, k := range keys[i:end] {
			fmt.Fprintf(&line, "  %-18s", k)
		}
		fmt.Fprintln(w, strings.TrimRight(line.String(), " "))
	}
	fmt.Fprintln(w, "\n  all                 every flag, unabridged")
}

// ---------------------------------------------------------------------------
// Tier 2: one topic page
// ---------------------------------------------------------------------------

func printTopic(w io.Writer, t *helpTopic) {
	fmt.Fprintf(w, "ADCgo — %s\n", t.title)
	if len(t.usage) > 0 {
		fmt.Fprintln(w)
		for _, u := range t.usage {
			fmt.Fprintf(w, "  %s\n", u)
		}
	}
	if len(t.body) > 0 {
		fmt.Fprintln(w)
		writeBody(w, t.body)
	}
	if len(t.flags) > 0 {
		fmt.Fprintf(w, "\nFLAGS\n")
		for _, name := range t.flags {
			writeFlag(w, name)
		}
	}
	if len(t.see) > 0 {
		refs := make([]string, len(t.see))
		for i, s := range t.see {
			refs[i] = "adcgo -h " + s
		}
		fmt.Fprintf(w, "\nSEE ALSO\n  %s\n", strings.Join(refs, "    "))
	}
}

// writeBody renders a topic body: a line indented by two spaces is verbatim (examples
// and tables), a "- " line is a bullet with a hanging indent, anything else is a
// wrapped paragraph.
func writeBody(w io.Writer, lines []string) {
	for _, ln := range lines {
		switch {
		case strings.TrimSpace(ln) == "":
			fmt.Fprintln(w)
		case strings.HasPrefix(ln, "  "):
			fmt.Fprintln(w, ln)
		case strings.HasPrefix(ln, "- "):
			writeWrapped(w, ln, "  ", "    ")
		default:
			writeWrapped(w, ln, "  ", "  ")
		}
	}
}

// writeFlag renders one flag.CommandLine entry: its name and value type, its default
// when there is a meaningful one, and its own usage string wrapped underneath. The
// text comes from the flag itself, so a topic page can never describe a flag the
// binary no longer has.
func writeFlag(w io.Writer, name string) {
	f := flag.CommandLine.Lookup(name)
	if f == nil { // guarded by TestHelpTopicsCoverEveryFlag
		fmt.Fprintf(w, "  -%s (undefined)\n", name)
		return
	}
	kind, usage := flag.UnquoteUsage(f)
	head := "  -" + f.Name
	if kind != "" {
		head += " " + kind
	}
	if def := f.DefValue; def != "" && def != "false" && def != "0" {
		head += fmt.Sprintf("  (default %s)", def)
	}
	fmt.Fprintln(w, head)
	writeWrapped(w, usage, "      ", "      ")
}

// writeWrapped word-wraps text to helpWidth, with separate first-line and
// continuation indents.
func writeWrapped(w io.Writer, text, first, cont string) {
	words := strings.Fields(text)
	if len(words) == 0 {
		fmt.Fprintln(w)
		return
	}
	line := first + words[0]
	for _, word := range words[1:] {
		if len([]rune(line))+1+len([]rune(word)) > helpWidth {
			fmt.Fprintln(w, line)
			line = cont + word
			continue
		}
		line += " " + word
	}
	fmt.Fprintln(w, line)
}
