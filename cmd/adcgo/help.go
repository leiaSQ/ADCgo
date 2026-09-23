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
			"- DIP decay-channel spectra (-spectrum without it falls back to -bare),",
			"- transition dipole moments (-tdm),",
			"- per-channel partial widths under -fano.",
			"",
			"    python scripts/fixtures/gen_fcidump.py                 # testdata/h2o.fcidump + h2o.mo.json",
			"    adcgo -fcidump testdata/h2o.fcidump           # HF + MP2 ingestion check",
		},
		flags: []string{"fcidump", "mo"},
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
		title:   "single ionization — IP-ADC(n)  (-sip)",
		usage:   []string{"adcgo -fcidump FILE -sip [-order 2|3|4|22] [-sym SEL] [-mo FILE]"},
		body: []string{
			"Cation (doublet) states: ionization energies, spectroscopic factors, per-orbital one-hole overlaps. One sector per irrep.",
			"",
			"-order picks the ADC scheme; it is the choice that decides both accuracy and cost — see `adcgo -h order`. The static self-energy is added to the main block by default (`adcgo -h sigma`); omitting it shifts every main line by ~0.2-0.35 eV.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -sip -order 3 -sym all",
		},
		flags: []string{"sip", "order", "sym"},
		see:   []string{"order", "sigma", "fano", "tdm"},
	},
	{
		key:     "order",
		aliases: []string{"orders", "adc"},
		title:   "ADC orders for -sip",
		usage:   []string{"adcgo -sip -order 2|3|4|22 ...", "adcgo -h order2 | order3 | order4 | order22"},
		body: []string{
			"All four build the same kind of real-symmetric secular matrix and run through the same solvers; they differ in which blocks exist and to what order they are correct.",
			"",
			"                                                                          adcgo -h ...",
			"  -order 2    extended ADC(2), \"ADC(2)x\": 2nd-order 1h/1h, 1st-order          order2",
			"              1h/2h1p coupling, 1st-order 2h1p satellite",
			"  -order 3    non-Dyson IP-ADC(3) (default): adds the 3rd-order              order3",
			"              self-energy and the 2nd-order coupling",
			"  -order 4    CVS Dyson ADC(4) for core holes; needs -core                   order4",
			"  -order 22   ADC(2,2): explicit 3h2p class — what makes second-order        order22",
			"              decay (double Auger, double ICD) describable at all",
			"",
			"-dip is DIP-ADC(2) and takes no order of its own.",
		},
		flags: []string{"order"},
		see:   []string{"sip", "fano"},
	},
	{
		key:     "order2",
		aliases: []string{"adc2", "adc2x"},
		title:   "-order 2 — extended ADC(2) (ADC(2)x)",
		body: []string{
			"The reference's extended ADC(2). The 1h/1h main block carries the 2nd-order self-energy, the 1h/2h1p coupling is 1st order, and the 2h1p/2h1p satellite keeps its full 1st-order block (that first-order satellite is what the \"extended\" means — plain ADC(2) leaves it diagonal).",
			"",
			"Cheapest SIP scheme, and the base for Fano-ADC(2)x widths (`adcgo -h fano`).",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -sip -order 2 -sym all",
		},
		flags: []string{"order"},
		see:   []string{"order", "sigma", "fano"},
	},
	{
		key:     "order3",
		aliases: []string{"adc3"},
		title:   "-order 3 — non-Dyson IP-ADC(3)  (default)",
		body: []string{
			"Adds the 3rd-order hole/hole self-energy and the 2nd-order 1h/2h1p coupling on top of order 2. 1h main space, 2h1p satellite. This is the default and the production valence scheme; it is validated against pyscf's ip_adc on matched integrals.",
			"",
			"The 2h1p x 2h1p satellite is the memory ceiling at scale — TB-class for large systems — so a big order-3 sector needs -matfree (`adcgo -h matfree`).",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -sip -order 3 -sym all",
		},
		flags: []string{"order"},
		see:   []string{"order", "matfree", "sigma"},
	},
	{
		key:     "order4",
		aliases: []string{"adc4", "cvs", "core"},
		title:   "-order 4 — CVS Dyson ADC(4) for core holes",
		usage:   []string{"adcgo -fcidump FILE -sip -order 4 -core LIST -sym IRREP"},
		body: []string{
			"Core-valence-separated Dyson ADC(4). -core names the occupied core orbital(s), 0-based. Only the core orbital's irrep has a main block, so pin the sector with -sym.",
			"",
			"The bare core diagonal is Koopmans-level: use it for relative core-state structure, not absolute core binding energies. -wert3 (on by default) adds the WERT3 5th-order 3h2p-diagonal correction — the full EIGAB effective diagonal theADCcode itself uses, bit-exact against its FT19 tape.",
			"",
			"A handful of converged core roots is usually what is wanted, which is -solver davidson territory rather than a whole-band Lanczos sweep.",
			"",
			"    # O 1s of water (orbital 0, a1 sector), lowest 8 roots",
			"    adcgo -fcidump testdata/h2o.fcidump -sip -order 4 -core 0 -sym 0 \\",
			"        -solver davidson -nroots 8 -convthr 1e-3",
		},
		flags: []string{"order", "core", "wert3"},
		see:   []string{"davidson", "matfree", "tdm"},
	},
	{
		key:     "order22",
		aliases: []string{"adc22"},
		title:   "-order 22 — ADC(2,2)",
		usage:   []string{"adcgo -fcidump FILE -sip -order 22 [-adc22 f|x|m] ..."},
		body: []string{
			"The ADC(2,2) scheme of Kolorenc & Averbukh, J. Chem. Phys. 152, 214107 (2020). Its 1h/1h block stays 2nd order (emphatically not third), and it carries an explicit 3h2p class — the class that makes second-order decay, double Auger and double ICD, describable at all.",
			"",
			"-adc22 selects the Table I variant:",
			"- f  full — the paper's recommendation and the only variant that gets double Auger right,",
			"- x  drops the 2nd-order 1h/2h1p coupling (~24% overshoot in decay width),",
			"- m  also leaves the 1st-order 3h2p/3h2p block diagonal (~14% overshoot).",
			"m and x are diagnostics, not production settings.",
			"",
			"    adcgo -fcidump ne.fcidump -sip -order 22 -adc22 f -fano -fano-init 0 \\",
			"        -sym all -solver lanczos -matfree on",
		},
		flags: []string{"order", "adc22"},
		see:   []string{"fano", "matfree", "order"},
	},
	{
		key:     "fano",
		aliases: []string{"width", "widths", "gamma", "lifetime"},
		title:   "decay widths and lifetimes — Fano/Feshbach + Stieltjes  (-fano)",
		usage: []string{
			"adcgo -fcidump FILE -sip -order 2|3|22 -fano -fano-init K",
			"adcgo -fcidump FILE -dip -spin singlet|triplet -sym none|IRREP -fano -fano-init K",
			"       [-fano-q LIST] [-fano-rule any|all] [-fano-qp SPEC]",
		},
		body: []string{
			"Decay rates, not just channels: Gamma and tau = hbar/Gamma for a chosen vacancy, by the Fano/Feshbach method with Stieltjes imaging (Kolorenc & Averbukh 2020). The configuration space is partitioned into a bound half Q (still carrying the initial vacancy) and a pseudo-continuum P; |Phi> comes from the QMQ spectrum, the coupling density from PMP. Under -sip the vacancy fixes the target irrep, so -sym is determined rather than chosen. -dip -fano runs one DIP-ADC(2) sector (-spin singlet|triplet, -sym none|IRREP); its decay class is 3h1p, so -mo partial widths are not available there. -khci K -fano runs the same pipeline over k-hole CI (adcgo -h khci).",
			"",
			"Partition rules:",
			"- -fano-rule any — a configuration is bound if at least one hole is in the Q set. The Auger criterion: right for local decay.",
			"- -fano-rule all — bound only if every hole is on the donor subunit. Right for ICD/ETMD, where a hole OUTSIDE the donor is what marks a decay channel.",
			"- -fano-qp — a Q/P partition stated per excitation class, for the cases neither reading can express (see its flag text for the grammar and two worked atomic examples).",
			"",
			"With -mo, approximate partial widths per channel (Auger@A, ICD:A->B, ETMD, and `double` for the 3h2p/second-order channel), each imaged separately.",
			"",
			"BASIS REQUIREMENT — what decides whether a run is possible at all. Imaging can only evaluate Gamma(E_Phi) if the 2h1p pseudo-continuum brackets E_Phi, and a 2h1p state sits at eps_a - eps_k - eps_l. What matters is the energy span and level density of the VIRTUAL space near E_Phi, not diffuseness: for Ne 1s (E_Phi ~ 32 E_h) aug-cc-pVTZ tops out at 14.6 E_h and cannot describe the decay at all.",
			"",
			"Two printed diagnostics decide whether a width is trustworthy:",
			"- coupled channels — how many P configurations carry any coupling. A few tens gives a large Stieltjes spread however good the rest of the pipeline is.",
			"- sum-rule residual — the pseudo-continuum's total strength against the exact 2pi|g|^2. A large value means the Krylov space was truncated before it captured the coupling; raise -fano-blocks until it closes. An unconverged width can still carry a SMALL error bar, because the Stieltjes orders agree with each other about the wrong density.",
			"",
			"    # Ne+ (1s^-1) Auger width",
			"    adcgo -fcidump ne.fcidump -sip -order 22 -adc22 f -fano -fano-init 0 \\",
			"        -sym all -solver lanczos -matfree on",
			"",
			"    # interatomic decay: Q is every configuration with ALL holes on the donor",
			"    adcgo -fcidump dimer.fcidump -sip -order 22 -fano -fano-init 2 \\",
			"        -fano-q 2,3,4 -fano-rule all -mo dimer.mo.json -init-atom A",
		},
		flags: []string{"fano", "fano-init", "fano-q", "fano-rule", "fano-qp", "fano-nth", "fano-qmin", "fano-phi-holes", "fano-phi-tol"},
		see:   []string{"fanotuning", "stieltjes", "order22", "khci"},
	},
	{
		key:     "khci",
		aliases: []string{"kholeci", "quip"},
		title:   "k-hole CI Fano widths  (-khci K -fano)",
		usage: []string{
			"adcgo -fcidump FILE -khci K -fano -fano-init I -sym none|IRREP",
			"       [-mo LABELLED.mo.json -fano-qp 'p: charge A=1,B=1 & free=1'] [-khci-maxfree 1]",
		},
		body: []string{
			"The Fano width over configuration interaction in the classes Kh | (K+1)h1p | (K+2)h2p (K = 1..4) instead of an ADC matrix: M = H - E_HF with the general Fock matrix, so localized occupied and compact/free virtual orbitals are allowed (the spectrum is invariant under occupied-occupied and virtual-virtual rotations). For K = 4 the classes are 4h | 5h1p | 6h2p.",
			"",
			"With a labelled sidecar (dump_fcidump's &orbitals localized scheme: orb_kind, orb_atom), -fano-qp takes the net-charge grammar: net charge on atom X = holes on X - compact particles on X, and free particles belong to no atom. A hole list cannot express the relaxed open channel, whose (K+2)h2p configurations put a hole AND a compact particle on the same atom.",
			"",
			"The Q and P operators are stored as CSR when they fit -maxmem, since their solves apply them many times; otherwise every apply enumerates the couplings again.",
			"",
			"A lock-in measures the noise floor instead of assuming it. -fano-lambda 'C:D=0' scales the C-D transfer integrals to zero, so a width that needs that transfer vanishes exactly and what remains is the floor; with -fano-save-g at the four corners of (lambda_1, lambda_2) in {0,1}^2, the double difference g_AB = g(1,1) - g(1,0) - g(0,1) + g(0,0) is imaged with -fano-image-g. -fano-decompose splits the width by the excitation class of |Phi> the coupling starts from (the time orderings).",
			"",
			"    # a four-hole initial state; every atom +1 and one free electron is the continuum",
			"    adcgo -fcidump sys.fcidump -mo sys.mo.json -khci 4 -khci-maxfree 1 -fano \\",
			"        -fano-init 0 -sym none -fano-qp 'p: charge *=1 & free=1'",
		},
		flags: []string{"khci", "khci-maxclass", "khci-twoms", "khci-maxfree", "fano-lambda", "fano-save-g", "fano-image-g", "fano-decompose"},
		see:   []string{"fano", "fanotuning"},
	},
	{
		key:     "fanotuning",
		aliases: []string{"fanocuts", "continuum"},
		title:   "-fano tuning — the pseudo-continuum solve and its cuts",
		body: []string{
			"Which states enter the coupling density, and how the two halves are solved. The partition itself is on `adcgo -h fano`.",
			"",
			"The QMQ (bound) half wants a different solver from the PMP (continuum) half that -solver governs: only a few of QMQ's LOWEST roots are wanted, and under most partitions Q's main block is a single configuration, so -fano-qsolver defaults to davidson.",
			"",
			"The energy ceiling is the cut that matters most. The reference hard-codes 4 hartree, and that constant does NOT transfer — it is a polarization-propagator scale. -fano-emax-rel states it as a multiple of E_Phi instead, which is the transferable form; scanning it is the honest convergence check, and Gamma plateaus over roughly 3-10x.",
			"",
			"    # converge the width against the ceiling before believing it",
			"    for r in 2 3 5 8; do adcgo ... -fano -fano-init 0 -fano-emax-rel $r; done",
		},
		flags: []string{
			"fano-qsolver", "fano-qroots", "fano-block", "fano-blocks",
			"fano-emax", "fano-emax-rel", "fano-wmin", "fano-gmin", "fano-gmin-rel", "fano-at",
			"fano-engine", "fano-order", "fano-si-shift",
		},
		see: []string{"fano", "stieltjes"},
	},
	{
		key:     "stieltjes",
		aliases: []string{"imaging"},
		title:   "Stieltjes imaging — turning the pseudo-continuum into Gamma(E)",
		body: []string{
			"The discrete PMP spectrum is a set of L^2 pseudo-states, not a continuum; Stieltjes imaging reconstructs the coupling density from its moments and evaluates it at E_Phi. Each order is an independent estimate, and the spread over orders is the error bar.",
			"",
			"The maximum usable order is set purely by the arithmetic precision of the moment recurrence (-stieltjes-prec): on a Lorentzian model, 53 bits reaches order 17 with a 21% density error, 113 bits (the reference's REAL*16) order 34 with 2.9%, and the default 256 bits order 72 with 0.37%.",
			"",
			"-stieltjes-average paper (the default) is Kolorenc & Averbukh's own protocol — the mean over -stieltjes-window consecutive orders in the region of best convergence, with that window's standard deviation as the uncertainty, which is what their tabulated error bars are.",
		},
		flags: []string{"stieltjes-orders", "stieltjes-prec", "stieltjes-average", "stieltjes-window"},
		see:   []string{"fano"},
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
			"    adcgo -fcidump testdata/h2o.fcidump -dip -solver lanczos-lowmem -sym all",
		},
		flags: []string{"solver", "lowmem-block"},
		see:   []string{"mgpu", "matfree", "checkpoint"},
	},
	{
		key:     "davidson",
		aliases: []string{"davidsonliu"},
		title:   "-solver davidson — root-targeting block Davidson-Liu",
		body: []string{
			"Matrix-free block Davidson-Liu: targets the algebraically lowest -nroots eigenpairs and iterates each to a residual threshold with a diagonal (theta - D)^-1 preconditioner. When a handful of converged interior eigenvalues is what is wanted (the lowest ~20 core-edge roots, say) rather than a broad envelope, it hits the exact positions at a fraction of the Lanczos subspace size — this is what reproduces a legacy adc4_diag.x Davidson run directly.",
			"",
			"Works for -sip (including CVS -order 4) and -dip. It is also the default QMQ solver under -fano, where only the lowest few bound roots matter.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -sip -order 4 -core 0 -sym 0 \\",
			"        -solver davidson -nroots 8 -convthr 1e-3",
		},
		flags: []string{"solver", "nroots", "convthr", "maxdavsp", "maxdavit"},
		see:   []string{"order4", "fano", "solver"},
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
		see: []string{"convert", "tdm", "dip"},
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
	{
		key:     "tdm",
		aliases: []string{"rassi", "dipole", "dipoles", "emission"},
		title:   "transition dipole moments — -tdm (-rassi)",
		usage:   []string{"adcgo -fcidump FILE -sip -mo FILE -tdm [-order 4 -core LIST]"},
		body: []string{
			"RASSI-like transition properties along the ICD decay chain, from a single-ionization run. Requires -sip and a -mo sidecar carrying dipole integrals. Three sections:",
			"",
			"- emissions — ion->ion radiative transitions within a sector (mu, oscillator strength f, Einstein A in s^-1). Within one sector only the totally-symmetric dipole component connects states.",
			"- photoionization — each cation state's Dyson orbital contracted with the dipole integrals into an L^2 pseudo-spectrum mu(eps_a), one channel per virtual orbital (the ejected-electron proxy).",
			"- cross_emissions — for -order 4, core->valence X-ray emission between the CVS core sector and companion plain-ADC(3) valence sectors.",
			"",
			"-tdm-isr picks the ISR property-matrix order: 2 (correlation-corrected) is the default for -order 2/3 where it is order-consistent with the secular matrix; -order 4 defaults to 0 because an ADC(4)-consistent property matrix would need 3rd/4th-order terms that do not exist yet.",
			"",
			"    adcgo -fcidump testdata/h2o.fcidump -sip -order 3 \\",
			"        -mo testdata/h2o.mo.json -solver dense -tdm -out tdm.json",
			"    plotspec -mode tdm -in tdm.json -out tdm.png",
		},
		flags: []string{"tdm", "rassi", "tdm-osc-thresh", "tdm-isr"},
		see:   []string{"order4", "spectrum", "sip"},
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
			"By default the block-sparse ADC operator is materialized: every nonzero block is assembled once and kept resident, so each mat-vec is a batched GEMM. For a large sector the dominant blocks — the DIP 3h1p<->3h1p satellite region, the SIP ADC(4) 2h1p x 3h2p and ADC(3) 2h1p^2 coupling — are the resident-memory ceiling: hundreds of GB to several TB, larger than a whole 8-GPU node.",
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
			"-satchunk and -sat-trace tune and instrument the matrix-free satellite apply. See scripts/helix/HELIX.md for a complete SLURM job.",
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
	// "-h solver lanczos" / "-h order 4": the index name plus a member selects the
	// member's page. order2..order22 are real keys, so only the solver prefix is cut.
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
  adcgo -fcidump FILE (-dip | -sip [-order N]) [-solver S] [options] [-out FILE]
  adcgo -h TOPIC                    detail on one method, order, solver or knob
  adcgo -h all                      every flag, unabridged
  adcgo -h topics                   list every help topic

METHODS                                                        detail: adcgo -h ...
  -dip                    double ionization, DIP-ADC(2)                    dip
  -sip                    single ionization, IP-ADC(n)                     sip
  -order 2|3|4|22         ADC scheme: (2)x, (3), CVS Dyson (4), (2,2)      order
  -fano -fano-init K      decay width Gamma and lifetime tau               fano

SOLVERS  -solver S        same matrix, same answers — pick by size and band
  lanczos       (default) whole-band block-Lanczos, -blocks N              lanczos
  lanczos-lowmem          same band, few panels resident — the GPU mode    lowmem
  davidson                converge the lowest -nroots exactly              davidson
  dense                   full LAPACK diagonalization, every state         dense

OUTPUT                    default: solver document JSON (E, ps, configs)
  -spectrum               decay channels (DIP + -mo) or per orbital (SIP)  spectrum
  -bare                   one stick per state: energy + pole strength      spectrum
  -tdm                    transition dipoles, Dyson photoionization        tdm
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
