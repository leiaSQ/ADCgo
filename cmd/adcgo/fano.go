package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fano"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
	"github.com/leiaSQ/ADCgo/internal/adc/stieltjes"
)

// fano.go — the -fano driver: an electronic decay width by the Fano/Feshbach method with
// Stieltjes imaging, over any of the SIP secular matrices (ADC(2)x, ADC(3), ADC(2,2)m/x/f).
//
// The pipeline, each stage in its own package:
//
//	sip.NewSpace / NewSpace22        the configuration space
//	fano.NewPartition                Q (bound) / P (continuum), by hole localization
//	sip.Space.Restrict               QMQ and PMP as ordinary ADC matrices
//	solve QMQ -> fano.SelectDiscrete |Phi>, E_Phi
//	fano.Coupling                    g = P(M - E_Phi)|Phi>, one PARENT mat-vec
//	solve PMP -> fano.Widths         the discrete pseudo-continuum {eps_i, gamma_i}
//	fano.ImageWidth                  Gamma(E_Phi) by Stieltjes imaging
//	fano.PartialWidths/ImagePartial   per-channel widths, with -mo

// hartreeToEV converts hartree to eV, derived from fano's own meV constant so the two
// cannot drift.
const hartreeToEV = fano.HartreeToMeV / 1000

// FanoDocument is the -fano output.
type FanoDocument struct {
	NORB   int    `json:"norb"`
	NELEC  int    `json:"nelec"`
	Family string `json:"family"` // "sip", "dip" (ADC), or "khci" (k-hole CI)
	Order  int    `json:"order"`
	// Spin is the DIP spin sector ("singlet" or "triplet"); empty for SIP.
	Spin string `json:"spin,omitempty"`
	// K is the main-class hole count of a khci run; TwoMs its 2*Ms sector.
	K     int `json:"k,omitempty"`
	TwoMs int `json:"two_ms,omitempty"`
	// Census is the Q/P count per excitation class (fano.Partition.CensusString).
	Census string `json:"census,omitempty"`
	// Variant is the ADC(2,2) scheme ("m", "x" or "f"); empty for order 2/3.
	Variant string `json:"variant,omitempty"`
	Scheme  string `json:"scheme"` // Q/P selection scheme: "a" (hole localization)

	Vacancy   int    `json:"vacancy"`   // 0-based occupied orbital of the initial hole
	Irrep     int    `json:"irrep"`     // 1-based target irrep (SIP: fixed by the vacancy; DIP: -sym)
	Criterion string `json:"criterion"` // the Q/P predicate, as applied

	// The partition and the two sub-problems.
	Size   int `json:"size"`    // configurations in the parent space
	QSize  int `json:"q_size"`  // bound subspace
	PSize  int `json:"p_size"`  // continuum subspace
	QMain  int `json:"q_main"`  // main-class configurations in Q
	PMain  int `json:"p_main"`  // main-class configurations in P
	PDecay int `json:"p_decay"` // decay-continuum configurations in P (all satellite classes)
	Class3 int `json:"class_3h2p,omitempty"`

	// The discrete state.
	EPhiHartree float64 `json:"e_phi_hartree"`
	EPhiEV      float64 `json:"e_phi_ev"`
	PhiWeight   float64 `json:"phi_weight"` // its weight on the vacancy's 1h configuration
	PhiRoot     int     `json:"phi_root"`   // which QMQ root (-1: an interior state, -fano-phi-holes)
	// PhiResidual is ||(QHQ - E_Phi) Phi|| of an interior state and PhiLadder its value
	// before and after each polish step; PhiConverged says whether -fano-phi-tol was met.
	// The coupling vector's noise floor scales with this residual squared.
	PhiResidual  float64   `json:"phi_residual,omitempty"`
	PhiLadder    []float64 `json:"phi_residual_ladder,omitempty"`
	PhiConverged bool      `json:"phi_converged,omitempty"`
	// C4Audit lists the Q Ritz pairs within 0.5 eV of E_Phi from the final search space:
	// the states the discrete state could be mixed or confused with.
	C4Audit []FanoAudit `json:"c4_audit,omitempty"`
	// PhiAlternatives are other converged Q states with target weight (SelectInterior
	// candidates); PhiAmbiguous is set when one carries >= 80% of the selected weight,
	// i.e. charge resonance splits the target character and the selection can switch.
	PhiAlternatives []FanoAudit `json:"phi_alternatives,omitempty"`
	PhiAmbiguous    bool        `json:"phi_ambiguous,omitempty"`
	// PhiClassWeights is |Phi>'s squared weight per excitation class ("4h", "5h1p", ...):
	// how much of the decaying state lives beyond its main class.
	PhiClassWeights map[string]float64 `json:"phi_class_weights,omitempty"`
	// EvalEV is the energy Gamma was evaluated at when -fano-at overrode E_Phi.
	EvalEV float64 `json:"eval_ev,omitempty"`

	// The pseudo-continuum.
	CoupledChannels int `json:"coupled_channels"` // P configurations with nonzero g
	// Decomposition is the ordering decomposition (-fano-decompose): the width each Q
	// class of |Phi> carries alone, and the interference of each pair.
	Decomposition []FanoPiece `json:"decomposition,omitempty"`
	// Lambda is the transfer scaling of a lock-in run (-fano-lambda), as given.
	Lambda string `json:"lambda,omitempty"`
	// Engine is the width engine when not the eigen path (gauss, shiftinvert, kpm).
	Engine      string  `json:"engine,omitempty"`
	States      int     `json:"pseudo_states"`
	SumGamma    float64 `json:"sum_gamma_hartree2"` // zeroth moment, NOT a width
	SumRule     float64 `json:"sum_rule_hartree2"`  // 2 pi ||g||^2
	SumResidual float64 `json:"sum_rule_residual"`  // (rule - sum)/rule
	DroppedE    int     `json:"dropped_energy"`
	DroppedW    int     `json:"dropped_weight"`
	DroppedG    int     `json:"dropped_gamma"`

	// The width.
	WidthMeV      float64 `json:"width_mev"`
	WidthSigmaMeV float64 `json:"width_sigma_mev"`
	WidthEV       float64 `json:"width_ev"`
	LifetimeFS    float64 `json:"lifetime_fs"`

	// The pseudo-continuum's energy span, and where E_Phi sits inside it. PhiPosition is
	// (E_Phi - Emin)/(Emax - Emin): imaging is a reconstruction from samples, so a value
	// near 0 or 1 means it is working at the edge of its own data, which the reference
	// flags as "large inaccuracy expected" and is the difference between a width and an
	// extrapolation.
	PseudoEMinEV float64 `json:"pseudo_emin_ev"`
	PseudoEMaxEV float64 `json:"pseudo_emax_ev"`
	PhiPosition  float64 `json:"phi_position"`

	// Stieltjes diagnostics.
	StieltjesMaxOrder int   `json:"stieltjes_max_order"`
	StieltjesOrders   []int `json:"stieltjes_orders_used"`
	StieltjesLowOrder bool  `json:"stieltjes_low_order,omitempty"`
	// Orders that fell outside their OWN density grid and so returned a fallback rather
	// than an interpolation: the crude 0.5 g_1/e_1 estimate below the first midpoint, or
	// zero above the last. Either one folded into the order average corrupts it, and the
	// reference folds them in silently.
	StieltjesBelow int `json:"stieltjes_below,omitempty"`
	StieltjesAbove int `json:"stieltjes_above,omitempty"`

	// Partial widths (needs -mo). Approximate, per the paper.
	PartialWidths []FanoChannel `json:"partial_widths,omitempty"`
	PartialNote   string        `json:"partial_widths_note,omitempty"`

	Timing map[string]string `json:"timing,omitempty"`
}

// FanoPiece is one entry of the ordering decomposition: a class's own width, or (with
// Interference set) Gamma[g_A + g_B] - Gamma[g_A] - Gamma[g_B] for a pair.
type FanoPiece struct {
	Name         string  `json:"name"`
	WidthMeV     float64 `json:"width_mev"`
	SigmaMeV     float64 `json:"width_sigma_mev"`
	Norm2        float64 `json:"g_norm2,omitempty"`
	Interference bool    `json:"interference,omitempty"`
}

// FanoAudit is one Q state near E_Phi (a Ritz pair; Residual says how good).
type FanoAudit struct {
	EnergyEV     float64            `json:"energy_ev"`
	Residual     float64            `json:"residual"`
	TargetWeight float64            `json:"target_weight"`
	Classes      map[string]float64 `json:"classes"`
	Selected     bool               `json:"selected,omitempty"`
}

// FanoChannel is one decay channel's partial width.
type FanoChannel struct {
	Name     string  `json:"name"`
	WidthMeV float64 `json:"width_mev,omitempty"`
	SigmaMeV float64 `json:"width_sigma_mev,omitempty"`
	Ratio    float64 `json:"branching_ratio,omitempty"`
	Share    float64 `json:"strength_share"`
	Note     string  `json:"note,omitempty"` // why this channel could not be imaged
}

type fanoConfig struct {
	sip sipConfig // backend, solver, matfree, sigma, caches — all shared with -sip

	// dip selects double ionization (-dip -fano): the parent is the DIP-ADC(2) 2h | 3h1p
	// space of one spin sector and irrep. The fields below that name the ADC(2,2) variant
	// or the self-energy do not apply to it.
	dip  bool
	spin dip.Spin

	// khci > 0 selects the k-hole CI family (-khci K): the parent is a khci.Space
	// over kh | (k+1)h1p | (k+2)h2p and the matrix H - E_HF (khci.CI). -mo is then the
	// labelled sidecar (orb_kind / orb_atom), which may describe localized orbitals, and
	// -fano-qp accepts the net-charge grammar of fano.ParseConfigRule.
	khci         int
	khciMaxClass int   // 0 = K+2
	khciTwoMs    int   // 2*Ms of the N-K electron states
	khciMaxFree  int   // < 0: no limit; else at most this many free particles in (K+2)h2p
	khciCSR      int64 // byte budget for materializing each Q/P operator (0 = never)

	variant sip.Variant
	vacancy int   // -fano-init
	qOrbs   []int // -fano-q (empty = just the vacancy)
	rule    fano.HoleRule
	qpSpec  string // -fano-qp: a per-excitation-class Q/P rule, overriding -fano-q/-fano-rule
	nth     int    // -fano-nth: which qualifying QMQ root
	// phiHoles (-fano-phi-holes) selects |Phi> as the interior QMQ root heaviest on the
	// main-class configuration with these holes; phiTol is its residual goal.
	phiHoles []int
	phiTol   float64
	// engine (-fano-engine) is eigen (the PMP eigenvector path, default), gauss,
	// shiftinvert or kpm; order is the Lanczos length / KPM moment count of the latter
	// three, siShift the shift-invert offset from E_Phi (hartree).
	engine  string
	order   int
	siShift float64
	gminRel float64 // -fano-gmin-rel, the eigen path's relative gamma floor

	// The lock-in: lambdaSpec (-fano-lambda "A:B=0.5,C:D=0") scales the transfer
	// integrals of atom pairs (khci with labels); saveG / imageG write a coupling vector
	// or image one made elsewhere; decompose images the coupling per Q class and their
	// pairwise interference.
	lambdaSpec    string
	saveG, imageG string
	decompose     bool
	qMin          float64 // -fano-qmin: minimum |Phi> weight on the vacancy configuration
	qRoots        int     // -fano-qroots: QMQ roots to converge
	qSolver       string  // -fano-qsolver: solver for QMQ ("" = davidson, or dense when -solver dense)

	emax, emaxRel   float64 // final-state energy ceiling: absolute (Eh) or a multiple of E_Phi
	wmin, gmin      float64 // final-state cuts
	pBlock, pBlocks int     // PMP Krylov block width and block count
	atEV            float64 // -fano-at: evaluate Gamma here instead of at E_Phi (eV)

	stOrderLo, stOrderHi int
	stPrec               uint
	stAverage            stieltjes.AverageMode
	stWindow             int

	initSite string          // -init-atom, the initially ionized site for channel labels
	sites    []spectrum.Site // -group
	specOpts spectrum.Options
}

// parseOrbitalList parses a comma-separated list of 0-based occupied orbital indices.
func parseOrbitalList(flagName, s string) ([]int, error) {
	if strings.TrimSpace(s) == "" {
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
			return nil, fmt.Errorf("bad %s orbital %q (want a 0-based occupied index)", flagName, f)
		}
		out = append(out, v)
	}
	return out, nil
}

// parseHoleRule maps the -fano-rule flag onto the two readings of the scheme A predicate.
func parseHoleRule(s string) (fano.HoleRule, error) {
	switch s {
	case "any", "":
		return fano.AnyHole, nil
	case "all":
		return fano.AllHoles, nil
	}
	return fano.AnyHole, fmt.Errorf("bad -fano-rule %q (want any or all)", s)
}

// parseStieltjesOrders parses "-stieltjes-orders lo-hi", "lo-" or "" (both ends automatic).
func parseStieltjesOrders(s string) (lo, hi int, err error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, 0, nil
	}
	parts := strings.SplitN(s, "-", 2)
	if parts[0] != "" {
		if lo, err = strconv.Atoi(parts[0]); err != nil || lo < 1 {
			return 0, 0, fmt.Errorf("bad -stieltjes-orders lower bound %q", parts[0])
		}
	}
	if len(parts) == 2 && parts[1] != "" {
		if hi, err = strconv.Atoi(parts[1]); err != nil || hi < 1 {
			return 0, 0, fmt.Errorf("bad -stieltjes-orders upper bound %q", parts[1])
		}
	}
	if lo > 0 && hi > 0 && hi < lo {
		return 0, 0, fmt.Errorf("-stieltjes-orders %q has its bounds the wrong way round", s)
	}
	return lo, hi, nil
}

// parseAverageMode maps -stieltjes-average onto the two order-averaging protocols.
func parseAverageMode(s string) (stieltjes.AverageMode, error) {
	switch s {
	case "paper", "":
		return stieltjes.AveragePaper, nil
	case "reference":
		return stieltjes.AverageReference, nil
	}
	return stieltjes.AveragePaper, fmt.Errorf("bad -stieltjes-average %q (want paper or reference)", s)
}

// runFano computes one decay width.
func runFano(d *fcidump.Data, cfg fanoConfig) error {
	if !cfg.dip && cfg.khci == 0 && cfg.sip.order != 2 && cfg.sip.order != 3 && cfg.sip.order != sip.Order22 {
		return fmt.Errorf("-fano supports -order 2 (Fano-ADC(2)x), 3, or %d (Fano-ADC(2,2)); "+
			"got %d. Order 4 is the CVS Dyson scheme, which has no Fano path",
			sip.Order22, cfg.sip.order)
	}
	nocc := mp.NOcc(d)
	if cfg.vacancy < 0 || cfg.vacancy >= nocc {
		return fmt.Errorf("-fano-init %d is not an occupied orbital (this system has %d)",
			cfg.vacancy, nocc)
	}
	if err := validateSolver(cfg.sip.solver); err != nil {
		return err
	}
	if cfg.dip && cfg.sip.moPath != "" {
		// PartialWidths routes each final state through its two-hole (dication) site
		// populations, which is the decay class of single ionization (2h1p). The DIP decay
		// class is 3h1p, a trication, and spectrum.Classify has no three-site routing, so
		// a partial width here would be assigned to the wrong channels.
		return fmt.Errorf("-dip -fano has no partial widths: channel routing is defined for " +
			"two-hole final states (SIP) only; drop -mo")
	}
	timing := map[string]string{}
	stage := func(name string, run func() error) error {
		t0 := time.Now()
		fmt.Fprintf(os.Stderr, "adcgo: fano: %s ...\n", name)
		err := run()
		d := time.Since(t0).Round(time.Millisecond)
		timing[name] = d.String()
		fmt.Fprintf(os.Stderr, "adcgo: fano: %s done in %s\n", name, d)
		return err
	}

	eps := mp.OrbitalEnergies(d, nocc)
	ch, err := newChooser(cfg.sip.backend, cfg.sip.profile, cfg.sip.gpus)
	if err != nil {
		return err
	}

	orbSym, syms, err := selectSymmetry(cfg.sip.sym, d)
	if err != nil {
		return err
	}
	ints := integrals.New(d, nocc, orbSym)

	var (
		fam       fanoFamily
		parent    fano.Space
		targetSym int
		class3    int
	)
	switch {
	case cfg.khci > 0:
		if len(syms) != 1 {
			return fmt.Errorf("-khci -fano needs one sector: -sym none or a 0-based irrep index, not %q",
				cfg.sip.sym)
		}
		targetSym = syms[0]
		var md *mo.Data
		if cfg.sip.moPath != "" {
			// CI is invariant to occupied-occupied and virtual-virtual rotations, so the
			// sidecar may describe localized orbitals.
			if md, err = mo.ReadFile(cfg.sip.moPath); err != nil {
				return err
			}
		}
		opts := khci.Options{K: cfg.khci, NOcc: nocc, NVir: d.NORB - nocc, TwoMs: cfg.khciTwoMs,
			TargetIrrep: targetSym, MaxClass: cfg.khciMaxClass}
		if orbSym != nil {
			opts.OrbSym = make([]int, d.NORB)
			for p, g := range orbSym {
				opts.OrbSym[p] = g - 1
			}
		}
		if cfg.khciMaxFree >= 0 {
			if md == nil || !md.HasLabels {
				return fmt.Errorf("-khci-maxfree needs -mo with orbital labels (orb_kind): " +
					"free and compact virtuals are told apart by the sidecar")
			}
			opts.VirKind = make([]khci.VirKind, d.NORB-nocc)
			for a := range opts.VirKind {
				if md.OrbKind[nocc+a] == mo.OrbFree {
					opts.VirKind[a] = khci.Free
				}
			}
			opts.LimitFree, opts.MaxFreeTop = true, cfg.khciMaxFree
		}
		ksp, err := khci.NewSpace(opts)
		if err != nil {
			return err
		}
		if ksp.MainBlockSize() == 0 {
			return fmt.Errorf("the sector has no %dh configurations", cfg.khci)
		}
		if cfg.lambdaSpec != "" {
			// the lock-in's transfer scaling, on a private copy of the integrals
			if md == nil || !md.HasLabels {
				return fmt.Errorf("-fano-lambda needs -mo with orbital labels: the transfer pairs are atoms")
			}
			if d, err = scaleTransfers(d, md, cfg.lambdaSpec); err != nil {
				return err
			}
		}
		op, err := khci.NewCI(ksp, d, backend.Gonum{})
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "adcgo: fano: khci %s; E_HF = %.10f Eh (electronic)\n", ksp, op.EHF())
		fam, parent = khciFamily{parent: ksp, op: op, csrBytes: cfg.khciCSR, md: md, nocc: nocc}, ksp
	case cfg.dip:
		// A 2h discrete state's irrep is the product of its two holes, so one orbital does
		// not fix it: -sym names the sector (a single index, or none).
		if len(syms) != 1 {
			return fmt.Errorf("-dip -fano needs one sector: -sym none or a 0-based irrep index, not %q",
				cfg.sip.sym)
		}
		targetSym = syms[0]
		dsp := dip.NewSpace(nocc, d.NORB, orbSym, targetSym, cfg.spin)
		fam, parent = dipFamily{parent: dsp, ints: ints, eps: eps, cfg: cfg.sip}, dsp
		if dsp.MainBlockSize() == 0 {
			return fmt.Errorf("the %s sector of irrep %d has no 2h configurations", spinName(cfg.spin), targetSym+1)
		}
	default:
		// The vacancy fixes the target irrep: |Phi> has the 1h configuration's symmetry, and
		// the secular matrix is block diagonal in irreps, so no other sector can contribute to
		// its width. -sym is therefore not a choice here; it is determined.
		if orbSym != nil {
			targetSym = orbSym[cfg.vacancy] - 1
		}
		if cfg.sip.sig, err = buildSigma(cfg.sip, ints, eps, nocc, d.NORB); err != nil {
			return err
		}
		var ssp *sip.Space
		if cfg.sip.order == sip.Order22 {
			ssp = sip.NewSpace22(nocc, d.NORB, orbSym, targetSym)
		} else {
			ssp = sip.NewSpace(nocc, d.NORB, orbSym, targetSym)
		}
		if ssp.MainBlockSize() == 0 {
			return fmt.Errorf("irrep %d has no 1h configurations, so orbital %d cannot be the "+
				"initial vacancy there", targetSym+1, cfg.vacancy)
		}
		fam, parent = sipFamily{parent: ssp, ints: ints, eps: eps, cfg: cfg.sip, v: cfg.variant}, ssp
		class3 = len(ssp.Sat3)
	}

	// Scheme A: the Q predicate. With no -fano-q the set is the vacancy itself, which with
	// the default "any" rule is the Auger criterion — Q is every configuration that still
	// carries the initial hole, P every one that has filled it.
	var sel fano.Selector
	if kf, ok := fam.(khciFamily); ok && cfg.qpSpec != "" && kf.md != nil && kf.md.HasLabels {
		// The net-charge partition (fano.ParseConfigRule): a P such as "every atom +1 and
		// one free particle" cannot be expressed by a hole list. The grammar also takes hole terms.
		am, err := fano.AtomMapFromMO(kf.md, nocc)
		if err != nil {
			return err
		}
		cr, err := fano.ParseConfigRule(cfg.qpSpec, "initial state", am)
		if err != nil {
			return err
		}
		sel = cr
	} else if cfg.qpSpec != "" {
		// -fano-qp states the partition per excitation class, which is what Mg(2s^-1) and
		// Kr(3d^-1) need: their rules distinguish 2h1p from 3h2p and count holes rather
		// than test membership. It supersedes -fano-q/-fano-rule entirely.
		cr, err := fano.ParseClassRule(cfg.qpSpec, "initial vacancy")
		if err != nil {
			return err
		}
		sel = cr
	} else {
		qOrbs := cfg.qOrbs
		if len(qOrbs) == 0 {
			qOrbs = []int{cfg.vacancy}
		}
		hl, err := fano.NewHoleLocalization(cfg.rule, qOrbs, "initial vacancy")
		if err != nil {
			return err
		}
		sel = hl
	}
	part := fano.NewPartition(parent, sel)
	if err := part.Validate(); err != nil {
		return err
	}
	qsp, err := fam.restrict(part.Q)
	if err != nil {
		return err
	}
	psp, err := fam.restrict(part.P)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "adcgo: fano: irrep %d, %s\n", targetSym+1, part)
	fmt.Fprintf(os.Stderr, "adcgo: fano: Q/P census %s\n", part.CensusString())

	doc := FanoDocument{
		NORB: d.NORB, NELEC: d.NELEC, Family: "sip", Order: cfg.sip.order, Scheme: "a",
		Vacancy: cfg.vacancy, Irrep: targetSym + 1, Criterion: sel.String(),
		Size: parent.Size(), QSize: part.QSize(), PSize: part.PSize(),
		QMain: part.QMain, PMain: psp.MainBlockSize(), Class3: class3,
	}
	doc.Census = part.CensusString()
	doc.Lambda = cfg.lambdaSpec
	switch {
	case cfg.khci > 0:
		// CI is first order in every block (the ISR's first order apart from B02)
		doc.Family, doc.Order, doc.K, doc.TwoMs = "khci", 1, cfg.khci, cfg.khciTwoMs
	case cfg.dip:
		doc.Family, doc.Order, doc.Spin = "dip", 2, spinName(cfg.spin)
	case cfg.sip.order == sip.Order22:
		doc.Variant = cfg.variant.String()
	}

	// --- QMQ: the discrete state. ---
	//
	// The lowest roots are the right ones to converge, which is not obvious and is worth
	// recording: every configuration in Q carries the initial vacancy, and every one beyond
	// the 1h class carries an ADDITIONAL hole, so it lies higher. |Phi> is therefore the
	// bottom of the QMQ spectrum, and Davidson's lowest-roots bias is exactly what is wanted
	// even for a deep core hole whose absolute energy is the highest in the full spectrum.
	// The two sub-problems want different solvers, so QMQ gets its own. Only a few of its
	// LOWEST roots are wanted, which is Davidson's case exactly and not block-Lanczos':
	// under this partition Q's main block is often a single configuration, so a Lanczos
	// block seeded from it is one column wide.
	qSolver := cfg.qSolver
	if qSolver == "" {
		qSolver = "davidson"
		if cfg.sip.solver == "dense" {
			qSolver = "dense" // the small-case oracle: use it for both halves
		}
	}
	if err := validateSolver(qSolver); err != nil {
		return fmt.Errorf("-fano-qsolver: %w", err)
	}

	var (
		phi    fano.Discrete
		g      []float64
		pieces map[string][]float64
	)
	if cfg.imageG != "" {
		// -fano-image-g: image a coupling vector made elsewhere (a lock-in double
		// difference g_AB = g(1,1) - g(1,0) - g(0,1) + g(0,0)) against this run's PMP.
		var e float64
		if g, e, err = loadCoupling(cfg.imageG, part.PSize()); err != nil {
			return err
		}
		phi = fano.Discrete{Energy: e, Root: -1}
		doc.EPhiHartree, doc.EPhiEV, doc.PhiRoot = e, e*hartreeToEV, -1
		fmt.Fprintf(os.Stderr, "adcgo: fano: imaging the coupling vector of %s at E = %.6f Eh\n",
			cfg.imageG, e)
	} else {
		if cfg.phiHoles != nil {
			// -fano-phi-holes: |Phi> is an INTERIOR root of QMQ, picked by its weight on the
			// configuration with these holes and polished to -fano-phi-tol, not the nth
			// qualifying root from the bottom.
			err = stage(fmt.Sprintf("QMQ interior state (%d configurations)", qsp.Size()), func() error {
				qmx, err := fam.matrix(ch, "fano QMQ", qsp)
				if err != nil {
					return err
				}
				defer qmx.Release()
				be := qmx.Backend()
				dv := qmx.Diagonal(be)
				diag := be.Download(dv)
				be.Free(dv)
				targets := fano.TargetRows(qsp, cfg.phiHoles)
				if len(targets) == 0 {
					return fmt.Errorf("-fano-phi-holes %v: no main-class configuration of Q has these holes "+
						"(is it in P under the partition?)", cfg.phiHoles)
				}
				ir, err := fano.SelectInterior(qsp, qmx, diag, be, fano.InteriorOptions{
					Targets: targets, Tol: cfg.phiTol,
					Log: func(s string) { fmt.Fprintf(os.Stderr, "adcgo: fano: %s\n", s) },
				})
				if err != nil {
					return err
				}
				phi = ir.Discrete
				doc.PhiResidual, doc.PhiLadder, doc.PhiConverged = ir.Residual, ir.Ladder, ir.Converged
				doc.PhiAmbiguous = ir.Ambiguous
				for _, a := range ir.Alternatives {
					doc.PhiAlternatives = append(doc.PhiAlternatives, FanoAudit{EnergyEV: a.Energy * hartreeToEV,
						Residual: a.Residual, TargetWeight: a.TargetWeight, Classes: a.Classes})
				}
				for _, a := range ir.Audit {
					doc.C4Audit = append(doc.C4Audit, FanoAudit{EnergyEV: a.Energy * hartreeToEV,
						Residual: a.Residual, TargetWeight: a.TargetWeight, Classes: a.Classes, Selected: a.Selected})
				}
				return nil
			})
		} else {
			err = stage(fmt.Sprintf("QMQ solve (%d configurations, %s)", qsp.Size(), qSolver), func() error {
				qcfg := cfg.sip
				qcfg.nroots = cfg.qRoots
				qcfg.solver = qSolver
				res, qmx, err := solveFanoSpace(ch, "fano QMQ", fam, qsp, qcfg, lanczos.Options{}, true)
				if err != nil {
					return err
				}
				defer qmx.Release()
				phi, err = fano.SelectDiscrete(qsp, res, cfg.vacancy, cfg.nth, cfg.qMin)
				return err
			})
		}
		if err != nil {
			return err
		}
		doc.EPhiHartree, doc.EPhiEV = phi.Energy, phi.Energy*hartreeToEV
		doc.PhiWeight, doc.PhiRoot = phi.Weight, phi.Root
		doc.PhiClassWeights = fano.ClassWeights(qsp, phi.Vec)
		fmt.Fprintf(os.Stderr, "adcgo: fano: E_Phi = %.6f Eh = %.4f eV (QMQ root %d, weight %.4f)\n",
			phi.Energy, doc.EPhiEV, phi.Root, phi.Weight)

		// --- The coupling vector, from ONE parent mat-vec. ---
		err = stage("coupling vector", func() error {
			pmx, err := fam.matrix(ch, "fano parent", parent)
			if err != nil {
				return err
			}
			defer pmx.Release()
			g, err = fano.Coupling(pmx, part, phi, pmx.Backend())
			if err != nil || !cfg.decompose {
				return err
			}
			// the ordering decomposition: g = sum over Q classes of g_B
			pieces, err = fano.Decompose(pmx, part, phi, fano.ClassBlocks(qsp), pmx.Backend())
			return err
		})
		if err != nil {
			return err
		}
	}
	if cfg.saveG != "" {
		if err := saveCoupling(cfg.saveG, g, phi.Energy); err != nil {
			return err
		}
	}
	doc.SumRule = fano.SumRule(g)
	nCoupled, gMax := fano.CoupledChannels(g, 1e-6)
	doc.CoupledChannels = nCoupled
	fmt.Fprintf(os.Stderr, "adcgo: fano: %d of %d P configurations carry coupling above 1e-6 of "+
		"the largest (|g|max = %.4g Eh); 2pi||g||^2 = %.6g Eh^2\n",
		nCoupled, len(g), gMax, doc.SumRule)
	if nCoupled < 40 {
		fmt.Fprintf(os.Stderr, "adcgo: fano: WARNING only %d coupled channels. Stieltjes imaging "+
			"reconstructs a density from these alone, so the width will carry a large order "+
			"spread whatever else is right. The count is set by the basis: it is roughly the "+
			"number of final dication configurations times the discretized continuum orbitals "+
			"per channel, so a larger/harder virtual space is what increases it.\n", nCoupled)
	}

	if pieces != nil {
		if err := imageDecomposition(&doc, cfg, fam, ch, psp, pieces, phi, stage); err != nil {
			return err
		}
	}
	if cfg.engine != "" && cfg.engine != "eigen" {
		return runFanoEngine(&doc, cfg, fam, ch, psp, g, phi, timing, stage)
	}

	// --- PMP: the pseudo-continuum. ---
	//
	// Seeded with the most strongly coupled P configurations (the reference's fill_stvc) at
	// an explicit block width, because P's own main block is a poor choice: under this
	// partition it holds only the handful of 1h configurations the Q criterion did not
	// claim, and a Krylov block of one or two needs hundreds of iterations. The width is
	// nearly free when the operator is matrix-free — one element recompute serves every
	// column of the block.
	pBlock := cfg.pBlock
	if pBlock <= 0 {
		pBlock = min(64, psp.Size())
	}
	blocks := cfg.pBlocks
	if blocks <= 0 {
		blocks = cfg.sip.blocks
	}
	var ps *fano.Pseudo
	err = stage(fmt.Sprintf("PMP solve (%d configurations, block width %d)", psp.Size(), pBlock),
		func() error {
			lopts := lanczos.Options{
				MaxBlocks: blocks, WantFull: true,
				Block:     pBlock,
				StartRows: fano.StartRows(g, pBlock),
			}
			res, pmx, err := solveFanoSpace(ch, "fano PMP", fam, psp, cfg.sip, lopts, true)
			if err != nil {
				return err
			}
			defer pmx.Release()
			// The energy ceiling, as a multiple of E_Phi unless given absolutely. Scaling
			// it is what makes the reference's constant transferable: its purpose is to
			// drop states far ABOVE the decaying state, which contribute nothing to
			// Gamma(E_Phi) but dominate the moments the reconstruction is built from, and
			// "far above" is only meaningful relative to E_Phi.
			emax := cfg.emax
			if emax <= 0 && cfg.emaxRel > 0 {
				emax = cfg.emaxRel * phi.Energy
				fmt.Fprintf(os.Stderr, "adcgo: fano: energy ceiling %.4g Eh (%.3g x E_Phi)\n",
					emax, cfg.emaxRel)
			}
			ps, err = fano.Widths(psp, res, g, fano.Filter{
				EMax: emax, MinWeight: cfg.wmin, MinGamma: cfg.gmin, MinGammaRel: cfg.gminRel,
			})
			if err != nil {
				return err
			}
			return imageAndClassify(&doc, cfg, psp, res, ps, phi, nocc)
		})
	if err != nil {
		return err
	}
	doc.States = len(ps.Energy)
	doc.SumGamma, doc.SumResidual = ps.SumGamma, ps.Residual()
	doc.DroppedE, doc.DroppedW, doc.DroppedG = ps.DroppedEnergy, ps.DroppedWeight, ps.DroppedGamma
	doc.PDecay = ps.DecayRows
	doc.Timing = timing
	fmt.Fprintf(os.Stderr, "adcgo: fano: %s\n", ps)
	fmt.Fprintf(os.Stderr, "adcgo: fano: gamma removed by the cuts: energy %.3g, weight %.3g, "+
		"gamma floor %.3g (threshold %.3g) of 2pi||g||^2 = %.3g Eh^2\n",
		ps.LostEnergy, ps.LostWeight, ps.LostToGamma, ps.GammaThreshold, ps.SumRule)
	// The sum-rule residual is the one number that says whether the pseudo-continuum
	// actually sampled the coupling. It is exactly zero for a complete basis, so anything
	// large means the Krylov space was truncated before it captured the coupling — and an
	// unconverged width can still look perfectly reasonable, with a small Stieltjes spread,
	// because the orders agree with each other about the wrong density.
	if r := ps.Residual(); r > 0.02 {
		fmt.Fprintf(os.Stderr, "adcgo: fano: WARNING the pseudo-continuum captures only %.1f%% "+
			"of the sum rule 2pi||g||^2 (residual %.1f%%). The width below is NOT converged. "+
			"Raise -fano-blocks (the Krylov space is %d blocks of %d) or loosen -fano-wmin; "+
			"%.3g of %.3g Eh^2 was carried off by the cuts.\n",
			100*(1-r), 100*r, blocks, pBlock, ps.LostGamma, ps.SumRule)
	}
	fmt.Fprintf(os.Stderr, "adcgo: fano: Gamma = %.4f +/- %.4f meV, tau = %.4g fs\n",
		doc.WidthMeV, doc.WidthSigmaMeV, doc.LifetimeFS)
	return emitJSON(doc, cfg.sip.out)
}

// runFanoEngine is the PMP half without PMP eigenvectors: Gamma from the
// coupling vector and applications of PMP only, by the engine -fano-engine names.
//
//	gauss        W-B, Gauss rules of every order from -fano-order Lanczos steps seeded by g
//	shiftinvert  W-B2, the same on (PMP - sigma)^-1, sigma = E + -fano-si-shift
//	kpm          W-C, -fano-order Chebyshev moments with a Jackson kernel
//
// None has a final-state cut (the pseudo-continuum is never formed), so the result is
// homogeneous in g, which is what widths of 1e-15 Eh need.
func runFanoEngine(doc *FanoDocument, cfg fanoConfig, fam fanoFamily, ch *chooser, psp fano.Space,
	g []float64, phi fano.Discrete, timing map[string]string,
	stage func(string, func() error) error) error {

	m := cfg.order
	if m <= 0 {
		m = 200
	}
	at := phi.Energy
	if cfg.atEV != 0 {
		at = cfg.atEV / hartreeToEV
		doc.EvalEV = cfg.atEV
	}
	doc.Engine = cfg.engine
	err := stage(fmt.Sprintf("PMP %s engine (%d configurations, order %d)", cfg.engine, psp.Size(), m),
		func() error {
			pmx, err := fam.matrix(ch, "fano PMP", psp)
			if err != nil {
				return err
			}
			defer pmx.Release()
			be := pmx.Backend()
			n := psp.Size()
			out := be.Alloc(n)
			defer be.Free(out)
			apply := func(dst, src []float64) {
				v := be.Upload(src)
				pmx.ApplyFull(out, v)
				be.Free(v)
				copy(dst, be.Download(out))
			}
			var w fano.Width
			switch cfg.engine {
			case "gauss":
				w, _, err = fano.GaussWidth(apply, g, m, at, cfg.stieltjesOptions())
			case "shiftinvert":
				dv := pmx.Diagonal(be)
				diag := be.Download(dv)
				be.Free(dv)
				w, err = fano.ShiftInvertWidth(apply, g, m, at,
					fano.ShiftInvertOptions{Sigma: at + cfg.siShift, Diag: diag}, cfg.stieltjesOptions())
			case "kpm":
				lo, hi, e := fano.SpectralBounds(apply, g, min(80, n), 0.05)
				if e != nil {
					return e
				}
				var gam float64
				gam, err = fano.KPMWidth(apply, g, at, m, lo, hi)
				w = fano.NewWidth(gam, 0)
			default:
				return fmt.Errorf("unknown -fano-engine %q (want eigen, gauss, shiftinvert or kpm)", cfg.engine)
			}
			if err != nil {
				return err
			}
			doc.WidthMeV, doc.WidthSigmaMeV = w.MeV, w.SigmaMeV
			doc.WidthEV, doc.LifetimeFS = w.Gamma*hartreeToEV, w.Tau
			if w.Stieltjes != nil {
				doc.StieltjesMaxOrder = w.Stieltjes.MaxOrder
				doc.StieltjesOrders = w.Stieltjes.Used
				doc.StieltjesLowOrder = w.Stieltjes.LowOrder
				doc.StieltjesBelow, doc.StieltjesAbove = w.Stieltjes.Boundary()
			}
			return nil
		})
	if err != nil {
		return err
	}
	doc.Timing = timing
	if doc.StieltjesBelow+doc.StieltjesAbove > 0 {
		fmt.Fprintf(os.Stderr, "adcgo: fano: WARNING %d order(s) fell below and %d above their own "+
			"density grid: the evaluation energy is at the edge of the coupled pseudo-continuum, "+
			"and those orders returned the fallback, not an interpolation. Compare -fano-engine kpm, "+
			"which has no such fallback.\n", doc.StieltjesBelow, doc.StieltjesAbove)
	}
	fmt.Fprintf(os.Stderr, "adcgo: fano: Gamma = %.6g +/- %.3g meV (%s engine), tau = %.4g fs\n",
		doc.WidthMeV, doc.WidthSigmaMeV, cfg.engine, doc.LifetimeFS)
	return emitJSON(*doc, cfg.sip.out)
}

// stieltjesOptions assembles the imaging options from the flags.
func (cfg fanoConfig) stieltjesOptions() stieltjes.Options {
	return stieltjes.Options{
		Prec: cfg.stPrec, MinOrder: cfg.stOrderLo, MaxOrder: cfg.stOrderHi,
		Average: cfg.stAverage, Window: cfg.stWindow,
	}
}

// imageAndClassify runs the Stieltjes step and, with an MO sidecar, the channel
// decomposition. Called inside the PMP stage so the Ritz vectors are still in hand.
func imageAndClassify(doc *FanoDocument, cfg fanoConfig, psp fano.Space, res lanczos.Result,
	ps *fano.Pseudo, phi fano.Discrete, nocc int) error {

	// The energy Gamma is evaluated at. Normally E_Phi, but -fano-at pins it, which is how
	// two schemes are compared at fixed energy: a scheme that moves E_Phi changes Gamma
	// twice over — once through the coupling density it produces, and once by evaluating
	// that density somewhere else. Separating the two is the only way to attribute a
	// difference between schemes to the physics rather than to the shift.
	at := phi.Energy
	if cfg.atEV != 0 {
		at = cfg.atEV / hartreeToEV
		doc.EvalEV = cfg.atEV
		fmt.Fprintf(os.Stderr, "adcgo: fano: evaluating Gamma at %.4f eV (-fano-at) rather than "+
			"at E_Phi = %.4f eV\n", cfg.atEV, phi.Energy*hartreeToEV)
	}
	w, err := fano.ImageWidth(ps, at, cfg.stieltjesOptions())
	if err != nil {
		return err
	}
	doc.WidthMeV, doc.WidthSigmaMeV = w.MeV, w.SigmaMeV
	doc.WidthEV, doc.LifetimeFS = w.Gamma*hartreeToEV, w.Tau
	doc.StieltjesMaxOrder = w.Stieltjes.MaxOrder
	doc.StieltjesOrders = w.Stieltjes.Used
	doc.StieltjesLowOrder = w.Stieltjes.LowOrder
	doc.StieltjesBelow, doc.StieltjesAbove = w.Stieltjes.Boundary()
	lo, hi := ps.Energy[0], ps.Energy[len(ps.Energy)-1]
	doc.PseudoEMinEV, doc.PseudoEMaxEV = lo*hartreeToEV, hi*hartreeToEV
	if hi > lo {
		doc.PhiPosition = (at - lo) / (hi - lo)
	}
	fmt.Fprintf(os.Stderr, "adcgo: fano: pseudo-continuum spans [%.3f, %.3f] eV; the evaluation "+
		"energy sits at %.1f%% of that span\n", doc.PseudoEMinEV, doc.PseudoEMaxEV,
		100*doc.PhiPosition)
	if doc.PhiPosition < 0.05 || doc.PhiPosition > 0.95 {
		fmt.Fprintf(os.Stderr, "adcgo: fano: WARNING the evaluation energy is at the edge of the "+
			"pseudo-continuum (%.1f%% of its span). Imaging reconstructs a density from these "+
			"samples, so near an edge it is extrapolating, not interpolating.\n",
			100*doc.PhiPosition)
	}
	if doc.StieltjesBelow+doc.StieltjesAbove > 0 {
		fmt.Fprintf(os.Stderr, "adcgo: fano: WARNING %d Stieltjes order(s) fell below and %d above "+
			"their own density grid and returned a fallback, not an interpolation; those values "+
			"are folded into the order average and corrupt it.\n",
			doc.StieltjesBelow, doc.StieltjesAbove)
	}

	if cfg.sip.moPath == "" || cfg.khci > 0 {
		// For khci the sidecar carries the orbital labels of the partition, not the
		// populations channel routing needs, and its decay classes are not two-hole.
		return nil
	}
	md, err := mo.ReadCanonical(cfg.sip.moPath)
	if err != nil {
		return err
	}
	chn, err := fano.NewChannels(md, nocc, cfg.sites, cfg.initSite, cfg.specOpts)
	if err != nil {
		return err
	}
	pw, err := chn.PartialWidths(psp, res, ps)
	if err != nil {
		return err
	}
	cws, err := fano.ImagePartial(pw, at, cfg.stieltjesOptions())
	if err != nil {
		return err
	}
	for _, cw := range cws {
		fc := FanoChannel{Name: cw.Name, Share: cw.Share}
		if cw.Err != nil {
			fc.Note = cw.Err.Error()
		} else {
			fc.WidthMeV, fc.SigmaMeV, fc.Ratio = cw.Width.MeV, cw.Width.SigmaMeV, cw.Ratio
		}
		doc.PartialWidths = append(doc.PartialWidths, fc)
	}
	doc.PartialNote = "approximate: channel projectors built from the L2 intermediate states, " +
		"per Kolorenc & Averbukh (2020) Sec. III D, which labels the method an estimate"
	return nil
}

// newFanoMatrix builds a sip.Matrix for one (sub)space with the run's backend, matrix-free
// policy, self-energy and ADC(2,2) variant applied.
func newFanoMatrix(ch *chooser, label string, sp *sip.Space, ints *integrals.Store,
	eps []float64, cfg sipConfig, v sip.Variant) (*sip.Matrix, error) {

	be := ch.pickDense(label, sp.Size())
	mx := sip.New(sp, ints, eps, cfg.order, be)
	mx.SetMatFree(cfg.matFree, cfg.matFreeBudget)
	mx.SetVariant(v)
	if cfg.sig != nil {
		mx.SetStaticSelfEnergy(cfg.sig)
	}
	return mx, nil
}

// fanoMatrix is what the Fano driver needs of an ADC matrix: the operator every solver
// takes, the dense form, its backend, and release of its resident blocks. *sip.Matrix
// and *dip.Matrix satisfy it.
type fanoMatrix interface {
	lanczos.PreconOperator
	BuildMatrix() backend.Mat
	Backend() backend.Backend
	Release()
}

// fanoFamily is one ionization family's side of the Fano driver: sub-spaces of its parent
// configuration space (the scheme A Q and P) and the ADC matrix over any of them.
type fanoFamily interface {
	restrict(rows []int) (fano.Space, error)
	matrix(ch *chooser, label string, sp fano.Space) (fanoMatrix, error)
}

// sipFamily is single ionization: ADC(2)x, ADC(3) or ADC(2,2) with the run's variant and
// static self-energy.
type sipFamily struct {
	parent *sip.Space
	ints   *integrals.Store
	eps    []float64
	cfg    sipConfig
	v      sip.Variant
}

func (f sipFamily) restrict(rows []int) (fano.Space, error) { return f.parent.Restrict(rows), nil }

func (f sipFamily) matrix(ch *chooser, label string, sp fano.Space) (fanoMatrix, error) {
	return newFanoMatrix(ch, label, sp.(*sip.Space), f.ints, f.eps, f.cfg, f.v)
}

// dipFamily is double ionization: DIP-ADC(2) of one spin sector. Its restrictions keep
// 3h1p groups whole (dip.Space.Restrict), which a hole predicate always does.
type dipFamily struct {
	parent *dip.Space
	ints   *integrals.Store
	eps    []float64
	cfg    sipConfig
}

func (f dipFamily) restrict(rows []int) (fano.Space, error) { return f.parent.Restrict(rows) }

func (f dipFamily) matrix(ch *chooser, label string, sp fano.Space) (fanoMatrix, error) {
	dsp := sp.(*dip.Space)
	mx := dip.New(dsp, f.ints, f.eps, ch.pickDense(label, dsp.Size()))
	mx.SetMatFree(f.cfg.matFree, f.cfg.matFreeBudget)
	return mx, nil
}

// khciFamily is the k-hole CI family. Sub-spaces are khci.Space restrictions; their
// operators share the parent's integrals and are materialized as CSR when they fit
// csrBytes, since the QMQ and PMP solves apply them many times.
type khciFamily struct {
	parent   *khci.Space
	op       *khci.CI
	csrBytes int64
	md       *mo.Data // labelled sidecar, or nil
	nocc     int
}

func (f khciFamily) restrict(rows []int) (fano.Space, error) { return f.parent.Restrict(rows) }

func (f khciFamily) matrix(ch *chooser, label string, sp fano.Space) (fanoMatrix, error) {
	ksp := sp.(*khci.Space)
	if ksp == f.parent {
		return f.op, nil // one apply (the coupling vector): enumerate, do not store
	}
	rows := make([]int, ksp.Size())
	for i := range rows {
		rows[i] = ksp.Parent(i)
	}
	op, err := f.op.Restrict(rows)
	if err != nil {
		return nil, err
	}
	if f.csrBytes > 0 {
		if op.Materialize(f.csrBytes) {
			fmt.Fprintf(os.Stderr, "adcgo: fano: %s: %d couplings stored (CSR)\n", label, op.NNZ())
		} else {
			fmt.Fprintf(os.Stderr, "adcgo: fano: %s: couplings exceed %.3g GB, applied by enumeration\n",
				label, float64(f.csrBytes)/(1<<30))
		}
	}
	return op, nil
}

// spinName is the -spin spelling of a DIP spin sector.
func spinName(s dip.Spin) string {
	if s == dip.Triplet {
		return "triplet"
	}
	return "singlet"
}

// solveFanoSpace solves one Fano sub-problem. lopts carries the caller's Krylov settings
// (block width, start rows) for the PMP solve; a zero value takes the -solver defaults,
// which is what the QMQ solve wants.
func solveFanoSpace(ch *chooser, label string, fam fanoFamily, sp fano.Space, cfg sipConfig,
	lopts lanczos.Options, wantFull bool) (lanczos.Result, fanoMatrix, error) {

	mx, err := fam.matrix(ch, label, sp)
	if err != nil {
		return lanczos.Result{}, nil, err
	}
	be := mx.Backend()
	if lopts.MaxBlocks == 0 {
		lopts.MaxBlocks = cfg.blocks
	}
	lopts.WantFull = wantFull

	var res lanczos.Result
	switch cfg.solver {
	case "dense":
		res = lanczos.SolveDense(mx, be)
	case "lanczos":
		res = lanczos.Solve(mx, be, lopts)
	case "davidson":
		res = lanczos.SolveDavidson(mx, be,
			davidsonOpts(cfg.nroots, cfg.maxdavsp, cfg.maxdavit, cfg.convthr, wantFull))
	default:
		mx.Release()
		return lanczos.Result{}, nil, fmt.Errorf("-fano needs the Ritz vectors, which -solver %q "+
			"does not keep; use lanczos, davidson or dense", cfg.solver)
	}
	if cfg.profile {
		reportTiming(label, sp.Size(), sp.MainBlockSize(), res.Timing)
	}
	return res, mx, nil
}

// couplingFile is the -fano-save-g / -fano-image-g format.
type couplingFile struct {
	EPhiHartree float64   `json:"e_phi_hartree"`
	PSize       int       `json:"p_size"`
	G           []float64 `json:"g"`
}

func saveCoupling(path string, g []float64, e float64) error {
	b, err := json.Marshal(couplingFile{EPhiHartree: e, PSize: len(g), G: g})
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func loadCoupling(path string, psize int) ([]float64, float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	var c couplingFile
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, 0, fmt.Errorf("%s: %w", path, err)
	}
	if len(c.G) != psize || c.PSize != psize {
		return nil, 0, fmt.Errorf("%s holds a coupling vector over %d P rows, this partition has %d",
			path, len(c.G), psize)
	}
	return c.G, c.EPhiHartree, nil
}

// scaleTransfers applies -fano-lambda "A:B=l,C:D=m" to a copy of the integrals: every
// charge distribution pairing an orbital of atom A with one of atom B is scaled by l
// (fcidump.ScaleTransfer), orbitals assigned by the sidecar labels (free virtuals belong
// to no atom and are never scaled).
func scaleTransfers(d *fcidump.Data, md *mo.Data, spec string) (*fcidump.Data, error) {
	idx := func(name string) (int, error) {
		for a, n := range md.AtomNames {
			if n == name {
				if md.GhostAtom[a] {
					return 0, fmt.Errorf("-fano-lambda: %q is a ghost centre", name)
				}
				return a, nil
			}
		}
		return 0, fmt.Errorf("-fano-lambda: unknown atom %q (the sidecar has %v)", name, md.AtomNames)
	}
	if len(md.OrbAtom) != d.NORB {
		return nil, fmt.Errorf("-fano-lambda: the sidecar labels %d orbitals, the FCIDUMP has %d",
			len(md.OrbAtom), d.NORB)
	}
	out := d.Clone()
	for _, term := range strings.Split(spec, ",") {
		term = strings.TrimSpace(term)
		pair, val, ok := strings.Cut(term, "=")
		names := strings.Split(pair, ":")
		if !ok || len(names) != 2 {
			return nil, fmt.Errorf("-fano-lambda: bad term %q (want A:B=lambda)", term)
		}
		lam, err := strconv.ParseFloat(strings.TrimSpace(val), 64)
		if err != nil {
			return nil, fmt.Errorf("-fano-lambda: bad lambda in %q", term)
		}
		a, err := idx(strings.TrimSpace(names[0]))
		if err != nil {
			return nil, err
		}
		b, err := idx(strings.TrimSpace(names[1]))
		if err != nil {
			return nil, err
		}
		if err := out.ScaleTransfer(md.OrbAtom, a, b, lam); err != nil {
			return nil, err
		}
		fmt.Fprintf(os.Stderr, "adcgo: fano: transfer integrals %s <-> %s scaled by %g\n",
			md.AtomNames[a], md.AtomNames[b], lam)
	}
	return out, nil
}

// imageDecomposition images each class piece of the coupling vector and every pair's
// interference with the Gauss engine (W-B, -fano-order steps): it needs no PMP
// eigenvector, so it costs one Lanczos run per piece and pair whatever -fano-engine is.
func imageDecomposition(doc *FanoDocument, cfg fanoConfig, fam fanoFamily, ch *chooser, psp fano.Space,
	pieces map[string][]float64, phi fano.Discrete, stage func(string, func() error) error) error {

	names := make([]string, 0, len(pieces))
	for n := range pieces {
		names = append(names, n)
	}
	sort.Strings(names)
	m := cfg.order
	if m <= 0 {
		m = 200
	}
	at := phi.Energy
	if cfg.atEV != 0 {
		at = cfg.atEV / hartreeToEV
	}
	return stage(fmt.Sprintf("ordering decomposition (%d classes)", len(names)), func() error {
		pmx, err := fam.matrix(ch, "fano PMP (decomposition)", psp)
		if err != nil {
			return err
		}
		defer pmx.Release()
		be := pmx.Backend()
		n := psp.Size()
		out := be.Alloc(n)
		defer be.Free(out)
		apply := func(dst, src []float64) {
			v := be.Upload(src)
			pmx.ApplyFull(out, v)
			be.Free(v)
			copy(dst, be.Download(out))
		}
		width := func(g []float64) (fano.Width, error) {
			if fano.SumRule(g) == 0 {
				return fano.Width{}, nil // a class with no coupling carries no width
			}
			w, _, err := fano.GaussWidth(apply, g, m, at, cfg.stieltjesOptions())
			return w, err
		}
		own := map[string]fano.Width{}
		for _, name := range names {
			w, err := width(pieces[name])
			if err != nil {
				return fmt.Errorf("class %s: %w", name, err)
			}
			own[name] = w
			doc.Decomposition = append(doc.Decomposition, FanoPiece{Name: name, WidthMeV: w.MeV,
				SigmaMeV: w.SigmaMeV, Norm2: fano.SumRule(pieces[name]) / (2 * math.Pi)})
		}
		for i, a := range names {
			for _, b := range names[i+1:] {
				sum := make([]float64, n)
				for r := range sum {
					sum[r] = pieces[a][r] + pieces[b][r]
				}
				w, err := width(sum)
				if err != nil {
					return fmt.Errorf("classes %s+%s: %w", a, b, err)
				}
				doc.Decomposition = append(doc.Decomposition, FanoPiece{Name: a + "+" + b,
					WidthMeV: w.MeV - own[a].MeV - own[b].MeV,
					SigmaMeV: w.SigmaMeV + own[a].SigmaMeV + own[b].SigmaMeV, Interference: true})
			}
		}
		for _, p := range doc.Decomposition {
			fmt.Fprintf(os.Stderr, "adcgo: fano: decomposition %-12s %12.6g +/- %.3g meV\n",
				p.Name, p.WidthMeV, p.SigmaMeV)
		}
		return nil
	})
}
