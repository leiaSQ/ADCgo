package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/leiaSQ/ADCgo/internal/adc/fano"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
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
	NORB  int `json:"norb"`
	NELEC int `json:"nelec"`
	Order int `json:"order"`
	// Variant is the ADC(2,2) scheme ("m", "x" or "f"); empty for order 2/3.
	Variant string `json:"variant,omitempty"`
	Scheme  string `json:"scheme"` // Q/P selection scheme: "a" (hole localization)

	Vacancy   int    `json:"vacancy"`   // 0-based occupied orbital of the initial hole
	Irrep     int    `json:"irrep"`     // 1-based target irrep (fixed by the vacancy)
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
	PhiRoot     int     `json:"phi_root"`   // which QMQ root
	// EvalEV is the energy Gamma was evaluated at when -fano-at overrode E_Phi.
	EvalEV float64 `json:"eval_ev,omitempty"`

	// The pseudo-continuum.
	CoupledChannels int     `json:"coupled_channels"` // P configurations with nonzero g
	States          int     `json:"pseudo_states"`
	SumGamma        float64 `json:"sum_gamma_hartree2"` // zeroth moment, NOT a width
	SumRule         float64 `json:"sum_rule_hartree2"`  // 2 pi ||g||^2
	SumResidual     float64 `json:"sum_rule_residual"`  // (rule - sum)/rule
	DroppedE        int     `json:"dropped_energy"`
	DroppedW        int     `json:"dropped_weight"`
	DroppedG        int     `json:"dropped_gamma"`

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

	variant sip.Variant
	vacancy int   // -fano-init
	qOrbs   []int // -fano-q (empty = just the vacancy)
	rule    fano.HoleRule
	qpSpec  string  // -fano-qp: a per-excitation-class Q/P rule, overriding -fano-q/-fano-rule
	nth     int     // -fano-nth: which qualifying QMQ root
	qMin    float64 // -fano-qmin: minimum |Phi> weight on the vacancy configuration
	qRoots  int     // -fano-qroots: QMQ roots to converge
	qSolver string  // -fano-qsolver: solver for QMQ ("" = davidson, or dense when -solver dense)

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
	if cfg.sip.order != 2 && cfg.sip.order != 3 && cfg.sip.order != sip.Order22 {
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

	// The vacancy fixes the target irrep: |Phi> has the 1h configuration's symmetry, and the
	// secular matrix is block diagonal in irreps, so no other sector can contribute to its
	// width. -sym is therefore not a choice here; it is determined.
	orbSym, _, err := selectSymmetry(cfg.sip.sym, d)
	if err != nil {
		return err
	}
	targetSym := 0
	if orbSym != nil {
		targetSym = orbSym[cfg.vacancy] - 1
	}
	ints := integrals.New(d, nocc, orbSym)
	if cfg.sip.sig, err = buildSigma(cfg.sip, ints, eps, nocc, d.NORB); err != nil {
		return err
	}

	var parent *sip.Space
	if cfg.sip.order == sip.Order22 {
		parent = sip.NewSpace22(nocc, d.NORB, orbSym, targetSym)
	} else {
		parent = sip.NewSpace(nocc, d.NORB, orbSym, targetSym)
	}
	if parent.MainBlockSize() == 0 {
		return fmt.Errorf("irrep %d has no 1h configurations, so orbital %d cannot be the "+
			"initial vacancy there", targetSym+1, cfg.vacancy)
	}

	// Scheme A: the Q predicate. With no -fano-q the set is the vacancy itself, which with
	// the default "any" rule is the Auger criterion — Q is every configuration that still
	// carries the initial hole, P every one that has filled it.
	var sel fano.Selector
	if cfg.qpSpec != "" {
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
	qsp := parent.Restrict(part.Q)
	psp := parent.Restrict(part.P)
	fmt.Fprintf(os.Stderr, "adcgo: fano: irrep %d, %s\n", targetSym+1, part)

	doc := FanoDocument{
		NORB: d.NORB, NELEC: d.NELEC, Order: cfg.sip.order, Scheme: "a",
		Vacancy: cfg.vacancy, Irrep: targetSym + 1, Criterion: sel.String(),
		Size: parent.Size(), QSize: part.QSize(), PSize: part.PSize(),
		QMain: part.QMain, PMain: psp.MainBlockSize(), Class3: len(parent.Sat3),
	}
	if cfg.sip.order == sip.Order22 {
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

	var phi fano.Discrete
	err = stage(fmt.Sprintf("QMQ solve (%d configurations, %s)", qsp.Size(), qSolver), func() error {
		qcfg := cfg.sip
		qcfg.nroots = cfg.qRoots
		qcfg.solver = qSolver
		res, qmx, err := solveFanoSpace(ch, "fano QMQ", qsp, ints, eps, qcfg, cfg.variant,
			lanczos.Options{}, true)
		if err != nil {
			return err
		}
		defer qmx.Release()
		phi, err = fano.SelectDiscrete(qsp, res, cfg.vacancy, cfg.nth, cfg.qMin)
		return err
	})
	if err != nil {
		return err
	}
	doc.EPhiHartree, doc.EPhiEV = phi.Energy, phi.Energy*hartreeToEV
	doc.PhiWeight, doc.PhiRoot = phi.Weight, phi.Root
	fmt.Fprintf(os.Stderr, "adcgo: fano: E_Phi = %.6f Eh = %.4f eV (QMQ root %d, weight %.4f)\n",
		phi.Energy, doc.EPhiEV, phi.Root, phi.Weight)

	// --- The coupling vector, from ONE parent mat-vec. ---
	var g []float64
	err = stage("coupling vector", func() error {
		pmx, err := newFanoMatrix(ch, "fano parent", parent, ints, eps, cfg.sip, cfg.variant)
		if err != nil {
			return err
		}
		defer pmx.Release()
		g, err = fano.Coupling(pmx, part, phi, pmx.Backend())
		return err
	})
	if err != nil {
		return err
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
			res, pmx, err := solveFanoSpace(ch, "fano PMP", psp, ints, eps, cfg.sip, cfg.variant,
				lopts, true)
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
				EMax: emax, MinWeight: cfg.wmin, MinGamma: cfg.gmin,
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

// stieltjesOptions assembles the imaging options from the flags.
func (cfg fanoConfig) stieltjesOptions() stieltjes.Options {
	return stieltjes.Options{
		Prec: cfg.stPrec, MinOrder: cfg.stOrderLo, MaxOrder: cfg.stOrderHi,
		Average: cfg.stAverage, Window: cfg.stWindow,
	}
}

// imageAndClassify runs the Stieltjes step and, with an MO sidecar, the channel
// decomposition. Called inside the PMP stage so the Ritz vectors are still in hand.
func imageAndClassify(doc *FanoDocument, cfg fanoConfig, psp *sip.Space, res lanczos.Result,
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

	if cfg.sip.moPath == "" {
		return nil
	}
	md, err := mo.ReadFile(cfg.sip.moPath)
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

// solveFanoSpace solves one Fano sub-problem. lopts carries the caller's Krylov settings
// (block width, start rows) for the PMP solve; a zero value takes the -solver defaults,
// which is what the QMQ solve wants.
func solveFanoSpace(ch *chooser, label string, sp *sip.Space, ints *integrals.Store,
	eps []float64, cfg sipConfig, v sip.Variant, lopts lanczos.Options, wantFull bool) (
	lanczos.Result, *sip.Matrix, error) {

	mx, err := newFanoMatrix(ch, label, sp, ints, eps, cfg, v)
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
		return lanczos.Result{}, nil, fmt.Errorf("unknown solver %q", cfg.solver)
	}
	if cfg.profile {
		reportTiming(label, sp.Size(), sp.MainBlockSize(), res.Timing)
	}
	return res, mx, nil
}
