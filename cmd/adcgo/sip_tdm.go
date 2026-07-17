package main

import (
	"fmt"
	"os"
	"time"

	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/selfenergy"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// SIPTDMDocument is the -tdm output: the standard cation states, plus the two dipole
// elements of the ICD decay chain (docs/adc4_rassi_plan.md, Chunk 6).
//
//   - Emissions: ion→ion radiative transitions within a sector (element 1).
//   - Photoionization: each state's Dyson L² photoionization pseudo-spectrum
//     (element 2). The channel strengths are discrete; a smooth σ_ion(ω) needs
//     Stieltjes imaging, which lives in Track W, so the asymptotic ICD width is not
//     assembled here.
//   - CrossEmissions: core→valence X-ray emission for an -order 4 (CVS) run, between
//     the core sector and companion plain-ADC(3) valence solves.
type SIPTDMDocument struct {
	NORB   int    `json:"norb"`
	NELEC  int    `json:"nelec"`
	Order  int    `json:"order"`
	Solver string `json:"solver"`

	Sectors         []analyze.SIPSector          `json:"sectors"`
	Emissions       []analyze.SIPTransition      `json:"emissions,omitempty"`
	Photoionization []analyze.SIPPhotoionization `json:"photoionization,omitempty"`
	CrossEmissions  []analyze.SIPTransition      `json:"cross_emissions,omitempty"`
}

// isrOptions resolves -tdm-isr into the ISR property-matrix configuration, building the
// ground-state density once for the whole run.
//
// The correlation corrections (sip/isrdipole_corr.go) are the O(2)/O(1) ISR expansion that
// matches an ADC(2)/ADC(3) secular matrix, so they are on by default at those orders. They are
// *not* order-consistent for ADC(4) — that would need the 3rd/4th-order terms built from
// t₂⁽²⁾/t₁⁽²⁾/t₃⁽²⁾, which do not exist in the tree — so -order 4 leaves them off unless asked.
func isrOptions(d *fcidump.Data, ints *integrals.Store, eps []float64, nocc int, cfg sipConfig) (*sip.ISROptions, error) {
	order := cfg.tdmISR
	if order < 0 {
		order = 2
		if cfg.order == 4 {
			order = 0
		}
	}
	if order == 0 {
		return nil, nil
	}
	// ρ⁽²⁾: what the legacy property module uses, and the order the corrections are consistent
	// with. It is the same object the static self-energy is built from.
	rho, err := selfenergy.Density(ints, eps, nocc, d.NORB, 2)
	if err != nil {
		return nil, fmt.Errorf("ISR ground-state density: %w", err)
	}
	if cfg.order == 4 {
		fmt.Fprintln(os.Stderr, "note: -tdm-isr 2 with -order 4 applies the 2nd-order ISR corrections to a "+
			"CVS ADC(4) space; they are a partial, order-inconsistent improvement there (no 3h2p correction).")
	}
	return &sip.ISROptions{Ints: ints, Eps: eps, Rho: rho.Func()}, nil
}

// solvedSIP is one sector whose operator is kept live (not released) so the
// transition-moment layer can reach its FMatrix and Dyson amplitudes.
type solvedSIP struct {
	sp     *sip.Space
	mx     *sip.Matrix
	res    lanczos.Result
	sector analyze.SIPSector
}

// solveSIPSpace solves one SIP configuration space and returns the result together with
// the still-live operator (the caller must Release it). wantFull retains the satellite
// rows of every Ritz vector — a hard requirement for the transition-moment machinery,
// and a no-op cost for the dense solver, which produces them anyway.
func solveSIPSpace(ch *chooser, label string, sp *sip.Space, ints *integrals.Store, eps []float64, order int, cfg sipConfig, wantFull bool) (lanczos.Result, *sip.Matrix, error) {
	lopts := lanczos.Options{MaxBlocks: cfg.blocks, WantFull: wantFull}
	// Checkpoint the block-Krylov build so a walltime kill can resume in a successor job.
	// Per-sector path (irrep-keyed) so concurrent sectors never share a file. Lanczos only —
	// the dense/Davidson drivers hold no resumable Krylov basis.
	if cfg.solver == "lanczos" && cfg.ckpt != "" {
		lopts.Checkpoint = &lanczos.Checkpoint{
			Path:  fmt.Sprintf("%s.i%d", cfg.ckpt, sp.Sym),
			Every: cfg.ckptEvery,
			Stop:  cfg.stop,
		}
	}
	davOpts := davidsonOpts(cfg.nroots, cfg.maxdavsp, cfg.maxdavit, cfg.convthr, wantFull)
	n, b := sp.Size(), sp.MainBlockSize()
	subspace := lanczos.SubspaceDim(n, b, lopts)
	if cfg.solver == "davidson" {
		subspace = lanczos.DavidsonSubspaceDim(n, davOpts)
	}

	var be backend.Backend
	if cfg.solver == "dense" {
		be = ch.pickDense(label, n)
	} else {
		be = ch.pickLanczos(label, n, b, subspace,
			func(cand backend.Backend) time.Duration {
				m := sip.New(sp, ints, eps, order, cand)
				m.SetMatFree(cfg.matFree, cfg.matFreeBudget)
				defer m.Release()
				return timeApplyBlock(cand, n, b, m.ApplyBlock)
			})
	}
	mx := sip.New(sp, ints, eps, order, be)
	mx.SetMatFree(cfg.matFree, cfg.matFreeBudget)
	mx.SetWert3(cfg.wert3)
	if cfg.sig != nil {
		mx.SetStaticSelfEnergy(cfg.sig)
	}

	var res lanczos.Result
	switch cfg.solver {
	case "dense":
		res = lanczos.SolveDense(mx, be)
	case "lanczos":
		res = lanczos.Solve(mx, be, lopts)
	case "davidson":
		res = lanczos.SolveDavidson(mx, be, davOpts)
	default:
		mx.Release()
		return lanczos.Result{}, nil, fmt.Errorf("unknown solver %q (want lanczos, davidson or dense)", cfg.solver)
	}
	if res.Interrupted {
		// A stop signal checkpointed and bailed out mid-build; propagate so main exits with
		// the "resume needed" code rather than emitting a partial spectrum.
		return res, mx, errInterrupted
	}
	if cfg.profile {
		reportTiming(label, sp.Size(), sp.MainBlockSize(), res.Timing)
	}
	return res, mx, nil
}

// buildSIPTDMDoc assembles the transition-moment document from the solved sectors. For
// an -order 4 run it additionally solves companion plain-ADC(3) valence sectors — the
// middle states of the X-ray-emission leg — over every irrep with an occupied orbital,
// and cross-emits each core sector against them.
func buildSIPTDMDoc(ch *chooser, d *fcidump.Data, ints *integrals.Store, orbSym []int, eps []float64, solved []solvedSIP, md *mo.Data, opts analyze.Options, cfg sipConfig) (*SIPTDMDocument, error) {
	nocc := mp.NOcc(d)
	tdmOpts := analyze.SIPTDMOptions{OscThresh: cfg.tdmOsc}

	isr, err := isrOptions(d, ints, eps, nocc, cfg)
	if err != nil {
		return nil, err
	}
	tdmOpts.ISR = isr

	td := &SIPTDMDocument{NORB: d.NORB, NELEC: d.NELEC, Order: cfg.order, Solver: cfg.solver}
	for _, s := range solved {
		td.Sectors = append(td.Sectors, s.sector)
	}

	// Element 1 within a sector, plus element 2 for every state. The square ISR dipole
	// is undefined over a 3h2p (ADC(4)) space, so within-sector emissions are skipped
	// there — those states emit via the cross-space path below — while the Dyson
	// photoionization, which skips its own 3h2p rows, runs for every order.
	for _, s := range solved {
		if len(s.sp.Sat3) == 0 {
			ems, err := analyze.BuildSIPEmissions(s.sp, s.res, s.mx.FMatrix(), md, opts, tdmOpts)
			if err != nil {
				return nil, err
			}
			td.Emissions = append(td.Emissions, ems...)
		}
		ph, err := analyze.BuildSIPPhotoionization(s.sp, s.mx, s.res, s.mx.FMatrix(), md, eps, opts, tdmOpts)
		if err != nil {
			return nil, err
		}
		td.Photoionization = append(td.Photoionization, ph...)
	}

	// Element 1 *across* sectors. Inside one sector the square D only sees the totally
	// symmetric dipole component, but a deep hole and an outer-valence hole generally carry
	// different irreps (O 1s is a₁ in H2O, 1b₁ is b₁), so the x/y-polarized lines — most of
	// an X-ray emission spectrum — live entirely in the rectangular cross-sector D. The two
	// sectors are symmetry blocks of one secular matrix, so these moments are exact, not an
	// approximation: they land in Emissions alongside the within-sector ones. Ordered pairs
	// with CrossEmissions' ω > 0 filter visit each transition exactly once.
	if cfg.order != 4 {
		for _, bra := range solved {
			for _, ket := range solved {
				if bra.sp.Sym == ket.sp.Sym {
					continue
				}
				ems, err := analyze.BuildSIPCrossEmissions(
					bra.sp, bra.res, bra.mx.FMatrix(),
					ket.sp, ket.res, ket.mx.FMatrix(), md, opts, tdmOpts)
				if err != nil {
					return nil, err
				}
				td.Emissions = append(td.Emissions, ems...)
			}
		}
		return td, nil
	}

	// Cross-space X-ray emission: companion valence middle states, plain ADC(3).
	type valence struct {
		sp  *sip.Space
		mx  *sip.Matrix
		res lanczos.Result
	}
	var kets []valence
	release := func() {
		for _, k := range kets {
			k.mx.Release()
		}
	}
	for g := 0; g < numIrreps(orbSym, d.NORB); g++ {
		vsp := sip.NewSpace(nocc, d.NORB, orbSym, g)
		if vsp.MainBlockSize() == 0 {
			continue
		}
		vres, vmx, err := solveSIPSpace(ch, fmt.Sprintf("sip valence irrep=%d", g+1), vsp, ints, eps, 3, cfg, true)
		if err != nil {
			release()
			return nil, err
		}
		kets = append(kets, valence{vsp, vmx, vres})
	}
	defer release()

	for _, s := range solved {
		for _, k := range kets {
			cem, err := analyze.BuildSIPCrossEmissions(
				s.sp, s.res, s.mx.FMatrix(),
				k.sp, k.res, k.mx.FMatrix(), md, opts, tdmOpts)
			if err != nil {
				return nil, err
			}
			td.CrossEmissions = append(td.CrossEmissions, cem...)
		}
	}
	return td, nil
}
