package main

import (
	"fmt"
	"time"

	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
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
	n, b := sp.Size(), sp.MainBlockSize()

	var be backend.Backend
	if cfg.solver == "dense" {
		be = ch.pickDense(label, n)
	} else {
		be = ch.pickLanczos(label, n, b, lanczos.SubspaceDim(n, b, lopts),
			func(cand backend.Backend) time.Duration {
				m := sip.New(sp, ints, eps, order, cand)
				defer m.Release()
				return timeApplyBlock(cand, n, b, m.ApplyBlock)
			})
	}
	mx := sip.New(sp, ints, eps, order, be)

	var res lanczos.Result
	switch cfg.solver {
	case "dense":
		res = lanczos.SolveDense(mx, be)
	case "lanczos":
		res = lanczos.Solve(mx, be, lopts)
	default:
		mx.Release()
		return lanczos.Result{}, nil, fmt.Errorf("unknown solver %q (want lanczos or dense)", cfg.solver)
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
			ems, err := analyze.BuildSIPEmissions(s.sp, s.res, s.mx.FMatrix(), md, opts)
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

	if cfg.order != 4 {
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
				k.sp, k.res, k.mx.FMatrix(), md, opts)
			if err != nil {
				return nil, err
			}
			td.CrossEmissions = append(td.CrossEmissions, cem...)
		}
	}
	return td, nil
}
