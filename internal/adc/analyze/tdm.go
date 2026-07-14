package analyze

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// The RASSI-like transition-property output layer (docs/adc4_rassi_plan.md, Chunk 6):
// it turns solved SIP sectors into the two dipole elements of the ICD decay chain —
// the ion→ion emission moment (element 1) and the per-state Dyson photoionization
// pseudo-spectrum (element 2) — and, for a CVS ADC(4) run, the core→valence X-ray
// emission across two spaces. Everything here is assembly over the already-validated
// sip primitives; no new physics.

// auToPerSec converts an Einstein A coefficient from atomic units of inverse time to
// s⁻¹: 1/τ_au with τ_au = 2.4188843e-17 s.
const auToPerSec = 4.1341373e16

// SIPTransition is one radiative transition dipole between two cationic states — the
// ion→ion emission of element 1. For a within-sector transition InitIrrep == MidIrrep
// and Overlap is omitted; for a cross-space (CVS core → valence) transition Cross is
// set and Overlap carries the state overlap (nonzero, and a gauge-origin warning, only
// when the two states share an irrep).
type SIPTransition struct {
	InitIrrep int `json:"init_irrep"`
	MidIrrep  int `json:"mid_irrep"`
	Init      int `json:"init"` // 1-based state index within its sector
	Mid       int `json:"mid"`

	InitEV  float64 `json:"init_ev"`
	MidEV   float64 `json:"mid_ev"`
	OmegaEV float64 `json:"omega_ev"` // E_init − E_mid, > 0 for emission

	Mu         [3]float64 `json:"mu"`           // transition dipole, a.u.
	Osc        float64    `json:"osc"`          // oscillator strength f
	RatePerSec float64    `json:"rate_per_sec"` // Einstein A, s⁻¹

	Cross   bool    `json:"cross,omitempty"`
	Overlap float64 `json:"overlap,omitempty"`
}

// SIPChannel is one discretized photoionization channel: the cationic state left
// behind, with the ejected electron parked in a virtual MO as an L² continuum proxy.
type SIPChannel struct {
	Vir     int        `json:"vir"`      // 0-based virtual position; absolute orbital = Nocc + Vir
	EpsEV   float64    `json:"eps_ev"`   // ε_a, the proxy photoelectron energy
	OmegaEV float64    `json:"omega_ev"` // photon energy E_state + ε_a
	Mu      [3]float64 `json:"mu"`       // ⟨φ_a|r|d⟩, a.u.
	Osc     float64    `json:"osc"`      // oscillator strength (2/3)·ω·|μ|²
}

// SIPPhotoionization is the L² photoionization pseudo-spectrum of one cationic state
// (element 2). The channel strengths are discrete; converting them to a smooth
// σ_ion(ω) needs Stieltjes imaging (Track W) and is not done here.
type SIPPhotoionization struct {
	Irrep      int          `json:"irrep"`
	State      int          `json:"state"` // 1-based
	EnergyEV   float64      `json:"energy_ev"`
	SpecFactor float64      `json:"spec_factor"` // Σ_occ d² — the pole strength of the Dyson orbital
	Channels   []SIPChannel `json:"channels,omitempty"`
}

// SIPTDMOptions tunes the transition-property output.
type SIPTDMOptions struct {
	// OscThresh drops photoionization channels below this oscillator strength. The
	// pseudo-spectrum has one channel per virtual orbital, so a threshold keeps the
	// output readable without discarding anything physical.
	OscThresh float64

	// ISR selects the correlation-corrected ISR property matrix (sip/isrdipole_corr.go).
	// nil leaves the operator at zeroth order in the fluctuation potential.
	ISR *sip.ISROptions
}

// colIndex maps each raw solver column to its 1-based position in cols.
func colIndex(cols []int) map[int]int {
	m := make(map[int]int, len(cols))
	for i, c := range cols {
		m[c] = i + 1
	}
	return m
}

// BuildSIPEmissions returns the within-sector ion→ion emission dipoles of one solved
// SIP sector (element 1): every pair of surviving states in the emission direction
// (ω > 0). sp must be a plain (order ≤ 3) space — the square ISR dipole is not defined
// over a 3h2p space; use BuildSIPCrossEmissions for a CVS run. md must carry dipole
// integrals, and res must retain the satellite rows (lanczos.Options.WantFull).
func BuildSIPEmissions(sp *sip.Space, res lanczos.Result, fmat backend.Mat, md *mo.Data, opts Options, tdm SIPTDMOptions) ([]SIPTransition, error) {
	if !md.HasDipole {
		return nil, fmt.Errorf("analyze: the MO sidecar has no dipole integrals")
	}
	ops, err := sip.NewISRDipolesWithCorr(sp, md.DipMO, tdm.ISR)
	if err != nil {
		return nil, err
	}
	sec, cols := buildSIPSector(sp, res, fmat, opts)
	ems, err := sip.Emissions(ops, res.Values, res.FullVecs, cols, cols)
	if err != nil {
		return nil, err
	}
	idx := colIndex(cols)
	var out []SIPTransition
	for _, e := range ems {
		if e.Omega <= 0 {
			continue // keep the emission direction only (deeper hole → shallower)
		}
		out = append(out, SIPTransition{
			InitIrrep:  sec.Irrep,
			MidIrrep:   sec.Irrep,
			Init:       idx[e.Init],
			Mid:        idx[e.Mid],
			InitEV:     res.Values[e.Init] * au2eV,
			MidEV:      res.Values[e.Mid] * au2eV,
			OmegaEV:    e.Omega * au2eV,
			Mu:         e.Mu,
			Osc:        e.Osc,
			RatePerSec: e.Rate * auToPerSec,
		})
	}
	return out, nil
}

// BuildSIPPhotoionization returns the per-state Dyson photoionization pseudo-spectrum
// of one solved sector (element 2). It works for every order, including a CVS ADC(4)
// space, whose 3h2p rows the Dyson construction skips. eps is the full orbital-energy
// array (length Space.Norb).
func BuildSIPPhotoionization(sp *sip.Space, mx *sip.Matrix, res lanczos.Result, fmat backend.Mat, md *mo.Data, eps []float64, opts Options, tdm SIPTDMOptions) ([]SIPPhotoionization, error) {
	if !md.HasDipole {
		return nil, fmt.Errorf("analyze: the MO sidecar has no dipole integrals")
	}
	sec, cols := buildSIPSector(sp, res, fmat, opts)
	dy, err := mx.DysonOrbitals(res.FullVecs, cols)
	if err != nil {
		return nil, err
	}
	out := make([]SIPPhotoionization, 0, len(cols))
	for i, k := range cols {
		d := make([]float64, dy.Rows)
		for r := range dy.Rows {
			d[r] = dy.At(r, i)
		}
		eIP := res.Values[k]
		p := SIPPhotoionization{
			Irrep:      sec.Irrep,
			State:      i + 1,
			EnergyEV:   eIP * au2eV,
			SpecFactor: sip.SpectroscopicFactor(sp, d),
		}
		for _, cm := range sip.PhotoionizationMoments(md.DipMO, sp, eps, d, eIP, i) {
			if cm.Eps <= 0 {
				continue // ε_a < 0 are bound-like virtuals, not continuum proxies
			}
			if cm.Osc < tdm.OscThresh {
				continue
			}
			p.Channels = append(p.Channels, SIPChannel{
				Vir:     cm.Vir,
				EpsEV:   cm.Eps * au2eV,
				OmegaEV: cm.Omega * au2eV,
				Mu:      cm.Mu,
				Osc:     cm.Osc,
			})
		}
		out = append(out, p)
	}
	return out, nil
}

// BuildSIPCrossEmissions returns the emission dipoles between the states of two different
// SIP spaces, in the two regimes isrdipole_cross.go describes: a CVS ADC(4) core bra
// against a plain valence ket (Chunk 5), and one plain irrep sector against another. Both
// results must retain the satellite rows. Overlap is reported on each transition: it is
// zero when the two sectors carry different irreps (the intended, gauge-independent case)
// and nonzero, with an origin-dependent moment, when they share one.
func BuildSIPCrossEmissions(
	bra *sip.Space, resBra lanczos.Result, fbra backend.Mat,
	ket *sip.Space, resKet lanczos.Result, fket backend.Mat,
	md *mo.Data, opts Options, tdm SIPTDMOptions) ([]SIPTransition, error) {

	if !md.HasDipole {
		return nil, fmt.Errorf("analyze: the MO sidecar has no dipole integrals")
	}
	ops, err := sip.NewISRDipolesCrossWithCorr(bra, ket, md.DipMO, tdm.ISR)
	if err != nil {
		return nil, err
	}
	ov, err := sip.ConfigOverlap(bra, ket)
	if err != nil {
		return nil, err
	}
	braSec, braCols := buildSIPSector(bra, resBra, fbra, opts)
	ketSec, ketCols := buildSIPSector(ket, resKet, fket, opts)
	ems, err := sip.CrossEmissions(ops, ov, md.NuclearDipole(),
		resBra.Values, resKet.Values, resBra.FullVecs, resKet.FullVecs, braCols, ketCols)
	if err != nil {
		return nil, err
	}
	// Cross marks a transition between two *different truncated Hamiltonians* — the CVS
	// ADC(4) core space against a plain valence one. There the state overlap need not
	// vanish and the ⟨3h2p|D̂|2h1p⟩ block is dropped. Two plain sectors are symmetry blocks
	// of one and the same secular matrix: they share no configuration, so S ≡ 0 and the
	// moment is exactly as well defined as a within-sector one. Those are not cross
	// transitions, and the ω > 0 emission they carry is the ordinary x/y-polarized line
	// that no square, single-sector D can see.
	cross := len(bra.Sat3) != 0 || len(ket.Sat3) != 0

	initIdx, midIdx := colIndex(braCols), colIndex(ketCols)
	var out []SIPTransition
	for _, e := range ems {
		if e.Omega <= 0 {
			continue
		}
		out = append(out, SIPTransition{
			InitIrrep:  braSec.Irrep,
			MidIrrep:   ketSec.Irrep,
			Init:       initIdx[e.Init],
			Mid:        midIdx[e.Mid],
			InitEV:     resBra.Values[e.Init] * au2eV,
			MidEV:      resKet.Values[e.Mid] * au2eV,
			OmegaEV:    e.Omega * au2eV,
			Mu:         e.Mu,
			Osc:        e.Osc,
			RatePerSec: e.Rate * auToPerSec,
			Cross:      cross,
			Overlap:    e.Overlap,
		})
	}
	return out, nil
}
