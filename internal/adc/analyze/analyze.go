// Package analyze turns a solved DIP-ADC(2) sector (Lanczos or dense Ritz pairs)
// into the state records ADCgo emits: double-ionization energies, pole strengths,
// leading two-hole configurations, and — when MO coefficients and the AO overlap
// are supplied — atom-resolved two-hole populations.
//
// Field semantics mirror ADCanalysis internal/model (State/Config/PopRow) so the
// output feeds the existing classify/spectrum path with only a thin adapter (M4).
package analyze

import (
	"sort"

	"adcgo/internal/adc/dip"
	"adcgo/internal/adc/lanczos"
)

// au2eV matches the reference conversion (adc2_dip_analyzer.cpp:16).
const au2eV = 27.211396

// spurThresh is the reference's ghost-root cutoff on main-space weight
// (adc_diagonalizer.cpp:499).
const spurThresh = 1e-9

// Leading is one leading two-hole component of a state (a "<i,j|: coeff" entry).
// I and J are 1-based MO indices with I >= J, matching the reference tables.
type Leading struct {
	I     int     `json:"i"`
	J     int     `json:"j"`
	Coeff float64 `json:"coeff"`
}

// Pop is the atom-resolved two-hole population of a state: one-site (A⁻²) and
// two-site (A⁻¹B⁻¹) weights keyed by group / group-pair name. Their sum equals
// PSPercent/100 to table rounding.
type Pop struct {
	OneSite map[string]float64 `json:"one_site,omitempty"`
	TwoSite map[string]float64 `json:"two_site,omitempty"`
}

// State is one final dicationic state.
type State struct {
	Index     int     `json:"index"`
	EnergyEV  float64 `json:"energy_ev"`
	PSPercent float64 `json:"ps_percent"`
	// Residue is the Ritz residual ‖M y − θ y‖ in eV: how far this state is from
	// being a true eigenpair. It is the solver's only in-band convergence signal —
	// use it to tell a converged line from an unconverged one instead of guessing at
	// -blocks. Exactly 0 on the dense path, which is exact by construction (hence no
	// omitempty: a zero here is a statement, not a missing value).
	Residue float64   `json:"residue"`
	Leading []Leading `json:"leading"`
	Pop     *Pop      `json:"pop,omitempty"`
}

// Sector is all states of one (irrep, spin) block.
type Sector struct {
	Irrep  int     `json:"irrep"` // 1-based
	Spin   int     `json:"spin"`  // 1 = singlet, 3 = triplet
	States []State `json:"states"`
}

// Options controls which roots and components are reported.
type Options struct {
	PSThresh    float64 // drop roots with pole strength below this (percent)
	CoeffThresh float64 // drop leading components with |coeff| below this
}

// spinLabel maps the internal spin (0/2) to the reported value (1/3).
func spinLabel(s dip.Spin) int {
	if s == dip.Triplet {
		return 3
	}
	return 1
}

// Sector assembles a sector's states from a solved result, ordered by energy,
// with spurious and weak roots dropped and leading components sorted. If pe is
// non-nil, each state also carries its atom-resolved two-hole population.
func BuildSector(sp *dip.Space, res lanczos.Result, opts Options, pe *PopEngine) Sector {
	main := sp.MainBlockSize()

	order := make([]int, len(res.Values))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return res.Values[order[a]] < res.Values[order[b]] })

	var states []State
	for _, k := range order {
		if res.Spurious(k, spurThresh) || res.PS[k] < opts.PSThresh {
			continue
		}
		var leading []Leading
		for c := range main {
			coeff := res.MainVecs.At(c, k)
			if coeff < opts.CoeffThresh && coeff > -opts.CoeffThresh {
				continue
			}
			cfg := sp.Configs[c]
			leading = append(leading, Leading{I: cfg.Occ[0] + 1, J: cfg.Occ[1] + 1, Coeff: coeff})
		}
		sort.SliceStable(leading, func(a, b int) bool {
			return abs(leading[a].Coeff) > abs(leading[b].Coeff)
		})
		var pop *Pop
		if pe != nil {
			mv := make([]float64, main)
			for c := range main {
				mv[c] = res.MainVecs.At(c, k)
			}
			pop = pe.Compute(mv)
		}
		st := State{
			Index:     len(states) + 1,
			EnergyEV:  res.Values[k] * au2eV,
			PSPercent: res.PS[k],
			Leading:   leading,
			Pop:       pop,
		}
		if k < len(res.Residual) {
			st.Residue = res.Residual[k] * au2eV
		}
		states = append(states, st)
	}
	return Sector{Irrep: sp.Sym + 1, Spin: spinLabel(sp.Spin), States: states}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
