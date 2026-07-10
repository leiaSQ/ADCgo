package analyze

import (
	"sort"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// OrbWeight is one one-hole main-space overlap "<i|: coeff" of a SIP state: the
// (signed) effective amplitude for removing an electron from MO Orbital (1-based).
// Squared and summed, these give the state's spectroscopic factor (PSPercent/100).
type OrbWeight struct {
	Orbital int     `json:"orbital"`
	Coeff   float64 `json:"coeff"`
}

// SIPState is one final cationic (doublet) state from a single-ionization run:
// its ionization energy, spectroscopic factor, and one-hole main-space overlaps.
type SIPState struct {
	Index     int         `json:"index"`
	EnergyEV  float64     `json:"energy_ev"`
	PSPercent float64     `json:"ps_percent"`
	Main      []OrbWeight `json:"main"`
}

// SIPSector is all states of one target-symmetry block.
type SIPSector struct {
	Irrep  int        `json:"irrep"` // 1-based
	Spin   int        `json:"spin"`  // 2 = doublet (single ionization)
	States []SIPState `json:"states"`
}

// BuildSIPSector assembles a sector's cationic states from a solved result,
// ordered by energy, with spurious/weak roots dropped. The spectroscopic
// amplitude of each state is a = F·Y (F the ND-ADC transition-amplitude matrix,
// Y the state's 1h main-block eigenvector part); the spectroscopic factor is
// 100·‖a‖² and the per-orbital overlaps are the components of a.
func BuildSIPSector(sp *sip.Space, res lanczos.Result, fmat backend.Mat, opts Options) SIPSector {
	sec, _ := buildSIPSector(sp, res, fmat, opts)
	return sec
}

// buildSIPSector is BuildSIPSector plus the raw solver column each surviving state
// came from (the index into res.Values / res.FullVecs). cols[i] belongs to
// sec.States[i]; the transition-moment layer (tdm.go) needs it to select the right
// Ritz vectors after energy ordering and spurious/weak filtering have reshuffled them.
func buildSIPSector(sp *sip.Space, res lanczos.Result, fmat backend.Mat, opts Options) (SIPSector, []int) {
	main := sp.MainBlockSize()

	order := make([]int, len(res.Values))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return res.Values[order[a]] < res.Values[order[b]] })

	var states []SIPState
	var cols []int
	for _, k := range order {
		if res.Spurious(k, spurThresh) {
			continue
		}
		// Effective one-hole amplitude a = F·Y over the main block.
		y := make([]float64, main)
		for c := range main {
			y[c] = res.MainVecs.At(c, k)
		}
		a := fmat.MulVec(y)

		var ps float64
		for _, v := range a {
			ps += v * v
		}
		ps *= 100
		if ps < opts.PSThresh {
			continue
		}

		var overlaps []OrbWeight
		for c := range main {
			if a[c] < opts.CoeffThresh && a[c] > -opts.CoeffThresh {
				continue
			}
			overlaps = append(overlaps, OrbWeight{Orbital: sp.Configs[c].Occ[0] + 1, Coeff: a[c]})
		}
		sort.SliceStable(overlaps, func(a, b int) bool {
			return abs(overlaps[a].Coeff) > abs(overlaps[b].Coeff)
		})

		states = append(states, SIPState{
			Index:     len(states) + 1,
			EnergyEV:  res.Values[k] * au2eV,
			PSPercent: ps,
			Main:      overlaps,
		})
		cols = append(cols, k)
	}
	return SIPSector{Irrep: sp.Sym + 1, Spin: 2, States: states}, cols
}
