package analyze

import (
	"bufio"
	"fmt"
	"io"
	"sort"

	"github.com/leiaSQ/ADCgo/backend"
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
	Index     int     `json:"index"`
	EnergyEV  float64 `json:"energy_ev"`
	PSPercent float64 `json:"ps_percent"`
	// Residue is the Ritz residual ‖M y − θ y‖ in eV, the DIP twin's quantity and the
	// solver's only in-band convergence signal. Exactly 0 on the dense path, which is
	// exact by construction. theADCcode prints it in a.u.; WriteRef converts back.
	Residue float64     `json:"residue"`
	Main    []OrbWeight `json:"main"`
	// Root is the position among the NON-SPURIOUS Ritz roots in energy order — the number
	// theADCcode prints, which keeps the gaps left by roots dropped for weak pole
	// strength. Index renumbers the survivors 1..N instead. See analyze.State.Root.
	Root int `json:"root"`
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
	// See analyze.State.Root: incremented before the pole-strength test so the gaps the
	// reference leaves for weak states are preserved.
	root := 0
	for _, k := range order {
		if res.Spurious(k, spurThresh) {
			continue
		}
		root++
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

		st := SIPState{
			Index:     len(states) + 1,
			Root:      root,
			EnergyEV:  res.Values[k] * au2eV,
			PSPercent: ps,
			Main:      overlaps,
		}
		if k < len(res.Residual) {
			st.Residue = res.Residual[k] * au2eV
		}
		states = append(states, st)
		cols = append(cols, k)
	}
	return SIPSector{Irrep: sp.Sym + 1, Spin: 2, States: states}, cols
}

// WriteRef writes the sector's cation states in theADCcode's "Eigenvalue (eV), ps (%),
// residue" format, the single-ionization twin of Sector.WriteRef and byte-compatible with
// it — the block structure, the 17-column entries and the atomic-unit residue are shared,
// and only the configuration label differs: one hole "<i|" here against the DIP "<i,j|".
func (s SIPSector) WriteRef(w io.Writer) error {
	bw := bufio.NewWriter(w)
	writeRefHeader(bw)
	for _, st := range s.States {
		labels := make([]string, len(st.Main))
		coeffs := make([]float64, len(st.Main))
		for i, o := range st.Main {
			labels[i] = fmt.Sprintf("<%d|", o.Orbital)
			coeffs[i] = o.Coeff
		}
		writeRefState(bw, st.Root, st.EnergyEV, st.PSPercent, st.Residue, labels, coeffs)
	}
	return bw.Flush()
}
