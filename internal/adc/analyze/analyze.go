// Package analyze turns a solved DIP-ADC(2) sector (Lanczos or dense Ritz pairs)
// into the state records ADCgo emits: double-ionization energies, pole strengths,
// leading two-hole configurations, and — when MO coefficients and the AO overlap
// are supplied — atom-resolved two-hole populations.
//
// Field semantics mirror ADCanalysis internal/model (State/Config/PopRow) so the
// output feeds the existing classify/spectrum path with only a thin adapter (M4).
package analyze

import (
	"bufio"
	"fmt"
	"io"
	"sort"

	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
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
	//
	// Same quantity as the "residue" column of theADCcode's adcdip*.out, which prints
	// it in a.u. (adc_analyzer.cpp:48); divide by au2eV to compare.
	Residue float64   `json:"residue"`
	Leading []Leading `json:"leading"`
	Pop     *Pop      `json:"pop,omitempty"`
	// Root is this state's position among the NON-SPURIOUS Ritz roots in energy order,
	// which is the number theADCcode prints and WriteRef reproduces. It is not Index:
	// Index renumbers the surviving states 1..N, while Root keeps the gaps left by roots
	// dropped for weak pole strength — adcdip1.out's last two states are 105 and 107.
	Root int `json:"root"`
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

// BuildSector assembles a sector's states from a solved result, ordered by energy,
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
	// root counts the non-spurious roots, incremented BEFORE the pole-strength test so the
	// gaps left by weak states survive into it. That is theADCcode's own numbering — see
	// State.Root — and the reason the two filters cannot be collapsed back into one test.
	root := 0
	for _, k := range order {
		if res.Spurious(k, spurThresh) {
			continue
		}
		root++
		if res.PS[k] < opts.PSThresh {
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
			Root:      root,
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

// theADCcode's state-list format (adc_analyzer.cpp). Every constant here is load-bearing:
// the block is diffed against reference output byte for byte, so a changed width silently
// turns a passing comparison into a wall of noise.
//
//   - refRule is 34 dashes under a one-space-indented title.
//   - refLabelCols right-aligns the configuration label, which with the colon and the
//     9-column coefficient makes every entry exactly 17 wide, so columns line up whether
//     the label is "<4,4|" or "<21,19|". Labels wider than 7 (three-digit orbitals) push
//     their own line out rather than being truncated, as the Fortran does.
//   - refPerLine is how many entries share a line before it wraps.
const (
	refTitle     = " Eigenvalue (eV), ps (%), residue"
	refRule      = " ----------------------------------"
	refLabelCols = 7
	refPerLine   = 6
)

// writeRefHeader writes the block title, its rule and the two blank lines under it.
func writeRefHeader(bw *bufio.Writer) {
	bw.WriteString(refTitle + "\n")
	bw.WriteString(refRule + "\n\n\n")
}

// writeRefState writes one state: its scalar line, the overlap heading, the coefficient
// list wrapped at refPerLine, and the blank line that separates it from the next.
//
// residueEV is converted BACK to atomic units here, because that is the unit theADCcode
// prints in the residue column (adc_analyzer.cpp:48) while State.Residue holds eV.
//
// Only main-space overlaps are written. theADCcode additionally prints an "Overlaps with
// satellite-space configurations:" list when it has the satellite amplitudes; ADCgo keeps
// only the main block of each Ritz vector unless Options.WantFull was set, so there is
// nothing to print and inventing the heading with an empty list would be worse than
// omitting it.
func writeRefState(bw *bufio.Writer, root int, energyEV, psPercent, residueEV float64,
	labels []string, coeffs []float64) {
	fmt.Fprintf(bw, " %d: %.6f, %.2f, %.6f\n", root, energyEV, psPercent, residueEV/au2eV)
	bw.WriteString(" Overlaps with main-space configurations:\n")
	for i, lab := range labels {
		fmt.Fprintf(bw, "%*s:%9.6f", refLabelCols, lab, coeffs[i])
		if (i+1)%refPerLine == 0 || i == len(labels)-1 {
			bw.WriteByte('\n')
		}
	}
	bw.WriteByte('\n')
}

// WriteRef writes the sector's states in theADCcode's own "Eigenvalue (eV), ps (%),
// residue" format, so an ADCgo run can be diffed straight against adcdip*.out instead of
// being compared through a hand-written converter.
//
// The two-hole labels are "<i,j|" with i >= j in 1-based MO numbering, matching Leading.
//
// Diffing against a reference file: expect whole states to differ by a GLOBAL SIGN. An
// eigenvector's sign is arbitrary and neither code fixes a gauge — reproducing
// testdata/reference/adcdip1.out gives 60/60 state lines byte-identical and all 228 overlap
// magnitudes equal, with 34 of the 60 states sign-flipped. Compare |coeff|, not coeff.
// That run is `-dip -solver lanczos -blocks 100 -sym 0 -spin singlet -ps-thresh 0.5
// -coeff-thresh 0.01` against testdata/reference/h2o_dzp.matched.fcidump: theADCcode's own
// thresholds there are 0.5 and 0.01, not ADCgo's 1.0 and 0.1 defaults, and a mismatched
// threshold shows up as missing states or missing overlaps rather than as wrong numbers.
func (s Sector) WriteRef(w io.Writer) error {
	bw := bufio.NewWriter(w)
	writeRefHeader(bw)
	for _, st := range s.States {
		labels := make([]string, len(st.Leading))
		coeffs := make([]float64, len(st.Leading))
		for i, c := range st.Leading {
			labels[i] = fmt.Sprintf("<%d,%d|", c.I, c.J)
			coeffs[i] = c.Coeff
		}
		writeRefState(bw, st.Root, st.EnergyEV, st.PSPercent, st.Residue, labels, coeffs)
	}
	return bw.Flush()
}
