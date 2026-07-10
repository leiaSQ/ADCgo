package render

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/leiaSQ/ADCgo/internal/adcanalysis/model"
)

// SIPOptions configures BuildSIP. The fields populate Meta; single ionization
// has no decay-channel thresholds, sites, or singlet:triplet ratio.
type SIPOptions struct {
	Molecule    string
	Basis       string
	PointGroup  string
	SourceFiles []string
}

// BuildSIP flattens a single-ionization run into a Spectrum decomposed per
// occupied molecular orbital, reusing the same JSON schema as BuildDIP so
// the plotting layer (and a future combined mode) need no special case.
//
// One Line is emitted per (state, orbital): the state's ionization energy with
// that orbital's squared one-hole amplitude (Coeff²) as the intensity, and the
// orbital label ("MO 3 (sym 1)") as the channel. Summed over a state's orbitals
// these intensities recover its pole strength (ps/100). Channels list the
// orbitals that occur, ordered by MO index, so the plotting layer overlays one
// stick series per orbital — the per-orbital "stacking" of mk_ref_spec_*.
func BuildSIP(f *model.SIPOutFile, opts SIPOptions) (*Spectrum, error) {
	if f == nil || len(f.States) == 0 {
		return nil, fmt.Errorf("no states to build")
	}

	symOf := make(map[int]int, len(f.MOTable))
	for _, mo := range f.MOTable {
		symOf[mo.Index] = mo.Sym
	}

	spec := &Spectrum{
		Meta: Meta{
			Kind:        "sip",
			Molecule:    opts.Molecule,
			Basis:       opts.Basis,
			PointGroup:  opts.PointGroup,
			Irreps:      sipIrrepLabels(f),
			EnergyUnit:  "eV",
			SourceFiles: opts.SourceFiles,
		},
	}

	// Canonical channels: every orbital that occurs in a main-space overlap,
	// ordered by MO index.
	seen := map[int]bool{}
	var orbs []int
	for _, s := range f.States {
		for _, w := range s.Main {
			if !seen[w.Orbital] {
				seen[w.Orbital] = true
				orbs = append(orbs, w.Orbital)
			}
		}
	}
	sort.Ints(orbs)
	for _, o := range orbs {
		spec.Channels = append(spec.Channels, orbLabel(o, symOf[o]))
	}

	for i := range f.States {
		s := &f.States[i]
		ref := fmt.Sprintf("irrep%d/s%d/#%d", s.Irrep, s.Spin, s.Index)
		for _, w := range s.Main {
			spec.Lines = append(spec.Lines, Line{
				Energy:    s.EnergyEV,
				Intensity: w.Coeff * w.Coeff,
				Channel:   orbLabel(w.Orbital, symOf[w.Orbital]),
				Spin:      s.Spin,
				Irrep:     s.Irrep,
				StateRef:  ref,
				PSPercent: s.PSPercent,
			})
		}
	}
	return spec, nil
}

// orbLabel names an orbital channel, e.g. "MO 3 (sym 1)".
func orbLabel(orbital, sym int) string {
	return fmt.Sprintf("MO %d (sym %d)", orbital, sym)
}

// sipIrrepLabels returns the sorted distinct irrep numbers present, as strings.
func sipIrrepLabels(f *model.SIPOutFile) []string {
	seen := map[int]bool{}
	for _, s := range f.States {
		seen[s.Irrep] = true
	}
	nums := make([]int, 0, len(seen))
	for n := range seen {
		nums = append(nums, n)
	}
	sort.Ints(nums)
	labels := make([]string, len(nums))
	for i, n := range nums {
		labels[i] = strconv.Itoa(n)
	}
	return labels
}
