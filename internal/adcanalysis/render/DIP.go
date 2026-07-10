// Package spectrum flattens parsed-and-classified states into the JSON stick
// spectrum that is the Go<->Python/MATLAB contract (decay_analyzer_sketch.md §6).
//
// One Line is emitted per (state, channel): the state's energy with that
// channel's two-hole population as the intensity. Energies stay un-convoluted
// (sticks) and intensities are the raw per-channel weights — Gaussian
// broadening and the singlet:triplet height weighting are deliberately left to
// the plotting layer, which reads singlet_triplet_ratio from the meta block.
package render

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/leiaSQ/ADCgo/internal/adcanalysis/classify"
	"github.com/leiaSQ/ADCgo/internal/adcanalysis/model"
)

// Ionization records the initial core-ionized site (and optionally the specific
// orbital, e.g. "1s") that channels are defined relative to.
type Ionization struct {
	Atom    string `json:"atom"`
	Orbital string `json:"orbital,omitempty"`
}

// Meta is the run-level header of the spectrum (schema §6). Fields that were
// not supplied are omitted rather than emitted blank.
type Meta struct {
	Kind                string     `json:"kind,omitempty"` // "dip" or "sip"
	Molecule            string     `json:"molecule,omitempty"`
	Basis               string     `json:"basis,omitempty"`
	PointGroup          string     `json:"point_group,omitempty"`
	InitialIonization   Ionization `json:"initial_ionization"`
	Irreps              []string   `json:"irreps"`
	EnergyUnit          string     `json:"energy_unit"`
	SingletTripletRatio float64    `json:"singlet_triplet_ratio"`
	SourceFiles         []string   `json:"source_files"`
}

// Line is one (state, channel) stick.
type Line struct {
	Energy    float64 `json:"energy"`
	Intensity float64 `json:"intensity"`
	Channel   string  `json:"channel"`
	Spin      int     `json:"spin"`
	Irrep     int     `json:"irrep"`
	StateRef  string  `json:"state_ref"`
	PSPercent float64 `json:"ps_percent"`
}

// Spectrum is the full JSON document.
type Spectrum struct {
	Meta     Meta     `json:"meta"`
	Channels []string `json:"channels"`
	Lines    []Line   `json:"lines"`
}

// DIPOptions configures a BuildDIP. Classify carries the channel thresholds; the
// remaining fields populate Meta. SingletTripletRatio is recorded for the
// plotting layer only — intensities are never pre-weighted by it here.
type DIPOptions struct {
	InitialAtom         string
	InitialOrbital      string
	Classify            classify.Options
	SingletTripletRatio float64
	Molecule            string
	Basis               string
	PointGroup          string
	SourceFiles         []string
}

// BuildDIP classifies every state in every parsed file against the initial site
// and flattens the result into a Spectrum. Each state's popana row is first
// folded into the user-defined sites (classify.Regroup) before classification.
// States with no joined population row are skipped (they carry no two-hole
// weight to route) and their count is returned so callers can warn.
//
// sites is the decay-unit grouping of the popana columns (each column its own
// site by default); it fixes the canonical channel ordering and is validated to
// contain opts.InitialAtom.
func BuildDIP(files []*model.OutFile, sites []model.Site, opts DIPOptions) (spec *Spectrum, skipped int, err error) {
	if err := classify.ValidateInitialAtom(opts.InitialAtom, sites); err != nil {
		return nil, 0, err
	}

	spec = &Spectrum{
		Meta: Meta{
			Kind:       "dip",
			Molecule:   opts.Molecule,
			Basis:      opts.Basis,
			PointGroup: opts.PointGroup,
			InitialIonization: Ionization{
				Atom:    opts.InitialAtom,
				Orbital: opts.InitialOrbital,
			},
			Irreps:              irrepLabels(files),
			EnergyUnit:          "eV",
			SingletTripletRatio: opts.SingletTripletRatio,
			SourceFiles:         opts.SourceFiles,
		},
		Channels: canonicalChannels(opts.InitialAtom, sites),
	}

	passive := classify.PassiveSet(sites)
	for _, f := range files {
		for i := range f.States {
			s := &f.States[i]
			if s.Pop == nil {
				skipped++
				continue
			}
			ref := fmt.Sprintf("irrep%d/s%d/#%d", s.Irrep, s.Spin, s.Index)
			row := classify.Regroup(classify.Discount(*s.Pop, passive), sites)
			for _, ch := range classify.Classify(opts.InitialAtom, sites, row, opts.Classify) {
				spec.Lines = append(spec.Lines, Line{
					Energy:    s.EnergyEV,
					Intensity: ch.Weight,
					Channel:   ch.Name,
					Spin:      s.Spin,
					Irrep:     s.Irrep,
					StateRef:  ref,
					PSPercent: s.PSPercent,
				})
			}
		}
	}
	return spec, skipped, nil
}

// canonicalChannels lists every channel the classifier can produce for these
// sites, in the same stable order Classify emits them: Auger@A, one ICD:A->B
// per other site (in site order), then ETMD(2) and ETMD(3). This is the full
// menu the plotting layer can choose to overlay/stack, independent of which
// channels a given state actually populates.
//
// ETMD channels are only listed when the grouping can support them: ETMD(2)
// lands both holes on a single neighbour site (needs >=1 site other than A) and
// ETMD(3) on two distinct neighbour sites (needs >=2). With a single site there
// is nowhere but A for the holes to go, so only Auger@A is possible — see
// classify.Classify, which gates the same way.
func canonicalChannels(initialAtom string, sites []model.Site) []string {
	out := []string{"Auger@" + initialAtom}
	others := 0
	for _, s := range sites {
		if s.Name == initialAtom {
			continue
		}
		others++
		out = append(out, "ICD:"+initialAtom+"->"+s.Name)
	}
	if others >= 1 {
		out = append(out, "ETMD(2)")
	}
	if others >= 2 {
		out = append(out, "ETMD(3)")
	}
	return out
}

// irrepLabels returns the sorted distinct irrep numbers present, as strings.
// (Mapping to point-group symbols like "a1" needs the &symmetry assignment,
// which v1 does not parse; the numbers are stable and unambiguous meanwhile.)
func irrepLabels(files []*model.OutFile) []string {
	seen := map[int]bool{}
	for _, f := range files {
		for _, s := range f.States {
			seen[s.Irrep] = true
		}
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
