package spectrum

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
)

// DIPOptions configures BuildDIP. Classify carries the channel thresholds; the
// remaining fields populate Meta. SingletTripletRatio is recorded for the
// plotting layer only — intensities are never pre-weighted by it here.
type DIPOptions struct {
	InitialAtom         string
	InitialOrbital      string
	Classify            Options
	SingletTripletRatio float64
	Molecule            string
	Basis               string
	PointGroup          string
	SourceFiles         []string
}

// BuildDIP classifies every state in every solved sector against the initial
// site and flattens the result into a Spectrum. Each state's population is first
// discounted for passive columns and folded into the user-defined sites
// (Regroup) before classification. States with no atom-resolved population
// (Pop == nil, i.e. the sector was solved without an MO sidecar) are skipped and
// their count returned so callers can warn.
//
// sites is the decay-unit grouping of the population columns (each column its own
// site by default); it fixes the canonical channel ordering and is validated to
// contain opts.InitialAtom.
func BuildDIP(secs []analyze.Sector, sites []Site, opts DIPOptions) (spec *Spectrum, skipped int, err error) {
	if err := ValidateInitialAtom(opts.InitialAtom, sites); err != nil {
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
			Irreps:              dipIrrepLabels(secs),
			EnergyUnit:          "eV",
			SingletTripletRatio: opts.SingletTripletRatio,
			SourceFiles:         opts.SourceFiles,
		},
		Channels: canonicalChannels(opts.InitialAtom, sites),
	}

	passive := PassiveSet(sites)
	for _, sec := range secs {
		for i := range sec.States {
			s := &sec.States[i]
			if s.Pop == nil {
				skipped++
				continue
			}
			ref := fmt.Sprintf("irrep%d/s%d/#%d", sec.Irrep, sec.Spin, s.Index)
			row := Row{OneSite: s.Pop.OneSite, TwoSite: s.Pop.TwoSite, EnergyEV: s.EnergyEV}
			row = Regroup(Discount(row, passive), sites)
			for _, ch := range Classify(opts.InitialAtom, sites, row, opts.Classify) {
				spec.Lines = append(spec.Lines, Line{
					Energy:    s.EnergyEV,
					Intensity: ch.Weight,
					Channel:   ch.Name,
					Spin:      sec.Spin,
					Irrep:     sec.Irrep,
					StateRef:  ref,
					PSPercent: s.PSPercent,
				})
			}
		}
	}
	return spec, skipped, nil
}

// SIPOptions configures BuildSIP. The fields populate Meta; single ionization
// has no decay-channel thresholds, sites, or singlet:triplet ratio.
type SIPOptions struct {
	Molecule    string
	Basis       string
	PointGroup  string
	SourceFiles []string
}

// BuildSIP flattens single-ionization sectors into a Spectrum decomposed per
// occupied molecular orbital, reusing the same JSON schema as BuildDIP so the
// plotting layer needs no special case.
//
// One Line is emitted per (state, orbital): the state's ionization energy with
// that orbital's squared one-hole amplitude (Coeff²) as the intensity, and the
// orbital label ("MO 3 (sym 1)") as the channel. Summed over a state's orbitals
// these intensities recover its spectroscopic factor (ps/100). orbSym is the
// per-orbital symmetry label (1-based MO index into a 0-based slice); a missing
// entry falls back to sym 0.
func BuildSIP(secs []analyze.SIPSector, orbSym []int, opts SIPOptions) (*Spectrum, error) {
	var nStates int
	for _, sec := range secs {
		nStates += len(sec.States)
	}
	if nStates == 0 {
		return nil, fmt.Errorf("no states to build")
	}

	symOf := func(orbital int) int {
		if i := orbital - 1; i >= 0 && i < len(orbSym) {
			return orbSym[i]
		}
		return 0
	}

	spec := &Spectrum{
		Meta: Meta{
			Kind:        "sip",
			Molecule:    opts.Molecule,
			Basis:       opts.Basis,
			PointGroup:  opts.PointGroup,
			Irreps:      sipIrrepLabels(secs),
			EnergyUnit:  "eV",
			SourceFiles: opts.SourceFiles,
		},
	}

	// Canonical channels: every orbital that occurs in a main-space overlap,
	// ordered by MO index.
	seen := map[int]bool{}
	var orbs []int
	for _, sec := range secs {
		for i := range sec.States {
			for _, w := range sec.States[i].Main {
				if !seen[w.Orbital] {
					seen[w.Orbital] = true
					orbs = append(orbs, w.Orbital)
				}
			}
		}
	}
	sort.Ints(orbs)
	for _, o := range orbs {
		spec.Channels = append(spec.Channels, orbLabel(o, symOf(o)))
	}

	for _, sec := range secs {
		for i := range sec.States {
			s := &sec.States[i]
			ref := fmt.Sprintf("irrep%d/s%d/#%d", sec.Irrep, sec.Spin, s.Index)
			for _, w := range s.Main {
				spec.Lines = append(spec.Lines, Line{
					Energy:    s.EnergyEV,
					Intensity: w.Coeff * w.Coeff,
					Channel:   orbLabel(w.Orbital, symOf(w.Orbital)),
					Spin:      sec.Spin,
					Irrep:     sec.Irrep,
					StateRef:  ref,
					PSPercent: s.PSPercent,
				})
			}
		}
	}
	return spec, nil
}

// BareOptions populates the Meta of a bare-eigenvalue spectrum (BuildBareDIP /
// BuildBareSIP). It carries only run-level annotations; the bare spectrum has no
// decay-channel thresholds, sites, initial site, or singlet:triplet ratio.
type BareOptions struct {
	Molecule    string
	Basis       string
	PointGroup  string
	SourceFiles []string
}

// bareChannel is the single channel label a bare-eigenvalue spectrum uses: every
// state is one stick on this one channel.
const bareChannel = "states"

// bareState is the minimal per-state information a bare spectrum needs — the same
// fields DIP and SIP states share, so the two sector types feed one core.
type bareState struct {
	EnergyEV, PSPercent float64
	Spin, Irrep, Index  int
}

// BuildBareDIP flattens solved DIP sectors into a bare-eigenvalue stick spectrum:
// one Line per state, energy = EnergyEV, intensity = PSPercent/100, all on a single
// "states" channel. Unlike BuildDIP it needs no MO sidecar or decay-channel
// classification, so it turns the plain eigenvalue document into a plottable
// Spectrum (same schema the renderer consumes).
func BuildBareDIP(secs []analyze.Sector, opts BareOptions) *Spectrum {
	var states []bareState
	for _, sec := range secs {
		for i := range sec.States {
			s := &sec.States[i]
			states = append(states, bareState{
				EnergyEV: s.EnergyEV, PSPercent: s.PSPercent,
				Spin: sec.Spin, Irrep: sec.Irrep, Index: s.Index,
			})
		}
	}
	return buildBare(states, "dip", dipIrrepLabels(secs), opts)
}

// BuildBareSIP is the single-ionization counterpart of BuildBareDIP: one Line per
// SIP state, decomposed by nothing (no per-orbital split, unlike BuildSIP).
func BuildBareSIP(secs []analyze.SIPSector, opts BareOptions) *Spectrum {
	var states []bareState
	for _, sec := range secs {
		for i := range sec.States {
			s := &sec.States[i]
			states = append(states, bareState{
				EnergyEV: s.EnergyEV, PSPercent: s.PSPercent,
				Spin: sec.Spin, Irrep: sec.Irrep, Index: s.Index,
			})
		}
	}
	return buildBare(states, "sip", sipIrrepLabels(secs), opts)
}

// buildBare is the shared core: it emits one stick per flattened state on the single
// bareChannel, mirroring the StateRef format and fraction intensity of BuildDIP /
// BuildSIP so the bare spectrum plots identically as a one-channel curve.
func buildBare(states []bareState, kind string, irreps []string, opts BareOptions) *Spectrum {
	spec := &Spectrum{
		Meta: Meta{
			Kind:        kind,
			Molecule:    opts.Molecule,
			Basis:       opts.Basis,
			PointGroup:  opts.PointGroup,
			Irreps:      irreps,
			EnergyUnit:  "eV",
			SourceFiles: opts.SourceFiles,
		},
		Channels: []string{bareChannel},
	}
	for _, s := range states {
		spec.Lines = append(spec.Lines, Line{
			Energy:    s.EnergyEV,
			Intensity: s.PSPercent / 100,
			Channel:   bareChannel,
			Spin:      s.Spin,
			Irrep:     s.Irrep,
			StateRef:  fmt.Sprintf("irrep%d/s%d/#%d", s.Irrep, s.Spin, s.Index),
			PSPercent: s.PSPercent,
		})
	}
	return spec
}

// canonicalChannels lists every channel the classifier can produce for these
// sites, in the same stable order Classify emits them: Auger@A, one ICD:A->B per
// other site (in site order), then ETMD(2) and ETMD(3). ETMD channels are only
// listed when the grouping can support them (ETMD(2) needs >=1 site other than
// A, ETMD(3) needs >=2), matching Classify's gating.
func canonicalChannels(initialAtom string, sites []Site) []string {
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

// orbLabel names an orbital channel, e.g. "MO 3 (sym 1)".
func orbLabel(orbital, sym int) string {
	return fmt.Sprintf("MO %d (sym %d)", orbital, sym)
}

// dipIrrepLabels returns the sorted distinct irrep numbers across the DIP
// sectors, as strings.
func dipIrrepLabels(secs []analyze.Sector) []string {
	seen := map[int]bool{}
	for _, sec := range secs {
		seen[sec.Irrep] = true
	}
	return sortedLabels(seen)
}

// sipIrrepLabels returns the sorted distinct irrep numbers across the SIP
// sectors, as strings.
func sipIrrepLabels(secs []analyze.SIPSector) []string {
	seen := map[int]bool{}
	for _, sec := range secs {
		seen[sec.Irrep] = true
	}
	return sortedLabels(seen)
}

func sortedLabels(seen map[int]bool) []string {
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
