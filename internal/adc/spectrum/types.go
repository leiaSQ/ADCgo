// Package spectrum turns solved ADCgo sectors into the decay-channel stick
// spectrum JSON. It absorbs the classify + spectrum layer of the sibling tool
// ADCanalysis: each dicationic (DIP) state's atom-resolved two-hole population is
// routed onto named decay channels (Auger / ICD / ETMD) relative to a chosen
// initial core-ionized site, and flattened into one stick per (state, channel).
// Single-ionization (SIP) states flatten per occupied orbital instead.
//
// The emitted JSON matches ADCanalysis's schema field-for-field (Meta / Channels
// / Lines), so its plotspec renderer and reference spectra consume ADCgo output
// unchanged. Unlike ADCanalysis, which parses text output, this package consumes
// the in-memory analyze.Sector / analyze.SIPSector directly — there is no text
// round-trip.
package spectrum

// Site is a user-defined decay unit: a named group of one or more population
// columns (the analyze.Pop atom names like "O", "H1"). It is the unit channels
// are resolved against — Auger@A, ICD:A->B, ETMD between sites.
//
// Grouping several columns into one Site folds the two-hole weight *between*
// those columns into the Site's one-site total. That fold is what lets ETMD(2)
// (both final holes on a single neighbour site) be told apart from ETMD(3) (the
// two holes on two different neighbour sites).
//
// Members lists the columns the Site covers. A Site with no Members is treated
// as covering the single column equal to its Name (the identity default).
//
// Passive is a subset of Members whose holes are discounted from the spectrum:
// the columns still belong to the Site, but each hole landing on a Passive
// column is not counted, so a config is scaled by the fraction of its holes on
// active columns (see Discount).
type Site struct {
	Name    string
	Members []string
	Passive []string
}

// Channel is the contribution of a single decay channel to one state's
// intensity: a stable label ("Auger@O", "ICD:O->H1", "ETMD(2)", "ETMD(3)") and
// the two-hole population routed to it.
type Channel struct {
	Name   string
	Weight float64
}

// Row is a state's two-hole population decomposition, keyed by group / group-pair
// name. One-site keys (no slash) are the A⁻² weights; two-site keys ("O/H1") are
// the A⁻¹B⁻¹ weights. It mirrors analyze.Pop plus a carried energy, so the
// classify routines that ADCanalysis proved on model.PopRow apply unchanged.
type Row struct {
	EnergyEV float64
	OneSite  map[string]float64
	TwoSite  map[string]float64
}

// Sum returns the total two-hole population of the row (one-site + two-site).
// This equals the matching state's ps/100 to rounding.
func (r Row) Sum() float64 {
	var s float64
	for _, v := range r.OneSite {
		s += v
	}
	for _, v := range r.TwoSite {
		s += v
	}
	return s
}

// Ionization records the initial core-ionized site (and optionally the specific
// orbital) that channels are defined relative to.
type Ionization struct {
	Atom    string `json:"atom"`
	Orbital string `json:"orbital,omitempty"`
}

// Meta is the run-level header of the spectrum. Fields that were not supplied
// are omitted rather than emitted blank.
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
