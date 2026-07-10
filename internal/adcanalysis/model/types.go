// Package model holds the shared data types for the ADC decay-channel analyzer.
//
// These mirror the structures described in the implementation sketch
// (decay_analyzer_sketch.md, §4). Only the fields the v1 parser populates are
// included here; v2 (eigenvector / from-scratch population) types can be added
// alongside without disturbing these.
package model

// MO is one row of the molecular-orbital table printed at the top of an
// adcdip{i}.out file: an index, its symmetry (irrep) label, and its SCF
// orbital energy in atomic units.
type MO struct {
	Index    int
	Sym      int
	EnergyAU float64
}

// Config is a single leading two-hole component of a state, i.e. one
// "<i,j|: coeff" entry under "Overlaps with main-space configurations".
// I and J are MO indices (the two holes); Coeff is the (signed) overlap
// amplitude. By convention the file prints I >= J.
type Config struct {
	I, J  int
	Coeff float64
}

// PopRow is one row of the "ADC two-hole population analysis" table for a
// single final dicationic state. It is keyed by the group / group-pair names
// declared in the &popana block (e.g. "O", "O/H1", "H1/H2"). One-site keys
// (no slash) are the A^-2 weights; two-site keys ("X/Y") are the A^-1 B^-1
// weights. Per the sketch (§1) these weights sum, per row, to the state's
// total two-hole pole strength ~= ps/100.
type PopRow struct {
	EnergyEV float64
	OneSite  map[string]float64 // "O"    -> A^-2 weight
	TwoSite  map[string]float64 // "O/H1" -> A^-1 B^-1 weight
}

// Sum returns the total two-hole population of the row (one-site + two-site).
// This should equal the matching state's ps/100 to within table rounding.
func (p PopRow) Sum() float64 {
	var s float64
	for _, v := range p.OneSite {
		s += v
	}
	for _, v := range p.TwoSite {
		s += v
	}
	return s
}

// AtomGroup is one atomic site (or symmetry-equivalent set of basis functions)
// declared in the &popana block of dip.in, e.g. {Name:"O", Orbitals: 1..19}.
// The Name is what appears as a one-site column ("O") and, paired, as the
// two-site columns ("O/H1") in the popana table. Orbitals is carried for the
// v2 from-scratch path and may be empty in v1.
type AtomGroup struct {
	Name     string
	Orbitals []int
}

// Site is a user-defined decay unit: a named group of one or more popana
// population columns (the OneSite atom names like "O", "H1"). It is the unit
// channels are resolved against — Auger@A, ICD:A->B, ETMD between sites.
//
// Grouping several columns into one Site folds the two-hole weight *between*
// those columns into the Site's one-site total. That fold is exactly what lets
// ETMD(2) (both final holes on a single neighbour site) be told apart from
// ETMD(3) (the two holes on two different neighbour sites): without the
// grouping, two holes on two atoms of the same physical unit would be
// miscounted as a three-centre process.
//
// Members lists the popana columns the Site covers. A Site with no Members is
// treated as covering the single column equal to its Name (the identity /
// "each column is its own site" default).
//
// Passive is a subset of Members whose holes are discounted from the spectrum:
// the columns still belong to the Site (so a two-hole configuration spanning a
// Passive and an active column of the same Site still folds into this one Site),
// but each hole landing on a Passive column is not counted. A two-hole
// configuration is scaled by the fraction of its holes on *active* columns, so a
// config with one hole on an active column and one on a Passive column counts at
// half weight, and a config with both holes on Passive columns drops out
// entirely. This expresses "ionize this unit but only credit the holes that
// stay on the heavy atom" — e.g. a water Site {O, ~H1, ~H2} whose Auger weight
// counts the O-localized holes but discounts the hydrogens. See classify.Discount.
type Site struct {
	Name    string
	Members []string
	Passive []string
}

// Channel is the contribution of a single decay channel to one state's
// intensity. Name is a stable, human-readable label:
//
//	"Auger@O"     both holes on the initial atom A (local Auger)
//	"ICD:O->H1"   one hole on A, one on neighbour B (interatomic Coulombic decay)
//	"ETMD"        no hole on A (electron-transfer-mediated decay / spectator)
//
// Weight is the two-hole population routed to that channel, which in v1 is the
// per-channel intensity used to build the stick spectrum.
type Channel struct {
	Name   string
	Weight float64
}

// State is one final dicationic ("DIP") state from a real diagonalization
// pass, joined to its two-hole population row.
type State struct {
	Irrep     int      // symmetry block (1..N), from "symmetry X"
	Spin      int      // 1 = singlet, 3 = triplet, from "spin Y"
	Index     int      // root number as printed (NOT contiguous: gaps exist)
	EnergyEV  float64  // double-ionization energy, eV
	PSPercent float64  // pole strength, percent
	Residue   float64  // residue column (0 for real states)
	Leading   []Config // parsed <i,j|: coeff list, in printed order
	Pop       *PopRow  // matching popana row; nil if none could be joined
}

// OrbWeight is one one-hole main-space overlap "<i|: coeff" of a SIP state:
// the (signed) amplitude for removing an electron from molecular orbital
// Orbital. Squared, it is that orbital's contribution to the state's pole
// strength (Σ Coeff² over the main-space overlaps ≈ ps/100).
type OrbWeight struct {
	Orbital int
	Coeff   float64
}

// SIPState is one final cationic state from a single-ionization (ndadc3ip) run.
// It mirrors State but carries one-hole main-space overlaps instead of two-hole
// configurations, and has no joined population row: the per-orbital
// decomposition is the Main list itself. Spin is the run's doublet value (2).
type SIPState struct {
	Irrep     int         // symmetry block (1..N), from "symmetry X"
	Spin      int         // 2 = doublet (single ionization)
	Index     int         // root number as printed (NOT contiguous: gaps exist)
	EnergyEV  float64     // single-ionization energy, eV
	PSPercent float64     // pole strength, percent
	Residue   float64     // residue column (0 for real states)
	Main      []OrbWeight // one-hole main-space overlaps, in printed order
}

// SIPOutFile is the parsed contents of a single-ionization ADC.out file. Unlike
// the per-irrep adcdip{i}.out of a DIP run, one ADC.out holds every symmetry
// block, so States spans all irreps.
type SIPOutFile struct {
	MOTable []MO       // the MO table at the top of the file
	States  []SIPState // states across all symmetry blocks
}

// OutFile is the parsed contents of one adcdip{i}.out file.
type OutFile struct {
	Path     int       // symmetry/irrep number this file covers
	Symmetry int       // same value, named for clarity (from the headers)
	MOTable  []MO      // the MO table at the top of the file
	States   []State   // real states only (eigenvector-save passes excluded)
	Groups   PopGroups // group names discovered in the popana header
}

// PopGroups records the one-site and two-site group names found in the popana
// table header, in the order they were printed.
type PopGroups struct {
	OneSite []string // e.g. ["O", "H1", "H2"]
	TwoSite []string // e.g. ["O/H1", "O/H2", "H1/H2"]
}
