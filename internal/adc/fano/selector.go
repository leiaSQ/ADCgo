package fano

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// selector.go — the paper's scheme A, hole localization (ADC22.pdf §III B 1, p. 6).
//
// A configuration goes to Q or P by a predicate on its holes alone. No basis
// transformation is involved, so QMQ and PMP are index sub-blocks of M and the
// Feshbach projectors are exact: Q + P = 1 and QP = 0, by construction.
//
// This is what ../ADC/adc2_pol/select_fano.f90 implements. Two things are worth
// recording about that code, because both are easy to mis-transfer:
//
//   - The energy criteria in select_atom_d are DEAD. Every `if (einit .le. ...)` in
//     it has the identical body in both branches (select_fano.f90:261-271, 308-318,
//     423-433), so the selection is purely by hole locality. Only select_singles,
//     which the Fano driver does not call, actually branches on energy.
//   - select_atom_is and select_atom_fs are NOT complementary. The initial space
//     takes holes from hcentre, the final space from hneighb, and the mixed cases —
//     one hole from each — are commented out ("aukommentier -> Ne sonst HF",
//     select_fano.f90:438-497). That is why partgammas.f90:648 has to warn about
//     non-orthogonal subspaces. This package partitions instead: P is the complement
//     of Q, so the subspaces are orthogonal by construction and the -E_Phi delta term
//     in the coupling vanishes identically rather than approximately.

// HoleRule selects between the two readings of the scheme A predicate. The paper uses
// both, for different decay processes, and they are not interchangeable.
type HoleRule int

const (
	// AllHoles puts a configuration in Q only when EVERY one of its holes lies in the
	// Q orbital set. This is the "all holes localized on subunit A" criterion, and the
	// right one for interatomic/intermolecular decay: in ICD between subunits A and B
	// the initial vacancy is on A, and the final state has one hole on A and one on B,
	// so it is the presence of a hole OUTSIDE A that marks a decay channel.
	//
	// It is also what select_atom_is implements, taking every hole from hcentre.
	AllHoles HoleRule = iota

	// AnyHole puts a configuration in Q when AT LEAST ONE of its holes lies in the Q
	// orbital set. This is the "retains the initial vacancy" criterion, and the right
	// one for local decay: in Ne 1s Auger the initial state is 1s^-1 and the final
	// states are 2p^-2 + e^-, so a configuration still carrying the 1s hole has not
	// decayed, whatever its other holes are, while one that has filled it has.
	AnyHole
)

func (r HoleRule) String() string {
	if r == AnyHole {
		return "any hole in"
	}
	return "all holes in"
}

// HoleLocalization is the scheme A selector: Q is defined by a set of occupied
// orbitals and one of the two rules above.
type HoleLocalization struct {
	rule  HoleRule
	orbs  []int // ascending, distinct, absolute 0-based occupied indices
	label string
}

// NewHoleLocalization builds the selector. orbitals are absolute 0-based occupied
// orbital indices; label is an optional human name for the set ("core", "site A")
// used in the run log, and may be empty.
//
// An empty orbital set is rejected rather than silently accepted: under AllHoles it
// would make Q empty (no bound state, so no width), and under AnyHole it would make Q
// empty as well. Either way the run cannot produce a rate, and failing here says so
// while the caller still knows which flag was wrong.
func NewHoleLocalization(rule HoleRule, orbitals []int, label string) (*HoleLocalization, error) {
	if len(orbitals) == 0 {
		return nil, fmt.Errorf("fano: empty Q orbital set; scheme A needs at least one orbital " +
			"to define the bound subspace")
	}
	o := slices.Clone(orbitals)
	slices.Sort(o)
	o = slices.Compact(o)
	for _, x := range o {
		if x < 0 {
			return nil, fmt.Errorf("fano: negative Q orbital index %d", x)
		}
	}
	return &HoleLocalization{rule: rule, orbs: o, label: label}, nil
}

// Bound reports whether a configuration with these holes belongs to Q.
//
// A configuration with no holes at all cannot occur in an ionization space (every
// class has at least one), but were it to, it is not bound: AllHoles would otherwise
// admit it vacuously.
func (h *HoleLocalization) Bound(holes []int) bool {
	if len(holes) == 0 {
		return false
	}
	if h.rule == AnyHole {
		for _, x := range holes {
			if slices.Contains(h.orbs, x) {
				return true
			}
		}
		return false
	}
	for _, x := range holes {
		if !slices.Contains(h.orbs, x) {
			return false
		}
	}
	return true
}

// Orbitals returns the Q orbital set.
func (h *HoleLocalization) Orbitals() []int { return slices.Clone(h.orbs) }

// String describes the criterion, for the run log.
func (h *HoleLocalization) String() string {
	var b strings.Builder
	b.WriteString("hole localization (scheme A): ")
	b.WriteString(h.rule.String())
	b.WriteString(" {")
	for i, o := range h.orbs {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "%d", o)
	}
	b.WriteString("}")
	if h.label != "" {
		fmt.Fprintf(&b, " [%s]", h.label)
	}
	return b.String()
}

// Selector decides, per configuration, whether it belongs to the bound subspace Q.
// P is always the complement, so a Selector defines a genuine Feshbach partition.
type Selector interface {
	// Bound reports whether a configuration with these holes is in Q. It must be a
	// pure function of holes — NewPartition evaluates it concurrently over the rows.
	Bound(holes []int) bool
	// String describes the criterion for the run log.
	String() string
}

// ---------------------------------------------------------------------------
// Per-excitation-class rules — scheme A in its general form.
// ---------------------------------------------------------------------------
//
// HoleLocalization asks one question of every configuration, whatever its excitation
// class. Two of the four atoms in the paper's Table V need more than that, and the
// supplementary material states them per class:
//
//	Mg+(2s^-1): "P subspace spanned by 2h1p ISs with at least one 3s hole and by 3h2p
//	             ISs with two 3s holes, with the exception of ISs with at least one 2s
//	             vacancy, which belong naturally to the Q subspace"
//	Kr+(3d^-1): "Q subspace is defined by configurations with at least one 3d vacancy
//	             plus all 3h2p ISs associated with Kr+(4s^-2 4p^-1 nl^+2) configurations"
//
// Neither is a predicate on the hole set alone: Mg's depends on whether the
// configuration is 2h1p or 3h2p, and asks for a hole COUNT of two rather than presence;
// Kr's adds a whole class-specific family to Q on top of an any-hole rule.
//
// Both matter physically rather than cosmetically. Mg's 2p^-2 channel lies ABOVE its 2s
// vacancy and is therefore closed, so 2p^-2 configurations must stay in Q; the rule
// above achieves that by naming only 3s-hole configurations as continuum. And the paper
// is explicit that Mg's 3h2p ISs from 2p^-1 3s^-2 "belong to both P and Q subspaces"
// physically — they carry both an open shake-up channel and a closed triply-ionized one
// — and that it assigned them to P alone "as it is paramount in the Fano theory to fully
// eliminate continuum from the Q subspace", which is the choice it blames for Mg being
// its worst case. Getting that assignment right is a prerequisite for reproducing the
// number at all.

// Term is a count predicate on one configuration's holes. Class is the excitation class
// as a hole count — 1 is 1h, 2 is 2h1p, 3 is 3h2p — and 0 matches every class. The Term
// holds when the number of holes lying in Set is at least Min and at most Max; a
// negative Max is unbounded.
//
// Counting rather than testing membership is what expresses "two 3s holes" (Min = 2) as
// distinct from "at least one 3s hole" (Min = 1). A doubly occupied orbital emptied twice
// appears twice in the hole list, which is what makes the count meaningful.
type Term struct {
	Class    int
	Set      []int
	Min, Max int
}

func (t Term) match(holes []int) bool {
	if t.Class != 0 && len(holes) != t.Class {
		return false
	}
	n := 0
	for _, h := range holes {
		if slices.Contains(t.Set, h) {
			n++
		}
	}
	return n >= t.Min && (t.Max < 0 || n <= t.Max)
}

func (t Term) String() string {
	var b strings.Builder
	if t.Class != 0 {
		fmt.Fprintf(&b, "%dh: ", t.Class)
	}
	b.WriteString("holes in {")
	for i, o := range t.Set {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "%d", o)
	}
	b.WriteString("}")
	switch {
	case t.Max < 0:
		fmt.Fprintf(&b, " >= %d", t.Min)
	case t.Min == t.Max:
		fmt.Fprintf(&b, " == %d", t.Min)
	default:
		fmt.Fprintf(&b, " in [%d,%d]", t.Min, t.Max)
	}
	return b.String()
}

// Clause is a conjunction of Terms. An empty Clause never matches, so that a malformed
// spec cannot silently become "always true".
type Clause []Term

func (c Clause) match(holes []int) bool {
	if len(c) == 0 {
		return false
	}
	for _, t := range c {
		if !t.match(holes) {
			return false
		}
	}
	return true
}

func (c Clause) String() string {
	parts := make([]string, len(c))
	for i, t := range c {
		parts[i] = t.String()
	}
	return strings.Join(parts, " and ")
}

// ClassRule is the general scheme A selector: a configuration is bound when some Q
// clause matches it, or when P clauses were given and none of them matches it.
//
// The two halves mirror the two ways the supplementary material phrases a partition.
// Ne and Ar are stated as Q rules ("all ISs characterized by at least one 1s vacancy
// belong to Q"), which is Q alone with P left unrestricted. Mg is stated as a P rule
// with a Q exception, which is both. Kr is a Q rule with an extra class-specific family,
// which is two Q clauses. P remains the exact complement of Q in every case, so QP = 0
// still holds by construction and the coupling's -E_Phi delta term still vanishes.
type ClassRule struct {
	q, p  []Clause
	label string
	spec  string
}

// NewClassRule builds the selector. At least one clause is required overall: with
// neither Q nor P clauses every configuration would land in P, leaving no bound state to
// select and hence no width.
func NewClassRule(q, p []Clause, label string) (*ClassRule, error) {
	if len(q) == 0 && len(p) == 0 {
		return nil, fmt.Errorf("fano: empty Q/P rule; scheme A needs at least one clause " +
			"to define the bound subspace")
	}
	for _, set := range [][]Clause{q, p} {
		for _, c := range set {
			if len(c) == 0 {
				return nil, fmt.Errorf("fano: empty clause in the Q/P rule")
			}
			for _, t := range c {
				if len(t.Set) == 0 {
					return nil, fmt.Errorf("fano: term with an empty orbital set")
				}
				if t.Min < 0 {
					return nil, fmt.Errorf("fano: negative Min %d in a Q/P term", t.Min)
				}
				if t.Max >= 0 && t.Max < t.Min {
					return nil, fmt.Errorf("fano: Max %d below Min %d in a Q/P term", t.Max, t.Min)
				}
				if t.Class < 0 {
					return nil, fmt.Errorf("fano: negative excitation class %d in a Q/P term", t.Class)
				}
				for _, o := range t.Set {
					if o < 0 {
						return nil, fmt.Errorf("fano: negative orbital index %d in a Q/P term", o)
					}
				}
			}
		}
	}
	return &ClassRule{q: q, p: p, label: label}, nil
}

// Bound reports whether a configuration with these holes belongs to Q.
func (r *ClassRule) Bound(holes []int) bool {
	if len(holes) == 0 {
		return false
	}
	for _, c := range r.q {
		if c.match(holes) {
			return true
		}
	}
	if len(r.p) == 0 {
		return false
	}
	for _, c := range r.p {
		if c.match(holes) {
			return false
		}
	}
	return true
}

// String describes the criterion for the run log.
func (r *ClassRule) String() string {
	var b strings.Builder
	b.WriteString("class rule (scheme A): Q if ")
	if len(r.q) == 0 {
		b.WriteString("never")
	} else {
		parts := make([]string, len(r.q))
		for i, c := range r.q {
			parts[i] = "(" + c.String() + ")"
		}
		b.WriteString(strings.Join(parts, " or "))
	}
	if len(r.p) > 0 {
		parts := make([]string, len(r.p))
		for i, c := range r.p {
			parts[i] = "(" + c.String() + ")"
		}
		b.WriteString(", else Q unless P: " + strings.Join(parts, " or "))
	}
	if r.spec != "" {
		fmt.Fprintf(&b, " [spec %s]", r.spec)
	}
	if r.label != "" {
		fmt.Fprintf(&b, " [%s]", r.label)
	}
	return b.String()
}

// parseOrbSet reads "0,1,2" or "9-13" or a mixture into an ascending distinct set.
func parseOrbSet(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		if lo, hi, ok := strings.Cut(f, "-"); ok {
			a, err := strconv.Atoi(strings.TrimSpace(lo))
			if err != nil {
				return nil, fmt.Errorf("bad orbital range start %q", lo)
			}
			b, err := strconv.Atoi(strings.TrimSpace(hi))
			if err != nil {
				return nil, fmt.Errorf("bad orbital range end %q", hi)
			}
			if b < a {
				return nil, fmt.Errorf("orbital range %q runs backwards", f)
			}
			for x := a; x <= b; x++ {
				out = append(out, x)
			}
			continue
		}
		x, err := strconv.Atoi(f)
		if err != nil {
			return nil, fmt.Errorf("bad orbital index %q", f)
		}
		out = append(out, x)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("empty orbital set")
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}

// ParseClassRule reads a Q/P rule from a flag string.
//
// The grammar is clauses separated by ';', each prefixed "q:" or "p:", each a
// conjunction of terms separated by '&', each term
//
//	[class/]orbitals:min[:max]
//
// where orbitals is a comma-separated list of indices and a-b ranges, class is an
// excitation class as a hole count (omitted or 0 = every class), and max omitted is
// unbounded. The four atoms of the paper's Table V are then, with 0-based occupied
// indices AFTER any frozen core:
//
//	Ne+(1s^-1)  q:0:1                          at least one 1s vacancy
//	Ar+(2p^-1)  q:0-3:1                        at least one 2s or 2p vacancy
//	Mg+(2s^-1)  q:0:1;p:2/4:1;p:3/4:2          2s in Q; P is 2h1p with a 3s hole
//	                                           and 3h2p with two of them
//	Kr+(3d^-1)  q:0-4:1;q:3/5:2:2&3/6-8:1      3d in Q, plus 3h2p 4s^-2 4p^-1
func ParseClassRule(spec, label string) (*ClassRule, error) {
	var q, p []Clause
	for _, cl := range strings.Split(spec, ";") {
		cl = strings.TrimSpace(cl)
		if cl == "" {
			continue
		}
		kind, body, ok := strings.Cut(cl, ":")
		if !ok {
			return nil, fmt.Errorf("fano: clause %q has no q:/p: prefix", cl)
		}
		var clause Clause
		for _, tm := range strings.Split(body, "&") {
			tm = strings.TrimSpace(tm)
			if tm == "" {
				return nil, fmt.Errorf("fano: empty term in clause %q", cl)
			}
			if isConfigTerm(tm) {
				return nil, fmt.Errorf("fano: term %q needs particles and an atom map; "+
					"parse the rule with ParseConfigRule", tm)
			}
			t, err := parseHoleTerm(tm)
			if err != nil {
				return nil, err
			}
			clause = append(clause, t)
		}
		switch strings.ToLower(strings.TrimSpace(kind)) {
		case "q":
			q = append(q, clause)
		case "p":
			p = append(p, clause)
		default:
			return nil, fmt.Errorf("fano: clause prefix %q is neither q nor p", kind)
		}
	}
	r, err := NewClassRule(q, p, label)
	if err != nil {
		return nil, err
	}
	r.spec = spec
	return r, nil
}

// splitClass strips an optional "class/" prefix from a term.
func splitClass(tm string) (int, string, error) {
	c, rest, has := strings.Cut(tm, "/")
	if !has {
		return 0, tm, nil
	}
	n, err := strconv.Atoi(strings.TrimSpace(c))
	if err != nil {
		return 0, "", fmt.Errorf("fano: bad excitation class %q in term %q", c, tm)
	}
	if n < 0 {
		return 0, "", fmt.Errorf("fano: negative excitation class %d in term %q", n, tm)
	}
	return n, strings.TrimSpace(rest), nil
}

// parseHoleTerm reads one hole-count term, [class/]orbitals:min[:max].
func parseHoleTerm(tm string) (Term, error) {
	fields := strings.Split(tm, ":")
	if len(fields) < 2 || len(fields) > 3 {
		return Term{}, fmt.Errorf("fano: term %q wants orbitals:min[:max]", tm)
	}
	class, orbSpec, err := splitClass(fields[0])
	if err != nil {
		return Term{}, err
	}
	set, err := parseOrbSet(orbSpec)
	if err != nil {
		return Term{}, fmt.Errorf("fano: term %q: %w", tm, err)
	}
	min, err := strconv.Atoi(strings.TrimSpace(fields[1]))
	if err != nil {
		return Term{}, fmt.Errorf("fano: bad min %q in term %q", fields[1], tm)
	}
	max := -1
	if len(fields) == 3 {
		if max, err = strconv.Atoi(strings.TrimSpace(fields[2])); err != nil {
			return Term{}, fmt.Errorf("fano: bad max %q in term %q", fields[2], tm)
		}
	}
	return Term{Class: class, Set: set, Min: min, Max: max}, nil
}
