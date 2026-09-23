package fano

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/leiaSQ/ADCgo/internal/adc/mo"
)

// charge.go — the net-charge Q/P partition of multiply ionized clusters.
//
// Hole localization (scheme A) cannot express an open channel whose final ions are
// relaxed: a relaxed He+ is a hole AND a compact particle on the same atom, whose hole
// set looks exactly like a closed He2+ configuration. A rule such as
//
//	P  <=>  every real atom has net charge +1  and  exactly one particle is free,
//
// is stated instead, with net charge on X = (holes on X) - (compact particles on X). Free
// particles belong to no atom: they are the diffuse and ghost-centre complement where the
// outgoing electron lives. Orbital-to-atom labels come from a labelled sidecar
// (scripts/fcidump/orbitals.py, internal/adc/mo readLabels).

// AtomMap assigns occupied and virtual spatial orbitals to atoms.
type AtomMap struct {
	Names   []string
	Ghost   []bool
	OccAtom []int // per occupied spatial orbital: atom index
	VirAtom []int // per virtual spatial orbital: atom index, or -1 when free
}

// AtomMapFromMO builds the map from a labelled sidecar. nocc is the number of
// occupied orbitals of the configuration space, which must match the sidecar's
// occupied labels (no frozen core is supported with labelled orbitals).
func AtomMapFromMO(md *mo.Data, nocc int) (*AtomMap, error) {
	if !md.HasLabels {
		return nil, fmt.Errorf("fano: the MO sidecar carries no orbital labels (orb_kind/orb_atom); " +
			"write it with dump_fcidump's &orbitals localized scheme")
	}
	if n := md.NOccLabelled(); n != nocc {
		return nil, fmt.Errorf("fano: sidecar labels %d occupied orbitals, the space has %d", n, nocc)
	}
	am := &AtomMap{Names: md.AtomNames, Ghost: md.GhostAtom,
		OccAtom: slices.Clone(md.OrbAtom[:nocc]), VirAtom: slices.Clone(md.OrbAtom[nocc:])}
	return am, am.validate()
}

func (am *AtomMap) validate() error {
	if len(am.Ghost) != len(am.Names) {
		return fmt.Errorf("fano: atom map has %d ghost flags for %d atoms", len(am.Ghost), len(am.Names))
	}
	for i, a := range am.OccAtom {
		if a < 0 || a >= len(am.Names) || am.Ghost[a] {
			return fmt.Errorf("fano: occupied orbital %d assigned to atom %d, not a real atom", i, a)
		}
	}
	for i, a := range am.VirAtom {
		if a >= len(am.Names) || (a >= 0 && am.Ghost[a]) {
			return fmt.Errorf("fano: virtual orbital %d assigned to atom %d, not a real atom", i, a)
		}
	}
	return nil
}

// atomIndex resolves a real atom name; ghosts and unknown names are errors, because a
// ghost carries no nucleus and cannot be ionized.
func (am *AtomMap) atomIndex(name string) (int, error) {
	i := slices.Index(am.Names, name)
	if i < 0 {
		return 0, fmt.Errorf("fano: unknown atom %q (atoms: %v)", name, am.Names)
	}
	if am.Ghost[i] {
		return 0, fmt.Errorf("fano: %q is a ghost centre; ghosts carry no nucleus and "+
			"cannot appear in charge rules", name)
	}
	return i, nil
}

func (am *AtomMap) realAtoms() []int {
	var out []int
	for i, g := range am.Ghost {
		if !g {
			out = append(out, i)
		}
	}
	return out
}

// cterm is one predicate of a ConfigRule clause.
type cterm interface {
	match(holes, parts []int, am *AtomMap) bool
	String() string
}

type holeTerm struct{ t Term }

func (h holeTerm) match(holes, _ []int, _ *AtomMap) bool { return h.t.match(holes) }
func (h holeTerm) String() string                        { return h.t.String() }

// chargeTerm requires the listed atoms to carry exactly the given net charge.
type chargeTerm struct {
	class  int
	atoms  []int
	charge []int
	names  []string
}

func netCharges(holes, parts []int, am *AtomMap) []int {
	q := make([]int, len(am.Names))
	for _, h := range holes {
		q[am.OccAtom[h]]++
	}
	for _, v := range parts {
		if a := am.VirAtom[v]; a >= 0 {
			q[a]--
		}
	}
	return q
}

func (c chargeTerm) match(holes, parts []int, am *AtomMap) bool {
	if c.class != 0 && len(holes) != c.class {
		return false
	}
	q := netCharges(holes, parts, am)
	for i, a := range c.atoms {
		if q[a] != c.charge[i] {
			return false
		}
	}
	return true
}

func (c chargeTerm) String() string {
	var b strings.Builder
	if c.class != 0 {
		fmt.Fprintf(&b, "%dh: ", c.class)
	}
	b.WriteString("charge ")
	for i, n := range c.names {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "%s=%+d", n, c.charge[i])
	}
	return b.String()
}

// freeTerm counts particles in free (atom-less) virtuals.
type freeTerm struct {
	class    int
	min, max int
}

func (f freeTerm) match(holes, parts []int, am *AtomMap) bool {
	if f.class != 0 && len(holes) != f.class {
		return false
	}
	n := 0
	for _, v := range parts {
		if am.VirAtom[v] < 0 {
			n++
		}
	}
	return n >= f.min && (f.max < 0 || n <= f.max)
}

func (f freeTerm) String() string {
	pre := ""
	if f.class != 0 {
		pre = fmt.Sprintf("%dh: ", f.class)
	}
	switch {
	case f.max < 0:
		return fmt.Sprintf("%sfree particles >= %d", pre, f.min)
	case f.min == f.max:
		return fmt.Sprintf("%sfree particles == %d", pre, f.min)
	default:
		return fmt.Sprintf("%sfree particles in [%d,%d]", pre, f.min, f.max)
	}
}

// ConfigRule is the net-charge generalization of ClassRule: the same Q/P clause logic
// (Q if a Q clause matches; else, when P clauses exist, Q unless a P clause matches),
// with terms that may look at particles.
type ConfigRule struct {
	q, p  [][]cterm
	am    *AtomMap
	label string
	spec  string
}

var _ ConfigSelector = (*ConfigRule)(nil)

// Bound cannot be answered from holes alone; NewPartition calls BoundConfig instead.
func (r *ConfigRule) Bound(holes []int) bool {
	panic("fano: ConfigRule needs particles; use NewPartition with a ParticleSpace")
}

func matchAll(c []cterm, holes, parts []int, am *AtomMap) bool {
	if len(c) == 0 {
		return false
	}
	for _, t := range c {
		if !t.match(holes, parts, am) {
			return false
		}
	}
	return true
}

// BoundConfig reports whether a configuration with these holes (occupied spatial
// orbitals) and particles (virtual spatial orbitals) belongs to Q.
func (r *ConfigRule) BoundConfig(holes, parts []int) bool {
	if len(holes) == 0 {
		return false
	}
	for _, c := range r.q {
		if matchAll(c, holes, parts, r.am) {
			return true
		}
	}
	if len(r.p) == 0 {
		return false
	}
	for _, c := range r.p {
		if matchAll(c, holes, parts, r.am) {
			return false
		}
	}
	return true
}

func (r *ConfigRule) String() string {
	join := func(cs [][]cterm) string {
		parts := make([]string, len(cs))
		for i, c := range cs {
			ts := make([]string, len(c))
			for j, t := range c {
				ts[j] = t.String()
			}
			parts[i] = "(" + strings.Join(ts, " and ") + ")"
		}
		return strings.Join(parts, " or ")
	}
	var b strings.Builder
	b.WriteString("net-charge rule: Q if ")
	if len(r.q) == 0 {
		b.WriteString("never")
	} else {
		b.WriteString(join(r.q))
	}
	if len(r.p) > 0 {
		b.WriteString(", else Q unless P: ")
		b.WriteString(join(r.p))
	}
	if r.spec != "" {
		fmt.Fprintf(&b, " [spec %s]", r.spec)
	}
	if r.label != "" {
		fmt.Fprintf(&b, " [%s]", r.label)
	}
	return b.String()
}

// isConfigTerm reports whether a term needs particles (charge or free terms).
func isConfigTerm(tm string) bool {
	_, rest, err := splitClass(tm)
	if err != nil {
		rest = tm
	}
	rest = strings.TrimSpace(rest)
	return strings.HasPrefix(rest, "charge") || strings.HasPrefix(rest, "free")
}

// ParseConfigRule reads a net-charge Q/P rule. The grammar is ParseClassRule's
// (clauses separated by ';', prefixed q: or p:, terms joined by '&', hole terms
// [class/]orbitals:min[:max]) plus two particle-aware terms:
//
//	[class/]charge NAME=q,NAME=q,...   the named real atoms carry exactly net charge q
//	[class/]charge *=q                 every real atom carries net charge q
//	[class/]free=min[:max]             number of particles in free virtuals
//
// For example, the open channel "every one of five atoms singly charged, one free
// electron" is
//
//	p: charge A=1,B=1,C=1,D=1,E=1 & free=1
//
// Ghost and unknown atom names are errors.
func ParseConfigRule(spec, label string, am *AtomMap) (*ConfigRule, error) {
	if am == nil {
		return nil, fmt.Errorf("fano: ParseConfigRule needs an atom map")
	}
	if err := am.validate(); err != nil {
		return nil, err
	}
	r := &ConfigRule{am: am, label: label, spec: spec}
	for _, cl := range strings.Split(spec, ";") {
		cl = strings.TrimSpace(cl)
		if cl == "" {
			continue
		}
		kind, body, ok := strings.Cut(cl, ":")
		if !ok {
			return nil, fmt.Errorf("fano: clause %q has no q:/p: prefix", cl)
		}
		var clause []cterm
		for _, tm := range strings.Split(body, "&") {
			tm = strings.TrimSpace(tm)
			if tm == "" {
				return nil, fmt.Errorf("fano: empty term in clause %q", cl)
			}
			t, err := parseConfigTerm(tm, am)
			if err != nil {
				return nil, err
			}
			clause = append(clause, t)
		}
		switch strings.ToLower(strings.TrimSpace(kind)) {
		case "q":
			r.q = append(r.q, clause)
		case "p":
			r.p = append(r.p, clause)
		default:
			return nil, fmt.Errorf("fano: clause prefix %q is neither q nor p", kind)
		}
	}
	if len(r.q) == 0 && len(r.p) == 0 {
		return nil, fmt.Errorf("fano: empty Q/P rule")
	}
	return r, nil
}

func parseConfigTerm(tm string, am *AtomMap) (cterm, error) {
	if !isConfigTerm(tm) {
		t, err := parseHoleTerm(tm)
		if err != nil {
			return nil, err
		}
		if len(t.Set) == 0 || t.Min < 0 || (t.Max >= 0 && t.Max < t.Min) {
			return nil, fmt.Errorf("fano: invalid hole term %q", tm)
		}
		return holeTerm{t}, nil
	}
	class, rest, err := splitClass(tm)
	if err != nil {
		return nil, err
	}
	if after, ok := strings.CutPrefix(rest, "free"); ok {
		after = strings.TrimSpace(after)
		v, ok := strings.CutPrefix(after, "=")
		if !ok {
			return nil, fmt.Errorf("fano: term %q wants free=min[:max]", tm)
		}
		lo, hi, hasHi := strings.Cut(strings.TrimSpace(v), ":")
		min, err := strconv.Atoi(strings.TrimSpace(lo))
		if err != nil || min < 0 {
			return nil, fmt.Errorf("fano: bad free count %q in term %q", lo, tm)
		}
		max := min
		if hasHi {
			if max, err = strconv.Atoi(strings.TrimSpace(hi)); err != nil || max < min {
				return nil, fmt.Errorf("fano: bad free maximum %q in term %q", hi, tm)
			}
		}
		return freeTerm{class: class, min: min, max: max}, nil
	}
	body := strings.TrimSpace(strings.TrimPrefix(rest, "charge"))
	if body == "" {
		return nil, fmt.Errorf("fano: term %q names no atoms", tm)
	}
	ct := chargeTerm{class: class}
	for _, kv := range strings.Split(body, ",") {
		name, val, ok := strings.Cut(strings.TrimSpace(kv), "=")
		if !ok {
			return nil, fmt.Errorf("fano: charge entry %q wants NAME=q", kv)
		}
		name = strings.TrimSpace(name)
		q, err := strconv.Atoi(strings.TrimSpace(val))
		if err != nil {
			return nil, fmt.Errorf("fano: bad charge %q in term %q", val, tm)
		}
		if name == "*" {
			for _, a := range am.realAtoms() {
				ct.atoms = append(ct.atoms, a)
				ct.charge = append(ct.charge, q)
				ct.names = append(ct.names, am.Names[a])
			}
			continue
		}
		a, err := am.atomIndex(name)
		if err != nil {
			return nil, err
		}
		if slices.Contains(ct.atoms, a) {
			return nil, fmt.Errorf("fano: atom %q listed twice in term %q", name, tm)
		}
		ct.atoms = append(ct.atoms, a)
		ct.charge = append(ct.charge, q)
		ct.names = append(ct.names, name)
	}
	return ct, nil
}
