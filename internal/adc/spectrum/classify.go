// classify maps a state's two-hole population row to decay channels relative to
// a user-specified initial core-ionized site.
//
// The physics: a core hole on site A decays into a dicationic state with two
// valence holes. Where those two holes sit relative to A *is* the decay channel:
//
//	both holes on A             -> local Auger ("Auger@A"),  from one-site A
//	one hole on A, one on B     -> ICD         ("ICD:A->B"), from two-site A/B
//	both holes on one site B!=A  -> ETMD(2)    ("ETMD(2)"),  from one-site B
//	holes on two sites B,C != A  -> ETMD(3)    ("ETMD(3)"),  from two-site B/C
//
// The population row already decomposes each state into one-site and two-site
// weights, so classification is a routing of the row's named columns onto
// channel labels — no eigenvector work. The columns are first folded into
// user-defined Sites (see Regroup); the ETMD(2)/ETMD(3) split is meaningful only
// once those sites are defined. This is a verbatim port of ADCanalysis's
// internal/classify (which the same test suite guards here).
package spectrum

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// pairSep separates the two group names in a two-site key ("O/H1").
const pairSep = "/"

// Regroup folds an atom-keyed population row into a site-keyed row using the
// given Sites:
//
//   - one-site weight moves to its column's Site;
//   - two-site weight between two columns in the *same* Site becomes one-site
//     weight on that Site (both holes on one unit — an ETMD(2) candidate);
//   - two-site weight between columns in *different* Sites stays two-site, keyed
//     by the two Site names ordered by Site-declaration order.
//
// A column not listed in any Site is kept as its own single-column site, so no
// population is lost and the identity grouping reproduces the atom-resolved row.
func Regroup(row Row, sites []Site) Row {
	siteOf := make(map[string]string)
	order := make(map[string]int)
	for i, s := range sites {
		order[s.Name] = i
		if len(s.Members) == 0 {
			siteOf[s.Name] = s.Name
		}
		for _, m := range s.Members {
			siteOf[m] = s.Name
		}
	}
	nameOf := func(col string) string {
		if s, ok := siteOf[col]; ok {
			return s
		}
		return col // ungrouped column is its own site
	}
	rank := func(site string) int {
		if i, ok := order[site]; ok {
			return i
		}
		return math.MaxInt // sites not in the declared order sort last, deterministically
	}

	out := Row{
		EnergyEV: row.EnergyEV,
		OneSite:  make(map[string]float64),
		TwoSite:  make(map[string]float64),
	}
	for col, w := range row.OneSite {
		out.OneSite[nameOf(col)] += w
	}
	for key, w := range row.TwoSite {
		a, b, ok := splitPair(key)
		if !ok {
			continue
		}
		sa, sb := nameOf(a), nameOf(b)
		if sa == sb {
			out.OneSite[sa] += w // intra-site pair: both holes on one site
			continue
		}
		if ra, rb := rank(sa), rank(sb); ra > rb || (ra == rb && sa > sb) {
			sa, sb = sb, sa
		}
		out.TwoSite[sa+pairSep+sb] += w
	}
	return out
}

// PassiveSet collects the population columns marked Passive across all sites.
// The result is the column set whose holes Discount discounts; it is empty when
// no site declares any passive members, in which case Discount is a no-op.
func PassiveSet(sites []Site) map[string]bool {
	var passive map[string]bool
	for _, s := range sites {
		for _, m := range s.Passive {
			if passive == nil {
				passive = make(map[string]bool)
			}
			passive[m] = true
		}
	}
	return passive
}

// Discount scales each two-hole configuration in row by the fraction of its two
// holes that sit on *active* (non-passive) columns, discarding the rest, and
// returns the resulting atom-keyed row. It is meant to run before Regroup.
//
//   - a one-site weight survives in full if that column is active, and drops to
//     zero if it is passive;
//   - a two-site weight is kept in full when both columns are active, halved when
//     exactly one is passive (the passive hole discarded), and dropped when both
//     are passive.
//
// With no passive columns the row is returned unchanged.
func Discount(row Row, passive map[string]bool) Row {
	if len(passive) == 0 {
		return row
	}
	out := Row{
		EnergyEV: row.EnergyEV,
		OneSite:  make(map[string]float64),
		TwoSite:  make(map[string]float64),
	}
	for col, w := range row.OneSite {
		if !passive[col] {
			out.OneSite[col] = w
		}
	}
	for key, w := range row.TwoSite {
		a, b, ok := splitPair(key)
		if !ok {
			continue
		}
		kept := 0
		if !passive[a] {
			kept++
		}
		if !passive[b] {
			kept++
		}
		if kept == 0 {
			continue
		}
		out.TwoSite[key] = w * float64(kept) / 2
	}
	return out
}

// Options tunes which channels are emitted. The zero value keeps every channel
// with a strictly positive weight, discarding the rounding-noise entries the
// table prints (e.g. "-0.0000").
type Options struct {
	// MinWeight drops any channel whose weight is <= MinWeight. The default 0
	// removes zero and rounding-negative contributions while keeping all genuine
	// population. Raise it to suppress small but nonzero channels.
	MinWeight float64

	// MinFraction drops any channel whose weight, as a fraction of the row's
	// total two-hole population, is below MinFraction (0..1). Default 0 disables
	// it. Ignored when the row total is non-positive.
	MinFraction float64

	// IncludeZero emits the full canonical channel set for the given groups even
	// when a weight is zero or below the thresholds, giving a stable,
	// identically-keyed channel list across every state. Default false.
	IncludeZero bool
}

// DefaultOptions returns the recommended thresholds: keep every channel with a
// positive weight, no fractional cut, no zero padding.
func DefaultOptions() Options {
	return Options{MinWeight: 0, MinFraction: 0, IncludeZero: false}
}

// ValidateInitialAtom reports whether initialAtom names one of the sites.
// Callers run this once (a user-config check), then call Classify per state.
func ValidateInitialAtom(initialAtom string, sites []Site) error {
	for _, s := range sites {
		if s.Name == initialAtom {
			return nil
		}
	}
	names := make([]string, len(sites))
	for i, s := range sites {
		names[i] = s.Name
	}
	return fmt.Errorf("initial site %q is not one of the defined sites (have: %s)",
		initialAtom, strings.Join(names, ", "))
}

// Classify routes a single state's two-hole population row onto decay channels,
// relative to the initial core-ionized site A == initialAtom. The row should
// already be site-keyed (see Regroup). Weights are read from the row, never
// invented. The returned slice is in a stable canonical order — Auger@A, then
// ICD:A->B for each other site in `sites` order, then any ICD partners not in
// `sites`, then ETMD(2), then ETMD(3) — filtered per opts.
func Classify(initialAtom string, sites []Site, row Row, opts Options) []Channel {
	total := row.Sum()

	// Auger@A: both holes on the initial atom.
	auger := Channel{
		Name:   "Auger@" + initialAtom,
		Weight: row.OneSite[initialAtom],
	}

	// ICD:A->B: two-site weight for every pair that includes A, keyed by the
	// partner atom B. Tolerate either key ordering.
	icd := make(map[string]float64)
	for key, w := range row.TwoSite {
		a, b, ok := splitPair(key)
		if !ok {
			continue
		}
		switch initialAtom {
		case a:
			icd[b] += w
		case b:
			icd[a] += w
		}
	}

	// ETMD: no hole on A. ETMD(2) lands both holes on a single neighbour site
	// (one-site weight on a site != A); ETMD(3) puts the holes on two different
	// neighbour sites (two-site weight whose pair excludes A). ETMD(2) needs >=1
	// site other than A, ETMD(3) needs >=2.
	others := 0
	for _, s := range sites {
		if s.Name != initialAtom {
			others++
		}
	}
	var etmd2, etmd3 float64
	for name, w := range row.OneSite {
		if name != initialAtom {
			etmd2 += w
		}
	}
	for key, w := range row.TwoSite {
		a, b, ok := splitPair(key)
		if !ok {
			continue
		}
		if a != initialAtom && b != initialAtom {
			etmd3 += w
		}
	}

	out := make([]Channel, 0, len(icd)+3)
	add := func(c Channel) {
		if keep(c.Weight, total, opts) {
			out = append(out, c)
		}
	}

	add(auger)

	seen := make(map[string]bool, len(icd))
	for _, s := range sites {
		if s.Name == initialAtom {
			continue
		}
		seen[s.Name] = true
		add(Channel{Name: icdName(initialAtom, s.Name), Weight: icd[s.Name]})
	}
	var leftover []string
	for partner := range icd {
		if !seen[partner] {
			leftover = append(leftover, partner)
		}
	}
	sort.Strings(leftover)
	for _, partner := range leftover {
		add(Channel{Name: icdName(initialAtom, partner), Weight: icd[partner]})
	}

	if others >= 1 {
		add(Channel{Name: "ETMD(2)", Weight: etmd2})
	}
	if others >= 2 {
		add(Channel{Name: "ETMD(3)", Weight: etmd3})
	}

	return out
}

// keep applies the Options thresholds to a single channel weight.
func keep(weight, total float64, opts Options) bool {
	if opts.IncludeZero {
		return true
	}
	if weight <= opts.MinWeight {
		return false
	}
	if opts.MinFraction > 0 && total > 0 && weight/total < opts.MinFraction {
		return false
	}
	return true
}

// icdName builds the "ICD:A->B" label.
func icdName(a, b string) string { return "ICD:" + a + "->" + b }

// splitPair splits a two-site key ("O/H1") into its two group names. It reports
// ok=false for a malformed key (no separator or an empty side).
func splitPair(key string) (a, b string, ok bool) {
	a, b, ok = strings.Cut(key, pairSep)
	if !ok || a == "" || b == "" {
		return "", "", false
	}
	return a, b, true
}
