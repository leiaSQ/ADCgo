package fano

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
)

// channels.go — channel-resolved (partial) widths.
//
// APPROXIMATE, and the paper says so: it defines channel projectors from the L2
// intermediate states and calls the resulting partial widths an estimate, deferring the
// rigorous treatment to a follow-up (ADC22.pdf §III D). Any output carrying these
// numbers has to carry that caveat with them.
//
// What is computed here is the additive decomposition: each state's gamma_i is
// distributed over channels in proportion to that state's character,
//
//	gamma_{beta,i} = gamma_i * w_beta(chi_i) / sum_beta' w_beta'(chi_i),
//
// so sum_beta Gamma_beta = Gamma by construction. The alternative reading of the paper —
// gamma_{beta,i} = 2 pi <g|P_beta|chi_i>^2 — is NOT additive, since squaring does not
// distribute over the projector sum, and would not reproduce a branching ratio.
//
// The character w_beta comes from where the holes sit, which is exactly what
// spectrum.Classify already routes, and that routing is the one ADCanalysis validated:
//
//	both holes on the initial site A      -> Auger@A
//	one hole on A, one on B               -> ICD:A->B
//	both holes on one site B != A          -> ETMD(2)
//	holes on two sites B, C != A           -> ETMD(3)
//
// The second-order class (3h2p for single ionization) is a separate channel, DoubleName:
// its final states carry two free electrons, which the two-hole routing above has no
// label for. That is not a gap but the quantity the paper's Table VII reports — Kr 3d
// Auger decay split into AD (normal, one electron, the 2h1p channels above) and CAD
// (cascade/double, two electrons, this channel), for which it quotes a 2.1% branching
// ratio at ADC(2,2)f against 2.4% experimental.

// DoubleName labels the second-order decay channel — double Auger, double ICD — whose
// final states are the 3h2p configurations. The paper's "CAD".
const DoubleName = "double"

// Channels routes a pseudo-continuum state's hole character onto named decay channels.
// It is optional: partial widths need an MO sidecar to know where an orbital sits, and a
// run without one still gets a total width.
type Channels struct {
	pop     [][]float64 // per occupied orbital, gross population over cols (sums to 1)
	cols    []string    // population column names (atom labels)
	sites   []spectrum.Site
	initial string
	opts    spectrum.Options
}

// NewChannels builds the router from an MO sidecar.
//
// Each occupied orbital is assigned to atoms by its Mulliken gross population,
//
//	q_A(i) = sum_{p in A} sum_q C_pi C_qi S_pq,
//
// which sums to 1 over atoms for a normalized orbital. A hole in orbital i is then
// spread over sites with those weights rather than assigned to one site, which matters
// for exactly the orbitals that decide a branching ratio: an inner-valence MO delocalized
// over two subunits contributes to both the local and the interatomic channel, and
// forcing it onto its largest atom would hand its whole weight to one of them.
//
// sites and initial are the user's site definitions and the initially ionized site, the
// same ones the spectrum layer takes; opts is spectrum's channel-keeping policy.
//
// Mulliken populations can be slightly negative on an atom an orbital barely touches
// (water's O 1s comes out at 1.00013 on O and -6.6e-5 on each H), so a channel weight can
// carry a correspondingly tiny negative contribution. That is a property of the
// population analysis, not an error: the populations still sum to 1 per orbital, so the
// decomposition stays additive, which is what TestPartialWidthsAreAdditive checks.
func NewChannels(md *mo.Data, nocc int, sites []spectrum.Site, initial string, opts spectrum.Options) (*Channels, error) {
	if md == nil {
		return nil, fmt.Errorf("fano: partial widths need an MO sidecar to locate the orbitals")
	}
	if nocc <= 0 || nocc > md.NMO {
		return nil, fmt.Errorf("fano: %d occupied orbitals is outside the sidecar's %d MOs", nocc, md.NMO)
	}
	if err := spectrum.ValidateInitialAtom(initial, sites); err != nil {
		// A bare atom label with no Site declared for it is still usable: Regroup keeps an
		// ungrouped column as its own site, so the label names a site after all.
		if !hasColumn(md.AtomNames, initial) {
			return nil, fmt.Errorf("fano: %w", err)
		}
	}

	natom := len(md.AtomNames)
	pop := make([][]float64, nocc)
	for i := range nocc {
		q := make([]float64, natom)
		for p := range md.NAO {
			a := md.AOAtom[p]
			if a < 0 || a >= natom {
				continue
			}
			cpi := md.C.At(p, i)
			if cpi == 0 {
				continue
			}
			var s float64
			for qq := range md.NAO {
				s += md.C.At(qq, i) * md.S.At(p, qq)
			}
			q[a] += cpi * s
		}
		pop[i] = q
	}
	return &Channels{pop: pop, cols: md.AtomNames, sites: sites, initial: initial, opts: opts}, nil
}

func hasColumn(cols []string, name string) bool {
	for _, c := range cols {
		if c == name {
			return true
		}
	}
	return false
}

// Partial is the channel-resolved pseudo-continuum: one discrete spectrum per channel over
// the same energy grid, which ImagePartial then images channel by channel.
//
// Sum and Total are zeroth MOMENTS (energy squared), not widths — see Pseudo.SumGamma.
// They give a serviceable branching ratio because a ratio of moments is dimensionless, but
// the partial WIDTHS are what ImagePartial returns.
type Partial struct {
	Names  []string    // channel labels, canonical order, DoubleName last
	Energy []float64   // eps_i, shared with the Pseudo it came from
	Gamma  [][]float64 // Gamma[b][i]: channel b's share of gamma_i
	Sum    []float64   // Sum[b] = total over i: channel b's zeroth MOMENT, not its width

	// Unassigned is the width carried by character that is no decay channel at all —
	// mainly the main-class (1h) component of a pseudo-continuum state, plus anything
	// spectrum.Options filtered out. Reported rather than silently folded in, so the
	// channels and the total can be checked against each other.
	Unassigned float64
	Total      float64 // == Pseudo.SumGamma
}

// Ratios returns each channel's share of the total zeroth moment, Sum[b]/Total. It is an
// approximation to the branching ratio; the imaged one comes from ImagePartial.
func (p *Partial) Ratios() []float64 {
	out := make([]float64, len(p.Sum))
	if p.Total == 0 {
		return out
	}
	for b, s := range p.Sum {
		out[b] = s / p.Total
	}
	return out
}

// PartialWidths distributes each gamma_i over the decay channels.
//
// psp is the restricted P space, res its solved spectrum, and ps the Pseudo from Widths
// over the same res. Only the retained states contribute, so the channel sums add up to
// ps.SumGamma less Unassigned.
func (ch *Channels) PartialWidths(psp Space, res lanczos.Result, ps *Pseudo) (*Partial, error) {
	if !res.HasFull() {
		return nil, fmt.Errorf("fano: partial widths need the full Ritz vectors")
	}
	n := psp.Size()
	if res.FullVecs.Rows != n {
		return nil, fmt.Errorf("fano: the Ritz vectors have %d rows but the P space has %d",
			res.FullVecs.Rows, n)
	}

	// Split P's satellite rows by class: the decay class feeds the two-hole routing, every
	// deeper class feeds the second-order channel.
	main := psp.MainBlockSize()
	buf := make([]int, 0, 4)
	holes := make([][]int, n)
	for r := main; r < n; r++ {
		holes[r] = append([]int(nil), psp.Holes(r, buf[:0])...)
	}

	out := &Partial{Energy: ps.Energy, Total: ps.SumGamma}
	col := map[string]int{} // channel name -> index in out.Names, in first-seen order
	idx := func(name string) int {
		if b, ok := col[name]; ok {
			return b
		}
		b := len(out.Names)
		col[name] = b
		out.Names = append(out.Names, name)
		out.Gamma = append(out.Gamma, make([]float64, len(ps.Energy)))
		out.Sum = append(out.Sum, 0)
		return b
	}

	for i, j := range ps.Root {
		row := spectrum.Row{
			OneSite: map[string]float64{},
			TwoSite: map[string]float64{},
		}
		var norm2, norm3 float64
		for r := main; r < n; r++ {
			c := res.FullVecs.At(r, j)
			if c == 0 {
				continue
			}
			w := c * c
			if len(holes[r]) != ps.DecayHoles {
				norm3 += w
				continue
			}
			norm2 += w
			ch.spread(row, holes[r], w)
		}
		denom := norm2 + norm3
		if denom == 0 {
			out.Unassigned += ps.Gamma[i]
			continue
		}
		scale := ps.Gamma[i] / denom

		var assigned float64
		for _, c := range spectrum.Classify(ch.initial, ch.sites, spectrum.Regroup(row, ch.sites), ch.opts) {
			g := c.Weight * scale
			b := idx(c.Name)
			out.Gamma[b][i] += g
			out.Sum[b] += g
			assigned += g
		}
		if norm3 > 0 {
			g := norm3 * scale
			b := idx(DoubleName)
			out.Gamma[b][i] += g
			out.Sum[b] += g
			assigned += g
		}
		out.Unassigned += ps.Gamma[i] - assigned
	}

	// DoubleName last, so the AD/CAD split reads in that order.
	if b, ok := col[DoubleName]; ok && b != len(out.Names)-1 {
		out.moveLast(b)
	}
	return out, nil
}

// moveLast rotates channel b to the end, keeping the others' relative order.
func (p *Partial) moveLast(b int) {
	name, gam, sum := p.Names[b], p.Gamma[b], p.Sum[b]
	p.Names = append(append(p.Names[:b:b], p.Names[b+1:]...), name)
	p.Gamma = append(append(p.Gamma[:b:b], p.Gamma[b+1:]...), gam)
	p.Sum = append(append(p.Sum[:b:b], p.Sum[b+1:]...), sum)
}

// spread adds weight w to row, distributing a configuration's two holes over population
// columns by their Mulliken weights. A coincident hole pair (both electrons out of one
// orbital) is the same expression with k == l.
func (ch *Channels) spread(row spectrum.Row, holes []int, w float64) {
	if len(holes) != 2 {
		return
	}
	pk, pl := ch.popOf(holes[0]), ch.popOf(holes[1])
	if pk == nil || pl == nil {
		return
	}
	for a, qa := range pk {
		if qa == 0 {
			continue
		}
		for b, qb := range pl {
			if qb == 0 {
				continue
			}
			v := w * qa * qb
			switch {
			case a == b:
				row.OneSite[ch.cols[a]] += v
			case a < b:
				row.TwoSite[ch.cols[a]+"/"+ch.cols[b]] += v
			default:
				row.TwoSite[ch.cols[b]+"/"+ch.cols[a]] += v
			}
		}
	}
}

func (ch *Channels) popOf(orb int) []float64 {
	if orb < 0 || orb >= len(ch.pop) {
		return nil
	}
	return ch.pop[orb]
}

// String summarizes the channel decomposition for the run log, as shares of the total
// zeroth moment.
func (p *Partial) String() string {
	s := fmt.Sprintf("total strength %.4g Eh^2", p.Total)
	for b, name := range p.Names {
		s += fmt.Sprintf("; %s %.2f%%", name, 100*p.Sum[b]/nonZero(p.Total))
	}
	if p.Unassigned != 0 {
		s += fmt.Sprintf("; unassigned %.2f%%", 100*p.Unassigned/nonZero(p.Total))
	}
	return s
}

func nonZero(x float64) float64 {
	if x == 0 {
		return 1
	}
	return x
}
