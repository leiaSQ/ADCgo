package sip

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
)

// The correlation corrections to the ISR representation of a one-particle operator — the
// terms isrdipole.go's port left out. Ported from theADCcode's ndadc3_prop
// (my_calc_d11.c, my_calc_d12.c), which reaches them through ND_ADC3_CAP_matrix.
//
// The ISR property matrix carries the same order structure as the secular matrix it belongs
// to: the 1h/1h block is needed to second order, the 1h/2h1p block to first, and the
// 2h1p/2h1p block only to zeroth. So the complete set of missing terms is
//
//	1h/1h    (13a) the correlated operator moment D₀ = 2·Tr(d·ρ) on the diagonal
//	         (13c) a ground-state-density term, ρ from internal/adc/selfenergy
//	         (12a) (12b) (12c) the second-order self-energy terms
//	1h/2h1p  (8a) (8b) (8c) the first-order terms
//	2h1p/2h1p  — nothing. my_calc_d22_{diag,off}.c contain no energy denominators and no
//	         two-electron integrals at all; isrdipole.go's satsat is already complete.
//
// This is exactly the ADC(2)/ADC(3)-consistent property matrix: one implementation serves
// both orders. It is *not* order-consistent for ADC(4), which would additionally need the
// O(3)/O(4) terms built from t₂⁽²⁾/t₁⁽²⁾/t₃⁽²⁾ — none of which exist in the tree. Hence the
// CLI keeps it opt-in there.
//
// # No irrep gating
//
// The legacy loops restrict their summation ranges by irrep (b_sym == g_sym in (12a), p and
// q of one irrep in (8a)/(8b), and so on). Those restrictions are *derived* from the
// same-space case the legacy only ever handles, where bra and ket carry one target symmetry.
// The rectangular D of isrdipole_cross.go breaks that assumption: a cross-sector element has
// i in occ(bra.Sym) and j in occ(ket.Sym), and every one of those irrep windows shifts by
// bra.Sym ⊗ ket.Sym.
//
// Rather than re-derive each window and risk getting one of them subtly wrong, the loops
// below run over the *full* occupied and virtual ranges and let the integrals do the gating:
// every term is a product of two-electron integrals whose own symmetry already forces the
// summand to vanish outside the legacy's window. In the square case the two agree term for
// term; in the cross case only this form is right. The cost is nil — the vanishing summands
// are the ones the tensor contractions would compute anyway.
//
// # Sign
//
// The 1h/2h1p terms carry the overall sign flip that mainSat already applies to the legacy
// (7a)/(7b): theADCcode's cap_calc_d12 is negated relative to its own secular coupling block
// (nd_adc3_cap_matrix.cpp:335, "a sign error in Joerg's code"), and (7) and (8) sit in the
// same legacy expression, so their relative sign is fixed. Every (8) term below is therefore
// the negative of the code in my_calc_d12.c. TestCorrZerothOrderSignsMatchMainSat pins that
// against the zeroth-order block rather than assuming it.

// ISROptions carries the ingredients the correlation corrections need. isrdipole.go's
// operator is built from a Space and a dipole matrix alone; these terms additionally read
// the two-electron integrals, the orbital energies, and the ground-state density.
//
// A nil *ISROptions selects the plain zeroth-order operator, bit-identical to what
// NewISRDipole/NewISRDipoleCross produced before this file existed.
type ISROptions struct {
	Ints *integrals.Store
	Eps  []float64

	// Rho is the correlation part of the ground-state one-particle density matrix,
	// selfenergy.Density (the zeroth-order δ_ij n_i is *not* included — this is the
	// correction alone, and the terms below add the reference part where they need it).
	// Indices are absolute orbitals. nil disables (13a)/(13c) alone, leaving (12)/(8) on.
	Rho func(p, q int) float64
}

func (o *ISROptions) validate(sp *Space) error {
	if o.Ints == nil {
		return fmt.Errorf("sip: ISR corrections need the two-electron integrals")
	}
	if len(o.Eps) < sp.Norb {
		return fmt.Errorf("sip: ISR corrections need %d orbital energies, got %d", sp.Norb, len(o.Eps))
	}
	return nil
}

// isrCorr precomputes everything about the correction terms that does not depend on which
// Cartesian component of the operator is being represented, so the three components share
// the cost instead of tripling it.
type isrCorr struct {
	ints *integrals.Store
	eps  []float64
	rho  func(p, q int) float64

	nocc, nvir, norb int

	// p11 folds the *whole* 1h/1h correction — (13c) + (12a) + (12b) + (12c) — into one
	// norb×norb "transition density" per pair of 1h holes, so that
	//
	//	⟨i|D̂|j⟩_corr = Σ_pq p11[i][j][p,q] · d_pq
	//
	// which is how the legacy is organized too (its sums multiply CAP_PTR element by
	// element). Every term of the correction contracts with exactly one dipole element, so
	// they all land in this one matrix. Keyed by absolute occupied orbital; a pair that is
	// not a (bra hole, ket hole) combination is nil.
	p11 [][][]float64
}

// v is the reference V1212 integral in physicist ordering ⟨ab|cd⟩, matching elements.go.
func (c *isrCorr) v(a, b, cc, d int) float64 { return c.ints.Eri(a, cc, b, d) }

// newISRCorr builds the shared correction context for one bra/ket space pair.
func newISRCorr(bra, ket *Space, o *ISROptions) (*isrCorr, error) {
	if err := o.validate(bra); err != nil {
		return nil, err
	}
	c := &isrCorr{
		ints: o.Ints, eps: o.Eps, rho: o.Rho,
		nocc: bra.Nocc, nvir: bra.Nvir, norb: bra.Norb,
	}
	c.buildP11(bra, ket)
	return c, nil
}

// holes lists the 1h main configurations' orbitals of a space.
func holes(sp *Space) []int {
	out := make([]int, 0, sp.BeginSat)
	for i := range sp.BeginSat {
		out = append(out, sp.Configs[i].Occ[0])
	}
	return out
}

// buildP11 fills p11 for every (bra hole, ket hole) pair.
func (c *isrCorr) buildP11(bra, ket *Space) {
	c.p11 = make([][][]float64, c.nocc)
	for i := range c.nocc {
		c.p11[i] = make([][]float64, c.nocc)
	}
	for _, i := range holes(bra) {
		for _, j := range holes(ket) {
			if c.p11[i][j] != nil {
				continue
			}
			c.p11[i][j] = c.densityFor(i, j)
		}
	}
}

// densityFor builds the norb×norb contraction density of the 1h/1h correction for the hole
// pair (i, j). The expression is symmetric under i↔j once contracted with a symmetric d, so
// the operator stays symmetric — TestCorrDipoleSymmetric checks that rather than assuming it.
func (c *isrCorr) densityFor(i, j int) []float64 {
	n := c.norb
	p := make([]float64, n*n)

	c.term13c(p, i, j)
	c.term12a(p, i, j)
	c.term12b(p, i, j)
	c.term12c(p, i, j)
	return p
}

// term13c is the ground-state-density term (my_calc_d11.c:90-94):
//
//	w_ij -= Σ_c [ ρ(c,i)·d(j,c) + ρ(c,j)·d(i,c) ],  c over the virtuals
//
// ρ is irrep-diagonal, so the legacy's restriction of c to the target irrep is what its own
// ρ enforces anyway; here c runs over every virtual and ρ zeroes the rest. Only the
// hole/particle block of ρ is ever read.
func (c *isrCorr) term13c(p []float64, i, j int) {
	if c.rho == nil {
		return
	}
	n := c.norb
	for a := c.nocc; a < c.norb; a++ {
		p[j*n+a] -= c.rho(a, i)
		p[i*n+a] -= c.rho(a, j)
	}
}

// term12a is my_calc_d11.c:104-134,
//
//	w_ij += Σ_{b,g ∈ vir} d(b,g) · ( −Σ_{n ∈ occ, f ∈ vir}
//	          [2·⟨jn|fb⟩⟨fg|in⟩ − ⟨jn|bf⟩⟨fg|in⟩ − ⟨jn|fb⟩⟨fg|ni⟩ + 2·⟨jn|bf⟩⟨fg|ni⟩]
//	          / [(ε_j+ε_n−ε_f−ε_b)(ε_i+ε_n−ε_f−ε_g)] )
//
// The numerator factorizes as ⟨fg|in⟩·(2⟨jn|fb⟩ − ⟨jn|bf⟩) + ⟨fg|ni⟩·(2⟨jn|bf⟩ − ⟨jn|fb⟩),
// and the two denominators separate over b and g. So the inner (f,n) sum is a matrix product
// between a b-indexed and a g-indexed factor: building the two factors costs O(n_v²n_o)
// integral lookups instead of the O(n_v³n_o) the literal quadruple loop would pay, and the
// contraction itself is pure arithmetic. That is the whole optimization, and
// TestCorrMatchesLiteralLoops gates it against a verbatim transcription of the legacy.
func (c *isrCorr) term12a(p []float64, i, j int) {
	nocc, nvir, n := c.nocc, c.nvir, c.norb
	m := nvir * nocc // the composite (f,n) index

	// aFac[b][(f,n)] = (2⟨jn|fb⟩ − ⟨jn|bf⟩)/(ε_j+ε_n−ε_f−ε_b), and aSwap the exchanged pair.
	aFac := make([]float64, nvir*m)
	aSwap := make([]float64, nvir*m)
	for bi := range nvir {
		b := nocc + bi
		for fi := range nvir {
			f := nocc + fi
			for nn := range nocc {
				den := c.eps[j] + c.eps[nn] - c.eps[f] - c.eps[b]
				x, y := c.v(j, nn, f, b), c.v(j, nn, b, f)
				aFac[bi*m+fi*nocc+nn] = (2*x - y) / den
				aSwap[bi*m+fi*nocc+nn] = (2*y - x) / den
			}
		}
	}
	// bFac[g][(f,n)] = ⟨fg|in⟩/(ε_i+ε_n−ε_f−ε_g), and bSwap the ⟨fg|ni⟩ partner.
	bFac := make([]float64, nvir*m)
	bSwap := make([]float64, nvir*m)
	for gi := range nvir {
		g := nocc + gi
		for fi := range nvir {
			f := nocc + fi
			for nn := range nocc {
				den := c.eps[i] + c.eps[nn] - c.eps[f] - c.eps[g]
				bFac[gi*m+fi*nocc+nn] = c.v(f, g, i, nn) / den
				bSwap[gi*m+fi*nocc+nn] = c.v(f, g, nn, i) / den
			}
		}
	}

	for bi := range nvir {
		for gi := range nvir {
			var s float64
			for x := range m {
				s += aFac[bi*m+x]*bFac[gi*m+x] + aSwap[bi*m+x]*bSwap[gi*m+x]
			}
			p[(nocc+bi)*n+(nocc+gi)] -= s
		}
	}
}

// term12b is my_calc_d11.c:136-170,
//
//	w_ij += Σ_{k,n ∈ occ} d(n,k) · ( ½·Σ_{f,g ∈ vir}
//	          [2·⟨jk|fg⟩⟨fg|in⟩ − ⟨jk|gf⟩⟨fg|in⟩ − ⟨jk|fg⟩⟨fg|ni⟩ + 2·⟨jk|gf⟩⟨fg|ni⟩]
//	          / [(ε_j+ε_k−ε_f−ε_g)(ε_i+ε_n−ε_f−ε_g)] )
//
// Same factorization as (12a), now separating over the two occupied labels k and n.
func (c *isrCorr) term12b(p []float64, i, j int) {
	nocc, nvir, n := c.nocc, c.nvir, c.norb
	m := nvir * nvir // the composite (f,g) index

	kFac := make([]float64, nocc*m)
	kSwap := make([]float64, nocc*m)
	for k := range nocc {
		for fi := range nvir {
			f := nocc + fi
			for gi := range nvir {
				g := nocc + gi
				den := c.eps[j] + c.eps[k] - c.eps[f] - c.eps[g]
				x, y := c.v(j, k, f, g), c.v(j, k, g, f)
				kFac[k*m+fi*nvir+gi] = (2*x - y) / den
				kSwap[k*m+fi*nvir+gi] = (2*y - x) / den
			}
		}
	}
	nFac := make([]float64, nocc*m)
	nSwap := make([]float64, nocc*m)
	for nn := range nocc {
		for fi := range nvir {
			f := nocc + fi
			for gi := range nvir {
				g := nocc + gi
				den := c.eps[i] + c.eps[nn] - c.eps[f] - c.eps[g]
				nFac[nn*m+fi*nvir+gi] = c.v(f, g, i, nn) / den
				nSwap[nn*m+fi*nvir+gi] = c.v(f, g, nn, i) / den
			}
		}
	}

	for nn := range nocc {
		for k := range nocc {
			var s float64
			for x := range m {
				s += kFac[k*m+x]*nFac[nn*m+x] + kSwap[k*m+x]*nSwap[nn*m+x]
			}
			p[nn*n+k] += 0.5 * s
		}
	}
}

// term12c is my_calc_d11.c:173-218: two sums over (f,g ∈ vir, n ∈ occ), one contracting with
// d(j,k) and one with d(i,k), k over the occupied orbitals.
//
//	S1(k) = −¼·Σ [ ⟨nk|fg⟩⟨fg|in⟩ − 2⟨nk|gf⟩⟨fg|in⟩ − 2⟨nk|fg⟩⟨fg|ni⟩ + ⟨nk|gf⟩⟨fg|ni⟩ ]
//	              / [(ε_n+ε_k−ε_f−ε_g)(ε_i+ε_n−ε_f−ε_g)]
//	S2(k) = −¼·Σ [ ⟨fg|nk⟩⟨jn|fg⟩ − 2⟨fg|kn⟩⟨jn|fg⟩ − 2⟨fg|nk⟩⟨jn|gf⟩ + ⟨fg|kn⟩⟨jn|gf⟩ ]
//	              / [(ε_n+ε_k−ε_f−ε_g)(ε_j+ε_n−ε_f−ε_g)]
//	w_ij += S1(k)·d(j,k) + S2(k)·d(i,k)
//
// Cheap enough (O(n_o²n_v²)) to leave as the literal loop.
func (c *isrCorr) term12c(p []float64, i, j int) {
	nocc, n := c.nocc, c.norb
	for k := range nocc {
		var s1, s2 float64
		for f := nocc; f < c.norb; f++ {
			for g := nocc; g < c.norb; g++ {
				for nn := range nocc {
					dkn := c.eps[nn] + c.eps[k] - c.eps[f] - c.eps[g]

					vnkfg, vnkgf := c.v(nn, k, f, g), c.v(nn, k, g, f)
					vfgin, vfgni := c.v(f, g, i, nn), c.v(f, g, nn, i)
					s1 -= 0.25 * (vnkfg*vfgin - 2*vnkgf*vfgin - 2*vnkfg*vfgni + vnkgf*vfgni) /
						(dkn * (c.eps[i] + c.eps[nn] - c.eps[f] - c.eps[g]))

					vfgnk, vfgkn := c.v(f, g, nn, k), c.v(f, g, k, nn)
					vjnfg, vjngf := c.v(j, nn, f, g), c.v(j, nn, g, f)
					s2 -= 0.25 * (vfgnk*vjnfg - 2*vfgkn*vjnfg - 2*vfgnk*vjngf + vfgkn*vjngf) /
						(dkn * (c.eps[j] + c.eps[nn] - c.eps[f] - c.eps[g]))
				}
			}
		}
		p[j*n+k] += s1
		p[i*n+k] += s2
	}
}

// d11 is the correlation correction to ⟨i|D̂|j⟩ between two 1h configurations: the
// precomputed density of this hole pair, contracted with the operator.
func (o *ISRDipole) d11corr(i, j int) float64 {
	p := o.corr.p11[i][j]
	if p == nil {
		return 0
	}
	n := o.corr.norb
	var acc float64
	for r := range n {
		for s := range n {
			if v := p[r*n+s]; v != 0 {
				acc += v * o.d.At(r, s)
			}
		}
	}
	return acc
}

// ph8 is the (8a)/(8b) sum, which both terms share:
//
//	Σ_{p ∈ vir, q ∈ occ} d(p,q) · [2·⟨qk|pa⟩ − ⟨qk|ap⟩] / (ε_q+ε_k−ε_p−ε_a)
//
// It depends only on the spectator hole k and the particle a, so it is tabulated once per
// Cartesian component (nocc × nvir) rather than recomputed per matrix element.
func (o *ISRDipole) ph8(k, a int) float64 { return o.t8[k*o.corr.nvir+(a-o.corr.nocc)] }

// buildT8 tabulates ph8 over every (occupied, virtual) pair.
func (o *ISRDipole) buildT8() []float64 {
	c := o.corr
	t := make([]float64, c.nocc*c.nvir)
	for k := range c.nocc {
		for ai := range c.nvir {
			a := c.nocc + ai
			var s float64
			for p := c.nocc; p < c.norb; p++ {
				for q := range c.nocc {
					d := o.d.At(p, q)
					if d == 0 {
						continue
					}
					s += d * (2*c.v(q, k, p, a) - c.v(q, k, a, p)) /
						(c.eps[q] + c.eps[k] - c.eps[p] - c.eps[a])
				}
			}
			t[k*c.nvir+ai] = s
		}
	}
	return t
}

// mainSatCorr is the first-order correction to ⟨i|D̂|c⟩ between the 1h config i and the 2h1p
// config c (holes k ≤ l, particle a) — the (8a)/(8b)/(8c) terms of my_calc_d12.c, each
// negated to match the sign convention mainSat fixed against the secular coupling block.
//
// Note (8c) has no delta: it is nonzero for *every* 1h/2h1p pair, not only when i is one of
// the two holes. That is what makes the block dense, and why sparsify stops pruning it once
// corrections are on — the zeroth-order sparsity is not the operator's sparsity any more.
func (o *ISRDipole) mainSatCorr(i int, c Config) float64 {
	cr := o.corr
	k, l, a := c.Occ[0], c.Occ[1], o.vir(c)

	// (8c), summed over every virtual p; ⟨kl|pa⟩ vanishes unless irrep(p) matches, so the
	// legacy's irrep window on p is enforced by the integral itself.
	var s8c float64
	if k == l {
		for p := cr.nocc; p < cr.norb; p++ {
			den := 2*cr.eps[k] - cr.eps[a] - cr.eps[p]
			s8c += -cr.v(k, k, a, p) / den * o.d.At(p, i)
		}
		// Legacy (8b) rides on the same delta as the k == l zeroth-order term, with no
		// spin-coupling prefactor.
		v := -s8c
		if i == k {
			v -= o.ph8(k, a)
		}
		return v
	}

	for p := cr.nocc; p < cr.norb; p++ {
		den := cr.eps[k] + cr.eps[l] - cr.eps[p] - cr.eps[a]
		vklpa, vklap := cr.v(k, l, p, a), cr.v(k, l, a, p)
		var coef float64
		if c.Typ == 0 {
			coef = sqrt1_2 * (-vklpa - vklap) / den
		} else {
			coef = sqrt3_2 * (vklpa - vklap) / den
		}
		s8c += coef * o.d.At(p, i)
	}

	v := -s8c
	switch {
	case i == k: // (8a), riding on the same delta as the (7b) zeroth-order term
		if c.Typ == 0 {
			v -= sqrt1_2 * o.ph8(l, a)
		} else {
			v += sqrt3_2 * o.ph8(l, a)
		}
	case i == l: // (8b), riding on (7a)
		if c.Typ == 0 {
			v -= sqrt1_2 * o.ph8(k, a)
		} else {
			v -= sqrt3_2 * o.ph8(k, a)
		}
	}
	return v
}

// dNull is the correlated operator moment D₀⁽²⁾ = 2·Tr(d·ρ_full) that (13a) puts on the
// 1h/1h diagonal — my_calc_d_null.c, doubled by its driver
// (nd_adc3_cap_matrix.cpp:162, "D(2) must not include D(0) if the density is full").
//
// It is *not* a uniform shift of the operator: the 2h1p diagonal keeps the uncorrelated
// d_null_null = 2·Σ_occ d_kk (my_calc_d22_diag.c (1a)). The difference between the two lives
// on the 1h block alone, so unlike a true multiple of the identity it does not cancel out of
// a transition moment.
func (o *ISRDipole) dNull() float64 {
	if o.corr == nil || o.corr.rho == nil {
		return o.d0
	}
	acc := o.d0 // the reference part, 2·Σ_occ d_kk
	for p := range o.corr.norb {
		for q := range o.corr.norb {
			acc += 2 * o.d.At(p, q) * o.corr.rho(p, q)
		}
	}
	return acc
}

// NewISRDipolesWithCorr builds all three Cartesian components of the correlation-corrected
// ISR operator over a single space. A nil opts gives the plain zeroth-order operator.
func NewISRDipolesWithCorr(sp *Space, dmo [3]backend.Mat, opts *ISROptions) ([3]*ISRDipole, error) {
	return NewISRDipolesCrossWithCorr(sp, sp, dmo, opts)
}

// NewISRDipolesCrossWithCorr is the rectangular form: the three components between two
// spaces, sharing one precomputed correction context.
func NewISRDipolesCrossWithCorr(bra, ket *Space, dmo [3]backend.Mat, opts *ISROptions) ([3]*ISRDipole, error) {
	var ops [3]*ISRDipole
	if opts == nil {
		return NewISRDipolesCross(bra, ket, dmo)
	}
	// A CVS ADC(4) bra is allowed but is the caller's decision, not ours. The terms below are
	// the O(2)/O(1) ISR expansion that matches an ADC(2)/(3) secular matrix; over a 3h2p space
	// they are a partial, order-inconsistent improvement (an order-consistent ADC(4) property
	// matrix additionally needs the O(3)/O(4) terms built from t₂⁽²⁾/t₁⁽²⁾/t₃⁽²⁾, none of which
	// exist here, and there is no 3h2p correction at all). Structurally they apply cleanly —
	// the corrections only ever touch the 1h and 2h1p blocks, and the 3h2p rows stay zero just
	// as they already do — so the CLI gates this behind an opt-in flag rather than the API.
	corr, err := newISRCorr(bra, ket, opts)
	if err != nil {
		return ops, err
	}
	for x := range 3 {
		op, err := newISRDipoleCorr(bra, ket, dmo[x], corr)
		if err != nil {
			return ops, fmt.Errorf("component %d: %w", x, err)
		}
		ops[x] = op
	}
	return ops, nil
}

// newISRDipoleCorr assembles one component against a shared correction context.
func newISRDipoleCorr(bra, ket *Space, dmo backend.Mat, corr *isrCorr) (*ISRDipole, error) {
	if err := checkDipole(bra, dmo); err != nil {
		return nil, err
	}
	if err := compatible(bra, ket); err != nil {
		return nil, err
	}
	o := &ISRDipole{bra: bra, ket: ket, d: dmo, corr: corr}
	for i := range ket.Nocc {
		o.d0 += 2 * dmo.At(i, i)
	}
	o.dnull = o.dNull()
	o.t8 = o.buildT8()
	o.rows = o.sparsify() // must come last: it calls At, which reads dnull and t8
	return o, nil
}
