package sip

// spinadapt.go — expansion of a spin-adapted ADC configuration over the
// *primitive* spin-orbital configurations of Eq. (9) of Kolorenč/Averbukh,
// J. Chem. Phys. 152, 214107 (2020), in which Appendix A2-A22 is written.
//
// spincsf.go produces the doublet spin functions as coefficients over canonically
// ordered determinants. This file does the two remaining conversions:
//
//  1. determinant -> operator string. The paper's basis elements are operator
//     strings c_k (1h) and c†_a c_k c_l with k < l (2h1p), not sorted
//     determinants, and the two differ by a fermionic reordering sign that
//     depends on which *spins* were removed. detSign2 computes it.
//  2. role order -> paper order. A Config carries its holes in macro roles
//     (k from the outer symmetry loop, l from the inner; see config.go), which is
//     not the paper's k < l spin-orbital order, so the hole pair may need one
//     transposition — another sign.
//
// The result is a short list of (primitive, coefficient) pairs per spin-adapted
// configuration, which elements22.go contracts with the appendix formulas.

// sorb is a spin orbital: a spatial orbital index plus a spin. Its Idx is the
// canonical ordering used throughout — spatial orbital major, alpha before beta —
// which is what makes the reordering signs well defined.
type sorb struct {
	Orb int
	Dn  bool // false = alpha (up), true = beta (down)
}

// Idx is the canonical spin-orbital index 2*Orb (+1 for beta).
func (s sorb) Idx() int {
	if s.Dn {
		return 2*s.Orb + 1
	}
	return 2 * s.Orb
}

// prim2 is a primitive 2h1p spin-orbital configuration c†_A c_K c_L |Φ0>, with
// the holes in the paper's K < L spin-orbital order.
type prim2 struct {
	K, L sorb // holes, K.Idx() < L.Idx()
	A    sorb // particle
}

// term2 is one primitive and its weight in a spin-adapted configuration.
type term2 struct {
	P prim2
	C float64
}

// flip returns the opposite spin of the same spatial orbital. A hole determinant
// records the electron that *stayed*, so the electron that was removed carries
// the opposite spin — that inversion is the whole reason this helper exists.
func flip(s sorb) sorb { return sorb{Orb: s.Orb, Dn: !s.Dn} }

// detSign2 is the sign relating the operator string c†_A c_K c_L |Φ0> to the
// canonically ordered determinant that carries the same occupation.
//
// With |Φ0> holding every spin orbital of index < 2*nocc, annihilating L costs
// (-1)^L.Idx(); annihilating K then costs (-1)^(K.Idx() - [L.Idx() < K.Idx()]),
// because L is already gone from the string. Creating the virtual A costs
// (-1)^(N-2) = +1, since A sorts after every remaining occupied orbital and N is
// even. Modulo 2 the subtraction is an addition, which is how it is written here.
func detSign2(k, l sorb) float64 {
	e := k.Idx() + l.Idx()
	if l.Idx() < k.Idx() {
		e++
	}
	if e%2 == 0 {
		return 1
	}
	return -1
}

// expand2 expands a spin-adapted 2h1p configuration over primitives.
//
// The open shells are, in this order, the singly occupied hole orbitals followed
// by the particle. For k != l that is (k, l, a) — three open shells, two spin
// functions, selected by cfg.Typ. For k == l the orbital is emptied, leaving the
// particle as the only open shell: one spin function, and cfg.Typ is ignored (as
// config.go's addSat already guarantees by pushing only Typ 0 there).
func (e *elements) expand2(cfg Config) []term2 {
	k, l := cfg.Occ[0], cfg.Occ[1]
	a := e.nocc + cfg.Vir
	if k == l {
		// Both electrons of k removed: the determinant is fixed up to the
		// particle spin, which M_s = +1/2 pins to alpha.
		ha := sorb{Orb: k, Dn: false}
		hb := sorb{Orb: k, Dn: true}
		p := prim2{K: ha, L: hb, A: sorb{Orb: a, Dn: false}}
		return []term2{{P: p, C: detSign2(ha, hb)}}
	}
	b := doubletCSF(3)
	row := b.Coef[cfg.Typ]
	out := make([]term2, 0, len(b.Dets))
	for d, det := range b.Dets {
		c := row[d]
		if c == 0 {
			continue
		}
		// det[i] is the spin that REMAINS in open shell i, so the removed spin
		// is its flip; the particle's entry is the spin that was created.
		hk := flip(sorb{Orb: k, Dn: det[0] == spinDown})
		hl := flip(sorb{Orb: l, Dn: det[1] == spinDown})
		pa := sorb{Orb: a, Dn: det[2] == spinDown}
		coef := c * detSign2(hk, hl)
		if hk.Idx() > hl.Idx() { // put the pair in the paper's K < L order
			hk, hl = hl, hk
			coef = -coef
		}
		out = append(out, term2{P: prim2{K: hk, L: hl, A: pa}, C: coef})
	}
	return out
}

// expand1 expands the spin-adapted 1h configuration |j>. With one open shell the
// single doublet function is the determinant whose remaining electron is alpha,
// so the removed spin orbital is (j, beta) and the operator string c_(j,beta)
// relates to the sorted determinant by (-1)^(2j+1) = -1.
func expand1(j int) (sorb, float64) {
	return sorb{Orb: j, Dn: true}, -1
}

// eriSO is the antisymmetrized spin-orbital integral <PQ||RS> = <PQ|RS> - <PQ|SR>
// in physicist ordering, the paper's V_ab[ij]. A physicist integral <PQ|RS> is
// nonzero only when the spin of P matches R and that of Q matches S, which is the
// only place spin enters.
func (e *elements) eriSO(p, q, r, s sorb) float64 {
	var v float64
	if p.Dn == r.Dn && q.Dn == s.Dn {
		v += e.v(p.Orb, q.Orb, r.Orb, s.Orb)
	}
	if p.Dn == s.Dn && q.Dn == r.Dn {
		v -= e.v(p.Orb, q.Orb, s.Orb, r.Orb)
	}
	return v
}

// ---------------------------------------------------------------------------
// Expansion over determinants.
// ---------------------------------------------------------------------------
//
// expand2 above produces the paper's primitive operator strings, which is what
// Appendix A2-A14 is written in terms of. For every block that is a plain
// Hamiltonian matrix element — A7, A8, A15-A22 — it is more direct to expand over
// DETERMINANTS instead and hand them to the Slater-Condon machinery in slater.go.
//
// The two expansions carry the same coefficients. The operator-string sign
// detSign2 applies converting determinant -> string, and excite applies exactly
// the same sign converting back, so the round trip is the identity: a spin-adapted
// ADC configuration is simply its branching-diagram spin function over sorted
// determinants. TestSlaterGateC22 established that empirically for the 2h1p block,
// and TestExpandDetMatchesPrimitive keeps the two routes pinned to each other.

// detTerm is one determinant, as an excitation from |Φ0>, and its weight in a
// spin-adapted configuration.
type detTerm struct {
	E exc
	C float64
}

// mkExc builds an exc from hole and particle spin orbitals, sorting each side.
// The lists are at most three and two long, so an insertion sort is the whole
// cost.
func mkExc(holes, parts []sorb) exc {
	h := make([]int, len(holes))
	for i, s := range holes {
		h[i] = s.Idx()
	}
	p := make([]int, len(parts))
	for i, s := range parts {
		p[i] = s.Idx()
	}
	sortSmall(h)
	sortSmall(p)
	return exc{H: h, P: p}
}

func sortSmall(a []int) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j] < a[j-1]; j-- {
			a[j], a[j-1] = a[j-1], a[j]
		}
	}
}

// expandDet2 expands a spin-adapted 2h1p configuration over determinants. The
// open shells are taken in the Config's own role order (Occ[0], Occ[1], particle),
// which is what makes spin-function index 0/1 agree with Config.Typ.
func (e *elements) expandDet2(cfg Config) []detTerm {
	k, l := cfg.Occ[0], cfg.Occ[1]
	a := e.nocc + cfg.Vir
	if k == l {
		// The orbital is emptied, so the particle is the only open shell and
		// M_s = +1/2 fixes its spin to alpha.
		return []detTerm{{
			E: mkExc([]sorb{{Orb: k}, {Orb: k, Dn: true}}, []sorb{{Orb: a}}),
			C: 1,
		}}
	}
	b := doubletCSF(3)
	row := b.Coef[cfg.Typ]
	out := make([]detTerm, 0, len(b.Dets))
	for d, det := range b.Dets {
		if row[d] == 0 {
			continue
		}
		holes := []sorb{
			flip(sorb{Orb: k, Dn: det[0] == spinDown}),
			flip(sorb{Orb: l, Dn: det[1] == spinDown}),
		}
		parts := []sorb{{Orb: a, Dn: det[2] == spinDown}}
		out = append(out, detTerm{E: mkExc(holes, parts), C: row[d]})
	}
	return out
}

// holeShells3 classifies a 3h2p configuration's holes: the orbital emptied by a
// coincident pair (doubled, -1 when the three holes are distinct) and the singly
// occupied hole orbitals in role order.
func holeShells3(k, l, m int) (open []int, doubled int) {
	switch {
	case k == l:
		return []int{m}, k
	case l == m:
		return []int{k}, l
	case k == m:
		return []int{l}, k
	default:
		return []int{k, l, m}, -1
	}
}

// expandDet3 expands a spin-adapted 3h2p configuration over determinants.
//
// Open shells are the singly occupied hole orbitals in role order, followed by the
// singly occupied particle orbitals in role order. A coincident hole pair empties
// its orbital (both spins removed, no open shell); coincident particles doubly
// occupy theirs (both spins created, no open shell). That is exactly the counting
// behind openShells3/nSpin3, so the spin index runs 1..nSpin3 as Config3.Spin
// promises.
func (e *elements) expandDet3(cfg Config3) []detTerm {
	k, l, m := cfg.Core, cfg.L, cfg.M
	ai, aj := e.nocc+cfg.I, e.nocc+cfg.J
	openHoles, doubled := holeShells3(k, l, m)

	var openOrbs []int
	openOrbs = append(openOrbs, openHoles...)
	if cfg.I != cfg.J {
		openOrbs = append(openOrbs, ai, aj)
	}

	b := doubletCSF(len(openOrbs))
	row := b.Coef[cfg.Spin-1]
	out := make([]detTerm, 0, len(b.Dets))
	for d, det := range b.Dets {
		if row[d] == 0 {
			continue
		}
		holes := make([]sorb, 0, 3)
		if doubled >= 0 {
			holes = append(holes, sorb{Orb: doubled}, sorb{Orb: doubled, Dn: true})
		}
		for i, o := range openHoles {
			holes = append(holes, flip(sorb{Orb: o, Dn: det[i] == spinDown}))
		}
		parts := make([]sorb, 0, 2)
		if cfg.I == cfg.J {
			parts = append(parts, sorb{Orb: ai}, sorb{Orb: ai, Dn: true})
		} else {
			off := len(openHoles)
			parts = append(parts,
				sorb{Orb: ai, Dn: det[off] == spinDown},
				sorb{Orb: aj, Dn: det[off+1] == spinDown})
		}
		out = append(out, detTerm{E: mkExc(holes, parts), C: row[d]})
	}
	return out
}
