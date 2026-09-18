package sip

import "sort"

// slater.go — Slater-Condon matrix elements between determinants, in the
// spin-orbital basis, used as the exact oracle for the ADC(2,2) blocks.
//
// Why this exists. The ISR secular matrix is M_IJ = <Ψ~_I|H - E_0|Ψ~_J>, and
// through FIRST order the intermediate states are just the HF configurations
// themselves — the perturbation corrections to the ISs first contribute at second
// order. So every zeroth- and first-order block of ADC(2,2) is a plain
// Slater-Condon matrix element:
//
//	A7  M^0_{2h1p,2h1p}    A8  M^1_{2h1p,2h1p}
//	A15-A17 M^1_{2h1p,3h2p}
//	A18 M^0_{3h2p,3h2p}    A19-A22 M^1_{3h2p,3h2p}
//
// That covers every genuinely new block of ADC(2,2) except the second-order
// 2h1p/2h1p one (A9-A14) — which is exactly the block the paper singles out as
// the only thing not already present in non-Dyson ADC(3).
//
// Computing them this way rather than transcribing A15-A22 is deliberate. Those
// equations are dense in permutation operators (‖klm↔lmk, k'l'm'↔l'm'k'‖ and so
// on) whose sign conventions are easy to misread, and there is no Fortran
// counterpart in ../ADC to check a transcription against — adc2_pol is the
// polarization propagator and has no 3h2p class at all. Slater-Condon has no such
// ambiguity: it is the definition.
//
// This general form is the ORACLE, not the production path. It sums over all
// occupied spin orbitals per element, which is O(N^2) work for a diagonal
// element; the production blocks use the normal-ordered closed forms and are
// gated against this.

// det is a determinant as its sorted occupied spin-orbital indices (so.Idx()).
type det []int

// refDet is the closed-shell reference |Φ0>: every spin orbital below 2*nocc.
func refDet(nocc int) det {
	d := make(det, 2*nocc)
	for i := range d {
		d[i] = i
	}
	return d
}

// excite applies c†_A c_K c_L ... to the reference: removes the hole spin
// orbitals, adds the particle ones, and returns the sorted determinant together
// with the sign relating it to the operator string written holes-innermost.
//
// The sign is accumulated the same way detSign2 does it for the 2h1p case, but
// for an arbitrary number of holes and particles: annihilate the holes in the
// order given (each costing the parity of its position in the *current* string),
// then create the particles (each sorting in from the left of the remainder).
func excite(nocc int, holes, parts []sorb) (det, float64) {
	cur := refDet(nocc)
	sign := 1.0
	for _, h := range holes {
		x := h.Idx()
		pos := sort.SearchInts(cur, x)
		if pos >= len(cur) || cur[pos] != x {
			return nil, 0 // annihilating an empty spin orbital
		}
		if pos%2 == 1 {
			sign = -sign
		}
		cur = append(cur[:pos:pos], cur[pos+1:]...)
	}
	for _, p := range parts {
		x := p.Idx()
		pos := sort.SearchInts(cur, x)
		if pos < len(cur) && cur[pos] == x {
			return nil, 0 // creating on an occupied spin orbital
		}
		// c†_x written leftmost must move past `pos` occupied orbitals to reach
		// its sorted place.
		if pos%2 == 1 {
			sign = -sign
		}
		cur = append(cur, 0)
		copy(cur[pos+1:], cur[pos:])
		cur[pos] = x
	}
	return cur, sign
}

// spinOrbOf turns a canonical spin-orbital index back into (orbital, spin).
func spinOrbOf(x int) sorb { return sorb{Orb: x / 2, Dn: x%2 == 1} }

// eriIdx is <PQ||RS> for canonical spin-orbital indices.
func (e *elements) eriIdx(p, q, r, s int) float64 {
	return e.eriSO(spinOrbOf(p), spinOrbOf(q), spinOrbOf(r), spinOrbOf(s))
}

// hCore is the one-electron integral h_PQ in the spin-orbital basis, recovered
// from the canonical HF data this package already carries: the Fock matrix is
// diagonal with F_PQ = ε_P δ_PQ, and F_PQ = h_PQ + Σ_I <PI||QI> over occupied
// spin orbitals I, so h_PQ = ε_P δ_PQ − Σ_I <PI||QI>. No new input is needed —
// which is the point, since FCIDUMP's own h is in the spatial AO-to-MO basis and
// the rest of this package never touches it.
func (e *elements) hCore(p, q int) float64 {
	var v float64
	if p == q {
		v = e.eps[p/2]
	}
	if spinOrbOf(p).Dn != spinOrbOf(q).Dn {
		return v // h is spin diagonal; the sum below would vanish anyway
	}
	for i := range 2 * e.nocc {
		v -= e.eriIdx(p, i, q, i)
	}
	return v
}

// diffDet reports the spin orbitals occupied in a but not in b ("from"), those in
// b but not in a ("to"), and the permutation sign that lines the two
// determinants up so Slater-Condon applies. ok is false when they differ by more
// than two spin orbitals, where H has no matrix element.
//
// Both determinants are sorted, so moving the differing orbitals to the front of
// each costs the sum of their positions in transpositions — which is the whole
// sign.
func diffDet(a, b det) (from, to []int, sign float64, ok bool) {
	inB := func(x int) bool {
		i := sort.SearchInts(b, x)
		return i < len(b) && b[i] == x
	}
	inA := func(x int) bool {
		i := sort.SearchInts(a, x)
		return i < len(a) && a[i] == x
	}
	pos := 0
	for i, x := range a {
		if !inB(x) {
			if len(from) == 2 {
				return nil, nil, 0, false
			}
			from = append(from, x)
			pos += i
		}
	}
	for i, x := range b {
		if !inA(x) {
			if len(to) == 2 {
				return nil, nil, 0, false
			}
			to = append(to, x)
			pos += i
		}
	}
	if len(from) != len(to) {
		return nil, nil, 0, false // different electron counts
	}
	sign = 1
	if pos%2 == 1 {
		sign = -1
	}
	return from, to, sign, true
}

// hamElem is the Slater-Condon matrix element <a|H|b> in the spin-orbital basis.
// Callers wanting the secular matrix want <a|H - E_0|b>, i.e. this minus
// hamElem(ref, ref) on the diagonal; see shiftedHamElem.
func (e *elements) hamElem(a, b det) float64 {
	from, to, sign, ok := diffDet(a, b)
	if !ok {
		return 0
	}
	switch len(from) {
	case 0:
		var v float64
		for _, i := range a {
			v += e.hCore(i, i)
		}
		for _, i := range a {
			for _, j := range a {
				v += 0.5 * e.eriIdx(i, j, i, j)
			}
		}
		return v
	case 1:
		p, q := from[0], to[0]
		v := e.hCore(p, q)
		for _, i := range a {
			if i == p {
				continue
			}
			v += e.eriIdx(p, i, q, i)
		}
		return sign * v
	default:
		return sign * e.eriIdx(from[0], from[1], to[0], to[1])
	}
}

// shiftedHamElem is <a|H - E_0|b>, the secular-matrix element through first order.
func (e *elements) shiftedHamElem(a, b det, e0 float64) float64 {
	v := e.hamElem(a, b)
	if len(a) == len(b) {
		same := true
		for i := range a {
			if a[i] != b[i] {
				same = false
				break
			}
		}
		if same {
			v -= e0
		}
	}
	return v
}

// refEnergy is <Φ0|H|Φ0>, the shift E_0 that the secular matrix is measured from.
func (e *elements) refEnergy() float64 {
	r := refDet(e.nocc)
	return e.hamElem(r, r)
}

// ---------------------------------------------------------------------------
// Normal-ordered closed forms — the production path.
// ---------------------------------------------------------------------------
//
// hamElem above sums over every occupied spin orbital, which is O(N^2) work for a
// diagonal element. That is fine for an oracle on H2O and hopeless for a real
// 3h2p space. Writing H - E_0 normal ordered with respect to |Φ0> collapses those
// sums onto the handful of orbitals a configuration actually disturbs, giving an
// O(1) element. exc/hamNO are that form; TestHamNOMatchesOracle gates them
// against hamElem.

// exc is a determinant written as its excitation from |Φ0>: the spin orbitals
// emptied (H) and filled (P), each sorted ascending. Every determinant in the ADC
// space is at most a 3h2p excitation, so both slices are tiny.
type exc struct {
	H []int // holes, all < 2*nocc
	P []int // particles, all >= 2*nocc
}

// pos is the index of spin orbital o within this determinant's sorted occupation
// list. Computing it from the excitation rather than from a materialized list is
// what keeps the element O(1): a hole below o shifts everything after it down, a
// particle below o shifts it up.
func (x exc) pos(nocc, o int) int {
	if o >= 2*nocc {
		n := 2*nocc - len(x.H)
		for _, p := range x.P {
			if p < o {
				n++
			}
		}
		return n
	}
	n := o
	for _, h := range x.H {
		if h < o {
			n--
		}
	}
	return n
}

// has reports membership in a small sorted slice.
func has(s []int, x int) bool {
	for _, v := range s {
		if v == x {
			return true
		}
	}
	return false
}

// diffExc is diffDet expressed on excitations. A spin orbital is occupied in A but
// not in B when it is a particle of A that B does not have, or a hole of B that A
// does not have — the second case being an orbital both determinants inherit from
// |Φ0| but only B removes.
//
// The sign is the parity of the summed positions of the differing orbitals in
// their own determinants, exactly as in diffDet: bringing k differing orbitals to
// the front of a list costs sum(positions) - k(k-1)/2 transpositions, and the
// correction is even for k <= 2.
//
// Both lists are sorted by spin-orbital index before being returned, which is what
// PAIRS them: from[i] must be the orbital that to[i] replaces. Within a
// determinant, pos is monotonic in the index (a particle always sorts after every
// occupied orbital), so an index sort is a position sort. Collecting particles and
// holes separately without this sort is only self-consistent while both
// determinants belong to the same excitation class — it silently mispairs a
// 2h1p against a 3h2p, which is precisely the 2h1p/3h2p coupling block.
func diffExc(nocc int, a, b exc) (from, to []int, sign float64, ok bool) {
	add := func(dst *[]int, x int) bool {
		if len(*dst) == 2 {
			return false
		}
		*dst = append(*dst, x)
		return true
	}
	parity := 0
	for _, p := range a.P {
		if !has(b.P, p) {
			if !add(&from, p) {
				return nil, nil, 0, false
			}
			parity += a.pos(nocc, p)
		}
	}
	for _, h := range b.H {
		if !has(a.H, h) {
			if !add(&from, h) {
				return nil, nil, 0, false
			}
			parity += a.pos(nocc, h)
		}
	}
	for _, p := range b.P {
		if !has(a.P, p) {
			if !add(&to, p) {
				return nil, nil, 0, false
			}
			parity += b.pos(nocc, p)
		}
	}
	for _, h := range a.H {
		if !has(b.H, h) {
			if !add(&to, h) {
				return nil, nil, 0, false
			}
			parity += b.pos(nocc, h)
		}
	}
	if len(from) != len(to) {
		return nil, nil, 0, false
	}
	sortSmall(from)
	sortSmall(to)
	sign = 1
	if parity%2 == 1 {
		sign = -1
	}
	return from, to, sign, true
}

// hamNO is <D_a|H - E_0|D_b> for determinants given as excitations from |Φ0>, in
// closed normal-ordered form.
//
//	no difference: sum_a eps_a - sum_i eps_i
//	               + sum_{a<b} <ab||ab> + sum_{i<j} <ij||ij> - sum_{a,i} <ai||ai>
//	one  (p -> r): f_pr + sum_{u in P_a} <pu||ru> - sum_{u in H_a} <pu||ru>,
//	               and f_pr = eps_p delta_pr vanishes because canonical HF orbitals
//	               make the Fock matrix diagonal and p != r here. The u = p term of
//	               either sum vanishes identically, <pp||rp> = 0, so no exclusion
//	               is needed.
//	two (pq -> rs): <pq||rs>
//
// each times the alignment sign from diffExc.
func (e *elements) hamNO(a, b exc) float64 {
	from, to, sign, ok := diffExc(e.nocc, a, b)
	if !ok {
		return 0
	}
	switch len(from) {
	case 0:
		var v float64
		for _, p := range a.P {
			v += e.eps[p/2]
		}
		for _, h := range a.H {
			v -= e.eps[h/2]
		}
		for i, p := range a.P {
			for _, q := range a.P[i+1:] {
				v += e.eriIdx(p, q, p, q)
			}
		}
		for i, h := range a.H {
			for _, g := range a.H[i+1:] {
				v += e.eriIdx(h, g, h, g)
			}
		}
		for _, p := range a.P {
			for _, h := range a.H {
				v -= e.eriIdx(p, h, p, h)
			}
		}
		return v
	case 1:
		p, r := from[0], to[0]
		var v float64
		for _, u := range a.P {
			v += e.eriIdx(p, u, r, u)
		}
		for _, u := range a.H {
			v -= e.eriIdx(p, u, r, u)
		}
		return sign * v
	default:
		return sign * e.eriIdx(from[0], from[1], to[0], to[1])
	}
}
