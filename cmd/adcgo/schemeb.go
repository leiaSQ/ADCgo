package main

import (
	"fmt"
	"math"
	"os"
	"slices"

	"gonum.org/v1/gonum/mat"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fano"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// schemeb.go — the driver side of Fano scheme B (-fano-scheme b): adapted intermediate
// states (fano/adapted.go) built from the SIP secular matrix, classified by the two-hole
// character of each adapted state over SITE-localized occupied orbitals.
//
// For every particle orbital p, the 2h1p configurations with particle p span a block of
// the secular matrix whose eigenvectors are the adapted states: correlated dication
// states with an extra electron in p. Each adapted state is classified on its own:
//
//   - inner: its two-hole weight on pairs containing an inner-valence hole
//     (-fano-b-inner) — part of the decaying state's dressing, Q;
//   - one-site: its weight with both holes on one site, measured after rotating the
//     occupied orbitals onto the sites — shake-up satellites and closed double
//     ionization, excluded from both by default (-fano-b-onesite x), or bound (q);
//   - otherwise two-site: an open ICD/ETMD channel, P.
//
// The site character is computed from the spin-orbital two-hole amplitude, rotated, so it
// needs no spin-coupling convention: sip.DetExpansion gives each configuration over
// determinants, and the norm of the rotated antisymmetric amplitude restricted to same-
// site pairs is the one-site weight. 1h rows are Q; 3h2p rows keep the class the scheme A
// rule (-fano-qp / -fano-q) gives them — for a valence vacancy every triply ionized
// channel is closed, so they are not rotated.

// schemeBStats is what the run records about the adapted basis.
type schemeBStats struct {
	Blocks   int `json:"blocks"`
	AdaptedQ int `json:"adapted_q"`
	AdaptedP int `json:"adapted_p"`
	AdaptedX int `json:"adapted_x"`
	// AdaptedXBound counts two-site adapted states excluded for a compact (bound) particle,
	// and CompactVirtuals the virtuals classified compact (-fano-b-particle free).
	AdaptedXBound   int      `json:"adapted_x_bound_particle"`
	CompactVirtuals int      `json:"compact_virtuals"`
	LocalizeMin     float64  `json:"site_localization_min"` // smallest kept site-projection eigenvalue
	Sites           []string `json:"sites"`
	InnerOrbs       []int    `json:"inner_orbitals"`
}

// siteLocalize returns L (nocc x nocc, orthogonal): column k' is a site-localized
// occupied orbital over the canonical occupied ones, and owner[k'] its site. For each
// site the occupied space's projection onto the site's AO span, W_F = (S C_o)_F^T S_FF^-1
// (S C_o)_F, has eigenvalues near 1 for the directions the site owns.
//
// The site spans are not orthogonal to each other, so a direction can project strongly
// onto two of them — measured on the water dimer with a KBJ continuum set on each oxygen:
// the donor's span took 5 directions above 1/2 for its 4 orbitals. Directions are
// therefore assigned competitively: every site's eigenpairs in descending eigenvalue,
// each accepted only if more than half of it is new (orthogonal to those already
// accepted), until nocc are placed. Ghost-centre AOs are left out of the site spans when
// the sidecar marks them (atom_charges): occupied orbitals have no business there, and a
// diffuse set only blurs the boundary. The accepted directions are Loewdin-
// orthonormalized; the smallest accepted eigenvalue is the localization quality.
func siteLocalize(md *mo.Data, nocc int, siteOf []int, nsites int) (*mat.Dense, []int, float64, error) {
	nao := md.NAO
	SC := mat.NewDense(nao, nocc, nil)
	for mu := range nao {
		for k := range nocc {
			var s float64
			for nu := range nao {
				s += md.S.At(mu, nu) * md.C.At(nu, k)
			}
			SC.Set(mu, k, s)
		}
	}
	type candidate struct {
		site int
		w    float64
		vec  []float64
	}
	var cands []candidate
	var cols [][]float64
	var owner []int
	minKept := math.Inf(1)
	counts := make([]int, nsites)
	for f := range nsites {
		var aos []int
		for mu := range nao {
			a := md.AOAtom[mu]
			if a < 0 || a >= len(siteOf) || siteOf[a] != f {
				continue
			}
			if md.HasDipole && len(md.AtomCharges) == len(md.AtomNames) && md.AtomCharges[a] == 0 {
				continue // ghost centre
			}
			aos = append(aos, mu)
		}
		if len(aos) == 0 {
			continue
		}
		m := len(aos)
		Sff := mat.NewSymDense(m, nil)
		for i, mu := range aos {
			for j, nu := range aos {
				Sff.SetSym(i, j, md.S.At(mu, nu))
			}
		}
		var ev mat.EigenSym
		if !ev.Factorize(Sff, true) {
			return nil, nil, 0, fmt.Errorf("site %d: overlap eigendecomposition failed", f)
		}
		w := ev.Values(nil)
		var V mat.Dense
		ev.VectorsTo(&V)
		Sinv := mat.NewDense(m, m, nil) // pseudo-inverse: diffuse sets can be near-dependent
		cut := 1e-10 * slices.Max(w)
		for k, wk := range w {
			if wk <= cut {
				continue
			}
			for i := range m {
				for j := range m {
					Sinv.Set(i, j, Sinv.At(i, j)+V.At(i, k)*V.At(j, k)/wk)
				}
			}
		}
		B := mat.NewDense(m, nocc, nil)
		for i, mu := range aos {
			for k := range nocc {
				B.Set(i, k, SC.At(mu, k))
			}
		}
		var t1, W mat.Dense
		t1.Mul(Sinv, B)
		W.Mul(B.T(), &t1)
		Ws := mat.NewSymDense(nocc, nil)
		for i := range nocc {
			for j := range nocc {
				Ws.SetSym(i, j, 0.5*(W.At(i, j)+W.At(j, i)))
			}
		}
		var ew mat.EigenSym
		if !ew.Factorize(Ws, true) {
			return nil, nil, 0, fmt.Errorf("site %d: projection eigendecomposition failed", f)
		}
		wv := ew.Values(nil)
		var U mat.Dense
		ew.VectorsTo(&U)
		for k, wk := range wv {
			if wk > 0.1 {
				col := make([]float64, nocc)
				for i := range nocc {
					col[i] = U.At(i, k)
				}
				cands = append(cands, candidate{f, wk, col})
			}
		}
	}
	slices.SortStableFunc(cands, func(a, b candidate) int {
		switch {
		case a.w > b.w:
			return -1
		case a.w < b.w:
			return 1
		}
		return 0
	})
	var basis [][]float64 // orthonormal span of the accepted directions
	for _, c := range cands {
		if len(cols) == nocc {
			break
		}
		r := slices.Clone(c.vec)
		for _, b := range basis {
			var d float64
			for i := range nocc {
				d += b[i] * r[i]
			}
			for i := range nocc {
				r[i] -= d * b[i]
			}
		}
		var nr float64
		for _, x := range r {
			nr += x * x
		}
		if nr <= 0.5 {
			continue // mostly a direction another site already owns
		}
		for i := range r {
			r[i] /= math.Sqrt(nr)
		}
		basis = append(basis, r)
		cols = append(cols, c.vec)
		owner = append(owner, c.site)
		counts[c.site]++
		minKept = math.Min(minKept, c.w)
	}
	if len(cols) != nocc {
		return nil, nil, 0, fmt.Errorf("site localization found %d site-owned occupied directions "+
			"(per site %v), the space has %d occupied orbitals; the -group sites must cover the "+
			"molecules the occupied orbitals live on", len(cols), counts, nocc)
	}
	X := mat.NewDense(nocc, nocc, nil)
	for k, c := range cols {
		X.SetCol(k, c)
	}
	// Loewdin: L = X (X^T X)^-1/2
	var G mat.Dense
	G.Mul(X.T(), X)
	Gs := mat.NewSymDense(nocc, nil)
	for i := range nocc {
		for j := range nocc {
			Gs.SetSym(i, j, 0.5*(G.At(i, j)+G.At(j, i)))
		}
	}
	var eg mat.EigenSym
	if !eg.Factorize(Gs, true) {
		return nil, nil, 0, fmt.Errorf("site localization: Loewdin eigendecomposition failed")
	}
	gv := eg.Values(nil)
	if slices.Min(gv) < 1e-8 {
		return nil, nil, 0, fmt.Errorf("site localization: the site-owned directions are linearly "+
			"dependent (smallest Gram eigenvalue %.3g); the sites overlap", slices.Min(gv))
	}
	var Ug mat.Dense
	eg.VectorsTo(&Ug)
	isq := mat.NewDense(nocc, nocc, nil)
	for i := range nocc {
		for j := range nocc {
			var s float64
			for k := range nocc {
				s += Ug.At(i, k) * Ug.At(j, k) / math.Sqrt(gv[k])
			}
			isq.Set(i, j, s)
		}
	}
	var L mat.Dense
	L.Mul(X, isq)
	return &L, owner, minKept, nil
}

// twoHoleCharacter returns, for an adapted state with amplitudes u over the 2h1p rows of
// one particle block, its inner-valence weight and its one-site weight. dets[i] is
// rows[i]'s determinant expansion; L and owner are the site localization; inner flags the
// inner-valence spatial orbitals.
func twoHoleCharacter(u []float64, dets [][]sip.DetTerm, nocc int, L *mat.Dense, owner []int,
	inner []bool) (innerW, oneSite float64) {
	nso := 2 * nocc
	amps := map[int]*mat.Dense{} // per particle spin orbital: antisymmetric hole amplitude
	for i, terms := range dets {
		for _, t := range terms {
			if len(t.Holes) != 2 || len(t.Parts) != 1 {
				continue
			}
			A := t.Parts[0]
			T, ok := amps[A]
			if !ok {
				T = mat.NewDense(nso, nso, nil)
				amps[A] = T
			}
			k, l := t.Holes[0], t.Holes[1]
			v := u[i] * t.Coef
			T.Set(k, l, T.At(k, l)+v)
			T.Set(l, k, T.At(l, k)-v)
		}
	}
	// Spin-orbital rotation: holes keep their spin; spatial part by L.
	Ls := mat.NewDense(nso, nso, nil)
	for k := range nocc {
		for kp := range nocc {
			Ls.Set(2*k, 2*kp, L.At(k, kp))
			Ls.Set(2*k+1, 2*kp+1, L.At(k, kp))
		}
	}
	var norm float64
	for _, T := range amps {
		for K := range nso {
			for Lh := K + 1; Lh < nso; Lh++ {
				v := T.At(K, Lh)
				norm += v * v
				if inner[K/2] || inner[Lh/2] {
					innerW += v * v
				}
			}
		}
		var t1, Tr mat.Dense
		t1.Mul(T, Ls)
		Tr.Mul(Ls.T(), &t1)
		for K := range nso {
			for Lh := K + 1; Lh < nso; Lh++ {
				if owner[K/2] == owner[Lh/2] {
					v := Tr.At(K, Lh)
					oneSite += v * v
				}
			}
		}
	}
	if norm == 0 {
		return 0, 0
	}
	return innerW / norm, oneSite / norm
}

// splitVirtuals rotates the canonical virtuals vs (relative indices of one irrep) into
// compact and free ones: compact are the directions of the real-atom AO span projected
// onto these virtuals with eigenvalue above thresh, free their orthogonal complement (all
// diffuse and ghost-centre content), each Fock-canonicalized. It returns V (column j is
// rotated virtual j over the canonical ones, compact first) and the kind per column.
// Ghost centres are the atoms of zero nuclear charge, so the sidecar must carry them.
func splitVirtuals(md *mo.Data, eps []float64, nocc int, vs []int, thresh float64) (*mat.Dense, []bool, error) {
	if !md.HasDipole || len(md.AtomCharges) != len(md.AtomNames) {
		return nil, nil, fmt.Errorf("-fano-b-particle free needs the sidecar's atom_charges to tell " +
			"real atoms from ghost centres")
	}
	var real []int
	for mu := range md.NAO {
		if a := md.AOAtom[mu]; a >= 0 && md.AtomCharges[a] > 0 {
			real = append(real, mu)
		}
	}
	nv, nr := len(vs), len(real)
	// Bm = C_v^T S chi_real (nv x nreal): the real-atom AOs over these virtuals.
	Bm := mat.NewDense(nv, nr, nil)
	for i, a := range vs {
		col := nocc + a
		for x, mu := range real {
			var sum float64
			for nu := range md.NAO {
				sum += md.C.At(nu, col) * md.S.At(nu, mu)
			}
			Bm.Set(i, x, sum)
		}
	}
	var G mat.Dense
	G.Mul(Bm.T(), Bm)
	Gs := mat.NewSymDense(nr, nil)
	for i := range nr {
		for j := range nr {
			Gs.SetSym(i, j, 0.5*(G.At(i, j)+G.At(j, i)))
		}
	}
	var eg mat.EigenSym
	if !eg.Factorize(Gs, true) {
		return nil, nil, fmt.Errorf("compact virtual split: eigendecomposition failed")
	}
	w := eg.Values(nil)
	var Ug mat.Dense
	eg.VectorsTo(&Ug)
	var comp [][]float64
	for k, wk := range w {
		if wk <= thresh {
			continue
		}
		c := make([]float64, nv)
		for i := range nv {
			var sum float64
			for x := range nr {
				sum += Bm.At(i, x) * Ug.At(x, k)
			}
			c[i] = sum / math.Sqrt(wk)
		}
		comp = append(comp, c)
	}
	nc := len(comp)
	// Free = the complement: eigenvectors of I - Cc Cc^T with eigenvalue ~1.
	P := mat.NewSymDense(nv, nil)
	for i := range nv {
		for j := range nv {
			var sum float64
			for k := range nc {
				sum += comp[k][i] * comp[k][j]
			}
			v := -sum
			if i == j {
				v++
			}
			P.SetSym(i, j, v)
		}
	}
	var ep mat.EigenSym
	if !ep.Factorize(P, true) {
		return nil, nil, fmt.Errorf("free virtual complement: eigendecomposition failed")
	}
	pw := ep.Values(nil)
	var Up mat.Dense
	ep.VectorsTo(&Up)
	var free [][]float64
	for k, v := range pw {
		if v > 0.5 {
			c := make([]float64, nv)
			for i := range nv {
				c[i] = Up.At(i, k)
			}
			free = append(free, c)
		}
	}
	if nc+len(free) != nv {
		return nil, nil, fmt.Errorf("compact/free split of %d virtuals gave %d + %d", nv, nc, len(free))
	}
	// Fock-canonicalize each set (the Fock matrix is diag(eps) over canonical virtuals).
	canon := func(set [][]float64) [][]float64 {
		m := len(set)
		if m == 0 {
			return nil
		}
		F := mat.NewSymDense(m, nil)
		for x := range m {
			for y := range m {
				var sum float64
				for i, a := range vs {
					sum += set[x][i] * eps[nocc+a] * set[y][i]
				}
				F.SetSym(x, y, sum)
			}
		}
		var ef mat.EigenSym
		ef.Factorize(F, true)
		var Uf mat.Dense
		ef.VectorsTo(&Uf)
		out := make([][]float64, m)
		for k := range m {
			c := make([]float64, nv)
			for x := range m {
				for i := range nv {
					c[i] += set[x][i] * Uf.At(x, k)
				}
			}
			out[k] = c
		}
		return out
	}
	all := append(canon(comp), canon(free)...)
	V := mat.NewDense(nv, nv, nil)
	kind := make([]bool, nv)
	for j, c := range all {
		V.SetCol(j, c)
		kind[j] = j < nc
	}
	return V, kind, nil
}

// schemeBSpace is a scheme B subspace: rows of the adapted parent, each keeping its
// parent row's holes (its class), which is all fano.SelectDiscrete, ClassWeights and the
// census read from it.
type schemeBSpace struct {
	parent fano.Space
	rows   []int
	main   int
}

func (s *schemeBSpace) Size() int          { return len(s.rows) }
func (s *schemeBSpace) MainBlockSize() int { return s.main }
func (s *schemeBSpace) Holes(r int, dst []int) []int {
	return s.parent.Holes(s.rows[r], dst)
}

// adaptedFamily is the fanoFamily of scheme B: restrictions of T^T M T.
type adaptedFamily struct {
	parent fano.Space
	full   *fano.AdaptedOperator
}

func (f adaptedFamily) restrict(rows []int) (fano.Space, error) {
	main := 0
	for _, r := range rows {
		if r < f.parent.MainBlockSize() {
			main++
		}
	}
	return &schemeBSpace{parent: f.parent, rows: rows, main: main}, nil
}

func (f adaptedFamily) matrix(_ *chooser, _ string, sp fano.Space) (fanoMatrix, error) {
	if sp == f.parent {
		return f.full, nil
	}
	s, ok := sp.(*schemeBSpace)
	if !ok {
		return nil, fmt.Errorf("fano: scheme B family given a %T", sp)
	}
	return fano.NewSubOperator(f.full, f.full.Backend(), s.rows, s.main, f.full.DiagonalHost()), nil
}

// buildSchemeB constructs the adapted basis, classifies it and returns the scheme B
// partition and family. base is the scheme A partition of the hole rule, which decides
// the 3h2p rows. pmx is the parent ADC matrix (the caller releases it).
func buildSchemeB(cfg fanoConfig, ssp *sip.Space, f sipFamily, pmx fanoMatrix, base *fano.Partition,
	eps []float64, nocc int, orbSym []int) (*fano.Partition, adaptedFamily, schemeBStats, error) {

	var st schemeBStats
	if cfg.sip.moPath == "" {
		return nil, adaptedFamily{}, st, fmt.Errorf("-fano-scheme b needs -mo: the adapted states are " +
			"classified by where their two holes sit")
	}
	md, err := mo.ReadCanonical(cfg.sip.moPath)
	if err != nil {
		return nil, adaptedFamily{}, st, err
	}
	return buildSchemeBWith(md, cfg, ssp, f, pmx, base, eps, nocc, orbSym)
}

// buildSchemeBWith is buildSchemeB over an already loaded sidecar.
func buildSchemeBWith(md *mo.Data, cfg fanoConfig, ssp *sip.Space, f sipFamily, pmx fanoMatrix,
	base *fano.Partition, eps []float64, nocc int, orbSym []int) (*fano.Partition, adaptedFamily, schemeBStats, error) {

	var st schemeBStats
	names, siteOf, err := columnSites(md, cfg.sites)
	if err != nil {
		return nil, adaptedFamily{}, st, err
	}
	L, owner, lmin, err := siteLocalize(md, nocc, siteOf, len(names))
	if err != nil {
		return nil, adaptedFamily{}, st, err
	}
	r, err := newOrbitalResolver(eps[:nocc], md, cfg.sites)
	if err != nil {
		return nil, adaptedFamily{}, st, err
	}
	innerList, err := r.list(cfg.bInner)
	if err != nil {
		return nil, adaptedFamily{}, st, fmt.Errorf("-fano-b-inner: %w", err)
	}
	inner := make([]bool, nocc)
	for _, o := range innerList {
		inner[o] = true
	}
	st.Sites, st.InnerOrbs, st.LocalizeMin = names, innerList, lmin
	fmt.Fprintf(os.Stderr, "adcgo: fano: scheme B: %d occupied orbitals localized onto sites %v "+
		"(smallest site projection %.4f); inner valence %v\n", nocc, names, lmin, innerList)

	n := ssp.Size()
	class := make([]fano.Class, n)
	for row := range n {
		switch {
		case row < ssp.MainBlockSize():
			class[row] = fano.ClassQ
		case base.QIndex(row) >= 0:
			class[row] = fano.ClassQ
		case base.PIndex(row) >= 0:
			class[row] = fano.ClassP
		default:
			class[row] = fano.ClassX
		}
	}

	// Virtual rotation per irrep: identity, or compact | free (-fano-b-particle free).
	norb := ssp.Norb
	irrepOf := func(orb int) int {
		if orbSym == nil {
			return 0
		}
		return orbSym[orb] - 1
	}
	vlist := map[int][]int{} // irrep -> relative virtual indices, ascending (sip's virBySym order)
	for a := range norb - nocc {
		g := irrepOf(nocc + a)
		vlist[g] = append(vlist[g], a)
	}
	irreps := make([]int, 0, len(vlist))
	for g := range vlist {
		irreps = append(irreps, g)
	}
	slices.Sort(irreps)
	V := map[int]*mat.Dense{}
	compact := map[int][]bool{}
	for _, g := range irreps {
		vs := vlist[g]
		if cfg.bParticle == "free" {
			Vg, cg, err := splitVirtuals(md, eps, nocc, vs, cfg.bCompactThresh)
			if err != nil {
				return nil, adaptedFamily{}, st, err
			}
			V[g], compact[g] = Vg, cg
			for _, c := range cg {
				if c {
					st.CompactVirtuals++
				}
			}
		} else {
			Vg := mat.NewDense(len(vs), len(vs), nil)
			for k := range vs {
				Vg.Set(k, k, 1)
			}
			V[g], compact[g] = Vg, make([]bool, len(vs))
		}
	}
	if cfg.bParticle == "free" {
		fmt.Fprintf(os.Stderr, "adcgo: fano: scheme B: %d of %d virtuals compact (real-atom AO span), "+
			"the rest free; a two-site adapted state is P only with a free particle\n",
			st.CompactVirtuals, norb-nocc)
	}

	// Hole groups: the 2h1p rows sharing (k, l, spin function), one row per particle of
	// the group's particle irrep, in that irrep's virtual order.
	type holeGroup struct {
		rows []int
		g    int
	}
	var groups []holeGroup
	key := map[[3]int]int{}
	for row := ssp.BeginSat; row < len(ssp.Configs); row++ {
		c := ssp.Configs[row]
		k := [3]int{c.Occ[0], c.Occ[1], c.Typ}
		gi, ok := key[k]
		if !ok {
			gi = len(groups)
			key[k] = gi
			groups = append(groups, holeGroup{g: irrepOf(nocc + c.Vir)})
		}
		groups[gi].rows = append(groups[gi].rows, row)
	}
	for _, hg := range groups {
		if len(hg.rows) != len(vlist[hg.g]) {
			return nil, adaptedFamily{}, st, fmt.Errorf("fano: scheme B: a hole group has %d particles, "+
				"its irrep has %d virtuals", len(hg.rows), len(vlist[hg.g]))
		}
	}

	// The dense 2h1p/2h1p block of the parent, exactly (a restriction, scheme A's argument).
	sat := make([]int, 0, len(ssp.Configs)-ssp.BeginSat)
	for row := ssp.BeginSat; row < len(ssp.Configs); row++ {
		sat = append(sat, row)
	}
	bm := sip.New(ssp.Restrict(sat), f.ints, eps, f.cfg.order, backend.Gonum{})
	bm.SetVariant(f.v)
	if f.cfg.sig != nil {
		bm.SetStaticSelfEnergy(f.cfg.sig)
	}
	M2 := bm.BuildMatrix()
	bm.Release()
	off := ssp.BeginSat

	// Particle blocks: for rotated particle j of irrep g, A_j[G][G'] = (V^T B_GG' V)[j][j]
	// with B_GG' the 2h1p block between hole groups G and G'. With the identity rotation
	// this is exactly the sub-block of M over the rows with particle j.
	var outerBlocks, blocks []fano.AdaptedBlock
	for _, g := range irreps {
		vs := vlist[g]
		var gs []int
		for gi, hg := range groups {
			if hg.g == g {
				gs = append(gs, gi)
			}
		}
		if len(gs) == 0 {
			continue
		}
		nv, ng := len(vs), len(gs)
		Vg := V[g]
		A := make([]*mat.SymDense, nv)
		for j := range nv {
			A[j] = mat.NewSymDense(ng, nil)
		}
		B := mat.NewDense(nv, nv, nil)
		var t1, VtBV mat.Dense
		for x, g1 := range gs {
			for y := x; y < ng; y++ {
				g2 := gs[y]
				for i, r1 := range groups[g1].rows {
					for k, r2 := range groups[g2].rows {
						B.Set(i, k, M2.At(r1-off, r2-off))
					}
				}
				t1.Mul(B, Vg)
				VtBV.Mul(Vg.T(), &t1)
				for j := range nv {
					A[j].SetSym(x, y, VtBV.At(j, j))
				}
			}
		}
		if cfg.bParticle == "free" {
			Ug := backend.NewMat(nv, nv)
			for a := range nv {
				for b := range nv {
					Ug.Set(a, b, Vg.At(a, b))
				}
			}
			for _, gi := range gs {
				outerBlocks = append(outerBlocks, fano.AdaptedBlock{Rows: groups[gi].rows, U: Ug,
					Values: make([]float64, nv)})
			}
		}
		for j := range nv {
			rows := make([]int, ng)
			Am := backend.NewMat(ng, ng)
			for x, gi := range gs {
				rows[x] = groups[gi].rows[j]
				for y := range ng {
					Am.Set(x, y, A[j].At(x, y))
				}
			}
			vals, U := backend.Gonum{}.SymEig(Am)
			blocks = append(blocks, fano.AdaptedBlock{Rows: rows, U: U, Values: vals})

			dets := make([][]sip.DetTerm, ng)
			for x, row := range rows {
				if dets[x], err = ssp.DetExpansion(row); err != nil {
					return nil, adaptedFamily{}, st, err
				}
			}
			u := make([]float64, ng)
			for q := range ng {
				for x := range ng {
					u[x] = U.At(x, q)
				}
				iw, one := twoHoleCharacter(u, dets, nocc, L, owner, inner)
				pos := rows[q]
				switch {
				case iw >= cfg.bCut:
					class[pos] = fano.ClassQ
					st.AdaptedQ++
				case one >= cfg.bCut:
					if cfg.bOneSite == "q" {
						class[pos] = fano.ClassQ
						st.AdaptedQ++
					} else {
						class[pos] = fano.ClassX
						st.AdaptedX++
					}
				case cfg.bParticle == "free" && compact[g][j]:
					class[pos] = fano.ClassX
					st.AdaptedXBound++
				default:
					class[pos] = fano.ClassP
					st.AdaptedP++
				}
			}
		}
	}
	slices.SortFunc(blocks, func(a, b fano.AdaptedBlock) int { return a.Rows[0] - b.Rows[0] })
	st.Blocks = len(blocks)
	ad, err := fano.NewAdapted(n, blocks)
	if err != nil {
		return nil, adaptedFamily{}, st, err
	}
	if len(outerBlocks) > 0 {
		outer, err := fano.NewAdapted(n, outerBlocks)
		if err != nil {
			return nil, adaptedFamily{}, st, err
		}
		if ad, err = fano.Compose(outer, ad); err != nil {
			return nil, adaptedFamily{}, st, err
		}
	}
	be := pmx.Backend()
	dv := pmx.Diagonal(be)
	pdiag := be.Download(dv)
	be.Free(dv)
	full, err := fano.NewAdaptedOperator(pmx, be, ad, pdiag)
	if err != nil {
		return nil, adaptedFamily{}, st, err
	}
	holes := func(row int) int { return len(ssp.Holes(row, nil)) }
	desc := fmt.Sprintf("scheme B: %d particle blocks (particle %s); adapted 2h1p states Q %d (inner >= %.2f), "+
		"P %d (two-site%s), %s %d (one-site >= %.2f), EXCLUDED %d (two-site, bound particle); 1h in Q; "+
		"3h2p by the hole rule [%s]",
		st.Blocks, cfg.bParticle, st.AdaptedQ, cfg.bCut, st.AdaptedP,
		map[string]string{"free": ", free particle", "any": ""}[cfg.bParticle],
		map[string]string{"x": "EXCLUDED", "q": "Q"}[cfg.bOneSite], st.AdaptedX, cfg.bCut, st.AdaptedXBound,
		base.Selector())
	part := fano.NewPartitionFromClasses(class, ssp.MainBlockSize(), holes, desc)
	return part, adaptedFamily{parent: ssp, full: full}, st, nil
}
