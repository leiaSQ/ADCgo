package sip

import "math"

// elements4.go — CVS IP-ADC(4) matrix elements (Dyson formulation), ported from
// ../ADC/adc4core/adc4_constr/{kopp1,kopp2,kopp3,kopp4,ab3,ab5}.F. See
// docs/adc4_sip_spec.md. These use the reference's Dyson spin conventions, which
// differ from the non-Dyson order-2/3 elements (elements.go) by basis phases — the
// order-4 secular matrix is therefore self-contained, never mixing the two.
//
// Convention bridge to the reference: a 2h1p config carries holes K,L and particle
// a; the reference names the particle "J" and the external 1h index "P". The
// physicist integral <P a|K L> is e.v(P,a,K,L) = ints.Eri(P,K,a,L). Direct vs
// exchange is the swap of the last two orbital arguments (indvx.F): A1=<Pa|KL>,
// A2=<Pa|LK>.
//
// Verification (matched-integral tapes, both sectors): every off-diagonal block is
// bit-exact against theADCcode's own matrix tape (TestADC4MatchedGate +
// TestADC4MatchedGateA1) — 2h1p/2h1p (c22elem4/WERT1), 2h1p<->3h2p (wert2elem4/WERT2),
// 1h<->2h1p (kopp1+kopp2+kopp3, KOPP1/2/3), and 1h<->3h2p (kopp4/KOPP4). The 1h/1h
// diagonal is −ε_core − Σ with the external static self-energy Σ supplied via
// Matrix.SetStaticSelfEnergy (see kopp3.go, matvec4.go).

// kopp1 is the 2nd-order 1h<->2h1p coupling <P|Ĥ|K L a,S> (kopp1.F). p is the
// external 1h hole (absolute occ index); cfg is the 2h1p satellite. The reference
// spin table SPIN(2,2) = [[1/√2, √(3/2)],[1/√2, −√(3/2)]] (column = spin function)
// contracts the direct (A1) and exchange (A2) integrals; FAKTOR = 1/√2 folds the
// K==L single spin function.
func (e *elements) kopp1(p int, cfg Config) float64 {
	k, l := cfg.Occ[0], cfg.Occ[1]
	a := e.nocc + cfg.Vir
	a1 := e.v(p, a, k, l) // <P a|K L> direct
	a2 := e.v(p, a, l, k) // <P a|L K> exchange
	if k == l {
		// MAXS=1, FKL=1/√2, SPIN(1,1)=SPIN(2,1)=1/√2: FKL*(A1+A2)/√2 = 0.5*(A1+A2);
		// A1==A2 here, so this is just <P a|K K>.
		return sqrt1_2 * sqrt1_2 * (a1 + a2)
	}
	if cfg.Typ == 0 { // spin function I: SPIN(1,1)=SPIN(2,1)=1/√2
		return sqrt1_2 * (a1 + a2)
	}
	// spin function II: SPIN(1,2)=+√(3/2), SPIN(2,2)=−√(3/2) (Dyson sign).
	return sqrt3_2 * (a1 - a2)
}

// kopp2 is the 3rd-order 1h<->2h1p coupling <P|Ĥ|K L a,S> (kopp2.F, Eq. A.3), added
// on top of kopp1. The reference has a single active contribution (sum over an
// intermediate non-core hole JJ and particle LL). p is the external core hole; cfg
// the 2h1p satellite (Occ[0]=K core, Occ[1]=L valence, Vir=a=particle J_ref,
// Typ=spin function). SPIN2(4,MS) inline table; FAKTOR/SUMFAK fold the (unused in
// CVS) K==L single. ASIG(IAB=1)=+1.
func (e *elements) kopp2(p int, cfg Config) float64 {
	so, ep := e.so, e.eps
	jp := e.nocc + cfg.Vir // particle J_ref
	k, l := cfg.Occ[0], cfg.Occ[1]
	maxs := 2
	if k == l {
		maxs = 1
	}
	if cfg.Typ >= maxs {
		return 0
	}
	fkl, sumkl := 1.0, 1.0
	if maxs == 1 {
		fkl, sumkl = sqrt1_2, 2.0
	}
	// SPIN2 column for this spin function (kopp2.F:41-44).
	var s1, s2, s3, s4 float64
	if cfg.Typ == 0 { // MS=1
		s1, s2, s3, s4 = sqrt1_2, sqrt1_2, -math.Sqrt2, sqrt1_2
	} else { // MS=2
		sqrt6 := math.Sqrt(6)
		s1, s2, s3, s4 = sqrt3_2, -sqrt3_2, -sqrt6, sqrt3_2
	}
	ifpk := so(p) ^ so(k)
	ifjl := so(jp) ^ so(l)
	var total float64
	for jj := range e.nocc {
		if e.sp.isCore(jj) {
			continue // IGRENZ = MAXCOR+1: intermediate hole excludes the core
		}
		for ll := e.nocc; ll < e.norb; ll++ {
			jjll := so(jj) ^ so(ll)
			if jjll != ifpk || jjll != ifjl {
				continue
			}
			e1 := ep[jp] + ep[ll] - ep[jj] - ep[l]
			a1 := e.v(jp, ll, jj, l)
			a2 := e.v(jp, ll, l, jj)
			a3 := e.v(p, jj, k, ll)
			a4 := e.v(p, jj, ll, k)
			sum := a1*a3*fkl*s1 + a1*a4*fkl*s2 + a2*a3*fkl*s3 + a2*a4*fkl*s4
			total += sumkl * sum / e1
		}
	}
	return total
}

// c22elem4 is the 2h1p x 2h1p block element between two CVS 2h1p configs, ported
// from WERT1_core (wert1.F, Eqs. A.7-A.9) — the zeroth-order K-matrix diagonal
// ε_J−ε_K−ε_L plus the 3rd-order interaction C(JKL,J'K'L'). row=(J,K,L),
// col=(J',K',L') with J,J' particles and K,K' core / L,L' valence holes; the value
// returned is the (row.Typ, col.Typ) spin-block entry (Typ 0 = intermediate spin 0,
// Typ 1 = spin 1). INDVZ == INDVX_core, so integrals are e.v. Hermitian. Includes the
// 4th-order exchange terms SUM1/SUM3/SUM4 (B1/B6/B7); bit-exact vs the reference tape
// (TestADC4MatchedGate, 2h1p/2h1p block maxdiff ~3e-15).
func (e *elements) c22elem4(row, col Config) float64 {
	jp, k, l := e.nocc+row.Vir, row.Occ[0], row.Occ[1]
	jjp, kk, ll := e.nocc+col.Vir, col.Occ[0], col.Occ[1]
	ep := e.eps
	var w [2][2]float64 // [spinRow][spinCol]

	if jp == jjp && k == kk && l == ll { // K-matrix (zeroth order): diagonal
		d := ep[jp] - ep[k] - ep[l]
		w[0][0], w[1][1] = d, d
	}
	var a1, a2, a3, a4, a5, a6 float64
	if jp == jjp {
		a1, a2 = e.v(k, l, kk, ll), e.v(k, l, ll, kk)
	}
	if k == kk {
		a3, a4 = e.v(jp, ll, jjp, l), e.v(jp, ll, l, jjp)
	}
	if l == ll {
		a5, a6 = e.v(jp, kk, jjp, k), e.v(jp, kk, k, jjp)
	}
	w[0][0] += (a1 + a2) - (a3 + a5) + 0.5*(a4+a6)
	w[1][1] += (a1 - a2) - (a3 + a5) + 1.5*(a4+a6)
	w[0][1] += sqrt3_4 * (a4 - a6)
	w[1][0] += sqrt3_4 * (a4 - a6)

	// 4th order (wert1.F:125-183) — only when K==K' (kk); SUM1 always, SUM3 if L==L',
	// SUM4 if J==J'. See sum1/sum3/sum4_4.
	if k == kk {
		b1 := e.sum1_4(jp, jjp, l, ll)
		var b6, b7 float64
		if l == ll {
			b6 = e.sum3_4(jp, jjp)
		}
		if jp == jjp {
			b7 = e.sum4_4(l, ll)
		}
		w[0][0] += -(b1[0] - b7) - b6
		w[1][1] += -(b1[1] - b7) - b6
		w[0][1] += -sqrt3_4 * b1[2]
		w[1][0] += -sqrt3_4 * b1[2]
	}
	return w[row.Typ][col.Typ]
}

// wert1spin is the SUM1 spin table SPIN(4,3) (ndriver.F:237), column-major -> rows
// here index the intermediate-spin block (0:W11, 1:W22, 2:W12), columns the four
// direct/exchange integral products (A1A3,A1A4,A2A3,A2A4).
var wert1spin = [3][4]float64{
	{0.5, -0.25, -0.25, 0.5},
	{0.5, -0.75, -0.75, 1.5},
	{0.0, -0.5, -0.5, 1.0},
}

// sum1_4 ports SUM1_core (wert1.F 4th-order Z(1)); called when K==K'. j,jj are the
// 2h1p/2h1p particles, l,ll the valence holes (absolute). Sums an intermediate
// non-core hole J" (jjj) and particle K" (kkk); returns the three spin shifts.
func (e *elements) sum1_4(j, jj, l, ll int) (b [3]float64) {
	so, ep := e.so, e.eps
	fjl1 := so(j) ^ so(l)
	fjl2 := so(jj) ^ so(ll)
	for jjj := range e.nocc {
		if e.sp.isCore(jjj) {
			continue
		}
		for kkk := e.nocc; kkk < e.norb; kkk++ {
			s := so(jjj) ^ so(kkk)
			if s != fjl1 || s != fjl2 {
				continue
			}
			a1 := e.v(j, kkk, jjj, l)
			a2 := e.v(j, kkk, l, jjj)
			a3 := e.v(jjj, ll, jj, kkk)
			a4 := e.v(jjj, ll, kkk, jj)
			e1 := ep[j] + ep[jj] - ep[l] - ep[ll] + 2*(ep[kkk]-ep[jjj])
			e1 /= ep[j] + ep[kkk] - ep[l] - ep[jjj]
			e1 /= ep[jj] + ep[kkk] - ep[ll] - ep[jjj]
			add := [4]float64{a1 * a3 * e1, a1 * a4 * e1, a2 * a3 * e1, a2 * a4 * e1}
			for i := range 3 {
				b[i] += add[0]*wert1spin[i][0] + add[1]*wert1spin[i][1] +
					add[2]*wert1spin[i][2] + add[3]*wert1spin[i][3]
			}
		}
	}
	return
}

// sum3_4 ports SUM3_core (wert1.F 4th-order Z(3)); called when K==K' and L==L'. j,jj
// are the particles. Sums a particle J" (jjj) and a non-core occupied pair K"<=L"
// (kkk,lll, half-weight on the diagonal).
func (e *elements) sum3_4(j, jj int) (b float64) {
	so, ep := e.so, e.eps
	for jjjSym := range e.sp.nSym {
		for _, jp := range e.sp.virBySym(jjjSym) {
			jjj := e.nocc + jp
			ifj1 := so(j) ^ jjjSym
			ifj2 := so(jj) ^ jjjSym
			for lllSym := range e.sp.nSym {
				lllList := e.sp.valOccBySym(lllSym)
				for kkkSym := 0; kkkSym <= lllSym; kkkSym++ {
					if kkkSym^lllSym != ifj1 || kkkSym^lllSym != ifj2 {
						continue
					}
					kkkList := e.sp.valOccBySym(kkkSym)
					for li, lll := range lllList {
						end := len(kkkList)
						if kkkSym == lllSym {
							end = li + 1
						}
						for _, kkk := range kkkList[:end] {
							f1 := 1.0
							if kkk == lll {
								f1 = 0.5
							}
							a1 := e.v(j, jjj, kkk, lll)
							a2 := e.v(j, jjj, lll, kkk)
							a3 := e.v(kkk, lll, jjj, jj)
							a4 := e.v(kkk, lll, jj, jjj)
							e1 := 2*(ep[jjj]-ep[kkk]-ep[lll]) + ep[j] + ep[jj]
							e1 /= ep[j] + ep[jjj] - ep[kkk] - ep[lll]
							e1 /= ep[jj] + ep[jjj] - ep[kkk] - ep[lll]
							e1 *= f1
							b += a1*a3*e1*0.5 - a1*a4*e1 - a2*a3*e1 + a2*a4*e1*0.5
						}
					}
				}
			}
		}
	}
	return
}

// sum4_4 ports SUM4_core (wert1.F 4th-order Z(2)); called when K==K' and J==J'. l,ll
// are the valence holes. Sums a non-core occupied J" (jjj) and a virtual pair
// K"<=L" (kkk,lll, half-weight on the diagonal).
func (e *elements) sum4_4(l, ll int) (b float64) {
	so, ep := e.so, e.eps
	for jjj := range e.nocc {
		if e.sp.isCore(jjj) {
			continue
		}
		ifjl1 := so(l) ^ so(jjj)
		ifjl2 := so(ll) ^ so(jjj)
		for lllSym := range e.sp.nSym {
			lllList := e.sp.virBySym(lllSym)
			for kkkSym := 0; kkkSym <= lllSym; kkkSym++ {
				if kkkSym^lllSym != ifjl1 || kkkSym^lllSym != ifjl2 {
					continue
				}
				kkkList := e.sp.virBySym(kkkSym)
				for li, lp := range lllList {
					lll := e.nocc + lp
					end := len(kkkList)
					if kkkSym == lllSym {
						end = li + 1
					}
					for _, kp := range kkkList[:end] {
						kkk := e.nocc + kp
						f1 := 1.0
						if kkk == lll {
							f1 = 0.5
						}
						a1 := e.v(kkk, lll, jjj, l)
						a2 := e.v(kkk, lll, l, jjj)
						a3 := e.v(jjj, ll, kkk, lll)
						a4 := e.v(jjj, ll, lll, kkk)
						e1 := 2*(ep[kkk]+ep[lll]-ep[jjj]) - ep[l] - ep[ll]
						e1 /= ep[kkk] + ep[lll] - ep[jjj] - ep[l]
						e1 /= ep[kkk] + ep[lll] - ep[jjj] - ep[ll]
						e1 *= f1
						b += a1*a3*e1 - a1*a4*e1*0.5 - a2*a3*e1*0.5 + a2*a4*e1
					}
				}
			}
		}
	}
	return
}

// wert2elem4 is the 2h1p x 3h2p effective coupling (1st order) between a 2h1p row
// and a 3h2p column, ported from WERT2_core (wert2.F, Eqs. A.10-A.11 of Giuliana's
// thesis). Reference naming: the 2h1p row carries particle J=nocc+row.Vir, core hole
// K=row.Occ[0], valence hole L=row.Occ[1]; the 3h2p column carries particles II,JJ
// and holes KK(core),LL,MM = col.I,col.J,col.Core,col.L,col.M (state.F role map, all
// direct). INDVZ(N,a,b,c,d)->e.v(a,b,c,d). The reference builds -C, so the returned
// value is the final (positive-eigenvalue) matrix element; NS=0 (2h1p), NSS=ns3(col).
// The conditional branches gate which VINT integrals are active (the diagram's
// Kronecker deltas); inactive entries stay zero and drop out of the spin contraction.
func (e *elements) wert2elem4(row Config, col Config3) float64 {
	jp := e.nocc + row.Vir // J  (particle)
	kc := row.Occ[0]       // K  (core hole)
	lv := row.Occ[1]       // L  (valence hole)
	ii := e.nocc + col.I   // II (particle)
	jj := e.nocc + col.J   // JJ (particle)
	kk := col.Core         // KK (core hole)
	ll := col.L            // LL (valence hole)
	mm := col.M            // MM (valence hole)
	v := e.v

	var vint [31]float64
	// wert2.F:72-103 — J==II branch.
	if jp == ii {
		if kc == kk { // :73
			vint[1] = v(ll, mm, lv, jj)
			vint[2] = v(ll, mm, jj, lv)
		}
		if lv == ll { // :81
			vint[9] = v(kk, mm, kc, jj)
			vint[10] = v(kk, mm, jj, kc)
			if ll == mm { // :89
				vint[11] = vint[9]
				vint[12] = vint[10]
			}
		} else if lv == mm { // :96
			vint[11] = v(kk, ll, kc, jj)
			vint[12] = v(kk, ll, jj, kc)
		}
	}
	// wert2.F:105-144 — J==JJ branch.
	if jp == jj {
		if ii == jj { // :106 copy VINT(1..12)->VINT(13..24)
			for n := 1; n <= 12; n++ {
				vint[12+n] = vint[n]
			}
		} else {
			if kc == kk { // :114
				vint[13] = v(ll, mm, lv, ii)
				vint[14] = v(ll, mm, ii, lv)
			}
			if lv == ll { // :122
				vint[21] = v(kk, mm, kc, ii)
				vint[22] = v(kk, mm, ii, kc)
				if ll == mm { // :130
					vint[23] = vint[21]
					vint[24] = vint[22]
				}
			} else if lv == mm { // :137
				vint[23] = v(kk, ll, kc, ii)
				vint[24] = v(kk, ll, ii, kc)
			}
		}
	}
	// wert2.F:146-161 — independent coincidence terms.
	if kc == kk && lv == ll { // :146
		vint[25] = v(mm, jp, ii, jj)
		vint[26] = v(mm, jp, jj, ii)
	}
	if kc == kk && lv == mm { // :154
		vint[27] = v(ll, jp, ii, jj)
		vint[28] = v(ll, jp, jj, ii)
	}

	// wert2.F:163-171 — spin contraction. NS=0, MS=row.Typ+1, NSS=ns3(col), MSS=Spin.
	col2 := ns3(col) + col.Spin - 1
	var add float64
	sp := &coeff1[row.Typ]
	for n := 1; n <= 30; n++ {
		if vint[n] != 0 {
			add += vint[n] * sp[col2][n-1]
		}
	}
	return -add
}

// ns3 is the KOPP4/AB5 spin-table row offset for a 3h2p config (kopp4.F:116-118):
// the coincidences (L==M, I==J) plus the multiplicity select which of the 13 rows
// of the coeff tables applies. Row = Spin + ns3 (1-based).
func ns3(cfg Config3) int {
	maxs := maxS3(cfg.L == cfg.M, cfg.I == cfg.J)
	switch {
	case cfg.L == cfg.M && maxs == 2:
		return 7
	case cfg.I == cfg.J && maxs == 2:
		return 9
	case cfg.L == cfg.M && maxs == 1:
		return 12
	default:
		return 0
	}
}

// wert3elem is the 3h2p↔3h2p secular block element −K(IJKLM,I'J'K'L'M') −
// C(IJKLM,I'J'K'L'M') for CVS IP-ADC(4) (Giuliana's thesis Eq. A.12–A.13), a verbatim
// port of ../ADC/adc4core/adc4_constr/wert3.F. row is the bra (I,J,K,L,M), col the ket
// (I',J',K',L',M'). theADCcode evaluates this only on the diagonal (bra==ket, selec.F:120),
// where it is the 3h2p effective diagonal EIGAB = the 0th-order orbital-energy sum plus the
// 5th-order CI diagonal correction. Returned as the [5][5] intermediate-spin block; the
// caller (sat3Diag) takes W[Spin-1][Spin-1].
//
// INDVZ(a,b,c,d) → e.v(a,b,c,d); exchange = swap the last two args. The reference tracks
// which VINT slots are active via IQUAL; here the inactive slots stay zero, so the spin
// contraction simply sums over all 52 (0·coeff = 0), dropping the IQUAL bookkeeping.
func (e *elements) wert3elem(row, col Config3) [5][5]float64 {
	i, j := e.nocc+row.I, e.nocc+row.J // particles I,J (absolute)
	k, l, m := row.Core, row.L, row.M  // core hole K, valence holes L,M
	ii, jj := e.nocc+col.I, e.nocc+col.J
	kk, ll, mm := col.Core, col.L, col.M
	maxs := maxS3(row.L == row.M, row.I == row.J)
	maxss := maxS3(col.L == col.M, col.I == col.J)
	ns, nss := ns3(row), ns3(col)
	v, ep := e.v, e.eps

	var w [5][5]float64
	var vint [53]float64 // 1-indexed (slot 0 unused), mirrors wert3.F VINT(1..52)

	// 0th-order K matrix: diagonal in the configuration indices (wert3.F:79-85).
	if i == ii && j == jj && k == kk && l == ll && m == mm {
		d := ep[i] + ep[j] - ep[k] - ep[l] - ep[m]
		for s := 0; s < 5; s++ {
			w[s][s] = d
		}
	}

	// 5th-order CI matrix (wert3.F:95-424). Near-verbatim: labels Lnnn mirror the
	// Fortran statement numbers; goto follows the reference's branch cascade exactly.
	if k != kk || l != ll || m != mm {
		goto L101
	}
	vint[1] = v(i, j, ii, jj)
	vint[2] = v(i, j, jj, ii)
L101:
	if i != ii || j != jj {
		goto L200
	}
	if k != kk {
		goto L105
	}
	vint[3] = v(ll, mm, l, m)
	vint[4] = v(ll, mm, m, l)
L105:
	if l != ll {
		goto L106
	}
	vint[5] = v(kk, mm, k, m)
	vint[6] = v(kk, mm, m, k)
L106:
	if m != ll {
		goto L108
	}
	vint[7] = v(kk, mm, k, l)
	vint[8] = v(kk, mm, l, k)
L108:
	if l != mm {
		goto L109
	}
	vint[9] = v(kk, ll, k, m)
	vint[10] = v(kk, ll, m, k)
L109:
	if m != mm {
		goto L200
	}
	vint[11] = v(kk, ll, k, l)
	vint[12] = v(kk, ll, l, k)
L200:
	if i != ii {
		goto L300
	}
	if k != kk || l != ll {
		goto L201
	}
	vint[13] = v(j, mm, jj, m)
	vint[14] = v(j, mm, m, jj)
L201:
	if k != kk || m != ll {
		goto L205
	}
	vint[15] = v(j, mm, jj, l)
	vint[16] = v(j, mm, l, jj)
L205:
	if l != ll || m != mm {
		goto L206
	}
	vint[17] = v(j, kk, jj, k)
	vint[18] = v(j, kk, k, jj)
L206:
	if k != kk || l != mm {
		goto L207
	}
	vint[19] = v(j, ll, jj, m)
	vint[20] = v(j, ll, m, jj)
L207:
	if k != kk || m != mm {
		goto L209
	}
	vint[21] = v(j, ll, jj, l)
	vint[22] = v(j, ll, l, jj)
L209:
	if i == j && ii == jj {
		goto L301
	}
	if i == j {
		goto L302
	}
	if ii == jj {
		goto L303
	}
	goto L700
L301:
	for ms := 1; ms <= 10; ms++ {
		vint[22+ms] = vint[12+ms]
		vint[32+ms] = vint[12+ms]
		vint[42+ms] = vint[12+ms]
	}
	goto L1000
L302:
	for ms := 1; ms <= 10; ms++ {
		vint[22+ms] = vint[12+ms]
	}
	goto L1000
L303:
	for ms := 1; ms <= 10; ms++ {
		vint[32+ms] = vint[12+ms]
	}
	goto L1000
L300:
	if j != ii {
		goto L500
	}
	if k != kk || l != ll {
		goto L401
	}
	vint[23] = v(i, mm, jj, m)
	vint[24] = v(i, mm, m, jj)
L401:
	if k != kk || m != ll {
		goto L405
	}
	vint[25] = v(i, mm, jj, l)
	vint[26] = v(i, mm, l, jj)
L405:
	if l != ll || m != mm {
		goto L406
	}
	vint[27] = v(i, kk, jj, k)
	vint[28] = v(i, kk, k, jj)
L406:
	if k != kk || l != mm {
		goto L407
	}
	vint[29] = v(i, ll, jj, m)
	vint[30] = v(i, ll, m, jj)
L407:
	if k != kk || m != mm {
		goto L409
	}
	vint[31] = v(i, ll, jj, l)
	vint[32] = v(i, ll, l, jj)
L409:
	if ii != jj {
		goto L1000
	}
	for ms := 1; ms <= 10; ms++ {
		vint[42+ms] = vint[22+ms]
	}
	goto L1000
L500:
	if i != jj {
		goto L700
	}
	if k != kk || l != ll {
		goto L601
	}
	vint[33] = v(j, mm, ii, m)
	vint[34] = v(j, mm, m, ii)
L601:
	if k != kk || m != ll {
		goto L605
	}
	vint[35] = v(j, mm, ii, l)
	vint[36] = v(j, mm, l, ii)
L605:
	if l != ll || m != mm {
		goto L606
	}
	vint[37] = v(j, kk, ii, k)
	vint[38] = v(j, kk, k, ii)
L606:
	if k != kk || l != mm {
		goto L607
	}
	vint[39] = v(j, ll, ii, m)
	vint[40] = v(j, ll, m, ii)
L607:
	if k != kk || m != mm {
		goto L609
	}
	vint[41] = v(j, ll, ii, l)
	vint[42] = v(j, ll, l, ii)
L609:
	if i != j {
		goto L1000
	}
	for ms := 1; ms <= 10; ms++ {
		vint[42+ms] = vint[32+ms]
	}
	goto L1000
L700:
	if j != jj {
		goto L1000
	}
	if k != kk || l != ll {
		goto L801
	}
	vint[43] = v(i, mm, ii, m)
	vint[44] = v(i, mm, m, ii)
L801:
	if k != kk || m != ll {
		goto L805
	}
	vint[45] = v(i, mm, ii, l)
	vint[46] = v(i, mm, l, ii)
L805:
	if l != ll || m != mm {
		goto L806
	}
	vint[47] = v(i, kk, ii, k)
	vint[48] = v(i, kk, k, ii)
L806:
	if k != kk || l != mm {
		goto L807
	}
	vint[49] = v(i, ll, ii, m)
	vint[50] = v(i, ll, m, ii)
L807:
	if k != kk || m != mm {
		goto L1000
	}
	vint[51] = v(i, ll, ii, l)
	vint[52] = v(i, ll, l, ii)
L1000:
	// Spin contraction W(MS,MSS) += Σ_n VINT(n)·SPIN(NS+MS, NSS+MSS, n) (wert3.F:415-423).
	for ms := 0; ms < maxs; ms++ {
		for mss := 0; mss < maxss; mss++ {
			var add float64
			for n := 1; n <= 52; n++ {
				if vint[n] != 0 {
					add += vint[n] * coeff2[ns+ms][nss+mss][n-1]
				}
			}
			w[ms][mss] += add
		}
	}
	return w
}

// kopp4 is the 4th-order 1h<->3h2p coupling <P|Ĥ|I J K L M,S> (kopp4.F, Eq. A.6).
// p is the external core hole (absolute occ index); cfg the 3h2p satellite. Two
// contributions: a sum over an intermediate particle KKK (SUM2+SUM5) and over an
// intermediate non-core hole KK (SUM1+SUM3), each a direct/exchange integral pair
// contracted with the coeff0 (/DREI/) spin table at row Spin+ns3 and the fixed
// column groups {5..8},{17..20} (KKK) and {25..28},{33..36} (KK). Direct vs
// exchange is the swap of the last two integral args (e.v). Symmetry guards mirror
// the reference (symmetry-forbidden integrals are zero).
func (e *elements) kopp4(p int, cfg Config3) float64 {
	so := e.so
	i := e.nocc + cfg.I // particle (absolute)
	j := e.nocc + cfg.J // particle (absolute)
	k, l, m := cfg.Core, cfg.L, cfg.M
	ep := e.eps
	r := cfg.Spin + ns3(cfg) - 1 // 0-based coeff row
	c0 := &coeff0

	var total float64

	// Contribution 1: sum over intermediate particle KKK (all virtuals).
	lmSym := so(l) ^ so(m)
	pkSymJ := so(j) ^ so(k) // IFJK
	pkSymI := so(i) ^ so(k) // IFIK
	for kkk := e.nocc; kkk < e.norb; kkk++ {
		var a3, a4, a9, a10 float64
		if lmSym == so(i)^so(kkk) {
			a3 = e.v(l, m, i, kkk)
			a4 = e.v(l, m, kkk, i)
		}
		if lmSym == so(j)^so(kkk) {
			a9 = e.v(l, m, j, kkk)
			a10 = e.v(l, m, kkk, j)
		}
		pkkk := so(p) ^ so(kkk)
		var a15, a16, a21, a22 float64
		if pkSymJ == pkkk {
			a15 = e.v(k, kkk, p, j)
			a16 = e.v(k, kkk, j, p)
		}
		if pkSymI == pkkk {
			a21 = e.v(k, kkk, p, i)
			a22 = e.v(k, kkk, i, p)
		}
		e2 := ep[i] + ep[kkk] - ep[l] - ep[m]
		e5 := ep[j] + ep[kkk] - ep[l] - ep[m]
		sum2 := (a3*a15*c0[r][4] + a4*a15*c0[r][5] + a3*a16*c0[r][6] + a4*a16*c0[r][7]) / e2
		sum5 := (a9*a21*c0[r][16] + a10*a21*c0[r][17] + a9*a22*c0[r][18] + a10*a22*c0[r][19]) / e5
		total += sum2 + sum5
	}

	// Contribution 2: sum over intermediate non-core hole KK.
	ijSym := so(i) ^ so(j)
	klSym := so(k) ^ so(l) // IFKL
	kmSym := so(k) ^ so(m) // IFKM
	for kk := range e.nocc {
		if e.sp.isCore(kk) {
			continue // IGREKK = MAXCOR+1: intermediate hole excludes the core
		}
		var a1, a2, a5, a6 float64
		if ijSym == so(m)^so(kk) {
			a1 = e.v(m, kk, i, j)
			a2 = e.v(m, kk, j, i)
		}
		if ijSym == so(l)^so(kk) {
			a5 = e.v(l, kk, i, j)
			a6 = e.v(l, kk, j, i)
		}
		pkk := so(p) ^ so(kk)
		var a7, a8, a11, a12 float64
		if klSym == pkk {
			a7 = e.v(k, l, p, kk)
			a8 = e.v(k, l, kk, p)
		}
		if kmSym == pkk {
			a11 = e.v(k, m, p, kk)
			a12 = e.v(k, m, kk, p)
		}
		e1 := ep[i] + ep[j] - ep[m] - ep[kk]
		e3 := ep[i] + ep[j] - ep[l] - ep[kk]
		sum1 := (a1*a7*c0[r][24] + a2*a7*c0[r][25] + a1*a8*c0[r][26] + a2*a8*c0[r][27]) / e1
		sum3 := (a5*a11*c0[r][32] + a6*a11*c0[r][33] + a5*a12*c0[r][34] + a6*a12*c0[r][35]) / e3
		total += sum1 + sum3
	}
	return total
}
