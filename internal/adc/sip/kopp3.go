package sip

// kopp3.go — the 4th-order 1h<->2h1p coupling KOPP3 (kopp3.F), ported as
// K2P2H + K1P3H + K3P1H (K4P vanishes in the core approximation). Adds to the
// kopp1+kopp2 (2nd+3rd order) coupling; validated bit-exact against the A1 reference
// tape (TestADC4MatchedGateA1). Reference naming: the 2h1p config carries particle J,
// core hole K, valence hole L; P is the external core hole. INDVZ(a,b,c,d)=e.v(a,b,c,d).
//
// The reference iterates intermediate orbitals by symmetry block (IT2SYM>=IT1SYM, …),
// which fixes the role assignment of the unordered intermediate tuples (T1<=T2<=T3) and
// the coincidence factors (½ / 1/6). We reproduce that block-major ordering exactly but
// drop the reference's symmetry *screening* IF-GOTOs: symmetry-forbidden integrals are
// already zero (e.v), and every energy denominator is a virtual−occupied gap, so never
// zero. The two returned values are the spin-function-0 and spin-function-1 sums before
// the global SPIN(1)=1/√2, SPIN(2)=√(3/2) prefactors (applied in kopp3).

// kopp3 is the total 4th-order 1h<->2h1p coupling for external core hole p and 2h1p
// config cfg (Occ[0]=K core, Occ[1]=L valence, Vir=J particle, Typ=spin function).
func (e *elements) kopp3(p int, cfg Config) float64 {
	j := e.nocc + cfg.Vir
	k, l := cfg.Occ[0], cfg.Occ[1]
	vir, occ := e.symLists()
	a2, b2 := e.k2p2h(p, j, k, l, vir, occ)
	a1, b1 := e.k1p3h(p, j, k, l, vir, occ)
	a3, b3 := e.k3p1h(p, j, k, l, vir, occ)
	if cfg.Typ == 0 {
		return sqrt1_2 * (a1 + a2 + a3)
	}
	return sqrt3_2 * (b1 + b2 + b3)
}

// symLists returns the per-irrep absolute virtual indices and non-core occupied indices
// (ascending), used to reproduce the reference's block-major intermediate ordering.
func (e *elements) symLists() (vir, occ [][]int) {
	ns := e.sp.nSym
	vir = make([][]int, ns)
	occ = make([][]int, ns)
	for s := range ns {
		for _, pos := range e.sp.virBySym(s) {
			vir[s] = append(vir[s], e.nocc+pos)
		}
		occ[s] = e.sp.valOccBySym(s)
	}
	return
}

// k2p2h ports K2P2H_core (Z(3)-Z(11)): sum over a virtual pair KKK<=LLL and a non-core
// occupied pair KK<=LL.
func (e *elements) k2p2h(p, j, k, l int, vir, occ [][]int) (accA, accB float64) {
	ns := len(vir)
	for t2 := range ns { // LLL (T2)
		for t1 := 0; t1 <= t2; t1++ { // KKK (T1) <= LLL
			for i2, LLL := range vir[t2] {
				kmax := len(vir[t1])
				if t1 == t2 {
					kmax = i2 + 1
				}
				for _, KKK := range vir[t1][:kmax] {
					ft := 1.0
					if KKK == LLL {
						ft = 0.5
					}
					for m2 := range ns { // LL (L2)
						for m1 := 0; m1 <= m2; m1++ { // KK (L1) <= LL
							for j2, LL := range occ[m2] {
								kkmax := len(occ[m1])
								if m1 == m2 {
									kkmax = j2 + 1
								}
								for _, KK := range occ[m1][:kkmax] {
									fl := 1.0
									if KK == LL {
										fl = 0.5
									}
									a, b := e.k2p2hTerm(p, j, k, l, KKK, LLL, KK, LL)
									accA += a * ft * fl
									accB += b * ft * fl
								}
							}
						}
					}
				}
			}
		}
	}
	return
}

func (e *elements) k2p2hTerm(p, J, K, L, KKK, LLL, KK, LL int) (suma, sumb float64) {
	v, ep := e.v, e.eps
	const D05, DUE = 0.5, 2.0
	A1 := v(KK, LL, KKK, LLL)
	A2 := v(KK, LL, LLL, KKK)
	A3 := v(KK, LLL, LL, KKK)
	E1 := 1 / (ep[KKK] + ep[LLL] - ep[KK] - ep[LL])
	A4 := v(KKK, LLL, KK, J)
	A5 := v(KKK, LLL, J, KK)
	A6 := v(KKK, LLL, LL, J)
	A7 := v(KKK, LLL, J, LL)
	A20 := v(KK, LL, KKK, J)
	A21 := v(KK, LL, J, KKK)
	A22 := v(KK, J, LL, KKK)
	A23 := v(KK, LL, LLL, J)
	A24 := v(KK, LL, J, LLL)
	A25 := v(KK, J, LL, LLL)
	A26 := v(KK, LL, KKK, L)
	A27 := v(KK, LL, L, KKK)
	A29 := v(KK, LL, LLL, L)
	A30 := v(KK, LL, L, LLL)
	A38 := v(KKK, LLL, KK, L)
	A39 := v(KKK, LLL, L, KK)
	A40 := v(KKK, L, LLL, KK)
	A41 := v(KKK, LLL, LL, L)
	A42 := v(KKK, LLL, L, LL)
	A43 := v(KKK, L, LLL, LL)
	A62 := v(L, LLL, KK, J)
	A63 := v(L, LLL, J, KK)
	A64 := v(L, J, LLL, KK)
	A65 := v(L, LLL, LL, J)
	A66 := v(L, LLL, J, LL)
	A67 := v(L, J, LLL, LL)
	A68 := v(L, KKK, KK, J)
	A69 := v(L, KKK, J, KK)
	A70 := v(L, J, KKK, KK)
	A71 := v(L, KKK, LL, J)
	A72 := v(L, KKK, J, LL)
	A73 := v(L, J, KKK, LL)
	E2 := 1 / (ep[J] - ep[LL])
	E3 := 1 / (ep[J] - ep[KK])
	E4 := 1 / (ep[LLL] - ep[L])
	E5 := 1 / (ep[KKK] - ep[L])
	E8 := 1 / (ep[J] + ep[KKK] - ep[KK] - ep[LL])
	E9 := 1 / (ep[J] + ep[LLL] - ep[KK] - ep[LL])
	E10 := 1 / (ep[KKK] + ep[LLL] - ep[KK] - ep[L])
	E11 := 1 / (ep[KKK] + ep[LLL] - ep[LL] - ep[L])
	E18 := 1 / (ep[J] + ep[KKK] - ep[LL] - ep[L])
	E19 := 1 / (ep[J] + ep[KKK] - ep[KK] - ep[L])
	E20 := 1 / (ep[J] + ep[LLL] - ep[LL] - ep[L])
	E21 := 1 / (ep[J] + ep[LLL] - ep[KK] - ep[L])
	E25 := 1 / (ep[KKK] + ep[LLL] - ep[KK] - ep[L])
	E26 := 1 / (ep[KKK] + ep[LLL] - ep[LL] - ep[L])
	A80 := v(K, L, p, LL)
	A81 := v(K, L, LL, p)
	A82 := v(K, L, p, KK)
	A83 := v(K, L, KK, p)
	A84 := v(K, L, p, LLL)
	A85 := v(K, L, LLL, p)
	A86 := v(K, L, p, KKK)
	A87 := v(K, L, KKK, p)
	A88 := v(K, LLL, p, J)
	A89 := v(K, LLL, J, p)
	A90 := v(K, KKK, p, J)
	A91 := v(K, KKK, J, p)
	A96 := v(K, LL, p, J)
	A97 := v(K, LL, J, p)
	A98 := v(K, KK, p, J)
	A99 := v(K, KK, J, p)
	A124 := v(KKK, K, p, LL)
	A125 := v(KKK, K, LL, p)
	A127 := v(KKK, K, p, KK)
	A128 := v(KKK, K, KK, p)
	A130 := v(LLL, K, p, LL)
	A131 := v(LLL, K, LL, p)
	A133 := v(LLL, K, p, KK)
	A134 := v(LLL, K, KK, p)

	var TIMP, TAMP, TEMP, TUMP, TOMP float64
	// SUM1
	TIMP = -A80 - A81
	TAMP = -A80 + A81
	TOMP = (DUE*A1-A2)*A4 + (DUE*A2-A1)*A5
	suma += TIMP * TOMP * E1 * E2
	sumb += TAMP * TOMP * E1 * E2
	// SUM2
	TIMP = -A82 - A83
	TAMP = -A82 + A83
	TOMP = (DUE*A2-A1)*A6 + (DUE*A1-A2)*A7
	suma += TIMP * TOMP * E1 * E3
	sumb += TAMP * TOMP * E1 * E3
	// SUM3
	TIMP = -A84 - A85
	TAMP = -A84 + A85
	TOMP = (DUE*A1-A2)*A20 + (DUE*A2-A1)*A21
	suma += TIMP * TOMP * D05 * E1 * E8
	sumb += TAMP * TOMP * D05 * E1 * E8
	// SUM4
	TIMP = -A86 - A87
	TAMP = -A86 + A87
	TOMP = (DUE*A2-A1)*A23 + (DUE*A1-A2)*A24
	suma += TIMP * TOMP * D05 * E1 * E9
	sumb += TAMP * TOMP * D05 * E1 * E9
	// SUM5
	TIMP = -A88 - A89
	TAMP = -A88 + A89
	TOMP = (DUE*A1-A2)*A26 + (DUE*A2-A1)*A27
	suma += TIMP * TOMP * E1 * E4
	sumb += TAMP * TOMP * E1 * E4
	// SUM6
	TIMP = -A90 - A91
	TAMP = -A90 + A91
	TOMP = (DUE*A2-A1)*A29 + (DUE*A1-A2)*A30
	suma += TIMP * TOMP * E1 * E5
	sumb += TAMP * TOMP * E1 * E5
	// SUM9
	TIMP = -A96 - A97
	TAMP = -A96 + A97
	TOMP = (DUE*A1-A2)*A38 + (DUE*A2-A1)*A39
	suma += TIMP * TOMP * D05 * E1 * E10
	sumb += TAMP * TOMP * D05 * E1 * E10
	// SUM10
	TIMP = -A98 - A99
	TAMP = -A98 + A99
	TOMP = (DUE*A2-A1)*A41 + (DUE*A1-A2)*A42
	suma += TIMP * TOMP * D05 * E1 * E11
	sumb += TAMP * TOMP * D05 * E1 * E11
	// SUM17
	TIMP = A1 - DUE*A2
	TAMP = A62 - DUE*A63
	TEMP = A124 * ((A1 + A2) * A62)
	TUMP = A124 * ((-A1 + A2) * A62)
	TOMP = A124*TIMP*A63 + A125*TIMP*TAMP
	suma += (TEMP + TOMP) * E1 * E18
	sumb += (TUMP + TOMP) * E1 * E18
	// SUM18
	TIMP = A2 - DUE*A1
	TAMP = A65 - DUE*A66
	TEMP = A127 * ((A2 + A1) * A65)
	TUMP = A127 * ((-A2 + A1) * A65)
	TOMP = A127*TIMP*A66 + A128*TIMP*TAMP
	suma += (TEMP + TOMP) * E1 * E19
	sumb += (TUMP + TOMP) * E1 * E19
	// SUM19
	TIMP = A2 - DUE*A1
	TAMP = A68 - DUE*A69
	TEMP = A130 * ((A2 + A1) * A68)
	TUMP = A130 * ((-A2 + A1) * A68)
	TOMP = A130*TIMP*A69 + A131*TIMP*TAMP
	suma += (TEMP + TOMP) * E1 * E20
	sumb += (TUMP + TOMP) * E1 * E20
	// SUM20
	TIMP = A1 - DUE*A2
	TAMP = A71 - DUE*A72
	TEMP = A133 * ((A1 + A2) * A71)
	TUMP = A133 * ((-A1 + A2) * A71)
	TOMP = A133*TIMP*A72 + A134*TIMP*TAMP
	suma += (TEMP + TOMP) * E1 * E21
	sumb += (TUMP + TOMP) * E1 * E21
	// SUM25
	TIMP = A1 - DUE*A2
	TAMP = A73 - DUE*A72
	suma += A134 * TIMP * TAMP * D05 * E1 * E18
	sumb += A134 * TIMP * TAMP * D05 * E1 * E18
	// SUM26
	TIMP = A2 - DUE*A1
	TAMP = A67 - DUE*A66
	suma += A128 * TIMP * TAMP * D05 * E1 * E20
	sumb += A128 * TIMP * TAMP * D05 * E1 * E20
	// SUM27
	TIMP = A2 - DUE*A1
	TAMP = A70 - DUE*A69
	suma += A131 * TIMP * TAMP * D05 * E1 * E19
	sumb += A131 * TIMP * TAMP * D05 * E1 * E19
	// SUM28
	TIMP = A1 - DUE*A2
	TAMP = A64 - DUE*A63
	suma += A125 * TIMP * TAMP * D05 * E1 * E21
	sumb += A125 * TIMP * TAMP * D05 * E1 * E21
	// SUM35
	TAMP = A3 - DUE*A2
	TIMP = A64 - DUE*A63
	TUMP = A125 * TAMP * TIMP
	suma += (A124*((A2+A3)*A64+TAMP*A63) + TUMP) * E18 * E21
	sumb += (A124*((A2-A3)*A64+TAMP*A63) + TUMP) * E18 * E21
	// SUM36
	TAMP = A3 - DUE*A1
	TIMP = A67 - DUE*A66
	TUMP = A128 * TAMP * TIMP
	suma += (A127*((A1+A3)*A67+TAMP*A66) + TUMP) * E19 * E20
	sumb += (A127*((A1-A3)*A67+TAMP*A66) + TUMP) * E19 * E20
	// SUM37
	TAMP = A3 - DUE*A1
	TIMP = A70 - DUE*A69
	TUMP = A131 * TAMP * TIMP
	suma += (A130*((A1+A3)*A70+TAMP*A69) + TUMP) * E19 * E20
	sumb += (A130*((A1-A3)*A70+TAMP*A69) + TUMP) * E19 * E20
	// SUM38
	TAMP = A3 - DUE*A2
	TIMP = A73 - DUE*A72
	TUMP = A134 * TAMP * TIMP
	suma += (A133*((A2+A3)*A73+TAMP*A72) + TUMP) * E18 * E21
	sumb += (A133*((A2-A3)*A73+TAMP*A72) + TUMP) * E18 * E21
	// SUM43
	TAMP = DUE*A23 - A24
	TEMP = A23 - DUE*A24
	TIMP = A23 + A24
	TUMP = -A23 + A24
	TOMP = -A125 * (TAMP*A39 - TEMP*A40)
	suma += (-A124*(TAMP*A39-TIMP*A40) + TOMP) * E9 * E18
	sumb += (-A124*(-TAMP*A39-TUMP*A40) + TOMP) * E9 * E18
	// SUM44
	TAMP = DUE*A24 - A23
	TEMP = A24 - DUE*A23
	TIMP = A24 + A23
	TUMP = -A24 + A23
	TOMP = -A128 * (TAMP*A42 - TEMP*A43)
	suma += (-A127*(TAMP*A42-TIMP*A43) + TOMP) * E9 * E19
	sumb += (-A127*(-TAMP*A42-TUMP*A43) + TOMP) * E9 * E19
	// SUM45
	TAMP = DUE*A20 - A21
	TEMP = A20 - DUE*A21
	TIMP = A20 + A21
	TUMP = -A20 + A21
	TOMP = -A131 * (TAMP*A38 - TEMP*A40)
	suma += (-A130*(TAMP*A38-TIMP*A40) + TOMP) * E8 * E20
	sumb += (-A130*(-TAMP*A38-TUMP*A40) + TOMP) * E8 * E20
	// SUM46
	TAMP = DUE*A21 - A20
	TEMP = A21 - DUE*A20
	TIMP = A21 + A20
	TUMP = -A21 + A20
	TOMP = -A134 * (TAMP*A41 - TEMP*A43)
	suma += (-A133*(TAMP*A41-TIMP*A43) + TOMP) * E8 * E21
	sumb += (-A133*(-TAMP*A41-TUMP*A43) + TOMP) * E8 * E21
	// SUM51
	TAMP = A25 - DUE*A23
	TEMP = A25 - A23
	TIMP = A25 + A23
	TUMP = DUE*A25 - A23
	TOMP = A125 * (TAMP*A39 - TUMP*A38)
	suma += (-A124*(-TAMP*A39-TIMP*A38) + TOMP) * E18 * E25
	sumb += (-A124*(TAMP*A39-TEMP*A38) + TOMP) * E18 * E25
	// SUM52
	TAMP = A25 - DUE*A24
	TEMP = A25 - A24
	TIMP = A25 + A24
	TUMP = DUE*A25 - A24
	TOMP = A128 * (TAMP*A42 - TUMP*A41)
	suma += (-A127*(-TAMP*A42-TIMP*A41) + TOMP) * E19 * E26
	sumb += (-A127*(TAMP*A42-TEMP*A41) + TOMP) * E19 * E26
	// SUM53
	TAMP = A22 - DUE*A20
	TEMP = A22 - A20
	TIMP = A22 + A20
	TUMP = DUE*A22 - A20
	TOMP = A131 * (TAMP*A38 - TUMP*A39)
	suma += (-A130*(-TAMP*A38-TIMP*A39) + TOMP) * E20 * E25
	sumb += (-A130*(TAMP*A38-TEMP*A39) + TOMP) * E20 * E25
	// SUM54
	TAMP = A22 - DUE*A21
	TEMP = A22 - A21
	TIMP = A22 + A21
	TUMP = DUE*A22 - A21
	TOMP = A134 * (TAMP*A41 - TUMP*A42)
	suma += (-A133*(-TAMP*A41-TIMP*A42) + TOMP) * E21 * E26
	sumb += (-A133*(TAMP*A41-TEMP*A42) + TOMP) * E21 * E26
	return
}

// k1p3h ports K1P3H_core (Z(12),Z(13)): one particle KKK and a non-core occupied triple
// KK<=LL<=MM.
func (e *elements) k1p3h(p, j, k, l int, vir, occ [][]int) (accA, accB float64) {
	ns := len(vir)
	for ts := range ns {
		for _, KKK := range vir[ts] {
			for s3 := range ns { // MM (L3)
				for s2 := 0; s2 <= s3; s2++ { // LL (L2)
					for s1 := 0; s1 <= s2; s1++ { // KK (L1)
						for i3, MM := range occ[s3] {
							max2 := len(occ[s2])
							if s2 == s3 {
								max2 = i3 + 1
							}
							for i2 := 0; i2 < max2; i2++ {
								LL := occ[s2][i2]
								max1 := len(occ[s1])
								if s1 == s2 {
									max1 = i2 + 1
								}
								for _, KK := range occ[s1][:max1] {
									f := k1p3hFac(KK, LL, MM)
									a, b := e.k1p3hTerm(p, j, k, l, KKK, KK, LL, MM)
									accA += a * f
									accB += b * f
								}
							}
						}
					}
				}
			}
		}
	}
	return
}

// k1p3hFac is the K1P3H coincidence factor for the hole triple (KK<=LL<=MM).
func k1p3hFac(KK, LL, MM int) float64 {
	switch {
	case LL == MM && KK == MM:
		return 1.0 / 6.0
	case KK == LL && LL != MM, LL == MM && KK != MM:
		return 0.5
	default:
		return 1.0
	}
}

func (e *elements) k1p3hTerm(p, J, K, L, KKK, KK, LL, MM int) (suma, sumb float64) {
	v, ep := e.v, e.eps
	const DUE = 2.0
	A1 := v(KK, LL, KKK, MM)
	A2 := v(KK, LL, MM, KKK)
	A3 := v(KK, MM, LL, KKK)
	A4 := v(KK, LL, KKK, J)
	A5 := v(KK, LL, J, KKK)
	A6 := v(MM, LL, KKK, J)
	A7 := v(MM, LL, J, KKK)
	A8 := v(KK, MM, KKK, J)
	A9 := v(KK, MM, J, KKK)
	A10 := v(KK, LL, L, MM)
	A11 := v(KK, LL, MM, L)
	A12 := v(KK, MM, LL, L)
	E1 := 1 / (ep[J] - ep[MM])
	E2 := 1 / (ep[J] - ep[KK])
	E3 := 1 / (ep[J] - ep[LL])
	E4 := 1 / (ep[J] + ep[KKK] - ep[KK] - ep[LL])
	E5 := 1 / (ep[J] + ep[KKK] - ep[LL] - ep[MM])
	E6 := 1 / (ep[J] + ep[KKK] - ep[KK] - ep[MM])
	E7 := 1 / (ep[J] + ep[KKK] - ep[MM] - ep[L])
	E8 := 1 / (ep[J] + ep[KKK] - ep[LL] - ep[L])
	E9 := 1 / (ep[J] + ep[KKK] - ep[KK] - ep[L])
	A16 := v(K, L, p, MM)
	A17 := v(K, L, MM, p)
	A18 := v(K, L, p, KK)
	A19 := v(K, L, KK, p)
	A20 := v(K, L, p, LL)
	A21 := v(K, L, LL, p)
	A22 := v(K, KKK, p, MM)
	A23 := v(K, KKK, MM, p)
	A24 := v(K, KKK, p, LL)
	A25 := v(K, KKK, LL, p)
	A26 := v(K, KKK, p, KK)
	A27 := v(K, KKK, KK, p)

	var TAMP, TIMP, TOMP float64
	// SUM1
	TAMP = DUE*A1 - A2
	TIMP = A1 - DUE*A2
	TOMP = TAMP*A4 - TIMP*A5
	suma += (A16 + A17) * TOMP * E1 * E4
	sumb += (A16 - A17) * TOMP * E1 * E4
	// SUM2
	TAMP = DUE*A3 - A2
	TIMP = A3 - DUE*A2
	TOMP = TAMP*A6 - TIMP*A7
	suma += (A18 + A19) * TOMP * E2 * E5
	sumb += (A18 - A19) * TOMP * E2 * E5
	// SUM3
	TAMP = DUE*A1 - A3
	TIMP = A1 - DUE*A3
	TOMP = TAMP*A8 - TIMP*A9
	suma += (A20 + A21) * TOMP * E3 * E6
	sumb += (A20 - A21) * TOMP * E3 * E6
	// SUM4
	TAMP = A4 - DUE*A5
	TIMP = DUE*A4 - A5
	TOMP = -A22 * (TAMP*A10 - TIMP*A11)
	suma += (TOMP - A23*(A4+A5)*(A10+A11)) * E4 * E7
	sumb += (TOMP + A23*(A4-A5)*(A10-A11)) * E4 * E7
	// SUM5
	TAMP = A8 - DUE*A9
	TIMP = DUE*A8 - A9
	TOMP = -A24 * (TAMP*A10 - TIMP*A12)
	suma += (TOMP - A25*(A8+A9)*(A10+A12)) * E6 * E8
	sumb += (TOMP + A25*(A8-A9)*(A10-A12)) * E6 * E8
	// SUM6
	TAMP = A6 - DUE*A7
	TIMP = DUE*A6 - A7
	TOMP = -A26 * (TAMP*A12 - TIMP*A11)
	suma += (TOMP - A27*(A6+A7)*(A12+A11)) * E5 * E9
	sumb += (TOMP + A27*(A6-A7)*(A12-A11)) * E5 * E9
	return
}

// k3p1h ports K3P1H_core (Z(1),Z(2)): one non-core hole KK and a virtual triple
// KKK<=LLL<=MMM.
func (e *elements) k3p1h(p, j, k, l int, vir, occ [][]int) (accA, accB float64) {
	ns := len(vir)
	for ls := range ns {
		for _, KK := range occ[ls] {
			for s3 := range ns { // MMM (T3)
				for s2 := 0; s2 <= s3; s2++ { // LLL (T2)
					for s1 := 0; s1 <= s2; s1++ { // KKK (T1)
						for i3, MMM := range vir[s3] {
							max2 := len(vir[s2])
							if s2 == s3 {
								max2 = i3 + 1
							}
							for i2 := 0; i2 < max2; i2++ {
								LLL := vir[s2][i2]
								max1 := len(vir[s1])
								if s1 == s2 {
									max1 = i2 + 1
								}
								for _, KKK := range vir[s1][:max1] {
									f := k3p1hFac(KKK, LLL, MMM)
									a, b := e.k3p1hTerm(p, j, k, l, KK, KKK, LLL, MMM)
									accA += a * f
									accB += b * f
								}
							}
						}
					}
				}
			}
		}
	}
	return
}

// k3p1hFac is the K3P1H coincidence factor for the particle triple (KKK<=LLL<=MMM).
func k3p1hFac(KKK, LLL, MMM int) float64 {
	switch {
	case KKK == LLL && KKK == MMM:
		return 1.0 / 6.0
	case LLL == MMM && KKK != LLL, KKK == LLL && LLL != MMM:
		return 0.5
	default:
		return 1.0
	}
}

func (e *elements) k3p1hTerm(p, J, K, L, KK, KKK, LLL, MMM int) (suma, sumb float64) {
	v, ep := e.v, e.eps
	const DUE = 2.0
	A1 := v(KKK, LLL, KK, MMM)
	A2 := v(KKK, LLL, MMM, KK)
	A3 := v(KKK, MMM, LLL, KK)
	A22 := v(KKK, LLL, KK, L)
	A23 := v(KKK, LLL, L, KK)
	A25 := v(LLL, MMM, KK, L)
	A26 := v(LLL, MMM, L, KK)
	A28 := v(KKK, MMM, KK, L)
	A29 := v(KKK, MMM, L, KK)
	A49 := v(KKK, LLL, MMM, J)
	A50 := v(KKK, LLL, J, MMM)
	A51 := v(KKK, J, LLL, MMM)
	E1 := 1 / (ep[KKK] - ep[L])
	E2 := 1 / (ep[LLL] - ep[L])
	E3 := 1 / (ep[MMM] - ep[L])
	E7 := 1 / (ep[KKK] + ep[LLL] - ep[KK] - ep[L])
	E8 := 1 / (ep[LLL] + ep[MMM] - ep[KK] - ep[L])
	E9 := 1 / (ep[KKK] + ep[MMM] - ep[KK] - ep[L])
	E16 := 1 / (ep[J] + ep[KKK] - ep[KK] - ep[L])
	E17 := 1 / (ep[J] + ep[LLL] - ep[KK] - ep[L])
	E18 := 1 / (ep[J] + ep[MMM] - ep[KK] - ep[L])
	A76 := v(K, KKK, p, J)
	A77 := v(K, KKK, J, p)
	A78 := v(K, LLL, p, J)
	A79 := v(K, LLL, J, p)
	A80 := v(K, MMM, p, J)
	A81 := v(K, MMM, J, p)
	A88 := v(K, KKK, p, KK)
	A89 := v(K, KKK, KK, p)
	A90 := v(K, LLL, p, KK)
	A91 := v(K, LLL, KK, p)
	A92 := v(K, MMM, p, KK)
	A93 := v(K, MMM, KK, p)

	var TUMP, TIMP, TIMP1, TAMP, TAMP1, TOMP float64
	// SUM1
	TUMP = A80 + A81
	TIMP = A80 - A81
	TOMP = (DUE*A1-A2)*A22 + (DUE*A2-A1)*A23
	suma += TUMP * TOMP * E3 * E7
	sumb += TIMP * TOMP * E3 * E7
	// SUM2
	TUMP = A76 + A77
	TIMP = A76 - A77
	TOMP = (DUE*A3-A2)*A26 + (DUE*A2-A3)*A25
	suma += TUMP * TOMP * E1 * E8
	sumb += TIMP * TOMP * E1 * E8
	// SUM3
	TUMP = A78 + A79
	TIMP = A78 - A79
	TOMP = (DUE*A1-A3)*A28 + (DUE*A3-A1)*A29
	suma += TUMP * TOMP * E2 * E9
	sumb += TIMP * TOMP * E2 * E9
	// SUM19
	TIMP = A22 + A23
	TIMP1 = A22 - A23
	TAMP = A49 + A50
	TAMP1 = A49 - A50
	TOMP = (DUE*A22-A23)*A49 + (DUE*A23-A22)*A50
	suma += (A92*TOMP - A93*TIMP*TAMP) * E7 * E18
	sumb += (A92*TOMP - A93*TIMP1*TAMP1) * E7 * E18
	// SUM20
	TIMP = A26 + A25
	TIMP1 = A26 - A25
	TAMP = A49 + A51
	TAMP1 = A49 - A51
	TOMP = (DUE*A26-A25)*A49 + (DUE*A25-A26)*A51
	suma += (A88*TOMP - A89*TIMP*TAMP) * E8 * E16
	sumb += (A88*TOMP - A89*TIMP1*TAMP1) * E8 * E16
	// SUM21
	TIMP = A28 + A29
	TIMP1 = A28 - A29
	TAMP = A51 + A50
	TAMP1 = A51 - A50
	TOMP = (DUE*A28-A29)*A51 + (DUE*A29-A28)*A50
	suma += (A90*TOMP - A91*TIMP*TAMP) * E9 * E17
	sumb += (A90*TOMP - A91*TIMP1*TAMP1) * E9 * E17
	return
}
