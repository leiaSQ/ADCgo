package dip

import "github.com/leiaSQ/ADCgo/internal/adc/backend"

// singlet implements the singlet ADC2-DIP matrix elements
// (../ADC/adc2_dip/singlet.cpp). Trust this transcription over Tarantelli's
// tables — the reference header warns the paper has misprints.
type singlet struct{ base }

// iiJJ: <ii|H|jj> (eq. A.5).
func (s *singlet) iiJJ(row, col Config) (float64, bool) {
	i, j := row.Occ[0], col.Occ[0]
	deltaIJ := i == j
	var W, U float64
	for r := s.nocc(); r < s.norb(); r++ {
		for ss := s.nocc(); ss <= r; ss++ {
			if deltaIJ {
				W += s.wTerm(i, i, ss, r)
			}
			if s.symOrb(r) == s.symOrb(ss) {
				U += s.uTerm(i, i, j, j, ss, r, true)
			}
		}
	}
	el := s.v(i, j, i, j) - 0.5*U
	if deltaIJ {
		el += 2 * (W - s.energy(i))
	}
	return el, true
}

// ijKK: <ij|H|kk> (eq. A.4).
func (s *singlet) ijKK(row, col Config) (float64, bool) {
	i, j, k := row.Occ[0], row.Occ[1], col.Occ[0]
	deltaIK, deltaJK := i == k, j == k
	var W, U float64
	for r := s.nocc(); r < s.norb(); r++ {
		for ss := s.nocc(); ss <= r; ss++ {
			if deltaIK || deltaJK {
				W += s.wTerm(i, j, ss, r)
			}
			if s.symOrb(r) == s.symOrb(ss) {
				U += s.uTerm(i, j, k, k, ss, r, true)
			}
		}
	}
	el := (s.v(i, k, j, k) - 0.5*U + W) * sqrt2
	return el, true
}

// ijKL: <ij|H|kl> (eq. A.1).
func (s *singlet) ijKL(row, col Config) (float64, bool) {
	i, j := row.Occ[0], row.Occ[1]
	k, l := col.Occ[0], col.Occ[1]
	deltaIK, deltaJK, deltaJL := i == k, j == k, j == l
	var W, U float64
	for r := s.nocc(); r < s.norb(); r++ {
		for ss := s.nocc(); ss <= r; ss++ {
			if deltaIK && s.symOrb(j) == s.symOrb(l) {
				W += s.wTerm(j, l, ss, r)
			}
			if deltaJK && s.symOrb(i) == s.symOrb(l) {
				W += s.wTerm(i, l, ss, r)
			}
			if deltaJL && s.symOrb(i) == s.symOrb(k) {
				W += s.wTerm(i, k, ss, r)
			}
			if symProduct(s.symOrb(i), s.symOrb(j)) == symProduct(s.symOrb(r), s.symOrb(ss)) {
				U += s.uTerm(i, j, k, l, ss, r, true)
			}
		}
	}
	el := s.vplus(i, k, j, l) + W - U
	if deltaIK && deltaJL {
		el += -(s.energy(i) + s.energy(j))
	}
	return el, true
}

// lkkII: <lkkr|H|ii> (Table A.2). Column vector over the row virtual group.
func (s *singlet) lkkII(row, col Config) (backend.Mat, bool) {
	l, k, i := row.Occ[0], row.Occ[1], col.Occ[0]
	deltaIK, deltaIL := i == k, i == l
	if !deltaIK && !deltaIL {
		return backend.Mat{}, false
	}
	rs := s.virSym(row)
	blk := backend.NewMat(s.sizeVirGroup(rs), 1)
	if deltaIK {
		blk.AddSubVec(0, 0, 2*sqrt2, s.V(i, i, l, rs))
		blk.AddSubVec(0, 0, -sqrt2, s.V(i, l, i, rs))
	}
	if deltaIL {
		blk.AddSubVec(0, 0, -sqrt2, s.V(i, k, k, rs))
	}
	return blk, true
}

// lkkIJ: <lkkr|H|ij> (Table A.2). Column vector over the row virtual group.
func (s *singlet) lkkIJ(row, col Config) (backend.Mat, bool) {
	l, k := row.Occ[0], row.Occ[1]
	i, j := col.Occ[0], col.Occ[1]
	deltaIK, deltaIL, deltaJK, deltaJL := i == k, i == l, j == k, j == l
	if !deltaIK && !deltaIL && !deltaJK && !deltaJL {
		return backend.Mat{}, false
	}
	rs := s.virSym(row)
	blk := backend.NewMat(s.sizeVirGroup(rs), 1)
	if deltaIK {
		blk.AddSubVec(0, 0, 2, s.V(i, j, l, rs))
		blk.AddSubVec(0, 0, -1, s.V(j, l, i, rs))
	}
	if deltaIL {
		blk.AddSubVec(0, 0, -1, s.V(j, k, k, rs))
	}
	if deltaJK {
		blk.AddSubVec(0, 0, 2, s.V(i, j, l, rs))
		blk.AddSubVec(0, 0, -1, s.V(i, l, j, rs))
	}
	if deltaJL {
		blk.AddSubVec(0, 0, -1, s.V(i, k, k, rs))
	}
	return blk, true
}

// klmII: <klmr|H|ii> (Table A.2). Two stacked spin parts, length 2*nvR.
func (s *singlet) klmII(row, col Config) (backend.Mat, bool) {
	k, l, m := row.Occ[0], row.Occ[1], row.Occ[2]
	i := col.Occ[0]
	deltaIK, deltaIL, deltaIM := i == k, i == l, i == m
	if !deltaIK && !deltaIL && !deltaIM {
		return backend.Mat{}, false
	}
	rs := s.virSym(row)
	nv := s.sizeVirGroup(rs)
	blk := backend.NewMat(2*nv, 1)
	if deltaIK {
		blk.AddSubVec(0, 0, 2, s.V(k, m, l, rs))
		blk.AddSubVec(0, 0, -1, s.V(k, l, m, rs))
		blk.AddSubVec(nv, 0, sqrt3, s.V(k, l, m, rs))
	}
	if deltaIL {
		blk.AddSubVec(0, 0, -1, s.V(k, l, m, rs))
		blk.AddSubVec(0, 0, -1, s.V(l, m, k, rs))
		blk.AddSubVec(nv, 0, sqrt3, s.V(k, l, m, rs))
		blk.AddSubVec(nv, 0, -sqrt3, s.V(l, m, k, rs))
	}
	if deltaIM {
		blk.AddSubVec(0, 0, 2, s.V(k, m, l, rs))
		blk.AddSubVec(0, 0, -1, s.V(l, m, k, rs))
		blk.AddSubVec(nv, 0, -sqrt3, s.V(l, m, k, rs))
	}
	return blk, true
}

// klmIJ: <klmr|H|ij> (Table A.2). Two stacked spin parts, length 2*nvR.
func (s *singlet) klmIJ(row, col Config) (backend.Mat, bool) {
	k, l, m := row.Occ[0], row.Occ[1], row.Occ[2]
	i, j := col.Occ[0], col.Occ[1]
	deltaIK, deltaIL, deltaIM := i == k, i == l, i == m
	deltaJK, deltaJL, deltaJM := j == k, j == l, j == m
	if !deltaIK && !deltaIL && !deltaIM && !deltaJK && !deltaJL && !deltaJM {
		return backend.Mat{}, false
	}
	rs := s.virSym(row)
	nv := s.sizeVirGroup(rs)
	blk := backend.NewMat(2*nv, 1)
	if deltaIK {
		blk.AddSubVec(0, 0, sqrt2, s.V(j, m, l, rs))
		blk.AddSubVec(0, 0, -sqrt1_2, s.V(j, l, m, rs))
		blk.AddSubVec(nv, 0, sqrt3_2, s.V(j, l, m, rs))
	}
	if deltaIL {
		blk.AddSubVec(0, 0, -sqrt1_2, s.V(j, k, m, rs))
		blk.AddSubVec(0, 0, -sqrt1_2, s.V(j, m, k, rs))
		blk.AddSubVec(nv, 0, sqrt3_2, s.V(j, k, m, rs))
		blk.AddSubVec(nv, 0, -sqrt3_2, s.V(j, m, k, rs))
	}
	if deltaIM {
		blk.AddSubVec(0, 0, sqrt2, s.V(j, k, l, rs))
		blk.AddSubVec(0, 0, -sqrt1_2, s.V(j, l, k, rs))
		blk.AddSubVec(nv, 0, -sqrt3_2, s.V(j, l, k, rs))
	}
	if deltaJK {
		blk.AddSubVec(0, 0, sqrt2, s.V(i, m, l, rs))
		blk.AddSubVec(0, 0, -sqrt1_2, s.V(i, l, m, rs))
		blk.AddSubVec(nv, 0, sqrt3_2, s.V(i, l, m, rs))
	}
	if deltaJL {
		blk.AddSubVec(0, 0, -sqrt1_2, s.V(i, k, m, rs))
		blk.AddSubVec(0, 0, -sqrt1_2, s.V(i, m, k, rs))
		blk.AddSubVec(nv, 0, sqrt3_2, s.V(i, k, m, rs))
		blk.AddSubVec(nv, 0, -sqrt3_2, s.V(i, m, k, rs))
	}
	if deltaJM {
		blk.AddSubVec(0, 0, sqrt2, s.V(i, k, l, rs))
		blk.AddSubVec(0, 0, -sqrt1_2, s.V(i, l, k, rs))
		blk.AddSubVec(nv, 0, -sqrt3_2, s.V(i, l, k, rs))
	}
	return blk, true
}

// jiiLKK: <jiir|H|lkkr'> (Table A.3). nvR×nvC over the row/col virtual groups.
func (s *singlet) jiiLKK(row, col Config) (backend.Mat, bool) {
	j, i := row.Occ[0], row.Occ[1]
	l, k := col.Occ[0], col.Occ[1]
	deltaIL, deltaJL := i == l, j == l
	deltaIK, deltaJK := i == k, j == k
	rowSym, colSym := s.virSym(row), s.virSym(col)
	deltaSym := rowSym == colSym
	if !(deltaIK || (deltaIK && deltaJL) || (deltaIL && deltaJK) ||
		(deltaSym && (deltaIK || deltaIL || deltaJK || deltaJL))) {
		return backend.Mat{}, false
	}
	blk := backend.NewMat(s.sizeVirGroup(rowSym), s.sizeVirGroup(colSym))
	if deltaIK {
		blk.AddSubMat(0, 0, 2, s.A(j, l, colSym))
		blk.AddSubMat(0, 0, -1, s.B(j, l, colSym))
	}
	if deltaIK && deltaJL {
		blk.AddSubMat(0, 0, 1, s.A(i, i, colSym))
		blk.AddSubMat(0, 0, -2, s.B(i, i, colSym))
	}
	if deltaIL && deltaJK {
		blk.AddSubMat(0, 0, 1, s.A(l, j, colSym))
		blk.AddSubMat(0, 0, 1, s.B(j, l, colSym))
	}
	if deltaSym {
		var diag float64
		if deltaIK {
			diag += 2*s.v(j, l, i, i) - s.v(i, j, i, l)
		}
		if deltaIL {
			diag -= s.v(j, k, l, k)
		}
		if deltaJK {
			diag -= s.v(i, j, i, l)
		}
		if deltaJL {
			diag += s.v(i, k, i, k)
		}
		if deltaJL && deltaIK {
			diag -= s.energy(j) + s.energy(i) + s.energy(i)
		}
		blk.AddSubDiagConst(0, 0, s.sizeVirGroup(rowSym), diag)
		if deltaJL && deltaIK {
			blk.AddSubDiagVec(0, 0, s.diagEnergies(colSym))
		}
	}
	return blk, true
}

// ijkMLL: <ijkr|H|mllr'> (Table A.4). 2*nvR × nvC.
func (s *singlet) ijkMLL(row, col Config) (backend.Mat, bool) {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	m, l := col.Occ[0], col.Occ[1]
	deltaIM, deltaJM, deltaKM := i == m, j == m, k == m
	deltaIL, deltaJL, deltaKL := i == l, j == l, k == l
	rowSym, colSym := s.virSym(row), s.virSym(col)
	deltaSym := rowSym == colSym
	if !((deltaIM && deltaJL) || (deltaIM && deltaKL) ||
		(deltaJM && deltaIL) || (deltaJM && deltaKL) ||
		(deltaKM && deltaIL) || (deltaKM && deltaJL) ||
		(deltaSym && (deltaIM || deltaIL || deltaJM || deltaJL || deltaKM || deltaKL))) {
		return backend.Mat{}, false
	}
	nv := s.sizeVirGroup(rowSym)
	blk := backend.NewMat(2*nv, s.sizeVirGroup(colSym))
	if deltaIM && deltaJL {
		blk.AddSubMat(0, 0, sqrt1_2, s.A(k, j, colSym))
		blk.AddSubMat(0, 0, sqrt1_2, s.B(j, k, colSym))
		blk.AddSubMat(nv, 0, -sqrt3_2, s.A(k, j, colSym))
		blk.AddSubMat(nv, 0, sqrt3_2, s.B(j, k, colSym))
	}
	if deltaIM && deltaKL {
		blk.AddSubMat(0, 0, -sqrt2, s.A(j, k, colSym))
		blk.AddSubMat(0, 0, sqrt1_2, s.B(j, k, colSym))
		blk.AddSubMat(nv, 0, sqrt3_2, s.B(j, k, colSym))
	}
	if deltaJM && deltaIL {
		blk.AddSubMat(0, 0, sqrt1_2, s.A(k, i, colSym))
		blk.AddSubMat(0, 0, -sqrt2, s.B(i, k, colSym))
		blk.AddSubMat(nv, 0, -sqrt3_2, s.A(k, i, colSym))
	}
	if deltaJM && deltaKL {
		blk.AddSubMat(0, 0, sqrt1_2, s.A(i, k, colSym))
		blk.AddSubMat(0, 0, -sqrt2, s.B(i, k, colSym))
		blk.AddSubMat(nv, 0, sqrt3_2, s.A(i, k, colSym))
	}
	if deltaKM && deltaIL {
		blk.AddSubMat(0, 0, -sqrt2, s.A(j, i, colSym))
		blk.AddSubMat(0, 0, sqrt1_2, s.B(i, j, colSym))
		blk.AddSubMat(nv, 0, -sqrt3_2, s.B(i, j, colSym))
	}
	if deltaKM && deltaJL {
		blk.AddSubMat(0, 0, sqrt1_2, s.A(i, j, colSym))
		blk.AddSubMat(0, 0, sqrt1_2, s.B(i, j, colSym))
		blk.AddSubMat(nv, 0, sqrt3_2, s.A(i, j, colSym))
		blk.AddSubMat(nv, 0, -sqrt3_2, s.B(i, j, colSym))
	}
	if deltaSym {
		var d0, d1 float64
		if deltaIM {
			d0 -= s.v(j, l, k, l)
			d1 -= s.v(j, l, k, l)
		}
		if deltaIL {
			d0 += 2*s.v(i, k, j, m) - s.v(i, j, k, m)
			d1 += s.v(i, j, k, m)
		}
		if deltaJM {
			d0 += 2 * s.v(i, l, k, l)
		}
		if deltaJL {
			d0 -= s.vplus(i, j, k, m)
			d1 += s.vminus(i, j, k, m)
		}
		if deltaKM {
			d0 -= s.v(i, l, j, l)
			d1 += s.v(i, l, j, l)
		}
		if deltaKL {
			d0 += 2*s.v(i, k, j, m) - s.v(i, m, j, k)
			d1 -= s.v(i, m, j, k)
		}
		blk.AddSubDiagConst(0, 0, nv, d0*sqrt1_2)
		blk.AddSubDiagConst(nv, 0, nv, d1*sqrt3_2)
	}
	return blk, true
}

// ijkLMN: <ijkr|H|lmnr'> (Table A.6). 2*nvR × 2*nvC (2×2 spin parts).
func (s *singlet) ijkLMN(row, col Config) (backend.Mat, bool) {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	l, m, n := col.Occ[0], col.Occ[1], col.Occ[2]
	deltaIL, deltaJL, deltaKL := i == l, j == l, k == l
	deltaJM, deltaKM := j == m, k == m
	deltaJN, deltaKN := j == n, k == n
	rowSym, colSym := s.virSym(row), s.virSym(col)
	deltaSym := rowSym == colSym
	if !((deltaIL && deltaJM) || (deltaIL && deltaKM) || (deltaIL && deltaKN) ||
		(deltaJL && deltaKM) || (deltaJL && deltaKN) || (deltaJM && deltaKN) ||
		(deltaSym && (deltaIL || deltaJL || deltaJM || deltaJN || deltaKL || deltaKM || deltaKN))) {
		return backend.Mat{}, false
	}
	nvR := s.sizeVirGroup(rowSym)
	nvC := s.sizeVirGroup(colSym)
	blk := backend.NewMat(2*nvR, 2*nvC)
	if deltaIL && deltaJM {
		blk.AddSubMat(0, 0, 0.5, s.A(k, n, colSym))
		blk.AddSubMat(0, 0, -1, s.B(k, n, colSym))
		blk.AddSubMat(0, nvC, -sqrt3_4, s.A(k, n, colSym))
		blk.AddSubMat(nvR, 0, -sqrt3_4, s.A(k, n, colSym))
		blk.AddSubMat(nvR, nvC, threehalves, s.A(k, n, colSym))
		blk.AddSubMat(nvR, nvC, -1, s.B(k, n, colSym))
	}
	if deltaIL && deltaKM {
		blk.AddSubMat(0, 0, -1, s.A(j, n, colSym))
		blk.AddSubMat(0, 0, 0.5, s.B(j, n, colSym))
		blk.AddSubMat(0, nvC, sqrt3, s.A(j, n, colSym))
		blk.AddSubMat(0, nvC, -sqrt3_4, s.B(j, n, colSym))
		blk.AddSubMat(nvR, 0, -sqrt3_4, s.B(j, n, colSym))
		blk.AddSubMat(nvR, nvC, -0.5, s.B(j, n, colSym))
	}
	if deltaIL && deltaKN {
		blk.AddSubMat(0, 0, 2, s.A(j, m, colSym))
		blk.AddSubMat(0, 0, -1, s.B(j, m, colSym))
		blk.AddSubMat(nvR, nvC, -1, s.B(j, m, colSym))
	}
	if deltaJL && deltaKM {
		blk.AddSubMat(0, 0, 0.5, s.A(i, n, colSym))
		blk.AddSubMat(0, 0, 0.5, s.B(i, n, colSym))
		blk.AddSubMat(0, nvC, -sqrt3_4, s.A(i, n, colSym))
		blk.AddSubMat(0, nvC, sqrt3_4, s.B(i, n, colSym))
		blk.AddSubMat(nvR, 0, sqrt3_4, s.A(i, n, colSym))
		blk.AddSubMat(nvR, 0, -sqrt3_4, s.B(i, n, colSym))
		blk.AddSubMat(nvR, nvC, -threehalves, s.A(i, n, colSym))
		blk.AddSubMat(nvR, nvC, 0.5, s.B(i, n, colSym))
	}
	if deltaJL && deltaKN {
		blk.AddSubMat(0, 0, -1, s.A(i, m, colSym))
		blk.AddSubMat(0, 0, 0.5, s.B(i, m, colSym))
		blk.AddSubMat(0, nvC, sqrt3_4, s.B(i, m, colSym))
		blk.AddSubMat(nvR, 0, -sqrt3, s.A(i, m, colSym))
		blk.AddSubMat(nvR, 0, sqrt3_4, s.B(i, m, colSym))
		blk.AddSubMat(nvR, nvC, -0.5, s.B(i, m, colSym))
	}
	if deltaJM && deltaKN {
		blk.AddSubMat(0, 0, 0.5, s.A(i, l, colSym))
		blk.AddSubMat(0, 0, -1, s.B(i, l, colSym))
		blk.AddSubMat(0, nvC, sqrt3_4, s.A(i, l, colSym))
		blk.AddSubMat(nvR, 0, sqrt3_4, s.A(i, l, colSym))
		blk.AddSubMat(nvR, nvC, threehalves, s.A(i, l, colSym))
		blk.AddSubMat(nvR, nvC, -1, s.B(i, l, colSym))
	}
	if deltaSym {
		var d00, d01, d10, d11 float64
		if deltaIL {
			d00 += s.v(j, m, k, n) - 0.5*s.v(j, n, k, m)
			d01 += s.v(j, n, k, m)
			d10 += s.v(j, n, k, m)
			d11 += s.v(j, m, k, n) + 0.5*s.v(j, n, k, m)
		}
		if deltaJL {
			d00 -= 0.5 * s.vplus(i, m, k, n)
			d01 -= s.vplus(i, m, k, n)
			d10 += s.vminus(i, n, k, m)
			d11 += 0.5 * s.vminus(i, m, k, n)
		}
		if deltaJM {
			d00 += s.vplus(i, l, k, n)
			d11 += s.vminus(i, l, k, n)
		}
		if deltaJN {
			d00 -= 0.5 * s.vplus(i, l, k, m)
			d01 += s.vplus(i, l, k, m)
			d10 += s.vminus(i, l, k, m)
			d11 += 0.5 * s.vminus(i, l, k, m)
		}
		if deltaKL {
			d00 += s.v(i, n, j, m) - 0.5*s.v(i, m, j, n)
			d01 += s.v(i, m, j, n)
			d10 -= s.v(i, m, j, n)
			d11 -= s.v(i, n, j, m) + 0.5*s.v(i, m, j, n)
		}
		if deltaKM {
			d00 -= 0.5 * s.vplus(i, l, j, n)
			d01 += s.vminus(i, l, j, n)
			d10 += s.vplus(i, l, j, n)
			d11 += 0.5 * s.vminus(i, l, j, n)
		}
		if deltaKN {
			d00 += s.v(i, l, j, m) - 0.5*s.v(i, m, j, l)
			d01 -= s.v(i, m, j, l)
			d10 -= s.v(i, m, j, l)
			d11 += s.v(i, l, j, m) + 0.5*s.v(i, m, j, l)
		}
		blk.AddSubDiagConst(0, 0, nvR, d00)
		blk.AddSubDiagConst(0, nvC, nvR, d01*sqrt3_4)
		blk.AddSubDiagConst(nvR, 0, nvR, d10*sqrt3_4)
		blk.AddSubDiagConst(nvR, nvC, nvR, d11)
		if deltaIL && deltaJM && deltaKN {
			epsi := -(s.energy(i) + s.energy(j) + s.energy(k))
			blk.AddSubDiagVec(0, 0, s.diagEnergies(colSym))
			blk.AddSubDiagConst(0, 0, nvR, epsi)
			blk.AddSubDiagVec(nvR, nvC, s.diagEnergies(colSym))
			blk.AddSubDiagConst(nvR, nvC, nvR, epsi)
		}
	}
	return blk, true
}
