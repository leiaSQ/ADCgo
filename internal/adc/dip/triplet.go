package dip

import "github.com/leiaSQ/ADCgo/internal/adc/backend"

// triplet implements the triplet ADC2-DIP matrix elements
// (../ADC/adc2_dip/triplet.cpp). Closed-shell blocks do not exist for triplet
// (triplet.hpp:29-32). Three spin functions per |ijkr> (multiplet = 3).
type triplet struct{ base }

func (t *triplet) iiJJ(row, col Config) (float64, bool)      { return 0, false }
func (t *triplet) ijKK(row, col Config) (float64, bool)      { return 0, false }
func (t *triplet) lkkII(row, col Config) (backend.Mat, bool) { return backend.Mat{}, false }
func (t *triplet) klmII(row, col Config) (backend.Mat, bool) { return backend.Mat{}, false }

// ijKL: <ij|H|kl> (eq. A.1).
func (t *triplet) ijKL(row, col Config) (float64, bool) {
	i, j := row.Occ[0], row.Occ[1]
	k, l := col.Occ[0], col.Occ[1]
	deltaIK, deltaJK, deltaJL := i == k, j == k, j == l
	var W, U float64
	for r := t.nocc(); r < t.norb(); r++ {
		for ss := t.nocc(); ss <= r; ss++ {
			if deltaIK && t.symOrb(j) == t.symOrb(l) {
				W += t.wTerm(j, l, ss, r)
			}
			if deltaJK && t.symOrb(i) == t.symOrb(l) {
				W -= t.wTerm(i, l, ss, r)
			}
			if deltaJL && t.symOrb(i) == t.symOrb(k) {
				W += t.wTerm(i, k, ss, r)
			}
			if symProduct(t.symOrb(i), t.symOrb(j)) == symProduct(t.symOrb(r), t.symOrb(ss)) {
				U += t.uTerm(i, j, k, l, ss, r, false)
			}
		}
	}
	el := t.vminus(i, k, j, l) + W - U
	if deltaIK && deltaJL {
		el += -(t.energy(i) + t.energy(j))
	}
	return el, true
}

// lkkIJ: <lkkr|H|ij> (Table A.2). Column vector over the row virtual group.
func (t *triplet) lkkIJ(row, col Config) (backend.Mat, bool) {
	l, k := row.Occ[0], row.Occ[1]
	i, j := col.Occ[0], col.Occ[1]
	deltaIK, deltaIL, deltaJK, deltaJL := i == k, i == l, j == k, j == l
	if !deltaIK && !deltaIL && !deltaJK && !deltaJL {
		return backend.Mat{}, false
	}
	rs := t.virSym(row)
	blk := backend.NewMat(t.sizeVirGroup(rs), 1)
	if deltaIK {
		blk.AddSubVec(0, 0, 1, t.V(j, l, i, rs))
	}
	if deltaIL {
		blk.AddSubVec(0, 0, -1, t.V(j, k, k, rs))
	}
	if deltaJK {
		blk.AddSubVec(0, 0, -1, t.V(i, l, j, rs))
	}
	if deltaJL {
		blk.AddSubVec(0, 0, 1, t.V(i, k, k, rs))
	}
	return blk, true
}

// klmIJ: <klmr|H|ij> (Table A.2). Three stacked spin parts, length 3*nvR.
func (t *triplet) klmIJ(row, col Config) (backend.Mat, bool) {
	k, l, m := row.Occ[0], row.Occ[1], row.Occ[2]
	i, j := col.Occ[0], col.Occ[1]
	deltaIK, deltaIL, deltaIM := i == k, i == l, i == m
	deltaJK, deltaJL, deltaJM := j == k, j == l, j == m
	if !deltaIK && !deltaIL && !deltaIM && !deltaJK && !deltaJL && !deltaJM {
		return backend.Mat{}, false
	}
	rs := t.virSym(row)
	nv := t.sizeVirGroup(rs)
	blk := backend.NewMat(3*nv, 1)
	if deltaIK {
		blk.AddSubVec(0, 0, -1, t.V(j, m, l, rs))
		blk.AddSubVec(nv, 0, 1, t.V(j, l, m, rs))
		blk.AddSubVec(2*nv, 0, 1, t.V(j, l, m, rs))
		blk.AddSubVec(2*nv, 0, -1, t.V(j, m, l, rs))
	}
	if deltaIL {
		blk.AddSubVec(0, 0, 1, t.V(j, m, k, rs))
		blk.AddSubVec(nv, 0, 1, t.V(j, m, k, rs))
		blk.AddSubVec(nv, 0, -1, t.V(j, k, m, rs))
		blk.AddSubVec(2*nv, 0, -1, t.V(j, k, m, rs))
	}
	if deltaIM {
		blk.AddSubVec(0, 0, 1, t.V(j, k, l, rs))
		blk.AddSubVec(0, 0, -1, t.V(j, l, k, rs))
		blk.AddSubVec(nv, 0, -1, t.V(j, l, k, rs))
		blk.AddSubVec(2*nv, 0, 1, t.V(j, k, l, rs))
	}
	if deltaJK {
		blk.AddSubVec(0, 0, 1, t.V(i, m, l, rs))
		blk.AddSubVec(nv, 0, -1, t.V(i, l, m, rs))
		blk.AddSubVec(2*nv, 0, -1, t.V(i, l, m, rs))
		blk.AddSubVec(2*nv, 0, 1, t.V(i, m, l, rs))
	}
	if deltaJL {
		blk.AddSubVec(0, 0, -1, t.V(i, m, k, rs))
		blk.AddSubVec(nv, 0, -1, t.V(i, m, k, rs))
		blk.AddSubVec(nv, 0, 1, t.V(i, k, m, rs))
		blk.AddSubVec(2*nv, 0, 1, t.V(i, k, m, rs))
	}
	if deltaJM {
		blk.AddSubVec(0, 0, -1, t.V(i, k, l, rs))
		blk.AddSubVec(0, 0, 1, t.V(i, l, k, rs))
		blk.AddSubVec(nv, 0, 1, t.V(i, l, k, rs))
		blk.AddSubVec(2*nv, 0, -1, t.V(i, k, l, rs))
	}
	return blk, true
}

// jiiLKK: <jiir|H|lkkr'> (Table A.3). nvR×nvC over the row/col virtual groups.
func (t *triplet) jiiLKK(row, col Config) (backend.Mat, bool) {
	j, i := row.Occ[0], row.Occ[1]
	l, k := col.Occ[0], col.Occ[1]
	deltaIL, deltaJL := i == l, j == l
	deltaIK, deltaJK := i == k, j == k
	rowSym, colSym := t.virSym(row), t.virSym(col)
	deltaSym := rowSym == colSym
	if !(deltaIK || (deltaIK && deltaJL) || (deltaIL && deltaJK) ||
		(deltaSym && (deltaIK || deltaIL || deltaJK || deltaJL))) {
		return backend.Mat{}, false
	}
	blk := backend.NewMat(t.sizeVirGroup(rowSym), t.sizeVirGroup(colSym))
	if deltaIK {
		blk.AddSubMat(0, 0, -1, t.B(j, l, colSym))
	}
	if deltaIK && deltaJL {
		blk.AddSubMat(0, 0, 1, t.A(i, i, colSym))
		blk.AddSubMat(0, 0, -2, t.B(i, i, colSym))
	}
	if deltaIL && deltaJK {
		blk.AddSubMat(0, 0, -1, t.A(l, j, colSym))
		blk.AddSubMat(0, 0, 1, t.B(j, l, colSym))
	}
	if deltaSym {
		var diag float64
		if deltaIK {
			diag += 2*t.v(j, l, i, i) - t.v(i, j, i, l)
		}
		if deltaIL {
			diag -= t.v(j, k, l, k)
		}
		if deltaJK {
			diag -= t.v(i, j, i, l)
		}
		if deltaJL {
			diag += t.v(i, k, i, k)
		}
		if deltaJL && deltaIK {
			diag -= t.energy(j) + t.energy(i) + t.energy(i)
		}
		blk.AddSubDiagConst(0, 0, t.sizeVirGroup(rowSym), diag)
		if deltaJL && deltaIK {
			blk.AddSubDiagVec(0, 0, t.diagEnergies(colSym))
		}
	}
	return blk, true
}

// ijkMLL: <ijkr|H|mllr'> (Table A.5). 3*nvR × nvC.
func (t *triplet) ijkMLL(row, col Config) (backend.Mat, bool) {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	m, l := col.Occ[0], col.Occ[1]
	deltaIM, deltaJM, deltaKM := i == m, j == m, k == m
	deltaIL, deltaJL, deltaKL := i == l, j == l, k == l
	rowSym, colSym := t.virSym(row), t.virSym(col)
	deltaSym := rowSym == colSym
	if !((deltaIM && deltaJL) || (deltaIM && deltaKL) ||
		(deltaJM && deltaIL) || (deltaJM && deltaKL) ||
		(deltaKM && deltaIL) || (deltaKM && deltaJL) ||
		(deltaSym && (deltaIM || deltaIL || deltaJM || deltaJL || deltaKM || deltaKL))) {
		return backend.Mat{}, false
	}
	nv := t.sizeVirGroup(rowSym)
	blk := backend.NewMat(3*nv, t.sizeVirGroup(colSym))
	if deltaIM && deltaJL {
		blk.AddSubMat(0, 0, -1, t.B(j, k, colSym))
		blk.AddSubMat(nv, 0, -1, t.A(k, j, colSym))
		blk.AddSubMat(nv, 0, 1, t.B(j, k, colSym))
		blk.AddSubMat(2*nv, 0, -1, t.A(k, j, colSym))
	}
	if deltaIM && deltaKL {
		blk.AddSubMat(0, 0, 1, t.A(j, k, colSym))
		blk.AddSubMat(0, 0, -1, t.B(j, k, colSym))
		blk.AddSubMat(nv, 0, 1, t.B(j, k, colSym))
		blk.AddSubMat(2*nv, 0, 1, t.A(j, k, colSym))
	}
	if deltaJM && deltaIL {
		blk.AddSubMat(0, 0, 1, t.B(i, k, colSym))
		blk.AddSubMat(nv, 0, 1, t.A(k, i, colSym))
		blk.AddSubMat(2*nv, 0, 1, t.A(k, i, colSym))
		blk.AddSubMat(2*nv, 0, -1, t.B(i, k, colSym))
	}
	if deltaJM && deltaKL {
		blk.AddSubMat(0, 0, -1, t.A(i, k, colSym))
		blk.AddSubMat(0, 0, 1, t.B(i, k, colSym))
		blk.AddSubMat(nv, 0, -1, t.A(i, k, colSym))
		blk.AddSubMat(2*nv, 0, -1, t.B(i, k, colSym))
	}
	if deltaKM && deltaIL {
		blk.AddSubMat(0, 0, -1, t.A(j, i, colSym))
		blk.AddSubMat(nv, 0, -1, t.B(i, j, colSym))
		blk.AddSubMat(2*nv, 0, -1, t.A(j, i, colSym))
		blk.AddSubMat(2*nv, 0, 1, t.B(i, j, colSym))
	}
	if deltaKM && deltaJL {
		blk.AddSubMat(0, 0, 1, t.A(i, j, colSym))
		blk.AddSubMat(nv, 0, 1, t.A(i, j, colSym))
		blk.AddSubMat(nv, 0, -1, t.B(i, j, colSym))
		blk.AddSubMat(2*nv, 0, 1, t.B(i, j, colSym))
	}
	if deltaSym {
		var d0, d1, d2 float64
		if deltaIM {
			d0 += t.v(j, l, k, l)
			d1 -= t.v(j, l, k, l)
		}
		if deltaIL {
			d0 -= t.v(i, k, j, m)
			d1 += t.v(i, j, k, m)
			d2 += t.vminus(i, k, m, j)
		}
		if deltaJM {
			d0 -= t.v(i, l, k, l)
			d2 += t.v(i, l, k, l)
		}
		if deltaJL {
			d0 += t.v(i, m, j, k)
			d1 += t.vminus(i, j, k, m)
			d2 -= t.v(i, j, k, m)
		}
		if deltaKM {
			d1 += t.v(i, l, j, l)
			d2 -= t.v(i, l, j, l)
		}
		if deltaKL {
			d0 += t.vminus(i, m, j, k)
			d1 -= t.v(i, m, j, k)
			d2 += t.v(i, k, j, m)
		}
		blk.AddSubDiagConst(0, 0, nv, d0)
		blk.AddSubDiagConst(nv, 0, nv, d1)
		blk.AddSubDiagConst(2*nv, 0, nv, d2)
	}
	return blk, true
}

// ijkLMN: <ijkr|H|lmnr'> (Table A.7). 3*nvR × 3*nvC (3×3 spin parts).
func (t *triplet) ijkLMN(row, col Config) (backend.Mat, bool) {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	l, m, n := col.Occ[0], col.Occ[1], col.Occ[2]
	deltaIL, deltaJL, deltaKL := i == l, j == l, k == l
	deltaJM, deltaKM := j == m, k == m
	deltaJN, deltaKN := j == n, k == n
	rowSym, colSym := t.virSym(row), t.virSym(col)
	deltaSym := rowSym == colSym
	if !((deltaIL && deltaJM) || (deltaIL && deltaKM) || (deltaIL && deltaKN) ||
		(deltaJL && deltaKM) || (deltaJL && deltaKN) || (deltaJM && deltaKN) ||
		(deltaSym && (deltaIL || deltaJL || deltaJM || deltaJN || deltaKL || deltaKM || deltaKN))) {
		return backend.Mat{}, false
	}
	nvR := t.sizeVirGroup(rowSym)
	nvC := t.sizeVirGroup(colSym)
	blk := backend.NewMat(3*nvR, 3*nvC)
	// row/col offsets for spin part [a][b] (rows use the row group, cols the col).
	offR := func(p int) int { return p * nvR }
	offC := func(p int) int { return p * nvC }
	if deltaIL && deltaJM {
		blk.AddSubMat(offR(0), offC(0), -1, t.B(k, n, colSym))
		blk.AddSubMat(offR(1), offC(1), 1, t.A(k, n, colSym))
		blk.AddSubMat(offR(1), offC(1), -1, t.B(k, n, colSym))
		blk.AddSubMat(offR(1), offC(2), 1, t.A(k, n, colSym))
		blk.AddSubMat(offR(2), offC(1), 1, t.A(k, n, colSym))
		blk.AddSubMat(offR(2), offC(2), 1, t.A(k, n, colSym))
		blk.AddSubMat(offR(2), offC(2), -1, t.B(k, n, colSym))
	}
	if deltaIL && deltaKM {
		blk.AddSubMat(offR(0), offC(1), -1, t.A(j, n, colSym))
		blk.AddSubMat(offR(0), offC(1), 1, t.B(j, n, colSym))
		blk.AddSubMat(offR(0), offC(2), -1, t.A(j, n, colSym))
		blk.AddSubMat(offR(1), offC(0), 1, t.B(j, n, colSym))
		blk.AddSubMat(offR(2), offC(1), -1, t.A(j, n, colSym))
		blk.AddSubMat(offR(2), offC(2), -1, t.A(j, n, colSym))
		blk.AddSubMat(offR(2), offC(2), 1, t.B(j, n, colSym))
	}
	if deltaIL && deltaKN {
		blk.AddSubMat(offR(0), offC(0), 1, t.A(j, m, colSym))
		blk.AddSubMat(offR(0), offC(0), -1, t.B(j, m, colSym))
		blk.AddSubMat(offR(0), offC(2), 1, t.A(j, m, colSym))
		blk.AddSubMat(offR(1), offC(1), -1, t.B(j, m, colSym))
		blk.AddSubMat(offR(2), offC(0), 1, t.A(j, m, colSym))
		blk.AddSubMat(offR(2), offC(2), 1, t.A(j, m, colSym))
		blk.AddSubMat(offR(2), offC(2), -1, t.B(j, m, colSym))
	}
	if deltaJL && deltaKM {
		blk.AddSubMat(offR(0), offC(1), 1, t.A(i, n, colSym))
		blk.AddSubMat(offR(0), offC(1), -1, t.B(i, n, colSym))
		blk.AddSubMat(offR(0), offC(2), 1, t.A(i, n, colSym))
		blk.AddSubMat(offR(1), offC(1), 1, t.A(i, n, colSym))
		blk.AddSubMat(offR(1), offC(2), 1, t.A(i, n, colSym))
		blk.AddSubMat(offR(1), offC(2), -1, t.B(i, n, colSym))
		blk.AddSubMat(offR(2), offC(0), -1, t.B(i, n, colSym))
	}
	if deltaJL && deltaKN {
		blk.AddSubMat(offR(0), offC(0), -1, t.A(i, m, colSym))
		blk.AddSubMat(offR(0), offC(0), 1, t.B(i, m, colSym))
		blk.AddSubMat(offR(0), offC(2), -1, t.A(i, m, colSym))
		blk.AddSubMat(offR(1), offC(0), -1, t.A(i, m, colSym))
		blk.AddSubMat(offR(1), offC(2), -1, t.A(i, m, colSym))
		blk.AddSubMat(offR(1), offC(2), 1, t.B(i, m, colSym))
		blk.AddSubMat(offR(2), offC(1), 1, t.B(i, m, colSym))
	}
	if deltaJM && deltaKN {
		blk.AddSubMat(offR(0), offC(0), 1, t.A(i, l, colSym))
		blk.AddSubMat(offR(0), offC(0), -1, t.B(i, l, colSym))
		blk.AddSubMat(offR(0), offC(1), 1, t.A(i, l, colSym))
		blk.AddSubMat(offR(1), offC(0), 1, t.A(i, l, colSym))
		blk.AddSubMat(offR(1), offC(1), 1, t.A(i, l, colSym))
		blk.AddSubMat(offR(1), offC(1), -1, t.B(i, l, colSym))
		blk.AddSubMat(offR(2), offC(2), -1, t.B(i, l, colSym))
	}
	if deltaSym {
		var d [3][3]float64
		if deltaIL {
			d[0][0] += t.v(j, m, k, n)
			d[0][1] -= t.v(j, n, k, m)
			d[1][0] -= t.v(j, n, k, m)
			d[1][1] += t.v(j, m, k, n)
			d[2][2] += t.vminus(j, m, k, n)
		}
		if deltaJL {
			d[0][0] -= t.v(i, m, k, n)
			d[0][1] += t.v(i, n, k, m)
			d[1][2] += t.vminus(i, n, k, m)
			d[2][0] += t.v(i, n, k, m)
			d[2][1] -= t.v(i, m, k, n)
		}
		if deltaJM {
			d[0][0] += t.v(i, l, k, n)
			d[0][2] -= t.v(i, n, k, l)
			d[1][1] += t.vminus(i, l, k, n)
			d[2][0] -= t.v(i, n, k, l)
			d[2][2] += t.v(i, l, k, n)
		}
		if deltaJN {
			d[0][1] -= t.v(i, l, k, m)
			d[0][2] += t.v(i, m, k, l)
			d[1][0] -= t.vminus(i, l, k, m)
			d[2][1] += t.v(i, m, k, l)
			d[2][2] -= t.v(i, l, k, m)
		}
		if deltaKL {
			d[0][2] += t.vminus(i, m, j, n)
			d[1][0] += t.v(i, m, j, n)
			d[1][1] -= t.v(i, n, j, m)
			d[2][0] -= t.v(i, n, j, m)
			d[2][1] += t.v(i, m, j, n)
		}
		if deltaKM {
			d[0][1] += t.vminus(i, n, j, l)
			d[1][0] -= t.v(i, l, j, n)
			d[1][2] += t.v(i, n, j, l)
			d[2][0] += t.v(i, n, j, l)
			d[2][2] -= t.v(i, l, j, n)
		}
		if deltaKN {
			d[0][0] += t.vminus(i, l, j, m)
			d[1][1] += t.v(i, l, j, m)
			d[1][2] -= t.v(i, m, j, l)
			d[2][1] -= t.v(i, m, j, l)
			d[2][2] += t.v(i, l, j, m)
		}
		for a := range 3 {
			for b := range 3 {
				blk.AddSubDiagConst(offR(a), offC(b), nvR, d[a][b])
			}
		}
		if deltaIL && deltaJM && deltaKN {
			epsi := -(t.energy(i) + t.energy(j) + t.energy(k))
			for p := range 3 {
				blk.AddSubDiagConst(offR(p), offC(p), nvR, epsi)
				blk.AddSubDiagVec(offR(p), offC(p), t.diagEnergies(colSym))
			}
		}
	}
	return blk, true
}
