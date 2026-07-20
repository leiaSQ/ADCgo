package dip

// satelem.go — scalar per-entry evaluation of the 3h1p↔3h1p satellite blocks.
//
// The dense block methods (singlet.go / triplet.go) assemble a whole nvR×nvC (or
// parts·nv) sub-matrix at once, by adding scaled A/B integral panels and structured
// diagonals. The matrix-free DEVICE path cannot do that: a GPU thread owns ONE output
// scalar (one 3h1p config = group × spin-part × virtual orbital) and must recompute
// just that entry from the device-resident ERI tensor. These functions are that
// per-entry form — the exact (row-part pr, row-virtual ra),(col-part pc, col-virtual sb)
// element of each block — and they are the single source of truth the CUDA kernel
// (adc2dip_kernels.cu) transcribes 1:1, exactly as sip/elements.go c22diag/c22off back
// the SIP kernel's d_c22diag/d_c22off.
//
// ra and sb are ABSOLUTE virtual orbital indices (nocc + vir). Every panel entry maps to
// a single ERI: A(x,y,·)[a,b] = (ra x | sb y) = v(ra,x,sb,y) and B(x,y,·)[a,b] =
// (ra sb | x y) = v(ra,sb,x,y). A structured-diagonal term contributes only when the row
// and column land on the same virtual position, which (the diagonals exist only in the
// rowSym==colSym blocks) is exactly ra==sb; the diagonal energy vector entry is then
// eps[ra]. Callers must have confirmed the block is nonzero (the …Gate); these assume it
// and recompute the δ-pattern themselves.
//
// TestSatelliteScalarMatchesDense (matfree_test.go) pins every entry of these against the
// dense blocks, so a transcription slip here or in the kernel is caught on the host.

// abEval returns the two panel-entry closures for a satellite block entry at absolute
// virtual orbitals (ra row, sb col): aVal(x,y)=A(x,y)[·]=v(ra,x,sb,y),
// bVal(x,y)=B(x,y)[·]=v(ra,sb,x,y).
func (b *base) abEval(ra, sb int) (aVal, bVal func(x, y int) float64) {
	aVal = func(x, y int) float64 { return b.v(ra, x, sb, y) }
	bVal = func(x, y int) float64 { return b.v(ra, sb, x, y) }
	return
}

// --- singlet ---------------------------------------------------------------

// jiiLKKElem is the scalar entry of singlet.jiiLKK at (row-virtual ra, col-virtual sb).
func (s *singlet) jiiLKKElem(row, col Config, ra, sb int) float64 {
	j, i := row.Occ[0], row.Occ[1]
	l, k := col.Occ[0], col.Occ[1]
	dIL, dJL := i == l, j == l
	dIK, dJK := i == k, j == k
	dSym := s.virSym(row) == s.virSym(col)
	aVal, bVal := s.abEval(ra, sb)
	var e float64
	if dIK {
		e += 2*aVal(j, l) - bVal(j, l)
	}
	if dIK && dJL {
		e += aVal(i, i) - 2*bVal(i, i)
	}
	if dIL && dJK {
		e += aVal(l, j) + bVal(j, l)
	}
	if dSym && ra == sb {
		var diag float64
		if dIK {
			diag += 2*s.v(j, l, i, i) - s.v(i, j, i, l)
		}
		if dIL {
			diag -= s.v(j, k, l, k)
		}
		if dJK {
			diag -= s.v(i, j, i, l)
		}
		if dJL {
			diag += s.v(i, k, i, k)
		}
		if dJL && dIK {
			diag -= s.energy(j) + s.energy(i) + s.energy(i)
			diag += s.energy(ra)
		}
		e += diag
	}
	return e
}

// ijkMLLElem is the scalar entry of singlet.ijkMLL at (row-part pr∈{0,1}, row-virtual ra,
// col-virtual sb). The column block is a JII type-I group (a single spin part), so pc≡0.
func (s *singlet) ijkMLLElem(row, col Config, pr, ra, sb int) float64 {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	m, l := col.Occ[0], col.Occ[1]
	dIM, dJM, dKM := i == m, j == m, k == m
	dIL, dJL, dKL := i == l, j == l, k == l
	dSym := s.virSym(row) == s.virSym(col)
	aVal, bVal := s.abEval(ra, sb)
	// c picks the part-pr coefficient (row offset 0 vs nv).
	c := func(c0, c1 float64) float64 {
		if pr == 0 {
			return c0
		}
		return c1
	}
	var e float64
	if dIM && dJL {
		e += c(sqrt1_2, -sqrt3_2)*aVal(k, j) + c(sqrt1_2, sqrt3_2)*bVal(j, k)
	}
	if dIM && dKL {
		e += c(-sqrt2, 0)*aVal(j, k) + c(sqrt1_2, sqrt3_2)*bVal(j, k)
	}
	if dJM && dIL {
		e += c(sqrt1_2, -sqrt3_2)*aVal(k, i) + c(-sqrt2, 0)*bVal(i, k)
	}
	if dJM && dKL {
		e += c(sqrt1_2, sqrt3_2)*aVal(i, k) + c(-sqrt2, 0)*bVal(i, k)
	}
	if dKM && dIL {
		e += c(-sqrt2, 0)*aVal(j, i) + c(sqrt1_2, -sqrt3_2)*bVal(i, j)
	}
	if dKM && dJL {
		e += c(sqrt1_2, sqrt3_2)*aVal(i, j) + c(sqrt1_2, -sqrt3_2)*bVal(i, j)
	}
	if dSym && ra == sb {
		var d0, d1 float64
		if dIM {
			d0 -= s.v(j, l, k, l)
			d1 -= s.v(j, l, k, l)
		}
		if dIL {
			d0 += 2*s.v(i, k, j, m) - s.v(i, j, k, m)
			d1 += s.v(i, j, k, m)
		}
		if dJM {
			d0 += 2 * s.v(i, l, k, l)
		}
		if dJL {
			d0 -= s.vplus(i, j, k, m)
			d1 += s.vminus(i, j, k, m)
		}
		if dKM {
			d0 -= s.v(i, l, j, l)
			d1 += s.v(i, l, j, l)
		}
		if dKL {
			d0 += 2*s.v(i, k, j, m) - s.v(i, m, j, k)
			d1 -= s.v(i, m, j, k)
		}
		if pr == 0 {
			e += d0 * sqrt1_2
		} else {
			e += d1 * sqrt3_2
		}
	}
	return e
}

// ijkLMNElem is the scalar entry of singlet.ijkLMN at (row-part pr∈{0,1}, row-virtual ra,
// col-part pc∈{0,1}, col-virtual sb).
func (s *singlet) ijkLMNElem(row, col Config, pr, ra, pc, sb int) float64 {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	l, m, n := col.Occ[0], col.Occ[1], col.Occ[2]
	dIL, dJL, dKL := i == l, j == l, k == l
	dJM, dKM := j == m, k == m
	dJN, dKN := j == n, k == n
	dSym := s.virSym(row) == s.virSym(col)
	aVal, bVal := s.abEval(ra, sb)
	// c picks the (pr,pc) spin-block coefficient from the 2×2 table.
	c := func(c00, c01, c10, c11 float64) float64 {
		switch pr*2 + pc {
		case 0:
			return c00
		case 1:
			return c01
		case 2:
			return c10
		default:
			return c11
		}
	}
	var e float64
	if dIL && dJM {
		e += c(0.5, -sqrt3_4, -sqrt3_4, threehalves)*aVal(k, n) + c(-1, 0, 0, -1)*bVal(k, n)
	}
	if dIL && dKM {
		e += c(-1, sqrt3, 0, 0)*aVal(j, n) + c(0.5, -sqrt3_4, -sqrt3_4, -0.5)*bVal(j, n)
	}
	if dIL && dKN {
		e += c(2, 0, 0, 0)*aVal(j, m) + c(-1, 0, 0, -1)*bVal(j, m)
	}
	if dJL && dKM {
		e += c(0.5, -sqrt3_4, sqrt3_4, -threehalves)*aVal(i, n) + c(0.5, sqrt3_4, -sqrt3_4, 0.5)*bVal(i, n)
	}
	if dJL && dKN {
		e += c(-1, 0, -sqrt3, 0)*aVal(i, m) + c(0.5, sqrt3_4, sqrt3_4, -0.5)*bVal(i, m)
	}
	if dJM && dKN {
		e += c(0.5, sqrt3_4, sqrt3_4, threehalves)*aVal(i, l) + c(-1, 0, 0, -1)*bVal(i, l)
	}
	if dSym && ra == sb {
		var d00, d01, d10, d11 float64
		if dIL {
			d00 += s.v(j, m, k, n) - 0.5*s.v(j, n, k, m)
			d01 += s.v(j, n, k, m)
			d10 += s.v(j, n, k, m)
			d11 += s.v(j, m, k, n) + 0.5*s.v(j, n, k, m)
		}
		if dJL {
			d00 -= 0.5 * s.vplus(i, m, k, n)
			d01 -= s.vplus(i, m, k, n)
			d10 += s.vminus(i, n, k, m)
			d11 += 0.5 * s.vminus(i, m, k, n)
		}
		if dJM {
			d00 += s.vplus(i, l, k, n)
			d11 += s.vminus(i, l, k, n)
		}
		if dJN {
			d00 -= 0.5 * s.vplus(i, l, k, m)
			d01 += s.vplus(i, l, k, m)
			d10 += s.vminus(i, l, k, m)
			d11 += 0.5 * s.vminus(i, l, k, m)
		}
		if dKL {
			d00 += s.v(i, n, j, m) - 0.5*s.v(i, m, j, n)
			d01 += s.v(i, m, j, n)
			d10 -= s.v(i, m, j, n)
			d11 -= s.v(i, n, j, m) + 0.5*s.v(i, m, j, n)
		}
		if dKM {
			d00 -= 0.5 * s.vplus(i, l, j, n)
			d01 += s.vminus(i, l, j, n)
			d10 += s.vplus(i, l, j, n)
			d11 += 0.5 * s.vminus(i, l, j, n)
		}
		if dKN {
			d00 += s.v(i, l, j, m) - 0.5*s.v(i, m, j, l)
			d01 -= s.v(i, m, j, l)
			d10 -= s.v(i, m, j, l)
			d11 += s.v(i, l, j, m) + 0.5*s.v(i, m, j, l)
		}
		e += c(d00, d01*sqrt3_4, d10*sqrt3_4, d11)
		if dIL && dJM && dKN && pr == pc {
			e += s.energy(ra) - (s.energy(i) + s.energy(j) + s.energy(k))
		}
	}
	return e
}

// --- triplet ---------------------------------------------------------------

// jiiLKKElem is the scalar entry of triplet.jiiLKK at (row-virtual ra, col-virtual sb).
func (t *triplet) jiiLKKElem(row, col Config, ra, sb int) float64 {
	j, i := row.Occ[0], row.Occ[1]
	l, k := col.Occ[0], col.Occ[1]
	dIL, dJL := i == l, j == l
	dIK, dJK := i == k, j == k
	dSym := t.virSym(row) == t.virSym(col)
	aVal, bVal := t.abEval(ra, sb)
	var e float64
	if dIK {
		e += -bVal(j, l)
	}
	if dIK && dJL {
		e += aVal(i, i) - 2*bVal(i, i)
	}
	if dIL && dJK {
		e += -aVal(l, j) + bVal(j, l)
	}
	if dSym && ra == sb {
		var diag float64
		if dIK {
			diag += 2*t.v(j, l, i, i) - t.v(i, j, i, l)
		}
		if dIL {
			diag -= t.v(j, k, l, k)
		}
		if dJK {
			diag -= t.v(i, j, i, l)
		}
		if dJL {
			diag += t.v(i, k, i, k)
		}
		if dJL && dIK {
			diag -= t.energy(j) + t.energy(i) + t.energy(i)
			diag += t.energy(ra)
		}
		e += diag
	}
	return e
}

// ijkMLLElem is the scalar entry of triplet.ijkMLL at (row-part pr∈{0,1,2}, row-virtual ra,
// col-virtual sb). The column block is a single spin part, so pc≡0.
func (t *triplet) ijkMLLElem(row, col Config, pr, ra, sb int) float64 {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	m, l := col.Occ[0], col.Occ[1]
	dIM, dJM, dKM := i == m, j == m, k == m
	dIL, dJL, dKL := i == l, j == l, k == l
	dSym := t.virSym(row) == t.virSym(col)
	aVal, bVal := t.abEval(ra, sb)
	c := func(c0, c1, c2 float64) float64 {
		switch pr {
		case 0:
			return c0
		case 1:
			return c1
		default:
			return c2
		}
	}
	var e float64
	if dIM && dJL {
		e += c(0, -1, -1)*aVal(k, j) + c(-1, 1, 0)*bVal(j, k)
	}
	if dIM && dKL {
		e += c(1, 0, 1)*aVal(j, k) + c(-1, 1, 0)*bVal(j, k)
	}
	if dJM && dIL {
		e += c(0, 1, 1)*aVal(k, i) + c(1, 0, -1)*bVal(i, k)
	}
	if dJM && dKL {
		e += c(-1, -1, 0)*aVal(i, k) + c(1, 0, -1)*bVal(i, k)
	}
	if dKM && dIL {
		e += c(-1, 0, -1)*aVal(j, i) + c(0, -1, 1)*bVal(i, j)
	}
	if dKM && dJL {
		e += c(1, 1, 0)*aVal(i, j) + c(0, -1, 1)*bVal(i, j)
	}
	if dSym && ra == sb {
		var d0, d1, d2 float64
		if dIM {
			d0 += t.v(j, l, k, l)
			d1 -= t.v(j, l, k, l)
		}
		if dIL {
			d0 -= t.v(i, k, j, m)
			d1 += t.v(i, j, k, m)
			d2 += t.vminus(i, k, m, j)
		}
		if dJM {
			d0 -= t.v(i, l, k, l)
			d2 += t.v(i, l, k, l)
		}
		if dJL {
			d0 += t.v(i, m, j, k)
			d1 += t.vminus(i, j, k, m)
			d2 -= t.v(i, j, k, m)
		}
		if dKM {
			d1 += t.v(i, l, j, l)
			d2 -= t.v(i, l, j, l)
		}
		if dKL {
			d0 += t.vminus(i, m, j, k)
			d1 -= t.v(i, m, j, k)
			d2 += t.v(i, k, j, m)
		}
		e += c(d0, d1, d2)
	}
	return e
}

// ijkLMNElem is the scalar entry of triplet.ijkLMN at (row-part pr∈{0,1,2}, row-virtual ra,
// col-part pc∈{0,1,2}, col-virtual sb).
func (t *triplet) ijkLMNElem(row, col Config, pr, ra, pc, sb int) float64 {
	i, j, k := row.Occ[0], row.Occ[1], row.Occ[2]
	l, m, n := col.Occ[0], col.Occ[1], col.Occ[2]
	dIL, dJL, dKL := i == l, j == l, k == l
	dJM, dKM := j == m, k == m
	dJN, dKN := j == n, k == n
	dSym := t.virSym(row) == t.virSym(col)
	aVal, bVal := t.abEval(ra, sb)
	// c picks the (pr,pc) spin-block coefficient from the 3×3 table (row-major).
	c := func(mtx [9]float64) float64 { return mtx[pr*3+pc] }
	var e float64
	if dIL && dJM {
		e += c([9]float64{0, 0, 0, 0, 1, 1, 0, 1, 1})*aVal(k, n) +
			c([9]float64{-1, 0, 0, 0, -1, 0, 0, 0, -1})*bVal(k, n)
	}
	if dIL && dKM {
		e += c([9]float64{0, -1, -1, 0, 0, 0, 0, -1, -1})*aVal(j, n) +
			c([9]float64{0, 1, 0, 1, 0, 0, 0, 0, 1})*bVal(j, n)
	}
	if dIL && dKN {
		e += c([9]float64{1, 0, 1, 0, 0, 0, 1, 0, 1})*aVal(j, m) +
			c([9]float64{-1, 0, 0, 0, -1, 0, 0, 0, -1})*bVal(j, m)
	}
	if dJL && dKM {
		e += c([9]float64{0, 1, 1, 0, 1, 1, 0, 0, 0})*aVal(i, n) +
			c([9]float64{0, -1, 0, 0, 0, -1, -1, 0, 0})*bVal(i, n)
	}
	if dJL && dKN {
		e += c([9]float64{-1, 0, -1, -1, 0, -1, 0, 0, 0})*aVal(i, m) +
			c([9]float64{1, 0, 0, 0, 0, 1, 0, 1, 0})*bVal(i, m)
	}
	if dJM && dKN {
		e += c([9]float64{1, 1, 0, 1, 1, 0, 0, 0, 0})*aVal(i, l) +
			c([9]float64{-1, 0, 0, 0, -1, 0, 0, 0, -1})*bVal(i, l)
	}
	if dSym && ra == sb {
		var d [3][3]float64
		if dIL {
			d[0][0] += t.v(j, m, k, n)
			d[0][1] -= t.v(j, n, k, m)
			d[1][0] -= t.v(j, n, k, m)
			d[1][1] += t.v(j, m, k, n)
			d[2][2] += t.vminus(j, m, k, n)
		}
		if dJL {
			d[0][0] -= t.v(i, m, k, n)
			d[0][1] += t.v(i, n, k, m)
			d[1][2] += t.vminus(i, n, k, m)
			d[2][0] += t.v(i, n, k, m)
			d[2][1] -= t.v(i, m, k, n)
		}
		if dJM {
			d[0][0] += t.v(i, l, k, n)
			d[0][2] -= t.v(i, n, k, l)
			d[1][1] += t.vminus(i, l, k, n)
			d[2][0] -= t.v(i, n, k, l)
			d[2][2] += t.v(i, l, k, n)
		}
		if dJN {
			d[0][1] -= t.v(i, l, k, m)
			d[0][2] += t.v(i, m, k, l)
			d[1][0] -= t.vminus(i, l, k, m)
			d[2][1] += t.v(i, m, k, l)
			d[2][2] -= t.v(i, l, k, m)
		}
		if dKL {
			d[0][2] += t.vminus(i, m, j, n)
			d[1][0] += t.v(i, m, j, n)
			d[1][1] -= t.v(i, n, j, m)
			d[2][0] -= t.v(i, n, j, m)
			d[2][1] += t.v(i, m, j, n)
		}
		if dKM {
			d[0][1] += t.vminus(i, n, j, l)
			d[1][0] -= t.v(i, l, j, n)
			d[1][2] += t.v(i, n, j, l)
			d[2][0] += t.v(i, n, j, l)
			d[2][2] -= t.v(i, l, j, n)
		}
		if dKN {
			d[0][0] += t.vminus(i, l, j, m)
			d[1][1] += t.v(i, l, j, m)
			d[1][2] -= t.v(i, m, j, l)
			d[2][1] -= t.v(i, m, j, l)
			d[2][2] += t.v(i, l, j, m)
		}
		e += d[pr][pc]
		if dIL && dJM && dKN && pr == pc {
			e += t.energy(ra) - (t.energy(i) + t.energy(j) + t.energy(k))
		}
	}
	return e
}
