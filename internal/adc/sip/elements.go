package sip

import (
	"math"

	"adcgo/internal/adc/integrals"
)

// Spin-coupling constants (adc_macros.h).
var (
	sqrt1_2 = math.Sqrt(0.5)     // SQRT_1_2
	sqrt3_2 = math.Sqrt(1.5)     // SQRT_3_2
	sqrt3_4 = math.Sqrt(0.75)    // SQRT_3_4
	sqrt1_8 = math.Sqrt(0.125)   // SQRT_1_8
	sqrt3_8 = math.Sqrt(3.0 / 8) // SQRT_3_8
)

// elements computes the ND-ADC(3) IP matrix elements, transcribed from
// ../ADC/ndadc3_ip/calc_*.c. order (2 or 3) gates the 3rd-order self-energy
// (calc_c11_3) and 2nd-order coupling (calc_c12_2). Integrals use the reference's
// V1212(A,B,C,D) = <AB|CD>_phys = (AC|BD)_chem = ints.Eri(A,C,B,D).
type elements struct {
	sp    *Space
	ints  *integrals.Store
	eps   []float64
	order int
	nocc  int
	norb  int
}

func newElements(sp *Space, ints *integrals.Store, eps []float64, order int) *elements {
	return &elements{sp: sp, ints: ints, eps: eps, order: order, nocc: sp.Nocc, norb: sp.Norb}
}

// v is the reference V1212 integral in physicist ordering <AB|CD>.
func (e *elements) v(a, b, c, d int) float64 { return e.ints.Eri(a, c, b, d) }

// so is the 0-based irrep of an absolute orbital.
func (e *elements) so(o int) int { return e.sp.irrep(o) }

// ---------------------------------------------------------------------------
// Main 1h/1h block: c11 = k1 + c11_2 (+ c11_3).
// ---------------------------------------------------------------------------

// c11 returns the (i,j) main-block element (i,j absolute occupied indices of the
// target irrep). Computed once per i>=j pair and mirrored by the caller.
func (e *elements) c11(i, j int) float64 {
	val := e.c11_2(i, j)
	if i == j {
		val -= e.eps[i] // k1: 0th order
	}
	if e.order >= 3 {
		val -= e.c11_3(i, j) // calc_c11_3: c_matrix -= c_ij (non-affinity)
	}
	return val
}

// c11_2 is the 2nd-order hole/hole self-energy (calc_c11_2.c).
func (e *elements) c11_2(i, j int) float64 {
	ei, ej := e.eps[i], e.eps[j]
	emean := 0.5 * (ei + ej)
	var cij float64
	for a := e.nocc; a < e.norb; a++ {
		for b := e.nocc; b < e.norb; b++ {
			for l := range e.nocc {
				// l_sym = sym ⊗ a_sym ⊗ b_sym  =>  a⊗b⊗l == i.
				if e.so(a)^e.so(b)^e.so(l) != e.so(i) {
					continue
				}
				ea, eb, el := e.eps[a], e.eps[b], e.eps[l]
				vabil := e.v(a, b, i, l)
				vabli := e.v(a, b, l, i)
				vabjl := e.v(a, b, j, l)
				vablj := e.v(a, b, l, j)
				vv := vabil*(2*vabjl-vablj) + vabli*(2*vablj-vabjl)
				eabl := ea + eb - el
				cij += 0.5 * vv * (eabl - emean) / ((eabl - ei) * (eabl - ej))
			}
		}
	}
	return cij
}

// c11_3 is the 3rd-order hole/hole self-energy (calc_c11_3.c), four diagrams
// A..D plus the hermitian F correction. Returns c_ij (the value subtracted from
// the main block). Also accumulates the amplitude sums f_ij, f_ji used for the
// F correction; those same sums feed the spectroscopic amplitudes (amplitudes.go
// recomputes them there).
func (e *elements) c11_3(i, j int) float64 {
	cij, fij, fji := e.c11_3sums(i, j)
	return cij + (e.eps[i]-e.eps[j])*(fij-fji)/2 // F_HERM
}

// c11_3sums returns the raw c_ij and the F-matrix sums f_ij, f_ji (before the
// hermitian correction), transcribed from calc_c11_3.c with SAFE_INT/F_HERM.
func (e *elements) c11_3sums(i, j int) (cij, fij, fji float64) {
	ep := e.eps
	ei, ej := ep[i], ep[j]
	no, nb := e.nocc, e.norb

	// C_ij^(A)
	for a := no; a < nb; a++ {
		for b := no; b < nb; b++ {
			for l := range no {
				if e.so(a)^e.so(b)^e.so(l) != e.so(i) {
					continue
				}
				ea, eb, el := ep[a], ep[b], ep[l]
				vabil := e.v(a, b, i, l)
				vabli := e.v(a, b, l, i)
				for c := no; c < nb; c++ {
					for d := no; d < nb; d++ {
						if e.so(c)^e.so(d) != e.so(a)^e.so(b) {
							continue
						}
						ec, ed := ep[c], ep[d]
						vcdjl := e.v(c, d, j, l)
						vcdlj := e.v(c, d, l, j)
						vcdab := e.v(c, d, a, b)
						vcdba := e.v(c, d, b, a)
						exe := (ea + eb - el - ei) * (ec + ed - el - ej)
						vvv := 0.25 / exe * (2*vabil*vcdjl*vcdab -
							vabil*vcdjl*vcdba - vabil*vcdlj*vcdab +
							2*vabil*vcdlj*vcdba - vabli*vcdjl*vcdab +
							2*vabli*vcdjl*vcdba + 2*vabli*vcdlj*vcdab -
							vabli*vcdlj*vcdba)
						cij += vvv
						fij += vvv / (ec + ed - el - ei)
						fji += vvv / (ea + eb - el - ej)
					}
				}
			}
		}
	}

	// C_ij^(B)
	for a := no; a < nb; a++ {
		for b := no; b < nb; b++ {
			for l := range no {
				if e.so(a)^e.so(b)^e.so(l) != e.so(i) {
					continue
				}
				ea, eb, el := ep[a], ep[b], ep[l]
				vabil := e.v(a, b, i, l)
				vabli := e.v(a, b, l, i)
				for c := no; c < nb; c++ {
					for m := range no {
						if e.so(a)^e.so(c)^e.so(m) != e.so(i) {
							continue
						}
						ec, em := ep[c], ep[m]
						vacjm := e.v(a, c, j, m)
						vacmj := e.v(a, c, m, j)
						vlcbm := e.v(l, c, b, m)
						vlcmb := e.v(l, c, m, b)
						exe := (ea + eb - el - ei) * (ea + ec - em - ej)
						vvv := 1 / exe * (4*vabil*vacjm*vlcbm -
							2*vabil*vacjm*vlcmb - 2*vabil*vacmj*vlcbm +
							vabil*vacmj*vlcmb - 2*vabli*vacjm*vlcbm +
							vabli*vacjm*vlcmb + vabli*vacmj*vlcbm -
							2*vabli*vacmj*vlcmb)
						cij += vvv
						fij += vvv / (ea + ec - em - ei)
						fji += vvv / (ea + eb - el - ej)
					}
				}
			}
		}
	}

	// C_ij^(C)
	for a := no; a < nb; a++ {
		for b := no; b < nb; b++ {
			for k := range no {
				if e.so(a)^e.so(b)^e.so(k) != e.so(i) {
					continue
				}
				ea, eb, ek := ep[a], ep[b], ep[k]
				vabkj := e.v(a, b, k, j)
				vabjk := e.v(a, b, j, k)
				vabki := e.v(a, b, k, i)
				vabik := e.v(a, b, i, k)
				for l := range no {
					for m := range no {
						if e.so(l)^e.so(m) != e.so(a)^e.so(b) {
							continue
						}
						el, em := ep[l], ep[m]
						vablm := e.v(a, b, l, m)
						vabml := e.v(a, b, m, l)
						vlmki := e.v(l, m, k, i)
						vlmik := e.v(l, m, i, k)
						vlmkj := e.v(l, m, k, j)
						vlmjk := e.v(l, m, j, k)
						exe := (ea + eb - el - em) * (ea + eb - ek - ej)
						vvv := 0.25 / exe * (2*vablm*vabkj*vlmki -
							vablm*vabkj*vlmik - vablm*vabjk*vlmki +
							2*vablm*vabjk*vlmik - vabml*vabkj*vlmki +
							2*vabml*vabkj*vlmik + 2*vabml*vabjk*vlmki -
							vabml*vabjk*vlmik)
						cij += vvv
						fij += vvv / (ea + eb - ei - ek)
						// h.c. part
						exe2 := (ea + eb - el - em) * (ea + eb - ek - ei)
						vvv2 := 0.25 / exe2 * (2*vablm*vabki*vlmkj -
							vablm*vabki*vlmjk - vablm*vabik*vlmkj +
							2*vablm*vabik*vlmjk - vabml*vabki*vlmkj +
							2*vabml*vabki*vlmjk + 2*vabml*vabik*vlmkj -
							vabml*vabik*vlmjk)
						cij += vvv2
						fji += vvv2 / (ea + eb - ej - ek)
					}
				}
			}
		}
	}

	// C_ij^(D)
	for a := no; a < nb; a++ {
		for b := no; b < nb; b++ {
			for c := no; c < nb; c++ {
				for m := range no {
					if e.so(b)^e.so(c)^e.so(m) != e.so(i) {
						continue
					}
					ea, eb, ec, em := ep[a], ep[b], ep[c], ep[m]
					vbcjm := e.v(b, c, j, m)
					vbcmj := e.v(b, c, m, j)
					vbcim := e.v(b, c, i, m)
					vbcmi := e.v(b, c, m, i)
					for l := range no {
						if e.so(a)^e.so(c)^e.so(l) != e.so(i) {
							continue
						}
						el := ep[l]
						vablm := e.v(a, b, l, m)
						vabml := e.v(a, b, m, l)
						vlcia := e.v(l, c, i, a)
						vlcai := e.v(l, c, a, i)
						vlcja := e.v(l, c, j, a)
						vlcaj := e.v(l, c, a, j)
						exe := (ea + eb - el - em) * (eb + ec - em - ej)
						vvv := 1 / exe * (vablm*vbcjm*vlcia -
							2*vablm*vbcjm*vlcai - 2*vablm*vbcmj*vlcia +
							4*vablm*vbcmj*vlcai - 2*vabml*vbcjm*vlcia +
							vabml*vbcjm*vlcai + vabml*vbcmj*vlcia -
							2*vabml*vbcmj*vlcai)
						cij += vvv
						fij += vvv / (eb + ec - ei - em)
						// h.c. part
						exe2 := (ea + eb - el - em) * (eb + ec - em - ei)
						vvv2 := 1 / exe2 * (vablm*vbcim*vlcja -
							2*vablm*vbcim*vlcaj - 2*vablm*vbcmi*vlcja +
							4*vablm*vbcmi*vlcaj - 2*vabml*vbcim*vlcja +
							vabml*vbcim*vlcaj + vabml*vbcmi*vlcja -
							2*vabml*vbcmi*vlcaj)
						cij += vvv2
						fji += vvv2 / (eb + ec - ej - em)
					}
				}
			}
		}
	}
	return cij, fij, fji
}

// ---------------------------------------------------------------------------
// Main/satellite coupling: c12 = c12_1 (+ c12_2).
// ---------------------------------------------------------------------------

// c12 returns the coupling between main orbital j (absolute occupied index of the
// target irrep) and satellite config cfg.
func (e *elements) c12(j int, cfg Config) float64 {
	val := e.c12_1(j, cfg)
	if e.order >= 3 {
		val += e.c12_2(j, cfg)
	}
	return val
}

// c12_1 is the 1st-order coupling (calc_c12_1.c).
func (e *elements) c12_1(j int, cfg Config) float64 {
	k, l := cfg.Occ[0], cfg.Occ[1]
	a := e.nocc + cfg.Vir
	if k == l { // akk single
		return e.v(j, a, k, k)
	}
	if cfg.Typ == 0 { // spin I
		return sqrt1_2 * (e.v(j, a, k, l) + e.v(j, a, l, k))
	}
	return -sqrt3_2 * (e.v(j, a, k, l) - e.v(j, a, l, k)) // spin II
}

// vv1 implements the VV1 macro (calc_c12_2.c).
func (e *elements) vv1(k, l, a, j, b, c int, s1, s2 float64) float64 {
	return (e.v(k, l, b, c) + s1*e.v(k, l, c, b)) * (e.v(j, a, b, c) + s2*e.v(j, a, c, b))
}

// vv2 implements the VV2 macro (calc_c12_2.c).
func (e *elements) vv2(k, l, a, j, b, i int, s1, s2, s3, s4 float64) float64 {
	return s1*e.v(k, i, b, a)*e.v(j, i, b, l) +
		s2*e.v(k, i, b, a)*e.v(j, i, l, b) +
		s3*e.v(k, i, a, b)*e.v(j, i, b, l) +
		s4*2*e.v(k, i, a, b)*e.v(j, i, l, b)
}

// c12_2 is the 2nd-order coupling (calc_c12_2.c).
func (e *elements) c12_2(j int, cfg Config) float64 {
	ep := e.eps
	k, l := cfg.Occ[0], cfg.Occ[1]
	a := e.nocc + cfg.Vir
	ea := ep[a]
	no, nb := e.nocc, e.norb

	if k == l { // akk single
		ek := ep[k]
		var x0 float64
		for b := no; b < nb; b++ {
			eb := ep[b]
			for c := no; c < nb; c++ { // c_sym == b_sym
				if e.so(c) != e.so(b) {
					continue
				}
				x0 += 0.25 * e.vv1(k, k, a, j, b, c, +1, +1) / (2*ek - eb - ep[c])
			}
			for ii := range no { // i_sym = b_sym ⊗ k_sym ⊗ sym
				if e.so(ii) != e.so(b)^e.so(k)^e.sp.Sym {
					continue
				}
				x0 -= e.vv2(k, k, a, j, b, ii, +1, +1, +1, -1) / (ek + ep[ii] - ea - eb)
			}
		}
		return x0
	}

	ek, el := ep[k], ep[l]
	klSym := e.so(k) ^ e.so(l)
	sym := e.sp.Sym
	var x float64 // spin I (x0) or spin II (x1) depending on cfg.Typ
	for b := no; b < nb; b++ {
		eb := ep[b]
		for c := no; c < nb; c++ { // c_sym = b_sym ⊗ kl_sym
			if e.so(b)^e.so(c) != klSym {
				continue
			}
			den := ek + el - eb - ep[c]
			if cfg.Typ == 0 {
				x += sqrt1_8 * e.vv1(k, l, a, j, b, c, +1, +1) / den
			} else {
				x -= sqrt3_8 * e.vv1(k, l, a, j, b, c, -1, -1) / den
			}
		}
		for ii := range no { // i_sym_kl = b_sym ⊗ l_sym ⊗ sym
			if e.so(ii) != e.so(b)^e.so(l)^sym {
				continue
			}
			den := ek + ep[ii] - ea - eb
			if cfg.Typ == 0 {
				x -= sqrt1_2 * e.vv2(k, l, a, j, b, ii, +1, +1, +1, -1) / den
			} else {
				x += sqrt3_2 * e.vv2(k, l, a, j, b, ii, +1, -1, -1, +1) / den
			}
		}
		for ii := range no { // (l<->k): i_sym_lk = b_sym ⊗ k_sym ⊗ sym
			if e.so(ii) != e.so(b)^e.so(k)^sym {
				continue
			}
			den := el + ep[ii] - ea - eb
			if cfg.Typ == 0 {
				x -= sqrt1_2 * e.vv2(l, k, a, j, b, ii, +1, +1, +1, -1) / den
			} else {
				x -= sqrt3_2 * e.vv2(l, k, a, j, b, ii, +1, -1, -1, +1) / den
			}
		}
	}
	return x
}

// ---------------------------------------------------------------------------
// Satellite 2h1p/2h1p block: diagonal (k2 + c22_1_diag) + off-diagonal (c22_1).
// ---------------------------------------------------------------------------

// c22diag returns the diagonal 2h1p element of cfg (k2 + calc_c22_1_diag).
func (e *elements) c22diag(cfg Config) float64 {
	ep := e.eps
	k, l := cfg.Occ[0], cfg.Occ[1]
	a := e.nocc + cfg.Vir
	ek, ea := ep[k], ep[a]

	if k == l { // akk single
		diag := ea - 2*ek
		off := e.v(k, k, k, k) -
			e.v(a, k, a, k) + 0.5*e.v(a, k, k, a) -
			e.v(a, k, a, k) + 0.5*e.v(a, k, k, a)
		return diag + off
	}
	el := ep[l]
	diag := ea - ek - el
	var off float64
	if cfg.Typ == 0 { // spin I
		off = e.v(k, l, k, l) + e.v(k, l, l, k) -
			e.v(a, l, a, l) + 0.5*e.v(a, l, l, a) -
			e.v(a, k, a, k) + 0.5*e.v(a, k, k, a)
	} else { // spin II
		off = e.v(k, l, k, l) - e.v(k, l, l, k) -
			e.v(a, l, a, l) + 1.5*e.v(a, l, l, a) -
			e.v(a, k, a, k) + 1.5*e.v(a, k, k, a)
	}
	return diag + off
}

// deltaV implements the DELTA_V_TERM macro of calc_c22_1_cols: when hole K equals
// hole M, add the spin-block contributions with the given ± signs and prefactor.
func (e *elements) deltaV(x00, x01, x10, x11 *float64, K, M, A, N, B, L int, s00, s01, s10, s11, pf float64) {
	if K != M {
		return
	}
	v1 := e.v(A, N, B, L)
	v2 := e.v(A, N, L, B)
	*x00 += s00 * pf * (-v1 + 0.5*v2)
	*x01 -= s01 * pf * (sqrt3_4 * v2)
	*x10 -= s10 * pf * (sqrt3_4 * v2)
	*x11 += s11 * pf * (-v1 + 1.5*v2)
}

// c22off returns the off-diagonal 1st-order coupling between two distinct 2h1p
// configs (calc_c22_1_off.c / calc_c22_1_cols). row = (k,l,a), col = (m,n,b).
func (e *elements) c22off(row, col Config) float64 {
	k, l, a := row.Occ[0], row.Occ[1], e.nocc+row.Vir
	m, n, b := col.Occ[0], col.Occ[1], e.nocc+col.Vir
	pf := 1.0
	if m == n {
		pf = sqrt1_2
	}
	var x00, x01, x10, x11 float64

	if k == l { // akk row branch
		if a == b {
			x00 += pf * 2 * e.v(m, n, k, k)
		}
		e.deltaV(&x00, &x01, &x10, &x11, k, m, a, n, b, k, +1, +1, +1, +1, pf)
		e.deltaV(&x00, &x01, &x10, &x11, k, n, a, m, b, k, +1, -1, -1, +1, pf)
		e.deltaV(&x00, &x01, &x10, &x11, k, m, a, n, b, k, +1, -1, +1, -1, pf)
		e.deltaV(&x00, &x01, &x10, &x11, k, n, a, m, b, k, +1, +1, -1, -1, pf)
		x00 *= sqrt1_2
		x10 *= sqrt1_2
		// row single couples to col spin I via x00, spin II via x10.
		if m == n || col.Typ == 0 {
			return x00
		}
		return x10
	}

	// akl row branch (k != l)
	if a == b {
		vmnkl := e.v(m, n, k, l)
		vmnlk := e.v(m, n, l, k)
		x00 += pf * (vmnkl + vmnlk)
		x11 += pf * (vmnkl - vmnlk)
	}
	e.deltaV(&x00, &x01, &x10, &x11, k, m, a, n, b, l, +1, +1, +1, +1, pf)
	e.deltaV(&x00, &x01, &x10, &x11, l, n, a, m, b, k, +1, -1, -1, +1, pf)
	e.deltaV(&x00, &x01, &x10, &x11, l, m, a, n, b, k, +1, -1, +1, -1, pf)
	e.deltaV(&x00, &x01, &x10, &x11, k, n, a, m, b, l, +1, +1, -1, -1, pf)

	if m == n { // col single: only spin-I column, rows I->x00, II->x01
		if row.Typ == 0 {
			return x00
		}
		return x01
	}
	switch {
	case row.Typ == 0 && col.Typ == 0:
		return x00
	case row.Typ == 1 && col.Typ == 0:
		return x01
	case row.Typ == 0 && col.Typ == 1:
		return x10
	default:
		return x11
	}
}
