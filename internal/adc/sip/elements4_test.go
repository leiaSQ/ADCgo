package sip

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// h2oElements builds an elements engine on the symmetry-off H2O fixture. When
// core is non-nil the underlying Space is the CVS ADC(4) space (order 4).
func h2oElements(t *testing.T, order int) *elements {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	sp := NewSpace(nocc, d.NORB, nil, 0)
	return newElements(sp, ints, eps, order)
}

// h2oElements4 builds an elements engine whose Space is the CVS ADC(4) space with
// the given core set (needed by kopp4's non-core hole loop).
func h2oElements4(t *testing.T, core []int) (*elements, *Space) {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	sp := NewSpace4(nocc, d.NORB, nil, 0, core)
	return newElements(sp, ints, eps, 4), sp
}

// TestKopp1Transcription checks kopp1 against a direct re-transcription of the F77
// SPIN-table contraction (kopp1.F): the independent oracle for the port.
func TestKopp1Transcription(t *testing.T) {
	e := h2oElements(t, 4)
	const eps = 1e-13
	// reference SPIN(2,2), column-major: SPIN(row,col).
	spin := [2][2]float64{{sqrt1_2, sqrt3_2}, {sqrt1_2, -sqrt3_2}}
	faktor := [2]float64{sqrt1_2, 1} // FAKTOR(MAXS): MAXS=1 -> 1/√2, MAXS=2 -> 1

	p := 0 // core hole (O 1s)
	check := func(k, l, aPos, typ int) {
		cfg := Config{Occ: [2]int{k, l}, Vir: aPos, Typ: typ}
		a := e.nocc + aPos
		a1, a2 := e.v(p, a, k, l), e.v(p, a, l, k)
		maxs := 2
		if k == l {
			maxs = 1
		}
		// F77: SUM(MS) = FKL*(A1*SPIN(1,MS) + A2*SPIN(2,MS)), MS = typ+1 (1-based).
		ms := typ
		want := faktor[maxs-1] * (a1*spin[0][ms] + a2*spin[1][ms])
		got := e.kopp1(p, cfg)
		if math.Abs(got-want) > eps {
			t.Errorf("kopp1(k=%d,l=%d,a=%d,typ=%d) = %.15g, want %.15g", k, l, aPos, typ, got, want)
		}
	}
	check(1, 2, 0, 0) // k!=l, spin I
	check(1, 2, 0, 1) // k!=l, spin II
	check(2, 4, 3, 0)
	check(2, 4, 3, 1)
	check(3, 3, 5, 0) // k==l single spin function
}

// TestKopp1EqualsC12_1UpToSpinPhase pins the Dyson-vs-non-Dyson relationship: the
// KOPP1 coupling equals the non-Dyson c12_1 for spin I and the K==L single, and is
// its negation for spin II (a basis phase). Guards against convention drift.
func TestKopp1EqualsC12_1UpToSpinPhase(t *testing.T) {
	e := h2oElements(t, 4)
	const eps = 1e-13
	p := 0
	cases := []Config{
		{Occ: [2]int{1, 2}, Vir: 0, Typ: 0},
		{Occ: [2]int{1, 2}, Vir: 0, Typ: 1},
		{Occ: [2]int{2, 4}, Vir: 7, Typ: 0},
		{Occ: [2]int{2, 4}, Vir: 7, Typ: 1},
		{Occ: [2]int{3, 3}, Vir: 5, Typ: 0},
	}
	for _, cfg := range cases {
		got := e.kopp1(p, cfg)
		ref := e.c12_1(p, cfg)
		want := ref
		if cfg.Occ[0] != cfg.Occ[1] && cfg.Typ == 1 {
			want = -ref // spin-II phase flip
		}
		if math.Abs(got-want) > eps {
			t.Errorf("kopp1 %v = %.15g, want %.15g (c12_1=%.15g)", cfg, got, want, ref)
		}
	}
}

// TestKopp2Finite checks kopp2 is finite over all H2O CVS 2h1p configs (guards
// against divide-by-zero denominators / index errors). Absolute correctness awaits
// the Hermiticity/dense oracle and the reference dump.
func TestKopp2Finite(t *testing.T) {
	e, sp := h2oElements4(t, []int{0})
	p := 0
	n := 0
	for idx := sp.BeginSat; idx < sp.Begin3h2p; idx++ {
		cfg := sp.Configs[idx]
		v := e.kopp2(p, cfg)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("cfg %d %+v: kopp2 = %v (non-finite)", idx, cfg, v)
		}
		n++
	}
	if n == 0 {
		t.Fatal("no 2h1p configs")
	}
}

// TestC22Elem4Hermitian checks the 2h1p block is symmetric — c22elem4(r,c) for
// (r.Typ,c.Typ) equals c22elem4(c,r) for (c.Typ,r.Typ) — and finite, over CVS 2h1p
// configs. Hermiticity is a genuine correctness property (integral bra/ket swap).
func TestC22Elem4Hermitian(t *testing.T) {
	e, sp := h2oElements4(t, []int{0})
	const eps = 1e-12
	cfgs := sp.Configs[sp.BeginSat:sp.Begin3h2p]
	// Cap the pair count for speed; step through a representative subset.
	step := 1
	if len(cfgs) > 60 {
		step = len(cfgs) / 60
	}
	for a := 0; a < len(cfgs); a += step {
		for b := 0; b < len(cfgs); b += step {
			ra, rb := cfgs[a], cfgs[b]
			x := e.c22elem4(ra, rb)
			y := e.c22elem4(rb, ra)
			if math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatalf("c22elem4(%+v,%+v) non-finite: %v", ra, rb, x)
			}
			if math.Abs(x-y) > eps {
				t.Fatalf("not symmetric: c22elem4(a,b)=%.15g c22elem4(b,a)=%.15g", x, y)
			}
		}
	}
}

// TestKopp4RowBoundsAndFinite checks the kopp4 spin-table row index stays in
// [0,12] and every coupling is finite over all H2O CVS 3h2p configs (guards
// against divide-by-zero denominators and coeff-table out-of-range).
func TestKopp4RowBoundsAndFinite(t *testing.T) {
	e, sp := h2oElements4(t, []int{0})
	if len(sp.Sat3) == 0 {
		t.Fatal("no 3h2p configs")
	}
	p := 0 // O 1s core hole (target irrep)
	for idx, cfg := range sp.Sat3 {
		r := cfg.Spin + ns3(cfg) - 1
		if r < 0 || r >= 13 {
			t.Fatalf("cfg %d: coeff row %d out of [0,13)", idx, r)
		}
		v := e.kopp4(p, cfg)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("cfg %d %+v: kopp4 = %v (non-finite)", idx, cfg, v)
		}
	}
}

// TestKopp4Retranscription recomputes kopp4 with an independently structured loop
// (accumulating the two contributions separately) to catch column-index / role
// typos in the port. This is a transcription cross-check, not an absolute-value
// gate (that awaits the reference dump).
func TestKopp4Retranscription(t *testing.T) {
	e, sp := h2oElements4(t, []int{0})
	const eps = 1e-12
	so := e.so
	ep := e.eps
	p := 0
	// reference column groups (1-based): KKK->{5,6,7,8},{17,18,19,20};
	// KK->{25,26,27,28},{33,34,35,36}.
	for idx, cfg := range sp.Sat3 {
		i, j := e.nocc+cfg.I, e.nocc+cfg.J
		k, l, m := cfg.Core, cfg.L, cfg.M
		r := cfg.Spin + ns3(cfg) - 1
		var want float64
		for kkk := e.nocc; kkk < e.norb; kkk++ {
			d3 := so(l)^so(m) == so(i)^so(kkk)
			d9 := so(l)^so(m) == so(j)^so(kkk)
			d15 := so(j)^so(k) == so(p)^so(kkk)
			d21 := so(i)^so(k) == so(p)^so(kkk)
			var a3, a4, a9, a10, a15, a16, a21, a22 float64
			if d3 {
				a3, a4 = e.v(l, m, i, kkk), e.v(l, m, kkk, i)
			}
			if d9 {
				a9, a10 = e.v(l, m, j, kkk), e.v(l, m, kkk, j)
			}
			if d15 {
				a15, a16 = e.v(k, kkk, p, j), e.v(k, kkk, j, p)
			}
			if d21 {
				a21, a22 = e.v(k, kkk, p, i), e.v(k, kkk, i, p)
			}
			e2 := ep[i] + ep[kkk] - ep[l] - ep[m]
			e5 := ep[j] + ep[kkk] - ep[l] - ep[m]
			want += (a3*coeff0[r][4] + a4*coeff0[r][5]) * a15 / e2
			want += (a3*coeff0[r][6] + a4*coeff0[r][7]) * a16 / e2
			want += (a9*coeff0[r][16] + a10*coeff0[r][17]) * a21 / e5
			want += (a9*coeff0[r][18] + a10*coeff0[r][19]) * a22 / e5
		}
		for kk := range e.nocc {
			if sp.isCore(kk) {
				continue
			}
			var a1, a2, a5, a6, a7, a8, a11, a12 float64
			if so(i)^so(j) == so(m)^so(kk) {
				a1, a2 = e.v(m, kk, i, j), e.v(m, kk, j, i)
			}
			if so(i)^so(j) == so(l)^so(kk) {
				a5, a6 = e.v(l, kk, i, j), e.v(l, kk, j, i)
			}
			if so(k)^so(l) == so(p)^so(kk) {
				a7, a8 = e.v(k, l, p, kk), e.v(k, l, kk, p)
			}
			if so(k)^so(m) == so(p)^so(kk) {
				a11, a12 = e.v(k, m, p, kk), e.v(k, m, kk, p)
			}
			e1 := ep[i] + ep[j] - ep[m] - ep[kk]
			e3 := ep[i] + ep[j] - ep[l] - ep[kk]
			want += (a1*coeff0[r][24] + a2*coeff0[r][25]) * a7 / e1
			want += (a1*coeff0[r][26] + a2*coeff0[r][27]) * a8 / e1
			want += (a5*coeff0[r][32] + a6*coeff0[r][33]) * a11 / e3
			want += (a5*coeff0[r][34] + a6*coeff0[r][35]) * a12 / e3
		}
		if got := e.kopp4(p, cfg); math.Abs(got-want) > eps {
			t.Fatalf("cfg %d: kopp4 = %.15g, retranscription = %.15g", idx, got, want)
		}
	}
}
