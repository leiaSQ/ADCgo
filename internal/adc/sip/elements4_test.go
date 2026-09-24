package sip

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
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

// elements4_test.go — the A1-sector matched-integral gate. A1 contains the core
// hole, so its matrix has the 1h main block absent from B2, exercising the 1h
// couplings KOPP1/KOPP2 (1h↔2h1p), KOPP4 (1h↔3h2p) and the 1h diagonal. Reference
// tape: testdata/reference/adc4_a1_tape (see README). Complements TestADC4MatchedGate.
//
// Bit-exact (asserted): 2h1p/2h1p (WERT1), 2h1p↔3h2p (WERT2, multiset), 1h↔3h2p (KOPP4,
// multiset) and 1h↔2h1p (KOPP1/2/3, element-wise).
//
// The 1h diagonal is −ε_core − Σ with Σ the external static self-energy (&self-energy
// infinite, computed outside the ADC4 matrix). Here Σ is still *back-solved* from the tape
// entry, which only exercises the SetStaticSelfEnergy wiring, not the value —
// TestADC4StaticSigmaGate gates the value itself against theADCcode's Σ dump.
func TestADC4MatchedGateA1(t *testing.T) {
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	dir := filepath.Join("..", "..", "..", "testdata", "reference", "adc4_a1_tape")
	if _, err := os.Stat(filepath.Join(dir, "FT21F001.ADC")); err != nil {
		t.Skipf("A1 reference tape unavailable: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace4(nocc, d.NORB, d.OrbSym, 0, []int{0}) // A1, core = orb 0
	if sp.Size() != 1712 || sp.BeginSat != 1 || sp.Begin3h2p-sp.BeginSat != 46 {
		t.Fatalf("A1 space = (size %d, 1h %d, 2h1p %d), want (1712, 1, 46)",
			sp.Size(), sp.BeginSat, sp.Begin3h2p-sp.BeginSat)
	}
	mx := New(sp, integrals.New(d, nocc, d.OrbSym), eps, 4, backend.Gonum{})
	M := mx.BuildMatrix()
	n := sp.Size()
	b1, b2 := sp.BeginSat, sp.Begin3h2p

	Ref := make([][]float64, n)
	for i := range Ref {
		Ref[i] = make([]float64, n)
	}
	rows, cols, vals := readTapeOff(t, filepath.Join(dir, "FT21F001.ADC"))
	for k := range rows {
		i, j := rows[k]-1, cols[k]-1
		Ref[i][j], Ref[j][i] = vals[k], vals[k]
	}
	for i, dv := range readTapeDiag(t, filepath.Join(dir, "FT18F001.ADC")) {
		Ref[i][i] = dv
	}

	// bit-exact blocks.
	if md := blockMaxDiff(Ref, M, b1, b2, b1, b2); md > 1e-12 {
		t.Errorf("2h1p/2h1p block maxdiff %g exceeds 1e-12", md)
	}
	if md := multisetMaxDiff(Ref, M, b1, b2, b2, n); md > 1e-12 {
		t.Errorf("2h1p↔3h2p multiset maxdiff %g exceeds 1e-12", md)
	}
	if md := multisetMaxDiff(Ref, M, 0, b1, b2, n); md > 1e-12 {
		t.Errorf("1h↔3h2p (KOPP4) multiset maxdiff %g exceeds 1e-12", md)
	}
	// 1h↔2h1p: kopp1+kopp2 (2nd+3rd) + kopp3 (4th, K2P2H+K1P3H+K3P1H) — bit-exact.
	if md := blockMaxDiff(Ref, M, 0, b1, b1, b2); md > 1e-12 {
		t.Errorf("1h↔2h1p (KOPP1/2/3) maxdiff %g exceeds 1e-12", md)
	}
	// 1h diagonal. Bare block is −ε_core (no static self-energy).
	p0 := sp.Configs[0].Occ[0]
	if md := math.Abs(M.At(0, 0) - (-eps[p0])); md > 1e-12 {
		t.Errorf("bare 1h diagonal %g != −ε_core %g", M.At(0, 0), -eps[p0])
	}
	// theADCcode folds in an external static self-energy Σ (&self-energy infinite):
	// diag = −ε_core − Σ. Verify the pluggable SetStaticSelfEnergy wiring reproduces the
	// tape diagonal given that Σ (the value itself is the self-energy module's output).
	sig := -eps[p0] - Ref[0][0] // ≈ −0.0116 a.u.
	mx.SetStaticSelfEnergy(func(i, j int) float64 {
		if i == p0 && j == p0 {
			return sig
		}
		return 0
	})
	if d := mx.BuildMatrix().At(0, 0) - Ref[0][0]; math.Abs(d) > 1e-12 {
		t.Errorf("with static Σ, 1h diagonal off by %g", d)
	}
	t.Logf("A1 gate: KOPP1/2/3 + KOPP4 bit-exact; external static Σ(core)=%.5f a.u. reproduces the 1h diagonal", sig)
}

func blockMaxDiff(Ref [][]float64, M backend.Mat, r0, r1, c0, c1 int) float64 {
	var md float64
	for i := r0; i < r1; i++ {
		for j := c0; j < c1; j++ {
			if dd := math.Abs(Ref[i][j] - M.At(i, j)); dd > md {
				md = dd
			}
		}
	}
	return md
}

func multisetMaxDiff(Ref [][]float64, M backend.Mat, r0, r1, c0, c1 int) float64 {
	var md float64
	for i := r0; i < r1; i++ {
		var rv, mv []float64
		for j := c0; j < c1; j++ {
			if Ref[i][j] != 0 {
				rv = append(rv, Ref[i][j])
			}
			if v := M.At(i, j); v != 0 {
				mv = append(mv, v)
			}
		}
		if len(rv) != len(mv) {
			return math.Inf(1)
		}
		sort.Float64s(rv)
		sort.Float64s(mv)
		for k := range rv {
			if dd := math.Abs(rv[k] - mv[k]); dd > md {
				md = dd
			}
		}
	}
	return md
}

// elements4_test.go — the two references that used to be missing from the tapes.
//
// theADCcode discards both before it writes anything a consumer could read: RSCRT1
// rewrites the diagonal tape (FT18) with only the first nh12 entries, throwing away the
// 3h2p effective diagonal, and the static self-energy is an *input* to adc_() that no
// tape records. ../ADC now dumps them (ab5.F -> FT19F001.ADC; egf.F -> SIGMA_STATIC.dat),
// which turns both of these from self-consistency checks into bit-exact value gates.
// FT21/FT18 are unchanged by that instrumentation.

// eigabTape is FT19F001.ADC: record 1 = IDIM NCOL NECORE N3H2P (4x int32),
// record 2 = N3H2P x float64, the 3h2p effective diagonal in ab5's pam/ELIM column
// order (same permutation FT21's 3h2p columns carry).
type eigabTape struct {
	idim, ncol, necore int
	diag               []float64
}

func readTapeEigab(t *testing.T, fn string) eigabTape {
	t.Helper()
	d, err := os.ReadFile(fn)
	if err != nil {
		t.Fatal(err)
	}
	le := binary.LittleEndian
	ln := int(int32(le.Uint32(d)))
	hdr := d[4 : 4+ln]
	out := eigabTape{
		idim:   int(int32(le.Uint32(hdr[0:]))),
		ncol:   int(int32(le.Uint32(hdr[4:]))),
		necore: int(int32(le.Uint32(hdr[8:]))),
	}
	n3 := int(int32(le.Uint32(hdr[12:])))
	off := 4 + ln + 4
	ln = int(int32(le.Uint32(d[off:])))
	off += 4
	if ln/8 != n3 {
		t.Fatalf("FT19 %s: header says %d 3h2p entries, record holds %d", fn, n3, ln/8)
	}
	out.diag = make([]float64, n3)
	for i := range out.diag {
		out.diag[i] = math.Float64frombits(le.Uint64(d[off+i*8:]))
	}
	return out
}

// TestADC4EigabGate is the bit-exact value gate for the 3h2p effective diagonal (WERT3).
// theADCcode's EIGAB is the 0th-order orbital-energy sum plus the 5th-order 3h2p-CI
// diagonal correction; ADCgo's sat3Diag reproduces it when WERT3 is on. The reference
// permutes 3h2p columns (ab5 pam/ELIM), so — exactly as the WERT2 coupling is compared —
// the diagonal is compared as a sorted multiset. ELIM's center-of-gravity fold is a
// truncation device inactive below MAXSTA (=10000), and both sectors are far under it,
// so the fold is a no-op here and the two must agree elementwise up to that permutation.
func TestADC4EigabGate(t *testing.T) {
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)

	for _, tc := range []struct {
		name            string
		dir             string
		sym             int
		size, n2h1p, n3 int
	}{
		{"A1", "adc4_a1_tape", 0, 1712, 46, 1665},
		{"B2", "adc4_b2_tape", 3, 1688, 42, 1646},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fn := filepath.Join("..", "..", "..", "testdata", "reference", tc.dir, "FT19F001.ADC")
			if _, err := os.Stat(fn); err != nil {
				t.Skipf("EIGAB tape unavailable: %v", err)
			}
			ref := readTapeEigab(t, fn)
			if ref.idim != tc.size || len(ref.diag) != tc.n3 {
				t.Fatalf("tape dims = (idim %d, 3h2p %d), want (%d, %d)",
					ref.idim, len(ref.diag), tc.size, tc.n3)
			}
			// ncol is the 2h1p count; necore the 1h count. Guards that we loaded the
			// sector we think we did.
			if ref.ncol != tc.n2h1p {
				t.Fatalf("tape 2h1p count = %d, want %d", ref.ncol, tc.n2h1p)
			}

			sp := NewSpace4(nocc, d.NORB, d.OrbSym, tc.sym, []int{0})
			if sp.Size() != tc.size {
				t.Fatalf("space size = %d, want %d", sp.Size(), tc.size)
			}
			mx := New(sp, integrals.New(d, nocc, d.OrbSym), eps, 4, backend.Gonum{})
			mx.SetWert3(true)
			got := mx.sat3Diag()
			if len(got) != len(ref.diag) {
				t.Fatalf("3h2p diagonal length = %d, want %d", len(got), len(ref.diag))
			}

			rv := append([]float64(nil), ref.diag...)
			mv := append([]float64(nil), got...)
			sort.Float64s(rv)
			sort.Float64s(mv)
			var maxd float64
			for i := range rv {
				if dd := math.Abs(rv[i] - mv[i]); dd > maxd {
					maxd = dd
				}
			}
			if maxd > 1e-12 {
				t.Errorf("3h2p effective diagonal (EIGAB/WERT3) multiset max diff %g exceeds 1e-12", maxd)
			}
			t.Logf("EIGAB gate %s: %d 3h2p entries, multiset maxdiff=%.2e", tc.name, len(rv), maxd)
		})
	}
}

// readSigmaStatic parses SIGMA_STATIC.dat (egf.F): a count line NECORE, then one row
// per lower-triangle element "i j ni nj value" with the value in a.u. — Σ as actually
// applied to the 1h block, i.e. after egf.F's own eV→a.u. conversion (FAKTOR =
// 27.211606, not the 27.211396 the C++ caller used on the way in) and after the SIGMPH
// 2p1h contribution. Reconstructing Σ from the C++ input instead leaves an ~9e-8 residual.
func readSigmaStatic(t *testing.T, fn string) map[[2]int]float64 {
	t.Helper()
	f, err := os.Open(fn)
	if err != nil {
		t.Skipf("static self-energy dump unavailable: %v", err)
	}
	defer f.Close()
	var n int
	if _, err := fmt.Fscan(f, &n); err != nil {
		t.Fatalf("SIGMA_STATIC.dat: bad header: %v", err)
	}
	out := make(map[[2]int]float64)
	for range n * (n + 1) / 2 {
		var i, j, ni, nj int
		var v float64
		if _, err := fmt.Fscan(f, &i, &j, &ni, &nj, &v); err != nil {
			t.Fatalf("SIGMA_STATIC.dat: bad row: %v", err)
		}
		out[[2]int{ni - 1, nj - 1}] = v // to 0-based absolute orbital indices
		out[[2]int{nj - 1, ni - 1}] = v // symmetric
	}
	return out
}

// TestADC4StaticSigmaGate closes the static self-energy on its real value. The 1h
// diagonal of the A1 tape is −ε_core − Σ(∞), with Σ supplied to adc_() from theADCcode's
// self-energy module (&self-energy infinite) and previously unavailable — the A1 gate had
// to *back-solve* Σ from the very tape entry it then checked, which asserts nothing about
// the value. Here Σ is read from theADCcode's own dump and must reproduce the tape.
func TestADC4StaticSigmaGate(t *testing.T) {
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	dir := filepath.Join("..", "..", "..", "testdata", "reference", "adc4_a1_tape")
	sig := readSigmaStatic(t, filepath.Join(dir, "SIGMA_STATIC.dat"))

	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace4(nocc, d.NORB, d.OrbSym, 0, []int{0}) // A1, core = orb 0
	mx := New(sp, integrals.New(d, nocc, d.OrbSym), eps, 4, backend.Gonum{})

	sigma := sig[[2]int{0, 0}]
	if sigma == 0 {
		t.Fatal("no Σ(0,0) in the dump — core orbital mismatch")
	}
	mx.SetStaticSelfEnergy(func(i, j int) float64 { return sig[[2]int{i, j}] })

	want := readTapeDiag(t, filepath.Join(dir, "FT18F001.ADC"))[0]
	got := mx.BuildMatrix().At(0, 0)
	if dd := math.Abs(got - want); dd > 1e-12 {
		t.Errorf("1h diagonal with theADCcode's Σ = %.15g, tape = %.15g (diff %g)", got, want, dd)
	}
	// Bare −ε_core must NOT reproduce the tape, or the gate is vacuous.
	if math.Abs(-eps[0]-want) < 1e-6 {
		t.Fatal("bare −ε_core already matches the tape — Σ is not being exercised")
	}
	t.Logf("static Σ gate: Σ(1,1) = %.12g Ha, 1h diagonal matches tape to %.2e", sigma, math.Abs(got-want))
}

// elements4_test.go — the matched-integral gate for CVS IP-ADC(4) (Track A / A2.6).
//
// It builds the ADC(4) B2 secular matrix on the DZP integrals that theADCcode itself
// used, and compares against theADCcode's own matrix tape (testdata/reference/
// adc4_b2_tape, see its README). B2 has no core hole in its irrep, so the matrix is
// 42 (2h1p) + 1646 (3h2p) with no 1h main block: the tape exercises the 2h1p/2h1p
// block (WERT1, 3rd+4th order) and the 2h1p<->3h2p coupling (WERT2). The reference
// reorders 3h2p columns internally (ab5.F pam/ELIM, eigenvalue-invariant), so the
// coupling is compared per-row by sorted multiset. The 3h2p effective diagonal
// (EIGAB) is not on the tape; the 1h KOPP couplings need an A1 tape — both are out of
// scope for this fixture.

const tapeDir = "../../../testdata/reference/adc4_b2_tape"

// TestADC4MatchedGate is bit-exact against theADCcode on matched integrals.
func TestADC4MatchedGate(t *testing.T) {
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	off := filepath.Join(tapeDir, "FT21F001.ADC")
	dia := filepath.Join(tapeDir, "FT18F001.ADC")
	if _, err := os.Stat(off); err != nil {
		t.Skipf("reference tape unavailable: %v", err)
	}

	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace4(nocc, d.NORB, d.OrbSym, 3, []int{0}) // B2 (0-based 3), core = orb 0
	if got, want := sp.Size(), 1688; got != want {
		t.Fatalf("B2 dimension = %d, want %d", got, want)
	}
	if n2 := sp.Begin3h2p - sp.BeginSat; n2 != 42 {
		t.Fatalf("2h1p count = %d, want 42", n2)
	}
	mx := New(sp, integrals.New(d, nocc, d.OrbSym), eps, 4, backend.Gonum{})
	M := mx.BuildMatrix()
	n := sp.Size()
	b := sp.Begin3h2p

	// Reference matrix from the tape (symmetric): off-diagonals + 2h1p diagonal.
	Ref := make([][]float64, n)
	for i := range Ref {
		Ref[i] = make([]float64, n)
	}
	rows, cols, vals := readTapeOff(t, off)
	for k := range rows {
		i, j := rows[k]-1, cols[k]-1
		Ref[i][j], Ref[j][i] = vals[k], vals[k]
	}
	for i, dv := range readTapeDiag(t, dia) {
		Ref[i][i] = dv
	}

	// 2h1p/2h1p block: rows and cols unpermuted -> element-wise bit-exact.
	var maxd float64
	for i := range b {
		for j := range b {
			if dd := math.Abs(Ref[i][j] - M.At(i, j)); dd > maxd {
				maxd = dd
			}
		}
	}
	if maxd > 1e-12 {
		t.Errorf("2h1p/2h1p block max diff %g exceeds 1e-12 (WERT1 3rd+4th order)", maxd)
	}

	// 2h1p<->3h2p coupling: reference permutes 3h2p columns (pam), so compare each
	// 2h1p row's coupling multiset to the 3h2p space.
	var worstMS float64
	for i := range b {
		var rv, mv []float64
		for j := b; j < n; j++ {
			if Ref[i][j] != 0 {
				rv = append(rv, Ref[i][j])
			}
			if v := M.At(i, j); v != 0 {
				mv = append(mv, v)
			}
		}
		if len(rv) != len(mv) {
			t.Fatalf("row %d: coupling nnz ref=%d mine=%d", i, len(rv), len(mv))
		}
		sort.Float64s(rv)
		sort.Float64s(mv)
		for k := range rv {
			if dd := math.Abs(rv[k] - mv[k]); dd > worstMS {
				worstMS = dd
			}
		}
	}
	if worstMS > 1e-12 {
		t.Errorf("2h1p<->3h2p coupling multiset max diff %g exceeds 1e-12 (WERT2)", worstMS)
	}

	// 3h2p block carries no off-diagonal in the reference (diagonal-only).
	for i := b; i < n; i++ {
		for j := b; j < n; j++ {
			if i != j && Ref[i][j] != 0 {
				t.Fatalf("unexpected reference 3h2p off-diagonal at (%d,%d)=%g", i, j, Ref[i][j])
			}
		}
	}
	t.Logf("matched gate: 2h1p block maxdiff=%.2e, coupling multiset maxdiff=%.2e", maxd, worstMS)
}

func readTapeOff(t *testing.T, fn string) (rows, cols []int, vals []float64) {
	t.Helper()
	d, err := os.ReadFile(fn)
	if err != nil {
		t.Fatal(err)
	}
	le := binary.LittleEndian
	for off := 0; off < len(d); {
		ln := int(int32(le.Uint32(d[off:])))
		off += 4
		body := d[off : off+ln]
		off += ln + 4
		cnt := int(int32(le.Uint32(body[16000:])))
		for k := range cnt {
			vals = append(vals, math.Float64frombits(le.Uint64(body[k*8:])))
			rows = append(rows, int(int32(le.Uint32(body[8000+k*4:]))))
			cols = append(cols, int(int32(le.Uint32(body[12000+k*4:]))))
		}
	}
	return
}

func readTapeDiag(t *testing.T, fn string) []float64 {
	t.Helper()
	d, err := os.ReadFile(fn)
	if err != nil {
		t.Fatal(err)
	}
	le := binary.LittleEndian
	ln := int(int32(le.Uint32(d))) // header record
	off := 4 + ln + 4
	ln = int(int32(le.Uint32(d[off:])))
	off += 4
	out := make([]float64, ln/8)
	for i := range out {
		out[i] = math.Float64frombits(le.Uint64(d[off+i*8:]))
	}
	return out
}
