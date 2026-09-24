package ip

import (
	"math"
	"path/filepath"
	"sort"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// types_test.go (hand-written; survives regeneration): the adcgen-generated
// IP-ADC evaluators against the hand-ported package sip.

func h2o(t *testing.T) refSystem {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join("..", "..", "..", "..", "testdata", "h2o.fcidump"))
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	return refSystem{d, nocc, mp.OrbitalEnergies(d, nocc), integrals.New(d, nocc, nil)}
}

// containsAll reports the largest distance from an eigenvalue of want to the nearest
// eigenvalue of got (both sorted).
func containsAll(got, want []float64) float64 {
	var worst float64
	for _, w := range want {
		k := sort.SearchFloat64s(got, w)
		d := math.Inf(1)
		for _, c := range []int{k - 1, k} {
			if c >= 0 && c < len(got) {
				d = min(d, math.Abs(got[c]-w))
			}
		}
		worst = max(worst, d)
	}
	return worst
}

func spinOrbitalSpectrum(t *testing.T, sys refSystem, scheme string, maxClass int) ([]float64, *khci.Space, backend.Mat) {
	t.Helper()
	el, err := New(sys.ints, sys.eps, sys.nocc, scheme)
	if err != nil {
		t.Fatal(err)
	}
	sp, err := khci.NewSpace(khci.Options{K: 1, NOcc: sys.nocc, NVir: sys.d.NORB - sys.nocc,
		TwoMs: 1, MaxClass: maxClass})
	if err != nil {
		t.Fatal(err)
	}
	M := el.BuildDense(sp)
	w, _ := backend.Gonum{}.SymEig(M)
	sort.Float64s(w)
	return w, sp, M
}

// TestMainBlockMatchesSipOrder2: through second order C11 is the same quantity in the
// strict ISR and in the ported ndadc3. Beta holes are the Ms = +1/2 doublet components.
func TestMainBlockMatchesSipOrder2(t *testing.T) {
	sys := h2o(t)
	el, err := New(sys.ints, sys.eps, sys.nocc, "adc22m") // C11 through order 2
	if err != nil {
		t.Fatal(err)
	}
	sp, err := khci.NewSpace(khci.Options{K: 1, NOcc: sys.nocc, NVir: sys.d.NORB - sys.nocc,
		TwoMs: 1, MaxClass: 1})
	if err != nil {
		t.Fatal(err)
	}
	R := sip.New(sip.NewSpace(sys.nocc, sys.d.NORB, nil, 0), sys.ints, sys.eps, 2, backend.Gonum{}).BuildMatrix()
	for r := range sp.Size() {
		for c := range sp.Size() {
			i, j := sp.Holes(r, nil)[0], sp.Holes(c, nil)[0]
			if d := math.Abs(el.Element(sp, r, c) - R.At(i, j)); d > 1e-12 {
				t.Errorf("C11(%d,%d) differs from sip by %g", i, j, d)
			}
		}
	}
}

// TestCouplingMatchesSipOrder3: with sip's main block swapped in, the strict:3
// spin-orbital spectrum contains every sip ADC(3) doublet. C11 itself differs at third
// order by design: sip applies the static self-energy outside the matrix.
func TestCouplingMatchesSipOrder3(t *testing.T) {
	sys := h2o(t)
	el, err := New(sys.ints, sys.eps, sys.nocc, "strict:3")
	if err != nil {
		t.Fatal(err)
	}
	sp, err := khci.NewSpace(khci.Options{K: 1, NOcc: sys.nocc, NVir: sys.d.NORB - sys.nocc,
		TwoMs: 1, MaxClass: 2})
	if err != nil {
		t.Fatal(err)
	}
	M := el.BuildDense(sp)
	R := sip.New(sip.NewSpace(sys.nocc, sys.d.NORB, nil, 0), sys.ints, sys.eps, 3, backend.Gonum{}).BuildMatrix()
	main := sp.MainBlockSize()
	for r := range main {
		for c := range main {
			M.Set(r, c, R.At(sp.Holes(r, nil)[0], sp.Holes(c, nil)[0]))
		}
	}
	got, _ := backend.Gonum{}.SymEig(M)
	want, _ := backend.Gonum{}.SymEig(R)
	sort.Float64s(got)
	if gap := containsAll(got, want); gap > 1e-9 {
		t.Errorf("sip ADC(3) eigenvalue missing from the generated spectrum, max gap %g", gap)
	}
}

// TestADC22MatchesSip: the generated ADC(2,2)m/x/f spectra contain every doublet of
// sip's hand-transcribed ADC(2,2) (A9-A22), on the random canonical reference system.
func TestADC22MatchesSip(t *testing.T) {
	_, sys := loadRef(t)
	be := backend.Gonum{}
	for _, v := range []struct {
		scheme string
		sv     sip.Variant
	}{{"adc22m", sip.VariantM}, {"adc22x", sip.VariantX}, {"adc22f", sip.VariantF}} {
		got, _, _ := spinOrbitalSpectrum(t, sys, v.scheme, 3)
		mx := sip.New(sip.NewSpace22(sys.nocc, sys.d.NORB, nil, 0), sys.ints, sys.eps, sip.Order22, be)
		mx.SetVariant(v.sv)
		want, _ := be.SymEig(mx.BuildMatrix())
		gap := containsAll(got, want)
		t.Logf("%s: %d spin-orbital roots, %d sip doublets, max gap %.2e", v.scheme, len(got), len(want), gap)
		if gap > 1e-9 {
			t.Errorf("%s: sip ADC(2,2) eigenvalue missing from the generated spectrum, max gap %g", v.scheme, gap)
		}
	}
}

// sipClass names the class of a sip ADC(2,2) row: 0 = 1h, 1 = 2h1p, 2 = 3h2p.
func sipClass(s *sip.Space, r int) int {
	switch {
	case r < s.MainBlockSize():
		return 0
	case r < len(s.Configs):
		return 1
	}
	return 2
}

// sipRowsInKhci expresses every spin-adapted sip row as a combination of khci rows:
// sip's DetExpansion gives canonical determinants, and khci's Det sign converts each
// canonical determinant to its operator-string row (|row> = sign |det>, sign = +-1).
func sipRowsInKhci(t *testing.T, sp *khci.Space, s *sip.Space, nocc int) [][]khciTerm {
	t.Helper()
	out := make([][]khciTerm, s.Size())
	for r := range s.Size() {
		ex, err := s.DetExpansion(r)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range ex {
			var mask uint64
			for _, h := range d.Holes {
				mask |= 1 << uint(h)
			}
			parts := make([]int, len(d.Parts))
			for i, p := range d.Parts {
				parts[i] = p - 2*nocc
			}
			idx, ok := sp.Index(mask, parts)
			if !ok {
				t.Fatalf("sip row %d: determinant holes %v particles %v is not a khci row", r, d.Holes, d.Parts)
			}
			_, _, sign := sp.Det(idx)
			out[r] = append(out[r], khciTerm{idx, d.Coef * sign})
		}
	}
	return out
}

type khciTerm struct {
	row  int
	coef float64
}

// TestADC22ElementsMatchSip: second source for sip's hand-transcribed ADC(2,2), element
// by element. Every sip row is expanded over khci rows with sip's own spin functions,
// and <CSF|M|CSF'> from the generated adc22m/x/f blocks must equal sip's element for
// every row pair: 1h/1h through order 2, 1h/2h1p (c12_1, c12_2 for f), 2h1p/2h1p with
// the second-order A9-A14, 2h1p/3h2p (A15-A17) and 3h2p/3h2p (A18-A22). The spectral
// containment of TestADC22MatchesSip cannot see an error that a basis rotation absorbs;
// this test can.
func TestADC22ElementsMatchSip(t *testing.T) {
	_, sys := loadRef(t)
	sp, err := khci.NewSpace(khci.Options{K: 1, NOcc: sys.nocc, NVir: sys.d.NORB - sys.nocc,
		TwoMs: 1, MaxClass: 3})
	if err != nil {
		t.Fatal(err)
	}
	ssp := sip.NewSpace22(sys.nocc, sys.d.NORB, nil, 0)
	T := sipRowsInKhci(t, sp, ssp, sys.nocc)
	names := [3]string{"1h", "2h1p", "3h2p"}
	for _, v := range []struct {
		scheme string
		sv     sip.Variant
	}{{"adc22m", sip.VariantM}, {"adc22x", sip.VariantX}, {"adc22f", sip.VariantF}} {
		el, err := New(sys.ints, sys.eps, sys.nocc, v.scheme)
		if err != nil {
			t.Fatal(err)
		}
		M := el.BuildDense(sp)
		mx := sip.New(ssp, sys.ints, sys.eps, sip.Order22, backend.Gonum{})
		mx.SetVariant(v.sv)
		R := mx.BuildMatrix()
		var worst [3][3]float64
		for I := range ssp.Size() {
			for J := I; J < ssp.Size(); J++ {
				var g float64
				for _, a := range T[I] {
					for _, b := range T[J] {
						g += a.coef * b.coef * M.At(a.row, b.row)
					}
				}
				ci, cj := sipClass(ssp, I), sipClass(ssp, J)
				worst[ci][cj] = max(worst[ci][cj], math.Abs(g-R.At(I, J)))
			}
		}
		for ci := range 3 {
			for cj := ci; cj < 3; cj++ {
				t.Logf("%s %s/%s: max |generated - sip| %.2e", v.scheme, names[ci], names[cj], worst[ci][cj])
				if worst[ci][cj] > 1e-12 {
					t.Errorf("%s %s/%s: sip element differs from the adcgen derivation by %.2e",
						v.scheme, names[ci], names[cj], worst[ci][cj])
				}
			}
		}
	}
}
