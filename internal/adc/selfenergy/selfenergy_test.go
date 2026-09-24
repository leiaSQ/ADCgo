package selfenergy

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// The reference Σ for each scheme is theADCcode's own, dumped per symmetry from
// ndadc3_ip/nd_adc3_matrix.cpp on the matched h2o/DZP integrals:
// testdata/reference/h2o_dzp_sip_sigma/<scheme>/SIGMA_ND_sym{N}.dat — one file per irrep with
// occupied orbitals, rows "ni nj value" (1-based absolute orbitals, Hartree). Σ is irrep-diagonal
// and the ADC main block only ever reads its occupied/occupied block, so that is what is gated.

func refDir(scheme string) string {
	return filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp_sip_sigma", scheme)
}

// readRefSigma loads every SIGMA_ND_sym*.dat under dir into (i,j)→Σ, 0-based absolute orbitals.
func readRefSigma(t *testing.T, dir string) map[[2]int]float64 {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "SIGMA_ND_sym*.dat"))
	if err != nil || len(files) == 0 {
		t.Skipf("no reference Σ in %s", dir)
	}
	out := make(map[[2]int]float64)
	for _, fn := range files {
		f, err := os.Open(fn)
		if err != nil {
			t.Fatal(err)
		}
		var n int
		if _, err := fmt.Fscan(f, &n); err != nil {
			f.Close()
			t.Fatalf("%s: header: %v", fn, err)
		}
		for range n * (n + 1) / 2 {
			var ni, nj int
			var v float64
			if _, err := fmt.Fscan(f, &ni, &nj, &v); err != nil {
				f.Close()
				t.Fatalf("%s: row: %v", fn, err)
			}
			out[[2]int{ni - 1, nj - 1}] = v
			out[[2]int{nj - 1, ni - 1}] = v
		}
		f.Close()
	}
	return out
}

func loadH2O(t *testing.T) (*integrals.Store, []float64, int, int) {
	t.Helper()
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	return integrals.New(d, nocc, d.OrbSym), eps, nocc, d.NORB
}

// checkScheme gates one scheme against theADCcode's Σ on matched integrals.
func checkScheme(t *testing.T, scheme Scheme, tol float64) {
	t.Helper()
	checkSchemeOpts(t, scheme, Options{}, tol)
}

func checkSchemeOpts(t *testing.T, scheme Scheme, opts Options, tol float64) {
	t.Helper()
	ref := readRefSigma(t, refDir(scheme.String()))
	ints, eps, nocc, norb := loadH2O(t)

	sig, err := Static(ints, eps, nocc, norb, scheme, opts)
	if err != nil {
		t.Fatalf("Static(%v): %v", scheme, err)
	}

	var maxd float64
	var checked int
	for i := range nocc {
		for j := range nocc {
			want, ok := ref[[2]int{i, j}]
			if !ok {
				continue // different irreps: Σ is irrep-diagonal, reference stores nothing
			}
			if d := math.Abs(sig.At(i, j) - want); d > maxd {
				maxd = d
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no reference elements matched the occupied space")
	}
	if maxd > tol {
		for i := range nocc {
			if want, ok := ref[[2]int{i, i}]; ok {
				t.Logf("  Σ(%d,%d): got %+.10f  want %+.10f", i, i, sig.At(i, i), want)
			}
		}
		t.Errorf("%v: max |Σ − Σ_theADCcode| = %.3e over %d elements (tol %.1e)",
			scheme, maxd, checked, tol)
		return
	}
	t.Logf("%v: %d elements, max deviation %.2e", scheme, checked, maxd)
}

func TestSigmaThree(t *testing.T)    { checkScheme(t, Three, 1e-10) }
func TestSigmaFour(t *testing.T)     { checkScheme(t, Four, 1e-10) }
func TestSigmaFourPlus(t *testing.T) { checkScheme(t, FourPlus, 1e-10) }

// TestSigmaInfinite is the parity gate: with theADCcode's own iteration settings (Akrit 1e-9,
// MaxIt 30) ADCgo must reproduce its Σ(∞) — truncation and all. The iteration is what defines
// the reference's answer, so this is the honest comparison.
func TestSigmaInfinite(t *testing.T) {
	checkSchemeOpts(t, Infinite, TheADCcodeDefaults, 1e-10)
}

// TestDensityIsWhatStaticUses pins the exported ρ to the one Static builds internally: Σ⁽³⁾ is
// by definition ρ⁽²⁾ contracted with the integrals (eq. A25), so contracting the ρ that Density
// hands out must reproduce Static(Three) exactly. Σ⁽³⁾ is itself bit-exact against theADCcode,
// so this transfers that gate onto the object the ISR property matrix consumes.
func TestDensityIsWhatStaticUses(t *testing.T) {
	ints, eps, nocc, norb := loadH2O(t)

	rho, err := Density(ints, eps, nocc, norb, 2)
	if err != nil {
		t.Fatal(err)
	}
	got := newEngine(ints, eps, nocc, norb).rhoToSigma(rho)

	want, err := Static(ints, eps, nocc, norb, Three, Options{})
	if err != nil {
		t.Fatal(err)
	}
	for i := range norb {
		for j := range norb {
			if got.At(i, j) != want.At(i, j) {
				t.Fatalf("Σ from the exported ρ differs at (%d,%d): %.17g vs Static's %.17g",
					i, j, got.At(i, j), want.At(i, j))
			}
		}
	}

	// ρ must carry a nonzero hole/particle block — that is the only block (13c) reads, and a
	// ρ that silently lacked it would still pass a Σ-level check on the occupied block alone.
	var maxPH float64
	for a := nocc; a < norb; a++ {
		for k := range nocc {
			if v := math.Abs(rho.At(a, k)); v > maxPH {
				maxPH = v
			}
		}
	}
	if maxPH < 1e-8 {
		t.Errorf("ρ hole/particle block is empty (max |ρ_ak| = %.3e); (13c) would be inert", maxPH)
	}
}

// TestSchemeSpectrum is not a gate — it reports where each Σ scheme puts the h2o/DZP main
// lines, so the CLI default can be chosen on evidence. Reference points on these *same*
// integrals: pyscf IP-ADC(3) gives 15.037 (3a₁), 12.847 (1b₁), 19.643 (1b₂) eV, and
// theADCcode's Σ(∞) gives 14.882 / 12.687 / 19.507.
func TestSchemeSpectrum(t *testing.T) {
	const au2eV = 27.211396
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}
	ints := integrals.New(d, nocc, d.OrbSym)

	// sector -> (label, pyscf IP-ADC(3), theADCcode Σ(∞))
	sectors := []struct {
		sym        int
		label      string
		pyscf, adc float64
	}{
		{0, "3a1(A1)", 15.037, 14.882},
		{2, "1b1(B1)", 12.847, 12.687},
		{3, "1b2(B2)", 19.643, 19.507},
	}

	run := func(sig *Sigma) map[int]float64 {
		out := map[int]float64{}
		for _, s := range sectors {
			sp := sip.NewSpace(nocc, d.NORB, d.OrbSym, s.sym)
			mx := sip.New(sp, ints, eps, 3, be)
			if sig != nil {
				mx.SetStaticSelfEnergy(sig.Func())
			}
			res := lanczos.SolveDense(mx, be)
			lo := res.Values[0]
			for _, v := range res.Values {
				if v < lo {
					lo = v
				}
			}
			out[s.sym] = lo * au2eV
		}
		return out
	}

	type row struct {
		name string
		e    map[int]float64
	}
	rows := []row{{"no Σ", run(nil)}}
	for _, sc := range []Scheme{Three, Four, FourPlus} {
		sig, err := Static(ints, eps, nocc, d.NORB, sc, Options{})
		if err != nil {
			t.Fatalf("Static(%v): %v", sc, err)
		}
		rows = append(rows, row{sc.String(), run(sig)})
	}

	for _, s := range sectors {
		t.Logf("%s  pyscf=%.3f  theADCcode(Σ∞)=%.3f", s.label, s.pyscf, s.adc)
		for _, r := range rows {
			t.Logf("    ADCgo %-9s %8.3f eV   (vs pyscf %+.3f, vs theADCcode %+.3f)",
				r.name, r.e[s.sym], r.e[s.sym]-s.pyscf, r.e[s.sym]-s.adc)
		}
	}
}
