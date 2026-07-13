package selfenergy

import (
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

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
