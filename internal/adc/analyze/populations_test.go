package analyze

import (
	"math"
	"path/filepath"
	"testing"

	"adcgo/internal/adc/backend"
	"adcgo/internal/adc/dip"
	"adcgo/internal/adc/fcidump"
	"adcgo/internal/adc/integrals"
	"adcgo/internal/adc/lanczos"
	"adcgo/internal/adc/mo"
	"adcgo/internal/adc/mp"
)

// TestPopSumMatchesPS is the M1 population oracle: the atom-resolved two-hole
// populations (one-site + two-site) must sum to the state's pole strength / 100.
// This is the same invariant ADCanalysis's TestPopSumMatchesPS enforces on the
// parsed reference tables.
func TestPopSumMatchesPS(t *testing.T) {
	base := filepath.Join("..", "..", "..", "testdata")
	d, err := fcidump.ReadFile(filepath.Join(base, "h2o.fcidump"))
	if err != nil {
		t.Fatal(err)
	}
	md, err := mo.ReadFile(filepath.Join(base, "h2o.mo.json"))
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	be := backend.Gonum{}

	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		sp := dip.NewSpace(nocc, d.NORB, nil, 0, spin)
		mx := dip.New(sp, ints, eps, be)
		res := lanczos.SolveDense(mx, be)
		pe := NewPopEngine(sp, md)
		sec := BuildSector(sp, res, Options{PSThresh: 1, CoeffThresh: 0.1}, pe)

		for _, s := range sec.States {
			if s.Pop == nil {
				t.Fatalf("spin %d state %d has no population", spin, s.Index)
			}
			if d := math.Abs(s.Pop.Sum() - s.PSPercent/100); d > 1e-6 {
				t.Errorf("spin %d state %d: PopSum %.6f vs ps/100 %.6f (Δ=%.1e)",
					spin, s.Index, s.Pop.Sum(), s.PSPercent/100, d)
			}
		}
	}
}

// TestGroundStateLocalized checks the physical picture: the H2O dication ground
// state is a two-hole state localized on oxygen (Auger@O), so its O one-site
// weight dominates.
func TestGroundStateLocalized(t *testing.T) {
	base := filepath.Join("..", "..", "..", "testdata")
	d, _ := fcidump.ReadFile(filepath.Join(base, "h2o.fcidump"))
	md, _ := mo.ReadFile(filepath.Join(base, "h2o.mo.json"))
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}

	sp := dip.NewSpace(nocc, d.NORB, nil, 0, dip.Singlet)
	mx := dip.New(sp, integrals.New(d, nocc, nil), eps, be)
	res := lanczos.SolveDense(mx, be)
	pe := NewPopEngine(sp, md)
	sec := BuildSector(sp, res, Options{PSThresh: 20, CoeffThresh: 0.1}, pe)

	g := sec.States[0]
	if o := g.Pop.OneSite["O"]; o < 0.7 {
		t.Errorf("ground state O one-site weight %.3f, want > 0.7 (localized on O)", o)
	}
}
