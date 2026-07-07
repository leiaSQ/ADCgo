// Package validate cross-checks ADCgo's DIP-ADC(2) output against theADCcode's
// h2o DIP reference (../ADCanalysis/examples/DIP_h2o, vendored under
// testdata/reference) on *matched* integrals: scripts/gen_ref_fcidump.py
// reproduces the reference's exact DZP+diffuse basis, geometry, and frozen-core
// active space (gated on SCF = -76.0498071428 Ha), so any residual is ADC method,
// not basis.
//
// Finding, encoded in the tolerances below: ADCgo (strict ADC(2)) reproduces the
// reference's DIP *structure* exactly — every strong line's leading two-hole
// configuration and irrep match, pole strengths agree to a few percent — while
// its energies sit systematically ABOVE the reference by 0.04..3.2 eV. That
// positive shift is the reference's higher-order ("Order: 4+") static self-energy
// vs ADCgo's second-order one, a documented method difference, not an error. The
// exact ADCgo numbers are pinned separately by the in-process regeneration guard.
package validate

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"adcgo/internal/adc/analyze"
	"adcgo/internal/adc/backend"
	"adcgo/internal/adc/dip"
	"adcgo/internal/adc/fcidump"
	"adcgo/internal/adc/integrals"
	"adcgo/internal/adc/lanczos"
	"adcgo/internal/adc/mp"
	"adcgo/internal/adc/refout"
)

// Tolerances. The energy band is one-sided and wide because it spans the static
// self-energy-order gap; the tight guarantees are on structure (leading configs,
// irrep), pole strengths, the ground state, and populations.
const (
	psMainThresh = 65.0  // "strong line" cutoff (percent) for the comparison
	tolPS        = 5.0   // pole-strength agreement (percent); max observed ~3.7
	energyFloor  = -0.15 // ADCgo must not sit below the reference (allowing FP slack)
	energyCeil   = 3.5   // ... nor above it by more than the 4+ static self-energy gap
	tolGroundE   = 0.15  // ground state (¹A₁): energy to the reference (eV)
	tolGroundPS  = 1.0   // ground state: pole strength (percent)
	tolPopSum    = 1e-3  // per-state PopSum vs ps/100 (ADCgo's own invariant)
)

// adcgoIrrepToRefSym maps ADCgo's Molpro-ordered irrep index (1=A1,2=B1,3=B2,
// 4=A2) to theADCcode's C₂ᵥ file numbering (1=A1,2=A2,3=B1,4=B2). Verified against
// the reference MO-symmetry table and the 14/14 leading-config identity matches.
var adcgoIrrepToRefSym = map[int]int{1: 1, 2: 3, 3: 4, 4: 2}

type fxLeading struct {
	I, J  int
	Coeff float64
}
type fxPop struct {
	OneSite map[string]float64 `json:"one_site"`
	TwoSite map[string]float64 `json:"two_site"`
}
type fxState struct {
	Index    int         `json:"index"`
	EnergyEV float64     `json:"energy_ev"`
	PS       float64     `json:"ps_percent"`
	Leading  []fxLeading `json:"leading"`
	Pop      *fxPop      `json:"pop"`
}
type fxSector struct {
	Irrep  int       `json:"irrep"`
	Spin   int       `json:"spin"`
	States []fxState `json:"states"`
}
type fxDoc struct {
	Sectors []fxSector
}

func testdata(name string) string { return filepath.Join("..", "..", "..", "testdata", name) }

func loadFixture(t *testing.T) fxDoc {
	t.Helper()
	b, err := os.ReadFile(testdata("h2o_dzp.adcgo.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var d fxDoc
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return d
}

// byRefSectorindexes the ADCgo fixture states by (reference symmetry, spin).
func byRefSector(d fxDoc) map[[2]int][]fxState {
	m := map[[2]int][]fxState{}
	for _, sec := range d.Sectors {
		key := [2]int{adcgoIrrepToRefSym[sec.Irrep], sec.Spin}
		m[key] = append(m[key], sec.States...)
	}
	return m
}

func holes(l fxLeading) [2]int {
	if l.I >= l.J {
		return [2]int{l.I, l.J}
	}
	return [2]int{l.J, l.I}
}

// TestReferenceStructure: every strong reference line has an ADCgo state in the
// mapped irrep+spin with the same leading two-hole configuration, a pole strength
// within tolPS, and an energy in the (one-sided) static-self-energy band.
func TestReferenceStructure(t *testing.T) {
	adc := byRefSector(loadFixture(t))
	var matched int
	for _, name := range []string{"adcdip1.out", "adcdip2.out", "adcdip3.out", "adcdip4.out"} {
		f, err := refout.ParseFile(testdata(filepath.Join("reference", name)))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, rs := range f.States {
			if rs.PSPercent < psMainThresh || rs.EnergyEV < 30 || len(rs.Leading) == 0 {
				continue
			}
			cand := adc[[2]int{f.Symmetry, rs.Spin}]
			// Nearest ADCgo *strong* state by energy.
			best, bestErr := fxState{}, math.Inf(1)
			for _, a := range cand {
				if a.PS < 50 {
					continue
				}
				if e := math.Abs(a.EnergyEV - rs.EnergyEV); e < bestErr {
					bestErr, best = e, a
				}
			}
			if math.IsInf(bestErr, 1) {
				t.Errorf("sym%d spin%d %.3f eV: no ADCgo strong state in sector", f.Symmetry, rs.Spin, rs.EnergyEV)
				continue
			}
			matched++
			// Leading two-hole config identity.
			if len(best.Leading) == 0 || holes(best.Leading[0]) != [2]int{rs.Leading[0].I, rs.Leading[0].J} {
				t.Errorf("sym%d spin%d %.3f eV: leading config %v, ref %v",
					f.Symmetry, rs.Spin, rs.EnergyEV, best.Leading, rs.Leading[0])
			}
			if d := best.PS - rs.PSPercent; math.Abs(d) > tolPS {
				t.Errorf("sym%d spin%d %.3f eV: ps %.2f vs ref %.2f (Δ=%.2f)",
					f.Symmetry, rs.Spin, rs.EnergyEV, best.PS, rs.PSPercent, d)
			}
			if dev := best.EnergyEV - rs.EnergyEV; dev < energyFloor || dev > energyCeil {
				t.Errorf("sym%d spin%d ref %.3f eV: ADCgo %.3f (dev %+.3f) outside [%.2f,%.2f] band",
					f.Symmetry, rs.Spin, rs.EnergyEV, best.EnergyEV, dev, energyFloor, energyCeil)
			}
		}
	}
	if matched < 14 {
		t.Errorf("only %d strong reference lines matched, want >= 14", matched)
	}
}

// TestGroundStateAndPopulations: the ¹A₁ Auger@O ground state matches the
// reference tightly (energy, ps, O-localized population), and every ADCgo state
// with a population satisfies PopSum == ps/100.
func TestGroundStateAndPopulations(t *testing.T) {
	d := loadFixture(t)
	// ¹A₁ = ADCgo Irrep 1, spin 1; lowest state.
	var g *fxState
	for i := range d.Sectors {
		sec := d.Sectors[i]
		if sec.Irrep == 1 && sec.Spin == 1 && len(sec.States) > 0 {
			g = &sec.States[0]
			break
		}
	}
	if g == nil {
		t.Fatal("no ¹A₁ sector in fixture")
	}
	const refE, refPS, refO = 39.660357, 83.39, 0.8065
	if math.Abs(g.EnergyEV-refE) > tolGroundE {
		t.Errorf("ground state %.4f eV vs ref %.4f (tol %.2f)", g.EnergyEV, refE, tolGroundE)
	}
	if math.Abs(g.PS-refPS) > tolGroundPS {
		t.Errorf("ground state ps %.2f vs ref %.2f", g.PS, refPS)
	}
	if g.Pop == nil {
		t.Fatal("ground state has no population (regenerate fixture with -mo)")
	}
	if o := g.Pop.OneSite["O"]; math.Abs(o-refO) > 0.03 {
		t.Errorf("ground state O one-site %.4f vs ref %.4f", o, refO)
	}

	// PopSum == ps/100 for every state carrying a population.
	for _, sec := range d.Sectors {
		for _, s := range sec.States {
			if s.Pop == nil {
				continue
			}
			var sum float64
			for _, v := range s.Pop.OneSite {
				sum += v
			}
			for _, v := range s.Pop.TwoSite {
				sum += v
			}
			if dd := math.Abs(sum - s.PS/100); dd > tolPopSum {
				t.Errorf("irrep%d spin%d state%d: PopSum %.5f vs ps/100 %.5f",
					sec.Irrep, sec.Spin, s.Index, sum, s.PS/100)
			}
		}
	}
}

// TestFixtureMatchesSolver is the regeneration guard: it re-solves the ¹A₁
// singlet sector in-process on the committed FCIDUMP and asserts the fixture
// reproduces it, so the committed JSON cannot silently drift from the solver.
func TestFixtureMatchesSolver(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in-process re-solve in -short mode")
	}
	d, err := fcidump.ReadFile(testdata("h2o_dzp.fcidump"))
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}
	ints := integrals.New(d, nocc, d.OrbSym)

	sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, 0, dip.Singlet) // ¹A₁
	res := lanczos.SolveDense(dip.New(sp, ints, eps, be), be)
	sec := analyze.BuildSector(sp, res, analyze.Options{PSThresh: 1, CoeffThresh: 0.1}, nil)

	// Fixture's ¹A₁ singlet sector.
	fx := loadFixture(t)
	var want []fxState
	for _, s := range fx.Sectors {
		if s.Irrep == 1 && s.Spin == 1 {
			want = s.States
		}
	}
	if len(want) != len(sec.States) {
		t.Fatalf("¹A₁ state count: solver %d, fixture %d", len(sec.States), len(want))
	}
	for i := range sec.States {
		if de := math.Abs(sec.States[i].EnergyEV - want[i].EnergyEV); de > 1e-8 {
			t.Errorf("state %d energy: solver %.10f vs fixture %.10f (Δ=%.1e)",
				i, sec.States[i].EnergyEV, want[i].EnergyEV, de)
		}
		if dp := math.Abs(sec.States[i].PSPercent - want[i].PS); dp > 1e-6 {
			t.Errorf("state %d ps: solver %.6f vs fixture %.6f", i, sec.States[i].PSPercent, want[i].PS)
		}
	}
}
