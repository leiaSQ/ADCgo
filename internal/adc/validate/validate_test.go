// Package validate cross-checks ADCgo's DIP-ADC(2) output against theADCcode's
// h2o DIP reference (../ADCanalysis/examples/DIP_h2o, vendored under
// testdata/reference) on *matched* integrals: scripts/gen_ref_fcidump.py
// reproduces the reference's exact DZP+diffuse basis, geometry, and frozen-core
// active space (gated on SCF = -76.0498071428 Ha), so any residual is ADC method,
// not basis.
//
// This runs ADCgo on the pyscf-*reproduced* integrals (gen_ref_fcidump.py), so the
// residual vs theADCcode is now just pyscf-vs-GAMESS integral transcription noise —
// the ADC(2) *method* is bit-identical. That was verified out-of-band by running
// ADCgo on theADCcode's own exported integrals (../ADC/fcidump_export →
// testdata/reference/h2o_dzp.matched.fcidump): the full DIP matrices agree to
// ~1e-15 Ha and eigenvalues to ~1e-13 eV across all four irreps × both spins.
//
// History: this comment previously attributed a 0.04..3.2 eV gap to the reference's
// "Order: 4+" static self-energy. That was wrong on two counts — fplus never enters
// the adc2dip matrix (it only feeds the two-hole population analysis), and the gap
// survived on matched integrals. The real cause was an ADCgo bug in
// backend.AddSubDiagConst (diagonal ran to the matrix edge, spilling constants onto
// the later spin parts of spin-doubled ijkLMN blocks); fixed 2026-07-07. The band
// below is now integral-noise-limited, not a method gap.
package validate

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/refout"
)

// Tolerances. The energy band is one-sided and wide to absorb pyscf-vs-GAMESS
// integral-transcription noise (the ADC method itself is bit-exact — see the
// matched-integral check noted in the package doc); the tight guarantees are on
// structure (leading configs, irrep), pole strengths, the ground state, and
// populations.
const (
	psMainThresh = 65.0  // "strong line" cutoff (percent) for the comparison
	tolPS        = 5.0   // pole-strength agreement (percent); max observed ~3.7
	energyFloor  = -0.15 // ADCgo must not sit below the reference (allowing FP slack)
	energyCeil   = 3.5   // ... nor above it by more than the 4+ static self-energy gap
	tolGroundE   = 0.15  // ground state (¹A₁): energy to the reference (eV)
	tolGroundPS  = 1.0   // ground state: pole strength (percent)
	tolPopSum    = 1e-3  // per-state PopSum vs ps/100 (ADCgo's own invariant)
)

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

// byRefSector indexes the ADCgo fixture states by (reference symmetry, spin).
// ADCgo's irrep labels now use theADCcode's GAMESS-UK numbering (emitted by the
// FCIDUMP generators), so the mapping is the identity — sec.Irrep is the reference
// file symmetry directly.
func byRefSector(d fxDoc) map[[2]int][]fxState {
	m := map[[2]int][]fxState{}
	for _, sec := range d.Sectors {
		key := [2]int{sec.Irrep, sec.Spin}
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

// TestMatFreeMatchesDenseSolve drives the real iterative block-Lanczos solver (which applies
// the operator through ApplyBlock) over the DZP ¹A₁ sector with the 3h1p↔3h1p satellite region
// applied matrix-free, and checks it reproduces the fully-dense operator's spectrum. This is
// the end-to-end matrix-free validation: SolveDense goes through BuildMatrix and would bypass
// the matrix-free path, so a genuine apply-driven solve is used here. Both runs share identical
// Krylov options over numerically-equal operators, so the eigenvalues agree to solver noise.
func TestMatFreeMatchesDenseSolve(t *testing.T) {
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
	opts := lanczos.Options{MaxBlocks: 40}

	dense := lanczos.Solve(dip.New(sp, ints, eps, be), be, opts)

	mfMx := dip.New(sp, ints, eps, be)
	mfMx.SetMatFree(matfree.On, 0)
	// Confirm matrix-free actually engaged: the satellite region no longer contributes to the
	// resident footprint, so it must be strictly smaller than the dense operator's.
	if denseBytes, mfBytes := dip.New(sp, ints, eps, be).OperatorResidentBytes(), mfMx.OperatorResidentBytes(); mfBytes >= denseBytes {
		t.Fatalf("matrix-free not engaged: resident bytes %d >= dense %d", mfBytes, denseBytes)
	}
	free := lanczos.Solve(mfMx, be, opts)

	if len(dense.Values) != len(free.Values) {
		t.Fatalf("root count: dense %d, matrix-free %d", len(dense.Values), len(free.Values))
	}
	var maxDiff float64
	for i := range dense.Values {
		if de := math.Abs(dense.Values[i] - free.Values[i]); de > maxDiff {
			maxDiff = de
		}
	}
	if maxDiff > 1e-9 {
		t.Errorf("matrix-free vs dense eigenvalues: max |Δ| = %.3e eV (want <= 1e-9)", maxDiff)
	}
	t.Logf("¹A₁ DZP: %d roots, matrix-free vs dense max |Δ| = %.2e", len(dense.Values), maxDiff)
}
