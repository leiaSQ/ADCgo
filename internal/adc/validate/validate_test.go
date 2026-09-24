package validate

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/analyze"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/refout"
	"github.com/leiaSQ/ADCgo/internal/adc/selfenergy"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
)

// Package validate cross-checks ADCgo's DIP-ADC(2) output against theADCcode's
// h2o DIP reference (../ADCanalysis/examples/DIP_h2o, vendored under
// testdata/reference) on *matched* integrals: scripts/fixtures/gen_ref_fcidump.py
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

// Dyson-orbital cross-validation: ADCgo's Dyson amplitudes (internal/adc/sip/dyson.go)
// against pyscf's, on matched H2O/cc-pVDZ integrals.
//
// The comparison has to be *term-matched* to mean anything. pyscf's Dyson amplitude for a
// virtual orbital carries the first-order 2h1p term f⁽¹⁾ *and* the second-order singles
// t₁⁽²⁾ (and, at adc(3), the second-order doubles t₂⁽²⁾). ADCgo implements f⁽¹⁾ alone
// (docs/adc4_rassi_plan.md, Chunk 4). scripts/fixtures/gen_sip_ref.py therefore dumps a `dyson_o1`
// block computed with approx_trans_moments=True on adc(2)-x, where pyscf's virtual block
// collapses to exactly the one term ADCgo has.
//
// The finding, encoded in the tolerances: ADCgo's `-order 2` and pyscf's `adc(2)-x` are
// the *same method*, and on the same integrals they agree to solver noise — ~1e-9 in the
// ionization energies and ~1e-8 per Dyson component, virtual block included. This is not
// the loose band TestSIPvsPyscf documents. That test pairs `-order 2` against pyscf's
// plain `adc(2)`, which lacks the first-order 2h1p/2h1p block ADCgo's order 2 carries, and
// pairs `-order 3` against `adc(3)`, where the non-Dyson and Dyson self-energy
// formulations genuinely part ways. Extended ADC(2) has no such freedom.
//
// So this is a near-exact gate on the whole assembled Dyson orbital — the F-matrix
// occupied block, the f⁽¹⁾ virtual block, and the relative sign between them, which is the
// one thing no norm can see. The determinant gate in sip/dyson_test.go pins f⁽¹⁾ from
// second quantization alone; this pins the assembly against an independent implementation.

const (
	dysonStrongSF = 0.5  // the main lines; satellites are too dense to pair up by energy
	dysonTolE     = 1e-6 // pyscf's Davidson residual, not a method difference
	dysonTolComp  = 1e-6 // per-component agreement, occupied and virtual alike
	dysonMinVir   = 1e-5 // the reference must carry virtual weight, or the test is vacuous
)

type pyscfDysonRoot struct {
	E  float64   `json:"e_ha"`
	SF float64   `json:"sf"`
	D  []float64 `json:"d"`
}
type pyscfDyson struct {
	Method string           `json:"method"`
	Roots  []pyscfDysonRoot `json:"roots"`
}

// TestDysonvsPyscf pairs every strong pyscf main line with the nearest ADCgo root and
// compares Dyson orbitals component by component.
//
// The virtual block carries only ~0.05% of the norm, so it is checked on its own absolute
// scale rather than the vector's: a test comparing ‖d‖, or a cosine similarity, would pass
// with the virtual components zeroed — and those components are the entire point of the
// chunk.
func TestDysonvsPyscf(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Dyson cross-validation in -short mode")
	}
	b, err := os.ReadFile(testdata("h2o_sip.pyscf.json"))
	if err != nil {
		t.Fatalf("read pyscf ref: %v", err)
	}
	var doc struct {
		Dyson pyscfDyson `json:"dyson_o1"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("unmarshal pyscf ref: %v", err)
	}
	if len(doc.Dyson.Roots) == 0 {
		t.Fatal("h2o_sip.pyscf.json has no dyson_o1 block; rerun scripts/fixtures/gen_sip_ref.py")
	}

	d, err := fcidump.ReadFile(testdata("h2o.fcidump"))
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}
	sp := sip.NewSpace(nocc, d.NORB, nil, 0) // symmetry off: all irreps in one sector
	mx := sip.New(sp, integrals.New(d, nocc, nil), eps, 2, be)
	res := lanczos.SolveDense(mx, be)

	states := make([]int, len(res.Values))
	for i := range states {
		states[i] = i
	}
	dy, err := mx.DysonOrbitals(res.FullVecs, states)
	if err != nil {
		t.Fatal(err)
	}

	var matched int
	for _, r := range doc.Dyson.Roots {
		if r.SF < dysonStrongSF {
			continue
		}
		if len(r.D) != d.NORB {
			t.Fatalf("pyscf Dyson vector has %d components, want %d", len(r.D), d.NORB)
		}
		best, bestErr := 0, math.Inf(1)
		for k := range res.Values {
			if e := math.Abs(res.Values[k] - r.E); e < bestErr {
				bestErr, best = e, k
			}
		}
		if bestErr > dysonTolE {
			t.Errorf("pyscf %.8f Ha (SF %.3f): nearest ADCgo root %.8f is %.1e away — "+
				"extended ADC(2) should reproduce adc(2)-x on matched integrals",
				r.E, r.SF, res.Values[best], bestErr)
			continue
		}
		matched++

		got := make([]float64, d.NORB)
		for p := range d.NORB {
			got[p] = dy.At(p, best)
		}
		// An eigenvector's overall phase is arbitrary; align on the dominant component,
		// which for a main line is the hole orbital itself. Everything after this is a
		// genuine comparison, the relative occupied/virtual sign included.
		lead := 0
		for p := range d.NORB {
			if math.Abs(r.D[p]) > math.Abs(r.D[lead]) {
				lead = p
			}
		}
		if got[lead]*r.D[lead] < 0 {
			for p := range got {
				got[p] = -got[p]
			}
		}

		var virWant float64
		for p := nocc; p < d.NORB; p++ {
			virWant += r.D[p] * r.D[p]
		}
		if virWant < dysonMinVir {
			t.Fatalf("pyscf %.5f Ha: reference virtual weight is %g — nothing to compare",
				r.E, virWant)
		}
		for p := range d.NORB {
			if e := math.Abs(got[p] - r.D[p]); e > dysonTolComp {
				block := "occupied"
				if p >= nocc {
					block = "virtual"
				}
				t.Errorf("pyscf %.5f Ha: %s component %d = %.9f, ADCgo %.9f (Δ=%.1e)",
					r.E, block, p, r.D[p], got[p], e)
			}
		}
	}
	if matched < 3 {
		t.Errorf("only %d strong pyscf lines matched, want >= 3", matched)
	}
}

// TestDIPMatchedIntegrals is the bit-exactness guard for the DIP-ADC(2) secular
// matrix. It runs ADCgo on theADCcode's *own* exported integrals
// (testdata/reference/h2o_dzp.matched.fcidump, written by ../ADC/fcidump_export on
// the GAMESS dfile/vfile), so there is zero basis/integral transcription noise —
// any discrepancy would be pure ADC-method difference.
//
// On matched integrals ADCgo reproduces every well-converged reference line
// (adcdip{1..4}.out, ps >= 5 %) to the reference's ~1e-4 eV print/convergence
// precision. This is the regression guard for the backend.AddSubDiagConst
// diagonal-length fix (2026-07-07): before it, high-lying 3h1p satellite diagonals
// were inflated by ~4.5 Ha, shifting the physical lines by up to ~3 eV.
//
// Weak reference satellites (ps < 5 %) are intentionally excluded: theADCcode's
// own Lanczos (iter 100) does not converge them, so their printed energies are not
// a faithful eigenvalue to compare against. The full dense matrices were verified
// element-wise (~1e-15 Ha) out of band via ../ADC/matrix_dump.

const (
	au2eV          = 27.211396 // matches internal/adc/analyze
	matchedPSConv  = 5.0       // reference lines this strong are Lanczos-converged
	matchedTolEV   = 2e-4      // eV; ~400× the observed worst deviation (0.0005 meV)
	matchedMinScan = 20        // sanity floor (23 lines qualify at ps >= 5 %)
)

// reference DIP-block markers. theADCcode prints two eigenvalue blocks per
// (sym,spin): the real spectrum ("Computing the spectrum …") and a follow-up ISR
// property pass ("Computing spectrum …", no "the") whose first "eigenvalue" is a
// spurious 0.0 eV reference state. Only the former is a DIP eigenvalue list.
var (
	reComputeBlock = regexp.MustCompile(`Computing (the )?spectrum for symmetry \d+, spin (\d+)`)
	reStateLine    = regexp.MustCompile(`^\s*\d+:\s*(-?[\d.]+),\s*(-?[\d.]+),`)
)

type refLine struct {
	spin   int
	energy float64 // eV
	ps     float64 // percent
}

// parseDIPBlock1 returns the states of the real DIP spectrum blocks only.
func parseDIPBlock1(t *testing.T, path string) []refLine {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var out []refLine
	spin, inBlock := 0, false
	for _, ln := range strings.Split(string(data), "\n") {
		if m := reComputeBlock.FindStringSubmatch(ln); m != nil {
			inBlock = m[1] == "the " // "the spectrum" == real DIP eigenvalues
			spin, _ = strconv.Atoi(m[2])
			continue
		}
		if !inBlock {
			continue
		}
		if m := reStateLine.FindStringSubmatch(ln); m != nil {
			e, _ := strconv.ParseFloat(m[1], 64)
			ps, _ := strconv.ParseFloat(m[2], 64)
			out = append(out, refLine{spin: spin, energy: e, ps: ps})
		}
	}
	return out
}

func TestDIPMatchedIntegrals(t *testing.T) {
	path := testdata(filepath.Join("reference", "h2o_dzp.matched.fcidump"))
	d, err := fcidump.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skip("matched FCIDUMP not present (regenerate with ../ADC/fcidump_export)")
		}
		t.Fatalf("read matched fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}
	ints := integrals.New(d, nocc, d.OrbSym)

	// The matched FCIDUMP carries theADCcode's own (GAMESS-UK ordered) ORBSYM, so
	// ADCgo sector index N corresponds to reference file adcdip{N+1}.out directly.
	spins := []struct {
		s    dip.Spin
		code int
	}{{dip.Singlet, 1}, {dip.Triplet, 3}}

	scanned := 0
	for sym := range 4 {
		refs := parseDIPBlock1(t, testdata(filepath.Join("reference", fmt.Sprintf("adcdip%d.out", sym+1))))
		for _, sp := range spins {
			space := dip.NewSpace(nocc, d.NORB, d.OrbSym, sym, sp.s)
			res := lanczos.SolveDense(dip.New(space, ints, eps, be), be)
			for _, r := range refs {
				if r.spin != sp.code || r.ps < matchedPSConv {
					continue
				}
				best := math.Inf(1)
				for _, v := range res.Values {
					if de := math.Abs(v*au2eV - r.energy); de < best {
						best = de
					}
				}
				scanned++
				if best > matchedTolEV {
					t.Errorf("sym%d spin%d ref %.6f eV (ps %.1f%%): nearest ADCgo eigenvalue off by %.4f meV (> %.4f)",
						sym, sp.code, r.energy, r.ps, best*1e3, matchedTolEV*1e3)
				}
			}
		}
	}
	if scanned < matchedMinScan {
		t.Errorf("only scanned %d converged reference lines, want >= %d", scanned, matchedMinScan)
	}
}

// TestSIPMatchedReference is the matched-integral gate for SIP against theADCcode
// itself, the check ADCgo never had: until now the only SIP reference was pyscf's ISR
// IP-ADC (validate_test.go), which uses a different self-energy formulation and so
// leaves an irreducible ~0.03–0.16 eV gap on the main lines — wide enough to hide a real
// porting error. internal/adc/sip is a port of theADCcode's own ndadc3_ip, so run against
// theADCcode's own ndadc3ip output on theADCcode's own integrals there is no method gap
// left and the tolerance can be the reference's print precision.
//
// Reference: testdata/reference/h2o_dzp.sip.ADC.out — ndadc3ip, spin 2, SYMGRP C2v,
// &self-energy infinite, &diagonalizer full, produced on the same GAMESS dfile/vfile that
// h2o_dzp.matched.fcidump was exported from. Both sides therefore see identical integrals
// and identical (GAMESS-UK ordered) ORBSYM, so ADCgo sector N is reference symmetry N+1.
//
// A full diagonalization has no Lanczos convergence caveat, so — unlike the DIP gate —
// weak satellites are legitimate eigenvalues too; the pole-strength floor here only keeps
// the comparison to lines the reference actually prints.

const (
	sipMatchedPSFloor = 0.1  // %, the deck's own print threshold (&eigen ps 0.1)
	sipMatchedTolEV   = 1e-5 // eV; ~20x the observed worst deviation (5e-7 eV = the print precision)
	sipMatchedMinScan = 60   // sanity floor (70 lines qualify at ps >= 0.1 %)
)

var (
	reSIPBlock = regexp.MustCompile(`Computing spectrum for symmetry (\d+), spin (\d+)`)
	reSIPState = regexp.MustCompile(`^\s*\d+:\s*(-?[\d.]+),\s*(-?[\d.]+),`)
)

type sipRefLine struct {
	sym    int // 1-based, as printed
	energy float64
	ps     float64
}

func parseSIPRef(t *testing.T, path string) []sipRefLine {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var out []sipRefLine
	sym := 0
	for _, ln := range strings.Split(string(data), "\n") {
		if m := reSIPBlock.FindStringSubmatch(ln); m != nil {
			sym, _ = strconv.Atoi(m[1])
			continue
		}
		if sym == 0 {
			continue
		}
		if m := reSIPState.FindStringSubmatch(ln); m != nil {
			e, _ := strconv.ParseFloat(m[1], 64)
			ps, _ := strconv.ParseFloat(m[2], 64)
			out = append(out, sipRefLine{sym: sym, energy: e, ps: ps})
		}
	}
	return out
}

func TestSIPMatchedReference(t *testing.T) {
	fc := testdata(filepath.Join("reference", "h2o_dzp.matched.fcidump"))
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skip("matched FCIDUMP not present (regenerate with ../ADC/fcidump_export)")
		}
		t.Fatalf("read matched fcidump: %v", err)
	}
	refPath := testdata(filepath.Join("reference", "h2o_dzp.sip.ADC.out"))
	if _, err := os.Stat(refPath); err != nil {
		t.Skipf("SIP reference unavailable: %v", err)
	}
	refs := parseSIPRef(t, refPath)
	if len(refs) == 0 {
		t.Fatal("no reference states parsed")
	}

	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}
	ints := integrals.New(d, nocc, d.OrbSym)

	// ADCgo's OWN Σ(∞) — no reference values injected. theADCcode's iteration settings are used
	// so the truncated resolvent matches its own (converging tighter would leave the ~1e-6
	// truncation difference and shift the lines by ~0.005 meV).
	sig, err := selfenergy.Static(ints, eps, nocc, d.NORB, selfenergy.Infinite,
		selfenergy.TheADCcodeDefaults)
	if err != nil {
		t.Fatalf("Σ(∞): %v", err)
	}

	scanned, worst := 0, 0.0
	for sym := range 4 {
		space := sip.NewSpace(nocc, d.NORB, d.OrbSym, sym)
		mx := sip.New(space, ints, eps, 3, be) // ndadc3ip == order 3
		mx.SetStaticSelfEnergy(sig.Func())
		res := lanczos.SolveDense(mx, be)
		for _, r := range refs {
			if r.sym != sym+1 || r.ps < sipMatchedPSFloor {
				continue
			}
			best := math.Inf(1)
			for _, v := range res.Values {
				if de := math.Abs(v*au2eV - r.energy); de < best {
					best = de
				}
			}
			scanned++
			if best > worst {
				worst = best
			}
			if best > sipMatchedTolEV {
				t.Errorf("sym%d ref %.6f eV (ps %.2f%%): nearest ADCgo eigenvalue off by %.4f meV (> %.4f)",
					sym+1, r.energy, r.ps, best*1e3, sipMatchedTolEV*1e3)
			}
		}
	}
	if scanned < sipMatchedMinScan {
		t.Errorf("only scanned %d reference lines, want >= %d", scanned, sipMatchedMinScan)
	}
	t.Logf("SIP matched gate: %d reference lines, worst deviation %.4f meV", scanned, worst*1e3)
}

// SIP cross-validation: ADCgo's IP-ADC(2)/(3) vs pyscf's IP-ADC on *matched*
// integrals (both read the same H2O/cc-pVDZ MO integrals — ADCgo from
// testdata/h2o.fcidump, pyscf from the identical mol in scripts/fixtures/gen_sip_ref.py),
// so any residual is ADC method, not basis.
//
// Finding, encoded in the tolerances: the 2h1p satellite roots and the
// spectroscopic factors agree with pyscf to ~1e-5 / ~5e-3, confirming the
// configuration space, the c22/c12 blocks and the F-matrix. The strong 1h main
// lines sit systematically ABOVE pyscf by ~0.001..0.006 Ha (~0.03..0.16 eV) — a
// small self-energy-formulation difference between ndadc3_ip and pyscf's ISR
// IP-ADC, one-sided and monotone. The exact ADCgo numbers are pinned separately
// by the in-process regeneration guard.

const (
	sipStrongSF   = 0.8    // clean valence main-line cutoff (fraction) for tight checks
	sipEnergyLo   = -0.001 // ADCgo main line must not sit below pyscf (FP slack)
	sipEnergyHi   = 0.008  // ... nor above by more than the self-energy-formulation gap (Ha)
	sipTolSF      = 0.01   // spectroscopic-factor agreement (fraction)
	sipMinStrong  = 3      // strong valence lines that must match per order
	sipTolFixture = 1e-8   // regen guard: fixture vs in-process solve
)

type pyscfRoot struct {
	E  float64 `json:"e_ha"`
	SF float64 `json:"sf"`
}
type pyscfRef struct {
	EScf  float64                `json:"e_scf"`
	Roots map[string][]pyscfRoot `json:"roots"`
}

// adcgoState is one solved ADCgo cationic state (energy in Ha, spectroscopic
// factor as a fraction).
type adcgoState struct {
	E  float64
	SF float64
}

// solveSIP diagonalizes the symmetry-off IP-ADC(order) matrix on h2o.fcidump and
// returns its states ordered by energy, each with its F-matrix spectroscopic
// factor (the full spectrum, to compare against pyscf's all-irrep roots).
func solveSIP(t *testing.T, order int) []adcgoState {
	t.Helper()
	d, err := fcidump.ReadFile(testdata("h2o.fcidump"))
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}
	sp := sip.NewSpace(nocc, d.NORB, nil, 0)
	mx := sip.New(sp, integrals.New(d, nocc, nil), eps, order, be)
	evals, evecs := be.SymEig(mx.BuildMatrix())
	fmat := mx.FMatrix()
	main := mx.MainBlockSize()

	states := make([]adcgoState, len(evals))
	for k := range evals {
		y := make([]float64, main)
		for c := range main {
			y[c] = evecs.At(c, k)
		}
		a := fmat.MulVec(y)
		var sf float64
		for _, v := range a {
			sf += v * v
		}
		states[k] = adcgoState{E: evals[k], SF: sf}
	}
	sort.Slice(states, func(i, j int) bool { return states[i].E < states[j].E })
	return states
}

func loadPyscfRef(t *testing.T) pyscfRef {
	t.Helper()
	b, err := os.ReadFile(testdata("h2o_sip.pyscf.json"))
	if err != nil {
		t.Fatalf("read pyscf ref: %v", err)
	}
	var r pyscfRef
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal pyscf ref: %v", err)
	}
	return r
}

// TestSIPvsPyscf: every strong pyscf valence line has a nearby ADCgo state with a
// matching spectroscopic factor, sitting in the one-sided self-energy band.
func TestSIPvsPyscf(t *testing.T) {
	ref := loadPyscfRef(t)
	for _, order := range []int{2, 3} {
		states := solveSIP(t, order)
		var matched int
		for _, r := range ref.Roots[strconv.Itoa(order)] {
			if r.SF < sipStrongSF {
				continue
			}
			// Nearest ADCgo state by energy.
			best, bestErr := adcgoState{}, math.Inf(1)
			for _, s := range states {
				if e := math.Abs(s.E - r.E); e < bestErr {
					bestErr, best = e, s
				}
			}
			matched++
			if dev := best.E - r.E; dev < sipEnergyLo || dev > sipEnergyHi {
				t.Errorf("order %d: pyscf %.5f Ha (SF %.3f): ADCgo %.5f (dev %+.5f) outside [%.3f,%.3f] band",
					order, r.E, r.SF, best.E, dev, sipEnergyLo, sipEnergyHi)
			}
			if d := math.Abs(best.SF - r.SF); d > sipTolSF {
				t.Errorf("order %d: pyscf %.5f Ha: SF %.4f vs ADCgo %.4f (Δ=%.4f)",
					order, r.E, r.SF, best.SF, d)
			}
		}
		if matched < sipMinStrong {
			t.Errorf("order %d: only %d strong pyscf lines matched, want >= %d", order, matched, sipMinStrong)
		}
	}
}

// fxSIPState / fxSIPSector mirror the committed h2o_sip.adcgo.json fixture.
type fxSIPState struct {
	EnergyEV float64 `json:"energy_ev"`
	PS       float64 `json:"ps_percent"`
}
type fxSIPSector struct {
	Irrep  int          `json:"irrep"`
	States []fxSIPState `json:"states"`
}
type fxSIPDoc struct {
	Order   int           `json:"order"`
	Sectors []fxSIPSector `json:"sectors"`
}

// TestSIPFixtureMatchesSolver is the regeneration guard: it re-solves the A1
// (irrep 1) sector in-process on the committed FCIDUMP at order 3 and asserts the
// committed fixture reproduces it, so the JSON cannot silently drift.
func TestSIPFixtureMatchesSolver(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in-process SIP re-solve in -short mode")
	}
	b, err := os.ReadFile(testdata("h2o_sip.adcgo.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx fxSIPDoc
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	var want []fxSIPState
	for _, s := range fx.Sectors {
		if s.Irrep == 1 {
			want = s.States
		}
	}
	if want == nil {
		t.Fatal("no A1 (irrep 1) sector in fixture")
	}

	d, err := fcidump.ReadFile(testdata("h2o.fcidump"))
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}
	sp := sip.NewSpace(nocc, d.NORB, d.OrbSym, 0) // A1
	mx := sip.New(sp, integrals.New(d, nocc, d.OrbSym), eps, fx.Order, be)
	res := lanczos.SolveDense(mx, be)
	sec := analyze.BuildSIPSector(sp, res, mx.FMatrix(), analyze.Options{PSThresh: 1, CoeffThresh: 0.1})

	if len(sec.States) != len(want) {
		t.Fatalf("A1 state count: solver %d, fixture %d", len(sec.States), len(want))
	}
	for i := range sec.States {
		if de := math.Abs(sec.States[i].EnergyEV - want[i].EnergyEV); de > sipTolFixture {
			t.Errorf("state %d energy: solver %.10f vs fixture %.10f (Δ=%.1e)",
				i, sec.States[i].EnergyEV, want[i].EnergyEV, de)
		}
		if dp := math.Abs(sec.States[i].PSPercent - want[i].PS); dp > 1e-6 {
			t.Errorf("state %d ps: solver %.6f vs fixture %.6f", i, sec.States[i].PSPercent, want[i].PS)
		}
	}
}

// Decay-channel spectrum regeneration guard: re-solves the DIP-ADC(2) problem on
// the matched DZP integrals in-process, classifies it into the Auger/ICD/ETMD
// stick spectrum exactly as `adcgo -dip -mo … -spectrum -init-atom O` does, and
// asserts the committed testdata/h2o_dzp.spec.json fixture reproduces it — so the
// spectrum JSON (the contract with ADCanalysis's plotspec) cannot silently drift.

const specTolFixture = 1e-8

// numIrreps mirrors cmd/adcgo's helper: the number of symmetry groups implied by
// the ORBSYM labels (the smallest power of two spanning them).
func numIrreps(orbSym []int, norb int) int {
	if orbSym == nil {
		return 1
	}
	max0 := 0
	for o := range norb {
		if lab := orbSym[o] - 1; lab > max0 {
			max0 = lab
		}
	}
	n := 1
	for n < max0+1 {
		n <<= 1
	}
	return n
}

// solveDIPSpectrum reproduces runDIP + buildDIPSpectrum for the DZP fixture:
// singlet then triplet, per irrep, dense solve with atom-resolved populations,
// classified against O with each atom its own site.
func solveDIPSpectrum(t *testing.T) *spectrum.Spectrum {
	t.Helper()
	d, err := fcidump.ReadFile(testdata("h2o_dzp.fcidump"))
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	moData, err := mo.ReadFile(testdata("h2o_dzp.mo.json"))
	if err != nil {
		t.Fatalf("read mo: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}
	ints := integrals.New(d, nocc, d.OrbSym)
	opts := analyze.Options{PSThresh: 1.0, CoeffThresh: 0.1} // cmd/adcgo defaults

	nsym := numIrreps(d.OrbSym, d.NORB)
	var secs []analyze.Sector
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		for targetSym := range nsym {
			sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, targetSym, spin)
			if sp.Size() == 0 {
				continue
			}
			mx := dip.New(sp, ints, eps, be)
			res := lanczos.SolveDense(mx, be)
			pe := analyze.NewPopEngine(sp, moData)
			secs = append(secs, analyze.BuildSector(sp, res, opts, pe))
		}
	}

	sites := make([]spectrum.Site, len(moData.AtomNames))
	for i, name := range moData.AtomNames {
		sites[i] = spectrum.Site{Name: name, Members: []string{name}}
	}
	spec, _, err := spectrum.BuildDIP(secs, sites, spectrum.DIPOptions{
		InitialAtom:         "O",
		Classify:            spectrum.DefaultOptions(),
		SingletTripletRatio: 3.0,
	})
	if err != nil {
		t.Fatalf("build spectrum: %v", err)
	}
	return spec
}

func TestSpectrumFixtureMatchesSolver(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in-process DIP spectrum re-solve in -short mode")
	}
	b, err := os.ReadFile(testdata("h2o_dzp.spec.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var want spectrum.Spectrum
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	got := solveDIPSpectrum(t)

	if len(got.Channels) != len(want.Channels) {
		t.Fatalf("channel count: solver %d, fixture %d", len(got.Channels), len(want.Channels))
	}
	for i := range got.Channels {
		if got.Channels[i] != want.Channels[i] {
			t.Errorf("channel %d: solver %q vs fixture %q", i, got.Channels[i], want.Channels[i])
		}
	}
	if got.Meta.Kind != want.Meta.Kind || got.Meta.InitialIonization.Atom != want.Meta.InitialIonization.Atom {
		t.Errorf("meta: solver %+v vs fixture %+v", got.Meta, want.Meta)
	}

	if len(got.Lines) != len(want.Lines) {
		t.Fatalf("line count: solver %d, fixture %d", len(got.Lines), len(want.Lines))
	}
	for i := range got.Lines {
		g, w := got.Lines[i], want.Lines[i]
		if g.Channel != w.Channel || g.StateRef != w.StateRef || g.Spin != w.Spin || g.Irrep != w.Irrep {
			t.Errorf("line %d labels: solver %+v vs fixture %+v", i, g, w)
			continue
		}
		if de := math.Abs(g.Energy - w.Energy); de > specTolFixture {
			t.Errorf("line %d (%s) energy: solver %.10f vs fixture %.10f (Δ=%.1e)", i, g.Channel, g.Energy, w.Energy, de)
		}
		if di := math.Abs(g.Intensity - w.Intensity); di > specTolFixture {
			t.Errorf("line %d (%s) intensity: solver %.10f vs fixture %.10f (Δ=%.1e)", i, g.Channel, g.Intensity, w.Intensity, di)
		}
		if dp := math.Abs(g.PSPercent - w.PSPercent); dp > specTolFixture {
			t.Errorf("line %d (%s) ps: solver %.10f vs fixture %.10f", i, g.Channel, g.PSPercent, w.PSPercent)
		}
	}
}
