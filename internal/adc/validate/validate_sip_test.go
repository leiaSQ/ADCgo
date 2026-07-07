// SIP cross-validation: ADCgo's IP-ADC(2)/(3) vs pyscf's IP-ADC on *matched*
// integrals (both read the same H2O/cc-pVDZ MO integrals — ADCgo from
// testdata/h2o.fcidump, pyscf from the identical mol in scripts/gen_sip_ref.py),
// so any residual is ADC method, not basis.
//
// Finding, encoded in the tolerances: the 2h1p satellite roots and the
// spectroscopic factors agree with pyscf to ~1e-5 / ~5e-3, confirming the
// configuration space, the c22/c12 blocks and the F-matrix. The strong 1h main
// lines sit systematically ABOVE pyscf by ~0.001..0.006 Ha (~0.03..0.16 eV) — a
// small self-energy-formulation difference between ndadc3_ip and pyscf's ISR
// IP-ADC, one-sided and monotone. The exact ADCgo numbers are pinned separately
// by the in-process regeneration guard.
package validate

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"strconv"
	"testing"

	"adcgo/internal/adc/analyze"
	"adcgo/internal/adc/backend"
	"adcgo/internal/adc/fcidump"
	"adcgo/internal/adc/integrals"
	"adcgo/internal/adc/lanczos"
	"adcgo/internal/adc/mp"
	"adcgo/internal/adc/sip"
)

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
