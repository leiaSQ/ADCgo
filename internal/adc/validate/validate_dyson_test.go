// Dyson-orbital cross-validation: ADCgo's Dyson amplitudes (internal/adc/sip/dyson.go)
// against pyscf's, on matched H2O/cc-pVDZ integrals.
//
// The comparison has to be *term-matched* to mean anything. pyscf's Dyson amplitude for a
// virtual orbital carries the first-order 2h1p term f⁽¹⁾ *and* the second-order singles
// t₁⁽²⁾ (and, at adc(3), the second-order doubles t₂⁽²⁾). ADCgo implements f⁽¹⁾ alone
// (docs/adc4_rassi_plan.md, Chunk 4). scripts/gen_sip_ref.py therefore dumps a `dyson_o1`
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
package validate

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

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
		t.Fatal("h2o_sip.pyscf.json has no dyson_o1 block; rerun scripts/gen_sip_ref.py")
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
