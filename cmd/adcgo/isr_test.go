package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestISRDenseMatchesDip: -isr dip:adc2x -solver dense reproduces every pole of package
// dip's dense spectrum (h2o, symmetry off) above the pole-strength threshold, singlet and
// triplet, energy and pole strength.
func TestISRDenseMatchesDip(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "isr.json")
	cfg := isrConfig{variant: "dip", scheme: "adc2x", solver: "dense", spinSel: "singlet,triplet",
		sym: "none", backend: "gonum", out: out, twoMs: -1, psThresh: 1, coeffThresh: 0.1}
	if err := runISR(d, cfg); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var doc isrDocument
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	be := backend.Gonum{}
	for _, sec := range doc.Sectors {
		spin := dip.Singlet
		if sec.Multiplicity == 3 {
			spin = dip.Triplet
		}
		sp := dip.NewSpace(nocc, d.NORB, nil, 0, spin)
		vals, vecs := be.SymEig(dip.New(sp, ints, eps, be).BuildMatrix())
		type pole struct{ e, ps float64 }
		var want []pole
		for k, v := range vals {
			var ps float64
			for r := range sp.MainBlockSize() {
				ps += vecs.At(r, k) * vecs.At(r, k)
			}
			if 100*ps >= cfg.psThresh {
				want = append(want, pole{v * isrAu2eV, 100 * ps})
			}
		}
		if len(want) != len(sec.States) {
			t.Fatalf("multiplicity %d: %d -isr states, %d dip poles", sec.Multiplicity, len(sec.States), len(want))
		}
		for i, st := range sec.States {
			if math.Abs(st.EnergyEV-want[i].e) > 1e-9 || math.Abs(st.PSPercent-want[i].ps) > 1e-7 {
				t.Fatalf("multiplicity %d state %d: %.10f eV %.6f%%, dip %.10f eV %.6f%%", sec.Multiplicity, i,
					st.EnergyEV, st.PSPercent, want[i].e, want[i].ps)
			}
		}
		t.Logf("multiplicity %d: %d poles agree", sec.Multiplicity, len(want))
	}
	if len(doc.Sectors) != 2 {
		t.Fatalf("%d sectors, want singlet and triplet", len(doc.Sectors))
	}
}

func TestParseISR(t *testing.T) {
	if v, s, err := parseISR("dip:adc2x"); err != nil || v != "dip" || s != "adc2x" {
		t.Fatalf("dip:adc2x -> %q %q %v", v, s, err)
	}
	for _, bad := range []string{"dip", "dip:nope", "zip:adc2x"} {
		if _, _, err := parseISR(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	if got := schemeMaxClass(2, isrVariants["dip"].schemes["ci"]); got != 4 {
		t.Errorf("dip ci max class %d, want 4", got)
	}
	if got := schemeMaxClass(2, isrVariants["dip"].schemes["adc2x"]); got != 3 {
		t.Errorf("dip adc2x max class %d, want 3", got)
	}
}
