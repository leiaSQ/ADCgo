package main

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/fano"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/stieltjes"
)

func testFCIDUMP(t *testing.T) *fcidump.Data {
	t.Helper()
	d, err := fcidump.ReadFile("../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	return d
}

func TestParseOrbitalList(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []int
		bad  bool
	}{
		{"", nil, false},
		{"   ", nil, false},
		{"0", []int{0}, false},
		{"0,3, 7 ", []int{0, 3, 7}, false},
		{"1,,2", []int{1, 2}, false},
		{"-1", nil, true},
		{"a", nil, true},
		{"1.5", nil, true},
	} {
		got, err := parseOrbitalList("-fano-q", c.in)
		if c.bad {
			if err == nil {
				t.Errorf("parseOrbitalList(%q) accepted", c.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseOrbitalList(%q): %v", c.in, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("parseOrbitalList(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestParseHoleRule(t *testing.T) {
	for _, c := range []struct {
		in   string
		want fano.HoleRule
		bad  bool
	}{
		{"", fano.AnyHole, false},
		{"any", fano.AnyHole, false},
		{"all", fano.AllHoles, false},
		{"ANY", fano.AnyHole, true},
		{"either", fano.AnyHole, true},
	} {
		got, err := parseHoleRule(c.in)
		if c.bad != (err != nil) {
			t.Errorf("parseHoleRule(%q): err = %v, want bad = %v", c.in, err, c.bad)
		}
		if err == nil && got != c.want {
			t.Errorf("parseHoleRule(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestParseStieltjesOrders(t *testing.T) {
	for _, c := range []struct {
		in     string
		lo, hi int
		bad    bool
	}{
		{"", 0, 0, false},
		{"5-40", 5, 40, false},
		{"8-", 8, 0, false},
		{"-40", 0, 40, false},
		{"7-7", 7, 7, false},
		{"40-5", 0, 0, true},
		{"0-10", 0, 0, true},
		{"x-10", 0, 0, true},
	} {
		lo, hi, err := parseStieltjesOrders(c.in)
		if c.bad != (err != nil) {
			t.Errorf("parseStieltjesOrders(%q): err = %v, want bad = %v", c.in, err, c.bad)
			continue
		}
		if err == nil && (lo != c.lo || hi != c.hi) {
			t.Errorf("parseStieltjesOrders(%q) = (%d,%d), want (%d,%d)", c.in, lo, hi, c.lo, c.hi)
		}
	}
}

func TestParseAverageMode(t *testing.T) {
	if m, err := parseAverageMode(""); err != nil || m != stieltjes.AveragePaper {
		t.Errorf("the default averaging mode is %v (err %v), want the paper's", m, err)
	}
	if m, err := parseAverageMode("reference"); err != nil || m != stieltjes.AverageReference {
		t.Errorf("parseAverageMode(reference) = %v, err %v", m, err)
	}
	if _, err := parseAverageMode("mean"); err == nil {
		t.Error("an unknown averaging mode was accepted")
	}
}

// TestFanoRunRejectsBadConfig covers the guards that would otherwise produce a confident
// wrong number: an order with no Fano path, and a vacancy that is not occupied.
func TestFanoRunRejectsBadConfig(t *testing.T) {
	d := testFCIDUMP(t)
	for _, c := range []struct {
		name string
		cfg  fanoConfig
		want string
	}{
		{"order 4", fanoConfig{sip: sipConfig{order: 4, solver: "dense"}, vacancy: 0}, "-order"},
		{"unoccupied vacancy", fanoConfig{sip: sipConfig{order: 2, solver: "dense"}, vacancy: 999}, "occupied"},
		{"bad solver", fanoConfig{sip: sipConfig{order: 2, solver: "magic"}, vacancy: 0}, "solver"},
	} {
		err := runFano(d, c.cfg)
		if err == nil {
			t.Errorf("%s: accepted", c.name)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %q does not mention %q", c.name, err, c.want)
		}
	}
}

// TestFanoDocumentSchema is the schema guard: the JSON keys a downstream parser or plotting
// script reads must not drift silently. Fields that are only sometimes present carry
// omitempty and are checked to be absent from a minimal document.
func TestFanoDocumentSchema(t *testing.T) {
	b, err := json.Marshal(FanoDocument{})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"norb", "nelec", "order", "scheme", "vacancy", "irrep", "criterion",
		"size", "q_size", "p_size", "q_main", "p_main", "p_decay",
		"e_phi_hartree", "e_phi_ev", "phi_weight", "phi_root",
		"pseudo_states", "sum_gamma_hartree2", "sum_rule_hartree2", "sum_rule_residual",
		"dropped_energy", "dropped_weight", "dropped_gamma", "coupled_channels",
		"width_mev", "width_sigma_mev", "width_ev", "lifetime_fs",
		"stieltjes_max_order", "stieltjes_orders_used", "pseudo_emin_ev", "pseudo_emax_ev", "phi_position",
	}
	for _, k := range want {
		if _, ok := got[k]; !ok {
			t.Errorf("the Fano document is missing the required key %q", k)
		}
	}
	for _, k := range []string{
		"variant", "class_3h2p", "stieltjes_low_order", "partial_widths", "eval_ev",
		"partial_widths_note", "timing", "stieltjes_below", "stieltjes_above",
	} {
		if _, ok := got[k]; ok {
			t.Errorf("key %q should carry omitempty: a minimal document must not emit it", k)
		}
	}
	if len(got) != len(want) {
		t.Errorf("the Fano document has %d always-present keys, want %d — a new field needs "+
			"adding to this test's list (or omitempty)", len(got), len(want))
	}

	// The channel record's own schema.
	cb, err := json.Marshal(FanoChannel{Name: "Auger@O", Share: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	var cm map[string]any
	if err := json.Unmarshal(cb, &cm); err != nil {
		t.Fatal(err)
	}
	if _, ok := cm["name"]; !ok {
		t.Error("a channel record must always carry its name")
	}
	if _, ok := cm["strength_share"]; !ok {
		t.Error("a channel record must always carry its strength share")
	}
	if _, ok := cm["width_mev"]; ok {
		t.Error("width_mev should carry omitempty: a channel that could not be imaged has none")
	}
}
