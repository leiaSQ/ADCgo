package classify

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adcanalysis/model"
)

// h2oGroups mirrors the &popana grouping of the H2O test data, with each atom
// column as its own site (the identity grouping).
var h2oGroups = []model.Site{
	{Name: "O"}, {Name: "H1"}, {Name: "H2"},
}

// firstStateRow is the popana row of the 39.6604 eV state in adcdip1.out:
//
//	O      O/H1    O/H2    H1      H1/H2   H2
//	0.8065 0.0131  0.0131  0.0004  0.0002  0.0004
func firstStateRow() model.PopRow {
	return model.PopRow{
		EnergyEV: 39.6604,
		OneSite:  map[string]float64{"O": 0.8065, "H1": 0.0004, "H2": 0.0004},
		TwoSite:  map[string]float64{"O/H1": 0.0131, "O/H2": 0.0131, "H1/H2": 0.0002},
	}
}

const eps = 1e-9

func weightOf(chs []model.Channel, name string) (float64, bool) {
	for _, c := range chs {
		if c.Name == name {
			return c.Weight, true
		}
	}
	return 0, false
}

func TestClassifyRoutesPopulation(t *testing.T) {
	chs := Classify("O", h2oGroups, firstStateRow(), DefaultOptions())

	want := map[string]float64{
		"Auger@O":   0.8065,
		"ICD:O->H1": 0.0131,
		"ICD:O->H2": 0.0131,
		"ETMD(2)":   0.0004 + 0.0004, // both H one-site: both holes on one neighbour
		"ETMD(3)":   0.0002,          // H1/H2 two-site: holes on two neighbours
	}
	if len(chs) != len(want) {
		t.Fatalf("got %d channels %+v, want %d", len(chs), chs, len(want))
	}
	for name, w := range want {
		got, ok := weightOf(chs, name)
		if !ok {
			t.Errorf("channel %q missing", name)
			continue
		}
		if math.Abs(got-w) > eps {
			t.Errorf("channel %q weight = %.6f, want %.6f", name, got, w)
		}
	}
}

// The sum of all channel weights must equal the row's total two-hole
// population: classification routes, it never creates or destroys weight.
func TestClassifyConservesWeight(t *testing.T) {
	row := firstStateRow()
	chs := Classify("O", h2oGroups, row, Options{IncludeZero: true})
	var sum float64
	for _, c := range chs {
		sum += c.Weight
	}
	if d := math.Abs(sum - row.Sum()); d > eps {
		t.Errorf("channel weights sum to %.6f, row total %.6f (|Δ|=%.2e)", sum, row.Sum(), d)
	}
}

// Canonical order: Auger@A first, ICDs in site order, then ETMD(2), ETMD(3).
func TestClassifyCanonicalOrder(t *testing.T) {
	chs := Classify("O", h2oGroups, firstStateRow(), Options{IncludeZero: true})
	wantOrder := []string{"Auger@O", "ICD:O->H1", "ICD:O->H2", "ETMD(2)", "ETMD(3)"}
	if len(chs) != len(wantOrder) {
		t.Fatalf("got %d channels, want %d: %+v", len(chs), len(wantOrder), chs)
	}
	for i, name := range wantOrder {
		if chs[i].Name != name {
			t.Errorf("channel[%d] = %q, want %q", i, chs[i].Name, name)
		}
	}
}

// A different initial atom re-routes the same row: H1-initiated decay turns the
// O/H1 weight into ICD:H1->O and folds the O one-site weight into ETMD.
func TestClassifyDifferentInitialAtom(t *testing.T) {
	chs := Classify("H1", h2oGroups, firstStateRow(), DefaultOptions())

	if w, ok := weightOf(chs, "Auger@H1"); !ok || math.Abs(w-0.0004) > eps {
		t.Errorf("Auger@H1 = %.6f (ok=%v), want 0.0004", w, ok)
	}
	if w, ok := weightOf(chs, "ICD:H1->O"); !ok || math.Abs(w-0.0131) > eps {
		t.Errorf("ICD:H1->O = %.6f (ok=%v), want 0.0131", w, ok)
	}
	if w, ok := weightOf(chs, "ICD:H1->H2"); !ok || math.Abs(w-0.0002) > eps {
		t.Errorf("ICD:H1->H2 = %.6f (ok=%v), want 0.0002", w, ok)
	}
	// ETMD(2) = O one-site + H2 one-site (both holes on one non-H1 site).
	wantETMD2 := 0.8065 + 0.0004
	if w, ok := weightOf(chs, "ETMD(2)"); !ok || math.Abs(w-wantETMD2) > eps {
		t.Errorf("ETMD(2) = %.6f (ok=%v), want %.6f", w, ok, wantETMD2)
	}
	// ETMD(3) = O/H2 two-site (holes on two non-H1 sites).
	if w, ok := weightOf(chs, "ETMD(3)"); !ok || math.Abs(w-0.0131) > eps {
		t.Errorf("ETMD(3) = %.6f (ok=%v), want 0.0131", w, ok)
	}
}

// MinWeight drops small channels; the surviving set is exactly those above it.
func TestClassifyMinWeight(t *testing.T) {
	chs := Classify("O", h2oGroups, firstStateRow(), Options{MinWeight: 0.01})
	for _, c := range chs {
		if c.Weight <= 0.01 {
			t.Errorf("channel %q weight %.6f survived MinWeight=0.01", c.Name, c.Weight)
		}
	}
	// Auger and the two ICDs clear 0.01; ETMD(2) (0.0008) and ETMD(3) (0.0002)
	// do not.
	if _, ok := weightOf(chs, "ETMD(2)"); ok {
		t.Errorf("ETMD(2) should have been dropped by MinWeight")
	}
	if _, ok := weightOf(chs, "ETMD(3)"); ok {
		t.Errorf("ETMD(3) should have been dropped by MinWeight")
	}
	if len(chs) != 3 {
		t.Errorf("got %d channels, want 3: %+v", len(chs), chs)
	}
}

// MinFraction is relative to the row total (~0.8337 here). 0.5 keeps only
// Auger@O (0.8065/0.8337 ~= 0.967).
func TestClassifyMinFraction(t *testing.T) {
	chs := Classify("O", h2oGroups, firstStateRow(), Options{MinFraction: 0.5})
	if len(chs) != 1 || chs[0].Name != "Auger@O" {
		t.Fatalf("MinFraction=0.5 gave %+v, want only Auger@O", chs)
	}
}

// IncludeZero pads the canonical set even where weights are zero/negative.
func TestClassifyIncludeZero(t *testing.T) {
	row := model.PopRow{
		EnergyEV: 58.8850,
		OneSite:  map[string]float64{"O": 0.5624, "H1": -0.0000, "H2": -0.0000},
		TwoSite:  map[string]float64{"O/H1": 0.1179, "O/H2": 0.1179, "H1/H2": -0.0000},
	}
	chs := Classify("O", h2oGroups, row, Options{IncludeZero: true})
	if len(chs) != 5 {
		t.Fatalf("IncludeZero gave %d channels, want 5: %+v", len(chs), chs)
	}
	// Without IncludeZero, the negative-rounding ETMD buckets disappear.
	bare := Classify("O", h2oGroups, row, DefaultOptions())
	if _, ok := weightOf(bare, "ETMD(2)"); ok {
		t.Errorf("ETMD(2) (negative-rounding) should be dropped by default")
	}
	if _, ok := weightOf(bare, "ETMD(3)"); ok {
		t.Errorf("ETMD(3) (negative-rounding) should be dropped by default")
	}
}

func TestValidateInitialAtom(t *testing.T) {
	if err := ValidateInitialAtom("O", h2oGroups); err != nil {
		t.Errorf("O should be valid: %v", err)
	}
	if err := ValidateInitialAtom("N", h2oGroups); err == nil {
		t.Errorf("N should be rejected")
	}
}

// hSite groups H1 and H2 into a single "H" decay unit.
var hSite = []model.Site{
	{Name: "O", Members: []string{"O"}},
	{Name: "H", Members: []string{"H1", "H2"}},
}

// Regroup folds intra-site two-hole weight into the site's one-site bucket and
// merges parallel two-site entries, conserving the total population.
func TestRegroupFoldsIntraSitePairs(t *testing.T) {
	got := Regroup(firstStateRow(), hSite)

	if w := got.OneSite["O"]; math.Abs(w-0.8065) > eps {
		t.Errorf("O one-site = %.6f, want 0.8065", w)
	}
	// H one-site = H1 + H2 one-site + the folded H1/H2 two-site weight.
	if w := got.OneSite["H"]; math.Abs(w-(0.0004+0.0004+0.0002)) > eps {
		t.Errorf("H one-site = %.6f, want 0.0010", w)
	}
	// O/H1 and O/H2 collapse into a single O/H two-site entry.
	if w := got.TwoSite["O/H"]; math.Abs(w-(0.0131+0.0131)) > eps {
		t.Errorf("O/H two-site = %.6f, want 0.0262", w)
	}
	if len(got.TwoSite) != 1 {
		t.Errorf("got %d two-site entries, want 1: %+v", len(got.TwoSite), got.TwoSite)
	}
	if d := math.Abs(got.Sum() - firstStateRow().Sum()); d > eps {
		t.Errorf("regroup changed total: %.6f vs %.6f", got.Sum(), firstStateRow().Sum())
	}
}

// Grouping reclassifies ETMD: with H1,H2 as one site, O-initiated decay sees the
// former H1/H2 weight as ETMD(2) (both holes on the H unit), not ETMD(3).
func TestClassifyETMDWithGrouping(t *testing.T) {
	row := Regroup(firstStateRow(), hSite)
	chs := Classify("O", hSite, row, Options{IncludeZero: true})

	if w, ok := weightOf(chs, "ETMD(2)"); !ok || math.Abs(w-(0.0004+0.0004+0.0002)) > eps {
		t.Errorf("ETMD(2) = %.6f (ok=%v), want 0.0010", w, ok)
	}
	// With O and H the only sites, there is no second neighbour for the two
	// holes to split across, so ETMD(3) is not a possible channel and is not
	// emitted at all — even with IncludeZero.
	if w, ok := weightOf(chs, "ETMD(3)"); ok {
		t.Errorf("ETMD(3) should be absent with only one neighbour site, got %.6f", w)
	}
	if w, ok := weightOf(chs, "ICD:O->H"); !ok || math.Abs(w-0.0262) > eps {
		t.Errorf("ICD:O->H = %.6f (ok=%v), want 0.0262", w, ok)
	}
}

// With every column folded into one site, that site is the only place the two
// holes can sit, so the sole possible channel is local Auger. ICD and both ETMD
// channels need a neighbour site and must not appear — not even at zero weight
// with IncludeZero. (Regression: a single group used to still advertise
// ETMD(2)/ETMD(3).)
func TestClassifySingleSiteOnlyAuger(t *testing.T) {
	water := []model.Site{{Name: "water", Members: []string{"O", "H1", "H2"}}}
	row := Regroup(firstStateRow(), water)
	chs := Classify("water", water, row, Options{IncludeZero: true})

	if len(chs) != 1 {
		t.Fatalf("got %d channels %+v, want exactly Auger@water", len(chs), chs)
	}
	if w, ok := weightOf(chs, "Auger@water"); !ok || math.Abs(w-firstStateRow().Sum()) > eps {
		t.Errorf("Auger@water = %.6f (ok=%v), want all weight %.6f", w, ok, firstStateRow().Sum())
	}
}

// Discount drops a passive column's own one-site weight, halves a two-site weight
// that straddles an active and a passive column (crediting only the active hole),
// and removes a two-site weight whose both columns are passive — leaving an
// active-only one-site weight untouched.
func TestDiscountHalvesPassivePairs(t *testing.T) {
	passive := map[string]bool{"H1": true, "H2": true}
	got := Discount(firstStateRow(), passive)

	wantOne := map[string]float64{"O": 0.8065} // H1, H2 one-site dropped
	wantTwo := map[string]float64{"O/H1": 0.0131 / 2, "O/H2": 0.0131 / 2} // H1/H2 dropped
	if len(got.OneSite) != len(wantOne) || len(got.TwoSite) != len(wantTwo) {
		t.Fatalf("got one=%v two=%v, want one=%v two=%v", got.OneSite, got.TwoSite, wantOne, wantTwo)
	}
	for k, w := range wantOne {
		if math.Abs(got.OneSite[k]-w) > eps {
			t.Errorf("OneSite[%q] = %.6f, want %.6f", k, got.OneSite[k], w)
		}
	}
	for k, w := range wantTwo {
		if math.Abs(got.TwoSite[k]-w) > eps {
			t.Errorf("TwoSite[%q] = %.6f, want %.6f", k, got.TwoSite[k], w)
		}
	}
}

// Discount with no passive columns is the identity: every weight is preserved.
func TestDiscountNoPassiveIsIdentity(t *testing.T) {
	row := firstStateRow()
	got := Discount(row, PassiveSet([]model.Site{{Name: "O"}, {Name: "H1"}, {Name: "H2"}}))
	if math.Abs(got.Sum()-row.Sum()) > eps {
		t.Errorf("discounted total %.6f, want unchanged %.6f", got.Sum(), row.Sum())
	}
}

// End to end for the motivating case: a water site whose hydrogens are passive.
// Auger@water credits the O one-site weight in full and the two O/H pairs at half
// (one hole still on O), and discards the all-hydrogen H1, H2, H1/H2 weight.
func TestPassiveWaterDiscountsHydrogens(t *testing.T) {
	water := []model.Site{{
		Name:    "water",
		Members: []string{"O", "H1", "H2"},
		Passive: []string{"H1", "H2"},
	}}
	row := Regroup(Discount(firstStateRow(), PassiveSet(water)), water)
	chs := Classify("water", water, row, Options{IncludeZero: true})

	want := 0.8065 + 0.0131/2 + 0.0131/2 // O full + two O/H halves
	if len(chs) != 1 {
		t.Fatalf("got %d channels %+v, want exactly Auger@water", len(chs), chs)
	}
	if w, ok := weightOf(chs, "Auger@water"); !ok || math.Abs(w-want) > eps {
		t.Errorf("Auger@water = %.6f (ok=%v), want %.6f", w, ok, want)
	}
}
