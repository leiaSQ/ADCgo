package parse

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adcanalysis/model"
)

var update = flag.Bool("update", false, "regenerate golden files")

// popSumTolerance: the popana table is rounded to 4 dp across (here) six
// columns and ps to 2 dp, so the worst-case sum vs ps/100 gap is a few e-4.
// Measured max over all four files is 2e-4; 1e-3 is a safe assertion bound.
const popSumTolerance = 1e-3

var outFiles = []string{
	"adcdip1.out",
	"adcdip2.out",
	"adcdip3.out",
	"adcdip4.out",
}

// stateSnapshot is the per-state projection captured in the golden file.
type stateSnapshot struct {
	Irrep      int     `json:"irrep"`
	Spin       int     `json:"spin"`
	Index      int     `json:"index"`
	EnergyEV   float64 `json:"energy_ev"`
	PSPercent  float64 `json:"ps_percent"`
	Residue    string  `json:"residue"` // string: residue is NaN for triplets
	NumLeading int     `json:"num_leading"`
	HasPop     bool    `json:"has_pop"`
	PopSum     float64 `json:"pop_sum"` // rounded; 0 when HasPop is false
}

type fileSnapshot struct {
	Symmetry      int             `json:"symmetry"`
	NumMOs        int             `json:"num_mos"`
	OneSiteGroups []string        `json:"one_site_groups"`
	TwoSiteGroups []string        `json:"two_site_groups"`
	NumStates     int             `json:"num_states"`
	States        []stateSnapshot `json:"states"`
}

func snapshot(o *model.OutFile) fileSnapshot {
	fs := fileSnapshot{
		Symmetry:      o.Symmetry,
		NumMOs:        len(o.MOTable),
		OneSiteGroups: o.Groups.OneSite,
		TwoSiteGroups: o.Groups.TwoSite,
		NumStates:     len(o.States),
	}
	for _, s := range o.States {
		ss := stateSnapshot{
			Irrep: s.Irrep, Spin: s.Spin, Index: s.Index,
			EnergyEV: round(s.EnergyEV, 6), PSPercent: s.PSPercent,
			Residue: strconv.FormatFloat(s.Residue, 'g', -1, 64), NumLeading: len(s.Leading),
			HasPop: s.Pop != nil,
		}
		if s.Pop != nil {
			ss.PopSum = round(s.Pop.Sum(), 6)
		}
		fs.States = append(fs.States, ss)
	}
	return fs
}

func round(v float64, dp int) float64 {
	p := math.Pow10(dp)
	return math.Round(v*p) / p
}

func goldenPath(name string) string {
	return filepath.Join("testdata", "golden", name+".json")
}

// TestParseDIPFileGolden parses each provided file and compares a structural
// snapshot against a stored golden JSON. Run `go test -update` to regenerate.
func TestParseDIPFileGolden(t *testing.T) {
	for _, name := range outFiles {
		name := name
		t.Run(name, func(t *testing.T) {
			o, err := ParseDIPFile(filepath.Join("testdata", name))
			if err != nil {
				t.Fatalf("ParseDIPFile(%s): %v", name, err)
			}
			got, err := json.MarshalIndent(snapshot(o), "", "  ")
			if err != nil {
				t.Fatalf("marshal snapshot: %v", err)
			}
			gp := goldenPath(name)
			if *update {
				if err := os.MkdirAll(filepath.Dir(gp), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(gp, append(got, '\n'), 0o644); err != nil {
					t.Fatal(err)
				}
				t.Logf("updated golden %s", gp)
				return
			}
			want, err := os.ReadFile(gp)
			if err != nil {
				t.Fatalf("read golden %s (run with -update first): %v", gp, err)
			}
			if string(got)+"\n" != string(want) {
				t.Errorf("snapshot mismatch for %s.\n--- got ---\n%s\n--- want ---\n%s",
					name, got, want)
			}
		})
	}
}

// TestPopSumMatchesPS asserts every joined popana row sums to ~= ps/100. This
// is the core physics invariant from the design doc (§2.1): the six one-site /
// two-site columns are the pole-strength decomposition of each state.
func TestPopSumMatchesPS(t *testing.T) {
	for _, name := range outFiles {
		o, err := ParseDIPFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatalf("ParseDIPFile(%s): %v", name, err)
		}
		checked := 0
		for _, s := range o.States {
			if s.Pop == nil {
				continue
			}
			got, want := s.Pop.Sum(), s.PSPercent/100.0
			if d := math.Abs(got - want); d > popSumTolerance {
				t.Errorf("%s irrep%d/s%d/#%d @ %.4f eV: pop sum %.5f vs ps/100 %.5f (|Δ|=%.5f > %.0e)",
					name, s.Irrep, s.Spin, s.Index, s.EnergyEV, got, want, d, popSumTolerance)
			}
			checked++
		}
		if checked == 0 {
			t.Errorf("%s: no states had a joined popana row to check", name)
		}
		t.Logf("%s: verified %d popana sums within %.0e", name, checked, popSumTolerance)
	}
}

// TestNoSentinelStates guards against the eigenvector-saving pass leaking in:
// those emit (0 eV, 100%%, nan) sentinels. Real states must be non-degenerate
// junk-free and every one must carry a population row in these test files.
func TestNoSentinelStates(t *testing.T) {
	for _, name := range outFiles {
		o, err := ParseDIPFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatalf("ParseDIPFile(%s): %v", name, err)
		}
		for _, s := range o.States {
			if s.EnergyEV == 0 && s.PSPercent == 100 {
				t.Errorf("%s: sentinel state leaked in: %+v", name, s)
			}
			if s.Pop == nil {
				t.Errorf("%s irrep%d/s%d/#%d @ %.4f eV: no popana row joined", name, s.Irrep, s.Spin, s.Index, s.EnergyEV)
			}
		}
	}
}

// TestMOTable spot-checks the MO table parse on file 1.
func TestMOTable(t *testing.T) {
	o, err := ParseDIPFile(filepath.Join("testdata", "adcdip1.out"))
	if err != nil {
		t.Fatal(err)
	}
	if len(o.MOTable) != 29 {
		t.Fatalf("MO count = %d, want 29", len(o.MOTable))
	}
	first, last := o.MOTable[0], o.MOTable[len(o.MOTable)-1]
	if first.Index != 1 || first.Sym != 1 || first.EnergyAU != -1.35904 {
		t.Errorf("first MO = %+v", first)
	}
	if last.Index != 29 || last.Sym != 1 || last.EnergyAU != 4.36972 {
		t.Errorf("last MO = %+v", last)
	}
}

// TestBothSpinsPresent confirms each file yields both spin-1 and spin-3 states.
func TestBothSpinsPresent(t *testing.T) {
	for _, name := range outFiles {
		o, err := ParseDIPFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		var s1, s3 int
		for _, s := range o.States {
			switch s.Spin {
			case 1:
				s1++
			case 3:
				s3++
			}
		}
		if s1 == 0 || s3 == 0 {
			t.Errorf("%s: spin coverage s1=%d s3=%d (both must be > 0)", name, s1, s3)
		}
	}
}

// TestLeadingOverlaps spot-checks the first state's parsed <i,j| components.
func TestLeadingOverlaps(t *testing.T) {
	o, err := ParseDIPFile(filepath.Join("testdata", "adcdip1.out"))
	if err != nil {
		t.Fatal(err)
	}
	var first *model.State
	for i := range o.States {
		if o.States[i].Spin == 1 {
			first = &o.States[i]
			break
		}
	}
	if first == nil {
		t.Fatal("no spin-1 state found")
	}
	if len(first.Leading) != 5 {
		t.Fatalf("state #%d leading count = %d, want 5", first.Index, len(first.Leading))
	}
	got := fmt.Sprintf("%d,%d=%.6f", first.Leading[0].I, first.Leading[0].J, first.Leading[0].Coeff)
	if got != "4,4=-0.889428" {
		t.Errorf("first overlap = %s, want 4,4=-0.889428", got)
	}
}
