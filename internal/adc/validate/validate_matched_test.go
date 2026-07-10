package validate

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

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

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
