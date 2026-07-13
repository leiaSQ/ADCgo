package validate

// TestSIPMatchedReference is the matched-integral gate for SIP against theADCcode
// itself, the check ADCgo never had: until now the only SIP reference was pyscf's ISR
// IP-ADC (validate_sip_test.go), which uses a different self-energy formulation and so
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

import (
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/selfenergy"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

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
