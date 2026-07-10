package sip

import (
	"math"
	"path/filepath"
	"sort"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

func symSetup(t *testing.T) (*fcidump.Data, int, []float64) {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	if d.OrbSym == nil {
		t.Fatal("testdata FCIDUMP has no ORBSYM; regenerate with symmetry")
	}
	nocc := mp.NOcc(d)
	return d, nocc, mp.OrbitalEnergies(d, nocc)
}

// TestSymmetryPartitionsSpace: the per-irrep sectors tile the full symmetry-off
// configuration space (no configuration lost or double-counted).
func TestSymmetryPartitionsSpace(t *testing.T) {
	d, nocc, _ := symSetup(t)
	nsym := integrals.New(d, nocc, d.OrbSym).NSym()
	full := NewSpace(nocc, d.NORB, nil, 0).Size()
	sum := 0
	for sym := range nsym {
		sum += NewSpace(nocc, d.NORB, d.OrbSym, sym).Size()
	}
	if sum != full {
		t.Errorf("per-irrep sizes sum to %d, want full size %d", sum, full)
	}
}

// TestSymmetryBlockingSpectrum is the M2 gate for SIP: because the H2O integrals
// carry C2v symmetry, the symmetry-off IP matrix is exactly block-diagonal by
// irrep, so the union of the per-irrep spectra reproduces the full symmetry-off
// spectrum eigenvalue-for-eigenvalue (both ADC orders).
func TestSymmetryBlockingSpectrum(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full-vs-blocked dense spectrum comparison in -short mode")
	}
	d, nocc, eps := symSetup(t)
	be := backend.Gonum{}
	intsOff := integrals.New(d, nocc, nil)
	intsSym := integrals.New(d, nocc, d.OrbSym)
	nsym := intsSym.NSym()

	spectrum := func(mx *Matrix) []float64 { ev, _ := be.SymEig(mx.BuildMatrix()); return ev }

	for _, order := range []int{2, 3} {
		full := spectrum(New(NewSpace(nocc, d.NORB, nil, 0), intsOff, eps, order, be))

		var blocked []float64
		for sym := range nsym {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym)
			if sp.Size() == 0 {
				continue
			}
			blocked = append(blocked, spectrum(New(sp, intsSym, eps, order, be))...)
		}
		if len(blocked) != len(full) {
			t.Fatalf("order %d: blocked dim %d != full dim %d", order, len(blocked), len(full))
		}
		sort.Float64s(blocked)
		var maxErr float64
		for i := range full {
			if e := math.Abs(full[i] - blocked[i]); e > maxErr {
				maxErr = e
			}
		}
		if maxErr > 1e-8 {
			t.Errorf("order %d: symmetry-blocked spectrum differs from full by %g (>1e-8)", order, maxErr)
		}
	}
}
