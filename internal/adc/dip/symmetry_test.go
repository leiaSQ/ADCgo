package dip

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

// symSetup loads the (symmetric) H2O FCIDUMP and returns the pieces needed to
// build either a symmetry-off or a per-irrep DIP matrix.
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

// denseSpectrum diagonalizes a matrix and returns the ascending eigenvalues.
func denseSpectrum(mx *Matrix, be backend.Backend) []float64 {
	ev, _ := be.SymEig(mx.BuildMatrix())
	return ev
}

// TestSymmetryBlockingPartitionsSpace: the per-irrep sectors tile the full
// symmetry-off configuration space (no configuration lost or double-counted).
func TestSymmetryBlockingPartitionsSpace(t *testing.T) {
	d, nocc, _ := symSetup(t)
	nsym := integrals.New(d, nocc, d.OrbSym).NSym()
	for _, spin := range []Spin{Singlet, Triplet} {
		full := NewSpace(nocc, d.NORB, nil, 0, spin).Size()
		sum := 0
		for sym := range nsym {
			sum += NewSpace(nocc, d.NORB, d.OrbSym, sym, spin).Size()
		}
		if sum != full {
			t.Errorf("spin %d: per-irrep sizes sum to %d, want full size %d", spin, sum, full)
		}
	}
}

// TestSymmetryBlockingSpectrum is the M2 correctness gate: because the H2O
// integrals genuinely carry C2v symmetry, the symmetry-off DIP matrix is exactly
// block-diagonal by irrep, so the union of the per-irrep spectra must reproduce
// the full symmetry-off spectrum eigenvalue-for-eigenvalue.
func TestSymmetryBlockingSpectrum(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full-vs-blocked dense spectrum comparison in -short mode")
	}
	d, nocc, eps := symSetup(t)
	be := backend.Gonum{}
	intsOff := integrals.New(d, nocc, nil)
	intsSym := integrals.New(d, nocc, d.OrbSym)
	nsym := intsSym.NSym()

	for _, spin := range []Spin{Singlet, Triplet} {
		// Full spectrum, symmetry off.
		full := denseSpectrum(New(NewSpace(nocc, d.NORB, nil, 0, spin), intsOff, eps, be), be)

		// Union of the per-irrep spectra.
		var blocked []float64
		for sym := range nsym {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			if sp.Size() == 0 {
				continue
			}
			blocked = append(blocked, denseSpectrum(New(sp, intsSym, eps, be), be)...)
		}

		if len(blocked) != len(full) {
			t.Fatalf("spin %d: blocked dim %d != full dim %d", spin, len(blocked), len(full))
		}
		sort.Float64s(blocked)
		var maxErr float64
		for i := range full {
			if e := math.Abs(full[i] - blocked[i]); e > maxErr {
				maxErr = e
			}
		}
		if maxErr > 1e-8 {
			t.Errorf("spin %d: symmetry-blocked spectrum differs from full by %g (>1e-8)", spin, maxErr)
		}
	}
}
