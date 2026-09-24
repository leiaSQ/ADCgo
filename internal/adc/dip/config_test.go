package dip

import (
	"math"
	"path/filepath"
	"sort"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// H2O/cc-pVDZ (symmetry off): nocc=5, norb=24, nvir=19.
const (
	testNocc = 5
	testNorb = 24
	testNvir = 19
)

func TestConfigCountsSinglet(t *testing.T) {
	s := NewSpace(testNocc, testNorb, nil, 0, Singlet)

	// |ii> = 5, |ij> (i>j) = C(5,2)=10 → main block 15.
	if s.BeginIJ != 5 {
		t.Errorf("BeginIJ=%d want 5", s.BeginIJ)
	}
	if s.MainBlockSize() != 15 {
		t.Errorf("main block=%d want 15", s.MainBlockSize())
	}
	// |jiir>: 5*4=20 groups × 19 virtuals = 380.
	jiirDim := s.BeginIJK - s.BeginJII
	if jiirDim != 20*testNvir {
		t.Errorf("|jiir> dim=%d want %d", jiirDim, 20*testNvir)
	}
	if len(s.JII) != 20 {
		t.Errorf("JII groups=%d want 20", len(s.JII))
	}
	// |ijkr,T>: C(5,3)=10 groups × mult(2) × 19 = 380.
	ijkrDim := s.Size() - s.BeginIJK
	if ijkrDim != 10*2*testNvir {
		t.Errorf("|ijkr> dim=%d want %d", ijkrDim, 10*2*testNvir)
	}
	if len(s.IJK) != 10 {
		t.Errorf("IJK groups=%d want 10", len(s.IJK))
	}
	if s.Size() != 15+380+380 {
		t.Errorf("total size=%d want 775", s.Size())
	}
}

func TestConfigCountsTriplet(t *testing.T) {
	s := NewSpace(testNocc, testNorb, nil, 0, Triplet)

	// No |ii>; |ij| (i>j) = 10 → main block 10.
	if s.BeginIJ != 0 {
		t.Errorf("BeginIJ=%d want 0", s.BeginIJ)
	}
	if s.MainBlockSize() != 10 {
		t.Errorf("main block=%d want 10", s.MainBlockSize())
	}
	// |ijkr,T>: 10 groups × mult(3) × 19 = 570.
	ijkrDim := s.Size() - s.BeginIJK
	if ijkrDim != 10*3*testNvir {
		t.Errorf("|ijkr> dim=%d want %d", ijkrDim, 10*3*testNvir)
	}
	if s.Size() != 10+380+570 {
		t.Errorf("total size=%d want 960", s.Size())
	}
}

// TestGroupBoundaries checks the group-start arrays are consistent strides.
func TestGroupBoundaries(t *testing.T) {
	s := NewSpace(testNocc, testNorb, nil, 0, Singlet)

	if s.JII[0] != s.BeginJII {
		t.Errorf("JII[0]=%d want BeginJII=%d", s.JII[0], s.BeginJII)
	}
	if s.IJK[0] != s.BeginIJK {
		t.Errorf("IJK[0]=%d want BeginIJK=%d", s.IJK[0], s.BeginIJK)
	}
	// Each type-I group spans exactly nvir rows.
	for m := range s.JII {
		end := s.BeginIJK
		if m+1 < len(s.JII) {
			end = s.JII[m+1]
		}
		if end-s.JII[m] != testNvir {
			t.Fatalf("JII group %d stride=%d want %d", m, end-s.JII[m], testNvir)
		}
	}
	// Each type-II group spans mult*nvir rows.
	for m := range s.IJK {
		end := s.Size()
		if m+1 < len(s.IJK) {
			end = s.IJK[m+1]
		}
		if end-s.IJK[m] != s.Mult*testNvir {
			t.Fatalf("IJK group %d stride=%d want %d", m, end-s.IJK[m], s.Mult*testNvir)
		}
	}
}

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
