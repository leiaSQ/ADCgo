package sip

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// adc4_eigab_gate_test.go — the two references that used to be missing from the tapes.
//
// theADCcode discards both before it writes anything a consumer could read: RSCRT1
// rewrites the diagonal tape (FT18) with only the first nh12 entries, throwing away the
// 3h2p effective diagonal, and the static self-energy is an *input* to adc_() that no
// tape records. ../ADC now dumps them (ab5.F -> FT19F001.ADC; egf.F -> SIGMA_STATIC.dat),
// which turns both of these from self-consistency checks into bit-exact value gates.
// FT21/FT18 are unchanged by that instrumentation.

// eigabTape is FT19F001.ADC: record 1 = IDIM NCOL NECORE N3H2P (4x int32),
// record 2 = N3H2P x float64, the 3h2p effective diagonal in ab5's pam/ELIM column
// order (same permutation FT21's 3h2p columns carry).
type eigabTape struct {
	idim, ncol, necore int
	diag               []float64
}

func readTapeEigab(t *testing.T, fn string) eigabTape {
	t.Helper()
	d, err := os.ReadFile(fn)
	if err != nil {
		t.Fatal(err)
	}
	le := binary.LittleEndian
	ln := int(int32(le.Uint32(d)))
	hdr := d[4 : 4+ln]
	out := eigabTape{
		idim:   int(int32(le.Uint32(hdr[0:]))),
		ncol:   int(int32(le.Uint32(hdr[4:]))),
		necore: int(int32(le.Uint32(hdr[8:]))),
	}
	n3 := int(int32(le.Uint32(hdr[12:])))
	off := 4 + ln + 4
	ln = int(int32(le.Uint32(d[off:])))
	off += 4
	if ln/8 != n3 {
		t.Fatalf("FT19 %s: header says %d 3h2p entries, record holds %d", fn, n3, ln/8)
	}
	out.diag = make([]float64, n3)
	for i := range out.diag {
		out.diag[i] = math.Float64frombits(le.Uint64(d[off+i*8:]))
	}
	return out
}

// TestADC4EigabGate is the bit-exact value gate for the 3h2p effective diagonal (WERT3).
// theADCcode's EIGAB is the 0th-order orbital-energy sum plus the 5th-order 3h2p-CI
// diagonal correction; ADCgo's sat3Diag reproduces it when WERT3 is on. The reference
// permutes 3h2p columns (ab5 pam/ELIM), so — exactly as the WERT2 coupling is compared —
// the diagonal is compared as a sorted multiset. ELIM's center-of-gravity fold is a
// truncation device inactive below MAXSTA (=10000), and both sectors are far under it,
// so the fold is a no-op here and the two must agree elementwise up to that permutation.
func TestADC4EigabGate(t *testing.T) {
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)

	for _, tc := range []struct {
		name            string
		dir             string
		sym             int
		size, n2h1p, n3 int
	}{
		{"A1", "adc4_a1_tape", 0, 1712, 46, 1665},
		{"B2", "adc4_b2_tape", 3, 1688, 42, 1646},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fn := filepath.Join("..", "..", "..", "testdata", "reference", tc.dir, "FT19F001.ADC")
			if _, err := os.Stat(fn); err != nil {
				t.Skipf("EIGAB tape unavailable: %v", err)
			}
			ref := readTapeEigab(t, fn)
			if ref.idim != tc.size || len(ref.diag) != tc.n3 {
				t.Fatalf("tape dims = (idim %d, 3h2p %d), want (%d, %d)",
					ref.idim, len(ref.diag), tc.size, tc.n3)
			}
			// ncol is the 2h1p count; necore the 1h count. Guards that we loaded the
			// sector we think we did.
			if ref.ncol != tc.n2h1p {
				t.Fatalf("tape 2h1p count = %d, want %d", ref.ncol, tc.n2h1p)
			}

			sp := NewSpace4(nocc, d.NORB, d.OrbSym, tc.sym, []int{0})
			if sp.Size() != tc.size {
				t.Fatalf("space size = %d, want %d", sp.Size(), tc.size)
			}
			mx := New(sp, integrals.New(d, nocc, d.OrbSym), eps, 4, backend.Gonum{})
			mx.SetWert3(true)
			got := mx.sat3Diag()
			if len(got) != len(ref.diag) {
				t.Fatalf("3h2p diagonal length = %d, want %d", len(got), len(ref.diag))
			}

			rv := append([]float64(nil), ref.diag...)
			mv := append([]float64(nil), got...)
			sort.Float64s(rv)
			sort.Float64s(mv)
			var maxd float64
			for i := range rv {
				if dd := math.Abs(rv[i] - mv[i]); dd > maxd {
					maxd = dd
				}
			}
			if maxd > 1e-12 {
				t.Errorf("3h2p effective diagonal (EIGAB/WERT3) multiset max diff %g exceeds 1e-12", maxd)
			}
			t.Logf("EIGAB gate %s: %d 3h2p entries, multiset maxdiff=%.2e", tc.name, len(rv), maxd)
		})
	}
}

// readSigmaStatic parses SIGMA_STATIC.dat (egf.F): a count line NECORE, then one row
// per lower-triangle element "i j ni nj value" with the value in a.u. — Σ as actually
// applied to the 1h block, i.e. after egf.F's own eV→a.u. conversion (FAKTOR =
// 27.211606, not the 27.211396 the C++ caller used on the way in) and after the SIGMPH
// 2p1h contribution. Reconstructing Σ from the C++ input instead leaves an ~9e-8 residual.
func readSigmaStatic(t *testing.T, fn string) map[[2]int]float64 {
	t.Helper()
	f, err := os.Open(fn)
	if err != nil {
		t.Skipf("static self-energy dump unavailable: %v", err)
	}
	defer f.Close()
	var n int
	if _, err := fmt.Fscan(f, &n); err != nil {
		t.Fatalf("SIGMA_STATIC.dat: bad header: %v", err)
	}
	out := make(map[[2]int]float64)
	for range n * (n + 1) / 2 {
		var i, j, ni, nj int
		var v float64
		if _, err := fmt.Fscan(f, &i, &j, &ni, &nj, &v); err != nil {
			t.Fatalf("SIGMA_STATIC.dat: bad row: %v", err)
		}
		out[[2]int{ni - 1, nj - 1}] = v // to 0-based absolute orbital indices
		out[[2]int{nj - 1, ni - 1}] = v // symmetric
	}
	return out
}

// TestADC4StaticSigmaGate closes the static self-energy on its real value. The 1h
// diagonal of the A1 tape is −ε_core − Σ(∞), with Σ supplied to adc_() from theADCcode's
// self-energy module (&self-energy infinite) and previously unavailable — the A1 gate had
// to *back-solve* Σ from the very tape entry it then checked, which asserts nothing about
// the value. Here Σ is read from theADCcode's own dump and must reproduce the tape.
func TestADC4StaticSigmaGate(t *testing.T) {
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	dir := filepath.Join("..", "..", "..", "testdata", "reference", "adc4_a1_tape")
	sig := readSigmaStatic(t, filepath.Join(dir, "SIGMA_STATIC.dat"))

	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace4(nocc, d.NORB, d.OrbSym, 0, []int{0}) // A1, core = orb 0
	mx := New(sp, integrals.New(d, nocc, d.OrbSym), eps, 4, backend.Gonum{})

	sigma := sig[[2]int{0, 0}]
	if sigma == 0 {
		t.Fatal("no Σ(0,0) in the dump — core orbital mismatch")
	}
	mx.SetStaticSelfEnergy(func(i, j int) float64 { return sig[[2]int{i, j}] })

	want := readTapeDiag(t, filepath.Join(dir, "FT18F001.ADC"))[0]
	got := mx.BuildMatrix().At(0, 0)
	if dd := math.Abs(got - want); dd > 1e-12 {
		t.Errorf("1h diagonal with theADCcode's Σ = %.15g, tape = %.15g (diff %g)", got, want, dd)
	}
	// Bare −ε_core must NOT reproduce the tape, or the gate is vacuous.
	if math.Abs(-eps[0]-want) < 1e-6 {
		t.Fatal("bare −ε_core already matches the tape — Σ is not being exercised")
	}
	t.Logf("static Σ gate: Σ(1,1) = %.12g Ha, 1h diagonal matches tape to %.2e", sigma, math.Abs(got-want))
}
