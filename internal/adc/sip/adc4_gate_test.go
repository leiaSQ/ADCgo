package sip

import (
	"encoding/binary"
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

// adc4_gate_test.go — the matched-integral gate for CVS IP-ADC(4) (Track A / A2.6).
//
// It builds the ADC(4) B2 secular matrix on the DZP integrals that theADCcode itself
// used, and compares against theADCcode's own matrix tape (testdata/reference/
// adc4_b2_tape, see its README). B2 has no core hole in its irrep, so the matrix is
// 42 (2h1p) + 1646 (3h2p) with no 1h main block: the tape exercises the 2h1p/2h1p
// block (WERT1, 3rd+4th order) and the 2h1p<->3h2p coupling (WERT2). The reference
// reorders 3h2p columns internally (ab5.F pam/ELIM, eigenvalue-invariant), so the
// coupling is compared per-row by sorted multiset. The 3h2p effective diagonal
// (EIGAB) is not on the tape; the 1h KOPP couplings need an A1 tape — both are out of
// scope for this fixture.

const tapeDir = "../../../testdata/reference/adc4_b2_tape"

// TestADC4MatchedGate is bit-exact against theADCcode on matched integrals.
func TestADC4MatchedGate(t *testing.T) {
	fc := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(fc)
	if err != nil {
		t.Skipf("matched fcidump unavailable: %v", err)
	}
	off := filepath.Join(tapeDir, "FT21F001.ADC")
	dia := filepath.Join(tapeDir, "FT18F001.ADC")
	if _, err := os.Stat(off); err != nil {
		t.Skipf("reference tape unavailable: %v", err)
	}

	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace4(nocc, d.NORB, d.OrbSym, 3, []int{0}) // B2 (0-based 3), core = orb 0
	if got, want := sp.Size(), 1688; got != want {
		t.Fatalf("B2 dimension = %d, want %d", got, want)
	}
	if n2 := sp.Begin3h2p - sp.BeginSat; n2 != 42 {
		t.Fatalf("2h1p count = %d, want 42", n2)
	}
	mx := New(sp, integrals.New(d, nocc, d.OrbSym), eps, 4, backend.Gonum{})
	M := mx.BuildMatrix()
	n := sp.Size()
	b := sp.Begin3h2p

	// Reference matrix from the tape (symmetric): off-diagonals + 2h1p diagonal.
	Ref := make([][]float64, n)
	for i := range Ref {
		Ref[i] = make([]float64, n)
	}
	rows, cols, vals := readTapeOff(t, off)
	for k := range rows {
		i, j := rows[k]-1, cols[k]-1
		Ref[i][j], Ref[j][i] = vals[k], vals[k]
	}
	for i, dv := range readTapeDiag(t, dia) {
		Ref[i][i] = dv
	}

	// 2h1p/2h1p block: rows and cols unpermuted -> element-wise bit-exact.
	var maxd float64
	for i := range b {
		for j := range b {
			if dd := math.Abs(Ref[i][j] - M.At(i, j)); dd > maxd {
				maxd = dd
			}
		}
	}
	if maxd > 1e-12 {
		t.Errorf("2h1p/2h1p block max diff %g exceeds 1e-12 (WERT1 3rd+4th order)", maxd)
	}

	// 2h1p<->3h2p coupling: reference permutes 3h2p columns (pam), so compare each
	// 2h1p row's coupling multiset to the 3h2p space.
	var worstMS float64
	for i := range b {
		var rv, mv []float64
		for j := b; j < n; j++ {
			if Ref[i][j] != 0 {
				rv = append(rv, Ref[i][j])
			}
			if v := M.At(i, j); v != 0 {
				mv = append(mv, v)
			}
		}
		if len(rv) != len(mv) {
			t.Fatalf("row %d: coupling nnz ref=%d mine=%d", i, len(rv), len(mv))
		}
		sort.Float64s(rv)
		sort.Float64s(mv)
		for k := range rv {
			if dd := math.Abs(rv[k] - mv[k]); dd > worstMS {
				worstMS = dd
			}
		}
	}
	if worstMS > 1e-12 {
		t.Errorf("2h1p<->3h2p coupling multiset max diff %g exceeds 1e-12 (WERT2)", worstMS)
	}

	// 3h2p block carries no off-diagonal in the reference (diagonal-only).
	for i := b; i < n; i++ {
		for j := b; j < n; j++ {
			if i != j && Ref[i][j] != 0 {
				t.Fatalf("unexpected reference 3h2p off-diagonal at (%d,%d)=%g", i, j, Ref[i][j])
			}
		}
	}
	t.Logf("matched gate: 2h1p block maxdiff=%.2e, coupling multiset maxdiff=%.2e", maxd, worstMS)
}

func readTapeOff(t *testing.T, fn string) (rows, cols []int, vals []float64) {
	t.Helper()
	d, err := os.ReadFile(fn)
	if err != nil {
		t.Fatal(err)
	}
	le := binary.LittleEndian
	for off := 0; off < len(d); {
		ln := int(int32(le.Uint32(d[off:])))
		off += 4
		body := d[off : off+ln]
		off += ln + 4
		cnt := int(int32(le.Uint32(body[16000:])))
		for k := range cnt {
			vals = append(vals, math.Float64frombits(le.Uint64(body[k*8:])))
			rows = append(rows, int(int32(le.Uint32(body[8000+k*4:]))))
			cols = append(cols, int(int32(le.Uint32(body[12000+k*4:]))))
		}
	}
	return
}

func readTapeDiag(t *testing.T, fn string) []float64 {
	t.Helper()
	d, err := os.ReadFile(fn)
	if err != nil {
		t.Fatal(err)
	}
	le := binary.LittleEndian
	ln := int(int32(le.Uint32(d))) // header record
	off := 4 + ln + 4
	ln = int(int32(le.Uint32(d[off:])))
	off += 4
	out := make([]float64, ln/8)
	for i := range out {
		out[i] = math.Float64frombits(le.Uint64(d[off+i*8:]))
	}
	return out
}
