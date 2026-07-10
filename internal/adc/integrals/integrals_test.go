package integrals

import (
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

func load(t *testing.T) *Store {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	return New(d, mp.NOcc(d), d.OrbSym)
}

// virsOf returns the virtual positions of symmetry group sym in ascending order —
// the ordering the accessors and the dip configuration enumeration must share.
func (s *Store) virsOf(sym int) []int {
	var out []int
	for rp := range s.nvir {
		if s.OrbIrrep(s.nocc+rp) == sym {
			out = append(out, rp)
		}
	}
	return out
}

// TestGroupsPartitionVirtuals: the symmetry groups tile the virtual space exactly.
func TestGroupsPartitionVirtuals(t *testing.T) {
	s := load(t)
	total := 0
	for sym := range s.NSym() {
		total += s.SizeVirGroup(sym)
	}
	if total != s.NVir() {
		t.Fatalf("group sizes sum to %d, want NVir=%d", total, s.NVir())
	}
}

// TestBlockConventions pins the index conventions of the symmetry-restricted
// V/A/B accessors against the dense (pq|rs) store: each block runs over the
// correct virtual groups, in ascending-orbital order.
func TestBlockConventions(t *testing.T) {
	s := load(t)
	i, j, k := 0, 2, 1 // arbitrary occupied indices (nocc=5)
	nocc := s.NOcc()

	for sym := range s.NSym() {
		rs := s.virsOf(sym) // r group for V (and column group for A/B)
		v := s.V(i, j, k, sym)
		if len(v) != len(rs) {
			t.Fatalf("V sym %d length %d, want %d", sym, len(v), len(rs))
		}
		for idx, rp := range rs {
			if want := s.Eri(nocc+rp, k, i, j); v[idx] != want { // (rk|ij)
				t.Fatalf("V(%d,%d,%d;%d)[%d]=%g want %g", i, j, k, sym, idx, v[idx], want)
			}
		}

		// A/B: rows over the group fixed by symmetry, cols over sym.
		rowSym := symProduct(s.OrbIrrep(i), s.OrbIrrep(j), sym)
		rows := s.virsOf(rowSym)
		a := s.A(i, j, sym)
		b := s.B(i, j, sym)
		if a.Rows != len(rows) || a.Cols != len(rs) {
			t.Fatalf("A sym %d shape %dx%d, want %dx%d", sym, a.Rows, a.Cols, len(rows), len(rs))
		}
		for ri, rp := range rows {
			for ci, sp := range rs {
				if got, want := a.At(ri, ci), s.Eri(nocc+rp, i, nocc+sp, j); got != want {
					t.Fatalf("A(%d,%d;%d)[%d,%d]=%g want %g", i, j, sym, ri, ci, got, want)
				}
				if got, want := b.At(ri, ci), s.Eri(nocc+rp, nocc+sp, i, j); got != want {
					t.Fatalf("B(%d,%d;%d)[%d,%d]=%g want %g", i, j, sym, ri, ci, got, want)
				}
			}
		}
	}
}
