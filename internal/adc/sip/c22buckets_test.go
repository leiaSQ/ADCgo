package sip

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

func c22Fixture(t *testing.T, dump string, sym int) (*Space, *elements) {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join("..", "..", "..", "testdata", dump))
	if err != nil {
		t.Skipf("fcidump unavailable: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace(nocc, d.NORB, d.OrbSym, sym)
	if sp.MainBlockSize() == 0 || sp.Size() == sp.MainBlockSize() {
		t.Skipf("%s sym=%d has no satellite space", dump, sym)
	}
	return sp, newElements(sp, integrals.New(d, nocc, d.OrbSym), eps, 3)
}

// TestC22GateIsNecessary is the safety proof for the candidate buckets: EVERY nonzero
// c22off element must satisfy the gate the buckets encode (share the particle, or share a
// hole). A gate that is merely usually-true would silently drop operator contributions and
// shift the ionization spectrum without failing anything, so this checks the full n2×n2
// square exhaustively rather than sampling.
func TestC22GateIsNecessary(t *testing.T) {
	for _, dump := range []string{"h2o.fcidump", "h2o_dzp.fcidump"} {
		for sym := range 4 {
			sp, el := c22Fixture(t, dump, sym)
			rows := sp.Configs[sp.BeginSat:]
			n2 := len(rows)
			checked, nonzero := 0, 0
			for r := range n2 {
				rc := rows[r]
				for c := range n2 {
					if c == r {
						continue
					}
					lo, hi := rc, rows[c]
					if c < r {
						lo, hi = rows[c], rc
					}
					g := el.c22off(lo, hi)
					checked++
					if g == 0 {
						continue
					}
					nonzero++
					cc := rows[c]
					shareHole := rc.Occ[0] == cc.Occ[0] || rc.Occ[0] == cc.Occ[1] ||
						rc.Occ[1] == cc.Occ[0] || rc.Occ[1] == cc.Occ[1]
					if !shareHole && rc.Vir != cc.Vir {
						t.Fatalf("%s sym=%d: c22off(r=%d %v, c=%d %v) = %g is NONZERO but the "+
							"pair shares neither a hole nor a particle — the bucket gate would "+
							"drop it", dump, sym, r, rc, c, cc, g)
					}
				}
			}
			if nonzero == 0 {
				t.Fatalf("%s sym=%d: no nonzero off-diagonal elements in %d pairs; the test "+
					"would pass vacuously", dump, sym, checked)
			}
			t.Logf("%s sym=%d: %d/%d off-diagonal pairs nonzero, all gated", dump, sym, nonzero, checked)
		}
	}
}

// TestC22BucketsCoverEveryNonzeroColumn pins the merge itself: for every row the candidate
// list must be ascending, deduplicated, and a superset of the columns with a nonzero element.
// Ascending order is what makes the bucketed applier bit-exact — the applier already skips
// g == 0, so walking a superset in the original order replays the identical accumulation.
func TestC22BucketsCoverEveryNonzeroColumn(t *testing.T) {
	sp, el := c22Fixture(t, "h2o.fcidump", 0)
	rows := sp.Configs[sp.BeginSat:]
	n2 := len(rows)
	bk := buildC22Buckets(rows, sp.Nocc, sp.Nvir)

	var cand []int32
	for r := range n2 {
		rc := rows[r]
		cand = bk.candidates(cand, rc.Occ[0], rc.Occ[1], rc.Vir, r)
		inCand := make(map[int]bool, len(cand))
		prev := int32(-1)
		for _, c := range cand {
			if c <= prev {
				t.Fatalf("row %d: candidates not strictly ascending at %d (prev %d)", r, c, prev)
			}
			prev = c
			inCand[int(c)] = true
		}
		for c := range n2 {
			if c == r {
				continue
			}
			lo, hi := rc, rows[c]
			if c < r {
				lo, hi = rows[c], rc
			}
			if el.c22off(lo, hi) != 0 && !inCand[c] {
				t.Fatalf("row %d: column %d has a nonzero element but is not a candidate", r, c)
			}
		}
	}
}

// TestC22BucketedApplyMatchesFullScan runs the shipped matrix-free applier against a literal
// replica of the exhaustive n2² scan it replaces and requires BIT equality, not tolerance.
func TestC22BucketedApplyMatchesFullScan(t *testing.T) {
	for _, dump := range []string{"h2o.fcidump", "h2o_dzp.fcidump"} {
		sp, el := c22Fixture(t, dump, 0)
		main := sp.BeginSat
		rows := sp.Configs[main:]
		n2, n := len(rows), sp.Size()
		const cols = 3

		rng := rand.New(rand.NewSource(int64(n2)))
		hostIn := make([]float64, n*cols)
		for i := range hostIn {
			hostIn[i] = rng.NormFloat64()
		}

		// Reference: the original full-scan applier, verbatim.
		want := make([]float64, n*cols)
		for r := range n2 {
			rc := rows[r]
			for c := range n2 {
				var g float64
				switch {
				case c == r:
					g = el.c22diag(rc)
				case r < c:
					g = el.c22off(rc, rows[c])
				default:
					g = el.c22off(rows[c], rc)
				}
				if g == 0 {
					continue
				}
				for j := range cols {
					want[main+r+j*n] += g * hostIn[main+c+j*n]
				}
			}
		}

		be := backend.Gonum{}
		mx := New(sp, integrals.New(mustDump(t, dump), mp.NOcc(mustDump(t, dump)), mustDump(t, dump).OrbSym),
			mp.OrbitalEnergies(mustDump(t, dump), mp.NOcc(mustDump(t, dump))), 3, be)
		part := mx.newC22MatFreeO3()
		defer part.release()

		inV := be.Upload(hostIn)
		outBuf := be.Alloc(n * cols)
		be.Zero(outBuf)
		part.apply(
			backend.BlockView{V: inV, Rows: n, Cols: cols, Ld: n},
			backend.BlockView{V: outBuf, Rows: n, Cols: cols, Ld: n},
		)
		got := be.Download(outBuf)

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: element %d = %.17g, full scan = %.17g (diff %g) — the bucketed "+
					"applier must be BIT-identical to the exhaustive scan",
					dump, i, got[i], want[i], math.Abs(got[i]-want[i]))
			}
		}
	}
}

func mustDump(t *testing.T, dump string) *fcidump.Data {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join("..", "..", "..", "testdata", dump))
	if err != nil {
		t.Skipf("fcidump unavailable: %v", err)
	}
	return d
}
