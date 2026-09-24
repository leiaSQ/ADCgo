package dip

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestSatelliteScalarMatchesDense pins the per-entry scalar functions (satelem.go) against the
// dense block methods: for every satellite group pair and every (spin-part, virtual) entry, the
// scalar jiiLKKElem/ijkMLLElem/ijkLMNElem must reproduce the dense block cell. These scalars are
// the single source of truth the CUDA kernel transcribes, so this is the host-side guard that a
// transcription slip (here or, by extension, in the kernel) is caught without a GPU.
func TestSatelliteScalarMatchesDense(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		mx := New(sp, ints, eps, be)
		blk := mx.blk
		parts := 2
		if spin == Triplet {
			parts = 3
		}

		// cmp asserts scalar == dense[rowIdx,colIdx], allowing only reassociation error.
		cmp := func(name string, gr, gc int, m backend.Mat, rowIdx, colIdx int, scalar float64) {
			want := m.At(rowIdx, colIdx)
			if math.Abs(scalar-want) > 1e-12*(1+math.Abs(want)) {
				t.Errorf("spin=%v sym=%d %s(g%d,g%d)[%d,%d]: scalar=%.15g dense=%.15g", spin, sym, name, gr, gc, rowIdx, colIdx, scalar, want)
			}
		}

		// jiiLKK: 1 spin part per side, nvR×nvC.
		for gr := range sp.JII {
			for gc := range sp.JII {
				rc, cc := sp.Configs[sp.JII[gr]], sp.Configs[sp.JII[gc]]
				m, ok := blk.jiiLKK(rc, cc)
				if !ok {
					continue
				}
				rowOrbs, colOrbs := blk.virOrbs(blk.virSym(rc)), blk.virOrbs(blk.virSym(cc))
				for a, ra := range rowOrbs {
					for b, sb := range colOrbs {
						cmp("jiiLKK", gr, gc, m, a, b, blk.jiiLKKElem(rc, cc, ra, sb))
					}
				}
			}
		}

		// ijkMLL: parts spin parts on the row side, 1 on the column side.
		for gr := range sp.IJK {
			for gc := range sp.JII {
				rc, cc := sp.Configs[sp.IJK[gr]], sp.Configs[sp.JII[gc]]
				m, ok := blk.ijkMLL(rc, cc)
				if !ok {
					continue
				}
				rowOrbs, colOrbs := blk.virOrbs(blk.virSym(rc)), blk.virOrbs(blk.virSym(cc))
				nvR := len(rowOrbs)
				for pr := range parts {
					for a, ra := range rowOrbs {
						for b, sb := range colOrbs {
							cmp("ijkMLL", gr, gc, m, pr*nvR+a, b, blk.ijkMLLElem(rc, cc, pr, ra, sb))
						}
					}
				}
			}
		}

		// ijkLMN: parts spin parts on both sides.
		for gr := range sp.IJK {
			for gc := range sp.IJK {
				rc, cc := sp.Configs[sp.IJK[gr]], sp.Configs[sp.IJK[gc]]
				m, ok := blk.ijkLMN(rc, cc)
				if !ok {
					continue
				}
				rowOrbs, colOrbs := blk.virOrbs(blk.virSym(rc)), blk.virOrbs(blk.virSym(cc))
				nvR, nvC := len(rowOrbs), len(colOrbs)
				for pr := range parts {
					for pc := range parts {
						for a, ra := range rowOrbs {
							for b, sb := range colOrbs {
								cmp("ijkLMN", gr, gc, m, pr*nvR+a, pc*nvC+b, blk.ijkLMNElem(rc, cc, pr, ra, pc, sb))
							}
						}
					}
				}
			}
		}
	})
}

// TestApplyBlockSatellite checks that the gated apply equals the full operator with the 2h
// main block and the 2h↔3h1p couplings zeroed — i.e. only the 3h1p↔3h1p satellite blocks
// act. This is the operator half of the lanczos.SolveLowMem Mode B Tarantelli gate.
func TestApplyBlockSatellite(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)
	be := backend.Gonum{}
	rng := rand.New(rand.NewSource(7))

	tested := 0
	for _, spin := range []Spin{Singlet, Triplet} {
		for sym := range 4 {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			n := sp.Size()
			main := sp.MainBlockSize()
			if n == 0 || n == main { // need a satellite space to test
				continue
			}
			mx := New(sp, ints, eps, be)

			// Dense reference: M with rows/cols in the main space zeroed out.
			M := mx.BuildMatrix()
			for i := range n {
				for j := range n {
					if i < main || j < main {
						M.Set(i, j, 0)
					}
				}
			}

			// Random n-vector; compare M_sat·x (dense) to ApplyBlockSatellite (1 column).
			x := make([]float64, n)
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			want := M.MulVec(x)

			in := backend.BlockView{V: be.Upload(x), Rows: n, Cols: 1, Ld: n}
			out := backend.BlockView{V: be.Alloc(n), Rows: n, Cols: 1, Ld: n}
			mx.ApplyBlockSatellite(out, in)
			got := be.Download(out.V)

			var maxDiff, scale float64
			for i := range n {
				scale = math.Max(scale, math.Abs(want[i]))
				maxDiff = math.Max(maxDiff, math.Abs(want[i]-got[i]))
			}
			// Also assert the main-space rows of the output are exactly zero (the gate must
			// not touch the main space).
			for i := range main {
				if got[i] != 0 {
					t.Errorf("spin=%v sym=%d: satellite apply wrote main row %d = %g (want 0)", spin, sym, i, got[i])
				}
			}
			if rel := maxDiff / math.Max(scale, 1e-300); rel > 1e-12 {
				t.Errorf("spin=%v sym=%d n=%d main=%d: satellite apply relative diff %.3e", spin, sym, n, main, rel)
			}
			tested++
		}
	}
	if tested == 0 {
		t.Fatal("no sectors with a satellite space exercised")
	}
	t.Logf("%d sectors: ApplyBlockSatellite == masked dense", tested)
}
