package dip

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
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
