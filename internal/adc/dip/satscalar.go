package dip

import (
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// satscalar.go — the per-output-scalar form of the 3h1p↔3h1p satellite apply.
//
// The host block applier (matfree.go newSatelliteMatFree) recomputes whole nvR×nvC panels
// and GEMVs them in two symmetric passes. That is the fast CPU path, but it is NOT the shape
// a GPU wants: a CUDA thread owns ONE output scalar and must accumulate its row of the
// operator directly. This file is that shape — one output 3h1p configuration (group × spin
// part × virtual orbital) sums Σ_C M[R,C]·in[C] over the candidate columns C — and it is the
// exact algorithm the CUDA kernel (adc2dip_kernels.cu, Phase C) implements thread-per-row.
//
// It needs no transpose pass: like the SIP c22 kernel's lo=min(r,c) orientation, each
// element is evaluated in the same orientation the dense assemble used (the block whose ROW
// is the higher-indexed group, IJK-as-row for the mixed ijkMLL block), so the value returned
// is M[R,C] directly — symmetry is honoured by construction, not by a second sweep.
//
// The candidate columns of R are pruned to the groups sharing an occupied index with R (the
// Kronecker-δ necessary condition, satBuckets) — the same O(G·k) pruning the block applier
// uses. TestSatelliteScalarApplyEqualsDense pins the whole apply against the dense operator;
// together with TestSatelliteScalarMatchesDense (per-entry) it is the host-side proof the
// CUDA kernel's design is correct before it ever runs on a GPU.

// satScalarPlan holds the config struct-of-arrays and occ buckets the per-scalar apply reads.
// The group arrays are indexed by group id; the row arrays by 3h1p offset (global index minus
// the main-block size). This is exactly the data the device path uploads once per sector.
type satScalarPlan struct {
	mx    *Matrix
	main  int
	parts int

	// Per-group metadata (JII = type I, IJK = type II).
	jCfg []Config // group representative config (for the Elem occ/δ pattern)
	jSt  []int    // group start global index
	jVir [][]int  // absolute virtual orbitals in block order
	iCfg []Config
	iSt  []int
	iVir [][]int

	// Per-row metadata over the 3h1p region (index = global - main).
	rTyp  []int8  // 0 = JII, 1 = IJK
	rGrp  []int32 // group id
	rPart []int32 // spin part
	rVir  []int32 // absolute virtual orbital

	bk satBuckets
}

// buildSatScalarPlan assembles the SoA + buckets for the per-scalar satellite apply.
func (mx *Matrix) buildSatScalarPlan() *satScalarPlan {
	sp := mx.sp
	p := &satScalarPlan{mx: mx, main: sp.BeginJII, parts: sp.Mult, bk: mx.buildSatBuckets()}

	n := sp.Size()
	p.rTyp = make([]int8, n-p.main)
	p.rGrp = make([]int32, n-p.main)
	p.rPart = make([]int32, n-p.main)
	p.rVir = make([]int32, n-p.main)

	for g, r0 := range sp.JII {
		cfg := sp.Configs[r0]
		vir := mx.blk.virOrbs(mx.blk.virSym(cfg))
		p.jCfg = append(p.jCfg, cfg)
		p.jSt = append(p.jSt, r0)
		p.jVir = append(p.jVir, vir)
		for a, orb := range vir {
			ri := r0 - p.main + a
			p.rTyp[ri], p.rGrp[ri], p.rPart[ri], p.rVir[ri] = 0, int32(g), 0, int32(orb)
		}
	}
	for g, r0 := range sp.IJK {
		cfg := sp.Configs[r0]
		vir := mx.blk.virOrbs(mx.blk.virSym(cfg))
		nv := len(vir)
		p.iCfg = append(p.iCfg, cfg)
		p.iSt = append(p.iSt, r0)
		p.iVir = append(p.iVir, vir)
		for part := range p.parts {
			for a, orb := range vir {
				ri := r0 - p.main + part*nv + a
				p.rTyp[ri], p.rGrp[ri], p.rPart[ri], p.rVir[ri] = 1, int32(g), int32(part), int32(orb)
			}
		}
	}
	return p
}

// elem returns the oriented operator element M[R,C] for a row scalar R (type rTyp, group
// rGrp, part rPart, virtual rVir) and a column scalar C (type cTyp, group cGrp, part cPart,
// virtual cVir). The block is evaluated in the orientation the dense assemble used, so the
// returned value is M[R,C] with no transpose bookkeeping.
func (p *satScalarPlan) elem(rTyp int8, rGrp int32, rPart, rVir int, cTyp int8, cGrp int32, cPart, cVir int) float64 {
	blk := p.mx.blk
	switch {
	case rTyp == 0 && cTyp == 0: // jiiLKK: row is the higher-indexed group
		if rGrp >= cGrp {
			return blk.jiiLKKElem(p.jCfg[rGrp], p.jCfg[cGrp], rVir, cVir)
		}
		return blk.jiiLKKElem(p.jCfg[cGrp], p.jCfg[rGrp], cVir, rVir)
	case rTyp == 1 && cTyp == 0: // ijkMLL: R (IJK) is the row, C (JII) the column
		return blk.ijkMLLElem(p.iCfg[rGrp], p.jCfg[cGrp], rPart, rVir, cVir)
	case rTyp == 0 && cTyp == 1: // ijkMLL transposed: C (IJK) is the row, R (JII) the column
		return blk.ijkMLLElem(p.iCfg[cGrp], p.jCfg[rGrp], cPart, cVir, rVir)
	default: // ijkLMN: row is the higher-indexed group
		if rGrp >= cGrp {
			return blk.ijkLMNElem(p.iCfg[rGrp], p.iCfg[cGrp], rPart, rVir, cPart, cVir)
		}
		return blk.ijkLMNElem(p.iCfg[cGrp], p.iCfg[rGrp], cPart, cVir, rPart, rVir)
	}
}

// apply accumulates the satellite region's contribution into out (+= semantics) on a HostData
// backend, resolving the panels to host slices and delegating to applyHost.
func (p *satScalarPlan) apply(in, out backend.BlockView) {
	hd := p.mx.be.(backend.HostData)
	p.applyHost(hd.HostSlice(in.V), hd.HostSlice(out.V), in.Cols, in.Ld, out.Ld)
}

// applyHost is the core per-scalar accumulation on plain host slices (xin read, yout += ),
// one output scalar at a time — the CPU twin of the CUDA kernel and the shared kernel for both
// the HostData apply and the distributed gather-apply-scatter path (matfree_dist.go). Rows are
// chunked across the worker pool; each worker owns a disjoint band of output rows, so there is
// no reduction or locking. b is the panel column count; ldi/ldo the input/output leading dims.
func (p *satScalarPlan) applyHost(xin, yout []float64, b, ldi, ldo int) {
	main := p.main
	nsat := len(p.rTyp)
	njii, nijk := len(p.jCfg), len(p.iCfg)

	parallel.Chunks(nsat, parallel.ChunkWorkers(nsat), func(_, lo, hi int) {
		stampJ, stampI := newStamp(njii), newStamp(nijk)
		var candJ, candI []int32
		gen := int32(0)
		for ri := lo; ri < hi; ri++ {
			R := main + ri
			rTyp, rGrp, rPart, rVir := p.rTyp[ri], p.rGrp[ri], int(p.rPart[ri]), int(p.rVir[ri])
			var rOcc [3]int
			var rn int
			if rTyp == 0 {
				rOcc, rn = p.jCfg[rGrp].Occ, 2
			} else {
				rOcc, rn = p.iCfg[rGrp].Occ, 3
			}

			// Candidate column groups: those sharing an occupied index with R. A fresh
			// generation per row keeps the dedup stamp valid without clearing it.
			gen++
			candJ = gatherCand(rOcc[:rn], p.bk.jii, stampJ, gen, candJ)
			candI = gatherCand(rOcc[:rn], p.bk.ijk, stampI, gen, candI)

			// JII column groups (single spin part).
			for _, cg := range candJ {
				vir := p.jVir[cg]
				cst := p.jSt[cg]
				for cb, sb := range vir {
					g := p.elem(rTyp, rGrp, rPart, rVir, 0, cg, 0, sb)
					if g == 0 {
						continue
					}
					C := cst + cb
					for j := range b {
						yout[R+j*ldo] += g * xin[C+j*ldi]
					}
				}
			}
			// IJK column groups (parts spin parts).
			for _, cg := range candI {
				vir := p.iVir[cg]
				nv := len(vir)
				cst := p.iSt[cg]
				for cpart := range p.parts {
					for cb, sb := range vir {
						g := p.elem(rTyp, rGrp, rPart, rVir, 1, cg, cpart, sb)
						if g == 0 {
							continue
						}
						C := cst + cpart*nv + cb
						for j := range b {
							yout[R+j*ldo] += g * xin[C+j*ldi]
						}
					}
				}
			}
		}
	})
}
