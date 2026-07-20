package dip

import (
	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// matfree.go — matrix-free application of the 3h1p↔3h1p satellite region.
//
// The block-sparse operator (matvec.go) assembles every block densely and reuses the
// resident copy each mat-vec. For a large DIP sector the satellite blocks (jiiLKK/ijkMLL/
// ijkLMN) are the overwhelming majority of that footprint — hundreds of GB to multiple TB —
// bigger than any single GPU (docs/dip_operator_memory.md). The matrix-free path recomputes
// each satellite block on the fly during the apply and never stores it, collapsing the
// resident footprint to the panels plus the tiny main+coupling blocks. This is the direct
// σ-build theADCcode uses (../ADC/adc2_dip, Adc2_matrix::operator()).
//
// It is affordable because every nonzero satellite block requires the row and column 3h1p
// configurations to share at least one occupied index (the Kronecker-δ guards in the block
// gates, blocks.go). So the applier visits, per group, only the candidate groups pulled from
// occupied-index buckets — O(G·k) instead of O(G²) group-pairs — recomputing and applying
// just those. Without this pruning the one-time O(G²) dense assemble would become an O(G²)
// cost paid on every mat-vec.
//
// The applier realizes the symmetric operator in two barrier-separated passes with disjoint
// per-worker output ownership: pass 1 accumulates the forward direction (out[row] += A·in[col])
// parallelized over row-groups; pass 2 accumulates the transpose (out[col] += Aᵀ·in[row], off
// -diagonal blocks only) parallelized over col-groups. No two workers ever write the same
// output band, so no reduction and no locking are needed. Host-only for now (a CUDA
// DeviceKernels twin is future work; see docs/dip_operator_memory.md).

// matFreePart is a satellite region applied by recomputing its blocks each mat-vec instead of
// storing them. apply accumulates the region's contribution into out (both directions), and
// runs after the dense GEMMs, so it uses += semantics. release frees any scratch (a no-op for
// the host applier, which reuses per-worker buffers within a call).
type matFreePart struct {
	apply   func(in, out backend.BlockView)
	release func()
}

// SetMatFree configures matrix-free assembly of the satellite region for this sector.
// budgetBytes is the dense-size threshold used by Auto (ignored for Off/On).
func (mx *Matrix) SetMatFree(mode matfree.Mode, budgetBytes int64) {
	mx.matFree = mode
	mx.matFreeBudget = budgetBytes
}

// matFreeSatellite reports whether the 3h1p↔3h1p satellite region is applied matrix-free for
// this sector. It is supported on a host backend (HostData, the block applier) and on a device
// backend that provides the DIP satellite kernel (DeviceKernels, the per-scalar CUDA applier);
// any other backend falls back to dense. Auto compares the satellite region's dense size
// against the budget.
func (mx *Matrix) matFreeSatellite() bool {
	if mx.matFree == matfree.Auto {
		return matfree.Decide(matfree.Auto, int64(mx.satelliteResidentBytes()), mx.matFreeBudget, mx.matFreeSupported())
	}
	return matfree.Decide(mx.matFree, 0, mx.matFreeBudget, mx.matFreeSupported())
}

// matFreeSupported reports whether the current backend can apply the satellite region
// matrix-free: a DeviceKernels device (per-scalar CUDA kernel), a row-partitioned backend
// (PanelScatterAdd, the -mgpu gather-apply-scatter path), or a HostData host (block applier).
// The order matters: the distributed backend also satisfies HostData through its embedded
// Gonum, so PanelScatterAdd must be tested before HostData or the host applier would call
// HostSlice on a distributed vector.
func (mx *Matrix) matFreeSupported() bool {
	if _, ok := mx.be.(backend.DeviceKernels); ok {
		return true
	}
	if _, ok := mx.be.(backend.PanelScatterAdd); ok {
		return true
	}
	_, host := mx.be.(backend.HostData)
	return host
}

// newSatelliteMatFreePart builds the satellite matrix-free applier for the current backend: the
// per-scalar CUDA kernel on a DeviceKernels device, the gather-apply-scatter path on a
// row-partitioned (PanelScatterAdd) backend, else the host block applier. Same ordering
// rationale as matFreeSupported (distributed satisfies HostData via embedded Gonum).
func (mx *Matrix) newSatelliteMatFreePart() matFreePart {
	if dk, ok := mx.be.(backend.DeviceKernels); ok {
		return mx.newSatelliteMatFreeDevice(dk)
	}
	if pg, ok := mx.be.(backend.PanelScatterAdd); ok {
		return mx.newSatelliteMatFreeDistributed(pg)
	}
	return mx.newSatelliteMatFree()
}

// satelliteResidentBytes returns the dense resident size (Σ rows·cols·8) of the 3h1p↔3h1p
// satellite region, computed from the cheap block gates (occupied-index equality + virtual
// group sizes) without evaluating any integrals or allocating any block. It walks the same
// group-pair structure the dense satelliteTasks assemble, so it is the exact satellite
// footprint — the number the pre-flight device-fit guard needs, obtainable in seconds where a
// full block build would take hours.
func (mx *Matrix) satelliteResidentBytes() uint64 {
	sp := mx.sp
	njii, nijk := len(sp.JII), len(sp.IJK)
	sums := make([]uint64, njii+nijk)
	parallel.Rows(njii+nijk, func(t int) {
		var e uint64
		if t < njii {
			gr := t
			rc := sp.Configs[sp.JII[gr]]
			for gc := 0; gc <= gr; gc++ {
				if rows, cols, ok := mx.blk.jiiLKKGate(rc, sp.Configs[sp.JII[gc]]); ok {
					e += uint64(rows) * uint64(cols)
				}
			}
		} else {
			gr := t - njii
			rc := sp.Configs[sp.IJK[gr]]
			for _, c0 := range sp.JII {
				if rows, cols, ok := mx.blk.ijkMLLGate(rc, sp.Configs[c0]); ok {
					e += uint64(rows) * uint64(cols)
				}
			}
			for gc := 0; gc <= gr; gc++ {
				if rows, cols, ok := mx.blk.ijkLMNGate(rc, sp.Configs[sp.IJK[gc]]); ok {
					e += uint64(rows) * uint64(cols)
				}
			}
		}
		sums[t] = e
	})
	var total uint64
	for _, s := range sums {
		total += s
	}
	return total * 8 // sizeof(float64)
}

// satBuckets maps an occupied-orbital index to the 3h1p groups (JII, IJK) that contain it.
// A nonzero satellite block requires a shared occupied index between its row and column
// groups, so these buckets are the candidate lists the applier prunes with.
type satBuckets struct {
	jii [][]int32 // occ index -> JII group indices (into sp.JII)
	ijk [][]int32 // occ index -> IJK group indices (into sp.IJK)
}

func (mx *Matrix) buildSatBuckets() satBuckets {
	sp := mx.sp
	bk := satBuckets{jii: make([][]int32, sp.Norb), ijk: make([][]int32, sp.Norb)}
	for gc, c0 := range sp.JII {
		cfg := sp.Configs[c0]
		bk.jii[cfg.Occ[0]] = append(bk.jii[cfg.Occ[0]], int32(gc))
		bk.jii[cfg.Occ[1]] = append(bk.jii[cfg.Occ[1]], int32(gc))
	}
	for gc, c0 := range sp.IJK {
		cfg := sp.Configs[c0]
		bk.ijk[cfg.Occ[0]] = append(bk.ijk[cfg.Occ[0]], int32(gc))
		bk.ijk[cfg.Occ[1]] = append(bk.ijk[cfg.Occ[1]], int32(gc))
		bk.ijk[cfg.Occ[2]] = append(bk.ijk[cfg.Occ[2]], int32(gc))
	}
	return bk
}

// newStamp returns a dedup-stamp slice initialized so no generation matches (generations are
// >= 0). gatherCand uses it to visit each candidate group at most once per query.
func newStamp(n int) []int32 {
	s := make([]int32, n)
	for i := range s {
		s[i] = -1
	}
	return s
}

// gatherCand collects into dst (reused, cleared first) the deduplicated group indices that
// share an occupied index with occs, using the per-worker stamp keyed by the unique gen of
// this query. A group appearing under several shared occ indices is visited once.
func gatherCand(occs []int, bucket [][]int32, stamp []int32, gen int32, dst []int32) []int32 {
	dst = dst[:0]
	for _, o := range occs {
		for _, gc := range bucket[o] {
			if stamp[gc] == gen {
				continue
			}
			stamp[gc] = gen
			dst = append(dst, gc)
		}
	}
	return dst
}

// gemvForward accumulates out[rowOff + r + j·ldo] += Σ_c block[r,c]·in[colOff + c + j·ldi]
// for every column j of the panel — the block applied to the column band, into the row band.
func gemvForward(block backend.Mat, rowOff, colOff int, xin, yout []float64, b, ldi, ldo int) {
	rows, cols := block.Rows, block.Cols
	for r := range rows {
		base := block.Data[r*cols : r*cols+cols]
		for j := range b {
			inOff := colOff + j*ldi
			var acc float64
			for c := range cols {
				acc += base[c] * xin[inOff+c]
			}
			yout[rowOff+r+j*ldo] += acc
		}
	}
}

// gemvTranspose accumulates out[colOff + c + j·ldo] += Σ_r block[r,c]·in[rowOff + r + j·ldi]
// — the block's transpose applied to the row band, into the column band.
func gemvTranspose(block backend.Mat, rowOff, colOff int, xin, yout []float64, b, ldi, ldo int) {
	rows, cols := block.Rows, block.Cols
	for j := range b {
		inOff := rowOff + j*ldi
		outOff := colOff + j*ldo
		for r := range rows {
			x := xin[inOff+r]
			if x == 0 {
				continue
			}
			base := block.Data[r*cols : r*cols+cols]
			for c := range cols {
				yout[outOff+c] += base[c] * x
			}
		}
	}
}

// newSatelliteMatFree builds the matrix-free applier for the 3h1p↔3h1p satellite region, the
// on-the-fly equivalent of the dense satelliteTasks. See the file comment for the two-pass,
// occupied-index-pruned design.
func (mx *Matrix) newSatelliteMatFree() matFreePart {
	sp := mx.sp
	hd := mx.be.(backend.HostData)
	bk := mx.buildSatBuckets()
	njii, nijk := len(sp.JII), len(sp.IJK)

	apply := func(in, out backend.BlockView) {
		xin := hd.HostSlice(in.V)
		yout := hd.HostSlice(out.V)
		b, ldi, ldo := in.Cols, in.Ld, out.Ld

		// --- Pass 1: forward (out[row] += A·in[col]), parallelized over row-groups. Each
		// worker owns its row-groups' output bands exclusively. ---

		// 1a: JII row-groups → jiiLKK over candidate JII col-groups (gc <= gr).
		parallel.Chunks(njii, parallel.ChunkWorkers(njii), func(_, lo, hi int) {
			stamp := newStamp(njii)
			var cand []int32
			for gr := lo; gr < hi; gr++ {
				r0 := sp.JII[gr]
				rc := sp.Configs[r0]
				cand = gatherCand(rc.Occ[:2], bk.jii, stamp, int32(gr), cand)
				for _, gc32 := range cand {
					gc := int(gc32)
					if gc > gr {
						continue
					}
					c0 := sp.JII[gc]
					if blk, ok := mx.blk.jiiLKK(rc, sp.Configs[c0]); ok {
						gemvForward(blk, r0, c0, xin, yout, b, ldi, ldo)
					}
				}
			}
		})

		// 1b: IJK row-groups → ijkMLL over candidate JII col-groups (all) + ijkLMN over
		// candidate IJK col-groups (gc <= gr). Both write the same IJK row band, done by the
		// one worker that owns it.
		parallel.Chunks(nijk, parallel.ChunkWorkers(nijk), func(_, lo, hi int) {
			stampJ, stampI := newStamp(njii), newStamp(nijk)
			var candJ, candI []int32
			for gr := lo; gr < hi; gr++ {
				r0 := sp.IJK[gr]
				rc := sp.Configs[r0]
				candJ = gatherCand(rc.Occ[:3], bk.jii, stampJ, int32(gr), candJ)
				for _, gc32 := range candJ {
					c0 := sp.JII[int(gc32)]
					if blk, ok := mx.blk.ijkMLL(rc, sp.Configs[c0]); ok {
						gemvForward(blk, r0, c0, xin, yout, b, ldi, ldo)
					}
				}
				candI = gatherCand(rc.Occ[:3], bk.ijk, stampI, int32(gr), candI)
				for _, gc32 := range candI {
					gc := int(gc32)
					if gc > gr {
						continue
					}
					c0 := sp.IJK[gc]
					if blk, ok := mx.blk.ijkLMN(rc, sp.Configs[c0]); ok {
						gemvForward(blk, r0, c0, xin, yout, b, ldi, ldo)
					}
				}
			}
		})

		// --- Pass 2: transpose (out[col] += Aᵀ·in[row]), off-diagonal blocks only,
		// parallelized over col-groups. Each worker owns its col-groups' output bands. ---

		// 2a: JII col-groups → jiiLKK from candidate JII row-groups (gr > gc) + ijkMLL from
		// candidate IJK row-groups (all; ijkMLL is never diagonal). Both write the JII col band.
		parallel.Chunks(njii, parallel.ChunkWorkers(njii), func(_, lo, hi int) {
			stampJ, stampI := newStamp(njii), newStamp(nijk)
			var candJ, candI []int32
			for gc := lo; gc < hi; gc++ {
				c0 := sp.JII[gc]
				cc := sp.Configs[c0]
				candJ = gatherCand(cc.Occ[:2], bk.jii, stampJ, int32(gc), candJ)
				for _, gr32 := range candJ {
					gr := int(gr32)
					if gr <= gc {
						continue
					}
					r0 := sp.JII[gr]
					if blk, ok := mx.blk.jiiLKK(sp.Configs[r0], cc); ok {
						gemvTranspose(blk, r0, c0, xin, yout, b, ldi, ldo)
					}
				}
				candI = gatherCand(cc.Occ[:2], bk.ijk, stampI, int32(gc), candI)
				for _, gr32 := range candI {
					r0 := sp.IJK[int(gr32)]
					if blk, ok := mx.blk.ijkMLL(sp.Configs[r0], cc); ok {
						gemvTranspose(blk, r0, c0, xin, yout, b, ldi, ldo)
					}
				}
			}
		})

		// 2b: IJK col-groups → ijkLMN from candidate IJK row-groups (gr > gc).
		parallel.Chunks(nijk, parallel.ChunkWorkers(nijk), func(_, lo, hi int) {
			stamp := newStamp(nijk)
			var cand []int32
			for gc := lo; gc < hi; gc++ {
				c0 := sp.IJK[gc]
				cc := sp.Configs[c0]
				cand = gatherCand(cc.Occ[:3], bk.ijk, stamp, int32(gc), cand)
				for _, gr32 := range cand {
					gr := int(gr32)
					if gr <= gc {
						continue
					}
					r0 := sp.IJK[gr]
					if blk, ok := mx.blk.ijkLMN(sp.Configs[r0], cc); ok {
						gemvTranspose(blk, r0, c0, xin, yout, b, ldi, ldo)
					}
				}
			}
		})
	}
	return matFreePart{apply: apply, release: func() {}}
}
