package dip

import (
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// bench_satellite_test.go — a local, sub-second throughput signal for the 3h1p↔3h1p satellite
// σ-build, the term that is ~96.6% of DIP solver wall time.
//
// WHY THIS EXISTS. Until now the only per-mat-vec timing signal was a cluster job (job 14015067,
// scripts/uracil2W_mgpu_timing.sbatch), i.e. hours of queue per data point. That job measured the
// satellite apply at 0.196 TFLOP/s per H200 = 0.58% of fp64 vector peak — even after the
// per-device NVLink rewrite that was itself 12.6× faster than its predecessor. The contraction
// work planned on the back of that number (docs/sigma_build_contractions.md) needs a signal that
// can be read on every commit rather than every queue slot; this is it.
//
// WHAT IT MEASURES. ApplyBlockSatellite with the satellite region matrix-free — the exact path a
// contraction rewrite would replace — reporting achieved GFLOP/s under the same FLOP model the
// cluster measurement used, so host and device numbers are commensurable:
//
//	flop = 2·nnz·b        (each stored element multiply-adds across b panel columns)
//
// nnz comes from satelliteResidentBytes()/8, the same exact block-gate walk the sizing path uses
// — not an estimate.
//
// WHAT IT DOES NOT MEASURE. Absolute host GFLOP/s here is not comparable to a GPU's: the point is
// the RATIO before and after a change, on identical input. Nor is the h2o_dzp sector
// representative of melanin's scale — batch-plan overhead and cache behaviour at n≈10⁷ cannot be
// seen at n≈10², which is why the plan keeps the cluster re-measurement as a mandatory final step.

// benchSector builds the largest h2o_dzp sector that has a satellite space, which is the biggest
// system in-repo where this is still a sub-second benchmark. Panels are allocated fresh per call
// so a benchmark loop never measures a warm output buffer.
func benchSector(b *testing.B) (*Matrix, int, uint64) {
	b.Helper()
	d, err := fcidump.ReadFile("../../../testdata/h2o_dzp.fcidump")
	if err != nil {
		b.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)

	// Pick the sector with the largest satellite space across spin × irrep.
	var best *Space
	for _, spin := range []Spin{Singlet, Triplet} {
		for sym := range 4 {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			if sp.Size() == 0 || sp.Size() == sp.MainBlockSize() {
				continue
			}
			if best == nil || sp.Size()-sp.MainBlockSize() > best.Size()-best.MainBlockSize() {
				best = sp
			}
		}
	}
	if best == nil {
		b.Fatal("no sector with a satellite space")
	}

	mx := New(best, ints, eps, backend.Gonum{})
	mx.SetMatFree(matfree.On, 0)
	if !mx.matFreeSatellite() {
		b.Fatal("matFreeSatellite() false — benchmark would time the dense path instead")
	}
	return mx, best.Size(), mx.satelliteResidentBytes() / 8
}

// BenchmarkSatelliteApply times the matrix-free satellite apply at several panel widths. b=435
// matches the sector width of the cluster measurement (uracil2W_dz main=435); b=64 matches
// satChunkCols, the per-device gather chunk. Report is GFLOP/s via the 2·nnz·b model.
func BenchmarkSatelliteApply(bb *testing.B) {
	for _, cols := range []int{1, 64, 435} {
		bb.Run(widthName(cols), func(bb *testing.B) {
			mx, n, nnz := benchSector(bb)
			be := backend.Gonum{}

			host := make([]float64, n*cols)
			for i := range host {
				host[i] = float64(i%17) * 0.125 // deterministic, non-trivial, no RNG in the loop
			}
			in := backend.BlockView{V: be.Upload(host), Rows: n, Cols: cols, Ld: n}
			out := backend.BlockView{V: be.Alloc(n * cols), Rows: n, Cols: cols, Ld: n}

			// Force the one-time assemble before timing: the first apply builds op, which is
			// not per-mat-vec work and would otherwise be charged to iteration 1.
			mx.ApplyBlockSatellite(out, in)

			flop := 2 * float64(nnz) * float64(cols)
			bb.ResetTimer()
			for bb.Loop() {
				mx.ApplyBlockSatellite(out, in)
			}
			bb.StopTimer()

			// GFLOP/s at the achieved ns/op. ReportMetric is per-op, so divide by the op count.
			perOp := float64(bb.Elapsed().Nanoseconds()) / float64(bb.N)
			bb.ReportMetric(flop/perOp, "GFLOP/s")
			bb.ReportMetric(float64(nnz), "nnz")
		})
	}
}

func widthName(cols int) string {
	switch cols {
	case 1:
		return "b=1"
	case 64:
		return "b=64_satChunkCols"
	default:
		return "b=435_clusterwidth"
	}
}

// BenchmarkBlockApplyCrossover is the decisive measurement for the contraction plan's abandon
// criterion: at what block size does a BLAS dgemm beat the hand-written gemvForward loop?
//
// It exists because the whole-apply benchmark above CANNOT answer that on any in-repo system.
// TestSatelliteBlockShapes shows h2o_dzp's jiiLKK blocks are 2..11 on a side (mean 17-80
// elements), because its 25 virtuals are split across 4 irreps. Production is a different regime
// entirely: melanin is C1 (ONE irrep), nvir=154, so blocks are 154×154 = 23,716 elements — ~300×
// larger. A "BLAS loses" result at h2o_dzp scale would therefore be an artefact of the test
// system, not evidence about the rewrite.
//
// So this benchmarks the two kernels directly on synthetic blocks across a size sweep spanning
// both regimes, at a fixed panel width. The number to read off is the crossover: if it sits well
// below 154, the melanin-scale rewrite is justified even though the small-system apply benchmark
// may show BLAS losing.
func BenchmarkBlockApplyCrossover(bb *testing.B) {
	const b = 64 // panel width; satChunkCols, the per-device gather chunk
	be := backend.Gonum{}

	for _, dim := range []int{8, 11, 16, 32, 64, 77, 128, 154, 256} {
		// One square block, applied into a panel band. The surrounding panel is sized so the
		// row/col bands do not alias, mirroring how gemvForward addresses a real panel.
		n := 2 * dim
		blk := backend.NewMat(dim, dim)
		for i := range blk.Data {
			blk.Data[i] = float64(i%23) * 0.0625
		}
		xin := make([]float64, n*b)
		for i := range xin {
			xin[i] = float64(i%29) * 0.03125
		}

		bb.Run("loop/dim="+itoa(dim), func(bb *testing.B) {
			yout := make([]float64, n*b)
			bb.ResetTimer()
			for bb.Loop() {
				gemvForward(blk, 0, dim, xin, yout, b, n, n)
			}
		})

		bb.Run("blas/dim="+itoa(dim), func(bb *testing.B) {
			dm := be.UploadMat(blk)
			inV := be.Upload(xin)
			outV := be.Alloc(n * b)
			// Same shape gemvForward realizes: block (dim×dim) times the col band at offset dim,
			// accumulating (beta=1) into the row band at offset 0.
			src := backend.BlockView{V: inV, Rows: n, Cols: b, Ld: n}.RowRange(dim, dim)
			dst := backend.BlockView{V: outV, Rows: n, Cols: b, Ld: n}.RowRange(0, dim)
			bb.ResetTimer()
			for bb.Loop() {
				be.GemmMat(false, 1, dm, src, 1, dst)
			}
		})
	}
}

// BenchmarkBlockBuildVsApply splits the matrix-free satellite cost into its two halves for the
// SAME blocks: rebuilding each block (mx.blk.jiiLKK — evaluates integrals, allocates a Mat) and
// applying it (gemvForward). Both happen on every mat-vec in the matrix-free path.
//
// This bounds what the planned BLAS/contraction swap can possibly buy. The rewrite replaces only
// the APPLY half; if BUILD dominates, Amdahl caps the win no matter how fast the GEMM is, and the
// real target would instead be the block recompute (or caching it, which is what does not fit in
// memory and is why the path is matrix-free at all).
func BenchmarkBlockBuildVsApply(bb *testing.B) {
	const b = 64
	mx, n, _ := benchSectorForBB(bb)
	sp := mx.sp

	// Collect the (row, col) config pairs of every nonzero jiiLKK block in this sector.
	type pair struct{ rc, cc Config }
	var pairs []pair
	for gr := range sp.JII {
		rc := sp.Configs[sp.JII[gr]]
		for gc := 0; gc <= gr; gc++ {
			cc := sp.Configs[sp.JII[gc]]
			if _, _, ok := mx.blk.jiiLKKGate(rc, cc); ok {
				pairs = append(pairs, pair{rc, cc})
			}
		}
	}
	if len(pairs) == 0 {
		bb.Skip("no jiiLKK blocks in sector")
	}

	xin := make([]float64, n*b)
	for i := range xin {
		xin[i] = float64(i%29) * 0.03125
	}

	bb.Run("build", func(bb *testing.B) {
		for bb.Loop() {
			for _, p := range pairs {
				if blk, ok := mx.blk.jiiLKK(p.rc, p.cc); ok {
					_ = blk
				}
			}
		}
	})

	// Pre-build so the apply loop times only the gemv.
	built := make([]backend.Mat, 0, len(pairs))
	for _, p := range pairs {
		if blk, ok := mx.blk.jiiLKK(p.rc, p.cc); ok {
			built = append(built, blk)
		}
	}
	bb.Run("apply_loop", func(bb *testing.B) {
		yout := make([]float64, n*b)
		bb.ResetTimer()
		for bb.Loop() {
			for _, blk := range built {
				gemvForward(blk, 0, 0, xin, yout, b, n, n)
			}
		}
	})
}

// BenchmarkJIIApplyLoopVsBatched compares the two jiiLKK appliers end-to-end on a real sector:
// the existing gemvForward/gemvTranspose loops vs the batched-GEMM path (matfree_batched.go).
//
// Read this against TestSatelliteBlockShapes: h2o_dzp's blocks are 2..11 on a side, which is the
// PESSIMISTIC end of BenchmarkBlockApplyCrossover (BLAS ~1.1-1.3× there, ~3.8× at melanin's 154).
// So a modest ratio here is the expected floor, not the production figure.
func BenchmarkJIIApplyLoopVsBatched(bb *testing.B) {
	const b = 64
	mx, n, _ := benchSectorForBB(bb)
	sp := mx.sp
	be := backend.Gonum{}

	panel := make([]float64, n*b)
	for i := range panel {
		panel[i] = float64(i%29) * 0.03125
	}

	bb.Run("loop", func(bb *testing.B) {
		out := make([]float64, n*b)
		bb.ResetTimer()
		for bb.Loop() {
			for gr := range sp.JII {
				r0 := sp.JII[gr]
				rc := sp.Configs[r0]
				for gc := 0; gc <= gr; gc++ {
					c0 := sp.JII[gc]
					blk, ok := mx.blk.jiiLKK(rc, sp.Configs[c0])
					if !ok {
						continue
					}
					gemvForward(blk, r0, c0, panel, out, b, n, n)
					if gr != gc {
						gemvTranspose(blk, r0, c0, panel, out, b, n, n)
					}
				}
			}
		}
	})

	bb.Run("batched", func(bb *testing.B) {
		part := mx.newJIIMatFreeBatched()
		inV := be.Upload(panel)
		outV := be.Alloc(n * b)
		inView := backend.BlockView{V: inV, Rows: n, Cols: b, Ld: n}
		outView := backend.BlockView{V: outV, Rows: n, Cols: b, Ld: n}
		bb.ResetTimer()
		for bb.Loop() {
			part.apply(inView, outView)
		}
	})
}

// benchSectorForBB is benchSector for a *testing.B nested one level down (sub-benchmarks get a
// fresh B, so the helper cannot capture the outer one).
func benchSectorForBB(bb *testing.B) (*Matrix, int, uint64) {
	bb.Helper()
	return benchSector(bb)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var d [8]byte
	i := len(d)
	for v > 0 {
		i--
		d[i] = byte('0' + v%10)
		v /= 10
	}
	return string(d[i:])
}

// TestSatelliteBlockShapes reports the jiiLKK block-size distribution for the benchmark sector.
//
// This is diagnostic, not an assertion, and it exists because it decides whether the planned
// contraction rewrite can pay off AT ALL: routing a block through OpenBLAS/cuBLAS `dgemm` only
// beats the hand-written triple loop in gemvForward once the block is large enough to amortize
// the BLAS call overhead. A sector whose blocks are ~10×10 will favour the loop no matter how
// good the batching is — so a disappointing BenchmarkSatelliteApply result must be read against
// these numbers before concluding the technique fails.
//
// Block dims are sizeVirGroup(rowSym) × sizeVirGroup(colSym) (blocks.go jiiLKKShape), i.e. set by
// the VIRTUAL-orbital group sizes, which grow with basis set — so production sectors have much
// larger blocks than any in-repo test system. Run with -v.
func TestSatelliteBlockShapes(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o_dzp.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)

	for _, spin := range []Spin{Singlet, Triplet} {
		for sym := range 4 {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			if sp.Size() == 0 || sp.Size() == sp.MainBlockSize() {
				continue
			}
			mx := New(sp, ints, eps, backend.Gonum{})
			var nblk, minD, maxD int
			var sumElems int64
			minD = 1 << 30
			for gr := range sp.JII {
				rc := sp.Configs[sp.JII[gr]]
				for gc := 0; gc <= gr; gc++ {
					r, c, ok := mx.blk.jiiLKKGate(rc, sp.Configs[sp.JII[gc]])
					if !ok {
						continue
					}
					nblk++
					sumElems += int64(r) * int64(c)
					minD = min(minD, min(r, c))
					maxD = max(maxD, max(r, c))
				}
			}
			if nblk == 0 {
				continue
			}
			t.Logf("spin=%v sym=%d: n=%5d nsat=%5d | jiiLKK blocks=%4d  dims %d..%d  mean elems/block=%.0f",
				spin, sym, sp.Size(), sp.Size()-sp.MainBlockSize(), nblk, minD, maxD,
				float64(sumElems)/float64(nblk))
		}
	}
}
