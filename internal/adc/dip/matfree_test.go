package dip

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// h2oSectors builds every (spin, sym) DIP sector of h2o that has a satellite space, calling
// fn with a fresh dense Matrix for each. Shared setup for the matrix-free tests.
func h2oSectors(t *testing.T, fn func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend)) {
	t.Helper()
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)
	be := backend.Gonum{}
	tested := 0
	for _, spin := range []Spin{Singlet, Triplet} {
		for sym := range 4 {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			if sp.Size() == 0 || sp.Size() == sp.MainBlockSize() {
				continue // need a satellite space
			}
			fn(spin, sym, sp, ints, eps, be)
			tested++
		}
	}
	if tested == 0 {
		t.Fatal("no sectors with a satellite space exercised")
	}
}

// TestSatelliteMatFreeEqualsDense checks that applying the operator with the satellite region
// matrix-free (SetMatFree On) reproduces the fully-dense operator, for ApplyFull (one vector)
// and ApplyBlock (a panel of columns). The matrix-free path recomputes the same blocks and
// sums them in a different order, so equality is to a tight numerical tolerance, not bitwise.
func TestSatelliteMatFreeEqualsDense(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		n := sp.Size()
		dense := New(sp, ints, eps, be)
		free := New(sp, ints, eps, be)
		free.SetMatFree(matfree.On, 0)
		if !free.matFreeSatellite() {
			t.Fatalf("spin=%v sym=%d: matFreeSatellite() false on host backend", spin, sym)
		}

		// ApplyFull, single vector.
		x := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		wantV := be.Alloc(n)
		gotV := be.Alloc(n)
		dense.ApplyFull(wantV, be.Upload(x))
		free.ApplyFull(gotV, be.Upload(x))
		assertClose(t, spin, sym, "ApplyFull", be.Download(wantV), be.Download(gotV))

		// ApplyBlock, a 3-column panel (Ld == Rows == n).
		const b = 3
		panel := make([]float64, n*b)
		for i := range panel {
			panel[i] = rng.NormFloat64()
		}
		inB := backend.BlockView{V: be.Upload(panel), Rows: n, Cols: b, Ld: n}
		wantB := backend.BlockView{V: be.Alloc(n * b), Rows: n, Cols: b, Ld: n}
		gotB := backend.BlockView{V: be.Alloc(n * b), Rows: n, Cols: b, Ld: n}
		dense.ApplyBlock(wantB, inB)
		free.ApplyBlock(gotB, inB)
		assertClose(t, spin, sym, "ApplyBlock", be.Download(wantB.V), be.Download(gotB.V))
	})
}

// TestSatelliteScalarApplyEqualsDense checks the per-output-scalar satellite applier
// (satscalar.go, the CUDA kernel's CPU twin) reproduces the dense satellite operator over a
// panel of columns. It validates the one-thread-per-row algorithm end to end: candidate
// pruning, block orientation, and the Elem transcription together must equal M_sat.
func TestSatelliteScalarApplyEqualsDense(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		n, main := sp.Size(), sp.MainBlockSize()
		mx := New(sp, ints, eps, be)

		// Dense satellite-only reference: M with the main rows/cols removed.
		M := mx.BuildMatrix()
		for i := range n {
			for j := range n {
				if i < main || j < main {
					M.Set(i, j, 0)
				}
			}
		}
		const b = 3
		panel := make([]float64, n*b)
		for i := range panel {
			panel[i] = rng.NormFloat64()
		}
		want := make([]float64, n*b)
		for c := range b {
			copy(want[c*n:(c+1)*n], M.MulVec(panel[c*n:(c+1)*n]))
		}

		plan := mx.buildSatScalarPlan()
		in := backend.BlockView{V: be.Upload(panel), Rows: n, Cols: b, Ld: n}
		out := backend.BlockView{V: be.Alloc(n * b), Rows: n, Cols: b, Ld: n}
		be.Zero(out.V)
		plan.apply(in, out)
		assertClose(t, spin, sym, "ScalarApply", want, be.Download(out.V))
	})
}

// TestApplyBlockSatelliteMatFree checks the gated satellite apply (Tarantelli Mode B gate)
// against the masked dense operator when the satellite region is matrix-free — the matrix-free
// twin of TestApplyBlockSatellite.
func TestApplyBlockSatelliteMatFree(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		n, main := sp.Size(), sp.MainBlockSize()
		mx := New(sp, ints, eps, be)
		mx.SetMatFree(matfree.On, 0)

		// Dense reference: M with the main-space rows/cols zeroed (only satellite acts).
		M := mx.BuildMatrix()
		for i := range n {
			for j := range n {
				if i < main || j < main {
					M.Set(i, j, 0)
				}
			}
		}
		x := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		want := M.MulVec(x)

		in := backend.BlockView{V: be.Upload(x), Rows: n, Cols: 1, Ld: n}
		out := backend.BlockView{V: be.Alloc(n), Rows: n, Cols: 1, Ld: n}
		mx.ApplyBlockSatellite(out, in)
		got := be.Download(out.V)
		for i := range main {
			if got[i] != 0 {
				t.Errorf("spin=%v sym=%d: matrix-free satellite wrote main row %d = %g (want 0)", spin, sym, i, got[i])
			}
		}
		assertClose(t, spin, sym, "ApplyBlockSatellite", want, got)
	})
}

// TestSatelliteMatFreeDistributedEqualsDense checks the -mgpu composition: a DIP operator on a
// row-partitioned (distributed) backend with the satellite region matrix-free reproduces the
// single-node dense operator, for ApplyBlock (full operator) and ApplyBlockSatellite (the Mode-B
// gate). It runs the distributed backend over gonum sub-backends, so the same gather-apply-scatter
// path that composes with real GPUs is validated on the host.
func TestSatelliteMatFreeDistributedEqualsDense(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		n, main := sp.Size(), sp.MainBlockSize()
		bounds := sp.PartitionBounds(2)
		npart := len(bounds) - 1
		if npart < 2 || n <= 2*main*main {
			return // sector too small to partition; the shape invariant would reject it
		}
		subs := make([]backend.Backend, npart)
		for i := range subs {
			subs[i] = backend.Gonum{}
		}
		dist, err := backend.NewDistributed(subs, n, main, bounds)
		if err != nil {
			t.Fatalf("spin=%v sym=%d: NewDistributed: %v", spin, sym, err)
		}

		dense := New(sp, ints, eps, be)
		free := New(sp, ints, eps, dist)
		free.SetMatFree(matfree.On, 0)
		if !free.matFreeSatellite() {
			t.Fatalf("spin=%v sym=%d: matFreeSatellite() false on distributed backend", spin, sym)
		}

		const b = 3
		panel := make([]float64, n*b)
		for i := range panel {
			panel[i] = rng.NormFloat64()
		}

		for _, tc := range []struct {
			name string
			run  func(mx *Matrix, out, in backend.BlockView)
		}{
			{"ApplyBlock", (*Matrix).ApplyBlock},
			{"ApplyBlockSatellite", (*Matrix).ApplyBlockSatellite},
		} {
			wantB := backend.BlockView{V: be.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			tc.run(dense, wantB, backend.BlockView{V: be.Upload(panel), Rows: n, Cols: b, Ld: n})

			gotB := backend.BlockView{V: dist.Alloc(n * b), Rows: n, Cols: b, Ld: n}
			tc.run(free, gotB, backend.BlockView{V: dist.Upload(panel), Rows: n, Cols: b, Ld: n})
			assertClose(t, spin, sym, "dist "+tc.name, be.Download(wantB.V), dist.Download(gotB.V))
		}
		free.Release()
	})
}

// TestSatelliteDistributedEdgeSweep runs the row-partitioned satellite apply across the
// DEGENERATE partitionings and chunk widths, against the single-backend dense reference.
//
// TestSatelliteMatFreeDistributedEqualsDense above pins the shape at one partition count (2) and
// one panel width (3). Every -mgpu defect this code has shipped lived outside that point: a
// partition owning no block at all, a panel narrower than one gather chunk, a chunk width that
// does not divide the panel. This sweeps those directly, over Gonum sub-backends, so it runs on
// any machine with no GPU and no CUDA toolkit — the cheap gate that belongs ahead of the
// hardware smoke, not behind it.
//
// SatChunkCols is latched when an applier is CONSTRUCTED (it sizes the slab), so it is set
// before New and restored after.
func TestSatelliteDistributedEdgeSweep(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	defer func(w int) { SatChunkCols = w }(SatChunkCols)

	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		n, main := sp.Size(), sp.MainBlockSize()
		if n <= 2*main*main {
			return // the distributed backend's shape invariant rejects it
		}
		dense := New(sp, ints, eps, be)

		for _, g := range []int{1, 2, 3, 4} {
			bounds := sp.PartitionBounds(g)
			npart := len(bounds) - 1
			if npart < 1 {
				continue
			}
			for _, chunk := range []int{1, 3, 1 << 20} { // sub-panel, odd, and wider than any panel
				for _, b := range []int{1, 3, chunk + 1} {
					if b < 1 || b > 64 {
						continue
					}
					SatChunkCols = chunk
					subs := make([]backend.Backend, npart)
					for i := range subs {
						subs[i] = backend.Gonum{}
					}
					dist, err := backend.NewDistributed(subs, n, main, bounds)
					if err != nil {
						t.Fatalf("spin=%v sym=%d g=%d: NewDistributed: %v", spin, sym, g, err)
					}
					free := New(sp, ints, eps, dist)
					free.SetMatFree(matfree.On, 0)

					panel := make([]float64, n*b)
					for i := range panel {
						panel[i] = rng.NormFloat64()
					}
					wantB := backend.BlockView{V: be.Alloc(n * b), Rows: n, Cols: b, Ld: n}
					dense.ApplyBlockSatellite(wantB, backend.BlockView{V: be.Upload(panel), Rows: n, Cols: b, Ld: n})

					gotB := backend.BlockView{V: dist.Alloc(n * b), Rows: n, Cols: b, Ld: n}
					free.ApplyBlockSatellite(gotB, backend.BlockView{V: dist.Upload(panel), Rows: n, Cols: b, Ld: n})

					assertClose(t, spin, sym,
						fmt.Sprintf("dist g=%d parts=%d chunk=%d b=%d", g, npart, chunk, b),
						be.Download(wantB.V), dist.Download(gotB.V))
					free.Release()
				}
			}
		}
	})
}

// TestSatelliteResidentBytes checks that the cheap gate-based satellite sizing equals the real
// dense satellite footprint, and that OperatorResidentBytes drops the satellite term when the
// region is matrix-free (the pre-flight guard depends on this).
func TestSatelliteResidentBytes(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		mx := New(sp, ints, eps, be)

		// Real dense satellite bytes: sum rows·cols·8 over the actually-assembled satellite blocks.
		var satNNZ uint64
		for _, task := range mx.satelliteTasks() {
			task(func(m backend.Mat, _, _ int, _ bool) {
				satNNZ += uint64(m.Rows) * uint64(m.Cols)
			})
		}
		wantSat := satNNZ * 8
		if got := mx.satelliteResidentBytes(); got != wantSat {
			t.Errorf("spin=%v sym=%d: satelliteResidentBytes=%d, want %d", spin, sym, got, wantSat)
		}

		full := mx.OperatorResidentBytes() // dense
		free := New(sp, ints, eps, be)
		free.SetMatFree(matfree.On, 0)
		if got, want := free.OperatorResidentBytes(), full-wantSat; got != want {
			t.Errorf("spin=%v sym=%d: matrix-free OperatorResidentBytes=%d, want full-sat=%d", spin, sym, got, want)
		}
		if wantSat == 0 {
			t.Errorf("spin=%v sym=%d: no satellite bytes — sector does not exercise the region", spin, sym)
		}
	})
}

// TestSatelliteGateExhaustive verifies, over every satellite group pair, that (1) the cheap
// gate's nonzero decision and dimensions match the value block exactly (no gate/value drift),
// and (2) every nonzero block has a shared occupied index between its row and column groups —
// the necessary condition the matrix-free applier's occ-index buckets rely on to prune.
func TestSatelliteGateExhaustive(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, _ *integrals.Store, eps []float64, _ backend.Backend) {
		d, _ := fcidump.ReadFile("../../../testdata/h2o.fcidump")
		ints := integrals.New(d, mp.NOcc(d), d.OrbSym)
		mx := New(sp, ints, eps, backend.Gonum{})
		blk := mx.blk

		check := func(name string, r, c Config, gr, gc int, ok bool, rows, cols int) {
			var vr, vc int
			var vok bool
			switch name {
			case "jiiLKK":
				m, o := blk.jiiLKK(r, c)
				vok, vr, vc = o, m.Rows, m.Cols
			case "ijkMLL":
				m, o := blk.ijkMLL(r, c)
				vok, vr, vc = o, m.Rows, m.Cols
			case "ijkLMN":
				m, o := blk.ijkLMN(r, c)
				vok, vr, vc = o, m.Rows, m.Cols
			}
			if ok != vok {
				t.Fatalf("spin=%v sym=%d %s(%d,%d): gate ok=%v but value ok=%v", spin, sym, name, gr, gc, ok, vok)
			}
			if ok && (rows != vr || cols != vc) {
				t.Fatalf("spin=%v sym=%d %s(%d,%d): gate dims %dx%d but value %dx%d", spin, sym, name, gr, gc, rows, cols, vr, vc)
			}
			if ok && !sharesOcc(name, r, c) {
				t.Fatalf("spin=%v sym=%d %s(%d,%d): nonzero block with no shared occ index (buckets would miss it)", spin, sym, name, gr, gc)
			}
		}

		for gr := range sp.JII {
			for gc := range sp.JII {
				r, c := sp.Configs[sp.JII[gr]], sp.Configs[sp.JII[gc]]
				rows, cols, ok := blk.jiiLKKGate(r, c)
				check("jiiLKK", r, c, gr, gc, ok, rows, cols)
			}
		}
		for gr := range sp.IJK {
			for gc := range sp.JII {
				r, c := sp.Configs[sp.IJK[gr]], sp.Configs[sp.JII[gc]]
				rows, cols, ok := blk.ijkMLLGate(r, c)
				check("ijkMLL", r, c, gr, gc, ok, rows, cols)
			}
			for gc := range sp.IJK {
				r, c := sp.Configs[sp.IJK[gr]], sp.Configs[sp.IJK[gc]]
				rows, cols, ok := blk.ijkLMNGate(r, c)
				check("ijkLMN", r, c, gr, gc, ok, rows, cols)
			}
		}
	})
}

// sharesOcc reports whether the row and column groups share an occupied index over the ranges
// the block actually reads (jiiLKK/ijkMLL/ijkLMN use 2 or 3 holes per side).
func sharesOcc(name string, r, c Config) bool {
	rn, cn := 3, 3
	switch name {
	case "jiiLKK":
		rn, cn = 2, 2
	case "ijkMLL":
		rn, cn = 3, 2
	}
	for i := range rn {
		for j := range cn {
			if r.Occ[i] == c.Occ[j] {
				return true
			}
		}
	}
	return false
}

func assertClose(t *testing.T, spin Spin, sym int, what string, want, got []float64) {
	t.Helper()
	var maxDiff, scale float64
	for i := range want {
		scale = math.Max(scale, math.Abs(want[i]))
		maxDiff = math.Max(maxDiff, math.Abs(want[i]-got[i]))
	}
	if rel := maxDiff / math.Max(scale, 1e-300); rel > 1e-10 {
		t.Errorf("spin=%v sym=%d %s: relative diff %.3e (want <= 1e-10)", spin, sym, what, rel)
	}
}

// TestRestrictIsSubBlock: a space restricted by a hole predicate (the Fano scheme A
// partition) assembles exactly the parent's sub-block, dense and with the satellite region
// matrix-free, for both halves of the partition; and a restriction that splits a 3h1p group
// is refused, since the operator builds each group as one panel.
func TestRestrictIsSubBlock(t *testing.T) {
	h2oSectors(t, func(spin Spin, sym int, sp *Space, ints *integrals.Store, eps []float64, be backend.Backend) {
		full := New(sp, ints, eps, be).BuildMatrix()
		var q, p []int
		for r := range sp.Size() {
			has := false
			for _, h := range sp.Holes(r, nil) {
				has = has || h == 1
			}
			if has {
				q = append(q, r)
			} else {
				p = append(p, r)
			}
		}
		for _, rows := range [][]int{q, p} {
			sub, err := sp.Restrict(rows)
			if err != nil {
				t.Fatalf("spin=%v sym=%d: %v", spin, sym, err)
			}
			for i, r := range rows {
				if got, want := sub.Holes(i, nil), sp.Holes(r, nil); fmt.Sprint(got) != fmt.Sprint(want) {
					t.Fatalf("spin=%v sym=%d: restricted row %d holes %v, parent row %d %v", spin, sym, i, got, r, want)
				}
			}
			S := New(sub, ints, eps, be).BuildMatrix()
			for i, r := range rows {
				for j, c := range rows {
					if S.At(i, j) != full.At(r, c) {
						t.Fatalf("spin=%v sym=%d: restricted (%d,%d)=%g, parent (%d,%d)=%g",
							spin, sym, i, j, S.At(i, j), r, c, full.At(r, c))
					}
				}
			}
			if sub.Size() == sub.MainBlockSize() {
				continue
			}
			free := New(sub, ints, eps, be)
			free.SetMatFree(matfree.On, 0)
			n := sub.Size()
			x := make([]float64, n)
			rng := rand.New(rand.NewSource(int64(n)))
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			y := be.Alloc(n)
			free.ApplyFull(y, be.Upload(x))
			yh := be.Download(y)
			var worst float64
			for i := range n {
				var s float64
				for j := range n {
					s += S.At(i, j) * x[j]
				}
				worst = max(worst, math.Abs(yh[i]-s))
			}
			free.Release()
			if worst > 1e-11 {
				t.Errorf("spin=%v sym=%d: matrix-free restricted apply differs from the sub-block by %.2e", spin, sym, worst)
			}
		}
		if len(sp.JII) > 0 {
			var split []int
			for r := range sp.Size() {
				if r != sp.JII[0] {
					split = append(split, r)
				}
			}
			if _, err := sp.Restrict(split); err == nil {
				t.Errorf("spin=%v sym=%d: a restriction splitting a 3h1p group was accepted", spin, sym)
			}
		}
	})
}

// matfree_test.go — a local, sub-second throughput signal for the 3h1p↔3h1p satellite
// σ-build, the term that is ~96.6% of DIP solver wall time.
//
// WHY THIS EXISTS. Until now the only per-mat-vec timing signal was a cluster job (job 14015067,
// scripts/gpu/uracil2W_mgpu_timing.sbatch), i.e. hours of queue per data point. That job measured the
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
// representative of the production system's scale — batch-plan overhead and cache behaviour at n≈10⁷ cannot be
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
// SatChunkCols, the per-device gather chunk. Report is GFLOP/s via the 2·nnz·b model.
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
		return "b=64_SatChunkCols"
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
// entirely: the production system is C1 (ONE irrep), nvir=154, so blocks are 154×154 = 23,716 elements — ~300×
// larger. A "BLAS loses" result at h2o_dzp scale would therefore be an artefact of the test
// system, not evidence about the rewrite.
//
// So this benchmarks the two kernels directly on synthetic blocks across a size sweep spanning
// both regimes, at a fixed panel width. The number to read off is the crossover: if it sits well
// below 154, the production-scale rewrite is justified even though the small-system apply benchmark
// may show BLAS losing.
func BenchmarkBlockApplyCrossover(bb *testing.B) {
	const b = 64 // panel width; SatChunkCols, the per-device gather chunk
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
// PESSIMISTIC end of BenchmarkBlockApplyCrossover (BLAS ~1.1-1.3× there, ~3.8× at the production system's 154).
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
