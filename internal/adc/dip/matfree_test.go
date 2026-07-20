package dip

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
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
