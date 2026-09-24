package dip

import (
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/leiaSQ/ADCgo/backend"
	dippkg "github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// sigma_generated_test.go (hand-written; survives regeneration): the generated σ-build
// against the generated per-element evaluators. Elem.BuildDense is checked against
// adcgen's fully expanded expressions (fidelity_test.go) and against package dip
// (types_test.go), so agreement here pins the σ program — its mvp derivation, spin
// integration, contraction order and normalization — to the same matrix.

type sigmaSystem struct {
	name string
	ints *integrals.Store
	eps  []float64
	nocc int
	nvir int
}

func loadSigmaSystem(t *testing.T, rel ...string) sigmaSystem {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join(append([]string{"..", "..", "..", ".."}, rel...)...))
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	return sigmaSystem{name: rel[len(rel)-1], ints: integrals.New(d, nocc, nil),
		eps: mp.OrbitalEnergies(d, nocc), nocc: nocc, nvir: d.NORB - nocc}
}

// denseApply is M·X for a row-major M and a column-major panel X.
func denseApply(m backend.Mat, x []float64, n, b int) []float64 {
	y := make([]float64, n*b)
	for c := range b {
		for r := range n {
			var s float64
			row := m.Data[r*n : (r+1)*n]
			for k, v := range row {
				s += v * x[c*n+k]
			}
			y[c*n+r] = s
		}
	}
	return y
}

func maxDev(a, b []float64) (dev, scale float64) {
	for i := range a {
		dev = math.Max(dev, math.Abs(a[i]-b[i]))
		scale = math.Max(scale, math.Abs(b[i]))
	}
	return dev, scale
}

func TestSigmaMatchesDense(t *testing.T) {
	h2o := loadSigmaSystem(t, "testdata", "h2o.fcidump")
	he3 := loadSigmaSystem(t, "testdata", "khci", "he3chain_631g_canonical.fcidump")
	cases := []struct {
		sys      sigmaSystem
		scheme   string
		twoMs    int
		maxClass int
	}{
		{h2o, "adc2x", 0, 3},
		{h2o, "adc2x", 2, 3},
		{h2o, "strict:2", 0, 3},
		{he3, "ci", 0, 4},
		{he3, "ci", 2, 4},
		{he3, "adc2x", 0, 3},
		{he3, "ci", 0, 3}, // B12 terms must drop when the 4h2p class is absent
	}
	be := backend.Gonum{}
	rng := rand.New(rand.NewPCG(11, 12))
	for _, c := range cases {
		sp, err := khci.NewSpace(khci.Options{K: 2, NOcc: c.sys.nocc, NVir: c.sys.nvir,
			TwoMs: c.twoMs, MaxClass: c.maxClass})
		if err != nil {
			t.Fatal(err)
		}
		el, err := New(c.sys.ints, c.sys.eps, c.sys.nocc, c.scheme)
		if err != nil {
			t.Fatal(err)
		}
		op, err := NewSigma(sp, c.sys.ints, c.sys.eps, c.sys.nocc, c.scheme, be)
		if err != nil {
			t.Fatal(err)
		}
		n, b := sp.Size(), 3
		m := el.BuildDense(sp)
		x := make([]float64, n*b)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		out := be.Alloc(n * b)
		op.ApplyBlock(backend.BlockView{V: out, Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: be.Upload(x), Rows: n, Cols: b, Ld: n})
		dev, scale := maxDev(be.Download(out), denseApply(m, x, n, b))
		tag := c.sys.name + " " + c.scheme
		t.Logf("%s 2Ms=%d class<=%d: n=%d, |σ - M·X| max %.2e (scale %.2e)", tag, c.twoMs, c.maxClass, n, dev, scale)
		if dev > 1e-11*math.Max(scale, 1) {
			t.Errorf("%s 2Ms=%d: σ differs from BuildDense·X by %.3e", tag, c.twoMs, dev)
		}

		// the diagonal
		dg := be.Download(op.Diagonal(be))
		var ddev float64
		for r := range n {
			ddev = math.Max(ddev, math.Abs(dg[r]-m.Data[r*n+r]))
		}
		if ddev > 1e-11 {
			t.Errorf("%s 2Ms=%d: Diagonal differs from diag(BuildDense) by %.3e", tag, c.twoMs, ddev)
		}

		// the satellite-only apply: M with every row and column of the main class removed
		main := sp.MainBlockSize()
		ms := backend.NewMat(n, n)
		for r := main; r < n; r++ {
			for k := main; k < n; k++ {
				ms.Data[r*n+k] = m.Data[r*n+k]
			}
		}
		op.ApplyBlockSatellite(backend.BlockView{V: out, Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: be.Upload(x), Rows: n, Cols: b, Ld: n})
		if dev, _ := maxDev(be.Download(out), denseApply(ms, x, n, b)); dev > 1e-11*math.Max(scale, 1) {
			t.Errorf("%s 2Ms=%d: satellite apply differs by %.3e", tag, c.twoMs, dev)
		}

		// a restricted space: the operator on it is the sub-block of M
		var rows []int
		for r := range n {
			if rng.IntN(3) > 0 {
				rows = append(rows, r)
			}
		}
		sub, err := sp.Restrict(rows)
		if err != nil {
			t.Fatal(err)
		}
		ops, err := NewSigma(sub, c.sys.ints, c.sys.eps, c.sys.nocc, c.scheme, be)
		if err != nil {
			t.Fatal(err)
		}
		ns := len(rows)
		mr := backend.NewMat(ns, ns)
		for i, r := range rows {
			for j, k := range rows {
				mr.Data[i*ns+j] = m.Data[r*n+k]
			}
		}
		xs := x[:ns*b]
		os := be.Alloc(ns * b)
		ops.ApplyBlock(backend.BlockView{V: os, Rows: ns, Cols: b, Ld: ns},
			backend.BlockView{V: be.Upload(xs), Rows: ns, Cols: b, Ld: ns})
		if dev, _ := maxDev(be.Download(os), denseApply(mr, xs, ns, b)); dev > 1e-11*math.Max(scale, 1) {
			t.Errorf("%s 2Ms=%d: restricted σ differs by %.3e", tag, c.twoMs, dev)
		}
	}
}

// TestSigmaSpinProjectedMatchesDipPackage is the physics gate: the spin-summed Ms = 0
// σ operator, run by block Lanczos from the pure-singlet (pure-triplet) main-class
// vectors, reproduces the singlet (triplet) poles of the hand-ported Tarantelli
// DIP-ADC(2) — every pole with weight, both ways, eigenvalue and pole strength. The
// Hamiltonian commutes with S², so the Krylov space never leaves the multiplicity.
func TestSigmaSpinProjectedMatchesDipPackage(t *testing.T) {
	sys := loadSigmaSystem(t, "testdata", "h2o.fcidump")
	be := backend.Gonum{}
	sp, err := khci.NewSpace(khci.Options{K: 2, NOcc: sys.nocc, NVir: sys.nvir, TwoMs: 0, MaxClass: 3})
	if err != nil {
		t.Fatal(err)
	}
	op, err := NewSigma(sp, sys.ints, sys.eps, sys.nocc, "adc2x", be)
	if err != nil {
		t.Fatal(err)
	}
	const psMin = 1e-4 // percent
	for _, tc := range []struct {
		spin dippkg.Spin
		twoS int
	}{{dippkg.Singlet, 0}, {dippkg.Triplet, 2}} {
		hsp := dippkg.NewSpace(sys.nocc, sys.nocc+sys.nvir, nil, 0, tc.spin)
		hm := dippkg.New(hsp, sys.ints, sys.eps, be).BuildMatrix()
		hv, hvec := be.SymEig(hm)
		hmain := hsp.MainBlockSize()
		var want, wantPS []float64
		for k := range hv {
			var ps float64
			for r := range hmain {
				ps += hvec.At(r, k) * hvec.At(r, k)
			}
			if 100*ps > psMin {
				want = append(want, hv[k])
				wantPS = append(wantPS, 100*ps)
			}
		}
		vecs, b, err := sp.MainSpinVectors(tc.twoS)
		if err != nil {
			t.Fatal(err)
		}
		if b != hmain {
			t.Fatalf("2S=%d: %d main-class spin vectors, dip has %d main rows", tc.twoS, b, hmain)
		}
		if err := op.SetSpin(tc.twoS); err != nil {
			t.Fatal(err)
		}
		res := lanczos.Solve(op, be, lanczos.Options{Block: b, StartVecs: vecs, MaxDim: sp.Size()})
		t.Logf("2S=%d: Krylov dim %d (the multiplicity sector of package dip has %d rows)",
			tc.twoS, len(res.Values), hsp.Size())
		var got, gotPS []float64
		for k, v := range res.Values {
			if res.PS[k] > psMin {
				got = append(got, v)
				gotPS = append(gotPS, res.PS[k])
			}
		}
		match := func(a, aps, b, bps []float64) (de, dps float64) {
			for i, x := range a {
				best := -1
				for j, y := range b {
					if best < 0 || math.Abs(y-x) < math.Abs(b[best]-x) {
						best = j
					}
				}
				de = math.Max(de, math.Abs(b[best]-x))
				dps = math.Max(dps, math.Abs(bps[best]-aps[i]))
			}
			return
		}
		de1, dp1 := match(want, wantPS, got, gotPS)
		de2, dp2 := match(got, gotPS, want, wantPS)
		t.Logf("2S=%d: %d dip poles, %d σ-Lanczos poles; |ΔE| %.1e/%.1e Eh, |ΔPS| %.1e/%.1e %%",
			tc.twoS, len(want), len(got), de1, de2, dp1, dp2)
		if de1 > 1e-8 || de2 > 1e-8 || dp1 > 1e-6 || dp2 > 1e-6 {
			t.Errorf("2S=%d: σ-Lanczos poles differ from package dip", tc.twoS)
		}
	}
}

// TestSigmaDeviceParity: the σ-build on the CUDA backend (TensorKernels) against the host
// engine — block apply at several panel widths, the diagonal, the satellite-only apply,
// the spin-projected apply and a restricted space.
func TestSigmaDeviceParity(t *testing.T) {
	dev, err := backend.New("cuda")
	if err != nil {
		t.Skipf("no cuda backend/device: %v", err)
	}
	host := backend.Gonum{}
	rng := rand.New(rand.NewPCG(21, 22))
	for _, c := range []struct {
		sys      sigmaSystem
		scheme   string
		maxClass int
	}{
		{loadSigmaSystem(t, "testdata", "h2o.fcidump"), "adc2x", 3},
		{loadSigmaSystem(t, "testdata", "khci", "he3chain_631g_canonical.fcidump"), "ci", 4},
	} {
		sp, err := khci.NewSpace(khci.Options{K: 2, NOcc: c.sys.nocc, NVir: c.sys.nvir, TwoMs: 0, MaxClass: c.maxClass})
		if err != nil {
			t.Fatal(err)
		}
		var rows []int
		for r := range sp.Size() {
			if rng.IntN(4) > 0 {
				rows = append(rows, r)
			}
		}
		sub, err := sp.Restrict(rows)
		if err != nil {
			t.Fatal(err)
		}
		for _, s := range []*khci.Space{sp, sub} {
			ho, err := NewSigma(s, c.sys.ints, c.sys.eps, c.sys.nocc, c.scheme, host)
			if err != nil {
				t.Fatal(err)
			}
			do, err := NewSigma(s, c.sys.ints, c.sys.eps, c.sys.nocc, c.scheme, dev)
			if err != nil {
				t.Fatal(err)
			}
			if do.Staged() {
				t.Fatal("the cuda backend runs staged: its TensorKernels were not picked up")
			}
			n := s.Size()
			for _, b := range []int{1, 7} {
				x := make([]float64, n*b)
				for i := range x {
					x[i] = rng.NormFloat64()
				}
				for _, mode := range []string{"full", "satellite"} {
					hout, dout := host.Alloc(n*b), dev.Alloc(n*b)
					hin := backend.BlockView{V: host.Upload(x), Rows: n, Cols: b, Ld: n}
					din := backend.BlockView{V: dev.Upload(x), Rows: n, Cols: b, Ld: n}
					hv := backend.BlockView{V: hout, Rows: n, Cols: b, Ld: n}
					dv := backend.BlockView{V: dout, Rows: n, Cols: b, Ld: n}
					if mode == "full" {
						ho.ApplyBlock(hv, hin)
						do.ApplyBlock(dv, din)
					} else {
						ho.ApplyBlockSatellite(hv, hin)
						do.ApplyBlockSatellite(dv, din)
					}
					dev2, scale := maxDev(dev.Download(dout), host.Download(hout))
					if dev2 > 1e-12*math.Max(scale, 1) {
						t.Errorf("%s %s n=%d b=%d: device σ differs by %.2e", c.scheme, mode, n, b, dev2)
					}
					dev.Free(dout)
					dev.Free(din.V)
				}
			}
			dd, _ := maxDev(dev.Download(do.Diagonal(dev)), host.Download(ho.Diagonal(host)))
			if dd > 1e-12 {
				t.Errorf("%s n=%d: device diagonal differs by %.2e", c.scheme, n, dd)
			}
			if s == sp {
				if err := ho.SetSpin(0); err != nil {
					t.Fatal(err)
				}
				if err := do.SetSpin(0); err != nil {
					t.Fatal(err)
				}
				x := make([]float64, n)
				for i := range x {
					x[i] = rng.NormFloat64()
				}
				hout, dout := host.Alloc(n), dev.Alloc(n)
				ho.ApplyFull(hout, host.Upload(x))
				do.ApplyFull(dout, dev.Upload(x))
				if d, scale := maxDev(dev.Download(dout), host.Download(hout)); d > 1e-12*math.Max(scale, 1) {
					t.Errorf("%s: spin-projected device σ differs by %.2e", c.scheme, d)
				}
			}
			do.Release()
		}
	}
}

// TestSigmaPartitionedMatchesHost: on a row-partitioned backend (three host partitions,
// so the host-staged band copies run) the column-split σ apply equals the single-backend
// one, for panels narrower than, equal to and wider than the partition count, and the
// diagonal and spin projection carry over.
func TestSigmaPartitionedMatchesHost(t *testing.T) {
	sys := loadSigmaSystem(t, "testdata", "h2o.fcidump")
	sp, err := khci.NewSpace(khci.Options{K: 2, NOcc: sys.nocc, NVir: sys.nvir, TwoMs: 0, MaxClass: 3})
	if err != nil {
		t.Fatal(err)
	}
	n := sp.Size()
	host := backend.Gonum{}
	subs := []backend.Backend{backend.Gonum{}, backend.Gonum{}, backend.Gonum{}}
	dist, err := backend.NewDistributed(subs, n, sp.MainBlockSize(), []int{0, n / 5, n / 2, n})
	if err != nil {
		t.Fatal(err)
	}
	ho, err := NewSigma(sp, sys.ints, sys.eps, sys.nocc, "adc2x", host)
	if err != nil {
		t.Fatal(err)
	}
	do, err := NewSigma(sp, sys.ints, sys.eps, sys.nocc, "adc2x", dist)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(31, 32))
	for _, spin := range []int{-1, 0} {
		if err := ho.SetSpin(spin); err != nil {
			t.Fatal(err)
		}
		if err := do.SetSpin(spin); err != nil {
			t.Fatal(err)
		}
		for _, b := range []int{1, 3, 5} {
			x := make([]float64, n*b)
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			hout, dout := host.Alloc(n*b), dist.Alloc(n*b)
			ho.ApplyBlock(backend.BlockView{V: hout, Rows: n, Cols: b, Ld: n},
				backend.BlockView{V: host.Upload(x), Rows: n, Cols: b, Ld: n})
			do.ApplyBlock(backend.BlockView{V: dout, Rows: n, Cols: b, Ld: n},
				backend.BlockView{V: dist.Upload(x), Rows: n, Cols: b, Ld: n})
			if d, scale := maxDev(dist.Download(dout), host.Download(hout)); d > 1e-12*math.Max(scale, 1) {
				t.Errorf("2S=%d b=%d: partitioned σ differs by %.2e", spin, b, d)
			}
		}
	}
	if d, _ := maxDev(dist.Download(do.Diagonal(dist)), host.Download(ho.Diagonal(host))); d > 1e-13 {
		t.Errorf("partitioned diagonal differs by %.2e", d)
	}
}

// TestSigmaPartitionedDeviceParity is TestSigmaPartitionedMatchesHost on real devices:
// every visible CUDA device a partition, band copies over peer access where available.
func TestSigmaPartitionedDeviceParity(t *testing.T) {
	subs, err := backend.NewAll("cuda", 0)
	if err != nil || len(subs) < 2 {
		t.Skipf("needs two or more cuda devices (%d, %v)", len(subs), err)
	}
	sys := loadSigmaSystem(t, "testdata", "h2o.fcidump")
	sp, err := khci.NewSpace(khci.Options{K: 2, NOcc: sys.nocc, NVir: sys.nvir, TwoMs: 0, MaxClass: 3})
	if err != nil {
		t.Fatal(err)
	}
	n := sp.Size()
	bounds := make([]int, len(subs)+1)
	for i := range bounds {
		bounds[i] = i * n / len(subs)
	}
	dist, err := backend.NewDistributed(subs, n, sp.MainBlockSize(), bounds)
	if err != nil {
		t.Fatal(err)
	}
	host := backend.Gonum{}
	ho, err := NewSigma(sp, sys.ints, sys.eps, sys.nocc, "adc2x", host)
	if err != nil {
		t.Fatal(err)
	}
	do, err := NewSigma(sp, sys.ints, sys.eps, sys.nocc, "adc2x", dist)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(41, 42))
	for _, b := range []int{1, len(subs) + 1, 16} {
		x := make([]float64, n*b)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		hout, dout := host.Alloc(n*b), dist.Alloc(n*b)
		ho.ApplyBlock(backend.BlockView{V: hout, Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: host.Upload(x), Rows: n, Cols: b, Ld: n})
		do.ApplyBlock(backend.BlockView{V: dout, Rows: n, Cols: b, Ld: n},
			backend.BlockView{V: dist.Upload(x), Rows: n, Cols: b, Ld: n})
		if d, scale := maxDev(dist.Download(dout), host.Download(hout)); d > 1e-12*math.Max(scale, 1) {
			t.Errorf("%d devices, b=%d: partitioned device σ differs by %.2e", len(subs), b, d)
		}
	}
	do.Release()
}

// sigma_generated_test.go (hand-written; survives regeneration): wall time of one block
// apply of the generated σ-build (Ms = 0, adc2x: every singlet, triplet and quintet)
// against package dip's production host path (singlet + triplet sectors, 3h1p↔3h1p
// satellites matrix-free), on one synthetic canonical C1 system. The checked-in systems
// have at most 5 occupied orbitals, far from the regime where the per-block costs
// separate (o⁵v² for the δ-gated satellite blocks against o⁴v² for the contractions).
//
//	SIGMABENCH=1 [SIGMA_NOCC=12] [SIGMA_NVIR=48] [SIGMA_COLS=1,16,64] [SIGMA_SKIPHAND=1] \
//	    [SIGMA_BACKEND=cuda [SIGMA_MGPU=4]] \
//	    go test ./internal/adc/isrgen/dip -run TestSigmaVsHandTiming -v -timeout 0

func envInt(key string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(key)); err == nil && v > 0 {
		return v
	}
	return def
}

func timeApply(op lanczos.Operator, n, b, reps int, be backend.Backend) time.Duration {
	rng := rand.New(rand.NewPCG(7, uint64(b)))
	x := make([]float64, n*b)
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	in := backend.BlockView{V: be.Upload(x), Rows: n, Cols: b, Ld: n}
	out := backend.BlockView{V: be.Alloc(n * b), Rows: n, Cols: b, Ld: n}
	op.ApplyBlock(out, in) // warm: assembly, intermediates, plans
	t0 := time.Now()
	for range reps {
		op.ApplyBlock(out, in)
	}
	return time.Since(t0) / time.Duration(reps)
}

func TestSigmaVsHandTiming(t *testing.T) {
	if os.Getenv("SIGMABENCH") == "" {
		t.Skip("set SIGMABENCH=1")
	}
	nocc, nvir := envInt("SIGMA_NOCC", 12), envInt("SIGMA_NVIR", 48)
	reps := envInt("SIGMA_REPS", 2)
	cols := []int{1, 16, 64}
	if s := os.Getenv("SIGMA_COLS"); s != "" {
		cols = nil
		for _, f := range splitComma(s) {
			if v, err := strconv.Atoi(f); err == nil && v > 0 {
				cols = append(cols, v)
			}
		}
	}
	norb := nocc + nvir
	eps := make([]float64, norb)
	for p := range norb {
		if p < nocc {
			eps[p] = -1.5 + float64(p)/float64(nocc)
		} else {
			eps[p] = 0.2 + 2*float64(p-nocc)/float64(nvir)
		}
	}
	rng := rand.New(rand.NewPCG(3, 5))
	t0 := time.Now()
	d, err := fcidump.Synthetic(norb, 2*nocc, eps, 0.05, rng.Float64)
	if err != nil {
		t.Fatal(err)
	}
	ints := integrals.New(d, nocc, nil)
	t.Logf("synthetic C1 system o=%d v=%d built in %v", nocc, nvir, time.Since(t0).Round(time.Millisecond))
	var be backend.Backend = backend.Gonum{}
	if name := os.Getenv("SIGMA_BACKEND"); name != "" {
		if be, err = backend.New(name); err != nil {
			t.Fatal(err)
		}
	}
	// SIGMA_MGPU=G: the σ apply split across G devices of that backend (the hand path stays
	// on one device: its -mgpu needs the low-memory Mode B solver, not a bare apply)
	sbe := be
	var mgpu []backend.Backend
	if g := envInt("SIGMA_MGPU", 1); g > 1 {
		if mgpu, err = backend.NewAll(os.Getenv("SIGMA_BACKEND"), g); err != nil {
			t.Fatal(err)
		}
	}

	var hand []*dippkg.Matrix
	handRows := 0
	for _, spin := range []dippkg.Spin{dippkg.Singlet, dippkg.Triplet} {
		sp := dippkg.NewSpace(nocc, norb, nil, 0, spin)
		mx := dippkg.New(sp, ints, eps, be)
		mx.SetMatFree(matfree.On, 0)
		hand = append(hand, mx)
		handRows += sp.Size()
	}
	ksp, err := khci.NewSpace(khci.Options{K: 2, NOcc: nocc, NVir: nvir, TwoMs: 0, MaxClass: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(mgpu) > 1 {
		n := ksp.Size()
		bounds := make([]int, len(mgpu)+1)
		for i := range bounds {
			bounds[i] = i * n / len(mgpu)
		}
		if sbe, err = backend.NewDistributed(mgpu, n, ksp.MainBlockSize(), bounds); err != nil {
			t.Fatal(err)
		}
	}
	t1 := time.Now()
	op, err := NewSigma(ksp, ints, eps, nocc, "adc2x", sbe)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("rows: dip singlet+triplet %d, σ Ms=0 %d; σ setup %v", handRows, ksp.Size(),
		time.Since(t1).Round(time.Millisecond))
	for _, b := range cols {
		var th time.Duration
		if os.Getenv("SIGMA_SKIPHAND") == "" {
			for _, mx := range hand {
				th += timeApply(mx, mx.Size(), b, reps, be)
			}
		}
		ts := timeApply(op, ksp.Size(), b, reps, sbe)
		t.Logf("b=%3d  hand dip %10v   σ %10v   σ/hand %.2f", b, th.Round(time.Millisecond),
			ts.Round(time.Millisecond), float64(ts)/float64(th))
	}
}

func splitComma(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == ',' {
			if i > start {
				out = append(out, s[start:i])
			}
			start = i + 1
		}
	}
	return out
}
