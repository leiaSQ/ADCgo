//go:build hip || cuda

package lanczos

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// gpuBackend takes testing.TB so benchmarks can use it too, not just tests.
func gpuBackend(t testing.TB) (backend.Backend, string) {
	t.Helper()
	for _, name := range backend.Available() {
		if name == "gonum" {
			continue
		}
		be, err := backend.New(name)
		if err != nil {
			t.Fatalf("New(%q): %v", name, err)
		}
		return be, name
	}
	t.Skip("no accelerated backend registered in this build")
	return nil, ""
}

func buildH2OWith(t *testing.T, spin dip.Spin, be backend.Backend) *dip.Matrix {
	t.Helper()
	path := filepath.Join("..", "..", "..", "testdata", "h2o.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := dip.NewSpace(nocc, d.NORB, nil, 0, spin)
	return dip.New(sp, integrals.New(d, nocc, nil), eps, be)
}

// TestGPULanczosMatchesDense is the M3 spectrum-parity gate: a full block-Lanczos
// solve with the device mat-vec must reproduce the pure-Go dense spectrum. This
// exercises the resident GEMV (ApplyFull) plus the on-device AXPY/DOT/NRM2/SCAL of
// the Gram–Schmidt recurrence end-to-end.
func TestGPULanczosMatchesDense(t *testing.T) {
	gpu, name := gpuBackend(t)
	ref := backend.Gonum{}

	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		dense, _ := ref.SymEig(buildH2OWith(t, spin, ref).BuildMatrix())

		res := Solve(buildH2OWith(t, spin, gpu), gpu, Options{}) // full subspace → exact

		var maxErr float64
		for _, e := range dense {
			best := math.Inf(1)
			for _, r := range res.Values {
				if d := math.Abs(r - e); d < best {
					best = d
				}
			}
			if best > maxErr {
				maxErr = best
			}
		}
		if maxErr > 1e-7 {
			t.Errorf("%s spin %d: device Lanczos vs dense max eigenvalue error %g (>1e-7)", name, spin, maxErr)
		}
	}
}

// TestGPUBandEigReplayParity is the gate on -eig-device: the device replay of the projected
// banded eigensolver's eigenvector rotations must be BIT-IDENTICAL to the host replay, not close
// to it. The comparison is on bits because these rows become pole strengths, and because the
// eigenvalue list carries Lanczos ghosts whose ordering a changed last bit can flip.
//
// The hazard this exists for is FMA contraction: Go on amd64 does not fuse a*b+c, nvcc and hipcc
// do by default, and a fused product rounds once where Go rounds twice. bandeig_kernels.cu uses
// explicit __dmul_rn/__dadd_rn/__dsub_rn to prevent that, and this test is what proves the
// prevention works on the hardware and toolchain actually in use — the parity cannot be checked
// on a machine without the device, so a green CPU suite says nothing about it.
//
// It drives the solver twice over identical matrices rather than comparing replay in isolation, so
// a mistake anywhere in the upload/flush/swap/download sequence is caught too, including the
// staleness rule that the host z is invalid while the device holds the accumulator.
func TestGPUBandEigReplayParity(t *testing.T) {
	be, name := gpuBackend(t)
	accel, ok := be.(backend.BandEigKernels)
	if !ok {
		t.Skipf("backend %q does not implement BandEigKernels", name)
	}
	for _, tc := range []struct{ dim, band int }{{40, 7}, {200, 2}, {512, 5}, {260, 128}, {1000, 10}} {
		seed := int64(1000 + tc.dim*100 + tc.band)
		main := (tc.band + 1) / 2

		_, hostBS := buildBanded(tc.dim, tc.band, seed)
		wantD, wantZ, wantLD, _ := bandSymDiagFastOpts(hostBS, bandEigOpts{topRows: main, botRows: main})

		_, devBS := buildBanded(tc.dim, tc.band, seed)
		gotD, gotZ, gotLD, _ := bandSymDiagFastOpts(devBS, bandEigOpts{topRows: main, botRows: main, accel: accel})

		if gotLD != wantLD {
			t.Fatalf("%s dim=%d band=%d: device ld=%d, host ld=%d", name, tc.dim, tc.band, gotLD, wantLD)
		}
		bad := 0
		for k := range wantD {
			if gotD[k] != wantD[k] {
				if bad++; bad <= 5 {
					t.Errorf("%s dim=%d band=%d: eval[%d] device %.17g, host %.17g",
						name, tc.dim, tc.band, k, gotD[k], wantD[k])
				}
			}
		}
		rows := 2 * main
		for k := range tc.dim {
			for r := range rows {
				if g, w := gotZ[k*gotLD+r], wantZ[k*wantLD+r]; g != w {
					if bad++; bad <= 5 {
						t.Errorf("%s dim=%d band=%d: z[eigenvector %d row %d] device %.17g, host %.17g",
							name, tc.dim, tc.band, k, r, g, w)
					}
				}
			}
		}
		if bad > 5 {
			t.Errorf("%s dim=%d band=%d: %d differing values in total", name, tc.dim, tc.band, bad)
		}
	}
}

// BenchmarkGPUBandEigReplay measures what the device replay actually achieves, in eigenvector
// updates per second, as a function of the two things that decide it: how many ops a concurrent run
// holds, and whether the run is replayed concurrently at all.
//
// WHY THE SWEEP IS THE MEASUREMENT. The sequential kernel's only parallelism is the accumulated row
// count, so at production's 3080 rows it runs ~96 warps on a 132-SM H200 and is latency-bound — flat
// across every block size from 32 to 1024 threads, 195 GB/s, 4% of HBM. The 2-D kernel adds the ops
// within a run as a second axis, so its throughput depends on the run length, and a real tape's run
// length is the chase length, dim/band: ~100 ops at band 3079 and ~2406 after a two-stage reduction
// to b2 = 128. ops=1 is the launch-overhead floor and is what sets whether the projected 7 h -> 0.5 h
// is reachable at all; ops=100 and ops=2406 are the two shapes production actually runs.
//
// The seq/par pair at each run length is the comparison that matters. par at ops=1 must cost the same
// as seq (the implementation falls back for a single-op run), so a difference there is measuring the
// harness, not the kernel.
//
// rows = 3080 is production's exact accumulated width, so these numbers transfer.
func BenchmarkGPUBandEigReplay(b *testing.B) {
	be, name := gpuBackend(b)
	accel, ok := be.(backend.BandEigKernels)
	if !ok {
		b.Skipf("backend %q does not implement BandEigKernels", name)
	}
	const rows, n, nops = 3080, 1 << 14, 1 << 14
	ld := rows
	z := make([]float64, ld*n)
	rng := rand.New(rand.NewSource(1))
	for i := range z {
		z[i] = rng.NormFloat64()
	}
	dev := accel.NewBandEigZ(z, rows, ld, n)
	defer dev.Free()

	for _, rs := range []int{1, 8, 100, 400, 2406} {
		// Columns must be pairwise disjoint WITHIN a run, or the concurrent form races and the
		// benchmark is measuring an incorrect computation. A rotTD on column c touches c and c+1, so
		// stride 3 separates them, and restarting the pattern at every run boundary keeps each run
		// self-contained. (The earlier version used (o*7)%(n-3) across the whole tape, which repeats
		// after n-3 ops and was therefore NOT disjoint despite the comment saying so.)
		if 2+3*(rs-1)+1 >= n {
			b.Fatalf("run size %d needs more than %d columns", rs, n)
		}
		kind := make([]uint8, nops)
		col := make([]int32, nops)
		f1 := make([]float64, nops)
		f2 := make([]float64, nops)
		for o := range nops {
			kind[o] = zOpRotTD
			col[o] = int32(2 + 3*(o%rs))
			f1[o], f2[o] = 0.6, -0.8
		}
		seg := make([]int32, 0, nops/rs+1)
		for hi := rs; hi <= nops; hi += rs {
			seg = append(seg, int32(hi))
		}
		for _, par := range []bool{false, true} {
			segPar := make([]bool, len(seg))
			for i := range segPar {
				segPar[i] = par
			}
			mode := "seq"
			if par {
				mode = "par"
			}
			b.Run(fmt.Sprintf("ops=%d/%s", rs, mode), func(b *testing.B) {
				// One rotTD touches 2 entries per row and writes both: 3 updates per row per op,
				// matching how the CPU-side BenchmarkZReplayBlock counts, so the two are directly
				// comparable. The reported MB/s is therefore eigenvector updates/s.
				b.SetBytes(int64(nops) * int64(rows) * 3)
				for b.Loop() {
					dev.Replay(kind, col, f1, f2, nops, seg, segPar)
				}
			})
		}
	}
}
