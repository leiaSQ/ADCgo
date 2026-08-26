//go:build cuda || hip

package backend

import "testing"

// TestGemmLeadingDimensionCrossesInt32 probes the one scale-dependent hazard the mgpu parity tests
// structurally cannot reach: a GEMM whose OUTPUT panel spans more than 2^31 elements because its
// leading dimension is a device row band, not because any single dimension is large.
//
// Why this shape and not a big square: distBackend.gemmMatOne applies the operator one block at a
// time, so m (the block's rows) and k stay small — but C is a view into the device's whole row band,
// so ldc is the partition height and the far column sits at (cols-1)*ldc. For the production DIP sector
// that killed job 14040960 (n=10,014,483 over 8 partitions, panel width 1711) that offset is
//
//	1710 * 1,251,810 + m  ~=  2.1418e9   vs   INT32_MAX = 2.1475e9
//
// i.e. 0.26% of headroom. The mgpu tests run at n ~= 200, four orders of magnitude below any of
// this, so they cannot say whether the vendor BLAS (or our wrapper) computes that offset in 32 or
// 64 bits. Job 14040960 died with cudaErrorLaunchFailure (719) reported at a later
// cudaDeviceSynchronize, which is exactly how a wrapped-index out-of-bounds store would surface.
//
// The parameters below deliberately STRADDLE the boundary: with ld = 1,300,000, column j starts at
// j*ld, which crosses 2^31 between j=1651 and j=1652. The check therefore reads a column below the
// boundary, the first column above it, and the last column. A 32-bit offset anywhere wraps negative
// and either faults or writes somewhere else, so both outcomes fail this test rather than passing
// quietly.
//
// Cost: ld*cols*8 = 17.8 GB of device memory, allocated (not uploaded) and never copied to the
// host in full — only three m-element columns come back. Skips when the device cannot spare it.
func TestGemmLeadingDimensionCrossesInt32(t *testing.T) {
	gpu, name := gpuUnderTest(t)

	const (
		ld   = 1_300_000 // device row band (production n/8 rounded up); the leading dimension of C
		cols = 1711      // the production system's panel width
		m    = 64        // one operator block's rows: small, as in gemmMatOne
		k    = 1
	)
	const total = uint64(ld) * uint64(cols) // 2.2243e9 elements
	if total <= 1<<31 {
		t.Fatalf("test misconfigured: %d elements does not cross 2^31", total)
	}
	need := total*elemSize + (1 << 30) // C plus slack for A, B and the context

	if dev, ok := gpu.(*gpuBackend); ok {
		var free uint64
		dev.do(func() { free, _ = devMemInfo() })
		if free < need {
			t.Skipf("%s: needs %.1f GB free, have %.1f GB", name, float64(need)/(1<<30), float64(free)/(1<<30))
		}
	}

	// C = A*B with A = ones(m x 1) and B = [1, 2, ... cols], so column j must hold j+1 in every
	// one of its m rows. A wrong-column write is therefore detected by VALUE, not just by a fault.
	aData := make([]float64, m*k)
	for i := range aData {
		aData[i] = 1
	}
	bData := make([]float64, k*cols)
	for j := range bData {
		bData[j] = float64(j + 1)
	}

	av := gpu.Upload(aData)
	defer gpu.Free(av)
	bv := gpu.Upload(bData)
	defer gpu.Free(bv)

	cv := gpu.Alloc(int(total)) // Alloc zeroes, so an un-written column reads back 0
	defer gpu.Free(cv)

	a := BlockView{V: av, Rows: m, Cols: k, Ld: m}
	b := BlockView{V: bv, Rows: k, Cols: cols, Ld: k}
	c := BlockView{V: cv, Rows: m, Cols: cols, Ld: ld}

	gpu.Gemm(false, false, 1, a, b, 0, c)

	// Below the boundary, the first column past it, and the last column.
	for _, j := range []int{0, 1651, 1652, cols - 1} {
		off := uint64(j) * uint64(ld)
		got := gpu.Download(cv.Slice(int(off), m))
		want := float64(j + 1)
		for i, v := range got {
			if v != want {
				t.Fatalf("%s: C[%d,%d] = %g, want %g (column offset %d, %s 2^31) — "+
					"the GEMM output addressing is not 64-bit clean at this leading dimension",
					name, i, j, v, want, off, map[bool]string{true: "above", false: "below"}[off >= 1<<31])
			}
		}
	}
}
