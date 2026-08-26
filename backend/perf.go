package backend

import (
	"math"
	"math/rand"
	"time"
)

// Backend selection by measurement rather than by hardcoded size thresholds.
//
// Whether the GPU wins depends on the card, not on the molecule: on this box's consumer
// RTX 3060 Ti, FP64 GEMM is only ~1.6-2x the 6-core CPU (consumer Ampere runs FP64 at
// 1/64 of FP32), while a datacenter card is 40-60x. A threshold tuned here would be
// wrong there. So Calibrate times the two kernels that dominate a solve and the chooser
// compares predicted seconds.
//
// The mat-vec (`apply`) is NOT modelled — it is measured. Its cost depends on the
// operator's block structure, not just on n and dim: on formic acid's largest sector the
// GPU's apply beats the CPU's 2.5x, while on CH2O's sectors it loses badly, because the
// batched calls are then too small to cover the launch overhead. A flop-based estimate
// gets the sign wrong. The chooser therefore assembles each candidate and times one real
// ApplyBlock, extrapolating by the number of block iterations. That costs a fraction of a
// second per sector and removes the largest source of modelling error.

// DeviceSymEigMin is the matrix dimension below which a device backend keeps SymEig on
// the host: the device path costs two n² transfers plus a workspace allocation, and the
// host LAPACK (dsyevd under the openblas tag) is already fast at small n. Defined here,
// outside the GPU build tags, because the cost model needs it too.
const DeviceSymEigMin = 2000

// Perf holds a backend's measured throughput, in FLOP/s.
type Perf struct {
	GemmFlops float64 // level-3 GEMM, the orthogonalization and projection kernel
	EigFlops  float64 // symmetric eigendecomposition of the projected matrix
}

// DeviceMemory is implemented by backends with a separate memory space. The chooser uses
// it to refuse a sector that would not fit; host backends do not implement it.
type DeviceMemory interface {
	DeviceMem() (free, total uint64)
}

// PeerCopier is implemented by device backends that can move resident data directly to a
// peer device (NVLink/xGMI) without staging through host memory. The distributed
// (multi-GPU) backend uses it to accelerate its one large cross-partition data mover —
// the mat-vec input gather — and gracefully falls back to Download/Upload host staging
// for any sub-backend that does not implement it (e.g. Gonum) or any device pair without
// peer access.
type PeerCopier interface {
	// EnablePeerAccess best-effort authorizes this backend's device to read the memory of
	// each backend in peers. Non-*gpuBackend peers, self, and pairs without peer capability
	// are skipped, leaving the host-staging fallback in place for those. Idempotent.
	EnablePeerAccess(peers []Backend)

	// PeerAvailable reports whether this backend can peer-copy from `from`'s device (i.e.
	// EnablePeerAccess succeeded for that pair).
	PeerAvailable(from Backend) bool

	// Sync blocks until this backend's queued device work completes. A peer read does NOT
	// drain the source device's stream the way the host-staged Download path implicitly does,
	// so the distributed backend calls Sync on the SOURCE before a peer copy to guarantee a
	// pending async write (e.g. a Krylov-panel scaling) has finished producing the band.
	Sync()

	// PeerCopy2D copies a rows×cols column-major band from src (resident on `from`, column
	// stride srcLd) into dst (resident on this backend, column stride dstLd), device-to-device.
	// dst and src must be this backend's / from's native Vectors. Runs on this backend's
	// owning thread.
	//
	// dstLd is explicit so bands can be scattered into a taller buffer: the -mgpu satellite
	// gather builds a full-height n×w slab out of every partition's band with dstLd = n and dst
	// pre-sliced to the band's row offset. Pass dstLd = rows for a compact destination.
	PeerCopy2D(dst, src Vector, from Backend, rows, cols, dstLd, srcLd int)
}

// calibGemm/calibEig are sized to be representative but cheap. The eig size must exceed
// gpuSymEigMin, or a GPU backend would silently fall back to the host and Calibrate would
// measure the wrong thing.
const (
	calibGemmN   = 4096
	calibGemmDim = 1024
	calibGemmB   = 64
	calibEigN    = 2176
)

// Calibrate measures a backend's GEMM and SymEig throughput. It runs a warm-up of each
// (cuBLAS and cuSOLVER both defer setup to first use), then takes the best of a few
// repeats — best-of-N, because the 5600X's boost clock makes single runs vary by ~20%.
//
// Cost is a few seconds. Only the `auto` backend calls it, and only when there is more
// than one candidate to choose between.
func Calibrate(be Backend) Perf {
	rng := rand.New(rand.NewSource(1))
	fill := func(n int) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		return v
	}

	// --- GEMM: P = Bᵀ·V, the projection shape the solver actually issues. ---
	b := BlockView{V: be.Upload(fill(calibGemmN * calibGemmDim)), Rows: calibGemmN, Cols: calibGemmDim, Ld: calibGemmN}
	v := BlockView{V: be.Upload(fill(calibGemmN * calibGemmB)), Rows: calibGemmN, Cols: calibGemmB, Ld: calibGemmN}
	p := BlockView{V: be.Alloc(calibGemmDim * calibGemmB), Rows: calibGemmDim, Cols: calibGemmB, Ld: calibGemmDim}
	defer be.Free(b.V)
	defer be.Free(v.V)
	defer be.Free(p.V)

	gemmFlops := 2.0 * calibGemmN * calibGemmDim * calibGemmB
	be.Gemm(true, false, 1, b, v, 0, p) // warm-up
	_ = be.Nrm2(p.V)                    // drain the device queue: Gemm is asynchronous
	best := math.Inf(1)
	for range 3 {
		t0 := time.Now()
		for range 4 {
			be.Gemm(true, false, 1, b, v, 0, p)
		}
		_ = be.Nrm2(p.V) // sync before stopping the clock
		if d := time.Since(t0).Seconds() / 4; d < best {
			best = d
		}
	}
	perf := Perf{GemmFlops: gemmFlops / best}

	// --- SymEig on a matrix large enough to take a GPU backend's device path. ---
	a := NewMat(calibEigN, calibEigN)
	src := fill(calibEigN * calibEigN)
	for i := range calibEigN {
		for j := i; j < calibEigN; j++ {
			val := src[i*calibEigN+j]
			a.Set(i, j, val)
			a.Set(j, i, val)
		}
	}
	eigFlops := (4.0 / 3.0) * math.Pow(calibEigN, 3)
	be.SymEig(a) // warm-up (cuSOLVER handle creation, workspace query)
	t0 := time.Now()
	be.SymEig(a)
	perf.EigFlops = eigFlops / time.Since(t0).Seconds()

	return perf
}

// SolveSeconds predicts the GEMM- and eigensolver-bound part of one sector's solve:
//
//	orthogonalization + projection ≈ 5·n·dim² flops of GEMM
//	Rayleigh-Ritz                  ≈ (4/3)·dim³ flops of SymEig
//
// The GEMM coefficient: the projected matrix costs n·dim² summed over block iterations,
// and one CGS2 (two passes, two GEMMs each) costs 4·n·dim². The caller adds the measured
// apply time; see the note at the top of this file for why that one is not modelled.
func (p Perf) SolveSeconds(n, dim int) float64 {
	nf, df := float64(n), float64(dim)
	return 5*nf*df*df/p.GemmFlops + (4.0/3.0)*df*df*df/p.EigFlops
}

// EigSeconds predicts a dense-path solve, which is one SymEig of the full sector.
func (p Perf) EigSeconds(n int) float64 {
	nf := float64(n)
	return (4.0 / 3.0) * nf * nf * nf / p.EigFlops
}

// SectorBytes estimates the device memory one sector needs: the Krylov basis, the two
// n×b work panels, the projection scratch, the operator, and — when the projected matrix
// is large enough for the device eigensolver — its matrix plus workspace.
//
// opFrac bounds the block-sparse operator's density. Measured at ~0.20 of n² across
// formic acid's sectors; 0.5 leaves a 2.5x margin, so the check errs toward the CPU.
func SectorBytes(n, dim, b int) uint64 {
	const opFrac = 0.5
	const eigWorkFactor = 2.5 // cuSOLVER's A plus its dsyevd workspace
	nf, df, bf := float64(n), float64(dim), float64(b)

	bytes := nf*df*8 + // basis
		2*nf*bf*8 + // W and V panels
		df*bf*8 + // projection scratch
		opFrac*nf*nf*8 // assembled operator

	if dim >= DeviceSymEigMin {
		bytes += eigWorkFactor * df * df * 8
	}
	return uint64(bytes)
}
