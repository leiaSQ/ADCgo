//go:build hip || cuda

// Shared device-resident backend orchestration for the GPU backends. The vector
// and matrix handle types, the Backend method bodies, and the row-major↔column-
// major GEMV convention live here once; each vendor's file (hip.go, cuda.go)
// supplies only the thin cgo shim — the handle constructor, the device memory
// ops, and the d-prefix BLAS calls — under a matching build tag. Because hipBLAS
// mirrors the cuBLAS API, the two shims are structural twins.
//
// SymEig is overridden for large matrices only (cuSOLVER/hipSOLVER dsyevd above
// gpuSymEigMin, subject to a device-memory check); below that, and whenever the
// device is short of room, it falls back to the embedded Gonum implementation.
//
// Thread affinity. Every device call is funnelled through one dedicated OS thread
// that owns the BLAS handle and the CUDA/HIP context. Goroutines migrate between OS
// threads, and a cuBLAS handle used from a thread other than its creator's fails
// with CUBLAS_STATUS_INTERNAL_ERROR (14) and no cudaError — a flaky, silent
// wrong-answer risk, since the old shim discarded every status. Serializing here is
// free for this workload (the solver issues one GEMM at a time, and a channel
// round-trip is ~1 us against a >100 us kernel) and it is a precondition for ever
// running sectors concurrently: one handle cannot be shared across worker
// goroutines.
package backend

import (
	"fmt"
	"runtime"
	"unsafe"
)

const elemSize = 8 // sizeof(float64)
const ptrSize = 8  // sizeof(void*) on the supported 64-bit targets

// peerAccessAlreadyEnabled is cudaErrorPeerAccessAlreadyEnabled / hipErrorPeerAccessAlreadyEnabled
// (both 704): a benign status meaning a prior EnablePeerAccess already set the pair up.
const peerAccessAlreadyEnabled = 704

// Registered as a multi-device backend: devCount enumerates the visible GPUs and
// newGPUOn binds an instance to one of them. RegisterMulti also wires the plain
// single-instance entry (device 0), so New(backendName) and -backend cuda|hip keep
// working; NewAll(backendName) fans out one backend per device.
func init() { RegisterMulti(backendName, devCount, newGPUOn) }

// gpuBackend holds the vendor BLAS handle (as an opaque unsafe.Pointer; the shim
// casts it back to the concrete handle type) and the channel onto its owning thread.
type gpuBackend struct {
	Gonum
	dev  int // physical device index this backend is pinned to (0 unless multi-GPU)
	h    unsafe.Pointer
	jobs chan func()

	// Device arrays of device pointers for the batched GEMM, plus their host staging
	// slices. Grown on demand and reused: a batch of 27k pointers is 216 kB, and
	// re-allocating it per call would trade one launch for three cudaMallocs. Only
	// ever touched from the device-owning goroutine, so no locking.
	ptrA, ptrB, ptrC    unsafe.Pointer
	ptrCap              int
	hostA, hostB, hostC []unsafe.Pointer
	// pinned backs hostA/B/C with one page-locked allocation when the backend can provide
	// it (nil otherwise — hostA/B/C are then ordinary Go slices). Freed only on regrow.
	pinned unsafe.Pointer

	// Scratch for DipSatFillJII: the materialized jiiLKK blocks of one apply. Same
	// grown-on-demand, reused-across-calls rationale as the pointer arrays above — this is
	// refilled every mat-vec, so allocating per call would put cudaMalloc on the hot path.
	jiiBuf unsafe.Pointer
	jiiCap int

	// solver is the dense-eigensolver handle (cuSOLVER / hipSOLVER), created lazily on
	// the device thread the first time a matrix large enough to justify it appears.
	solver unsafe.Pointer

	// peers holds the physical device indices this backend has been granted peer (NVLink)
	// read access to, populated by EnablePeerAccess. Written on the device thread during
	// one-time setup (before any solve), read by PeerAvailable — the do() round-trip in
	// EnablePeerAccess sequences the two, so no lock is needed for this workload.
	peers map[int]bool
}

// ensurePtrCap grows the pointer-array scratch to hold at least n entries.
// Must run on the device-owning thread.
func (b *gpuBackend) ensurePtrCap(n int) {
	if n <= b.ptrCap {
		return
	}
	for _, p := range []unsafe.Pointer{b.ptrA, b.ptrB, b.ptrC} {
		if p != nil {
			devFree(p)
		}
	}
	b.ptrA, b.ptrB, b.ptrC = devMalloc(n), devMalloc(n), devMalloc(n) // n*8 bytes each

	// Host staging for the three pointer arrays, uploaded on EVERY batched GEMM (three
	// devH2DPtrs per call). Page-lock them when the backend can: a pageable source makes the
	// driver stage through an internal bounce buffer, and pinned memory is the precondition
	// for ever issuing these asynchronously on a stream. One allocation backs all three, and
	// the pointers written into it are device pointers — not Go pointers — so storing them in
	// C memory does not violate the cgo pointer rules.
	//
	// devHostAlloc returns nil when pinned memory is unavailable (always, on hip; on cuda when
	// the driver declines, since it is a limited system-wide resource). The fallback is the
	// original Go-heap staging, so this is a performance path only, never a correctness one.
	if b.pinned != nil {
		devHostFree(b.pinned)
		b.pinned = nil
	}
	if p := devHostAlloc(3 * n * ptrSize); p != nil {
		b.pinned = p
		all := unsafe.Slice((*unsafe.Pointer)(p), 3*n)
		b.hostA, b.hostB, b.hostC = all[0:n:n], all[n:2*n:2*n], all[2*n:3*n:3*n]
	} else {
		b.hostA = make([]unsafe.Pointer, n)
		b.hostB = make([]unsafe.Pointer, n)
		b.hostC = make([]unsafe.Pointer, n)
	}
	b.ptrCap = n
}

// newGPUOn builds a backend pinned to physical device dev. Every op funnels through
// the one owning thread (do), which calls devSet(dev) once, before blasCreate, so the
// handle and all later allocations/kernels/constant memory live on that device — CUDA
// device selection is thread-current state, and the thread is never reused for another
// device. N independent backends (one per device) therefore share no device state, the
// precondition for running sectors concurrently across GPUs.
func newGPUOn(dev int) Backend {
	b := &gpuBackend{dev: dev, jobs: make(chan func())}
	ready := make(chan struct{})
	go func() {
		// Never unlocked: this goroutine and its OS thread exist to own the device
		// context for the process lifetime.
		runtime.LockOSThread()
		devSet(dev) // bind this thread to `dev` before the handle captures the device
		b.h = blasCreate()
		close(ready)
		for f := range b.jobs {
			f()
		}
	}()
	<-ready
	return b
}

// DeviceIndex reports the physical device this backend is pinned to (for logging and
// the multi-device tests).
func (b *gpuBackend) DeviceIndex() int { return b.dev }

// do runs f on the device-owning thread and blocks until it finishes, re-raising any
// panic (a failed status check) on the calling goroutine.
func (b *gpuBackend) do(f func()) {
	done := make(chan any, 1)
	b.jobs <- func() {
		defer func() { done <- recover() }()
		f()
	}
	if r := <-done; r != nil {
		panic(r)
	}
}

// devVec is a resident vector: a device allocation (base) plus an element offset
// and length, so Slice is a zero-copy view. Only base allocations (off==0) are
// ever Free'd.
type devVec struct {
	base unsafe.Pointer
	off  int
	n    int
}

func (v devVec) Len() int                { return v.n }
func (v devVec) Slice(off, n int) Vector { return devVec{base: v.base, off: v.off + off, n: n} }
func (v devVec) ptr() unsafe.Pointer     { return unsafe.Add(v.base, v.off*elemSize) }

// devMat is a resident matrix: the row-major block data uploaded verbatim. Read
// column-major by the vendor BLAS it is the transpose, which GemvN/GemvT account
// for by swapping the operation flag.
type devMat struct {
	p          unsafe.Pointer
	rows, cols int
}

func (m devMat) Dims() (int, int) { return m.rows, m.cols }

func (b *gpuBackend) Alloc(n int) Vector {
	var p unsafe.Pointer
	b.do(func() {
		p = devMalloc(n)
		devZero(p, n)
	})
	return devVec{base: p, n: n}
}

func (b *gpuBackend) Upload(hostv Vec) Vector {
	var p unsafe.Pointer
	b.do(func() {
		p = devMalloc(len(hostv))
		if len(hostv) > 0 {
			devH2D(p, hostv)
		}
	})
	return devVec{base: p, n: len(hostv)}
}

// Download2D satisfies backend.StridedDownloader: one cudaMemcpy2D for a strided rectangle
// instead of `cols` separate Download round-trips. cudaMemcpyDefault resolves device→host from
// the pointers under UVA, so this reuses the same shim the peer gather uses; the host end is
// pageable, which is correct but not the fastest possible (see the pinned-memory work).
func (b *gpuBackend) Download2D(v Vector, rows, cols, ld int) Vec {
	dv := v.(devVec)
	out := make([]float64, rows*cols)
	if rows > 0 && cols > 0 {
		b.do(func() {
			devMemcpy2D(unsafe.Pointer(&out[0]), rows*elemSize, dv.ptr(), ld*elemSize,
				rows*elemSize, cols)
		})
	}
	return out
}

func (b *gpuBackend) Download(v Vector) Vec {
	dv := v.(devVec)
	out := make([]float64, dv.n)
	if dv.n > 0 {
		b.do(func() { devD2H(out, dv.ptr()) })
	}
	return out
}

func (b *gpuBackend) Zero(v Vector) {
	dv := v.(devVec)
	b.do(func() { devZero(dv.ptr(), dv.n) })
}

func (b *gpuBackend) Copy(dst, src Vector) {
	d, s := dst.(devVec), src.(devVec)
	b.do(func() { devD2D(d.ptr(), s.ptr(), d.n) })
}

func (b *gpuBackend) Free(v Vector) {
	if dv := v.(devVec); dv.off == 0 && dv.base != nil {
		b.do(func() { devFree(dv.base) })
	}
}

func (b *gpuBackend) UploadMat(m Mat) DeviceMat {
	var p unsafe.Pointer
	b.do(func() {
		p = devMalloc(m.Rows * m.Cols)
		if len(m.Data) > 0 {
			devH2D(p, m.Data)
		}
	})
	return devMat{p: p, rows: m.Rows, cols: m.Cols}
}

// FreeMat releases an UploadMat allocation. Without it every uploaded operator
// block leaked device memory for the process lifetime.
func (b *gpuBackend) FreeMat(m DeviceMat) {
	if dm, ok := m.(devMat); ok && dm.p != nil {
		b.do(func() { devFree(dm.p) })
	}
}

// Gemm passes column-major panels straight to the vendor BLAS, which is itself
// column-major. Note the contrast with GemvN/GemvT below: those compensate for
// operator blocks that were uploaded row-major, and that flip must NOT be applied
// here — a BlockView is already in the device's native layout.
func (b *gpuBackend) Gemm(transA, transB bool, alpha float64, a, bb BlockView, beta float64, c BlockView) {
	k := a.Cols
	if transA {
		k = a.Rows
	}
	b.do(func() {
		blasGemm(b.h, transA, transB, c.Rows, c.Cols, k, alpha,
			a.V.(devVec).ptr(), a.Ld, bb.V.(devVec).ptr(), bb.Ld,
			beta, c.V.(devVec).ptr(), c.Ld)
	})
}

func (b *gpuBackend) Axpy(alpha float64, x, y Vector) {
	dx, dy := x.(devVec), y.(devVec)
	b.do(func() { blasAxpy(b.h, dx.ptr(), dy.ptr(), dx.n, alpha) })
}

// AxpyDiag applies a diagonal operator block: y += d ⊙ x. cuBLAS/hipBLAS expose no
// elementwise product, so this host-round-trips. It is used only for the large diagonal
// 3h2p block, once per apply — a cublasDdgmm-based device kernel is a future optimization
// — and keeps the device build correct in the meantime.
func (b *gpuBackend) AxpyDiag(d, x, y Vector) {
	hd, hx, hy := b.Download(d), b.Download(x), b.Download(y)
	for i, dv := range hd {
		hy[i] += dv * hx[i]
	}
	up := b.Upload(hy)
	b.Copy(y, up)
	b.Free(up)
}

func (b *gpuBackend) Dot(x, y Vector) float64 {
	dx, dy := x.(devVec), y.(devVec)
	var r float64
	b.do(func() { r = blasDot(b.h, dx.ptr(), dy.ptr(), dx.n) })
	return r
}

func (b *gpuBackend) Nrm2(x Vector) float64 {
	dx := x.(devVec)
	var r float64
	b.do(func() { r = blasNrm2(b.h, dx.ptr(), dx.n) })
	return r
}

func (b *gpuBackend) Scal(alpha float64, x Vector) {
	dx := x.(devVec)
	b.do(func() { blasScal(b.h, dx.ptr(), dx.n, alpha) })
}

// GemmMat: c := alpha*op(a)*b + beta*c for a row-major uploaded operator block a.
// Read column-major the stored block is aᵀ (dims cols×rows, lda=cols), so the
// operation flag inverts — exactly as in GemvN/GemvT, and unlike Gemm above, whose
// operands are already column-major BlockViews.
func (b *gpuBackend) GemmMat(transA bool, alpha float64, a DeviceMat, bb BlockView, beta float64, c BlockView) {
	m := a.(devMat)
	b.do(func() {
		blasGemm(b.h, !transA, false, c.Rows, c.Cols, bb.Rows, alpha,
			m.p, m.cols, bb.V.(devVec).ptr(), bb.Ld,
			beta, c.V.(devVec).ptr(), c.Ld)
	})
}

// GemmMatBatched issues one batched GEMM for a whole set of same-shaped operator
// blocks. This is the difference between 11.1 M cuBLAS calls per formic-acid sector and
// ~200 k: at ~16 µs of launch overhead per call, the un-batched apply spent 181 s of a
// 379 s sector doing nothing but dispatch.
//
// The batch members execute concurrently and accumulate into c (beta = 1), so the
// caller must supply pairwise non-overlapping c — which is precisely PlanBatches'
// contract. Shapes are read from element 0; the contract requires them uniform.
func (b *gpuBackend) GemmMatBatched(transA bool, alpha float64, a []DeviceMat, bb []BlockView, beta float64, c []BlockView) {
	n := len(a)
	if n == 0 {
		return
	}
	if n == 1 { // a batched call of one is pure overhead
		b.GemmMat(transA, alpha, a[0], bb[0], beta, c[0])
		return
	}
	m0 := a[0].(devMat)
	b.do(func() {
		b.ensurePtrCap(n)
		for i := range n {
			b.hostA[i] = a[i].(devMat).p
			b.hostB[i] = bb[i].V.(devVec).ptr()
			b.hostC[i] = c[i].V.(devVec).ptr()
		}
		devH2DPtrs(b.ptrA, b.hostA[:n])
		devH2DPtrs(b.ptrB, b.hostB[:n])
		devH2DPtrs(b.ptrC, b.hostC[:n])
		// Same row-major→column-major flag inversion as GemmMat.
		blasGemmBatched(b.h, !transA, false, c[0].Rows, c[0].Cols, bb[0].Rows, alpha,
			b.ptrA, m0.cols, b.ptrB, bb[0].Ld, beta, b.ptrC, c[0].Ld, n)
	})
}

// gpuSymEigMin is DeviceSymEigMin (perf.go), which lives outside the GPU build tags so
// the backend-selection cost model can see it. Above it the O(n³) factorization dominates
// everything: the projected Lanczos matrix reaches n ≈ 11600 for formic acid, where this
// phase was 120 s of a 379 s sector.
const gpuSymEigMin = DeviceSymEigMin

// symEigMargin is headroom left free on the device, so a SymEig never evicts the
// resident basis or operator blocks.
const symEigMargin = 128 << 20

// SymEig overrides the inherited host implementation with cuSOLVER's divide-and-conquer
// dsyevd, but only when the matrix is big enough to pay for the transfers AND the
// device has room. Otherwise it falls back to the embedded Gonum path, which under the
// openblas tag is LAPACKE_dsyevd. The fallback is not a slow path — it is the same
// algorithm on the host — so degrading is cheap and always safe.
func (b *gpuBackend) SymEig(a Mat) ([]float64, Mat) {
	n := a.Rows
	if n < gpuSymEigMin {
		return b.Gonum.SymEig(a)
	}

	out := NewMat(n, n)
	copy(out.Data, a.Data)
	evals := make([]float64, n)

	ok := false
	b.do(func() {
		bytes := uint64(n) * uint64(n) * elemSize
		if free, _ := devMemInfo(); free < bytes+uint64(n)*elemSize+symEigMargin {
			return // not enough room even for A and W; stay on the host
		}
		if b.solver == nil {
			b.solver = solverCreate()
		}
		dA := devMalloc(n * n)
		defer devFree(dA)
		dW := devMalloc(n)
		defer devFree(dW)
		devH2D(dA, out.Data)

		lwork := solverDsyevdBufferSize(b.solver, n, dA, n, dW)
		if free, _ := devMemInfo(); free < uint64(lwork)*elemSize+symEigMargin {
			return // workspace does not fit; the deferred frees release A and W
		}
		dWork := devMalloc(lwork)
		defer devFree(dWork)
		dInfo := devMalloc(1)
		defer devFree(dInfo)

		if info := solverDsyevd(b.solver, n, dA, n, dW, dWork, lwork, dInfo); info != 0 {
			panic(fmt.Sprintf("backend: cusolver dsyevd did not converge (devInfo=%d, n=%d)", info, n))
		}
		devD2H(out.Data, dA)
		devD2H(evals, dW)
		ok = true
	})
	if !ok {
		return b.Gonum.SymEig(a)
	}

	// The device left the eigenvectors as columns of a column-major matrix. Read back
	// linearly into row-major storage that is its transpose, so rows currently hold the
	// eigenvectors; transpose in place to restore the columns-are-eigenvectors contract.
	transposeSquareInPlace(out)
	return evals, out
}

// transposeSquareInPlace transposes an n×n row-major matrix without a second n²
// allocation — the matrices here reach 1.1 GB.
func transposeSquareInPlace(m Mat) {
	n := m.Rows
	for i := range n {
		for j := i + 1; j < n; j++ {
			m.Data[i*n+j], m.Data[j*n+i] = m.Data[j*n+i], m.Data[i*n+j]
		}
	}
}

// DeviceMem reports free and total device memory, so the backend chooser can refuse a
// sector that would not fit. Satisfies backend.DeviceMemory.
func (b *gpuBackend) DeviceMem() (free, total uint64) {
	b.do(func() { free, total = devMemInfo() })
	return free, total
}

// EnablePeerAccess authorizes this device to read the memory of each peer that supports it,
// so the distributed backend's mat-vec input gather can copy device-to-device (NVLink)
// instead of staging through the host. Runs cudaDeviceEnablePeerAccess on this backend's
// owning thread (device selection is thread-current state); a peer already enabled by an
// earlier call reports error 704, which counts as success. Satisfies backend.PeerCopier.
func (b *gpuBackend) EnablePeerAccess(peers []Backend) {
	if b.peers == nil {
		b.peers = make(map[int]bool)
	}
	for _, p := range peers {
		pg, ok := p.(*gpuBackend)
		if !ok || pg.dev == b.dev || !devCanPeer(b.dev, pg.dev) {
			continue
		}
		peerDev := pg.dev
		b.do(func() {
			if st := devEnablePeer(peerDev); st == 0 || st == peerAccessAlreadyEnabled {
				b.peers[peerDev] = true
			}
		})
	}
}

// PeerAvailable reports whether this backend can peer-copy from `from`'s device.
func (b *gpuBackend) PeerAvailable(from Backend) bool {
	pg, ok := from.(*gpuBackend)
	return ok && b.peers[pg.dev]
}

// Sync drains this backend's device stream (see PeerCopier.Sync). Runs on the owning thread.
func (b *gpuBackend) Sync() { b.do(func() { devSync() }) }

// PeerCopy2D pulls a rows×cols column-major band from src (resident on `from`, column stride
// srcLd) into dst (resident on this backend, column stride dstLd), device-to-device. It runs on
// this backend's owning thread, which already holds peer read access to `from` (granted by
// EnablePeerAccess); cudaMemcpy2D with cudaMemcpyDefault resolves both ends from their pointers
// under UVA. Satisfies backend.PeerCopier.
//
// dstLd is explicit so a caller can scatter bands into a taller buffer: the -mgpu satellite
// gather assembles a full-height n×w slab from every partition's band, passing dstLd = n and a
// dst already sliced to the band's row offset. Pass dstLd = rows for a compact destination.
func (b *gpuBackend) PeerCopy2D(dst, src Vector, from Backend, rows, cols, dstLd, srcLd int) {
	dp := dst.(devVec).ptr()
	sp := src.(devVec).ptr()
	b.do(func() {
		devMemcpy2D(dp, dstLd*elemSize, sp, srcLd*elemSize, rows*elemSize, cols)
	})
}

// GemvN: y += alpha*A*x. A is row-major rows×cols; stored column-major it is Aᵀ
// (cols rows, rows cols, lda=cols), so A*x = op(stored, T)*x.
func (b *gpuBackend) GemvN(alpha float64, a DeviceMat, x, y Vector) {
	b.gemv(true, a.(devMat), alpha, x, y)
}

// GemvT: y += alpha*Aᵀ*x = op(stored, N)*x.
func (b *gpuBackend) GemvT(alpha float64, a DeviceMat, x, y Vector) {
	b.gemv(false, a.(devMat), alpha, x, y)
}

func (b *gpuBackend) gemv(trans bool, m devMat, alpha float64, x, y Vector) {
	dx, dy := x.(devVec), y.(devVec)
	b.do(func() { blasGemv(b.h, trans, m.cols, m.rows, alpha, m.p, m.cols, dx.ptr(), 1, dy.ptr()) })
}

var (
	_ Backend    = (*gpuBackend)(nil)
	_ PeerCopier = (*gpuBackend)(nil)
)
