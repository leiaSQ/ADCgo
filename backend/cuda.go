//go:build cuda && !hip

// cuBLAS cgo shim for the shared GPU backend (gpu_device.go) — the NVIDIA twin of
// hip.go. Compiled here against the CUDA 13.x developer libraries; the actual GPU
// run happens on an NVIDIA system. Structurally identical to the hipBLAS shim
// (cuBLAS mirrors the hipBLAS API); only the headers, libraries, and enum/type
// names differ. The `!hip` tag keeps the two shims mutually exclusive in a build.
package backend

/*
// CUDA install prefix defaults to /usr/local/cuda; override it for a module-based
// toolkit (e.g. Helix's cuda/13.2) by exporting CGO_CFLAGS="-I$CUDA_HOME/include" and
// CGO_LDFLAGS="-L$CUDA_HOME/lib64" before `go build -tags cuda` — cgo appends those to
// the directives below.
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcublas -lcudart -lcusolver
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdlib.h>

// Every CUDA/cuBLAS entry point returns a status. Discarding them turns an OOM into
// a NULL pointer and a rejected kernel into a silent no-op; the latter surfaces only
// as a benchmark reporting throughput above the card's FP64 peak. Surface them.
static int  dev_last_error(void)                       { return (int)cudaGetLastError(); }
static int  dev_sync(void)                             { return (int)cudaDeviceSynchronize(); }

// Multi-GPU: dev_count enumerates visible devices (honours CUDA_VISIBLE_DEVICES);
// dev_set binds the CALLING thread's context to a device. Device is thread-current
// state, so a backend that pins one OS thread for life needs only one dev_set (before
// its handle is created) to route every later op to that device — see gpu_device.go.
static int  dev_count(void)                            { int n = 0; cudaGetDeviceCount(&n); return n; }
static int  dev_set(int dev)                           { return (int)cudaSetDevice(dev); }
static void* dev_malloc(size_t bytes)                  { void* p = NULL; cudaMalloc(&p, bytes); return p; }
static void  dev_free(void* p)                         { cudaFree(p); }
static int   dev_zero(void* p, size_t bytes)           { return (int)cudaMemset(p, 0, bytes); }
static int   dev_h2d(void* d, const void* s, size_t b) { return (int)cudaMemcpy(d, s, b, cudaMemcpyHostToDevice); }
static int   dev_d2h(void* d, const void* s, size_t b) { return (int)cudaMemcpy(d, s, b, cudaMemcpyDeviceToHost); }
static int   dev_d2d(void* d, const void* s, size_t b) { return (int)cudaMemcpy(d, s, b, cudaMemcpyDeviceToDevice); }

// Multi-GPU peer (NVLink) copy for the distributed backend's cross-partition input
// gather. dev_can_peer asks whether `dev` may read `peer`'s memory; dev_enable_peer
// authorizes the CALLING thread's current device to read `peer` (so it runs on the
// enabling device's thread, and returns cudaErrorPeerAccessAlreadyEnabled == 704 if a
// prior call set it up — the caller treats that as success). dev_memcpy2d copies a
// strided (column-major, lda > rows) band peer-to-peer in one call: with peer access
// enabled and UVA, cudaMemcpyDefault infers each end's device from its pointer.
static int  dev_can_peer(int dev, int peer) { int ok = 0; cudaDeviceCanAccessPeer(&ok, dev, peer); return ok; }
static int  dev_enable_peer(int peer)       { return (int)cudaDeviceEnablePeerAccess(peer, 0); }
static int  dev_memcpy2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
	return (int)cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDefault);
}

// Page-locked (pinned) host memory. A transfer from pageable memory is staged by the driver
// through an internal bounce buffer; a pinned source is DMA'd directly, and is additionally the
// precondition for cudaMemcpyAsync ever overlapping a copy with compute on a stream.
static int  dev_host_alloc(void** p, size_t bytes) { return (int)cudaHostAlloc(p, bytes, cudaHostAllocDefault); }
static int  dev_host_free(void* p)                 { return (int)cudaFreeHost(p); }

static cublasHandle_t blas_create(int* status) { cublasHandle_t h = NULL; *status = (int)cublasCreate(&h); return h; }

static void   blas_axpy(cublasHandle_t h, int n, double a, const double* x, double* y) { cublasDaxpy(h, n, &a, x, 1, y, 1); }
static double blas_dot (cublasHandle_t h, int n, const double* x, const double* y)     { double r = 0; cublasDdot(h, n, x, 1, y, 1, &r); return r; }
static double blas_nrm2(cublasHandle_t h, int n, const double* x)                      { double r = 0; cublasDnrm2(h, n, x, 1, &r); return r; }
static void   blas_scal(cublasHandle_t h, int n, double a, double* x)                  { cublasDscal(h, n, &a, x, 1); }

// y := alpha*op(A)*x + beta*y, op = trans?OP_T:OP_N, A column-major m×n, lda.
static void blas_gemv(cublasHandle_t h, int trans, int m, int n, double alpha,
                      const double* A, int lda, const double* x, double beta, double* y) {
	cublasOperation_t op = trans ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasDgemv(h, op, m, n, &alpha, A, lda, x, 1, &beta, y, 1);
}

// C := alpha*op(A)*op(B) + beta*C, all column-major. cuBLAS is column-major
// natively and BlockView already is, so the operands pass straight through --
// unlike blas_gemv above, whose caller compensates for row-major uploaded blocks.
// Returns the cuBLAS status: a rejected DGEMM is a silent no-op otherwise, which
// shows up only as an impossibly fast benchmark.
static int blas_gemm(cublasHandle_t h, int transA, int transB, int m, int n, int k,
                     double alpha, const double* A, int lda, const double* B, int ldb,
                     double beta, double* C, int ldc) {
	cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
	return (int)cublasDgemm(h, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// Batched DGEMM: one launch for `batch` same-shaped products. A, B, C are DEVICE
// arrays of device pointers. The batch members run concurrently, so the caller must
// guarantee the C pointers do not overlap (see backend.PlanBatches).
static int blas_gemm_batched(cublasHandle_t h, int transA, int transB, int m, int n, int k,
                             double alpha, const double* const* A, int lda,
                             const double* const* B, int ldb,
                             double beta, double* const* C, int ldc, int batch) {
	cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
	return (int)cublasDgemmBatched(h, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, batch);
}

// ---- cuSOLVER: divide-and-conquer symmetric eigensolver on the device ----
//
// The input is a fully symmetric matrix, so its row-major and column-major readings
// coincide and the uplo choice is immaterial. On exit A holds the eigenvectors as
// COLUMNS in column-major order; read back as row-major that is the transpose, which
// the Go caller undoes.
static cusolverDnHandle_t solver_create(int* status) {
	cusolverDnHandle_t h = NULL;
	*status = (int)cusolverDnCreate(&h);
	return h;
}
static void solver_destroy(cusolverDnHandle_t h) { cusolverDnDestroy(h); }

static int solver_dsyevd_bufsize(cusolverDnHandle_t h, int n, const double* A, int lda,
                                 const double* W, int* lwork) {
	return (int)cusolverDnDsyevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR,
	                                        CUBLAS_FILL_MODE_LOWER, n, A, lda, W, lwork);
}
static int solver_dsyevd(cusolverDnHandle_t h, int n, double* A, int lda, double* W,
                         double* work, int lwork, int* devInfo) {
	return (int)cusolverDnDsyevd(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
	                             n, A, lda, W, work, lwork, devInfo);
}

static int dev_read_int(const int* p) {
	int v = -1;
	cudaMemcpy(&v, p, sizeof(int), cudaMemcpyDeviceToHost);
	return v;
}
static int dev_mem_info(size_t* freeB, size_t* totalB) {
	return (int)cudaMemGetInfo(freeB, totalB);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

const backendName = "cuda"

// ckCuda panics on a non-zero cudaError_t. ckBlas does the same for cublasStatus_t.
func ckCuda(st C.int, op string) {
	if st != 0 {
		panic(fmt.Sprintf("backend: cuda %s failed (cudaError_t %d)", op, int(st)))
	}
}

func ckBlas(st C.int, op string) {
	if st != 0 {
		panic(fmt.Sprintf("backend: cublas %s failed (cublasStatus_t %d)", op, int(st)))
	}
}

// devCount returns the number of visible CUDA devices, or 0 if the driver is absent
// or reports an error — letting a build with no usable GPU fall back to a host backend.
func devCount() int {
	n := int(C.dev_count())
	if n < 0 {
		return 0
	}
	return n
}

// devSet binds the calling thread's CUDA context to device dev. Must run on the
// backend's owning thread before blasCreate (see newGPUOn).
func devSet(dev int) { ckCuda(C.dev_set(C.int(dev)), "cudaSetDevice") }

func blasCreate() unsafe.Pointer {
	var st C.int
	h := C.blas_create(&st)
	ckBlas(st, "cublasCreate")
	if h == nil {
		panic("backend: cublasCreate returned a nil handle")
	}
	return unsafe.Pointer(h)
}

func devMalloc(n int) unsafe.Pointer {
	p := C.dev_malloc(C.size_t(n * elemSize))
	if p == nil && n > 0 {
		// Drain the sticky error so later calls do not report this one.
		st := C.dev_last_error()
		// cudaErrorMemoryAllocation (2) is the common case here — a sector's assembled
		// operator plus its resident Lanczos panels outgrew the GPU. Say so, and point at
		// the fix, rather than surfacing a bare error code and a device stack trace.
		hint := ""
		if int(st) == 2 {
			hint = "; the GPU is out of memory (operator + Lanczos panels too large) — " +
				"use a larger-memory GPU (e.g. ADCGO_DIP_GRES=gpu:H200:2) or run with -backend gonum"
		}
		panic(fmt.Sprintf("backend: cudaMalloc(%d bytes) returned NULL (cudaError_t %d)%s", n*elemSize, int(st), hint))
	}
	return p
}

func devFree(p unsafe.Pointer)        { C.dev_free(p) }
func devZero(p unsafe.Pointer, n int) { ckCuda(C.dev_zero(p, C.size_t(n*elemSize)), "cudaMemset") }

func devH2D(dst unsafe.Pointer, src []float64) {
	ckCuda(C.dev_h2d(dst, unsafe.Pointer(&src[0]), C.size_t(len(src)*elemSize)), "cudaMemcpy H2D")
}

func devD2H(dst []float64, src unsafe.Pointer) {
	ckCuda(C.dev_d2h(unsafe.Pointer(&dst[0]), src, C.size_t(len(dst)*elemSize)), "cudaMemcpy D2H")
}

func devD2D(dst, src unsafe.Pointer, n int) {
	ckCuda(C.dev_d2d(dst, src, C.size_t(n*elemSize)), "cudaMemcpy D2D")
}

// devCanPeer reports whether device dev may access device peer's memory over NVLink.
func devCanPeer(dev, peer int) bool { return int(C.dev_can_peer(C.int(dev), C.int(peer))) != 0 }

// devEnablePeer authorizes the calling thread's current device to read `peer`'s memory,
// returning the cudaError_t (0 = enabled, 704 = already enabled). Must run on the
// enabling device's owning thread.
func devEnablePeer(peer int) int { return int(C.dev_enable_peer(C.int(peer))) }

// devMemcpy2D copies a strided rectangle peer-to-peer (or intra-device). Pitches and
// width are byte counts, height is a row count; cudaMemcpyDefault resolves the direction
// from the pointers (UVA + enabled peer access).
func devMemcpy2D(dst unsafe.Pointer, dpitch int, src unsafe.Pointer, spitch, width, height int) {
	ckCuda(C.dev_memcpy2d(dst, C.size_t(dpitch), src, C.size_t(spitch), C.size_t(width), C.size_t(height)), "cudaMemcpy2D peer")
}

// devHostAlloc allocates `bytes` of page-locked host memory, or returns nil if the driver
// declines. Deliberately NOT a ckCuda panic: pinned memory is a limited system-wide resource,
// so exhaustion must degrade to ordinary pageable staging rather than kill a long solve. Every
// caller therefore has to handle nil.
func devHostAlloc(bytes int) unsafe.Pointer {
	var p unsafe.Pointer
	if int(C.dev_host_alloc(&p, C.size_t(bytes))) != 0 {
		return nil
	}
	return p
}

// devHostFree releases a devHostAlloc block. Safe on nil.
func devHostFree(p unsafe.Pointer) {
	if p != nil {
		ckCuda(C.dev_host_free(p), "cudaFreeHost")
	}
}

// devSync blocks until all queued device work completes, and reports any error the
// asynchronous kernels raised. Needed for honest benchmarking.
func devSync() { ckCuda(C.dev_sync(), "cudaDeviceSynchronize") }

// devH2DPtrs uploads a host array of device pointers, as cublasDgemmBatched requires.
func devH2DPtrs(dst unsafe.Pointer, src []unsafe.Pointer) {
	ckCuda(C.dev_h2d(dst, unsafe.Pointer(&src[0]), C.size_t(len(src)*ptrSize)), "cudaMemcpy H2D (ptrs)")
}

// devMemInfo reports free and total device memory in bytes.
func devMemInfo() (free, total uint64) {
	var f, t C.size_t
	ckCuda(C.dev_mem_info(&f, &t), "cudaMemGetInfo")
	return uint64(f), uint64(t)
}

// devReadInt reads a single int32 (cuSOLVER's devInfo) back from the device.
func devReadInt(p unsafe.Pointer) int { return int(C.dev_read_int((*C.int)(p))) }

// solverCreate makes the dense-solver handle. Created lazily on the device thread.
func solverCreate() unsafe.Pointer {
	var st C.int
	h := C.solver_create(&st)
	if st != 0 || h == nil {
		panic(fmt.Sprintf("backend: cusolverDnCreate failed (status %d)", int(st)))
	}
	return unsafe.Pointer(h)
}

func solverDestroy(h unsafe.Pointer) { C.solver_destroy(C.cusolverDnHandle_t(h)) }

// solverDsyevdBufferSize returns the workspace size in float64 elements.
func solverDsyevdBufferSize(h unsafe.Pointer, n int, a unsafe.Pointer, lda int, w unsafe.Pointer) int {
	var lwork C.int
	st := C.solver_dsyevd_bufsize(C.cusolverDnHandle_t(h), C.int(n), (*C.double)(a), C.int(lda),
		(*C.double)(w), &lwork)
	if st != 0 {
		panic(fmt.Sprintf("backend: cusolverDnDsyevd_bufferSize failed (status %d, n=%d)", int(st), n))
	}
	return int(lwork)
}

// solverDsyevd overwrites a with the eigenvectors (columns, column-major) and w with
// the ascending eigenvalues. Returns cuSOLVER's devInfo: 0 on success.
func solverDsyevd(h unsafe.Pointer, n int, a unsafe.Pointer, lda int, w, work unsafe.Pointer, lwork int, info unsafe.Pointer) int {
	st := C.solver_dsyevd(C.cusolverDnHandle_t(h), C.int(n), (*C.double)(a), C.int(lda),
		(*C.double)(w), (*C.double)(work), C.int(lwork), (*C.int)(info))
	if st != 0 {
		panic(fmt.Sprintf("backend: cusolverDnDsyevd failed (status %d, n=%d)", int(st), n))
	}
	return devReadInt(info)
}

func handle(h unsafe.Pointer) C.cublasHandle_t { return C.cublasHandle_t(h) }

func blasAxpy(h, x, y unsafe.Pointer, n int, a float64) {
	C.blas_axpy(handle(h), C.int(n), C.double(a), (*C.double)(x), (*C.double)(y))
}

func blasDot(h, x, y unsafe.Pointer, n int) float64 {
	return float64(C.blas_dot(handle(h), C.int(n), (*C.double)(x), (*C.double)(y)))
}

func blasNrm2(h, x unsafe.Pointer, n int) float64 {
	return float64(C.blas_nrm2(handle(h), C.int(n), (*C.double)(x)))
}

func blasScal(h, x unsafe.Pointer, n int, a float64) {
	C.blas_scal(handle(h), C.int(n), C.double(a), (*C.double)(x))
}

func blasGemv(h unsafe.Pointer, trans bool, m, n int, alpha float64, a unsafe.Pointer, lda int, x unsafe.Pointer, beta float64, y unsafe.Pointer) {
	var t C.int
	if trans {
		t = 1
	}
	C.blas_gemv(handle(h), t, C.int(m), C.int(n), C.double(alpha),
		(*C.double)(a), C.int(lda), (*C.double)(x), C.double(beta), (*C.double)(y))
}

func blasGemm(h unsafe.Pointer, transA, transB bool, m, n, k int, alpha float64,
	a unsafe.Pointer, lda int, b unsafe.Pointer, ldb int, beta float64, c unsafe.Pointer, ldc int) {
	var ta, tb C.int
	if transA {
		ta = 1
	}
	if transB {
		tb = 1
	}
	st := C.blas_gemm(handle(h), ta, tb, C.int(m), C.int(n), C.int(k), C.double(alpha),
		(*C.double)(a), C.int(lda), (*C.double)(b), C.int(ldb),
		C.double(beta), (*C.double)(c), C.int(ldc))
	if st != 0 {
		// cuBLAS reports INTERNAL_ERROR for faults raised by earlier async work, so
		// surface the underlying cudaError_t too.
		cudaErr := int(C.dev_last_error())
		panic(fmt.Sprintf("backend: cublasDgemm failed (cublasStatus_t %d, cudaError_t %d): transA=%v transB=%v m=%d n=%d k=%d lda=%d ldb=%d ldc=%d",
			int(st), cudaErr, transA, transB, m, n, k, lda, ldb, ldc))
	}
}

// blasGemmBatched: a, b, c are device arrays of device pointers, batch elements long.
func blasGemmBatched(h unsafe.Pointer, transA, transB bool, m, n, k int, alpha float64,
	a unsafe.Pointer, lda int, b unsafe.Pointer, ldb int, beta float64, c unsafe.Pointer, ldc, batch int) {
	var ta, tb C.int
	if transA {
		ta = 1
	}
	if transB {
		tb = 1
	}
	st := C.blas_gemm_batched(handle(h), ta, tb, C.int(m), C.int(n), C.int(k), C.double(alpha),
		(**C.double)(a), C.int(lda), (**C.double)(b), C.int(ldb),
		C.double(beta), (**C.double)(c), C.int(ldc), C.int(batch))
	if st != 0 {
		cudaErr := int(C.dev_last_error())
		panic(fmt.Sprintf("backend: cublasDgemmBatched failed (cublasStatus_t %d, cudaError_t %d): transA=%v m=%d n=%d k=%d lda=%d ldb=%d ldc=%d batch=%d",
			int(st), cudaErr, transA, m, n, k, lda, ldb, ldc, batch))
	}
}
