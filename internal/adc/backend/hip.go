//go:build hip

// hipBLAS cgo shim for the shared GPU backend (gpu_device.go). The DIP-ADC(2)
// mat-vec and Lanczos recurrence run on an AMD GPU via ROCm/hipBLAS with vectors
// and the assembled operator kept resident on the device. Built and tested here
// on a Radeon 890M (gfx1150); the deployment target is AMD Instinct (CDNA).
//
// gfx1150 may need HSA_OVERRIDE_GFX_VERSION=11.0.0 in the environment so rocBLAS
// selects a compatible kernel library.
package backend

/*
#cgo CFLAGS: -I/opt/rocm/include -D__HIP_PLATFORM_AMD__
#cgo LDFLAGS: -L/opt/rocm/lib -lhipblas -lhipsolver -lamdhip64
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <stdlib.h>

// Memory helpers hide hip's pointer-to-pointer and enum signatures from cgo.
static void* dev_malloc(size_t bytes)                     { void* p = NULL; hipMalloc(&p, bytes); return p; }
static int   dev_last_error(void)                         { return (int)hipGetLastError(); }
static void  dev_free(void* p)                            { hipFree(p); }
static void  dev_zero(void* p, size_t bytes)              { hipMemset(p, 0, bytes); }
static void  dev_h2d(void* d, const void* s, size_t b)    { hipMemcpy(d, s, b, hipMemcpyHostToDevice); }
static void  dev_d2h(void* d, const void* s, size_t b)    { hipMemcpy(d, s, b, hipMemcpyDeviceToHost); }
static void  dev_d2d(void* d, const void* s, size_t b)    { hipMemcpy(d, s, b, hipMemcpyDeviceToDevice); }

// Multi-GPU peer copy for the distributed backend (twin of cuda.go). dev_can_peer asks
// whether `dev` may read `peer`'s memory; dev_enable_peer authorizes the calling thread's
// current device to read `peer` (returns hipErrorPeerAccessAlreadyEnabled == 704 if a
// prior call already did, which the caller treats as success). dev_memcpy2d copies a
// strided (column-major, lda > rows) band in one call; hipMemcpyDefault infers the
// direction from the pointers under UVA + enabled peer access.
static int  dev_can_peer(int dev, int peer) { int ok = 0; hipDeviceCanAccessPeer(&ok, dev, peer); return ok; }
static int  dev_enable_peer(int peer)       { return (int)hipDeviceEnablePeerAccess(peer, 0); }
static int  dev_memcpy2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
	return (int)hipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyDefault);
}

static hipblasHandle_t blas_create(void) { hipblasHandle_t h; hipblasCreate(&h); return h; }

static void   blas_axpy(hipblasHandle_t h, int n, double a, const double* x, double* y) { hipblasDaxpy(h, n, &a, x, 1, y, 1); }
static double blas_dot (hipblasHandle_t h, int n, const double* x, const double* y)     { double r = 0; hipblasDdot(h, n, x, 1, y, 1, &r); return r; }
static double blas_nrm2(hipblasHandle_t h, int n, const double* x)                      { double r = 0; hipblasDnrm2(h, n, x, 1, &r); return r; }
static void   blas_scal(hipblasHandle_t h, int n, double a, double* x)                  { hipblasDscal(h, n, &a, x, 1); }

// y := alpha*op(A)*x + beta*y, op = trans?OP_T:OP_N, A column-major m×n, lda.
static void blas_gemv(hipblasHandle_t h, int trans, int m, int n, double alpha,
                      const double* A, int lda, const double* x, double beta, double* y) {
	hipblasOperation_t op = trans ? HIPBLAS_OP_T : HIPBLAS_OP_N;
	hipblasDgemv(h, op, m, n, &alpha, A, lda, x, 1, &beta, y, 1);
}

// C := alpha*op(A)*op(B) + beta*C, all column-major (hipBLAS's native layout, so
// BlockView passes straight through -- see the note on gpuBackend.Gemm).
static int blas_gemm(hipblasHandle_t h, int transA, int transB, int m, int n, int k,
                     double alpha, const double* A, int lda, const double* B, int ldb,
                     double beta, double* C, int ldc) {
	hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
	hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
	return (int)hipblasDgemm(h, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// Batched DGEMM: one launch for `batch` same-shaped products. A, B, C are DEVICE
// arrays of device pointers; members run concurrently, so the C pointers must not
// overlap (see backend.PlanBatches).
static int blas_gemm_batched(hipblasHandle_t h, int transA, int transB, int m, int n, int k,
                             double alpha, const double* const* A, int lda,
                             const double* const* B, int ldb,
                             double beta, double* const* C, int ldc, int batch) {
	hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
	hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
	return (int)hipblasDgemmBatched(h, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, batch);
}

// ---- hipSOLVER: divide-and-conquer symmetric eigensolver on the device ----
//
// hipSOLVER deliberately mirrors the cuSOLVER API (it dispatches to rocSOLVER on AMD),
// so this is a line-for-line twin of cuda.go's block. The input is fully symmetric, so
// its row-major and column-major readings coincide and uplo is immaterial; on exit A
// holds the eigenvectors as COLUMNS in column-major order, which the Go caller
// transposes back.
static hipsolverHandle_t solver_create(int* status) {
	hipsolverHandle_t h = NULL;
	*status = (int)hipsolverCreate(&h);
	return h;
}
static void solver_destroy(hipsolverHandle_t h) { hipsolverDestroy(h); }

static int solver_dsyevd_bufsize(hipsolverHandle_t h, int n, double* A, int lda,
                                 double* W, int* lwork) {
	return (int)hipsolverDsyevd_bufferSize(h, HIPSOLVER_EIG_MODE_VECTOR,
	                                       HIPSOLVER_FILL_MODE_LOWER, n, A, lda, W, lwork);
}
static int solver_dsyevd(hipsolverHandle_t h, int n, double* A, int lda, double* W,
                         double* work, int lwork, int* devInfo) {
	return (int)hipsolverDsyevd(h, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER,
	                            n, A, lda, W, work, lwork, devInfo);
}

static int dev_read_int(const int* p) {
	int v = -1;
	hipMemcpy(&v, p, sizeof(int), hipMemcpyDeviceToHost);
	return v;
}
static int dev_mem_info(size_t* freeB, size_t* totalB) {
	return (int)hipMemGetInfo(freeB, totalB);
}

// Multi-GPU: enumerate visible devices and bind the calling thread's context to one.
// Device is thread-current state, so a backend pinning one thread for life needs a
// single dev_set before its handle is created (see gpu_device.go / cuda.go's twins).
static int dev_count(void)  { int n = 0; hipGetDeviceCount(&n); return n; }
static int dev_set(int dev) { return (int)hipSetDevice(dev); }
*/
import "C"

import (
	"fmt"
	"unsafe"
)

const backendName = "hip"

// devCount returns the number of visible HIP devices, or 0 if none / on error.
func devCount() int {
	n := int(C.dev_count())
	if n < 0 {
		return 0
	}
	return n
}

// devSet binds the calling thread's HIP context to device dev. Must run on the
// backend's owning thread before blasCreate (see newGPUOn).
func devSet(dev int) { C.dev_set(C.int(dev)) }

func blasCreate() unsafe.Pointer { return unsafe.Pointer(C.blas_create()) }

func devMalloc(n int) unsafe.Pointer {
	p := C.dev_malloc(C.size_t(n * elemSize))
	if p == nil && n > 0 {
		st := C.dev_last_error() // drain the sticky error
		// hipErrorOutOfMemory (2) mirrors CUDA: a sector's assembled operator plus its
		// resident Lanczos panels outgrew the GPU. Point at the fix rather than letting a
		// NULL pointer fault later.
		hint := ""
		if int(st) == 2 {
			hint = "; the GPU is out of memory (operator + Lanczos panels too large) — " +
				"use a larger-memory GPU or run with -backend gonum"
		}
		panic(fmt.Sprintf("backend: hipMalloc(%d bytes) returned NULL (hipError_t %d)%s", n*elemSize, int(st), hint))
	}
	return p
}
func devFree(p unsafe.Pointer)        { C.dev_free(p) }
func devZero(p unsafe.Pointer, n int) { C.dev_zero(p, C.size_t(n*elemSize)) }

func devH2D(dst unsafe.Pointer, src []float64) {
	C.dev_h2d(dst, unsafe.Pointer(&src[0]), C.size_t(len(src)*elemSize))
}

func devD2H(dst []float64, src unsafe.Pointer) {
	C.dev_d2h(unsafe.Pointer(&dst[0]), src, C.size_t(len(dst)*elemSize))
}

func devD2D(dst, src unsafe.Pointer, n int) {
	C.dev_d2d(dst, src, C.size_t(n*elemSize))
}

// devCanPeer reports whether device dev may access device peer's memory.
func devCanPeer(dev, peer int) bool { return int(C.dev_can_peer(C.int(dev), C.int(peer))) != 0 }

// devEnablePeer authorizes the calling thread's current device to read `peer`'s memory,
// returning the hipError_t (0 = enabled, 704 = already enabled). Runs on the enabling
// device's owning thread.
func devEnablePeer(peer int) int { return int(C.dev_enable_peer(C.int(peer))) }

// devMemcpy2D copies a strided rectangle peer-to-peer. Pitches and width are byte counts,
// height a row count; hipMemcpyDefault resolves the direction from the pointers.
func devMemcpy2D(dst unsafe.Pointer, dpitch int, src unsafe.Pointer, spitch, width, height int) {
	C.dev_memcpy2d(dst, C.size_t(dpitch), src, C.size_t(spitch), C.size_t(width), C.size_t(height))
}

func handle(h unsafe.Pointer) C.hipblasHandle_t { return C.hipblasHandle_t(h) }

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
		panic(fmt.Sprintf("backend: hipblasDgemm failed (status %d): transA=%v transB=%v m=%d n=%d k=%d", int(st), transA, transB, m, n, k))
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
		panic(fmt.Sprintf("backend: hipblasDgemmBatched failed (status %d): transA=%v m=%d n=%d k=%d batch=%d", int(st), transA, m, n, k, batch))
	}
}

// devH2DPtrs uploads a host array of device pointers, as hipblasDgemmBatched requires.
func devH2DPtrs(dst unsafe.Pointer, src []unsafe.Pointer) {
	C.dev_h2d(dst, unsafe.Pointer(&src[0]), C.size_t(len(src)*ptrSize))
}

// devMemInfo reports free and total device memory in bytes.
func devMemInfo() (free, total uint64) {
	var f, t C.size_t
	C.dev_mem_info(&f, &t)
	return uint64(f), uint64(t)
}

// devReadInt reads a single int32 (hipSOLVER's devInfo) back from the device.
func devReadInt(p unsafe.Pointer) int { return int(C.dev_read_int((*C.int)(p))) }

// solverCreate makes the dense-solver handle. Created lazily on the device thread.
func solverCreate() unsafe.Pointer {
	var st C.int
	h := C.solver_create(&st)
	if st != 0 || h == nil {
		panic(fmt.Sprintf("backend: hipsolverCreate failed (status %d)", int(st)))
	}
	return unsafe.Pointer(h)
}

func solverDestroy(h unsafe.Pointer) { C.solver_destroy(C.hipsolverHandle_t(h)) }

// solverDsyevdBufferSize returns the workspace size in float64 elements.
func solverDsyevdBufferSize(h unsafe.Pointer, n int, a unsafe.Pointer, lda int, w unsafe.Pointer) int {
	var lwork C.int
	st := C.solver_dsyevd_bufsize(C.hipsolverHandle_t(h), C.int(n), (*C.double)(a), C.int(lda),
		(*C.double)(w), &lwork)
	if st != 0 {
		panic(fmt.Sprintf("backend: hipsolverDsyevd_bufferSize failed (status %d, n=%d)", int(st), n))
	}
	return int(lwork)
}

// solverDsyevd overwrites a with the eigenvectors (columns, column-major) and w with
// the ascending eigenvalues. Returns hipSOLVER's devInfo: 0 on success.
func solverDsyevd(h unsafe.Pointer, n int, a unsafe.Pointer, lda int, w, work unsafe.Pointer, lwork int, info unsafe.Pointer) int {
	st := C.solver_dsyevd(C.hipsolverHandle_t(h), C.int(n), (*C.double)(a), C.int(lda),
		(*C.double)(w), (*C.double)(work), C.int(lwork), (*C.int)(info))
	if st != 0 {
		panic(fmt.Sprintf("backend: hipsolverDsyevd failed (status %d, n=%d)", int(st), n))
	}
	return devReadInt(info)
}
