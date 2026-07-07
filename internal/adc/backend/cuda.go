//go:build cuda && !hip

// cuBLAS cgo shim for the shared GPU backend (gpu_device.go) — the NVIDIA twin of
// hip.go. Compiled here against the CUDA 13.x developer libraries; the actual GPU
// run happens on an NVIDIA system. Structurally identical to the hipBLAS shim
// (cuBLAS mirrors the hipBLAS API); only the headers, libraries, and enum/type
// names differ. The `!hip` tag keeps the two shims mutually exclusive in a build.
package backend

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcublas -lcudart
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>

static void* dev_malloc(size_t bytes)                  { void* p = NULL; cudaMalloc(&p, bytes); return p; }
static void  dev_free(void* p)                         { cudaFree(p); }
static void  dev_zero(void* p, size_t bytes)           { cudaMemset(p, 0, bytes); }
static void  dev_h2d(void* d, const void* s, size_t b) { cudaMemcpy(d, s, b, cudaMemcpyHostToDevice); }
static void  dev_d2h(void* d, const void* s, size_t b) { cudaMemcpy(d, s, b, cudaMemcpyDeviceToHost); }
static void  dev_d2d(void* d, const void* s, size_t b) { cudaMemcpy(d, s, b, cudaMemcpyDeviceToDevice); }

static cublasHandle_t blas_create(void) { cublasHandle_t h; cublasCreate(&h); return h; }

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
*/
import "C"

import "unsafe"

const backendName = "cuda"

func blasCreate() unsafe.Pointer { return unsafe.Pointer(C.blas_create()) }

func devMalloc(n int) unsafe.Pointer  { return C.dev_malloc(C.size_t(n * elemSize)) }
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
