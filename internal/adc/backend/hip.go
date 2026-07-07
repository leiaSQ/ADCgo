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
#cgo LDFLAGS: -L/opt/rocm/lib -lhipblas -lamdhip64
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdlib.h>

// Memory helpers hide hip's pointer-to-pointer and enum signatures from cgo.
static void* dev_malloc(size_t bytes)                     { void* p = NULL; hipMalloc(&p, bytes); return p; }
static void  dev_free(void* p)                            { hipFree(p); }
static void  dev_zero(void* p, size_t bytes)              { hipMemset(p, 0, bytes); }
static void  dev_h2d(void* d, const void* s, size_t b)    { hipMemcpy(d, s, b, hipMemcpyHostToDevice); }
static void  dev_d2h(void* d, const void* s, size_t b)    { hipMemcpy(d, s, b, hipMemcpyDeviceToHost); }
static void  dev_d2d(void* d, const void* s, size_t b)    { hipMemcpy(d, s, b, hipMemcpyDeviceToDevice); }

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
*/
import "C"

import "unsafe"

const backendName = "hip"

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
