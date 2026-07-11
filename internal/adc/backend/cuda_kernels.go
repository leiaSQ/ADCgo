//go:build cuda

// cuda_kernels.go — gpuBackend implementation of DeviceKernels, calling the custom CUDA
// kernels in adc4_kernels.cu (compiled by nvcc into adc4_kernels.o and linked here). This
// is the device path for the matrix-free 2h1p×3h2p coupling (internal/adc/sip matfree.go).
//
// The object must be built before `go build -tags cuda`:
//
//go:generate nvcc -O3 -std=c++14 -c adc4_kernels.cu -o adc4_kernels.o
//
// It is a build artifact (git-ignored). See docs/adc4_matfree_gpu.md. Compiles and links
// against the CUDA toolkit here; it is exercised (parity test) on an NVIDIA GPU.

package backend

/*
#cgo LDFLAGS: ${SRCDIR}/adc4_kernels.o -L/usr/local/cuda/lib64 -lcudart -lstdc++
#include <cuda_runtime.h>
#include <stdlib.h>

// Launchers defined in adc4_kernels.cu (extern "C").
int adc4_set_coeff1(const double* h_coeff1);
int adc4_wert2_apply(int n2,int n3,int b,int ldIn,int ldOut,int mainOff,int off3,int norb,int nocc,
    const int* rVir,const int* rK,const int* rL,const int* rTyp,
    const int* cI,const int* cJ,const int* cK,const int* cL,const int* cM,const int* cSpin,
    const double* eri,const double* xin,double* yout);

// Byte-sized device allocation/copy helpers (this cgo file's own C context; the ones in
// cuda.go belong to a different translation unit and are not visible here).
static void* k_malloc(size_t bytes)            { void* p=NULL; cudaMalloc(&p,bytes); return p; }
static void  k_free(void* p)                   { cudaFree(p); }
static int   k_h2d(void* d,const void* s,size_t b){ return (int)cudaMemcpy(d,s,b,cudaMemcpyHostToDevice); }
*/
import "C"

import "unsafe"

// SetCoeff1 uploads the flattened [3][13][30] spin table to constant memory.
func (b *gpuBackend) SetCoeff1(coeff1 []float64) {
	b.do(func() { C.adc4_set_coeff1((*C.double)(unsafe.Pointer(&coeff1[0]))) })
}

// DeviceERI uploads the flat norb⁴ ERI tensor; returned pointer is freed via FreeDev.
func (b *gpuBackend) DeviceERI(eri []float64) unsafe.Pointer {
	var p unsafe.Pointer
	b.do(func() {
		p = C.k_malloc(C.size_t(len(eri) * elemSize))
		C.k_h2d(p, unsafe.Pointer(&eri[0]), C.size_t(len(eri)*elemSize))
	})
	return p
}

// UploadInts uploads an int32 config-SoA array; freed via FreeDev.
func (b *gpuBackend) UploadInts(x []int32) unsafe.Pointer {
	var p unsafe.Pointer
	b.do(func() {
		p = C.k_malloc(C.size_t(len(x) * 4))
		C.k_h2d(p, unsafe.Pointer(&x[0]), C.size_t(len(x)*4))
	})
	return p
}

// FreeDev frees a DeviceERI/UploadInts buffer.
func (b *gpuBackend) FreeDev(p unsafe.Pointer) { b.do(func() { C.k_free(p) }) }

// DevPtr is the device pointer backing a resident vector.
func (b *gpuBackend) DevPtr(v Vector) unsafe.Pointer { return v.(devVec).ptr() }

// Wert2Apply launches the matrix-free 2h1p×3h2p coupling apply (forward + transpose) on
// the device-owning thread, accumulating into a.Out.
func (b *gpuBackend) Wert2Apply(a Wert2Args) {
	xin := (*C.double)(a.In.(devVec).ptr())
	yout := (*C.double)(a.Out.(devVec).ptr())
	b.do(func() {
		C.adc4_wert2_apply(C.int(a.N2), C.int(a.N3), C.int(a.B), C.int(a.LdIn), C.int(a.LdOut),
			C.int(a.MainOff), C.int(a.Off3), C.int(a.Norb), C.int(a.Nocc),
			(*C.int)(a.RVir), (*C.int)(a.RK), (*C.int)(a.RL), (*C.int)(a.RTyp),
			(*C.int)(a.CI), (*C.int)(a.CJ), (*C.int)(a.CK), (*C.int)(a.CL), (*C.int)(a.CM), (*C.int)(a.CSpin),
			(*C.double)(a.ERI), xin, yout)
	})
}
