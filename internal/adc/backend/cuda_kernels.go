//go:build cuda

// cuda_kernels.go — gpuBackend implementation of DeviceKernels, calling the custom CUDA
// kernels in adc4_kernels.cu (compiled by nvcc into adc4_kernels.o and linked here). This
// is the device path for the matrix-free 2h1p×3h2p coupling (internal/adc/sip matfree.go).
//
// The objects must be built before `go build -tags cuda`:
//
//go:generate nvcc -O3 -std=c++14 -c adc4_kernels.cu -o adc4_kernels.o
//go:generate nvcc -O3 -std=c++14 -c adc2dip_kernels.cu -o adc2dip_kernels.o
//
// They are build artifacts (git-ignored). See docs/adc4_matfree_gpu.md. Compiles and links
// against the CUDA toolkit here; they are exercised (parity tests) on an NVIDIA GPU.

package backend

/*
#cgo LDFLAGS: ${SRCDIR}/adc4_kernels.o ${SRCDIR}/adc2dip_kernels.o -L/usr/local/cuda/lib64 -lcudart -lstdc++
#include <cuda_runtime.h>
#include <stdlib.h>

// Launchers defined in adc4_kernels.cu (extern "C").
int adc4_set_coeff1(const double* h_coeff1);
int adc4_wert2_apply(int n2,int n3,int b,int ldIn,int ldOut,int mainOff,int off3,int norb,int nocc,
    const int* rVir,const int* rK,const int* rL,const int* rTyp,
    const int* cI,const int* cJ,const int* cK,const int* cL,const int* cM,const int* cSpin,
    const double* eri,const double* xin,double* yout);
int adc4_c22_apply(int n2,int b,int ldIn,int ldOut,int mainOff,int norb,int nocc,
    const int* K,const int* L,const int* Vir,const int* Typ,
    const double* eri,const double* eps,const double* xin,double* yout);

// Launcher defined in adc2dip_kernels.cu (extern "C").
int adc2_dip_fill_sat(int nslot,int spin,int norb,int parts,int maxElems,
    const int* kind,
    const int* rowO0,const int* rowO1,const int* rowO2,
    const int* colO0,const int* colO1,const int* colO2,
    const int* rowVOff,const int* rowNv,const int* colVOff,const int* colNv,
    const int* bufOff,const int* virs,
    const double* eri,const double* eps,const int* osym,double* buf);
int adc2_dip_sat_apply(int nsat,int njii,int nijk,int b,int ldIn,int ldOut,
    int mainOff,int norb,int parts,int spin,
    int rowLo,int rowHi,int outRowOff,
    const int* rTyp,const int* rGrp,const int* rPart,const int* rVir,
    const int* jO0,const int* jO1,const int* jSt,const int* jVoff,const int* jNv,const int* jVir,
    const int* iO0,const int* iO1,const int* iO2,const int* iSt,const int* iVoff,const int* iNv,const int* iVir,
    const double* eri,const double* eps,const int* osym,
    const double* xin,double* yout);

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

// UploadFloats uploads a flat float64 array (e.g. orbital energies); freed via FreeDev.
func (b *gpuBackend) UploadFloats(x []float64) unsafe.Pointer {
	var p unsafe.Pointer
	b.do(func() {
		p = C.k_malloc(C.size_t(len(x) * elemSize))
		C.k_h2d(p, unsafe.Pointer(&x[0]), C.size_t(len(x)*elemSize))
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

// DipSatFillJII materializes a batch of jiiLKK blocks into a persistent device scratch buffer and
// returns DeviceMat handles into it.
//
// The scratch is grown on demand and reused across applies, following the same precedent as the
// batched-GEMM pointer arrays (ensurePtrCap): this runs once per mat-vec, so a cudaMalloc/cudaFree
// pair per call would put allocator churn on the hot path. Handles are invalidated by the next
// call, which is why the caller consumes them immediately via GemmMatBatched.
func (b *gpuBackend) DipSatFillJII(a DipFillJIIArgs) []DeviceMat {
	if a.NSlot == 0 {
		return nil
	}
	b.do(func() {
		if b.jiiCap < a.TotalElems {
			if b.jiiBuf != nil {
				devFree(b.jiiBuf)
			}
			b.jiiBuf = devMalloc(a.TotalElems)
			b.jiiCap = a.TotalElems
		}
		C.adc2_dip_fill_sat(C.int(a.NSlot), C.int(a.Spin), C.int(a.Norb), C.int(a.Parts),
			C.int(a.MaxElems), (*C.int)(a.Kind),
			(*C.int)(a.RowO0), (*C.int)(a.RowO1), (*C.int)(a.RowO2),
			(*C.int)(a.ColO0), (*C.int)(a.ColO1), (*C.int)(a.ColO2),
			(*C.int)(a.RowVOff), (*C.int)(a.RowNv), (*C.int)(a.ColVOff), (*C.int)(a.ColNv),
			(*C.int)(a.BufOff), (*C.int)(a.Virs),
			(*C.double)(a.ERI), (*C.double)(a.Eps), (*C.int)(a.OrbSym),
			(*C.double)(b.jiiBuf))
	})

	// Shape the handles from the host-side dims; offsets are the same prefix sum the kernel used.
	out := make([]DeviceMat, a.NSlot)
	off := 0
	for i := range a.NSlot {
		out[i] = devMat{p: unsafe.Add(b.jiiBuf, off*elemSize), rows: a.Rows[i], cols: a.Cols[i]}
		off += a.Rows[i] * a.Cols[i]
	}
	return out
}

// DownloadMat copies a resident block back to the host (row-major). Verification only.
func (b *gpuBackend) DownloadMat(m DeviceMat) []float64 {
	dm := m.(devMat)
	out := make([]float64, dm.rows*dm.cols)
	if len(out) == 0 {
		return out
	}
	b.do(func() { devD2H(out, dm.p) })
	return out
}

// C22Apply launches the matrix-free order-3 2h1p×2h1p satellite apply (single symmetric pass)
// on the device-owning thread, accumulating into a.Out.
func (b *gpuBackend) C22Apply(a C22Args) {
	xin := (*C.double)(a.In.(devVec).ptr())
	yout := (*C.double)(a.Out.(devVec).ptr())
	b.do(func() {
		C.adc4_c22_apply(C.int(a.N2), C.int(a.B), C.int(a.LdIn), C.int(a.LdOut),
			C.int(a.MainOff), C.int(a.Norb), C.int(a.Nocc),
			(*C.int)(a.K), (*C.int)(a.L), (*C.int)(a.Vir), (*C.int)(a.Typ),
			(*C.double)(a.ERI), (*C.double)(a.Eps), xin, yout)
	})
}

// DipSatApply launches the matrix-free DIP 3h1p↔3h1p satellite apply (one thread per output
// 3h1p row) on the device-owning thread, accumulating into a.Out.
func (b *gpuBackend) DipSatApply(a DipSatArgs) {
	xin := (*C.double)(a.In.(devVec).ptr())
	yout := (*C.double)(a.Out.(devVec).ptr())
	b.do(func() {
		C.adc2_dip_sat_apply(C.int(a.Nsat), C.int(a.Njii), C.int(a.Nijk), C.int(a.B),
			C.int(a.LdIn), C.int(a.LdOut), C.int(a.MainOff), C.int(a.Norb), C.int(a.Parts), C.int(a.Spin),
			C.int(a.RowLo), C.int(a.RowHi), C.int(a.OutRowOff),
			(*C.int)(a.RTyp), (*C.int)(a.RGrp), (*C.int)(a.RPart), (*C.int)(a.RVir),
			(*C.int)(a.JO0), (*C.int)(a.JO1), (*C.int)(a.JSt), (*C.int)(a.JVoff), (*C.int)(a.JNv), (*C.int)(a.JVir),
			(*C.int)(a.IO0), (*C.int)(a.IO1), (*C.int)(a.IO2), (*C.int)(a.ISt), (*C.int)(a.IVoff), (*C.int)(a.INv), (*C.int)(a.IVir),
			(*C.double)(a.ERI), (*C.double)(a.Eps), (*C.int)(a.OrbSym), xin, yout)
	})
}
