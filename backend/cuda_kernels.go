//go:build cuda

// cuda_kernels.go — gpuBackend implementation of DeviceKernels, calling the custom CUDA
// kernels in adc4_kernels.cu (compiled by nvcc into adc4_kernels.o and linked here). This
// is the device path for the matrix-free 2h1p×3h2p coupling (internal/adc/sip matfree.go).
//
// The objects must be built before `go build -tags cuda`:
//
//go:generate nvcc -O3 -std=c++14 -c adc4_kernels.cu -o adc4_kernels.o
//go:generate nvcc -O3 -std=c++14 -c adc2dip_kernels.cu -o adc2dip_kernels.o
//go:generate nvcc -O3 -std=c++14 -c tensor_kernels.cu -o tensor_kernels.o
//
// They are build artifacts (git-ignored). See docs/adc4_matfree_gpu.md. Compiles and links
// against the CUDA toolkit here; they are exercised (parity tests) on an NVIDIA GPU.

package backend

/*
#cgo LDFLAGS: ${SRCDIR}/adc4_kernels.o ${SRCDIR}/adc2dip_kernels.o ${SRCDIR}/tensor_kernels.o -L/usr/local/cuda/lib64 -lcudart -lstdc++
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

// Launchers defined in tensor_kernels.cu (extern "C").
int tensor_einsum_launch(int nOuter, int nInner, int hasB,
    const long long* outerDim, const long long* innerDim,
    const long long* oO, const long long* oA, const long long* oB,
    const long long* iA, const long long* iB, double alpha,
    double* out, const double* a, const double* b);
int tensor_unpack_launch(long long n, int cols, long long tsize, long long ld,
    const long long* elem, const int* row, const signed char* sign, const double* y, double* t);
int tensor_pack_launch(long long n, int cols, long long tsize, long long ld,
    const long long* elem, const int* row, const signed char* sign, const double* t, double* out);

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
	b.do(func() { ckLaunch(C.adc4_set_coeff1((*C.double)(unsafe.Pointer(&coeff1[0]))), "adc4_set_coeff1") })
}

// DeviceERI uploads the flat norb⁴ ERI tensor; returned pointer is freed via FreeDev.
func (b *gpuBackend) DeviceERI(eri []float64) unsafe.Pointer {
	var p unsafe.Pointer
	b.do(func() {
		p = ckDevAlloc(C.k_malloc(C.size_t(len(eri)*elemSize)), len(eri)*elemSize, "DeviceERI")
		ckCuda(C.k_h2d(p, unsafe.Pointer(&eri[0]), C.size_t(len(eri)*elemSize)), "cudaMemcpy H2D (ERI)")
	})
	return p
}

// UploadInts uploads an int32 config-SoA array; freed via FreeDev.
func (b *gpuBackend) UploadInts(x []int32) unsafe.Pointer {
	var p unsafe.Pointer
	b.do(func() {
		p = ckDevAlloc(C.k_malloc(C.size_t(len(x)*4)), len(x)*4, "UploadInts")
		ckCuda(C.k_h2d(p, unsafe.Pointer(&x[0]), C.size_t(len(x)*4)), "cudaMemcpy H2D (ints)")
	})
	return p
}

// UploadFloats uploads a flat float64 array (e.g. orbital energies); freed via FreeDev.
func (b *gpuBackend) UploadFloats(x []float64) unsafe.Pointer {
	var p unsafe.Pointer
	b.do(func() {
		p = ckDevAlloc(C.k_malloc(C.size_t(len(x)*elemSize)), len(x)*elemSize, "UploadFloats")
		ckCuda(C.k_h2d(p, unsafe.Pointer(&x[0]), C.size_t(len(x)*elemSize)), "cudaMemcpy H2D (floats)")
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
		ckLaunch(C.adc4_wert2_apply(C.int(a.N2), C.int(a.N3), C.int(a.B), C.int(a.LdIn), C.int(a.LdOut),
			C.int(a.MainOff), C.int(a.Off3), C.int(a.Norb), C.int(a.Nocc),
			(*C.int)(a.RVir), (*C.int)(a.RK), (*C.int)(a.RL), (*C.int)(a.RTyp),
			(*C.int)(a.CI), (*C.int)(a.CJ), (*C.int)(a.CK), (*C.int)(a.CL), (*C.int)(a.CM), (*C.int)(a.CSpin),
			(*C.double)(a.ERI), xin, yout), "adc4_wert2_apply")
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
	lo, hi := a.SlotLo, a.SlotHi
	n := hi - lo
	if n <= 0 {
		return nil
	}
	// The kernel indexes every per-slot array by its y-block, so passing the arrays offset by lo
	// and launching hi-lo y-blocks materializes exactly the global range [lo,hi) — no kernel-side
	// range parameter needed. Virs/ERI/Eps/OrbSym are shared (indexed by content, not slot) and
	// are NOT offset. BufOff is chunk-local, so its offsets and this call's scratch both start at 0.
	off32 := func(p unsafe.Pointer) *C.int { return (*C.int)(unsafe.Add(p, lo*4)) } // int32 arrays
	b.do(func() {
		if b.jiiCap < a.ChunkElems {
			if b.jiiBuf != nil {
				devFree(b.jiiBuf)
			}
			b.jiiBuf = devMalloc(a.ChunkElems)
			b.jiiCap = a.ChunkElems
		}
		ckLaunch(C.adc2_dip_fill_sat(C.int(n), C.int(a.Spin), C.int(a.Norb), C.int(a.Parts),
			C.int(a.MaxElems), off32(a.Kind),
			off32(a.RowO0), off32(a.RowO1), off32(a.RowO2),
			off32(a.ColO0), off32(a.ColO1), off32(a.ColO2),
			off32(a.RowVOff), off32(a.RowNv), off32(a.ColVOff), off32(a.ColNv),
			off32(a.BufOff), (*C.int)(a.Virs),
			(*C.double)(a.ERI), (*C.double)(a.Eps), (*C.int)(a.OrbSym),
			(*C.double)(b.jiiBuf)), "adc2_dip_fill_sat")
	})

	// Handles for [lo,hi): handle k is global slot lo+k, placed at the same chunk-local prefix
	// the kernel wrote to (BufOff is that same prefix, reset to 0 at the chunk boundary).
	out := make([]DeviceMat, n)
	off := 0
	for k := range n {
		i := lo + k
		out[k] = devMat{p: unsafe.Add(b.jiiBuf, off*elemSize), rows: a.Rows[i], cols: a.Cols[i]}
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
		ckLaunch(C.adc4_c22_apply(C.int(a.N2), C.int(a.B), C.int(a.LdIn), C.int(a.LdOut),
			C.int(a.MainOff), C.int(a.Norb), C.int(a.Nocc),
			(*C.int)(a.K), (*C.int)(a.L), (*C.int)(a.Vir), (*C.int)(a.Typ),
			(*C.double)(a.ERI), (*C.double)(a.Eps), xin, yout), "adc4_c22_apply")
	})
}

// DipSatApply launches the matrix-free DIP 3h1p↔3h1p satellite apply (one thread per output
// 3h1p row) on the device-owning thread, accumulating into a.Out.
func (b *gpuBackend) DipSatApply(a DipSatArgs) {
	xin := (*C.double)(a.In.(devVec).ptr())
	yout := (*C.double)(a.Out.(devVec).ptr())
	b.do(func() {
		ckLaunch(C.adc2_dip_sat_apply(C.int(a.Nsat), C.int(a.Njii), C.int(a.Nijk), C.int(a.B),
			C.int(a.LdIn), C.int(a.LdOut), C.int(a.MainOff), C.int(a.Norb), C.int(a.Parts), C.int(a.Spin),
			C.int(a.RowLo), C.int(a.RowHi), C.int(a.OutRowOff),
			(*C.int)(a.RTyp), (*C.int)(a.RGrp), (*C.int)(a.RPart), (*C.int)(a.RVir),
			(*C.int)(a.JO0), (*C.int)(a.JO1), (*C.int)(a.JSt), (*C.int)(a.JVoff), (*C.int)(a.JNv), (*C.int)(a.JVir),
			(*C.int)(a.IO0), (*C.int)(a.IO1), (*C.int)(a.IO2), (*C.int)(a.ISt), (*C.int)(a.IVoff), (*C.int)(a.INv), (*C.int)(a.IVir),
			(*C.double)(a.ERI), (*C.double)(a.Eps), (*C.int)(a.OrbSym), xin, yout), "adc2_dip_sat_apply")
	})
}

// ---- backend.TensorKernels (tensor.go; kernels in tensor_kernels.cu) ----

// devTensorMap is a resident signed index map.
type devTensorMap struct {
	elem, row, sign unsafe.Pointer
	n               int
}

func (m *devTensorMap) Len() int { return m.n }

// TensorEinsum launches the label kernel: one thread per assignment of the labels out
// carries, the others summed in the thread.
func (b *gpuBackend) TensorEinsum(alpha float64, out, a TensorOperand, bo *TensorOperand) error {
	ops := []TensorOperand{out, a}
	if bo != nil {
		ops = append(ops, *bo)
	}
	p, err := PlanTensor(ops...)
	if err != nil {
		return err
	}
	var outerDim, innerDim, oO, oA, oB, iA, iB []C.longlong
	for _, l := range p.LoopOrder() {
		sb := 0
		if bo != nil {
			sb = p.Strides[2][l]
		}
		if p.Strides[0][l] != 0 {
			outerDim = append(outerDim, C.longlong(p.Dims[l]))
			oO = append(oO, C.longlong(p.Strides[0][l]))
			oA = append(oA, C.longlong(p.Strides[1][l]))
			oB = append(oB, C.longlong(sb))
		} else {
			innerDim = append(innerDim, C.longlong(p.Dims[l]))
			iA = append(iA, C.longlong(p.Strides[1][l]))
			iB = append(iB, C.longlong(sb))
		}
	}
	first := func(x []C.longlong) *C.longlong {
		if len(x) == 0 {
			return nil
		}
		return &x[0]
	}
	hasB := C.int(0)
	var bp unsafe.Pointer
	if bo != nil {
		hasB = 1
		bp = bo.V.(devVec).ptr()
	}
	b.do(func() {
		ckLaunch(C.tensor_einsum_launch(C.int(len(outerDim)), C.int(len(innerDim)), hasB,
			first(outerDim), first(innerDim), first(oO), first(oA), first(oB), first(iA), first(iB),
			C.double(alpha), (*C.double)(out.V.(devVec).ptr()), (*C.double)(a.V.(devVec).ptr()),
			(*C.double)(bp)), "tensor_einsum")
	})
	return nil
}

// UploadTensorMap makes a signed index map resident.
func (b *gpuBackend) UploadTensorMap(elem []int64, row []int32, sign []int8) TensorMap {
	m := &devTensorMap{n: len(elem)}
	if m.n == 0 {
		return m
	}
	up := func(src unsafe.Pointer, bytes int, what string) unsafe.Pointer {
		p := ckDevAlloc(C.k_malloc(C.size_t(bytes)), bytes, what)
		ckCuda(C.k_h2d(p, src, C.size_t(bytes)), "cudaMemcpy H2D ("+what+")")
		return p
	}
	b.do(func() {
		m.elem = up(unsafe.Pointer(&elem[0]), 8*m.n, "tensor map elem")
		m.row = up(unsafe.Pointer(&row[0]), 4*m.n, "tensor map row")
		m.sign = up(unsafe.Pointer(&sign[0]), m.n, "tensor map sign")
	})
	return m
}

// FreeTensorMap releases an UploadTensorMap allocation.
func (b *gpuBackend) FreeTensorMap(tm TensorMap) {
	m := tm.(*devTensorMap)
	if m.n == 0 {
		return
	}
	b.do(func() {
		devFree(m.elem)
		devFree(m.row)
		devFree(m.sign)
	})
	m.n = 0
}

// TensorUnpack scatters a packed panel into a spin-block tensor.
func (b *gpuBackend) TensorUnpack(t Vector, tsize int, y Vector, ld, cols int, tm TensorMap) {
	m := tm.(*devTensorMap)
	if m.n == 0 || cols == 0 {
		return
	}
	b.do(func() {
		ckLaunch(C.tensor_unpack_launch(C.longlong(m.n), C.int(cols), C.longlong(tsize), C.longlong(ld),
			(*C.longlong)(m.elem), (*C.int)(m.row), (*C.schar)(m.sign),
			(*C.double)(y.(devVec).ptr()), (*C.double)(t.(devVec).ptr())), "tensor_unpack")
	})
}

// TensorPack gathers a spin-block tensor back into a packed panel (accumulating).
func (b *gpuBackend) TensorPack(out Vector, ld, cols int, t Vector, tsize int, tm TensorMap) {
	m := tm.(*devTensorMap)
	if m.n == 0 || cols == 0 {
		return
	}
	b.do(func() {
		ckLaunch(C.tensor_pack_launch(C.longlong(m.n), C.int(cols), C.longlong(tsize), C.longlong(ld),
			(*C.longlong)(m.elem), (*C.int)(m.row), (*C.schar)(m.sign),
			(*C.double)(t.(devVec).ptr()), (*C.double)(out.(devVec).ptr())), "tensor_pack")
	})
}

// GemmStridedBatched runs nb same-shaped products at fixed strides in one cuBLAS call.
func (b *gpuBackend) GemmStridedBatched(transA, transB bool, m, n, k, nb int, a Vector, lda int,
	bv Vector, ldb int, c Vector) {
	if nb == 0 || m == 0 || n == 0 {
		return
	}
	b.do(func() {
		blasGemmStridedBatched(b.h, transA, transB, m, n, k, 1,
			a.(devVec).ptr(), lda, m*k, bv.(devVec).ptr(), ldb, k*n, 0, c.(devVec).ptr(), m, m*n, nb)
	})
}
