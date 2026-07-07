//go:build hip || cuda

// Shared device-resident backend orchestration for the GPU backends. The vector
// and matrix handle types, the Backend method bodies, and the row-major↔column-
// major GEMV convention live here once; each vendor's file (hip.go, cuda.go)
// supplies only the thin cgo shim — the handle constructor, the device memory
// ops, and the d-prefix BLAS calls — under a matching build tag. Because hipBLAS
// mirrors the cuBLAS API, the two shims are structural twins.
//
// SymEig is inherited from the embedded Gonum backend: it only ever sees the
// small projected (Lanczos) or validation matrices, so it stays on the host.
package backend

import "unsafe"

const elemSize = 8 // sizeof(float64)

func init() { Register(backendName, newGPU) }

// gpuBackend holds the vendor BLAS handle (as an opaque unsafe.Pointer; the shim
// casts it back to the concrete handle type).
type gpuBackend struct {
	Gonum
	h unsafe.Pointer
}

func newGPU() Backend { return &gpuBackend{h: blasCreate()} }

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
	p := devMalloc(n)
	devZero(p, n)
	return devVec{base: p, n: n}
}

func (b *gpuBackend) Upload(hostv Vec) Vector {
	p := devMalloc(len(hostv))
	if len(hostv) > 0 {
		devH2D(p, hostv)
	}
	return devVec{base: p, n: len(hostv)}
}

func (b *gpuBackend) Download(v Vector) Vec {
	dv := v.(devVec)
	out := make([]float64, dv.n)
	if dv.n > 0 {
		devD2H(out, dv.ptr())
	}
	return out
}

func (b *gpuBackend) Zero(v Vector) {
	dv := v.(devVec)
	devZero(dv.ptr(), dv.n)
}

func (b *gpuBackend) Copy(dst, src Vector) {
	d, s := dst.(devVec), src.(devVec)
	devD2D(d.ptr(), s.ptr(), d.n)
}

func (b *gpuBackend) Free(v Vector) {
	if dv := v.(devVec); dv.off == 0 && dv.base != nil {
		devFree(dv.base)
	}
}

func (b *gpuBackend) UploadMat(m Mat) DeviceMat {
	p := devMalloc(m.Rows * m.Cols)
	if len(m.Data) > 0 {
		devH2D(p, m.Data)
	}
	return devMat{p: p, rows: m.Rows, cols: m.Cols}
}

func (b *gpuBackend) Axpy(alpha float64, x, y Vector) {
	dx, dy := x.(devVec), y.(devVec)
	blasAxpy(b.h, dx.ptr(), dy.ptr(), dx.n, alpha)
}

func (b *gpuBackend) Dot(x, y Vector) float64 {
	dx, dy := x.(devVec), y.(devVec)
	return blasDot(b.h, dx.ptr(), dy.ptr(), dx.n)
}

func (b *gpuBackend) Nrm2(x Vector) float64 {
	dx := x.(devVec)
	return blasNrm2(b.h, dx.ptr(), dx.n)
}

func (b *gpuBackend) Scal(alpha float64, x Vector) {
	dx := x.(devVec)
	blasScal(b.h, dx.ptr(), dx.n, alpha)
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
	blasGemv(b.h, trans, m.cols, m.rows, alpha, m.p, m.cols, dx.ptr(), 1, dy.ptr())
}

var _ Backend = (*gpuBackend)(nil)
