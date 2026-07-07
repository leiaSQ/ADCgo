// Package backend is the linear-algebra abstraction the ADC engine is written
// against. The DIP-ADC(2) matrix-vector product and the Lanczos driver express
// themselves purely as calls into a Backend operating on backend-owned handles,
// so a GPU (hipBLAS/cuBLAS) implementation can keep vectors and the assembled
// operator resident on the device across Lanczos iterations without touching the
// solver (see ADCgo_plan.md, milestone M3).
//
// DIP-ADC(2) is a real-symmetric secular problem, so the interface is real
// (float64) throughout — the reference theADCcode uses only d-prefix BLAS. All
// GEMV/AXPY operations accumulate into the output (β = 1), matching the additive
// character of σ = M·x.
package backend

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

// Vec is a dense real host vector with unit stride. It is the host-side data
// interchange type (upload/download, small analysis GEMVs); the solver hotpath
// works on opaque Vector handles instead.
type Vec = []float64

// Vector is an opaque handle to a backend-resident vector. Host backends wrap a
// []float64; device backends wrap a device pointer. Slice returns a view sharing
// the same storage, so a GEMV can write into a sub-range of a resident output
// vector without copying (the block offsets of the DIP mat-vec).
type Vector interface {
	Len() int
	Slice(off, n int) Vector
}

// DeviceMat is an opaque handle to a backend-resident dense matrix, uploaded once
// (the immutable integral/operator blocks) and reused on every apply.
type DeviceMat interface {
	Dims() (rows, cols int)
}

// Mat is a dense real matrix stored row-major (Data[i*Cols+j]), matching the
// gonum blas64.General convention. It is the host-side assembly type for the
// operator blocks and the dense validation path; UploadMat turns one into a
// backend-resident DeviceMat. Column-major mapping for cuBLAS/hipBLAS is handled
// inside those backends; it does not leak into the solver.
type Mat struct {
	Rows, Cols int
	Data       []float64
}

// NewMat allocates an r×c zero matrix.
func NewMat(r, c int) Mat { return Mat{Rows: r, Cols: c, Data: make([]float64, r*c)} }

// At returns element (i,j).
func (m Mat) At(i, j int) float64 { return m.Data[i*m.Cols+j] }

// Set assigns element (i,j).
func (m Mat) Set(i, j int, v float64) { m.Data[i*m.Cols+j] = v }

func (m Mat) general() blas64.General {
	return blas64.General{Rows: m.Rows, Cols: m.Cols, Stride: m.Cols, Data: m.Data}
}

// MulVec returns m·x as a fresh host vector (no accumulation). Used by the small,
// host-resident analysis GEMVs (population U/O transforms), which are off the
// Lanczos hotpath and need no device backend.
func (m Mat) MulVec(x Vec) Vec {
	y := make([]float64, m.Rows)
	blas64.Gemv(blas.NoTrans, 1, m.general(), vector(x), 0, vector(y))
	return y
}

// AddSubMat accumulates alpha*src into the sub-block of m whose top-left corner
// is (r0,c0): m[r0+i, c0+j] += alpha*src[i,j]. Mirrors a BLAS daxpy of an
// integral block into a matrix sub-region.
func (m Mat) AddSubMat(r0, c0 int, alpha float64, src Mat) {
	for i := range src.Rows {
		base := (r0+i)*m.Cols + c0
		srow := i * src.Cols
		for j := range src.Cols {
			m.Data[base+j] += alpha * src.Data[srow+j]
		}
	}
}

// AddSubVec accumulates alpha*v as a column at rows [r0, r0+len(v)) in column c0:
// m[r0+i, c0] += alpha*v[i].
func (m Mat) AddSubVec(r0, c0 int, alpha float64, v []float64) {
	for i, x := range v {
		m.Data[(r0+i)*m.Cols+c0] += alpha * x
	}
}

// AddSubDiagConst adds alpha to the first n diagonal elements of the sub-block
// anchored at (r0,c0): m[r0+d, c0+d] += alpha for d in [0,n). The caller passes
// the sub-block's diagonal length explicitly — inferring it from the matrix edge
// (min(Rows-r0, Cols-c0)) is wrong for a sub-block that is not flush with the
// bottom-right corner, e.g. the top-left spin part of a spin-doubled block,
// where it would spill the constant onto the other spin parts' diagonals.
func (m Mat) AddSubDiagConst(r0, c0, n int, alpha float64) {
	for d := range n {
		m.Data[(r0+d)*m.Cols+(c0+d)] += alpha
	}
}

// AddSubDiagVec adds v[d] to the diagonal of the sub-block anchored at (r0,c0).
func (m Mat) AddSubDiagVec(r0, c0 int, v []float64) {
	for d, x := range v {
		m.Data[(r0+d)*m.Cols+(c0+d)] += x
	}
}

// Backend is the set of real linear-algebra kernels the solver may use. Vectors
// and matrices are backend-resident handles (Vector, DeviceMat); Upload/Download
// move data across the host boundary, which host backends make ~free.
type Backend interface {
	// Memory management for resident vectors.
	Alloc(n int) Vector        // resident zero vector of length n
	Upload(host Vec) Vector    // resident copy of a host vector
	Download(v Vector) Vec     // host copy of a resident vector
	Zero(v Vector)             // v[:] = 0
	Copy(dst, src Vector)      // dst[:] = src
	Free(v Vector)             // release (no-op for host backends)
	UploadMat(m Mat) DeviceMat // resident copy of an (immutable) block

	// BLAS-1 on resident vectors (scalars are returned to the host).
	Axpy(alpha float64, x, y Vector) // y += alpha*x
	Dot(x, y Vector) float64         // xᵀy
	Nrm2(x Vector) float64           // ‖x‖₂
	Scal(alpha float64, x Vector)    // x *= alpha

	// BLAS-2 on resident vectors: y += alpha*a*x (GemvN) or y += alpha*aᵀ*x
	// (GemvT). x and y are typically Slice views onto the block offsets.
	GemvN(alpha float64, a DeviceMat, x, y Vector)
	GemvT(alpha float64, a DeviceMat, x, y Vector)

	// SymEig returns the ascending eigenvalues and eigenvectors (as columns of
	// the returned Mat) of the symmetric host matrix a. The lower triangle of a is
	// read. Used for the dense validation path and to diagonalize the (small)
	// projected matrix inside Lanczos — always small, hence host-side.
	SymEig(a Mat) (evals []float64, evecs Mat)
}

// hostVec is the Gonum backend's Vector: a plain host slice. Slice shares storage.
type hostVec struct{ d []float64 }

func (v hostVec) Len() int                { return len(v.d) }
func (v hostVec) Slice(off, n int) Vector { return hostVec{d: v.d[off : off+n]} }

// hostMat is the Gonum backend's DeviceMat: the host Mat itself (blocks are
// immutable, so no copy is needed).
type hostMat struct{ m Mat }

func (h hostMat) Dims() (int, int) { return h.m.Rows, h.m.Cols }

// Gonum is the pure-Go backend, wrapping gonum's blas64 / mat. It is the
// deterministic reference implementation; cross-backend agreement is asserted
// against it. With the `openblas` build tag its BLAS/LAPACK engine is swapped for
// multicore OpenBLAS (see openblas.go); the type and code are otherwise identical.
type Gonum struct{}

func vector(x Vec) blas64.Vector { return blas64.Vector{N: len(x), Inc: 1, Data: x} }

func host(v Vector) []float64 { return v.(hostVec).d }

func (Gonum) Alloc(n int) Vector { return hostVec{d: make([]float64, n)} }

func (Gonum) Upload(h Vec) Vector {
	d := make([]float64, len(h))
	copy(d, h)
	return hostVec{d: d}
}

func (Gonum) Download(v Vector) Vec {
	s := host(v)
	out := make([]float64, len(s))
	copy(out, s)
	return out
}

func (Gonum) Zero(v Vector) {
	d := host(v)
	for i := range d {
		d[i] = 0
	}
}

func (Gonum) Copy(dst, src Vector) { copy(host(dst), host(src)) }
func (Gonum) Free(Vector)          {}

func (Gonum) UploadMat(m Mat) DeviceMat { return hostMat{m: m} }

func (Gonum) Axpy(alpha float64, x, y Vector) { blas64.Axpy(alpha, vector(host(x)), vector(host(y))) }
func (Gonum) Dot(x, y Vector) float64         { return blas64.Dot(vector(host(x)), vector(host(y))) }
func (Gonum) Nrm2(x Vector) float64           { return blas64.Nrm2(vector(host(x))) }
func (Gonum) Scal(alpha float64, x Vector)    { blas64.Scal(alpha, vector(host(x))) }

func (Gonum) GemvN(alpha float64, a DeviceMat, x, y Vector) {
	blas64.Gemv(blas.NoTrans, alpha, a.(hostMat).m.general(), vector(host(x)), 1, vector(host(y)))
}

func (Gonum) GemvT(alpha float64, a DeviceMat, x, y Vector) {
	blas64.Gemv(blas.Trans, alpha, a.(hostMat).m.general(), vector(host(x)), 1, vector(host(y)))
}

func (Gonum) SymEig(a Mat) ([]float64, Mat) {
	n := a.Rows
	sym := mat.NewSymDense(n, nil)
	for i := range n {
		for j := i; j < n; j++ {
			sym.SetSym(i, j, a.At(i, j))
		}
	}
	var es mat.EigenSym
	if ok := es.Factorize(sym, true); !ok {
		panic("backend: symmetric eigendecomposition failed to converge")
	}
	evals := es.Values(nil)
	var ev mat.Dense
	es.VectorsTo(&ev)
	out := NewMat(n, n)
	for i := range n {
		for j := range n {
			out.Set(i, j, ev.At(i, j))
		}
	}
	return evals, out
}

// Ensure Gonum satisfies Backend.
var _ Backend = Gonum{}
