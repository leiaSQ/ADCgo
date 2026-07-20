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
	"unsafe"

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

// BlockView is a mutable, column-major view onto a backend-resident Vector: column
// j occupies [j*Ld, j*Ld+Rows). It is the level-3 counterpart of Vector — a panel
// of vectors that GEMM can consume in one call.
//
// Column-major is load-bearing. It makes each column contiguous, so a column is a
// plain V.Slice(j*Ld, Rows) usable anywhere a Vector is (in particular as the
// argument to a mat-vec), and it is cuBLAS's native layout, so the GPU shim passes
// it straight through. Contrast Mat, which is row-major and immutable and stays the
// host-side assembly type for the operator blocks.
type BlockView struct {
	V          Vector // backing storage; at least Ld*(Cols-1)+Rows long
	Rows, Cols int
	Ld         int // leading dimension, >= Rows
}

// Col returns column j as a Vector sharing storage.
func (b BlockView) Col(j int) Vector { return b.V.Slice(j*b.Ld, b.Rows) }

// Cut returns the leading n columns of b, sharing storage.
func (b BlockView) Cut(n int) BlockView { b.Cols = n; return b }

// ColRange returns columns [lo, hi) of b, sharing storage.
func (b BlockView) ColRange(lo, hi int) BlockView {
	return BlockView{V: b.V.Slice(lo*b.Ld, (hi-lo-1)*b.Ld+b.Rows), Rows: b.Rows, Cols: hi - lo, Ld: b.Ld}
}

// RowRange returns rows [r0, r0+rows) of every column, sharing storage. Column j of
// the result begins at offset j*Ld of the sliced buffer, so the leading dimension is
// unchanged — this is how an operator block addresses its row band of a panel.
func (b BlockView) RowRange(r0, rows int) BlockView {
	return BlockView{V: b.V.Slice(r0, (b.Cols-1)*b.Ld+rows), Rows: rows, Cols: b.Cols, Ld: b.Ld}
}

// general reinterprets the column-major block as the row-major blas64.General of its
// transpose: identical memory, dims swapped, Stride = Ld. Callers compensate by
// swapping the operand order and transpose flags (see Gonum.Gemm).
func (b BlockView) general(data []float64) blas64.General {
	return blas64.General{Rows: b.Cols, Cols: b.Rows, Stride: b.Ld, Data: data}
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

// MatMul returns a·b for row-major host matrices. Used for the Ritz back-transform
// (main × dim times dim × dim), which is off the device — the main-space slice of the
// basis is tiny next to the basis itself, so bringing it home beats uploading the
// dim×dim eigenvector matrix. Threaded when built with the openblas tag.
func MatMul(a, b Mat) Mat {
	if a.Cols != b.Rows {
		panic("backend: MatMul shape mismatch")
	}
	c := NewMat(a.Rows, b.Cols)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, a.general(), b.general(), 0, c.general())
	return c
}

// Transpose returns mᵀ as a fresh host matrix. MatMul takes no transpose flags — the
// solver never needs them — so the one-electron AO→MO transforms (Cᵀ·A·C) form the
// transpose explicitly. The matrices involved are nAO-sized, not sector-sized.
func Transpose(m Mat) Mat {
	t := NewMat(m.Cols, m.Rows)
	for i := range m.Rows {
		for j := range m.Cols {
			t.Set(j, i, m.At(i, j))
		}
	}
	return t
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
	FreeMat(m DeviceMat)       // release an UploadMat allocation (no-op for host backends)

	// BLAS-1 on resident vectors (scalars are returned to the host).
	Axpy(alpha float64, x, y Vector) // y += alpha*x
	Dot(x, y Vector) float64         // xᵀy
	Nrm2(x Vector) float64           // ‖x‖₂
	Scal(alpha float64, x Vector)    // x *= alpha

	// AxpyDiag applies a resident diagonal operator block: y += d ⊙ x (elementwise
	// product accumulate), with d, x, y equal-length views. It lets a purely diagonal
	// block — e.g. the CVS-ADC(4) 3h2p/3h2p block, whose off-diagonal WERT3 coupling is
	// deferred — be stored and applied as an O(n) vector instead of a dense n×n matrix,
	// which is the difference between a few MB and terabytes for a large satellite space.
	AxpyDiag(d, x, y Vector)

	// BLAS-2 on resident vectors: y += alpha*a*x (GemvN) or y += alpha*aᵀ*x
	// (GemvT). x and y are typically Slice views onto the block offsets.
	GemvN(alpha float64, a DeviceMat, x, y Vector)
	GemvT(alpha float64, a DeviceMat, x, y Vector)

	// BLAS-3 on resident column-major panels: c := alpha*op(a)*op(b) + beta*c,
	// where op(x) is xᵀ when the corresponding trans flag is set. This is what
	// keeps the block-Lanczos reorthogonalization and the projected-matrix build
	// off the BLAS-1 path, where per-call overhead (host sync on a GPU, no
	// threading on a CPU) dominates the O(n) arithmetic.
	Gemm(transA, transB bool, alpha float64, a, b BlockView, beta float64, c BlockView)

	// GemmMat is the level-3 counterpart of GemvN/GemvT: c := alpha*op(a)*b + beta*c
	// for a resident, row-major, immutable operator block a (as returned by
	// UploadMat) against a column-major panel b. It lets the mat-vec apply the whole
	// operator to a block of vectors at once, streaming each block once per block
	// rather than once per vector.
	GemmMat(transA bool, alpha float64, a DeviceMat, b BlockView, beta float64, c BlockView)

	// GemmMatBatched applies GemmMat to a whole batch in one call. Every a[i] must have
	// the same dimensions, every b[i] the same Rows/Cols/Ld, every c[i] the same
	// Rows/Cols/Ld — and the c[i] must be pairwise non-overlapping, because a batched
	// GEMM runs its members concurrently and they accumulate (beta = 1).
	// PlanBatches builds batches satisfying exactly that contract. On a GPU this is one
	// kernel launch instead of len(a); on a host backend it is the obvious loop.
	GemmMatBatched(transA bool, alpha float64, a []DeviceMat, b []BlockView, beta float64, c []BlockView)

	// SymEig returns the ascending eigenvalues and eigenvectors (as columns of
	// the returned Mat) of the symmetric host matrix a. The lower triangle of a is
	// read. Used for the dense validation path and to diagonalize the (small)
	// projected matrix inside Lanczos — always small, hence host-side.
	SymEig(a Mat) (evals []float64, evecs Mat)
}

// HostData is an optional capability implemented by host-resident backends (Gonum):
// it exposes the raw backing slice of a resident Vector so a matrix-free operator
// block can read/write panel data in place, without Download/Upload copies. Device
// backends deliberately do not implement it (their vectors have no host copy) — a
// matrix-free block selects a device-kernel path instead (see internal/adc/sip
// matfree.go).
type HostData interface {
	HostSlice(v Vector) []float64
}

// PanelScatterAdd is the optional capability a row-partitioned backend (the distributed
// multi-device backend) implements so a matrix-free operator can add a full host output panel
// back into a partitioned resident panel — each device receives its own row band. Together
// with Download (which gathers the full input panel to host), it lets the DIP satellite region
// run matrix-free under -mgpu: the dense main/coupling blocks and the Krylov panels stay
// partitioned across the devices, and the satellite region — the multi-TB memory hog — is
// recomputed instead of materialized (dip/matfree_dist.go). Host and single-device backends do
// not implement it (they take the HostData / DeviceKernels satellite paths directly).
type PanelScatterAdd interface {
	// AddPanel adds the full n×cols column-major host panel into dst (dst += full), scattering
	// each device its row band. cols is inferred from len(full) and the backend's row count.
	AddPanel(dst Vector, full []float64)
}

// DeviceKernels is the optional capability a device backend implements to run a
// matrix-free operator block on-device: the element recompute (wert2elem4) is a custom
// CUDA kernel reading a device-resident ERI tensor, so the large ADC(4) 2h1p×3h2p
// coupling never occupies VRAM. Device counterpart of HostData. Implemented by the cuda
// backend (cuda_kernels.go, kernel in adc4_kernels.cu); the hip backend and Gonum do not
// implement it, so those take the dense / HostData paths. See docs/adc4_matfree_gpu.md.
type DeviceKernels interface {
	// SetCoeff1 uploads the flattened [3][13][30] spin table to constant memory (once).
	SetCoeff1(coeff1 []float64)
	// DeviceERI uploads a flat norb⁴ ERI tensor to the device; returns its pointer. The
	// caller frees it via FreeDev.
	DeviceERI(eri []float64) unsafe.Pointer
	// UploadInts uploads an int32 config-SoA array to the device; freed via FreeDev.
	UploadInts(x []int32) unsafe.Pointer
	// UploadFloats uploads a flat float64 array (e.g. orbital energies) to the device;
	// freed via FreeDev.
	UploadFloats(x []float64) unsafe.Pointer
	// FreeDev frees a DeviceERI/UploadInts/UploadFloats buffer.
	FreeDev(p unsafe.Pointer)
	// DevPtr is the device pointer backing a resident float64 Vector.
	DevPtr(v Vector) unsafe.Pointer
	// Wert2Apply launches the matrix-free 2h1p×3h2p coupling apply (forward + transpose),
	// accumulating into a.Out.
	Wert2Apply(a Wert2Args)
	// C22Apply launches the matrix-free order-3 2h1p×2h1p satellite apply (symmetric block,
	// a single pass — one thread per output row), accumulating into a.Out.
	C22Apply(a C22Args)
	// DipSatApply launches the matrix-free DIP 3h1p↔3h1p satellite apply (one thread per
	// output 3h1p row, symmetry honoured by block orientation — no transpose pass),
	// accumulating into a.Out. Kernel in adc2dip_kernels.cu; host twin in dip/satscalar.go.
	DipSatApply(a DipSatArgs)
}

// Wert2Args carries the device buffers and dimensions for DeviceKernels.Wert2Apply. The
// R* / C* pointers are UploadInts results (row/col config struct-of-arrays); ERI is a
// DeviceERI result; In/Out are resident panels (their device pointers via DevPtr).
type Wert2Args struct {
	N2, N3, B, LdIn, LdOut, MainOff, Off3, Norb, Nocc int
	RVir, RK, RL, RTyp                                unsafe.Pointer
	CI, CJ, CK, CL, CM, CSpin                         unsafe.Pointer
	ERI                                               unsafe.Pointer
	In, Out                                           Vector
}

// C22Args carries the device buffers and dimensions for DeviceKernels.C22Apply. K/L/Vir/Typ
// are UploadInts results (the 2h1p config struct-of-arrays, length N2); ERI is a DeviceERI
// result and Eps an UploadFloats result (the norb orbital energies); In/Out are resident
// panels (device pointers via DevPtr). The 2h1p region starts at MainOff within each panel.
type C22Args struct {
	N2, B, LdIn, LdOut, MainOff, Norb, Nocc int
	K, L, Vir, Typ                          unsafe.Pointer
	ERI, Eps                                unsafe.Pointer
	In, Out                                 Vector
}

// DipSatArgs carries the device buffers and dimensions for DeviceKernels.DipSatApply. The
// R* pointers are the per-row config struct-of-arrays (length Nsat); J*/I* are the JII/IJK
// per-group struct-of-arrays with JVir/IVir the flat concatenated absolute virtual orbitals;
// ERI is a DeviceERI result, Eps and OrbSym UploadInts/UploadFloats results. In/Out are
// resident panels (device pointers via DevPtr). The 3h1p region starts at MainOff. Spin is
// 0 (singlet) or 1 (triplet); Parts is 2 or 3.
type DipSatArgs struct {
	Nsat, Njii, Nijk, B, LdIn, LdOut, MainOff, Norb, Parts, Spin int
	RTyp, RGrp, RPart, RVir                                      unsafe.Pointer
	JO0, JO1, JSt, JVoff, JNv, JVir                              unsafe.Pointer
	IO0, IO1, IO2, ISt, IVoff, INv, IVir                         unsafe.Pointer
	ERI, Eps, OrbSym                                             unsafe.Pointer
	In, Out                                                      Vector
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
func (Gonum) FreeMat(DeviceMat)         {}

// HostSlice exposes the resident vector's backing slice (Gonum vectors are host
// slices, so this is zero-copy). Lets a matrix-free operator block read/write panels
// in place. See backend.HostData.
func (Gonum) HostSlice(v Vector) []float64 { return host(v) }

// Gemm computes c := alpha*op(a)*op(b) + beta*c on column-major panels.
//
// gonum's blas64 is row-major, and the row-major reading of a column-major panel is
// its transpose. Transposing the identity C = op(A)·op(B) gives
// Cᵀ = op(B)ᵀ·op(A)ᵀ, and op(X)ᵀ is exactly the row-major view of X carrying the
// same trans flag. So the row-major call is the operands swapped, flags following
// their operands — no flag inversion.
func (Gonum) Gemm(transA, transB bool, alpha float64, a, b BlockView, beta float64, c BlockView) {
	t := func(v bool) blas.Transpose {
		if v {
			return blas.Trans
		}
		return blas.NoTrans
	}
	blas64.Gemm(t(transB), t(transA), alpha,
		b.general(host(b.V)), a.general(host(a.V)), beta, c.general(host(c.V)))
}

func (Gonum) AxpyDiag(d, x, y Vector) {
	dd, xd, yd := host(d), host(x), host(y)
	for i, dv := range dd {
		yd[i] += dv * xd[i]
	}
}

func (Gonum) Axpy(alpha float64, x, y Vector) { blas64.Axpy(alpha, vector(host(x)), vector(host(y))) }
func (Gonum) Dot(x, y Vector) float64         { return blas64.Dot(vector(host(x)), vector(host(y))) }
func (Gonum) Nrm2(x Vector) float64           { return blas64.Nrm2(vector(host(x))) }
func (Gonum) Scal(alpha float64, x Vector)    { blas64.Scal(alpha, vector(host(x))) }

// GemmMat computes c := alpha*op(a)*b + beta*c with a row-major (a) and column-major
// (b, c). Transposing, Cᵀ = Bᵀ·op(A)ᵀ: rmB and rmC are the row-major readings of the
// column-major panels, and op(A)ᵀ is the row-major general of a carrying the negated
// trans flag.
func (Gonum) GemmMat(transA bool, alpha float64, a DeviceMat, b BlockView, beta float64, c BlockView) {
	ta := blas.Trans // op(A)ᵀ = Aᵀ when transA is false
	if transA {
		ta = blas.NoTrans
	}
	blas64.Gemm(blas.NoTrans, ta, alpha,
		b.general(host(b.V)), a.(hostMat).m.general(), beta, c.general(host(c.V)))
}

// GemmMatBatched on the host is the loop the batch exists to avoid on a device: the
// per-call overhead of a host BLAS call is nanoseconds, not microseconds.
func (g Gonum) GemmMatBatched(transA bool, alpha float64, a []DeviceMat, b []BlockView, beta float64, c []BlockView) {
	for i := range a {
		g.GemmMat(transA, alpha, a[i], b[i], beta, c[i])
	}
}

func (Gonum) GemvN(alpha float64, a DeviceMat, x, y Vector) {
	blas64.Gemv(blas.NoTrans, alpha, a.(hostMat).m.general(), vector(host(x)), 1, vector(host(y)))
}

func (Gonum) GemvT(alpha float64, a DeviceMat, x, y Vector) {
	blas64.Gemv(blas.Trans, alpha, a.(hostMat).m.general(), vector(host(x)), 1, vector(host(y)))
}

// SymEig dispatches to symEig, which build-tagged files may replace. The default
// (symEigGonum) is gonum's mat.EigenSym, i.e. LAPACK dsyev — QR iteration, and by
// far the slowest symmetric driver: measured at 2.4 GFLOP/s here, against 23.8
// GFLOP/s for the divide-and-conquer dsyevd on the same OpenBLAS. Since the
// projected Lanczos matrix reaches dim ~11600 for formic acid, where this is
// O(dim³), openblas.go swaps in LAPACKE_dsyevd.
func (Gonum) SymEig(a Mat) ([]float64, Mat) { return symEig(a) }

// symEig is the symmetric eigensolver behind Gonum.SymEig. Replaced in an init()
// by the build-tagged accelerated implementations.
var symEig = symEigGonum

// symEigGonum is the pure-Go reference: LAPACK dsyev via gonum's mat.EigenSym.
// Cross-implementation agreement is asserted against it.
func symEigGonum(a Mat) ([]float64, Mat) {
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
