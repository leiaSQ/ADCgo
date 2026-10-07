// bandeig_kernels.cu — device replay of the projected banded eigensolver's deferred eigenvector
// rotations (internal/adc/lanczos/bandeig.go, zTape). Linked into the cuda build by
// cuda_kernels.go; the host twin is zAccum.replay.
//
// WHY THIS IS ON A DEVICE AT ALL. Mode B's band→tridiagonal reduction plus QL sweep is the stage
// a whole-band DIP run disappears into for days. At the production shape (dim 308000,
// half-bandwidth 3079) the eigenvector rotations alone are ~10^14 two-cell updates over a 7.6 GB
// accumulator: pure streaming over independent rows, no reduction, no reuse beyond one column.
// That is a bandwidth problem, and it is the one thing here a GPU is unambiguously better at —
// HBM3e delivers ~4.8 TB/s against a CPU socket's ~200 GB/s. Threading it on the CPU was measured
// SLOWER than serial (see bandEigDefaultWorkers), which is what a bandwidth-bound kernel on a
// bandwidth-starved machine looks like.
//
// What is NOT here: the band matrix itself. Its reduction is a sequential chase that carries
// scalars from step to step and synchronizes ~9.5e8 times; a kernel launch is microseconds, so
// that side stays on the host. Nor is cuSOLVER used — neither cuSOLVER nor hipSOLVER has a banded
// or tridiagonal eigensolver, and their dense syevd returns all dim x dim eigenvectors where this
// algorithm needs only ~3080 rows of them.
//
// BIT-EXACTNESS. The host reference is Go on amd64, which does not contract a*b+c into an FMA.
// nvcc and hipcc do by default (-fmad=true), and a contracted product rounds once where Go rounds
// twice, so every arithmetic step below uses the explicit round-to-nearest intrinsics __dmul_rn /
// __dadd_rn / __dsub_rn. Do not "simplify" them back to operators, and do not rely on the build
// passing -fmad=false: the intrinsics make the guarantee local to this file. Operand order is
// copied from bnd2td.f / tddiag.f exactly as in the host path, because that is the other way a bit
// can move. TestGPUBandEigReplayParity compares device against host with !=.
//
// Replaying by rows is a loop interchange, not a reassociation: for a fixed row the ops run in
// their recorded order with their recorded operand order, and distinct rows never interact. One
// thread owns one row, so consecutive threads touch consecutive addresses of a column and the
// accesses coalesce.

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdint.h>

// Op kinds, in lockstep with the zOp* constants in internal/adc/lanczos/bandeig.go. The Go side
// asserts this mapping (TestZOpKindsMatchDevice) because a silent renumbering here would apply the
// wrong rotation and show up only as a wrong eigenvector row.
#define Z_OP_ROT_A 0
#define Z_OP_ROT_B 1
#define Z_OP_ROT_TD 2
#define Z_OP_SCALE 3

// bandeig_replay_z applies nops recorded ops to rows [0,rows) of the column-major accumulator z
// (leading dimension ld). col[o] is the second (higher) column of a rotation; its partner is
// col-1 for ROT_A/ROT_B and col+1 for ROT_TD. Columns are 1-based, as in the Fortran.
// applyOp applies one recorded op to row r. Shared by both replay kernels so the arithmetic — which
// is transcribed operand for operand from bnd2td.f / tddiag.f and uses round-to-nearest intrinsics to
// stop nvcc contracting an FMA — exists in exactly one place.
__device__ __forceinline__ void applyOp(unsigned char kind, long long j, double a1, double a2,
		int ld, int r, double* z) {
	double* pj = z + (j - 1) * (long long)ld + r;
	switch (kind) {
	case Z_OP_ROT_A: {
		double* pj1 = pj - (long long)ld;
		const double c1 = *pj1, c = *pj;
		const double u = __dadd_rn(c1, __dmul_rn(a2, c));
		*pj = __dadd_rn(__dmul_rn(-a1, c1), c);
		*pj1 = u;
		break;
	}
	case Z_OP_ROT_B: {
		double* pj1 = pj - (long long)ld;
		const double c1 = *pj1, c = *pj;
		const double u = __dadd_rn(__dmul_rn(a2, c1), c);
		*pj = __dadd_rn(-c1, __dmul_rn(a1, c));
		*pj1 = u;
		break;
	}
	case Z_OP_ROT_TD: {
		double* pi1 = pj + (long long)ld;
		const double ci = *pj, hh = *pi1;
		*pi1 = __dadd_rn(__dmul_rn(a2, ci), __dmul_rn(a1, hh));
		*pj = __dsub_rn(__dmul_rn(a1, ci), __dmul_rn(a2, hh));
		break;
	}
	case Z_OP_SCALE:
		*pj = __dmul_rn(a1, *pj);
		break;
	}
}

// bandeig_replay_z_par applies ops [lo,hi) CONCURRENTLY: one thread per (row, op).
//
// Legal only when the run's ops touch pairwise-disjoint columns, which the host asserts by setting
// segPar (see zTape in internal/adc/lanczos/bandeig.go, and TestZTapeSegmentsCommute). Under that
// precondition distinct threads write distinct addresses and the result is bit-identical to applying
// them one at a time — the arithmetic per element is untouched, only the schedule changes.
//
// THE PRECONDITION IS THE HOST'S RESPONSIBILITY AND HAS BEEN VIOLATED ONCE. bnd2td's periodic
// underflow rescale (bnd2td.f:124-145) records scale ops that are disjoint among themselves but share
// columns with the NEXT chase's rotations, and nothing between the two closed the run. This kernel then
// applied both at once and returned 1.5e+109 where the host had -1.4e-16. It is invisible at small
// shapes because the rescale needs dim > 64 and a narrow band to fire at all — 526 firings at
// {512,5}, none at {200,20}. Anything that adds a new recorded op inside bnd2td's k loop must bracket
// it with za.mark(), and the shape it fires at must be in TestZTapeSegmentsCommute.
//
// This is the parallelism the sequential kernel below could not use. At the production shape a run
// holds ~dim/band ops, so (rows x ops) is ~3080 x 50 threads instead of 3080 — the difference between
// ~96 warps on a 132-SM GPU and filling it.
//
// x indexes rows so that consecutive threads touch consecutive addresses within a column and the
// accesses coalesce; y indexes ops.
__global__ void bandeig_replay_z_par(int rows, int ld, int lo, int hi,
		const unsigned char* __restrict__ kind, const int* __restrict__ col,
		const double* __restrict__ f1, const double* __restrict__ f2,
		double* z) {
	const int o = lo + blockIdx.y * blockDim.y + threadIdx.y;
	if (o >= hi) {
		return;
	}
	const long long j = (long long)col[o];
	const double a1 = f1[o], a2 = f2[o];
	for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x) {
		applyOp(kind[o], j, a1, a2, ld, r, z);
	}
}

__global__ void bandeig_replay_z(int rows, int ld, int lo, int hi,
		const unsigned char* __restrict__ kind, const int* __restrict__ col,
		const double* __restrict__ f1, const double* __restrict__ f2,
		double* z) {
	for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x) {
		for (int o = lo; o < hi; ++o) {
			applyOp(kind[o], (long long)col[o], f1[o], f2[o], ld, r, z);
		}
	}
}

// bandeig_swap_z exchanges columns i and k of z, for the eigenvalue sort.
__global__ void bandeig_swap_z(int rows, int ld, int i, int k, double* z) {
	double* ci = z + (long long)(i - 1) * ld;
	double* ck = z + (long long)(k - 1) * ld;
	for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x) {
		const double t = ci[r];
		ci[r] = ck[r];
		ck[r] = t;
	}
}

extern "C" {

// bandeig_replay_z_launch launches one flush. The grid covers rows once; there is no more
// parallelism available than rows, because a row's ops are strictly ordered.
//
// THE BLOCK SIZE IS A REAL TUNING KNOB, not boilerplate, and the obvious value is the wrong one.
// rows is only ~3080 at the production shape, so threads=256 yields 12 blocks — on a 132-SM H200
// that is ~9% of the machine, and the other 120 SMs contribute nothing. Since the parallelism is
// capped at rows, the lever is not more threads but wider SPREAD: threads=32 gives 97 single-warp
// blocks on 97 SMs, multiplying the aggregate L1 and issue bandwidth. The working set of
// consecutive ops is two 24 kB columns, which is L2-resident, so this is issue-rate bound rather
// than HBM bound and the spread should matter a lot. 0 means "caller has no opinion".
int bandeig_replay_z_launch(int rows, int ld, int lo, int hi, int threads,
		const unsigned char* kind, const int* col, const double* f1, const double* f2, double* z) {
	if (rows <= 0 || hi <= lo) {
		return 0;
	}
	if (threads <= 0) {
		threads = 256;
	}
	if (threads < 32) {
		threads = 32;
	}
	if (threads > 1024) {
		threads = 1024;
	}
	threads = (threads / 32) * 32;
	int blocks = (rows + threads - 1) / threads;
	bandeig_replay_z<<<blocks, threads>>>(rows, ld, lo, hi, kind, col, f1, f2, z);
#ifdef __HIP_PLATFORM_AMD__
	return (int)hipGetLastError();
#else
	return (int)cudaGetLastError();
#endif
}

// bandeig_replay_z_par_launch launches the concurrent form over (rows x ops).
//
// The block is 32 wide in rows so a warp stays inside one column's contiguous run, and as tall in ops
// as the budget allows — that is where the extra parallelism comes from, and it costs nothing when a
// run holds few ops because the grid's y extent shrinks with it.
int bandeig_replay_z_par(int rows, int ld, int lo, int hi, int threads,
		const unsigned char* kind, const int* col, const double* f1, const double* f2, double* z) {
	if (rows <= 0 || hi <= lo) {
		return 0;
	}
	if (threads <= 0) {
		threads = 256;
	}
	if (threads < 32) {
		threads = 32;
	}
	if (threads > 1024) {
		threads = 1024;
	}
	const int tx = 32;
	int ty = threads / tx;
	if (ty < 1) {
		ty = 1;
	}
	const int nops = hi - lo;
	dim3 blk(tx, ty);
	dim3 grd((rows + tx - 1) / tx, (nops + ty - 1) / ty);
	bandeig_replay_z_par<<<grd, blk>>>(rows, ld, lo, hi, kind, col, f1, f2, z);
#ifdef __HIP_PLATFORM_AMD__
	return (int)hipGetLastError();
#else
	return (int)cudaGetLastError();
#endif
}

int bandeig_swap_z_launch(int rows, int ld, int i, int k, double* z) {
	if (rows <= 0 || i == k) {
		return 0;
	}
	const int threads = 256;
	int blocks = (rows + threads - 1) / threads;
	bandeig_swap_z<<<blocks, threads>>>(rows, ld, i, k, z);
#ifdef __HIP_PLATFORM_AMD__
	return (int)hipGetLastError();
#else
	return (int)cudaGetLastError();
#endif
}

} // extern "C"
