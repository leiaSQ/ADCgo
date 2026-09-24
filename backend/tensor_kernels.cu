// tensor_kernels.cu — device kernels of backend.TensorKernels (tensor.go), the σ-build of
// the generated ISR operators (internal/adc/isrgen/sigma). Linked into the cuda build by
// cuda_kernels.go; host twins are sigma.einsumHost and the sigma host pack kernels.
//
// tensor_einsum: out += alpha · Σ a·b over label assignments. The labels are split into
// the ones out carries ("outer") and the ones it does not ("inner", summed). One thread
// owns one outer assignment — distinct outer assignments address distinct out elements,
// so there are no atomics — and loops over the inner assignments itself. The outer labels
// arrive ordered by out stride, first fastest, so consecutive threads write consecutive
// out elements.

#include <cuda_runtime.h>
#include <stdint.h>

#define TMAX 16

struct TPlan {
	int nOuter, nInner, hasB;
	long long outerDim[TMAX], innerDim[TMAX];
	long long oO[TMAX], oA[TMAX], oB[TMAX]; // outer label strides in out, a, b
	long long iA[TMAX], iB[TMAX];           // inner label strides in a, b
	long long totalOuter, totalInner;
	double alpha;
};

__global__ void tensor_einsum(TPlan p, double* __restrict__ out,
		const double* __restrict__ a, const double* __restrict__ b) {
	for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < p.totalOuter;
			t += (long long)gridDim.x * blockDim.x) {
		long long r = t, off0 = 0, offA = 0, offB = 0;
		for (int l = 0; l < p.nOuter; ++l) {
			long long x = r % p.outerDim[l];
			r /= p.outerDim[l];
			off0 += x * p.oO[l];
			offA += x * p.oA[l];
			offB += x * p.oB[l];
		}
		double acc = 0;
		if (p.nInner == 0) {
			acc = p.hasB ? a[offA] * b[offB] : a[offA];
		} else {
			long long cnt[TMAX];
			for (int l = 0; l < p.nInner; ++l) cnt[l] = 0;
			long long ia = offA, ib = offB;
			for (long long s = 0; s < p.totalInner; ++s) {
				acc += p.hasB ? a[ia] * b[ib] : a[ia];
				for (int l = 0; l < p.nInner; ++l) {
					ia += p.iA[l];
					ib += p.iB[l];
					if (++cnt[l] < p.innerDim[l]) break;
					ia -= cnt[l] * p.iA[l];
					ib -= cnt[l] * p.iB[l];
					cnt[l] = 0;
				}
			}
		}
		out[off0] += p.alpha * acc;
	}
}

__global__ void tensor_unpack(long long n, int cols, long long tsize, long long ld,
		const long long* __restrict__ elem, const int* __restrict__ row, const signed char* __restrict__ sign,
		const double* __restrict__ y, double* __restrict__ t) {
	long long total = n * cols;
	for (long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x; k < total;
			k += (long long)gridDim.x * blockDim.x) {
		long long i = k % n, c = k / n;
		t[elem[i] + c * tsize] = (double)sign[i] * y[row[i] + c * ld];
	}
}

__global__ void tensor_pack(long long n, int cols, long long tsize, long long ld,
		const long long* __restrict__ elem, const int* __restrict__ row, const signed char* __restrict__ sign,
		const double* __restrict__ t, double* __restrict__ out) {
	long long total = n * cols;
	for (long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x; k < total;
			k += (long long)gridDim.x * blockDim.x) {
		long long i = k % n, c = k / n;
		out[row[i] + c * ld] += (double)sign[i] * t[elem[i] + c * tsize];
	}
}

static int grid_for(long long n) {
	long long g = (n + 255) / 256;
	if (g < 1) g = 1;
	if (g > 65535LL * 32) g = 65535LL * 32;
	return (int)g;
}

extern "C" {

// Launchers return cudaGetLastError(); the Go side checks every status (ckLaunch).
int tensor_einsum_launch(int nOuter, int nInner, int hasB,
		const long long* outerDim, const long long* innerDim,
		const long long* oO, const long long* oA, const long long* oB,
		const long long* iA, const long long* iB, double alpha,
		double* out, const double* a, const double* b) {
	if (nOuter > TMAX || nInner > TMAX) return (int)cudaErrorInvalidValue;
	TPlan p;
	p.nOuter = nOuter;
	p.nInner = nInner;
	p.hasB = hasB;
	p.totalOuter = 1;
	p.totalInner = 1;
	for (int l = 0; l < nOuter; ++l) {
		p.outerDim[l] = outerDim[l];
		p.oO[l] = oO[l];
		p.oA[l] = oA[l];
		p.oB[l] = oB[l];
		p.totalOuter *= outerDim[l];
	}
	for (int l = 0; l < nInner; ++l) {
		p.innerDim[l] = innerDim[l];
		p.iA[l] = iA[l];
		p.iB[l] = iB[l];
		p.totalInner *= innerDim[l];
	}
	p.alpha = alpha;
	if (p.totalOuter == 0 || p.totalInner == 0) return (int)cudaSuccess;
	tensor_einsum<<<grid_for(p.totalOuter), 256>>>(p, out, a, b);
	return (int)cudaGetLastError();
}

int tensor_unpack_launch(long long n, int cols, long long tsize, long long ld,
		const long long* elem, const int* row, const signed char* sign, const double* y, double* t) {
	if (n == 0 || cols == 0) return (int)cudaSuccess;
	tensor_unpack<<<grid_for(n * cols), 256>>>(n, cols, tsize, ld, elem, row, sign, y, t);
	return (int)cudaGetLastError();
}

int tensor_pack_launch(long long n, int cols, long long tsize, long long ld,
		const long long* elem, const int* row, const signed char* sign, const double* t, double* out) {
	if (n == 0 || cols == 0) return (int)cudaSuccess;
	tensor_pack<<<grid_for(n * cols), 256>>>(n, cols, tsize, ld, elem, row, sign, t, out);
	return (int)cudaGetLastError();
}

}
