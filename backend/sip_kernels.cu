// sip_kernels.cu — matrix-free SIP-ADC(3) 2h1p×2h1p satellite block on NVIDIA GPUs.
//
// Device counterpart of internal/adc/sip/matfree.go newC22MatFreeO3: recomputes the
// satellite elements on the fly (reading a device-resident ERI tensor) so the
// ~n2²·8-byte block never occupies VRAM. Compiled with nvcc into an object linked by the
// cuda-tagged cgo shim (see cuda_kernels.go); the launcher is extern "C".
//
// Deterministic: one thread per OUTPUT row, summed in fixed index order — no atomics, no
// run-to-run reduction jitter.
//
// Build: nvcc -O3 -c sip_kernels.cu -o sip_kernels.o   (see cuda_kernels.go go:generate)

#include <cuda_runtime.h>

// d_eri: e.v(a,b,c,d) = ints.Eri(a,c,b,d) = TwoE(a,c,b,d) = eri[((a*n+c)*n+b)*n+d].
__device__ __forceinline__ double d_eri(const double *eri, int n, int a, int b, int c, int d) {
    return eri[(((long long)a * n + c) * n + b) * n + d];
}

// ============================================================================
// Order-3 SIP 2h1p×2h1p satellite block (matrix-free), the device counterpart of
// internal/adc/sip matfree.go newC22MatFreeO3 / matvec.go satBlock. The block is symmetric
// and on the operator diagonal, so it is applied in a single pass: one thread per output
// 2h1p row r accumulates out[main+r] += Σ_c S(r,c)·in[main+c]. The element is DIRECTIONAL —
// c22off(row,col) differs under argument swap — so element(r,c) is evaluated with the
// lower-indexed config as the row (lo=min(r,c)), matching the dense satBlock (which fills
// S(r,c)=S(c,r)=c22off(cfg_r,cfg_c) for r<c). d_c22diag/d_c22off are bit-for-bit transcriptions
// of sip/elements.go c22diag/c22off. Only ERIs + orbital energies are read (no spin table).
// ============================================================================

// Spin-coupling constants (sip/elements.go).
#define C22_SQRT1_2 0.70710678118654752440
#define C22_SQRT3_4 0.86602540378443864676

// d_c22diag: diagonal 2h1p element (k2 + c22_1_diag) of cfg (K=Occ0, L=Occ1, a=nocc+Vir).
__device__ double d_c22diag(int K, int L, int Vir, int Typ,
                            const double *eri, const double *eps, int norb, int nocc) {
    int a = nocc + Vir;
    double ek = eps[K], ea = eps[a];
#define V(p, q, r, s) d_eri(eri, norb, p, q, r, s)
    if (K == L) { // akk single
        double diag = ea - 2.0 * ek;
        double off = V(K, K, K, K)
                   - V(a, K, a, K) + 0.5 * V(a, K, K, a)
                   - V(a, K, a, K) + 0.5 * V(a, K, K, a);
        return diag + off;
    }
    double el = eps[L];
    double diag = ea - ek - el;
    double off;
    if (Typ == 0) { // spin I
        off = V(K, L, K, L) + V(K, L, L, K)
            - V(a, L, a, L) + 0.5 * V(a, L, L, a)
            - V(a, K, a, K) + 0.5 * V(a, K, K, a);
    } else { // spin II
        off = V(K, L, K, L) - V(K, L, L, K)
            - V(a, L, a, L) + 1.5 * V(a, L, L, a)
            - V(a, K, a, K) + 1.5 * V(a, K, K, a);
    }
#undef V
    return diag + off;
}

// d_deltaV mirrors sip/elements.go deltaV (the DELTA_V_TERM macro): when hole Kh == Mh, add
// the spin-block contributions with the given ± signs and prefactor pf.
__device__ __forceinline__ void d_deltaV(double &x00, double &x01, double &x10, double &x11,
                                         int Kh, int Mh, int A, int N, int B, int Lh,
                                         double s00, double s01, double s10, double s11, double pf,
                                         const double *eri, int norb) {
    if (Kh != Mh) return;
    double v1 = d_eri(eri, norb, A, N, B, Lh);  // V(A,N,B,L)
    double v2 = d_eri(eri, norb, A, N, Lh, B);  // V(A,N,L,B)
    x00 += s00 * pf * (-v1 + 0.5 * v2);
    x01 -= s01 * pf * (C22_SQRT3_4 * v2);
    x10 -= s10 * pf * (C22_SQRT3_4 * v2);
    x11 += s11 * pf * (-v1 + 1.5 * v2);
}

// d_c22off: off-diagonal 1st-order coupling between two distinct 2h1p configs. row=(K,L,a),
// col=(M,Nn,b). Caller passes the lower-indexed config as the row.
__device__ double d_c22off(int K, int L, int Vir, int Typ,
                           int M, int Nn, int VirC, int TypC,
                           const double *eri, int norb, int nocc) {
    int a = nocc + Vir, b = nocc + VirC;
    double pf = (M == Nn) ? C22_SQRT1_2 : 1.0;
    double x00 = 0.0, x01 = 0.0, x10 = 0.0, x11 = 0.0;
#define V(p, q, r, s) d_eri(eri, norb, p, q, r, s)
    if (K == L) { // akk row branch
        if (a == b) x00 += pf * 2.0 * V(M, Nn, K, K);
        d_deltaV(x00, x01, x10, x11, K, M, a, Nn, b, K, +1, +1, +1, +1, pf, eri, norb);
        d_deltaV(x00, x01, x10, x11, K, Nn, a, M, b, K, +1, -1, -1, +1, pf, eri, norb);
        d_deltaV(x00, x01, x10, x11, K, M, a, Nn, b, K, +1, -1, +1, -1, pf, eri, norb);
        d_deltaV(x00, x01, x10, x11, K, Nn, a, M, b, K, +1, +1, -1, -1, pf, eri, norb);
        x00 *= C22_SQRT1_2;
        x10 *= C22_SQRT1_2;
        // row single couples to col spin I via x00, spin II via x10.
        if (M == Nn || TypC == 0) return x00;
        return x10;
    }
    // akl row branch (k != l)
    if (a == b) {
        double vmnkl = V(M, Nn, K, L), vmnlk = V(M, Nn, L, K);
        x00 += pf * (vmnkl + vmnlk);
        x11 += pf * (vmnkl - vmnlk);
    }
    d_deltaV(x00, x01, x10, x11, K, M, a, Nn, b, L, +1, +1, +1, +1, pf, eri, norb);
    d_deltaV(x00, x01, x10, x11, L, Nn, a, M, b, K, +1, -1, -1, +1, pf, eri, norb);
    d_deltaV(x00, x01, x10, x11, L, M, a, Nn, b, K, +1, -1, +1, -1, pf, eri, norb);
    d_deltaV(x00, x01, x10, x11, K, Nn, a, M, b, L, +1, +1, -1, -1, pf, eri, norb);
#undef V
    if (M == Nn) { // col single: only spin-I column, rows I->x00, II->x01
        if (Typ == 0) return x00;
        return x01;
    }
    if (Typ == 0 && TypC == 0) return x00;
    if (Typ == 1 && TypC == 0) return x01;
    if (Typ == 0 && TypC == 1) return x10;
    return x11;
}

// c22_apply: one thread per 2h1p output row r. out[main+r] += Σ_c S(r,c)·in[main+c], with S
// the symmetric satellite block (diag = c22diag, off = c22off with lo=min(r,c) as the row).
// The element is computed once per (r,c) and applied across all b panel columns; the single
// owning thread makes the row's output race-free and reduction-jitter-free.
__global__ void c22_apply(int n2, int b, int ldIn, int ldOut, int mainOff, int norb, int nocc,
                          const int *__restrict__ K, const int *__restrict__ L,
                          const int *__restrict__ Vir, const int *__restrict__ Typ,
                          const double *__restrict__ eri, const double *__restrict__ eps,
                          const double *__restrict__ xin, double *__restrict__ yout) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n2) return;
    int Kr = K[r], Lr = L[r], Vr = Vir[r], Tr = Typ[r];
    for (int c = 0; c < n2; c++) {
        double g;
        if (c == r) {
            g = d_c22diag(Kr, Lr, Vr, Tr, eri, eps, norb, nocc);
        } else {
            // Necessary condition for a nonzero c22off: the two 2h1p configs must share the
            // PARTICLE or share a HOLE. Every term is gated that way — d_c22off's four deltaV
            // contributions each require one of k==m, l==n, l==m, k==n, and the only terms
            // outside them sit behind a==b.
            //
            // This is a pure early-out, not an approximation: the loop already discards
            // g == 0.0 below, so skipping pairs that provably evaluate to zero removes no
            // accumulation and changes no summation order. It is bit-identical by construction.
            //
            // It matters because this loop is O(n2²): at the production system's n2 = 518,056 that is 2.68e11
            // element evaluations PER MAT-VEC, and roughly 13 in 14 of those pairs are
            // structurally zero. The gate turns each of them into four integer compares instead
            // of a full d_c22off. The host applier prunes the same way, with buckets
            // (sip/matfree.go c22Buckets); a kernel cannot afford the per-thread dedup state a
            // bucket walk needs, so it tests the condition in place.
            int Kc = K[c], Lc = L[c], Vc = Vir[c];
            if (Vc != Vr && Kc != Kr && Kc != Lr && Lc != Kr && Lc != Lr) continue;
            if (r < c) {
                g = d_c22off(Kr, Lr, Vr, Tr, Kc, Lc, Vc, Typ[c], eri, norb, nocc);
            } else {
                g = d_c22off(Kc, Lc, Vc, Typ[c], Kr, Lr, Vr, Tr, eri, norb, nocc);
            }
        }
        if (g == 0.0) continue;
        for (int j = 0; j < b; j++) {
            yout[(mainOff + r) + j * ldOut] += g * xin[(mainOff + c) + j * ldIn];
        }
    }
}

// ---- extern "C" launcher (called from cuda_kernels.go via cgo) ----

extern "C" {

// sip_c22_apply runs the order-3 2h1p×2h1p satellite apply (single pass) on the default
// stream. All pointers are device pointers. Returns the first cudaError_t seen.
int sip_c22_apply(int n2, int b, int ldIn, int ldOut, int mainOff, int norb, int nocc,
                   const int *K, const int *L, const int *Vir, const int *Typ,
                   const double *eri, const double *eps, const double *xin, double *yout) {
    int T = 256;
    // Clear any error still pending from an earlier, unrelated CUDA call before launching, so
    // the status returned below describes THIS launch and nothing else. cudaGetLastError both
    // reads and RESETS the per-thread error state, and several statuses are deliberately
    // tolerated elsewhere without being cleared (EnablePeerAccess treats
    // cudaErrorPeerAccessAlreadyEnabled, 704, as benign) — without this reset the next launcher
    // to look would report that stale status as its own launch failure.
    cudaGetLastError();
    c22_apply<<<(n2 + T - 1) / T, T>>>(n2, b, ldIn, ldOut, mainOff, norb, nocc,
                                       K, L, Vir, Typ, eri, eps, xin, yout);
    return (int)cudaGetLastError();
}

} // extern "C"
