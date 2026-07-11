// adc4_kernels.cu — matrix-free CVS-ADC(4) 2h1p×3h2p coupling (WERT2) on NVIDIA GPUs.
//
// Device counterpart of internal/adc/sip/matfree.go newWert2MatFree: recomputes the
// coupling elements on the fly (reading a device-resident ERI tensor) so the ~n2·n3·8-byte
// block never occupies VRAM. Compiled with nvcc into an object linked by the cuda-tagged
// cgo shim (see cuda_kernels.go); the launchers are extern "C".
//
// Deterministic two-pass design: one thread per OUTPUT element (a 2h1p row in the forward
// pass, a 3h2p column in the transpose pass), so each output is summed by a single thread
// in fixed index order — no atomics, no run-to-run reduction jitter. It evaluates the
// element twice (once per direction); the fused single-eval variant needs a slab reduce
// (docs/adc4_matfree_gpu.md) and is a follow-up.
//
// Build: nvcc -O3 -c adc4_kernels.cu -o adc4_kernels.o   (see cuda_kernels.go go:generate)

#include <cuda_runtime.h>

// coeff1 = SPINA/[3][13][30] spin table, flat (Typ*13 + col2)*30 + (n-1). ≤ 64 KB const.
__constant__ double c_coeff1[3 * 13 * 30];

// d_eri: e.v(a,b,c,d) = ints.Eri(a,c,b,d) = TwoE(a,c,b,d) = eri[((a*n+c)*n+b)*n+d].
__device__ __forceinline__ double d_eri(const double *eri, int n, int a, int b, int c, int d) {
    return eri[(((long long)a * n + c) * n + b) * n + d];
}

// d_ns3 mirrors sip.ns3 (elements4.go): the KOPP4/AB5 spin-table row offset for a 3h2p cfg.
__device__ __forceinline__ int d_ns3(int L, int M, int I, int J) {
    int lEqM = (L == M), iEqJ = (I == J);
    int maxs = (lEqM && iEqJ) ? 1 : ((lEqM || iEqJ) ? 2 : 5);
    if (lEqM && maxs == 2) return 7;
    if (iEqJ && maxs == 2) return 9;
    if (lEqM && maxs == 1) return 12;
    return 0;
}

// d_wert2 mirrors sip.wert2elem4 (elements4.go): the 2h1p×3h2p coupling element. Row =
// (Vir,K=Occ0,L=Occ1,Typ); col = (I,J,Kc,Lc,Mc,Spin). Returns 0 for non-coupling pairs
// (the branch gates are the δ-sparsity filter), so callers need no separate gate.
__device__ double d_wert2(int Vir, int K, int Locc, int Typ,
                          int I, int J, int Kc, int Lc, int Mc, int Spin,
                          const double *eri, int norb, int nocc) {
    int jp = nocc + Vir, kc = K, lv = Locc;
    int ii = nocc + I, jj = nocc + J, kk = Kc, ll = Lc, mm = Mc;
    double vint[31];
#pragma unroll
    for (int t = 0; t < 31; t++) vint[t] = 0.0;
#define V(a, b, c, d) d_eri(eri, norb, a, b, c, d)
    if (jp == ii) {
        if (kc == kk) { vint[1] = V(ll, mm, lv, jj); vint[2] = V(ll, mm, jj, lv); }
        if (lv == ll) {
            vint[9] = V(kk, mm, kc, jj); vint[10] = V(kk, mm, jj, kc);
            if (ll == mm) { vint[11] = vint[9]; vint[12] = vint[10]; }
        } else if (lv == mm) {
            vint[11] = V(kk, ll, kc, jj); vint[12] = V(kk, ll, jj, kc);
        }
    }
    if (jp == jj) {
        if (ii == jj) {
            for (int n = 1; n <= 12; n++) vint[12 + n] = vint[n];
        } else {
            if (kc == kk) { vint[13] = V(ll, mm, lv, ii); vint[14] = V(ll, mm, ii, lv); }
            if (lv == ll) {
                vint[21] = V(kk, mm, kc, ii); vint[22] = V(kk, mm, ii, kc);
                if (ll == mm) { vint[23] = vint[21]; vint[24] = vint[22]; }
            } else if (lv == mm) {
                vint[23] = V(kk, ll, kc, ii); vint[24] = V(kk, ll, ii, kc);
            }
        }
    }
    if (kc == kk && lv == ll) { vint[25] = V(mm, jp, ii, jj); vint[26] = V(mm, jp, jj, ii); }
    if (kc == kk && lv == mm) { vint[27] = V(ll, jp, ii, jj); vint[28] = V(ll, jp, jj, ii); }
#undef V
    int col2 = d_ns3(Lc, Mc, I, J) + Spin - 1;
    const double *sp = &c_coeff1[(Typ * 13 + col2) * 30];
    double add = 0.0;
    for (int n = 1; n <= 30; n++) {
        if (vint[n] != 0.0) add += vint[n] * sp[n - 1];
    }
    return -add;
}

// Config struct-of-arrays passed as device int arrays. Row (2h1p): rVir,rK,rL,rTyp.
// Col (3h2p): cI,cJ,cK,cL,cM,cSpin.
struct RowSoA { const int *Vir, *K, *L, *Typ; };
struct ColSoA { const int *I, *J, *K, *L, *M, *Spin; };

// The 2h1p and 3h2p regions are sub-ranges of the SAME panel, so one leading dimension
// applies to each panel: ldIn for the input, ldOut for the output.
//
// Forward pass: y2[main+r] += Σ_c wert2(r,c)·x3[off3+c], one thread per 2h1p row r.
__global__ void wert2_fwd(int n2, int n3, int b, int ldIn, int ldOut, int mainOff, int off3,
                          int norb, int nocc, RowSoA rw, ColSoA cl,
                          const double *xin, double *yout, const double *eri) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n2) return;
    int Vir = rw.Vir[r], K = rw.K[r], L = rw.L[r], Typ = rw.Typ[r];
    for (int j = 0; j < b; j++) {
        double acc = 0.0;
        for (int c = 0; c < n3; c++) {
            double g = d_wert2(Vir, K, L, Typ, cl.I[c], cl.J[c], cl.K[c], cl.L[c], cl.M[c],
                               cl.Spin[c], eri, norb, nocc);
            if (g != 0.0) acc += g * xin[(off3 + c) + j * ldIn];
        }
        yout[(mainOff + r) + j * ldOut] += acc;
    }
}

// Transpose pass: y3[off3+c] += Σ_r wert2(r,c)·x2[main+r], one thread per 3h2p column c.
__global__ void wert2_trans(int n2, int n3, int b, int ldIn, int ldOut, int mainOff, int off3,
                            int norb, int nocc, RowSoA rw, ColSoA cl,
                            const double *xin, double *yout, const double *eri) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n3) return;
    int I = cl.I[c], J = cl.J[c], K = cl.K[c], L = cl.L[c], M = cl.M[c], Spin = cl.Spin[c];
    for (int j = 0; j < b; j++) {
        double acc = 0.0;
        for (int r = 0; r < n2; r++) {
            double g = d_wert2(rw.Vir[r], rw.K[r], rw.L[r], rw.Typ[r], I, J, K, L, M, Spin,
                               eri, norb, nocc);
            if (g != 0.0) acc += g * xin[(mainOff + r) + j * ldIn];
        }
        yout[(off3 + c) + j * ldOut] += acc;
    }
}

// ---- extern "C" launchers (called from cuda_kernels.go via cgo) ----

extern "C" {

// adc4_set_coeff1 uploads the [3][13][30] spin table to constant memory (once per run).
int adc4_set_coeff1(const double *h_coeff1) {
    return (int)cudaMemcpyToSymbol(c_coeff1, h_coeff1, sizeof(double) * 3 * 13 * 30);
}

// adc4_wert2_apply runs both passes (forward + transpose) on the default stream. All
// pointers are device pointers. Returns the first cudaError_t seen.
int adc4_wert2_apply(int n2, int n3, int b, int ldIn, int ldOut, int mainOff, int off3,
                     int norb, int nocc,
                     const int *rVir, const int *rK, const int *rL, const int *rTyp,
                     const int *cI, const int *cJ, const int *cK, const int *cL,
                     const int *cM, const int *cSpin,
                     const double *eri, const double *xin, double *yout) {
    RowSoA rw = {rVir, rK, rL, rTyp};
    ColSoA cl = {cI, cJ, cK, cL, cM, cSpin};
    int T = 256;
    wert2_fwd<<<(n2 + T - 1) / T, T>>>(n2, n3, b, ldIn, ldOut, mainOff, off3, norb, nocc, rw, cl,
                                       xin, yout, eri);
    wert2_trans<<<(n3 + T - 1) / T, T>>>(n2, n3, b, ldIn, ldOut, mainOff, off3, norb, nocc, rw, cl,
                                         xin, yout, eri);
    return (int)cudaGetLastError();
}

} // extern "C"
