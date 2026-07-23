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
// Determinism note (2026-07-23). wert2_fwd/wert2_trans accumulate straight into yout per
// element rather than into a per-column register and adding once at the end. Both forms sum
// in the same fixed index order, so the run-to-run guarantee above is unchanged — but the
// association differs: y + (g0x0 + g1x1 + ...) became ((y + g0x0) + g1x1) + ... . Where yout
// is non-zero on entry (appliers accumulate in sequence) results may move by an ulp against
// the pre-2026-07-23 kernel. Parity tests run at 1e-10..1e-14; this is ~1e-16 relative.
// Keeping the register accumulator would have required holding b of them, and b is unbounded
// on the SIP path — that is the register-tiling follow-up, not something to do here.
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
                          const double *__restrict__ xin, double *__restrict__ yout,
                          const double *__restrict__ eri) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n2) return;
    int Vir = rw.Vir[r], K = rw.K[r], L = rw.L[r], Typ = rw.Typ[r];
    // Element-outer, column-inner — the same nest c22_apply uses below. g depends only on
    // (r,c), so the previous column-outer form re-evaluated d_wert2 b times per (r,c): a
    // 31-entry local array, up to ~16 conditional ERI loads and a 30-term dot product, all
    // repeated per panel column. SIP passes b unchunked (sip/matfree.go), so that was a full
    // b-fold multiplier on the dominant cost, not an amortized chunking tax.
    for (int c = 0; c < n3; c++) {
        double g = d_wert2(Vir, K, L, Typ, cl.I[c], cl.J[c], cl.K[c], cl.L[c], cl.M[c],
                           cl.Spin[c], eri, norb, nocc);
        if (g == 0.0) continue;
        for (int j = 0; j < b; j++) {
            yout[(mainOff + r) + j * ldOut] += g * xin[(off3 + c) + j * ldIn];
        }
    }
}

// Transpose pass: y3[off3+c] += Σ_r wert2(r,c)·x2[main+r], one thread per 3h2p column c.
__global__ void wert2_trans(int n2, int n3, int b, int ldIn, int ldOut, int mainOff, int off3,
                            int norb, int nocc, RowSoA rw, ColSoA cl,
                            const double *__restrict__ xin, double *__restrict__ yout,
                            const double *__restrict__ eri) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n3) return;
    int I = cl.I[c], J = cl.J[c], K = cl.K[c], L = cl.L[c], M = cl.M[c], Spin = cl.Spin[c];
    // Element-outer, column-inner — see wert2_fwd above for why.
    for (int r = 0; r < n2; r++) {
        double g = d_wert2(rw.Vir[r], rw.K[r], rw.L[r], rw.Typ[r], I, J, K, L, M, Spin,
                           eri, norb, nocc);
        if (g == 0.0) continue;
        for (int j = 0; j < b; j++) {
            yout[(off3 + c) + j * ldOut] += g * xin[(mainOff + r) + j * ldIn];
        }
    }
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
        } else if (r < c) {
            g = d_c22off(Kr, Lr, Vr, Tr, K[c], L[c], Vir[c], Typ[c], eri, norb, nocc);
        } else {
            g = d_c22off(K[c], L[c], Vir[c], Typ[c], Kr, Lr, Vr, Tr, eri, norb, nocc);
        }
        if (g == 0.0) continue;
        for (int j = 0; j < b; j++) {
            yout[(mainOff + r) + j * ldOut] += g * xin[(mainOff + c) + j * ldIn];
        }
    }
}

// ---- extern "C" launchers (called from cuda_kernels.go via cgo) ----

extern "C" {

// adc4_set_coeff1 uploads the [3][13][30] spin table to constant memory (once per run).
int adc4_set_coeff1(const double *h_coeff1) {
    return (int)cudaMemcpyToSymbol(c_coeff1, h_coeff1, sizeof(double) * 3 * 13 * 30);
}

// adc4_c22_apply runs the order-3 2h1p×2h1p satellite apply (single pass) on the default
// stream. All pointers are device pointers. Returns the first cudaError_t seen.
int adc4_c22_apply(int n2, int b, int ldIn, int ldOut, int mainOff, int norb, int nocc,
                   const int *K, const int *L, const int *Vir, const int *Typ,
                   const double *eri, const double *eps, const double *xin, double *yout) {
    int T = 256;
    c22_apply<<<(n2 + T - 1) / T, T>>>(n2, b, ldIn, ldOut, mainOff, norb, nocc,
                                       K, L, Vir, Typ, eri, eps, xin, yout);
    return (int)cudaGetLastError();
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
