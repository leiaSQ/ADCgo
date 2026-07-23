// adc2dip_kernels.cu — matrix-free 3h1p↔3h1p satellite apply for DIP-ADC(2) on NVIDIA GPUs.
//
// Device counterpart of internal/adc/dip satscalar.go (satScalarPlan.apply): one thread owns
// ONE output 3h1p configuration (group × spin part × virtual orbital) and accumulates its row
// of the satellite operator, Σ_C M[R,C]·in[C], recomputing every element from the device-
// resident ERI tensor so the ~TB satellite block never occupies VRAM (docs/dip_operator_memory.md).
//
// The device element functions d_jii_{s,t} / d_ijkMLL_{s,t} / d_ijkLMN_{s,t} are line-for-line
// transcriptions of internal/adc/dip/satelem.go (singlet/triplet …Elem), which is pinned to the
// dense blocks by TestSatelliteScalarMatchesDense; the orientation logic (which config is the
// block's row) transcribes satscalar.go elem() and is pinned by TestSatelliteScalarApplyEqualsDense.
// So the host tests fix the physics; the on-hardware parity test (dip/matfree_cuda_test.go)
// fixes only the C transcription.
//
// Like the SIP c22 kernel (adc4_kernels.cu), each thread scans all candidate column groups with a
// cheap shared-occupied-index early-out — the Kronecker-δ necessary condition — and skips a group
// with no shared hole. (Bucketing the candidates to O(G·k), as the host block applier does, is the
// scale follow-up; this kernel matches the SIP precedent's loop-the-columns shape first.)
//
// Build: nvcc -O3 -c adc2dip_kernels.cu -o adc2dip_kernels.o  (see cuda_kernels.go go:generate)

#include <cuda_runtime.h>

// dd_eri(a,b,c,d) = (a b | c d) = integrals.Store.Eri(a,b,c,d), stored plain row-major:
// eri[((a·n+b)·n+c)·n+d]. (The DIP host fill in dip/matfree.go must match this layout.)
__device__ __forceinline__ double dd_eri(const double *e, int n, int a, int b, int c, int d) {
    return e[(((long long)a * n + b) * n + c) * n + d];
}
// V1122_PLUS / V1122_MINUS (integrals.go EriPlus/EriMinus).
__device__ __forceinline__ double dd_vp(const double *e, int n, int p, int q, int r, int s) {
    return dd_eri(e, n, p, q, r, s) + dd_eri(e, n, p, s, r, q);
}
__device__ __forceinline__ double dd_vm(const double *e, int n, int p, int q, int r, int s) {
    return dd_eri(e, n, p, q, r, s) - dd_eri(e, n, p, s, r, q);
}

// Spin-coupling constants (blocks.go).
#define DD_SQRT2 1.41421356237309504880
#define DD_S1_2 0.70710678118654752440  // sqrt(1/2)
#define DD_SQRT3 1.73205080756887729353
#define DD_S3_2 1.22474487139158904909  // sqrt(3/2)
#define DD_S3_4 0.86602540378443864676  // sqrt(3/4)
#define DD_1_5 1.5

// A(x,y)[ra,sb] = (ra x | sb y); B(x,y)[ra,sb] = (ra sb | x y).
#define AA(x, y) dd_eri(eri, n, ra, x, sb, y)
#define BB(x, y) dd_eri(eri, n, ra, sb, x, y)

// ------------------------------- singlet ------------------------------------

__device__ double d_jii_s(int j, int i, int l, int k, int ra, int sb,
                          const double *eri, const double *eps, const int *osym, int n) {
    int dIL = (i == l), dJL = (j == l), dIK = (i == k), dJK = (j == k), dSym = (osym[ra] == osym[sb]);
    double e = 0.0;
    if (dIK) e += 2 * AA(j, l) - BB(j, l);
    if (dIK && dJL) e += AA(i, i) - 2 * BB(i, i);
    if (dIL && dJK) e += AA(l, j) + BB(j, l);
    if (dSym && ra == sb) {
        double diag = 0.0;
        if (dIK) diag += 2 * dd_eri(eri, n, j, l, i, i) - dd_eri(eri, n, i, j, i, l);
        if (dIL) diag -= dd_eri(eri, n, j, k, l, k);
        if (dJK) diag -= dd_eri(eri, n, i, j, i, l);
        if (dJL) diag += dd_eri(eri, n, i, k, i, k);
        if (dJL && dIK) { diag -= eps[j] + eps[i] + eps[i]; diag += eps[ra]; }
        e += diag;
    }
    return e;
}

// pr picks the row spin part (0/1); the column side is a single part.
__device__ double d_ijkMLL_s(int i, int j, int k, int m, int l, int pr, int ra, int sb,
                             const double *eri, const int *osym, int n) {
    int dIM = (i == m), dJM = (j == m), dKM = (k == m);
    int dIL = (i == l), dJL = (j == l), dKL = (k == l), dSym = (osym[ra] == osym[sb]);
    double e = 0.0;
#define C(c0, c1) (pr == 0 ? (c0) : (c1))
    if (dIM && dJL) e += C(DD_S1_2, -DD_S3_2) * AA(k, j) + C(DD_S1_2, DD_S3_2) * BB(j, k);
    if (dIM && dKL) e += C(-DD_SQRT2, 0) * AA(j, k) + C(DD_S1_2, DD_S3_2) * BB(j, k);
    if (dJM && dIL) e += C(DD_S1_2, -DD_S3_2) * AA(k, i) + C(-DD_SQRT2, 0) * BB(i, k);
    if (dJM && dKL) e += C(DD_S1_2, DD_S3_2) * AA(i, k) + C(-DD_SQRT2, 0) * BB(i, k);
    if (dKM && dIL) e += C(-DD_SQRT2, 0) * AA(j, i) + C(DD_S1_2, -DD_S3_2) * BB(i, j);
    if (dKM && dJL) e += C(DD_S1_2, DD_S3_2) * AA(i, j) + C(DD_S1_2, -DD_S3_2) * BB(i, j);
    if (dSym && ra == sb) {
        double d0 = 0.0, d1 = 0.0;
        if (dIM) { d0 -= dd_eri(eri, n, j, l, k, l); d1 -= dd_eri(eri, n, j, l, k, l); }
        if (dIL) { d0 += 2 * dd_eri(eri, n, i, k, j, m) - dd_eri(eri, n, i, j, k, m); d1 += dd_eri(eri, n, i, j, k, m); }
        if (dJM) d0 += 2 * dd_eri(eri, n, i, l, k, l);
        if (dJL) { d0 -= dd_vp(eri, n, i, j, k, m); d1 += dd_vm(eri, n, i, j, k, m); }
        if (dKM) { d0 -= dd_eri(eri, n, i, l, j, l); d1 += dd_eri(eri, n, i, l, j, l); }
        if (dKL) { d0 += 2 * dd_eri(eri, n, i, k, j, m) - dd_eri(eri, n, i, m, j, k); d1 -= dd_eri(eri, n, i, m, j, k); }
        e += (pr == 0) ? d0 * DD_S1_2 : d1 * DD_S3_2;
    }
#undef C
    return e;
}

__device__ double d_ijkLMN_s(int i, int j, int k, int l, int m, int nn, int pr, int pc, int ra, int sb,
                             const double *eri, const double *eps, const int *osym, int n) {
    int dIL = (i == l), dJL = (j == l), dKL = (k == l);
    int dJM = (j == m), dKM = (k == m), dJN = (j == nn), dKN = (k == nn), dSym = (osym[ra] == osym[sb]);
    double e = 0.0;
#define C(c00, c01, c10, c11) ((pr * 2 + pc) == 0 ? (c00) : (pr * 2 + pc) == 1 ? (c01) : (pr * 2 + pc) == 2 ? (c10) : (c11))
    if (dIL && dJM) e += C(0.5, -DD_S3_4, -DD_S3_4, DD_1_5) * AA(k, nn) + C(-1, 0, 0, -1) * BB(k, nn);
    if (dIL && dKM) e += C(-1, DD_SQRT3, 0, 0) * AA(j, nn) + C(0.5, -DD_S3_4, -DD_S3_4, -0.5) * BB(j, nn);
    if (dIL && dKN) e += C(2, 0, 0, 0) * AA(j, m) + C(-1, 0, 0, -1) * BB(j, m);
    if (dJL && dKM) e += C(0.5, -DD_S3_4, DD_S3_4, -DD_1_5) * AA(i, nn) + C(0.5, DD_S3_4, -DD_S3_4, 0.5) * BB(i, nn);
    if (dJL && dKN) e += C(-1, 0, -DD_SQRT3, 0) * AA(i, m) + C(0.5, DD_S3_4, DD_S3_4, -0.5) * BB(i, m);
    if (dJM && dKN) e += C(0.5, DD_S3_4, DD_S3_4, DD_1_5) * AA(i, l) + C(-1, 0, 0, -1) * BB(i, l);
    if (dSym && ra == sb) {
        double d00 = 0, d01 = 0, d10 = 0, d11 = 0;
        if (dIL) {
            d00 += dd_eri(eri, n, j, m, k, nn) - 0.5 * dd_eri(eri, n, j, nn, k, m);
            d01 += dd_eri(eri, n, j, nn, k, m); d10 += dd_eri(eri, n, j, nn, k, m);
            d11 += dd_eri(eri, n, j, m, k, nn) + 0.5 * dd_eri(eri, n, j, nn, k, m);
        }
        if (dJL) {
            d00 -= 0.5 * dd_vp(eri, n, i, m, k, nn); d01 -= dd_vp(eri, n, i, m, k, nn);
            d10 += dd_vm(eri, n, i, nn, k, m); d11 += 0.5 * dd_vm(eri, n, i, m, k, nn);
        }
        if (dJM) { d00 += dd_vp(eri, n, i, l, k, nn); d11 += dd_vm(eri, n, i, l, k, nn); }
        if (dJN) {
            d00 -= 0.5 * dd_vp(eri, n, i, l, k, m); d01 += dd_vp(eri, n, i, l, k, m);
            d10 += dd_vm(eri, n, i, l, k, m); d11 += 0.5 * dd_vm(eri, n, i, l, k, m);
        }
        if (dKL) {
            d00 += dd_eri(eri, n, i, nn, j, m) - 0.5 * dd_eri(eri, n, i, m, j, nn);
            d01 += dd_eri(eri, n, i, m, j, nn); d10 -= dd_eri(eri, n, i, m, j, nn);
            d11 -= dd_eri(eri, n, i, nn, j, m) + 0.5 * dd_eri(eri, n, i, m, j, nn);
        }
        if (dKM) {
            d00 -= 0.5 * dd_vp(eri, n, i, l, j, nn); d01 += dd_vm(eri, n, i, l, j, nn);
            d10 += dd_vp(eri, n, i, l, j, nn); d11 += 0.5 * dd_vm(eri, n, i, l, j, nn);
        }
        if (dKN) {
            d00 += dd_eri(eri, n, i, l, j, m) - 0.5 * dd_eri(eri, n, i, m, j, l);
            d01 -= dd_eri(eri, n, i, m, j, l); d10 -= dd_eri(eri, n, i, m, j, l);
            d11 += dd_eri(eri, n, i, l, j, m) + 0.5 * dd_eri(eri, n, i, m, j, l);
        }
        e += C(d00, d01 * DD_S3_4, d10 * DD_S3_4, d11);
        if (dIL && dJM && dKN && pr == pc) e += eps[ra] - (eps[i] + eps[j] + eps[k]);
    }
#undef C
    return e;
}

// ------------------------------- triplet ------------------------------------

__device__ double d_jii_t(int j, int i, int l, int k, int ra, int sb,
                          const double *eri, const double *eps, const int *osym, int n) {
    int dIL = (i == l), dJL = (j == l), dIK = (i == k), dJK = (j == k), dSym = (osym[ra] == osym[sb]);
    double e = 0.0;
    if (dIK) e += -BB(j, l);
    if (dIK && dJL) e += AA(i, i) - 2 * BB(i, i);
    if (dIL && dJK) e += -AA(l, j) + BB(j, l);
    if (dSym && ra == sb) {
        double diag = 0.0;
        if (dIK) diag += 2 * dd_eri(eri, n, j, l, i, i) - dd_eri(eri, n, i, j, i, l);
        if (dIL) diag -= dd_eri(eri, n, j, k, l, k);
        if (dJK) diag -= dd_eri(eri, n, i, j, i, l);
        if (dJL) diag += dd_eri(eri, n, i, k, i, k);
        if (dJL && dIK) { diag -= eps[j] + eps[i] + eps[i]; diag += eps[ra]; }
        e += diag;
    }
    return e;
}

__device__ double d_ijkMLL_t(int i, int j, int k, int m, int l, int pr, int ra, int sb,
                             const double *eri, const int *osym, int n) {
    int dIM = (i == m), dJM = (j == m), dKM = (k == m);
    int dIL = (i == l), dJL = (j == l), dKL = (k == l), dSym = (osym[ra] == osym[sb]);
    double e = 0.0;
#define C(c0, c1, c2) (pr == 0 ? (c0) : pr == 1 ? (c1) : (c2))
    if (dIM && dJL) e += C(0, -1, -1) * AA(k, j) + C(-1, 1, 0) * BB(j, k);
    if (dIM && dKL) e += C(1, 0, 1) * AA(j, k) + C(-1, 1, 0) * BB(j, k);
    if (dJM && dIL) e += C(0, 1, 1) * AA(k, i) + C(1, 0, -1) * BB(i, k);
    if (dJM && dKL) e += C(-1, -1, 0) * AA(i, k) + C(1, 0, -1) * BB(i, k);
    if (dKM && dIL) e += C(-1, 0, -1) * AA(j, i) + C(0, -1, 1) * BB(i, j);
    if (dKM && dJL) e += C(1, 1, 0) * AA(i, j) + C(0, -1, 1) * BB(i, j);
    if (dSym && ra == sb) {
        double d0 = 0, d1 = 0, d2 = 0;
        if (dIM) { d0 += dd_eri(eri, n, j, l, k, l); d1 -= dd_eri(eri, n, j, l, k, l); }
        if (dIL) { d0 -= dd_eri(eri, n, i, k, j, m); d1 += dd_eri(eri, n, i, j, k, m); d2 += dd_vm(eri, n, i, k, m, j); }
        if (dJM) { d0 -= dd_eri(eri, n, i, l, k, l); d2 += dd_eri(eri, n, i, l, k, l); }
        if (dJL) { d0 += dd_eri(eri, n, i, m, j, k); d1 += dd_vm(eri, n, i, j, k, m); d2 -= dd_eri(eri, n, i, j, k, m); }
        if (dKM) { d1 += dd_eri(eri, n, i, l, j, l); d2 -= dd_eri(eri, n, i, l, j, l); }
        if (dKL) { d0 += dd_vm(eri, n, i, m, j, k); d1 -= dd_eri(eri, n, i, m, j, k); d2 += dd_eri(eri, n, i, k, j, m); }
        e += C(d0, d1, d2);
    }
#undef C
    return e;
}

__device__ double d_ijkLMN_t(int i, int j, int k, int l, int m, int nn, int pr, int pc, int ra, int sb,
                             const double *eri, const double *eps, const int *osym, int n) {
    int dIL = (i == l), dJL = (j == l), dKL = (k == l);
    int dJM = (j == m), dKM = (k == m), dJN = (j == nn), dKN = (k == nn), dSym = (osym[ra] == osym[sb]);
    double e = 0.0;
    int idx = pr * 3 + pc;
#define C9(a0, a1, a2, a3, a4, a5, a6, a7, a8) \
    (idx == 0 ? (a0) : idx == 1 ? (a1) : idx == 2 ? (a2) : idx == 3 ? (a3) : idx == 4 ? (a4) : idx == 5 ? (a5) : idx == 6 ? (a6) : idx == 7 ? (a7) : (a8))
    if (dIL && dJM) e += C9(0, 0, 0, 0, 1, 1, 0, 1, 1) * AA(k, nn) + C9(-1, 0, 0, 0, -1, 0, 0, 0, -1) * BB(k, nn);
    if (dIL && dKM) e += C9(0, -1, -1, 0, 0, 0, 0, -1, -1) * AA(j, nn) + C9(0, 1, 0, 1, 0, 0, 0, 0, 1) * BB(j, nn);
    if (dIL && dKN) e += C9(1, 0, 1, 0, 0, 0, 1, 0, 1) * AA(j, m) + C9(-1, 0, 0, 0, -1, 0, 0, 0, -1) * BB(j, m);
    if (dJL && dKM) e += C9(0, 1, 1, 0, 1, 1, 0, 0, 0) * AA(i, nn) + C9(0, -1, 0, 0, 0, -1, -1, 0, 0) * BB(i, nn);
    if (dJL && dKN) e += C9(-1, 0, -1, -1, 0, -1, 0, 0, 0) * AA(i, m) + C9(1, 0, 0, 0, 0, 1, 0, 1, 0) * BB(i, m);
    if (dJM && dKN) e += C9(1, 1, 0, 1, 1, 0, 0, 0, 0) * AA(i, l) + C9(-1, 0, 0, 0, -1, 0, 0, 0, -1) * BB(i, l);
    if (dSym && ra == sb) {
        double d[9];
#pragma unroll
        for (int t = 0; t < 9; t++) d[t] = 0.0;
        // d[a*3+b]
        if (dIL) {
            d[0] += dd_eri(eri, n, j, m, k, nn); d[1] -= dd_eri(eri, n, j, nn, k, m);
            d[3] -= dd_eri(eri, n, j, nn, k, m); d[4] += dd_eri(eri, n, j, m, k, nn);
            d[8] += dd_vm(eri, n, j, m, k, nn);
        }
        if (dJL) {
            d[0] -= dd_eri(eri, n, i, m, k, nn); d[1] += dd_eri(eri, n, i, nn, k, m);
            d[5] += dd_vm(eri, n, i, nn, k, m); d[6] += dd_eri(eri, n, i, nn, k, m); d[7] -= dd_eri(eri, n, i, m, k, nn);
        }
        if (dJM) {
            d[0] += dd_eri(eri, n, i, l, k, nn); d[2] -= dd_eri(eri, n, i, nn, k, l);
            d[4] += dd_vm(eri, n, i, l, k, nn); d[6] -= dd_eri(eri, n, i, nn, k, l); d[8] += dd_eri(eri, n, i, l, k, nn);
        }
        if (dJN) {
            d[1] -= dd_eri(eri, n, i, l, k, m); d[2] += dd_eri(eri, n, i, m, k, l);
            d[3] -= dd_vm(eri, n, i, l, k, m); d[7] += dd_eri(eri, n, i, m, k, l); d[8] -= dd_eri(eri, n, i, l, k, m);
        }
        if (dKL) {
            d[2] += dd_vm(eri, n, i, m, j, nn); d[3] += dd_eri(eri, n, i, m, j, nn);
            d[4] -= dd_eri(eri, n, i, nn, j, m); d[6] -= dd_eri(eri, n, i, nn, j, m); d[7] += dd_eri(eri, n, i, m, j, nn);
        }
        if (dKM) {
            d[1] += dd_vm(eri, n, i, nn, j, l); d[3] -= dd_eri(eri, n, i, l, j, nn);
            d[5] += dd_eri(eri, n, i, nn, j, l); d[6] += dd_eri(eri, n, i, nn, j, l); d[8] -= dd_eri(eri, n, i, l, j, nn);
        }
        if (dKN) {
            d[0] += dd_vm(eri, n, i, l, j, m); d[4] += dd_eri(eri, n, i, l, j, m);
            d[5] -= dd_eri(eri, n, i, m, j, l); d[7] -= dd_eri(eri, n, i, m, j, l); d[8] += dd_eri(eri, n, i, l, j, m);
        }
        e += d[idx];
        if (dIL && dJM && dKN && pr == pc) e += eps[ra] - (eps[i] + eps[j] + eps[k]);
    }
#undef C9
    return e;
}

// ------------------------------- driver -------------------------------------

// Group struct-of-arrays. JII (type I): occ0/occ1, start (global index), voff (offset into
// jVir), nv. IJK (type II): occ0/1/2, start, voff (into iVir), nv. jVir/iVir are the flat
// concatenated absolute virtual orbitals of each group.
struct JGroups { const int *o0, *o1, *st, *voff, *nv, *vir; };
struct IGroups { const int *o0, *o1, *o2, *st, *voff, *nv, *vir; };
// Row SoA over the 3h1p region (index = global − mainOff): type (0=JII,1=IJK), grp, part, vir.
struct RowSoA { const int *typ, *grp, *part, *vir; };

// dip_sat_apply: one thread per output 3h1p row. Accumulates out[main+ri] += Σ_C M[R,C]·in[C]
// over candidate column groups (shared-occ early-out). spin: 0 singlet, 1 triplet; parts: 2/3.
//
// ROW BAND (-mgpu per-device apply). [rowLo,rowHi) restricts the 3h1p rows this launch owns, and
// outRowOff is subtracted from the global row to index a row-partitioned output panel: device d
// passes rowLo/rowHi for its own band and outRowOff = bound[d]. `xin` is always indexed by the
// GLOBAL row, so the caller must hand this kernel a FULL-HEIGHT input (the gathered n×w slab) —
// a candidate column may live in any partition. Single-device callers pass
// rowLo=0, rowHi=nsat, outRowOff=0, which reproduces the whole-band behaviour exactly.
__global__ void dip_sat_apply(int nsat, int njii, int nijk, int b, int ldIn, int ldOut,
                              int mainOff, int norb, int parts, int spin,
                              int rowLo, int rowHi, int outRowOff,
                              RowSoA rw, JGroups jg, IGroups ig,
                              const double *__restrict__ eri, const double *__restrict__ eps,
                              const int *__restrict__ osym,
                              const double *__restrict__ xin, double *__restrict__ yout) {
    int ri = rowLo + blockIdx.x * blockDim.x + threadIdx.x;
    if (ri >= rowHi || ri >= nsat) return;
    int n = norb;
    int rTyp = rw.typ[ri], rGrp = rw.grp[ri], rPart = rw.part[ri], rVir = rw.vir[ri];

    int ro0, ro1, ro2, rn;
    if (rTyp == 0) { ro0 = jg.o0[rGrp]; ro1 = jg.o1[rGrp]; ro2 = -1; rn = 2; }
    else { ro0 = ig.o0[rGrp]; ro1 = ig.o1[rGrp]; ro2 = ig.o2[rGrp]; rn = 3; }

    int R = mainOff + ri;        // global row: indexes xin and the candidate comparisons
    int Rout = R - outRowOff;    // local row within this device's output panel
    // accumulate into a private register bank per panel column (b is small).
    // b can exceed a fixed cap; loop columns in the inner apply instead.

    // --- JII column groups (single spin part) ---
    for (int cg = 0; cg < njii; cg++) {
        int co0 = jg.o0[cg], co1 = jg.o1[cg];
        // shared-occ early-out
        int share = (co0 == ro0) || (co0 == ro1) || (co1 == ro0) || (co1 == ro1) ||
                    (rn == 3 && (co0 == ro2 || co1 == ro2));
        if (!share) continue;
        int cst = jg.st[cg], voff = jg.voff[cg], cnv = jg.nv[cg];
        for (int cb = 0; cb < cnv; cb++) {
            int sb = jg.vir[voff + cb];
            double g;
            if (rTyp == 0) { // jiiLKK, higher-index group is the row
                if (rGrp >= cg) g = (spin == 0) ? d_jii_s(ro0, ro1, co0, co1, rVir, sb, eri, eps, osym, n)
                                                : d_jii_t(ro0, ro1, co0, co1, rVir, sb, eri, eps, osym, n);
                else g = (spin == 0) ? d_jii_s(co0, co1, ro0, ro1, sb, rVir, eri, eps, osym, n)
                                     : d_jii_t(co0, co1, ro0, ro1, sb, rVir, eri, eps, osym, n);
            } else { // ijkMLL: R (IJK) is the row, C (JII) the column
                g = (spin == 0) ? d_ijkMLL_s(ro0, ro1, ro2, co0, co1, rPart, rVir, sb, eri, osym, n)
                                : d_ijkMLL_t(ro0, ro1, ro2, co0, co1, rPart, rVir, sb, eri, osym, n);
            }
            if (g == 0.0) continue;
            int C = cst + cb;
            for (int jc = 0; jc < b; jc++) yout[Rout + jc * ldOut] += g * xin[C + jc * ldIn];
        }
    }

    // --- IJK column groups (parts spin parts) ---
    for (int cg = 0; cg < nijk; cg++) {
        int co0 = ig.o0[cg], co1 = ig.o1[cg], co2 = ig.o2[cg];
        int share = (co0 == ro0) || (co0 == ro1) || (co1 == ro0) || (co1 == ro1) ||
                    (co2 == ro0) || (co2 == ro1) ||
                    (rn == 3 && (co0 == ro2 || co1 == ro2 || co2 == ro2));
        if (!share) continue;
        int cst = ig.st[cg], voff = ig.voff[cg], cnv = ig.nv[cg];
        for (int cpart = 0; cpart < parts; cpart++) {
            for (int cb = 0; cb < cnv; cb++) {
                int sb = ig.vir[voff + cb];
                double g;
                if (rTyp == 0) { // ijkMLL transposed: C (IJK) row, R (JII) column
                    g = (spin == 0) ? d_ijkMLL_s(co0, co1, co2, ro0, ro1, cpart, sb, rVir, eri, osym, n)
                                    : d_ijkMLL_t(co0, co1, co2, ro0, ro1, cpart, sb, rVir, eri, osym, n);
                } else { // ijkLMN: higher-index group is the row
                    if (rGrp >= cg) g = (spin == 0) ? d_ijkLMN_s(ro0, ro1, ro2, co0, co1, co2, rPart, cpart, rVir, sb, eri, eps, osym, n)
                                                    : d_ijkLMN_t(ro0, ro1, ro2, co0, co1, co2, rPart, cpart, rVir, sb, eri, eps, osym, n);
                    else g = (spin == 0) ? d_ijkLMN_s(co0, co1, co2, ro0, ro1, ro2, cpart, rPart, sb, rVir, eri, eps, osym, n)
                                         : d_ijkLMN_t(co0, co1, co2, ro0, ro1, ro2, cpart, rPart, sb, rVir, eri, eps, osym, n);
                }
                if (g == 0.0) continue;
                int C = cst + cpart * cnv + cb;
                for (int jc = 0; jc < b; jc++) yout[Rout + jc * ldOut] += g * xin[C + jc * ldIn];
            }
        }
    }
}

// ============================================================================
// jiiLKK block FILL — the contraction path (docs/sigma_build_contractions.md).
//
// dip_sat_apply above recomputes each element and immediately broadcasts it across the b panel
// columns with a scalar loop, which is bandwidth-bound and measured 0.58% of fp64 peak. This
// kernel instead MATERIALIZES a batch of jiiLKK blocks into scratch, so the panel multiply is
// then done by cuBLAS batched dgemm — each block element is read by a tiled GEMM b times instead
// of being re-broadcast by hand. At melanin's b=1653 that reuse is the whole point.
//
// The arithmetic is unchanged: it calls the SAME d_jii_s / d_jii_t the per-scalar kernel uses,
// which are pinned against the dense reference by TestSatelliteScalarMatchesDense. Only the
// consumer differs.
//
// Orientation: the Go planner enumerates gc <= gr, so the row config is always the higher-indexed
// group — matching d_jii_s's (j,i | l,k) order with row first, i.e. the `rGrp >= cg` branch of
// dip_sat_apply. Blocks are written ROW-MAJOR (buf[bufOff + r*nc + c]), which is what
// backend.Mat / devMat is.
// ============================================================================

// Slot kinds, mirroring dip.satKind. The three blocks differ in how a flat (r,c) element index
// decodes into (spin part, virtual orbital) — their dims already fold in the parts factor
// (blocks.go ijkMLLShape/ijkLMNShape), so the block is a plain dense Mat in every case:
//   JII    : nvR        × nvC          r = a
//   IJKMLL : parts·nvR  × nvC          r = pr·nvR + a
//   IJKLMN : parts·nvR  × parts·nvC    r = pr·nvR + a,  c = pc·nvC + bIdx
#define DIPK_JII 0
#define DIPK_IJKMLL 1
#define DIPK_IJKLMN 2

// One CUDA block-row per slot (blockIdx.y), grid-stride over that slot's elements. Blocks are
// written ROW-MAJOR (buf[bufOff + r*totalCols + c]), matching backend.Mat/devMat.
__global__ void dip_fill_sat(int nslot, int spin, int norb, int parts,
                             const int *__restrict__ kind,
                             const int *__restrict__ rowO0, const int *__restrict__ rowO1,
                             const int *__restrict__ rowO2,
                             const int *__restrict__ colO0, const int *__restrict__ colO1,
                             const int *__restrict__ colO2,
                             const int *__restrict__ rowVOff, const int *__restrict__ rowNv,
                             const int *__restrict__ colVOff, const int *__restrict__ colNv,
                             const int *__restrict__ bufOff, const int *__restrict__ virs,
                             const double *__restrict__ eri, const double *__restrict__ eps,
                             const int *__restrict__ osym,
                             double *__restrict__ buf) {
    int s = blockIdx.y;
    if (s >= nslot) return;
    int kd = kind[s];
    int nvR = rowNv[s], nvC = colNv[s];
    int rvo = rowVOff[s], cvo = colVOff[s], bo = bufOff[s];

    // Full block dims including the parts factor.
    int nr = (kd == DIPK_JII) ? nvR : parts * nvR;
    int nc = (kd == DIPK_IJKLMN) ? parts * nvC : nvC;
    int total = nr * nc;

    int ro0 = rowO0[s], ro1 = rowO1[s], ro2 = rowO2[s];
    int co0 = colO0[s], co1 = colO1[s], co2 = colO2[s];

    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
        int r = e / nc, c = e - r * nc;
        double v;
        if (kd == DIPK_JII) {
            int ra = virs[rvo + r], sb = virs[cvo + c];
            v = (spin == 0) ? d_jii_s(ro0, ro1, co0, co1, ra, sb, eri, eps, osym, norb)
                            : d_jii_t(ro0, ro1, co0, co1, ra, sb, eri, eps, osym, norb);
        } else if (kd == DIPK_IJKMLL) {
            int pr = r / nvR, a = r - pr * nvR;
            int ra = virs[rvo + a], sb = virs[cvo + c];
            v = (spin == 0) ? d_ijkMLL_s(ro0, ro1, ro2, co0, co1, pr, ra, sb, eri, osym, norb)
                            : d_ijkMLL_t(ro0, ro1, ro2, co0, co1, pr, ra, sb, eri, osym, norb);
        } else {
            int pr = r / nvR, a = r - pr * nvR;
            int pc = c / nvC, bIdx = c - pc * nvC;
            int ra = virs[rvo + a], sb = virs[cvo + bIdx];
            v = (spin == 0) ? d_ijkLMN_s(ro0, ro1, ro2, co0, co1, co2, pr, pc, ra, sb, eri, eps, osym, norb)
                            : d_ijkLMN_t(ro0, ro1, ro2, co0, co1, co2, pr, pc, ra, sb, eri, eps, osym, norb);
        }
        buf[bo + e] = v;
    }
}

#undef AA
#undef BB

extern "C" {

// adc2_dip_fill_sat materializes nslot satellite blocks (any kind) into buf. maxElems is the largest single
// block's element count, used only to size the grid's x extent (the grid-stride loop covers any
// remainder). Returns the first cudaError_t seen.
int adc2_dip_fill_sat(int nslot, int spin, int norb, int parts, int maxElems,
                      const int *kind,
                      const int *rowO0, const int *rowO1, const int *rowO2,
                      const int *colO0, const int *colO1, const int *colO2,
                      const int *rowVOff, const int *rowNv, const int *colVOff, const int *colNv,
                      const int *bufOff, const int *virs,
                      const double *eri, const double *eps, const int *osym, double *buf) {
    if (nslot <= 0) return 0;
    int T = 128;
    int gx = (maxElems + T - 1) / T;
    if (gx < 1) gx = 1;
    if (gx > 256) gx = 256;
    dim3 grid(gx, nslot);
    dip_fill_sat<<<grid, T>>>(nslot, spin, norb, parts, kind,
                              rowO0, rowO1, rowO2, colO0, colO1, colO2,
                              rowVOff, rowNv, colVOff, colNv, bufOff, virs,
                              eri, eps, osym, buf);
    return (int)cudaGetLastError();
}

// adc2_dip_sat_apply launches the satellite apply on the default stream. All pointers are
// device pointers. [rowLo,rowHi) is the 3h1p row band this launch owns and outRowOff the global
// row offset of the output panel (whole-band single-device: 0, nsat, 0). xin must be full height
// regardless of the band — candidate columns are global. A caller whose band is empty
// (rowHi <= rowLo) must skip the call; launching zero blocks here is treated as a bug, not a
// no-op, so an off-by-one in the band derivation is caught rather than silently doing nothing.
// Returns the first cudaError_t seen.
int adc2_dip_sat_apply(int nsat, int njii, int nijk, int b, int ldIn, int ldOut,
                       int mainOff, int norb, int parts, int spin,
                       int rowLo, int rowHi, int outRowOff,
                       const int *rTyp, const int *rGrp, const int *rPart, const int *rVir,
                       const int *jO0, const int *jO1, const int *jSt, const int *jVoff, const int *jNv, const int *jVir,
                       const int *iO0, const int *iO1, const int *iO2, const int *iSt, const int *iVoff, const int *iNv, const int *iVir,
                       const double *eri, const double *eps, const int *osym,
                       const double *xin, double *yout) {
    if (rowHi > nsat) rowHi = nsat;
    int rows = rowHi - rowLo;
    if (rows <= 0) return (int)cudaErrorInvalidValue;
    RowSoA rw = {rTyp, rGrp, rPart, rVir};
    JGroups jg = {jO0, jO1, jSt, jVoff, jNv, jVir};
    IGroups ig = {iO0, iO1, iO2, iSt, iVoff, iNv, iVir};
    int T = 128;
    dip_sat_apply<<<(rows + T - 1) / T, T>>>(nsat, njii, nijk, b, ldIn, ldOut, mainOff, norb, parts, spin,
                                             rowLo, rowHi, outRowOff,
                                             rw, jg, ig, eri, eps, osym, xin, yout);
    return (int)cudaGetLastError();
}

} // extern "C"
