package sip

import (
	"math"
	"unsafe"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/matfree"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// matfree.go — matrix-free application of the large ADC(4) coupling blocks.
//
// The block-sparse operator (matvec.go) assembles every block densely and reuses it
// each mat-vec. For a big sector the 2h1p×3h2p WERT2 coupling (coupling24_4) is the
// memory ceiling — ~3.8 GB for pyridine/cc-pVDZ. A matrix-free part instead recomputes
// its elements on the fly during ApplyFull/ApplyBlock, holding zero resident bytes.
//
// This is affordable because wert2elem4 is δ-sparse: an element is nonzero only when
// the 2h1p row's particle matches a 3h2p column particle (Vir==I or ==J) or its valence
// hole matches a column hole (Occ[1]==L or ==M), together with the core-hole coincidence
// (elements4.go, the vint[...] gates). So per column we visit only the candidate rows
// pulled from those index buckets — ~12% of the block for pyridine — and a false
// positive simply evaluates to 0. One element eval serves both the forward (y2 += G·x3)
// and transpose (y3 += Gᵀ·x2) directions.

// MatFreeMode and its constants alias the shared matfree policy (internal/adc/matfree),
// kept here so the CLI and existing callers keep using sip.MatFreeMode unchanged.
type MatFreeMode = matfree.Mode

const (
	MatFreeOff  = matfree.Off
	MatFreeAuto = matfree.Auto
	MatFreeOn   = matfree.On
)

// matFreePart is an operator block applied by recomputing its elements each mat-vec
// instead of storing a dense A. apply accumulates M_block·in into out over the block's
// row/col bands — both the block and its transpose, realizing the symmetric operator —
// and runs after the dense GEMM and diagonal loops (so it must use += semantics).
// release frees any scratch (a no-op for the host applier, which allocates per-call).
type matFreePart struct {
	apply   func(in, out backend.BlockView)
	release func()
}

// SetMatFree configures matrix-free assembly for this sector. budgetBytes is the
// per-block dense-size threshold used by MatFreeAuto (ignored for Off/On).
func (mx *Matrix) SetMatFree(mode MatFreeMode, budgetBytes int64) {
	mx.matFree = mode
	mx.matFreeBudget = budgetBytes
}

// matFreeDecision applies the mode to a block, given whether the backend supports a
// matrix-free path for that block. Unsupported → dense (correct fallback).
func (mx *Matrix) matFreeDecision(denseBytes int64, supported bool) bool {
	return matfree.Decide(mx.matFree, denseBytes, mx.matFreeBudget, supported)
}

// matFreeWert2 decides the 2h1p×3h2p coupling: supported on host (HostData) and on a
// device backend that provides the wert2 kernel (DeviceKernels).
func (mx *Matrix) matFreeWert2(denseBytes int64) bool {
	_, host := mx.be.(backend.HostData)
	_, dev := mx.be.(backend.DeviceKernels)
	return mx.matFreeDecision(denseBytes, host || dev)
}

// matFreeC22 decides the order-4 2h1p×2h1p block (c22elem4): host-only, because there is no
// device c22elem4 kernel — a device backend must fall back to the dense block.
//
// The DeviceKernels test MUST come first. gpuBackend embeds Gonum (backend/gpu_device.go), so it
// satisfies HostData through the promoted Gonum.HostSlice even though its vectors are devVec —
// testing HostData alone selects newC22MatFree on a CUDA backend and panics on the first
// HostSlice(devVec). Same ordering rationale as dip/matfree.go matFreeSupported. Pinned by
// TestMatFreeC22RejectsDeviceBackend (no GPU needed).
func (mx *Matrix) matFreeC22(denseBytes int64) bool {
	if _, dev := mx.be.(backend.DeviceKernels); dev {
		return false
	}
	_, host := mx.be.(backend.HostData)
	return mx.matFreeDecision(denseBytes, host)
}

// newWert2MatFree builds the matrix-free applier for the 2h1p×3h2p WERT2 coupling,
// the on-the-fly equivalent of coupling24_4 placed at (BeginSat, Begin3h2p). The
// candidate buckets over the small 2h1p side are precomputed once; the apply is fused
// over both directions and parallelized over the 3h2p columns (the large,
// contention-free side) with per-worker 2h1p partials reduced in fixed order.
func (mx *Matrix) newWert2MatFree() matFreePart {
	if dk, ok := mx.be.(backend.DeviceKernels); ok {
		return mx.newWert2MatFreeDevice(dk)
	}
	sp := mx.sp
	main := sp.BeginSat
	off3 := sp.Begin3h2p
	rows := sp.Configs[main:off3] // 2h1p configs (block rows)
	cols := sp.Sat3               // 3h2p configs (block cols)
	n2 := len(rows)
	el := mx.el
	hd := mx.be.(backend.HostData)

	// Candidate buckets: rows keyed by their particle (Vir) and valence hole (Occ[1]).
	byPart := make([][]int32, sp.Nvir)
	byHole := make([][]int32, sp.Norb)
	for r := range rows {
		byPart[rows[r].Vir] = append(byPart[rows[r].Vir], int32(r))
		byHole[rows[r].Occ[1]] = append(byHole[rows[r].Occ[1]], int32(r))
	}

	// W and the per-worker accumulators are latched HERE, not derived per apply.
	//
	// Determinism. The forward leg is the one real reduction in this tree: each worker
	// accumulates a private full-width 2h1p partial and they are summed afterwards, so W fixes
	// how the columns are grouped, and floating-point addition is not associative. Go 1.25+
	// re-reads GOMAXPROCS from the cgroup CPU limit while the process runs, so deriving W per
	// apply let the operator silently change its rounding mid-solve — Lanczos would carry on a
	// short recurrence whose alpha/beta came from a numerically different M, and the
	// orthogonality bound that suppresses ghost and duplicated roots no longer holds.
	//
	// Churn. partials is W*n2*b float64 — GB-scale at CVS ADC(4) widths, since it is GOMAXPROCS
	// copies of the whole 2h1p output panel — and allocating it per mat-vec made every Lanczos
	// iteration hand that back to the allocator. It is sized once and re-zeroed instead.
	//
	// Still open: a W latched here is stable within a process but not across machines, so a
	// -checkpoint continuation on a node with a different core count resumes with a different
	// grouping. Closing that needs a W independent of the hardware, which costs either a cap on
	// parallelism or a second wert2elem4 pass. Not decided here.
	W := parallel.ChunkWorkers(len(cols))
	var partials []float64 // per-worker forward (2h1p) accumulators, reused across applies

	apply := func(in, out backend.BlockView) {
		xin := hd.HostSlice(in.V)
		yout := hd.HostSlice(out.V)
		b := in.Cols
		ldi, ldo := in.Ld, out.Ld
		n3 := len(cols)
		if n3 == 0 || n2 == 0 {
			return
		}

		// b is the Krylov block width and may shrink on the last block, so size to the request
		// and keep the larger buffer. ApplyBlock drives one Matrix serially (it reuses op.sa/sb/sc
		// scratch), so this buffer needs no locking.
		if need := W * n2 * b; cap(partials) < need {
			partials = make([]float64, need)
		} else {
			partials = partials[:need]
			clear(partials)
		}

		parallel.Chunks(n3, W, func(w, c0, c1 int) {
			y2 := partials[w*n2*b : (w+1)*n2*b]
			stamp := make([]int32, n2) // per-column dedup of candidate rows
			for i := range stamp {
				stamp[i] = -1
			}
			for c := c0; c < c1; c++ {
				col := cols[c]
				gen := int32(c) // unique per column within this worker
				colBase := off3 + c
				visit := func(list []int32) {
					for _, r32 := range list {
						r := int(r32)
						if stamp[r] == gen {
							continue
						}
						stamp[r] = gen
						g := el.wert2elem4(rows[r], col)
						if g == 0 {
							continue
						}
						rowBase := main + r
						for j := 0; j < b; j++ {
							y2[r+j*n2] += g * xin[colBase+j*ldi]          // forward: y2 += g·x3
							yout[colBase+j*ldo] += g * xin[rowBase+j*ldi] // transpose: y3 += g·x2
						}
					}
				}
				visit(byPart[col.I])
				visit(byPart[col.J])
				visit(byHole[col.L])
				visit(byHole[col.M])
			}
		})

		// Reduce per-worker forward partials into the 2h1p output band, fixed order.
		for w := 0; w < W; w++ {
			y2 := partials[w*n2*b : (w+1)*n2*b]
			for j := 0; j < b; j++ {
				base := main + j*ldo
				col := j * n2
				for r := 0; r < n2; r++ {
					yout[base+r] += y2[col+r]
				}
			}
		}
	}
	return matFreePart{apply: apply, release: func() {}}
}

// newC22MatFree builds the matrix-free applier for the 2h1p×2h1p WERT1 satellite block
// (satBlock2_4), the symmetric diagonal block at (BeginSat, BeginSat). It recomputes each
// element on the fly, parallelized over rows (each owns its output cell — no reduction).
//
// Unlike the 2h1p×3h2p coupling this is not a memory win at pyridine scale (the block is
// n2²·8 ≈ 5.6 MB); it exists for generality — letting a sector whose satBlock2_4 would not
// fit RAM (n2 ≳ 10⁴) still run, trading compute for memory. It is compute-heavy for CVS,
// where k==kk always holds so every element carries the 4th-order SUM1 term (O(nval·nvir)
// per element); the factorized-SUM1 fast path (separable sum1_4 denominator → GEMMs) is a
// documented follow-up. So -matfree switches it only on the memory threshold, never for
// symmetry with the coupling block.
func (mx *Matrix) newC22MatFree() matFreePart {
	sp := mx.sp
	main := sp.BeginSat
	rows := sp.Configs[main:sp.Begin3h2p]
	n2 := len(rows)
	el := mx.el
	hd := mx.be.(backend.HostData)

	apply := func(in, out backend.BlockView) {
		xin := hd.HostSlice(in.V)
		yout := hd.HostSlice(out.V)
		b := in.Cols
		ldi, ldo := in.Ld, out.Ld
		parallel.Rows(n2, func(r int) {
			rowCfg := rows[r]
			for c := 0; c < n2; c++ {
				g := el.c22elem4(rowCfg, rows[c])
				if g == 0 {
					continue
				}
				for j := 0; j < b; j++ {
					yout[main+r+j*ldo] += g * xin[main+c+j*ldi]
				}
			}
		})
	}
	return matFreePart{apply: apply, release: func() {}}
}

// matFreeC22O3 decides the order-3 2h1p×2h1p satellite block (satBlock): supported
// matrix-free on the host (HostData) and on a device backend that provides the c22 kernel
// (DeviceKernels). Unlike matFreeC22 (order 4, host-only) this has a device path, because
// the order-3 element (c22diag/c22off) is a fixed set of ERI lookups — no SUM1 — so it maps
// to a GPU kernel (adc4_kernels.cu c22_apply).
func (mx *Matrix) matFreeC22O3(denseBytes int64) bool {
	_, host := mx.be.(backend.HostData)
	_, dev := mx.be.(backend.DeviceKernels)
	return mx.matFreeDecision(denseBytes, host || dev)
}

// newC22MatFreeO3 builds the matrix-free applier for the order-3 2h1p×2h1p satellite block —
// the on-the-fly equivalent of satBlock (matvec.go): the diagonal is c22diag, the off-diagonal
// is c22off. It dispatches to the GPU kernel when the backend supports it, else runs on the
// host parallelized over output rows (each row owns its output cell — no reduction).
//
// The block is symmetric and sits on the operator diagonal, so it is applied in a single pass
// (out[main+r] += Σ_c S(r,c)·in[main+c]); there is no transpose direction. The order-3 element
// is DIRECTIONAL — c22off(row,col) evaluates its k==l ("row single") and m==n ("col single")
// branches differently, and the dense satBlock fills S(r,c)=S(c,r)=c22off(cfg_r,cfg_c) only for
// r<c — so element(r,c) must be evaluated with the lower-indexed config as the row (lo=min(r,c))
// to stay bit-for-bit with the dense block. (The ADC(4) newC22MatFree can ignore this because
// c22elem4 is symmetric under argument swap.)
// c22Buckets prunes the 2h1p×2h1p candidate columns of a row to those that can be nonzero.
//
// c22off(row=(k,l,a), col=(m,n,b)) is nonzero ONLY when the two configs share the particle
// (a == b) or share a hole ({k,l} ∩ {m,n} ≠ ∅). Every term in the element is gated that way:
// the four deltaV calls return immediately unless their first two arguments are equal
// (elements.go), which is k==m, l==n, l==m and k==n respectively, and the only terms outside
// deltaV sit behind `a == b`. The k==l "akk" row branch is the same condition read with
// {k,l} = {k}.
//
// Without this the applier scans every column for every row: n2² = 518,056² ≈ 2.68e11 element
// evaluations PER MAT-VEC at production scale, which is hours of legitimate work per Lanczos block
// and is indistinguishable from a hang because nothing is logged per block. The gate cuts that
// to ~1/13 of the columns — a row's candidates are the ~2·nocc hole pairs touching k or l, times
// the virtuals, plus the one hole-pair family sharing its particle.
//
// The lists are built in ASCENDING row index and merged that way, which is what keeps the fix
// bit-exact: the applier already skips g == 0, so walking a superset of the nonzero columns in
// the original order replays exactly the same sequence of accumulations.
type c22Buckets struct {
	byHole [][]int32 // occupied index -> satellite rows holding it as k or l
	byVir  [][]int32 // virtual position -> satellite rows holding it as the particle
}

func buildC22Buckets(rows []Config, nocc, nvir int) *c22Buckets {
	b := &c22Buckets{byHole: make([][]int32, nocc), byVir: make([][]int32, nvir)}
	nh := make([]int, nocc)
	nv := make([]int, nvir)
	for _, c := range rows {
		nh[c.Occ[0]]++
		if c.Occ[1] != c.Occ[0] {
			nh[c.Occ[1]]++
		}
		nv[c.Vir]++
	}
	for i := range b.byHole {
		b.byHole[i] = make([]int32, 0, nh[i])
	}
	for i := range b.byVir {
		b.byVir[i] = make([]int32, 0, nv[i])
	}
	for r, c := range rows {
		b.byHole[c.Occ[0]] = append(b.byHole[c.Occ[0]], int32(r))
		if c.Occ[1] != c.Occ[0] {
			b.byHole[c.Occ[1]] = append(b.byHole[c.Occ[1]], int32(r))
		}
		b.byVir[c.Vir] = append(b.byVir[c.Vir], int32(r))
	}
	return b
}

// candidates merges the ascending lists for row (k,l,a) into dst, deduped and still ascending.
// Returns the merged slice (dst is reused across rows by the caller).
//
// `self` — the row's own index — is merged in as a fourth source so the DIAGONAL keeps its
// position in the ascending sweep. That is not cosmetic: the dense form accumulates column by
// column in ascending order, so lifting the diagonal out of the walk reassociates the sum and
// moves the result (measured at 7.1e-15 on h2o before this was fixed).
func (b *c22Buckets) candidates(dst []int32, k, l, vir, self int) []int32 {
	dst = dst[:0]
	x, y := b.byHole[k], b.byVir[vir]
	var z []int32
	if l != k {
		z = b.byHole[l]
	}
	one := [1]int32{int32(self)}
	w := one[:]
	i, j, m, q := 0, 0, 0, 0
	var last int32 = -1
	for i < len(x) || j < len(y) || m < len(z) || q < len(w) {
		v := int32(math.MaxInt32)
		if i < len(x) && x[i] < v {
			v = x[i]
		}
		if j < len(y) && y[j] < v {
			v = y[j]
		}
		if m < len(z) && z[m] < v {
			v = z[m]
		}
		if q < len(w) && w[q] < v {
			v = w[q]
		}
		for i < len(x) && x[i] == v {
			i++
		}
		for j < len(y) && y[j] == v {
			j++
		}
		for m < len(z) && z[m] == v {
			m++
		}
		for q < len(w) && w[q] == v {
			q++
		}
		if v != last {
			dst = append(dst, v)
			last = v
		}
	}
	return dst
}

func (mx *Matrix) newC22MatFreeO3() matFreePart {
	if dk, ok := mx.be.(backend.DeviceKernels); ok {
		return mx.newC22MatFreeO3Device(dk)
	}
	sp := mx.sp
	main := sp.BeginSat
	rows := sp.Configs[main:] // order-3 2h1p configs run to the end of Configs (no 3h2p)
	n2 := len(rows)
	el := mx.el
	hd := mx.be.(backend.HostData)
	bk := buildC22Buckets(rows, sp.Nocc, sp.Nvir)

	// HeavyRows, not Rows: n2 is large in production but a small sector can fall under Rows'
	// 2*GOMAXPROCS floor and silently run the whole apply on one core.
	apply := func(in, out backend.BlockView) {
		xin := hd.HostSlice(in.V)
		yout := hd.HostSlice(out.V)
		b := in.Cols
		ldi, ldo := in.Ld, out.Ld
		W := parallel.ChunkWorkers(n2)
		parallel.Chunks(n2, W, func(_, lo, hi int) {
			cand := make([]int32, 0, 1024)
			for r := lo; r < hi; r++ {
				rc := rows[r]
				cand = bk.candidates(cand, rc.Occ[0], rc.Occ[1], rc.Vir, r)
				for _, c32 := range cand {
					c := int(c32)
					var g float64
					switch {
					case c == r:
						g = el.c22diag(rc)
					case r < c:
						g = el.c22off(rc, rows[c]) // row = lower index
					default:
						g = el.c22off(rows[c], rc) // row = lower index (c < r)
					}
					if g == 0 {
						continue
					}
					for j := 0; j < b; j++ {
						yout[main+r+j*ldo] += g * xin[main+c+j*ldi]
					}
				}
			}
		})
	}
	return matFreePart{apply: apply, release: func() {}}
}

// deviceERI returns the flat norb⁴ ERI tensor the CUDA kernels index,
// eri[((p·n+q)·n+r)·n+s] = ints.Eri(p,q,r,s) — the layout of d_eri in adc4_kernels.cu and of
// dd_eri in adc2dip_kernels.cu. Built once per Matrix and memoized, exactly as dip's
// buildSatDeviceSoA does.
//
// Both matter at the production system's norb=212. The array is 2.02e9 doubles — 16.2 GB — and every
// device applier constructor (newC22MatFreeO3Device, newWert2MatFreeDevice) filled its own
// byte-identical copy from scratch with 2.0e9 scalar Eri() calls on ONE core, once per
// assemble; Release() followed by a fresh ApplyFull pays it again. Memoizing costs the 16 GB
// of host residency for the life of the Matrix, which is the same trade dip already makes and
// far cheaper than the rebuild.
//
// The p-slabs are disjoint output ranges of a preallocated array, so filling them concurrently
// is bit-exact — no arithmetic happens here at all, only ordered copies. HeavyRows rather than
// Rows because norb (212) is barely above Rows' 2*GOMAXPROCS fallback and would drop below it
// on a larger node, while one slab is norb³ = 9.5e6 loads.
//
// Assemble is single-goroutine (assembleStep runs its steps in sequence), so the memo needs no
// lock; the constructors are never called concurrently.
//
// Worth knowing for whoever needs the next 16 GB: this copy is redundant, not merely slow.
// integrals.Store.Eri forwards straight to fcidump.Data.TwoE, which reads its own dense
// NORB⁴ array at exactly this index expression, ((p·n+q)·n+r)·n+s — and Space.Norb *is*
// Data.NORB (cmd/adcgo passes d.NORB to NewSpace). So the loop below is an element-by-element
// self-copy, and an accessor handing back fcidump's slice would delete both the traversal and
// the second 16 GB. That accessor belongs in fcidump/integrals, which is why it is not here;
// dip's buildSatDeviceSoA carries the identical duplicate.
func (mx *Matrix) deviceERI() []float64 {
	if mx.flatERI != nil {
		return mx.flatERI
	}
	norb := mx.sp.Norb
	flat := make([]float64, norb*norb*norb*norb)
	parallel.HeavyRows(norb, func(p int) {
		for q := range norb {
			for r := range norb {
				base := ((p*norb+q)*norb + r) * norb
				for s := range norb {
					flat[base+s] = mx.el.ints.Eri(p, q, r, s)
				}
			}
		}
	})
	mx.flatERI = flat
	return flat
}

// newC22MatFreeO3Device is the GPU (cuda) applier for the order-3 2h1p×2h1p satellite block:
// it uploads the 2h1p config struct-of-arrays, the flat ERI tensor, and the orbital energies
// once, then each mat-vec launches the c22 recompute kernel (adc4_kernels.cu c22_apply),
// holding zero dense-block VRAM. The kernel is bit-for-bit the CPU element evaluation
// (c22diag/c22off, same lo=min(r,c) directionality); validate with the GPU parity test on
// real hardware (TestC22MatFreeO3DeviceParity).
func (mx *Matrix) newC22MatFreeO3Device(dk backend.DeviceKernels) matFreePart {
	sp := mx.sp
	main := sp.BeginSat
	rows := sp.Configs[main:]
	n2 := len(rows)
	norb, nocc := sp.Norb, mx.el.nocc

	// Config struct-of-arrays (int32), uploaded once. K=Occ[0], L=Occ[1], Vir, Typ.
	rK, rL, rVir, rTyp := make([]int32, n2), make([]int32, n2), make([]int32, n2), make([]int32, n2)
	for r, c := range rows {
		rK[r], rL[r], rVir[r], rTyp[r] = int32(c.Occ[0]), int32(c.Occ[1]), int32(c.Vir), int32(c.Typ)
	}

	// Flat ERI tensor eri[((a·n+c)·n+b)·n+d] = e.v(a,b,c,d) = ints.Eri(a,c,b,d) (== the wert2
	// layout, d_eri in adc4_kernels.cu). Shared and built in parallel — see deviceERI.
	eri := mx.deviceERI()

	dK, dL, dVir, dTyp := dk.UploadInts(rK), dk.UploadInts(rL), dk.UploadInts(rVir), dk.UploadInts(rTyp)
	dERI := dk.DeviceERI(eri)
	dEps := dk.UploadFloats(mx.el.eps)

	apply := func(in, out backend.BlockView) {
		dk.C22Apply(backend.C22Args{
			N2: n2, B: in.Cols, LdIn: in.Ld, LdOut: out.Ld,
			MainOff: main, Norb: norb, Nocc: nocc,
			K: dK, L: dL, Vir: dVir, Typ: dTyp,
			ERI: dERI, Eps: dEps, In: in.V, Out: out.V,
		})
	}
	release := func() {
		for _, p := range []unsafe.Pointer{dK, dL, dVir, dTyp, dERI, dEps} {
			dk.FreeDev(p)
		}
	}
	return matFreePart{apply: apply, release: release}
}

// newWert2MatFreeDevice is the GPU (cuda) applier for the 2h1p×3h2p coupling: it uploads
// the config struct-of-arrays, the flat ERI tensor, and the spin table once, then each
// mat-vec launches the recompute kernel (adc4_kernels.cu, both directions), holding zero
// dense-block VRAM. The kernel is bit-for-bit the CPU element evaluation; validate with
// the GPU parity test on real hardware (docs/adc4_matfree_gpu.md).
func (mx *Matrix) newWert2MatFreeDevice(dk backend.DeviceKernels) matFreePart {
	sp := mx.sp
	main, off3 := sp.BeginSat, sp.Begin3h2p
	rows, cols := sp.Configs[main:off3], sp.Sat3
	n2, n3 := len(rows), len(cols)
	norb, nocc := sp.Norb, mx.el.nocc

	// Config struct-of-arrays (int32), uploaded once.
	rVir, rK, rL, rTyp := make([]int32, n2), make([]int32, n2), make([]int32, n2), make([]int32, n2)
	for r, c := range rows {
		rVir[r], rK[r], rL[r], rTyp[r] = int32(c.Vir), int32(c.Occ[0]), int32(c.Occ[1]), int32(c.Typ)
	}
	cI, cJ, cK := make([]int32, n3), make([]int32, n3), make([]int32, n3)
	cL, cM, cSpin := make([]int32, n3), make([]int32, n3), make([]int32, n3)
	for c, cf := range cols {
		cI[c], cJ[c], cK[c] = int32(cf.I), int32(cf.J), int32(cf.Core)
		cL[c], cM[c], cSpin[c] = int32(cf.L), int32(cf.M), int32(cf.Spin)
	}

	// Flat ERI tensor eri[((p·n+q)·n+r)·n+s] = ints.Eri(p,q,r,s) (== TwoE) — the very same
	// bytes the c22 kernel indexes, so both come from deviceERI. The spin table
	// coeff1[3][13][30] is flattened to (Typ·13+col2)·30+(n-1).
	eri := mx.deviceERI()
	coeff1Flat := make([]float64, 3*13*30)
	for t := 0; t < 3; t++ {
		for c := 0; c < 13; c++ {
			copy(coeff1Flat[(t*13+c)*30:], coeff1[t][c][:])
		}
	}

	dRVir, dRK, dRL, dRTyp := dk.UploadInts(rVir), dk.UploadInts(rK), dk.UploadInts(rL), dk.UploadInts(rTyp)
	dCI, dCJ, dCK := dk.UploadInts(cI), dk.UploadInts(cJ), dk.UploadInts(cK)
	dCL, dCM, dCSpin := dk.UploadInts(cL), dk.UploadInts(cM), dk.UploadInts(cSpin)
	dERI := dk.DeviceERI(eri)
	dk.SetCoeff1(coeff1Flat)

	apply := func(in, out backend.BlockView) {
		dk.Wert2Apply(backend.Wert2Args{
			N2: n2, N3: n3, B: in.Cols, LdIn: in.Ld, LdOut: out.Ld,
			MainOff: main, Off3: off3, Norb: norb, Nocc: nocc,
			RVir: dRVir, RK: dRK, RL: dRL, RTyp: dRTyp,
			CI: dCI, CJ: dCJ, CK: dCK, CL: dCL, CM: dCM, CSpin: dCSpin,
			ERI: dERI, In: in.V, Out: out.V,
		})
	}
	release := func() {
		for _, p := range []unsafe.Pointer{dRVir, dRK, dRL, dRTyp, dCI, dCJ, dCK, dCL, dCM, dCSpin, dERI} {
			dk.FreeDev(p)
		}
	}
	return matFreePart{apply: apply, release: release}
}
