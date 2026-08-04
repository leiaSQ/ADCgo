# GPU performance backlog — what is left after the contraction rewrite

**Status:** survey, 2026-07-23. Nothing here is implemented. Written after the batched-GEMM
satellite path landed in the working tree, to separate *what that rewrite already covers* from
*what it does not*. Companions: [`sigma_build_contractions.md`](sigma_build_contractions.md) (the
rewrite itself), [`dip_operator_memory.md`](dip_operator_memory.md) (why the operator is
matrix-free), [`../internal/adc/backend/README.md`](../internal/adc/backend/README.md) (how a block
is split across devices).

## First: both companion docs are stale

`sigma_build_contractions.md` describes steps 5–7 (the device fill kernel) as open. They are not.
Present in the tree, uncommitted:

| piece | where |
|---|---|
| fill kernel, **all three** block kinds (JII/IJKMLL/IJKLMN) | `backend/adc2dip_kernels.cu:373-417` (`dip_fill_sat`) |
| Go binding, grow-on-demand scratch, no per-apply malloc | `backend/cuda_kernels.go:121-151` (`DipSatFillJII`) |
| single-GPU applier | `dip/matfree_batched.go:299-326` (`newSatBatchedDevice`) |
| multi-GPU twin | `dip/matfree_dist.go:181-288` (`newSatBatchedPerDevice`) |
| selection | `dip/matfree.go:94-104` — **unconditional** for any `DeviceKernels` backend |

> **RESOLVED 2026-07-23.** When first written this path was the default for the production system and had never
> run on a GPU. Two jobs have since validated it on real CUDA backends: `14023530` (1×A40 —
> `TestJIIFillDeviceMatchesHost`, `TestJIIMatFreeBatchedDeviceParity`, `TestSatelliteMatFreeDeviceParity`)
> and `14023720` (4×A40 — `TestSatBatchedPerDeviceParity`, `TestSatelliteMatFreePerDeviceParity`,
> max|Δ| ≈ 5–8e-14). Correctness is settled. **Performance is not:** both nodes were A40 with `SYS`
> (PCIe + cross-socket) peer paths, so no H200/NVLink number exists yet — see
> [`h200_optimization_plan.md`](h200_optimization_plan.md) Phase 0.

## Tier 1 — high impact, low effort

**All six LANDED 2026-07-23** (host tests green; the two `.cu` changes need a GPU build to
confirm — see the note after the list). Kept here as the record of what was wrong and why.

**1. The NVLink gather is serial.** `dip/matfree_dist.go:110-126` — the 8×8 peer-copy double loop
is a plain nested `for`, while `syncAll` fifteen lines above it (`:43-53`) already fans out with
`wg.Go`. `PeerCopy2D` bottoms out in a *blocking* `cudaMemcpy2D` (`backend/cuda.go:50-52,216-218`),
so each of the 64 copies fully blocks before the next is issued: ~1,664 serialized transfers per
mat-vec at the production system's 26 chunks, on a switch built for concurrent P2P. The fix is the pattern
already sitting directly above the loop.

**2. `wert2_fwd`/`wert2_trans` recompute the element `b` times.** `backend/adc4_kernels.cu:97-106`
and `:115-123` put `j` in the *outer* loop with `d_wert2` inside it, though `g` has no `j`
dependence — and `d_wert2` is not cheap (31-entry local array, ~16 conditional ERI loads, 30-term
dot product). `c22_apply` in the same file (`:237-250`) has the correct nest: `g` once per `c`, `j`
inner. SIP passes `b` through unchunked (`sip/matfree.go:314,380`), so this is a straight `b`-fold
blow-up, not an amortized chunking tax. The swap preserves summation order over `c`, so the result
is bit-identical.

**3. SIP's order-2/3 assemble is single-threaded.** `sip/matvec.go:98-126` — `coupling()` and
`satBlock()` are serial loops; `sip/matvec4.go:65` wraps the *identical shape* in `parallel.Rows`.
This is the drag `dip_operator_memory.md` records fixing for DIP, never applied to the ADC(2)/(3)
path production runs. It fires lazily on the first apply (`sip/matvec.go:277`) *inside* the apply
timer — so it is silently inflating the 96.6% figure that motivated the contraction work.

**4. Davidson's preconditioner stalls the device every iteration.** `lanczos/davidson.go:200-231`
downloads the full `n×nw` residual, runs a serial host double loop with a per-column
`make([]float64, n)`, and re-uploads. Davidson is the production DIP driver for the real the production system
runs. `parallel.Rows` applies directly.

**5. `distBackend.GemmMatBatched` un-batches itself.** `backend/distributed.go:516-520` loops
`gemmMatOne` per block, reintroducing the dispatch tax batching exists to remove — costed by its
own comment at `backend/gpu_device.go:290-296` at 181 s of a 379 s sector. Bucket by
`(owner, shape)` and issue one real batched call per bucket.

**6. `__restrict__` is absent** on every read-only kernel pointer in both `.cu` files
(`adc2dip_kernels.cu:26,30,33,267-272,373-381`; `adc4_kernels.cu:22,91-93,109-111,231-233`).
Zero-risk, and its absence is part of why nvcc cannot hoist #2 on its own.

> **Bit-exactness caveat on Tier 1 #2.** The `wert2` loop swap is *not* bit-identical to the
> previous kernel, contrary to what this document first claimed. Both forms sum in the same order
> over `c`, but the old one accumulated into a register and added to `yout` once — `y + (g₀x₀ +
> g₁x₁ + …)` — while the new one accumulates into `yout` per element: `((y + g₀x₀) + g₁x₁) + …`.
> Where `yout` is non-zero on entry (appliers accumulate in sequence) results can move by an ulp.
> Run-to-run determinism is unchanged. Recorded in `adc4_kernels.cu`'s header next to the existing
> determinism note.

## Tier 2 — worth doing

**All six LANDED 2026-07-23.** `satChunkCols` became `dip.SatChunkCols` + a `-satchunk` flag (and
`cmd/sizeprobe` now reads the real value instead of a hand-copied mirror); the all-reduce
downloads and uploads concurrently while keeping the *summation* serial in ascending device order
(fp addition is not associative — the order is deliberate); pinned host memory landed as a
`devHostAlloc` hook backing the batched-GEMM pointer staging, with a nil-returning HIP stub so
that backend behaves exactly as before; the SIP guard required a new `sip.OperatorResidentBytes`
(the DIP twin) because `backend.SectorBytes`' dense `n²` term is the upper bound `checkDeviceFit`
itself calls meaningless; the Ritz back-transform got a `StridedDownloader` capability
(one `cudaMemcpy2D` for the rectangle, per-column loop as fallback); `-Xptxas -v` is on by
default in the build script, silenced with `PTXAS_VERBOSE=0`.

| item | where | note |
|---|---|---|
| `satChunkCols` hardcoded at 64 | `dip/matfree_dist.go:36`, mirrored in `cmd/sizeprobe/main.go:23` | the budget table already says w=128 fits; also halves the barrier count in Tier 1 #1 |
| α/Gram/CGS2 all-reduce is 16 sequential blocking calls + per-call malloc/free | `backend/distributed.go:322-333` | runs several times per *iteration*, not per mat-vec; same `wg.Go` fix |
| no pinned host memory anywhere | `backend/cuda.go:193-199,225-227` | prerequisite for any async-transfer work |
| SIP has no pre-flight guard | `cmd/adcgo/sip_tdm.go:85-144` | `checkDeviceFit` is generic (`dispatch.go:130-150`) but never called; `pickLanczos`'s own check is skipped whenever a backend is pinned (`dispatch.go:101-107`) — i.e. always, in production. Oversized sectors hit a raw `cudaMalloc` panic |
| Ritz back-transform does `dim` separate small downloads | `lanczos/lanczos.go:502-509` | one bulk transfer instead |
| register spilling never measured | `scripts/build_adcgo_cuda_helix` | no `-Xptxas -v`; `d[9]` (`adc2dip_kernels.cu:208`) and `vint[31]` (`adc4_kernels.cu:44`) are plausible spillers |

## Tier 3 — structural

**ERI addressing vs. thread enumeration.** `dd_eri` (`adc2dip_kernels.cu:26-28`) puts `ra` in the
outermost index while rows are enumerated virtual-orbital-innermost (`dip/satscalar.go:59-85`), so
warp lanes gather ~76 MB apart at the production system's `norb=212` — plausibly the real cause of the measured
0.58% of peak, more than the candidate scan. Worth noting the scan (`adc2dip_kernels.cu:288,314`)
is **warp-uniform**, since consecutive threads share every occupied index: it is amortized
overhead, not divergence, and therefore smaller than the docs imply. The batched path moots this
for JII but not for IJK, and not at all for `adc4_kernels.cu`. A warp-per-row decomposition is the
fix, but it needs a deterministic shuffle-tree reduction to hold the bit-reproducibility bar
`adc4_kernels.cu:8-12` sets — only worth scoping if a profile still points here afterward.

**SIP has neither `lanczos-lowmem` nor `-mgpu`.** `sip.Matrix` does not implement
`ApplyBlockSatellite` (required at `lanczos/lanczos.go:167-170`), and `cmd/adcgo/sip_tdm.go:124-134`
has no lowmem case even though `validateSolver` (`cmd/adcgo/main.go:342-349`) accepts the string for
`-sip`. So a SIP sector that outgrows one device has no mitigation at all — none of the memory work
that made production DIP tractable was ported.

## A caveat on the number everything rests on

The 0.58%-of-fp64-peak figure (job `14015067`) was measured on the **per-scalar** kernel, before the
batched path became the default. Every priority argument downstream of it — including this
document's Tier 3 — should be re-derived from a fresh measurement on the current default path once
the validation job above has run.

## Order of attack

1. ~~Submit `scripts/test_jii_contraction_gpu.sbatch`~~ — done, passed (see above). Instead: re-run
   the timing job pinned to H200 (`--gres=gpu:H200:4`), since every number so far is from A40.
2. Tier 1 #1–#4 — all small, independent, and pattern-copied from code already in the repo.
3. `-Xptxas -v` plus an Nsight trace on the *current* path before anything in Tier 3.

Superseded by [`h200_optimization_plan.md`](h200_optimization_plan.md), which orders this backlog
into executable phases.
