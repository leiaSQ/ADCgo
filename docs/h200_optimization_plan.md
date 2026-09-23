# H200 optimization plan — SIP-ADC(3) / DIP-ADC(2) at production and uracil scale

**Status:** plan, 2026-07-23. Supersedes/operationalizes [`gpu_perf_backlog.md`](gpu_perf_backlog.md)
into ordered, verifiable work. Companions: [`dip_operator_memory.md`](dip_operator_memory.md),
[`sigma_build_contractions.md`](sigma_build_contractions.md), [`dip_lowmem_lanczos.md`](dip_lowmem_lanczos.md),
[`../internal/adc/backend/README.md`](../internal/adc/backend/README.md), [`../scripts/helix/HELIX.md`](../scripts/helix/HELIX.md).

## Thesis

Target: 8×H200 (141 GB, fp64 tensor cores: 67 TFLOP/s tensor / 34 vector). The measured ceiling —
0.58% of vector fp64 peak on the satellite apply (job `14015067`, per-scalar kernel, since
superseded as default by the batched path) — is not an arithmetic problem. It is orchestration:
the Go↔CUDA seam has no streams, no async transfers, no pinned memory, a serial NVLink gather
sitting directly beneath code that already knows how to fan out (`syncAll`), and a "batched" GEMM
path that un-batches itself one call at a time. **Kernels meshing well with the Go code is the
priority.** Fix the seam before the arithmetic — a faster kernel behind a serialized gather and
the null stream still waits on the same critical path.

SIP-ADC(3) gets first-class treatment because it is the weaker path today: no `-mgpu`, no
`lanczos-lowmem`, one coupling block with no matrix-free gate at all, and a serial assemble where
the identical DIP-side fix already shipped.

## Measure first

`backend/perf.go` `Calibrate` (`:92`) is a **pre-solve backend selector** — it times one
`ApplyBlock` to pick gonum vs. cuda before a sector runs. It is not a runtime profiler: nothing
reports `PeerCopy2D` bandwidth, `syncAll` latency, or the gather/kernel split within a chunk.
Every number in the companion docs was hand-instrumented per job.

| gap | missing | fix |
|---|---|---|
| per-chunk gather/kernel split | `dip/matfree_dist.go` apply loops have no timers around gather vs. `DipSatApply`/`DipSatFillJII` | `time.Since` behind `-profile`, matching `reportTiming` (`cmd/adcgo/sip_tdm.go`) |
| `PeerCopy2D`/`syncAll` latency | `gpu_device.go` `PeerCopy2D`/`Sync` (`:439,450-456`) are silent | same |
| register-spill visibility | `scripts/helix/build_adcgo_cuda_helix:62-67` has no `-Xptxas -v` | add it; `vint[31]` (`adc4_kernels.cu:44`) is a plausible spiller on `sm_90` |
| kernel occupancy/timeline | nothing wraps the binary | `nsys profile --trace=cuda,osrt -o <run> ./adcgo-cuda ...`; `ncu --set full --launch-count 3 ...` on `dip_sat_apply`/`dip_fill_sat`/`wert2_fwd`/`c22_apply` |
| the "0.58%" figure itself | measured on the **superseded per-scalar** kernel | re-run the timing job on the current default (`newSatBatchedDevice`/`newSatBatchedPerDevice`) before scoping Phase F |

`TestSatBatchedPerDeviceParity`/`TestSatelliteMatFreePerDeviceParity` passed on A40 (job
`14023720`), but that node's `nvidia-smi topo -m` shows **SYS** (PCIe, cross-socket), not NVLink —
a correctness signal only. **Before scoping Phase B4, get one `nsys` trace on an actual H200
node**; the gather design's whole performance argument assumes an NVLink/NVSwitch fabric not yet
observed on the target hardware.

## Phases

Ordered by dependency, not raw impact: pinned memory before async copies, the parallel-gather
pattern before streams make the overlap in B4 worth building.

### Phase 0 — Correctness validated; only the H200 *timing* gap remains (no code)

**The correctness gap is closed.** Two jobs on 2026-07-23 exercised the batched-GEMM satellite path
(`dip_fill_sat`, `newSatBatchedDevice`/`newSatBatchedPerDevice`) on **real CUDA backends** —
`matfree_batched_cuda_test.go:99` builds sub-backends with `backend.NewAll("cuda", …)`, not gonum
subs; gonum appears only as the host reference (`:34`):

| job | node | tests | result |
|---|---|---|---|
| `14023530` | p04c05, 1×A40 | `TestJIIFillDeviceMatchesHost`, `TestJIIMatFreeBatchedDeviceParity`, `TestSatelliteMatFreeDeviceParity` | PASS |
| `14023720` | o03c02, 4×A40 | `TestSatBatchedPerDeviceParity`, `TestSatelliteMatFreePerDeviceParity` | PASS, max\|Δ\| ≈ 5–8e-14 |

So `gpu_perf_backlog.md`'s "has never run on a GPU" is **stale** — it was written before these
landed. The fill kernel is pinned entry-by-entry against `mx.blk.jiiLKK`, which recovers the
per-element verifiability `sigma_build_contractions.md` warned a contraction rewrite would cost.

What remains is **performance**, not correctness: both nodes were A40 with `SYS` (PCIe +
cross-socket) between every GPU pair, so neither says anything about H200 or an NVLink/NVSwitch
fabric. Re-run the timing job pinned with `--gres=gpu:H200:4` (the script has no model pin;
pin the model explicitly in the `--gres` line) and take the `nsys`/`ncu` traces above on it.

- **Verify:** already green above; re-confirm on H200 alongside the timing run.
- **Effort:** none (submit + wait), but blocking — Phase F should not be scoped before real H200 numbers land.

### Phase B — The Go↔CUDA seam (top priority)

Cheap independent fixes first (B1–B3, pattern-copies of code already in the tree), then the
structural stream/event design (B4) they make worth doing, then two more seam items (B5–B6).

**B1. Parallelize the serial NVLink gather.** `dip/matfree_dist.go:110-126`
(`newSatelliteMatFreePerDevice`) and the identical loop at `:240-254` (`newSatBatchedPerDevice`)
run an 8×8 `for d { for src { PeerCopy2D } }` double loop *serially*, while `syncAll` fifteen
lines above (`:43-53`) already fans out with `wg.Go`. `PeerCopy2D` (`gpu_device.go:450-456`)
bottoms out in `devMemcpy2D`→`cudaMemcpy2D` (`cuda.go:216-218`) and round-trips through the
destination device's owning goroutine via `b.do()` (`:125-134`) — each of the 64 copies per chunk
fully blocks before the next issues. At the production system's 26 chunks/mat-vec that is ~1,664 serialized
transfers on a switch built for concurrent P2P. Fix: `wg.Go` the per-`d` outer body, same pattern
`syncAll` already uses. Copies, not arithmetic — bit-exactness-neutral.
- **Verify:** `TestSatelliteMatFreePerDeviceParity`, `TestSatBatchedPerDeviceParity` (≥4 peered GPUs), plus an `nsys` trace showing gather time drop.
- **Effort:** ~0.5 day — the pattern is fifteen lines above the bug.

**B2. Stop `distBackend.GemmMatBatched` un-batching itself.** `backend/distributed.go:516-520`
loops `gemmMatOne` once per block, reintroducing the dispatch tax batching exists to remove —
costed by `gpu_device.go:290-296` at 181 s of a 379 s formic-acid sector. Fix: bucket the
`a[i]/bb[i]/c[i]` triples by `(ownerOf(c[i]), shape)` (reuse `ownerOf`, `matfree_dist.go:159-166`),
gather each bucket's remote operands (existing `gemmMatOne` logic, `distributed.go:485-509`), and
issue one `sub.GemmMatBatched` per bucket. Preserves accumulation order only if blocks are
appended in `gemmMatOne`'s original order — assert with a dedicated ordering test.
- **Verify:** existing `distributed_test.go` coverage plus a new `TestGemmMatBatchedBucketingMatchesLoop` on gonum subs before trusting cuda.
- **Effort:** 1–2 days.

**B3. Pinned host memory.** `cuda.go:193-199,225-227` move through plain Go-heap slices via
synchronous `cudaMemcpy`; zero `cudaHostAlloc`/`cudaMallocHost` anywhere. Not a win alone, but the
**prerequisite** for B4: `cudaMemcpyAsync` on a non-default stream needs a pinned endpoint or it
silently degrades to synchronous. Fix: add `cudaHostAlloc`/`cudaFreeHost` to the cgo shim; route
the Davidson preconditioner buffer (Phase E) and the gather staging copies through it first.
- **Verify:** no correctness test (same bytes, different allocator); a pinned-vs-pageable H2D bandwidth microbenchmark is the acceptance check.
- **Effort:** ~1 day for the shim; call-site migration folds into B4/E.

**B4. Streams, events, and gather/kernel overlap.** Every device op funnels through
`gpuBackend.do()` (`:125-134`): one channel, one owning goroutine, one blocking round-trip per
call, on the **null stream** (`blasCreate()` never calls `cublasSetStream`). `Sync()` is
`cudaDeviceSynchronize` — a device-wide fence, not stream-scoped. This is *why* gather(chunk c+1)
cannot overlap kernel(chunk c): both share the synchronous channel, and `syncAll`
(`matfree_dist.go:105,129`, twice per chunk) is a hard barrier by construction.

Redesign, scoped to the satellite loop first (`matfree_dist.go:99-146`):
1. Second cuBLAS handle per `gpuBackend` bound to a **compute stream**, plus a **copy stream** for
   `PeerCopy2D`, both created once in `newGPUOn` (`:101-117`) alongside the null-stream handle.
   `PeerCopy2D`'s `cudaMemcpy2D` → `cudaMemcpy2DAsync` on the copy stream (device-to-device
   resolves fine under UVA without pinning; B3 only matters for host-staged fallback paths).
2. Replace `syncAll`'s device-wide sync with `cudaEvent_t`: record on the copy stream after the
   gather, `cudaStreamWaitEvent` on the compute stream — a stream-scoped dependency instead of a
   device-wide barrier. This is what makes overlap possible: chunk *c+1*'s gather can issue while
   chunk *c*'s kernel still runs, since they no longer share a sync point until actually needed.
3. **Go-side requirement.** `b.do()` assumes one synchronous op per round-trip; async needs
   *issue* (enqueue, return immediately) split from *wait* (sync an event) — two verbs where
   there is one today. Split `PeerCopy2D`/`DipSatApply` into an `Issue` that enqueues and returns
   a handle, and keep `Sync` for the explicit wait (before the next `Download`, or before freeing
   a buffer a copy still reads). The owning-goroutine model (`:19-22`) stays; its jobs just become
   "enqueue, don't block" on the issue side.
4. Double-buffer the gather slab (`matfree_dist.go:96`, one `slab[d]` per device today): two
   slabs, ping-ponged per chunk, so chunk *c+1*'s gather does not write what chunk *c*'s kernel
   reads. Doubles slab VRAM (~15.2 GB at `w=64`, production triplet, 8 devices — small against
   141 GB/card) and is what actually unlocks step 2's overlap.
- **Bit-exactness:** unaffected — changes *when*, not *what*/*order* (one thread per row still owns its output accumulation, per `matfree_dist.go`'s existing determinism argument).
- **Verify:** `TestSatBatchedPerDeviceParity`/`TestSatelliteMatFreePerDeviceParity` must stay bit-identical; a new chunk-wall-time benchmark with/without overlap, gated on Phase 0's trace confirming NVLink (not PCIe/SYS) is the actual bottleneck class here.
- **Effort:** 1–2 weeks — a real concurrency redesign, not a local patch. Start only after B1–B3 land and Phase 0's H200 trace confirms the gather (not the kernel or other host round-trips) dominates.

**B5. Batch the α/Gram/CGS2 all-reduce.** `backend/distributed.go:322-333`
(`Gemm(transA=true,...)`) does a per-device `Download`, a host sum, then per-device
`Upload`+`Copy`+`Free` — one `Alloc`/`Free` pair per call, called several times **per Lanczos
iteration**, not per mat-vec. Fix: `wg.Go` the download/upload legs (same pattern as B1), and
hoist a persistent scratch buffer out of the per-call path (mirrors the `ptrA/ptrB/ptrC`
grow-on-demand pattern already used for batched-GEMM pointer arrays, `gpu_device.go:56-58`).
- **Verify:** `distributed_test.go`'s existing `Gemm(transA=true)` coverage; a per-iteration reduction benchmark.
- **Effort:** ~1 day, independent of B1–B4.

**B6. Raise `satChunkCols`.** Hardcoded 64 at `dip/matfree_dist.go:36` (mirrored
`cmd/sizeprobe/main.go:23`). The measured production triplet budget (`dip_operator_memory.md`, Phase
D) shows ~26 GB of headroom at `-mgpu 8` beyond the `w=64` slab — room for 128, halving both
recompute overhead (~15%→~8%) and the barrier count B1/B4 touch. One-line change, but sequence
after B1 (halving 1,664 serialized transfers is a smaller win than parallelizing them) and
re-measure VRAM headroom after B4.5's double-buffering, which doubles slab residency.
- **Verify:** `dip/matfree_dist_test.go` row-band tests; `cmd/sizeprobe` re-run to update the `dip_operator_memory.md` budget table.
- **Effort:** trivial, sequenced last for the reason above.

### Phase C — SIP-ADC(3) first-class treatment

No `-mgpu`, no `lanczos-lowmem`, a serial assemble where DIP's identical fix already shipped, and
one coupling block that is unconditionally dense. Cheapest/lowest-risk first.

**C1. Parallelize `coupling()`/`satBlock()`.** `sip/matvec.go:98-126` are plain serial loops;
`sip/matvec4.go:66,97,113` (`coupling2_4`, `satBlock2_4`, `coupling24_4`) wrap the **identical
row-independent shape** in `parallel.Rows` for the ADC(4) path — a direct pattern-copy. Runs
lazily inside `ApplyFull`'s first call (`matvec.go:277-280`), inflating whatever "apply" bucket
wraps SIP's first iteration — note this when re-measuring Phase 0's numbers.
- **Verify:** existing dense-matrix tests (`TestSIPMatchedReference` etc.) — `parallel.Rows` gives each worker a disjoint output row, no reduction, **bit-identical** by construction (same argument `dip_operator_memory.md` makes for its parallel walk).
- **Effort:** ~1 hour.

**C2. Gate the 1h×2h1p coupling block matrix-free.** `sip/matvec.go:139-146` (`assemble()`)
uploads `coupling()` unconditionally whenever `main>0 && nSat>0` — no `matFree` check, unlike
every other block in this file and in `matvec4.go`. At production scale (`nSat`≈5×10⁵) that is
`main·nSat·8` bytes materialized regardless of `-matfree`. Port the `matfree.Decide`-gated pattern
already used for `matFreeC22O3`/`matFreeWert2` (`sip/matfree.go:59-81`): add `matFreeCoupling` and
a host `newCouplingMatFree()` (candidate-bucket structure mirrors `newWert2MatFree`,
`matfree.go:88-169` — c12 is δ-sparse on the shared occupied index too). A device kernel is a
stretch goal, lower priority than C3/C4 (the block is main×nSat, not nSat²).
- **Verify:** new `TestCouplingMatFreeMatchesDense` (host) mirroring `TestSatelliteScalarMatchesDense`'s per-element approach; `TestMatFreeMatchesDenseSolve`-style end-to-end check.
- **Effort:** 2–3 days for the host applier; device kernel is separate/later.

**C3. Port `-mgpu` to SIP.** Currently DIP-only (`main.go:85`'s flag text: `"-dip -solver
lanczos-lowmem -lowmem-block 0 only"`).

> **Sequencing correction: C4 gates C3.** `main.go:563-565` rejects `-mgpu` outright unless
> `-solver lanczos-lowmem`. Since SIP has no lowmem path at all, `-mgpu` is unreachable from
> `-sip` until C4 lands — so **do C4 first**, despite C3 appearing earlier here. The alternative
> (relaxing the `-mgpu`/lowmem coupling for SIP) means partitioning under the full-reorth
> `Solve()`, whose `n×maxdim` resident basis is exactly what lowmem exists to avoid — not worth it. SIP's `n≈518k` is far smaller than DIP's production sectors
(`dip_lowmem_lanczos.md`), so this is less urgent for memory — but a larger active space (more
virtuals, no CVS truncation) pushes `nSat` up fast, and "large systems" is the stated goal.
`backend.PartitionedDevices`/`PanelScatterAdd` (implemented by `distBackend`,
`distributed.go:377-423,341-368`) are backend-level, not DIP-specific. Porting needs: (1) a
row-partition-aware `ApplyBlock` on `sip.Matrix` composing with `distBackend`, transcribing
`dip.Matrix`'s `-mgpu` `ApplyBlock` pattern (block placements differ; row-partitioning does not);
(2) reusing `dip.PartitionBounds`'s group-alignment for SIP's 2h1p space, so a boundary never
straddles a block — needed for C2's matfree coupling under `-mgpu`; (3) `cmd/adcgo` plumbing —
`-mgpu` reachable from `-sip` (gated to `-dip` at `main.go:561-567`), and `checkSubsFit` (already
generic, `dispatch.go:152-...`) wired as `main.go:477` does for DIP.
- **Verify:** `TestSatelliteMatFreeDistributedEqualsDense`-style host coverage adapted for SIP; on-hardware parity mirroring `dip/matfree_mgpu_cuda_test.go`.
- **Effort:** 1–2 weeks — a genuine port, though every primitive it needs exists on the DIP side.

**C4. Port `ApplyBlockSatellite` + `lanczos-lowmem` to SIP.** `sip.Matrix` does not implement
`lanczos.SatelliteOperator` (`lanczos/lanczos.go:167-170`), and `sip_tdm.go:122-134`'s solver
switch has no `"lanczos-lowmem"` case, though `validateSolver` (`main.go:342-349`, shared with
`-sip` via `runSIP`, `main.go:770`) accepts the string — so `-sip -solver lanczos-lowmem` fails
late inside `solveSIPSpace` with a generic "unknown solver" error instead of at flag parse.
**Cheap fix, do first:** reject `lanczos-lowmem` for `-sip` explicitly until the real port lands.
**The real port:** implement `ApplyBlockSatellite` on `sip.Matrix` — apply only the
satellite↔satellite block(s), skipping main and the (C2-matfree-eligible) coupling block,
transcribing `dip.Matrix`'s block-selection logic (physics differs, contract does not). This is
what `SolveLowMem`'s Tarantelli subspace-iteration gate needs (`dip_lowmem_lanczos.md`, Mode B).
Given SIP's much smaller `n` (518k vs. DIP's 10–15M), Mode A (device-frugal, block < main) may be
usable for SIP where it was ruled structurally incomplete for DIP's full band — check the Mode
A/B findings there before assuming Mode B (fat-CPU-node) is the only target.
- **Verify:** mirror `TestSolveLowMemModeB_MatchesDense`/`TestSolveLowMemModeA_FullExact` for SIP; satellite gate vs. a masked dense operator, mirroring `dip/satellite_test.go`.
- **Effort:** 1–2 weeks, gated on C1+C2 (the satellite-only apply needs the coupling matfree gate to stay cheap under `-lowmem-block 0`, where panel width equals the full main space).

**C5. SIP pre-flight guard.** `sip_tdm.go:85-141`'s `solveSIPSpace` never calls `checkDeviceFit`
(exists at `dispatch.go:130-150`; DIP's `solveDIPSector` calls it, `main.go:399`). `pickLanczos`'s
own check (`chooser.fits`, `dispatch.go:112-129`) is skipped whenever there is one pinned backend
(`chooser.single()`, `:102-107` — true in production for `-backend cuda`), so an oversized SIP
sector hits a raw `cudaMalloc` panic. Fix: add the `checkDeviceFit` block `solveDIPSector` has
(`main.go:391-401`), sized from `backend.SectorBytes(n, dim, b)` (accurate for SIP's
`lanczos`/`dense` solvers, unlike DIP's matrix-free-operator case).
- **Verify:** a new test forcing a device-memory-too-small condition (mock `DeviceMemory`), asserting a clean error instead of a panic.
- **Effort:** ~2 hours — direct port of an existing call site.

### Phase D — Kernel loop-order and hygiene fixes (bit-exact, zero-risk, independent of B/C)

Touches only `.cu` files, individually zero-risk, no dependency on the seam or SIP work; batch
into one rebuild/validation cycle. Combined effort ~2 days incl. rebuild + hardware verification.

**D1. Swap the `wert2_fwd`/`wert2_trans` loop nest.** `adc4_kernels.cu:91-106`/`:109-123` put `j`
(panel column) **outer** and call `d_wert2` — 31-entry local array, ~16 conditional ERI loads, a
30-term dot product — **inside** it, though the element `g` has no `j` dependence. `c22_apply`
(`:231-250`) has the correct nest: element computed once per `c` (`:237,240,242`), applied over
all `j` in the inner loop (`:247-249`). SIP passes the panel width `b` unchunked
(`sip/matfree.go:314,380`), so `d_wert2` is evaluated ~`b`-fold more than necessary — at the production system's
`main≈1653`-scale widths, ~1653× per element. Swap the nest to match `c22_apply`. **Preserves
summation order over `c`/`r`** — only *when* each term computes changes — bit-identical. Verify
against existing GPU parity tests for the wert2 path on hardware; the
`TestSatelliteScalarMatchesDense`-equivalent host oracle stays authoritative regardless.

**D2. Add `__restrict__` to read-only kernel pointers.** Absent from every read-only pointer in
both `.cu` files (`grep -c __restrict__` = 0 for both). Zero risk, and its absence is part of why
nvcc cannot hoist D1's redundant load on its own (cannot assume `eri`/`eps`/`osym` unaliased with
`yout`). Add to `eri`, `eps`, `osym`, and every SoA pointer in `wert2_fwd`, `wert2_trans`,
`c22_apply`, `dip_sat_apply`, `dip_fill_sat`. Verify: bit-identical output on any existing kernel
test suite — a pure compiler hint, not a semantic change.

**D3. Measure register spilling.** `scripts/helix/build_adcgo_cuda_helix:62-67` has no `-Xptxas -v`.
`vint[31]` (`adc4_kernels.cu:44`, inside `d_wert2`) — 248 bytes of live state per thread — is the
standout candidate for spilling to local memory on `sm_90`, even before D1 changes call frequency.
Add the flag, capture `ptxas info` register/spill counts in the Phase 0 job log, and treat any
nonzero spill count on `d_wert2`/`d_jii_s`/`d_jii_t`/`d_ijkLMN_s`/`d_ijkLMN_t` as a Phase F
candidate (shrink the local array via early δ-gate termination, not a general rewrite).
Informational only — feeds Phase F prioritization, no correctness test.

### Phase E — Davidson preconditioner overlap

`lanczos/davidson.go:200-231`: every iteration downloads the full `n×nw` residual
(`rHost := be.Download(cbuf)`, `:200`), runs a **serial** host loop with a **per-column
`make([]float64, n)`** (`:207`), then re-uploads (`:231`). Davidson is the production DIP driver
(block-Davidson, lowest `-nroots` — the actual production runs, per `HELIX.md`), so this runs every
iteration. **Cheap:** `parallel.Rows`/`parallel.Chunks` over `k` (each worker owns disjoint output
columns — `col[j] = rHost[k*n+j]/a1` has no cross-column dependency); hoist the per-column `make`
into one `n*nunc`-sized buffer, pre-sized instead of `append`-grown. **Better, needs B3:** route
`rHost`/`cor` through pinned buffers so B4's async infrastructure can overlap this download/upload
with other device work instead of blocking every iteration.
- **Verify:** existing Davidson convergence tests (`TestSolveDavidson`-class); a reassociation-free elementwise transform, bit-exact by construction — no new tolerance needed.
- **Effort:** ~1 day for step 1; step 2 folds into B4.

### Phase F — Structural / contraction rewrite (gated behind Phase 0's fresh measurement)

**Do not start until Phase 0 re-measures the current default (batched) path's fraction of fp64
peak.** `sigma_build_contractions.md`'s own rule: ≳20% of peak → prefer cheaper kernel work
(Phase D, device-side candidate buckets); ≲5% → the contraction case is strong; in between →
reassess against production walltime. The 0.58% figure was measured on the **superseded per-scalar**
kernel — re-derive it on `newSatBatchedDevice`/`newSatBatchedPerDevice` first.

- **Extend batched-GEMM to `ijkMLL`/`ijkLMN`.** Only `jiiLKK` is batched on host
  (`matfree_batched.go`); the device fill kernel (`dip_fill_sat`, `adc2dip_kernels.cu:373-417`)
  already covers all three block kinds per `gpu_perf_backlog.md` — confirm against
  `matfree_batched.go`'s current dispatch before scoping new work; may be closer to done than the
  host-only framing suggests.
- **Warp-per-row `dd_eri` reorder.** `dd_eri` (`:26-28`) puts `ra` (virtual orbital) outermost,
  while rows enumerate virtual-innermost (`dip/satscalar.go:59-85`), so consecutive threads gather
  ERI elements ~76 MB apart at the production system's `norb=212`. The candidate scan (`:288` `njii` loop and the
  `nijk` loop below it) is **warp-uniform** — consecutive threads share every occupied-index
  comparison — amortized overhead, not divergence; the gather stride is the real candidate. A
  warp-per-row decomposition (cooperative load, shuffle-reduce) is the fix, but needs a
  **deterministic** shuffle tree to hold the bit-reproducibility bar `adc4_kernels.cu:8-12` sets —
  fp addition is not associative, so an arbitrary reduction tree changes the last bit. **Flag this
  loudest**: get it wrong and the answer silently drifts ~1e-14 per apply, compounding over
  hundreds of Lanczos iterations.
- **cuTENSOR.** Per `sigma_build_contractions.md`: not before cuBLAS-batched-GEMM is exhausted and
  only if permutation overhead then shows up in an `ncu` trace. Not speculative.
- **Verify:** `TestSatelliteScalarMatchesDense`, `TestSatelliteScalarApplyEqualsDense`,
  `TestJIIBatchPlanMatchesGateWalk`, `TestJIIMatFreeBatchedEqualsLoop`, `TestJIIBatchedSymmetryOff`,
  `BenchmarkBlockApplyCrossover`/`BenchmarkBlockBuildVsApply` as the baseline a device change must beat.
- **Effort:** multi-week per sub-item; do not commit until Phase 0's number is in hand.

## Sequencing summary

```
Phase 0 (measure current default) ──┬──▶ Phase B1–B3 ──▶ Phase B4 (streams) ──▶ Phase B6
                                     ├──▶ Phase C1 ──▶ Phase C2 ──▶ Phase C4 (needs C1+C2)
                                     │                └──▶ Phase C3 (indep. of C2/C4)
                                     ├──▶ Phase C5 (trivial, independent)
                                     ├──▶ Phase D1–D3 (independent of everything)
                                     ├──▶ Phase E (independent; step 2 folds into B4)
                                     └──▶ Phase F (gated on Phase 0's number, not before)
```

B5 is independent of B1–B4. Phases C, D, E touch disjoint files from Phase B and from each other,
so they proceed in parallel once Phase 0's numbers are in, subject to review bandwidth.

## Effort, risk, and bit-exactness

| item | effort | risk | preserves accumulation order? |
|---|---|---|---|
| 0 — validate default path on H200 | none (submit + wait) | blocks Phase F scoping | n/a |
| B1 — parallelize gather | ~0.5 day | none | yes — copies, not arithmetic |
| B2 — fix `GemmMatBatched` un-batching | 1–2 days | low (needs ordering test) | yes, *if* append order is asserted |
| B3 — pinned memory shim | ~1 day | none | n/a |
| B4 — streams/events + double-buffer | 1–2 weeks | medium (real concurrency redesign) | yes — changes *when*, not *what*/*order* |
| B5 — batch α/Gram/CGS2 reduce | ~1 day | none | yes — same partials, same sum order |
| B6 — raise `satChunkCols` | trivial | none (re-measure VRAM headroom first) | n/a |
| C1 — parallelize SIP coupling/satBlock | ~1 hour | none | yes — disjoint output rows, no reduction |
| C2 — matfree-gate SIP coupling | 2–3 days | low | yes, if built like `newWert2MatFree` |
| C3 — port `-mgpu` to SIP | 1–2 weeks | medium | n/a |
| C4 — port `ApplyBlockSatellite` + lowmem to SIP | 1–2 weeks | medium (gated on C1+C2) | n/a |
| C5 — SIP pre-flight guard | ~2 hours | none | n/a |
| D1 — wert2 loop swap | ~1 day | none | yes — same summation order over `c`/`r`, computes `g` once instead of `b` times |
| D2 — `__restrict__` | ~2 hours | none | yes — compiler hint only |
| D3 — `-Xptxas -v` + spill audit | few hours | none (informational) | n/a |
| E — Davidson preconditioner overlap | ~1 day (+ B4 for step 2) | none | yes — elementwise, no cross-column dependency |
| F — contraction/warp-per-row rewrite | multi-week, per sub-item | **high** on the shuffle-reduce sub-item | **no, unless the reduction tree is fixed and asserted** — fp addition is not associative |
