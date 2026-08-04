# The σ-build: per-scalar recompute vs tensor contractions (and where cuTENSOR fits)

**Status:** the gating measurement is DONE and clears the bar — see
[MEASURED](#measured-2026-07-23--the-gate-is-cleared). Implementation has started at step 1
(benchmarks only; no production code changed yet). Written 2026-07-22, prompted by the question
"would cuTENSOR help instead of Lanczos?". Companion to
[`dip_operator_memory.md`](dip_operator_memory.md) (why the operator is matrix-free) and
[`dip_lowmem_lanczos.md`](dip_lowmem_lanczos.md) (why the basis is not resident).

## The short answer

cuTENSOR is not an alternative to Lanczos. They sit at different layers, and conflating them hides
that there are **three independent problems**, each with its own solution:

| wall | production triplet (n = 14,766,249) | what answers it |
|---|---|---|
| can't **store** the matrix | sparse operator ≈ **13 TB** | matrix-free σ-build (shipped) |
| can't **diagonalize** it | dense form would be `n²·8` ≈ **1.7 PB**; O(n³) ≈ 3×10²¹ flops | Krylov / block Lanczos (shipped) |
| the σ-build is **slow** | ≈ 4–6 PFLOP per block mat-vec | ← this note |

Two things worth pinning down, because they are easy to garble:

- **13 TB is the *sparse* operator, not the dense one.** The dense matrix would be ~1.7 PB; the
  nonzeros are ~0.8% of it. Both are unstorable, but only the 13 TB figure appears in the sizing
  docs and it is a nonzero count, not `n²`.
- **Even if it fit, Lanczos would still be required.** Dense diagonalization is O(n³) — about
  3×10²¹ flops at the production system's n. Storage is not the only reason the solver is iterative, so no
  amount of memory or tensor-library cleverness removes the need for a Krylov method.

cuTENSOR addresses none of the above. It is a candidate tool for the third row only.

## Where contractions could genuinely help

The satellite kernel currently evaluates **one operator element at a time**. From
`backend/adc2dip_kernels.cu`, with row `(i,j,k,ra)` and column `(l,m,n,sb)`:

```cuda
if (dIL && dJM) e += C(...)*AA(k,nn) + C(...)*BB(k,nn);   // AA(x,y) = eri[ra,x,sb,y]
```

The Kronecker deltas pin `i=l, j=m`. Summing that term's contribution over the *free* column
indices collapses it to

```
σ[i,j,k,ra] += Σ_{n,sb}  coeff · eri[ra,k,sb,n] · c[i,j,n,sb]
```

— a contraction over two indices, batched over occupied pairs `(i,j)`. That is a **batched GEMM**.
Every δ-gated term in `dip/satelem.go` has this shape: the deltas are not an obstacle to
contraction, they are what *selects which* contraction each term is.

Expected gain, with the efficiency fractions stated as assumptions rather than measurements:

| | fraction of fp64 peak | note |
|---|---|---|
| current per-scalar kernel | ~2–5% *(assumed)* | one thread per output row, scattered 8-byte ERI gathers, divergent candidate lists |
| batched fp64 GEMM on H200 | ~50–70% | H200 has real fp64 tensor cores (67 TFLOP/s; 34 vector) |

Net is **not** the ratio of those: the contraction form densifies, giving up some δ-sparsity and
doing more raw flops. Realistically **~5–15×**.

## The uncomfortable part

`dip_operator_memory.md` already records that theADCcode is fast because it "forms σ on the fly by
**contracting the MO integrals per block**". The reference implementation *already uses the
contraction form*. ADCgo's per-scalar kernel is, algorithmically, a step backwards from it.

That was very likely the right trade at the time, and should not be read as an oversight: the
per-scalar shape is verifiable element-by-element against the dense blocks
(`TestSatelliteScalarMatchesDense`, `TestSatelliteScalarApplyEqualsDense`), which is exactly what
made the CUDA port cheap to trust — the host tests fix the physics, the GPU test fixes only the
transcription. A contraction rewrite gives that up: correctness would then rest on whole-block
numerical agreement, not per-element identity. The performance ceiling was traded for
verifiability at design time; it was not lost through cuTENSOR's absence.

## cuTENSOR specifically

Separate the two questions, because most of the win is not cuTENSOR's:

- **per-scalar recompute → contractions**: the actual prize. Achievable with cuBLAS batched GEMM,
  which is already linked.
- **cuBLAS batched GEMM → cuTENSOR**: second-order. cuTENSOR fuses the index permutations that
  would otherwise need explicit transposes between contractions. Real, but a refinement of the
  first step, not a substitute for it.

So: do not adopt cuTENSOR *in order to* get faster. Reformulate as contractions first; adopt
cuTENSOR only if permutation overhead then shows up in a profile.

## MEASURED (2026-07-25) — the contraction path is SLOWER than per-scalar for uracil2W

> **A direct A/B on 4×H200 contradicts the premise below.** Job `14040958`, uracil2W_dz singlet,
> same sector as the gate measurement, comparing the two device paths end to end:
>
> | path | wall | apply | note |
> |---|---:|---:|---|
> | per-scalar (host gather-scatter, `newSatelliteMatFree*`) | **278 s** | 213.7 s (96.8%) | recompute-in-register, nothing materialized |
> | contraction (per-device batched, `newSatBatchedPerDevice`) | **485 s** | 419.3 s (98.3%) | materializes the ~424 GB operator, chunked, then GEMMs it |
>
> The contraction path is **1.75× slower**, not the projected 5–15× faster. Mechanism: the batched
> GEMM densifies and *materializes* the block-sparse operator (hundreds of GB of writes, then reads
> it back), whereas the per-scalar kernel recomputes each element in registers and never
> materializes. For a memory-bound apply the materialization traffic dwarfs the recompute it
> replaces, and at uracil2W's panel width (b=435) the GEMM's b-fold reuse does not amortize it.
> Bounded chunking (JIIFillBudgetElems) is what lets it *fit* at all — the unchunked fill OOM'd at
> 424 GB (job 14026481) — but chunking per column-chunk also adds refill cost.
>
> **What is NOT yet known:** whether the production system's b=1653 crosses the amortization over (the crossover
> argument in "A measurement-validity trap", below, says larger blocks favour BLAS). The production system
> whole-band DIP run (job 14040960, `-matfree on`) is the first real test. Until it reports, the
> contraction path should not be assumed faster, and it should NOT be an unconditional default —
> the per-scalar path wins on every measurement so far.

## MEASURED (2026-07-23) — the gate is cleared

Job `14015067`, uracil2W_dz singlet, 4×H200, per-device NVLink path (i.e. *after* the 12.6× win):

| | |
|---|---|
| per mat-vec | 68.0 s (203.88 s / 3 blocks); `apply` = **96.6%** of solver wall time |
| FLOP/mat-vec | 53.4 TFLOP (43.0 apply + 10.4 recompute over 7 chunks) |
| achieved | 0.785 TFLOP/s over 4 GPUs = **0.196 TFLOP/s per GPU** |
| **fraction of fp64 peak** | **0.58% vector** / 0.29% tensor-core |

That is an order of magnitude below the ≲5% branch below, so **the case is strong**. It also
reframes the work: the production system whole-band DIP extrapolates to singlet 113 h + triplet 236 h = **349 h
against a 120 h walltime**, so contractions are plausibly what make the run feasible at all, not
merely faster.

**Host-side baseline** (`internal/adc/dip/bench_satellite_test.go`, added so this stops requiring a
cluster job — sub-second, runs on any machine):

- `BenchmarkSatelliteApply` on the largest h2o_dzp sector: 0.048 / 0.88 / 2.07 GFLOP/s at
  b = 1 / 64 / 435.
- `BenchmarkBlockApplyCrossover` — **BLAS beats the hand-written `gemvForward` loop at every block
  size tested**, and there is no crossover to design around: 1.10× at dim=8, 1.76× at 16, 3.18× at
  64, **3.84× at 154** (the production system's `sizeVirGroup`), 8.12× at 256.
- `BenchmarkBlockBuildVsApply` — the matrix-free cost splits **build 63.8 µs / apply 246.9 µs** at
  b=64, i.e. apply is 79%. Apply scales with panel width and build does not, so at the production system's
  b=1653 the apply share is ~99%. Amdahl therefore does **not** cap the rewrite: it targets the
  dominant half.

**A measurement-validity trap worth recording.** `TestSatelliteBlockShapes` shows h2o_dzp's jiiLKK
blocks are only **2..11 on a side** (mean 17–80 elements), because its 25 virtuals are split across
4 irreps. The production system is C1 — *one* irrep, nvir=154 — so its blocks are 154×154 ≈ 23,716 elements,
~300× larger. Any conclusion drawn from a whole-apply benchmark on a small symmetric test system is
therefore an artefact of the test system, not evidence about production. That is exactly why the
crossover benchmark sweeps synthetic block sizes across both regimes instead.

## Implementation status (steps 1–4 done, host only)

`internal/adc/dip/matfree_batched.go` routes the **jiiLKK** half of the matrix-free satellite
apply through `GemmMatBatched` instead of the `gemvForward`/`gemvTranspose` loops. It reuses
`backend.PlanBatches` unchanged — the same machinery the dense path uses — which also subsumes the
manual forward/transpose split, since a `gr==gc` block is exactly `backend.Block{Diag: true}`.

The enabling separation: **block existence and shape are static per sector** (the gates are pure
functions of two `Config`s), so the plan is built once with a shape-only `DeviceMat` stand-in and
only *values* are refilled per mat-vec.

Wired in for **host backends only** (`newSatelliteMatFreeParts`): batched jiiLKK + loop remainder.
Device, distributed and `PanelScatterAdd` backends keep their existing single applier untouched.

Validated:

- `TestJIIBatchPlanMatchesGateWalk` — plan block-set/shapes/diag flags cross-checked against an
  independent gate walk, **plus** that each block is applied exactly once (diagonal) or twice
  (off-diagonal), **plus** batch write-offset disjointness. This is the mitigation for the
  double-count/drop failure mode, which would otherwise surface only as small numeric drift.
- `TestJIIMatFreeBatchedEqualsLoop` — batched vs loop across the spin × 4-irrep sweep, and asserts
  main-space rows are never written (they must stay *literally* zero).
- `TestJIIBatchedSymmetryOff` — the symmetry-off regime (one virtual group, large uniform blocks,
  deep batches) that production actually runs in and the symmetric sweep never reaches.
- Whole suite green; **`TestMatFreeMatchesDenseSolve` drift went 1.95e-14 → 1.15e-14**, i.e. the
  reassociation introduced no measurable accumulation error.

Measured, host, h2o_dzp: `BenchmarkJIIApplyLoopVsBatched` = **1.03×**. That is the *pessimistic
floor*, not the expected production figure — this sector's blocks are 2..11 on a side, where the
crossover benchmark puts BLAS at only ~1.1×, further diluted by the unchanged block-build cost.
The production expectation rests on the crossover curve (3.84× at the production system's dim=154) plus the
build/apply split (~99% apply at the production system's b=1653), not on this number.

**This does not yet change any production run.** Production DIP runs `-backend cuda`, whose device
applier is untouched — the host path only affects `-backend gonum`. Steps 5–7 (the `DipSatFillJII`
device kernel) are what would move the production number.

## Decide it with a number

**Do not scope a σ-build rewrite until the current kernel's efficiency is measured.** The whole
case rests on the "~2–5% of peak" assumption above, which nobody has checked.

Job `14015067` (`scripts/uracil2W_mgpu_timing.sbatch`) produces the first real per-mat-vec time for
the satellite apply. Convert it to a fraction of peak, then:

- **≳20% of peak** — a contraction rewrite buys maybe 3×, and is probably not worth losing
  per-element verifiability over. Prefer cheaper kernel work: device-side candidate buckets
  (replacing the linear group scan at `adc2dip_kernels.cu:279,305`), and raising `satChunkCols`.
- **≲5% of peak** — the case is strong. Port one block first (the JII↔JII term is the simplest),
  validate it against the existing dense reference, and keep the per-scalar path as the oracle.
- **in between** — reassess against how much walltime the production run actually needs.

## Pointers

- per-scalar element evaluation (the thing a rewrite would replace):
  `internal/adc/dip/satelem.go`, `satscalar.go`; kernel `internal/adc/backend/adc2dip_kernels.cu`.
- correctness oracles a rewrite must keep passing: `TestSatelliteScalarMatchesDense`,
  `TestSatelliteScalarApplyEqualsDense`, `dip/matfree_test.go`.
- why the operator is matrix-free at all, and the measured production budget:
  [`dip_operator_memory.md`](dip_operator_memory.md).
- how a block is split across devices today: [`../internal/adc/backend/README.md`](../internal/adc/backend/README.md).
