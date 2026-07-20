# DIP/SIP GPU memory — the resident block-sparse operator is the real wall

**Status:** IMPLEMENTED (host path). The fix this note proposed — a matrix-free
3h1p↔3h1p satellite apply, plus a parallel operator walk — has shipped for the
host backend; see [Implementation](#implementation-shipped) at the end. The
analysis below is kept as the (still-correct) statement of the problem. Companion to
[`dip_lowmem_lanczos.md`](dip_lowmem_lanczos.md). That note fixed the *Krylov
basis* cost (the low-memory Lanczos keeps only ~4 panels resident instead of the
whole basis). This note records the wall that shows up **once the basis is no
longer the problem**: the materialized block-sparse ADC operator itself, which
for large satellite spaces is hundreds of GB to multiple TB — bigger than any
single GPU, and for melanin bigger than a whole 8×H200 node.

Written 2026-07-20 while sizing the uracil / melanin DIP runs on bwForCluster
Helix (see `scripts/HELIX.md`, `scripts/uracil_dip.sbatch`,
`scripts/melanin_dip_mgpu.sbatch`).

## The wall in one line

For a whole-band DIP-ADC(2) sector the resident device footprint is

```
need = 4·n·main·8   +   OperatorResidentBytes()
       └── Lanczos panels ──┘   └── block-sparse ADC operator ──┘
```

and **the operator term dominates by 30–80×**. It does *not* shrink with the
Lanczos band width (`-lowmem-block`), because the band only sizes the panels.
The only lever we have today is `-mgpu N`, which row-partitions the operator
across N GPUs — but N ≤ 8 on one NVLink node, so it buys at most an 8× division.

## What we measured

Exact figures, from the same estimator the solver's pre-flight guard uses
(`4·n·main·8 + Matrix.OperatorResidentBytes()`, `internal/adc/dip/matvec.go`),
run offline on each FCIDUMP:

| system / sector          |          n | main | panels  | **operator** | total need | verdict                    |
|--------------------------|-----------:|-----:|--------:|-------------:|-----------:|----------------------------|
| uracil1W DZP · singlet   |    681,525 |  325 |  6.6 GB | **532.8 GB** |   539.3 GB | needs `-mgpu ≥4`           |
| uracil2W DZP · singlet   |  ~1,226,000 | 435 | ~17 GB  | **~1518 GB** | 1535 GB¹   | does not fit 8×H200        |
| uracil2W DZ+fc · singlet |    625,675 |  435 |  8.1 GB | **395.3 GB** |   403.4 GB | `-mgpu ≥4` (100 GB/GPU)    |
| uracil2W DZ+fc · triplet |    907,004 |  406 | 11.0 GB | **847.3 GB** |   858.3 GB | `-mgpu 8` (107 GB/GPU)     |
| melanin DZP · singlet    | 10,014,483 | 1711 | 510.7 GB| **~6 TB (est.²)**  | ~6.5 TB² | does not fit 8×H200        |
| melanin DZP · triplet    | 14,766,249 | 1653 | 728 GB  | **~13 TB (est.²)** | ~14 TB²  | does not fit 8×H200        |

¹ Observed directly from the failed run (job 13996569); the operator is the total
minus the ~17 GB panels.
² **Estimated, not measured.** The exact melanin operator walk was abandoned: it
runs single-threaded (see the drag note below) and had not finished even the
singlet sector after ~1.5 h of wall time, so it is not worth a multi-hour job for
a number that only confirms "does not fit". The measured operator is ≈ **600–930 KB
per configuration** across every row above; at melanin's n that gives **~6 TB
(singlet) / ~13 TB (triplet)** — i.e. ~5–12× past the 1128 GB aggregate of a full
8×H200 node, *before* the 0.5–0.7 TB of panels. An 8-GPU whole-band melanin DIP
therefore cannot fit and would `cudaMalloc`-panic mid-assembly (the `-mgpu` path
has no pre-flight guard). This is why `melanin_dip.sbatch` uses block-Davidson
(lowest `-nroots` only) for the real runs, and why the whole band needs the fix
below.

> **Aside — the operator walk is single-threaded.** `visitBlocks` /
> `OperatorResidentBytes` (and, by extension, `assemble()`) evaluate every operator
> block serially on one core. That is tolerable at uracil scale (a few minutes) but
> becomes a real drag at melanin scale (n ≈ 10–15 M): sizing alone runs into hours,
> and the same serial walk sits on the critical path of the *actual* solve's first
> apply. The blocks are independent, so this walk is embarrassingly parallel —
> worth a worker pool (or the matrix-free rewrite below, which removes the resident
> assemble entirely). Flagged here so the drag is on record, not mistaken for the
> memory problem it sits next to.

**Takeaways:**
- Panels are cheap and already `-mgpu`-partitionable; the operator is the ceiling.
- Banding (`-lowmem-block b`) is not a memory lever here — at b=10% of `main`,
  uracil1W's need is still 533 GB.
- Dropping polarization (DZP→DZ) + frozen core cut uracil2W's operator ~4× (1518→847 GB),
  enough to fit `-mgpu 8`. It does **not** scale down enough for melanin.

## Why ADCgo pays this and theADCcode does not

ADCgo **materializes** the operator: `Matrix.assemble()` walks every nonzero
block (2h/2h main, 2h↔3h1p couplings, 3h1p↔3h1p satellite blocks), uploads each
as a device matrix, and reuses the resident copy for every mat-vec. This is a
deliberate speed choice — the applies become dense/batched GEMV/GEMM — but it
pins the entire block-sparse matrix in device memory.

theADCcode (`adc2dip … lanczos`) runs the identical physics on far smaller
machines because it is **direct / matrix-free**: the σ-vector σ = M·b is formed
on the fly by contracting the MO integrals per block, and **M is never stored**.
It trades recompute-per-apply for a near-zero resident operator. That is exactly
the trade that makes the whole-band melanin solve tractable there and not here.

## The fix: generalize the matrix-free apply (referencing theADCcode)

ADCgo already contains the mechanism — it is just scoped too narrowly. The
`-matfree {off|auto|on}` flag (`cmd/adcgo/main.go`, `-maxmem`) applies the large
CVS-ADC(4) 3h2p coupling blocks matrix-free (recompute vs store) precisely to
dodge a resident-memory blowup. The proposal:

1. **Port theADCcode's direct DIP σ-build** for the dominant blocks — the
   3h1p↔3h1p satellite block first (it is the bulk of `OperatorResidentBytes`),
   then the 2h↔3h1p coupling — as a matrix-free `ApplyBlock` path in
   `internal/adc/dip`, driven by the same integral contractions theADCcode uses.
2. **Extend `-matfree=auto`** to DIP/SIP: a block whose resident size exceeds
   `-maxmem` is applied on the fly; small blocks (2h/2h main) stay materialized.
   Reuse `OperatorResidentBytes()` per block as the switch.
3. With the satellite block matrix-free, the resident footprint collapses to the
   panels (+ the tiny main block) — tens of GB, already `-mgpu`-friendly. melanin
   whole-band DIP then fits a single node, and uracil2W fits without DZ.

**Cost/benefit:** matrix-free adds recompute per mat-vec (×`-blocks` iterations),
but the applies are compute-bound GEMMs on the H200 and the operator upload it
replaces is a one-time multi-hundred-GB transfer; removing the resident-operator
ceiling is very likely worth the extra flops. Validate against theADCcode
`adc2dip` sticks on a small case (h2o DIP, `examples/DIP_h2o`) before scaling.

## Implementation (shipped)

Both fixes this note proposed have landed for the **host** backend:

**Parallel operator walk.** `parRows`/`parChunks`/`chunkWorkers` were promoted out
of `sip` into a shared `internal/adc/parallel` package. `matvec.go`'s walk is now a
task-structured `mainCouplingTasks`/`satelliteTasks` (one task per 3h1p row-group);
`assemble` and `OperatorResidentBytes` evaluate the tasks across a worker pool into
disjoint per-task slots, concatenated in task order so the dense `parts` are
byte-identical to the old serial walk. The single-threaded sizing/assembly drag is gone.

**Matrix-free satellite region.** The `-matfree {off|auto|on}` / `-maxmem` mechanism was
generalized from CVS-ADC(4) to DIP (`internal/adc/matfree` holds the shared `Mode`/`Decide`
policy). `internal/adc/dip/matfree.go` applies the 3h1p↔3h1p satellite blocks
(`jiiLKK`/`ijkMLL`/`ijkLMN`) on the fly and never stores them: cheap per-block gates
(`…Gate`, occupied-index Kronecker-δ + virtual-group sizes, no integrals) size and prune;
occupied-index candidate buckets cut each apply to O(G·k) instead of O(G²) group-pairs; the
symmetric operator is realized in two barrier-separated passes (forward over row-groups,
transpose over col-groups) with disjoint per-worker output bands, so no locking or reduction.
`OperatorResidentBytes` drops the satellite term when matrix-free, and the `-mgpu` path gained
the pre-flight guard (`checkSubsFit`) it previously lacked. With the satellite region
matrix-free the resident footprint collapses to the panels + main/coupling blocks.

**Validation.** `internal/adc/dip/matfree_test.go` checks ApplyFull/ApplyBlock/
ApplyBlockSatellite matrix-free == dense (≤1e-10), exact gate sizing, and an exhaustive
gate/shared-occ audit; `validate` drives the real block-Lanczos solver over the DZP ¹A₁
reference sector matrix-free and reproduces the dense spectrum to **1.95e-14 eV** (139 roots,
theADCcode-matched integrals). Under the chaotic short-recurrence `lanczos-lowmem` solver the
dominant main lines still agree to ~1e-12 eV; individual low-pole-strength satellite lines
diverge as expected (short-recurrence ghost sensitivity, not a matrix-free error).

**Phase C (shipped, host-validated).** Both Phase-C pieces have landed:

- *CUDA `DeviceKernels` twin.* The satellite region is reformulated as a per-output-scalar
  apply (`dip/satelem.go` element functions + `dip/satscalar.go` one-thread-per-row driver),
  pinned bit-close to the dense blocks by `TestSatelliteScalarMatchesDense` /
  `TestSatelliteScalarApplyEqualsDense`. `backend/adc2dip_kernels.cu` is a line-for-line
  transcription of that scalar form; the `backend.DeviceKernels.DipSatApply` binding
  (`cuda_kernels.go`) and the device applier (`dip/matfree_device.go`) upload the config SoA +
  flat ERI once and launch the kernel each mat-vec. `matFreeSatellite` now accepts a
  `DeviceKernels` backend. On-hardware parity: `dip/matfree_cuda_test.go` (cuda-tagged).
- *`-mgpu` + matrix-free composition.* `distBackend` gained `PanelScatterAdd.AddPanel`; the DIP
  satellite runs gather-apply-scatter under the row-partitioned backend (`dip/matfree_dist.go`):
  the full input is gathered to host, the satellite contribution recomputed with the same
  per-scalar kernel, and scatter-added back into the partitioned output — so the dense
  main/coupling blocks and Krylov panels stay partitioned across devices while the multi-TB
  satellite region is never materialized. Validated over gonum sub-backends
  (`TestSatelliteMatFreeDistributedEqualsDense`) and end-to-end (`-mgpu 2 -matfree on`: main
  lines match the dense solve).

**Remaining (Phase C follow-up):** the `-mgpu` satellite currently recomputes on the host
(gather-apply-scatter); a per-partition *on-device* apply — each GPU recomputing only its own
output band with the CUDA kernel — is the performance step for the real 8×H200 melanin run.
The memory ceiling (the point of this note) is already removed.

## Pointers

- Operator materialization + sizing + parallel walk: `internal/adc/dip/matvec.go`
  (`assemble`, `mainCouplingTasks`/`satelliteTasks`, `OperatorResidentBytes`).
- Matrix-free satellite apply (host block GEMV): `internal/adc/dip/matfree.go`; shared policy
  `internal/adc/matfree`; shared worker pool `internal/adc/parallel`.
- Matrix-free satellite per-scalar form (CUDA source of truth + device/distributed paths):
  `internal/adc/dip/satelem.go`, `satscalar.go`, `matfree_device.go`, `matfree_dist.go`;
  kernel `internal/adc/backend/adc2dip_kernels.cu`; binding `internal/adc/backend/cuda_kernels.go`.
- Pre-flight guard (single-GPU path only): `cmd/adcgo/dispatch.go`
  (`checkDeviceFit`); the `-mgpu` path (`main.go` `solveDIPSectorMGPU`) has none —
  add one there as part of this work.
- Existing matrix-free precedent: `-matfree` / `-maxmem` for CVS-ADC(4).
- The basis-memory companion problem: [`dip_lowmem_lanczos.md`](dip_lowmem_lanczos.md).
