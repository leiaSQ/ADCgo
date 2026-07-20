# DIP/SIP GPU memory — the resident block-sparse operator is the real wall

**Status:** design note / future work. Companion to
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

## Pointers

- Operator materialization + sizing: `internal/adc/dip/matvec.go`
  (`assemble`, `visitBlocks`, `OperatorResidentBytes`).
- Pre-flight guard (single-GPU path only): `cmd/adcgo/dispatch.go`
  (`checkDeviceFit`); the `-mgpu` path (`main.go` `solveDIPSectorMGPU`) has none —
  add one there as part of this work.
- Existing matrix-free precedent: `-matfree` / `-maxmem` for CVS-ADC(4).
- The basis-memory companion problem: [`dip_lowmem_lanczos.md`](dip_lowmem_lanczos.md).
