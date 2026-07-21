# backend — how a Lanczos block is split and applied

Orientation for the compute layer: what a `Backend` is, and — the part that is hard to
reconstruct from the code alone — how one block-Lanczos mat-vec is partitioned across devices and
turned into kernel launches. Companions:
[`docs/dip_operator_memory.md`](../../../docs/dip_operator_memory.md) (why the satellite region is
matrix-free) and [`docs/dip_lowmem_lanczos.md`](../../../docs/dip_lowmem_lanczos.md) (why only a
few panels are resident).

## The pieces

| | |
|---|---|
| `Backend` | the BLAS-ish surface the solvers use: `Alloc/Upload/Download`, `Gemm`, `GemmMat(Batched)`, `AxpyDiag`, … Implementations: `Gonum` (host, reference), `gpuBackend` (cuda), `hip`, `distBackend`. |
| `HostData` | optional: exposes a resident vector's backing slice. Host backends only. **`gpuBackend` embeds `Gonum`, so it satisfies this interface too** — always test `DeviceKernels` *before* `HostData`, or a device backend takes a host path and panics on `HostSlice(devVec)`. |
| `DeviceKernels` | optional: the custom CUDA kernels (`Wert2Apply`, `C22Apply`, `DipSatApply`) plus the upload helpers they need. Implemented by the cuda backend only. |
| `PeerCopier` | optional: NVLink peer access — `EnablePeerAccess`, `PeerCopy2D`, `Sync`. |
| `PanelScatterAdd` | optional: `AddPanel` — add a full host panel into a partitioned one. Backs the host-fallback satellite path. |
| `PartitionedDevices` | optional: reach the individual partitions (`NumParts`, `Bounds`, `PartBackend`, `PartKernels`, `PartVector`, `AllPeered`). Backs the per-device satellite path. |

`distBackend` row-partitions the config (`n`) dimension across sub-backends. Row-partitioning is
chosen because every reduction the solver performs (α, Gram, CGS2 projections, `Dot`/`Nrm2`)
contracts that dimension into a `main×main` or scalar result — a local partial plus a tiny
all-reduce. Only the mat-vec needs real cross-device exchange.

## One mat-vec, end to end

Numbers below are the melanin triplet sector: `n = 14,766,249`, `b = main = 1653`, 8×H200.

```
① THE LANCZOS BLOCK — one n×b column-major panel, row-partitioned across devices
   (lanczos-lowmem hands ApplyBlock a column range of its ring buffer; rPrev
    ping-pongs, so the base pointer MOVES every iteration → resolve it per apply)

        b = 1653 cols
      ├──────────────┤
   ┌──┬──────────────┐  rows 0..main   ← 2h main block  (dense, tiny)
   │  │              │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ bounds[1] ── may straddle! ─ ─
 n │  │   panel      │  rows main..n   ← 3h1p satellite (matrix-free, multi-TB)
   │  │              │
   └──┴──────────────┘
        split by bounds[] (group-aligned, from dip.PartitionBounds)
   ┌──────────┬──────────┬─────┬──────────┐
   │  dev 0   │  dev 1   │ ... │  dev 7   │   each holds rowsOn(d)×b,  ld=rowsOn(d)
   └──────────┴──────────┴─────┴──────────┘   (~1.85M rows each)

② ONE ApplyBlock                              backend/distributed.go + dip/matvec.go
   ├─ dense blocks ──→ GemmMatBatched   per device, remote row band peer-copied if needed
   └─ satellite ─────→ matrix-free, chunked over columns (w = satChunkCols = 64):

      for c0 in 0, 64, 128, ... 1653:            ← ceil(b/w)=26 chunks
        syncAll()                                 fence producers (peer read ≠ drain src)
        ┌─ GATHER (NVLink) ──────────────────────────────────────────┐
        │  every device builds a FULL-HEIGHT n×64 slab, because a    │
        │  candidate column C can live in ANY partition:             │
        │                                                            │
        │   dev d slab        PeerCopy2D(rows,64, dstLd=n, srcLd=rd) │
        │   ┌────┐  ← rows[0..b1)   from dev 0                       │
        │   │    │  ← rows[b1..b2)  from dev 1                       │
        │   │ n  │  ← ...                                            │
        │   │    │  ← rows[b7..n)   from dev 7                       │
        │   └────┘  7.0 GB @ w=64                                    │
        └────────────────────────────────────────────────────────────┘
        syncAll()                                 fence gather before any kernel reads
        ┌─ LAUNCH: each device, ITS OWN band only ───────────────────┐
        │  rowLo = max(bounds[d],main)-main                          │
        │  rowHi = min(bounds[d+1],n)-main    (skip if rowHi<=rowLo) │
        │  dip_sat_apply<<<(rowHi-rowLo+127)/128, 128>>>             │
        └────────────────────────────────────────────────────────────┘
      syncAll()                                   fence outputs for the caller

③ INSIDE THE KERNEL — one thread per OUTPUT 3h1p row      adc2dip_kernels.cu
   ri = rowLo + tid;   R = main + ri          ← GLOBAL row: indexes slab + candidates
                       Rout = R - outRowOff   ← LOCAL row: indexes this device's panel

   thread ri:  for each candidate col group (shared-occ early-out)
                 g = d_jii_* / d_ijkMLL_* / d_ijkLMN_*   ← recomputed from device ERI
                                                            (15 GB, replicated per device)
                 for jc in 0..64:
                    yout[Rout + jc*ldOut] += g * xin[C + jc*ldIn]
                              ▲                        ▲
                     local, ld=rowsOn(d)      full-height slab, ld=n
```

## Why it is shaped this way

**The slab must be full height.** The output is partitioned, but the input is not separable: a
candidate column `C` for a row owned by device 3 can sit on device 6. That asymmetry is the whole
reason a gather exists — and why the earlier implementation took the simpler route of dragging the
entire panel to the host, contracting on CPU, and scattering it back (~137 GB each way).

**`R` and `Rout` are deliberately separate.** `R` stays global: it indexes the slab and drives the
candidate comparisons. Only the *write* is rebased into the device's local panel. Conflating the
two silently corrupts the output.

**Chunking trades recompute for residency.** `g` is recomputed once per `(row, candidate)` *per
chunk*, so 26 chunks means 26× the element evaluations — but each is amortized over `w` column
operations, so the overhead is ~15% at `w=64`, not 26×. Raising `w` cuts that (~8% at 128) and
costs proportionally more slab VRAM. That is the only knob `satChunkCols` turns.

**Never cache a device pointer across applies.** The Mode-B ring buffer moves the input panel's
column offset between iterations (`lanczos/lowmem.go`: `rPrev, rCur = rCur, rank`), and
`devVec.ptr()` resolves `base + off*elemSize` when queried. A pointer array uploaded once reads the
wrong half of the ring from the second iteration onward — a single-iteration parity test passes
while the solve is quietly wrong.

**Fences are not optional.** A peer read does *not* drain the source device's stream, so the gather
is fenced on both sides. The syncs are issued concurrently (one goroutine per device): each is a
blocking round-trip through that device's owning goroutine, and serialized over 8 devices twice per
apply that is pure added latency.

## Fallbacks

Path selection is `dip/matfree.go newSatelliteMatFreePart`, most capable first:

```
DeviceKernels ......... single-device CUDA kernel, whole band  (RowLo=0, RowHi=nsat, OutRowOff=0)
PartitionedDevices .... per-device kernel + NVLink gather   ← requires kernels on EVERY partition
   && AllPeered                                                and every pair peered
PanelScatterAdd ....... host gather-apply-scatter           ← always correct; the slow path
HostData .............. single-node host block applier
```

The host paths stay in place deliberately: they are what the gonum-backed correctness tests run,
and the only thing available when peer access is missing.

## Tests

| what | where | hardware |
|---|---|---|
| partition metadata, per-device panel slices, panic guards | `distributed_test.go` | none (gonum subs) |
| row-band derivation: straddling / empty / seam, tiling invariant | `dip/matfree_dist_test.go` | none |
| satellite matrix-free == dense, incl. distributed composition | `dip/matfree_test.go` | none (gonum subs) |
| per-scalar form == dense blocks (pins the kernel's design) | `dip/satscalar` tests | none |
| single-device kernel parity | `dip/matfree_cuda_test.go`, `sip/matfree_cuda_test.go` | 1 GPU |
| per-device parity == dense **and** == host path, bit-exact | `dip/matfree_mgpu_cuda_test.go` | ≥4 peered GPUs |

The host-side tests are the source of truth for the physics; the on-hardware tests fix only the
CUDA transcription. Build the kernels before any `-tags cuda` run — `cuda_kernels.go` links
**both** `adc4_kernels.o` and `adc2dip_kernels.o`, and omitting either fails the link:

```
scripts/build_adcgo_cuda_helix     # nvcc both .cu, then go build -tags cuda
```
