# Deferred: Go-level parallelism (goroutines)

**Status: shelved, deliberately.** Written 2026-07-08, after the level-3 BLAS rewrite.

This file records what was planned, why it was *not* done, and what would have to be
true for it to become worthwhile. The repo still contains **zero goroutines**
(`grep -rn "go func\|sync\.\|errgroup" --include=*.go .` matches only cgo `#include`
lines).

## Why it was shelved

The original motivation was that `adcgo` used one core while five OpenBLAS threads sat
idle. That was true, but the cause was not a missing `go func`: `lanczos.Solve` was
running GEMM-shaped work as O(dim²) level-1 BLAS calls, and OpenBLAS does not thread
`Dot`/`Axpy`. Fixing the BLAS *level* — not adding threads — recovered the cores:

| formic acid, cc-pVDZ, `-blocks 200`, 6 threads | wall |
|---|---|
| before (level-1 orthogonalization, `dsyev`) | ~64 min (modeled; a real run was killed at 45 min mid-way) |
| after (level-3 CGS2 + `ApplyBlock` + `dsyevd`) | **630 s** |

After that change, threaded GEMM already saturates all 6 cores for the phases that
matter. Goroutines would now contend with OpenBLAS threads rather than add
parallelism, and every candidate site is either memory-bound or already negligible.

## The four candidate sites, with measurements

Percentages are of a single sector's solve time (formic acid `s0`: n=12842, b=58).

1. **Sector loop** (`cmd/adcgo/main.go`, the `spin × irrep` nest) — up to 8 independent
   sectors. **Blocked by memory, not by code.** Peak RSS for the full FA run is
   **9.1 GB of 15 GB**, and a single sector (`s0`) accounts for most of it: basis
   1.11 GB + projected matrix T 1.00 GB + its copy + `dsyevd` workspace (~2·dim²) +
   operator blocks 0.25 GB. Two concurrent sectors would not fit. Any implementation
   must gate on *estimated bytes*, not on a worker count.

   Two further constraints, if it is ever attempted:
   - JSON sector order is load-bearing (`spectrum.BuildDIP` iterates slice order).
     Replace the `append` with an indexed write into a preallocated slice.
   - `gpuBackend` owns a single BLAS handle bound to one OS thread
     (see `gpu_device.go`). Concurrent sectors on a GPU backend need a handle and
     stream per worker, or must serialize the device and overlap only host work.

2. **Per-state populations** (`analyze.BuildSector` → `PopEngine.Compute`) — an
   O(nao⁴/4) `MulVec` per surviving state, independent across states, and too small for
   BLAS threading to help. This is the **most defensible remaining target**, but it is
   `back`-phase work: **0.1 % of `s0`'s runtime** (0.21 s of 272.8 s). Not worth the
   oversubscription risk on its own.

3. **DIP matrix assembly** (`dip/matvec.go:assemble`) — pure functions, independent
   output cells, backend-independent. **Measured: 1.08 s for all four FA sectors
   combined.** Skip it. (An earlier analysis called this "almost certainly where the
   wall clock is"; the flop count and the runtime thread accounting both say it is not.)

4. **SIP order-3 assembly** (`sip/elements.go:c11_3sums`, and `sip/amplitudes.go:FMatrix`,
   which recomputes from scratch what `mainBlock()` already computed and discarded).
   This is O(nvir⁴·nocc) *per element* and is the dominant cost of `-sip -order 3`.
   **This one is real** — but the fix is *caching*, not threading, and DIP is the
   current focus. Cache first, then measure again.

5. ~~Parallel `apply` fan-out inside a Lanczos block~~ — **obsoleted.** `ApplyBlock`
   now applies M to all `b` columns in one pass; there is nothing left to fan out.

## The oversubscription constraint

If this is ever revisited, it needs **one** knob, not several:

```
-jobs N    # Go-level parallelism; must also set openblas_set_num_threads(NumCPU/N)
```

Default `-jobs 1` (all cores to BLAS). Anything that parallelizes over goroutines must
first drop the BLAS thread count, or the two nest multiplicatively and the machine
thrashes. `fcidump.Data`'s integral store is immutable after `Read` returns, so
concurrent reads of `ints`/`eps`/`orbSym`/`d` are safe.

## What would change the calculus

Revisit if any of these become true:

- **Peak memory per sector drops enough for 2–3 concurrent sectors.** The obvious
  reduction is the projected matrix: `T` is allocated at `maxdim²` and then *copied*
  into a `dim²` matrix before `SymEig`, and `dsyevd` allocates ~2·dim² of workspace on
  top. Eliminating the copy and shrinking the subspace (see below) would roughly halve
  the sector footprint.
- **Band Lanczos lands.** With partial reorthogonalization and a banded projected
  matrix (`lanczos.go`'s own TODO, citing theADCcode's `bnd2td.f`/`tddiag.f`),
  orthogonalization drops from O(n·dim²) to O(n·dim·b) and `SymEig` from O(dim³) to
  O(dim·b²). For FA, `dim = 11600` against `n = 12842` — the "Krylov subspace" is 90 %
  of the full space, so the current solver is a dense diagonalization with extra steps.
  That is a far larger win than any threading, and it shrinks the memory that blocks
  sector parallelism.
- **A machine with substantially more RAM**, where the sector loop's memory gate stops
  binding.
- **The SIP order-3 path becomes the workload**, in which case cache `c11_3sums`
  first and re-measure before threading anything.
