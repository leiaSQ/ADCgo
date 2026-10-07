# DIP-ADC(2) low-memory Lanczos — the gap vs. theADCcode

**Status:** IMPLEMENTED (`-solver lanczos-lowmem`, `internal/adc/lanczos/lowmem.go`
+ `bandeig.go`). This note originally recorded why ADCgo's full-band DIP-ADC(2)
block-Lanczos was not runnable on the production problem while theADCcode's was, and
sketched the low-memory variant. That variant now exists; the sections below keep
the original analysis (still the correct explanation of the gap) and the
[Implementation](#what-was-implemented) / [Findings](#findings) sections at the end
record what shipped and what the work discovered.

Written 2026-07 while splitting the production pipeline into DIP=Davidson /
SIP=Lanczos (see `scripts/helix/HELIX.md`). The immediate production DIP runs use
block-Davidson (lowest `-nroots` roots); the low-memory solver is for whoever
wants the *whole* double-ionization band.

## The gap in one line

The DIP-ADC(2) full-band solve is blocked by an **algorithmic** memory cost, not
a hardware ceiling. A bigger node does not fix it — the current solver stores the
entire Krylov basis, and for the production system that basis is tens of terabytes. theADCcode
runs the same physics because it uses a **limited-memory** Lanczos (short
recurrence + banded tridiagonal solver + ghost filtering) that never stores the
full basis.

## Why ADCgo's `Solve` cannot run production DIP

`internal/adc/lanczos/lanczos.go`, `func Solve` (line 321), is a block-Krylov
Rayleigh–Ritz with **full reorthogonalization + deflation** (package comment,
lines 6–12). Full reorth means every Krylov vector must be kept for the life of
the run so each new block can be orthogonalized against all of them. The whole
basis is therefore allocated up front on the device:

```go
bbuf := be.Alloc(n * maxdim)   // lanczos.go:332  — the entire basis panel
```

with `maxdim = blocks · main`. That single allocation is the memory driver.

Measured production DIP spaces (real space builders, `-blocks 200`):

| Sector      | dim `n` | main block | one Krylov block `n·main·8` | full basis `n·maxdim·8` |
|-------------|--------:|-----------:|----------------------------:|------------------------:|
| DIP singlet | 10.0 M  | 1711       | ~137 GB                     | ~25 TB                  |
| DIP triplet | 14.8 M  | 1653       | ~137 GB                     | ~36 TB                  |

`ApplyBlock` needs a second `n·main` work panel (`wbuf`), so even **one** Lanczos
iteration wants ~2·137 = 274 GB resident. No Helix GPU (H200 = 141 GB) holds a
single block; the fattest CPU nodes (~2–4 TB) hold only ~15–30 blocks of a
200-block run. There is no node — current or in a realistic project request — with
25–36 TB of memory. **This is why a higher node tier does not help.**

SIP-ADC(3), by contrast, has `n ≈ 518 k` → full basis ~45 GB, which fits an
80 GB+ GPU. That is the whole reason the pipeline splits: SIP keeps full-reorth
Lanczos (with checkpointing), DIP falls back to Davidson.

## How theADCcode runs it: limited-memory Lanczos

theADCcode (`../ADC`, referenced in the package comment lines 3–4 and 14–16)
does *not* keep the basis. It uses the classic short-recurrence Lanczos:

- **3-term recurrence.** Only a handful of `n`-vectors are live at once
  (current, previous, next block), a few hundred MB — independent of the number
  of iterations. The tridiagonal `α`/`β` blocks accumulate; the basis is
  discarded as it is generated.
- **Banded tridiagonal eigensolver** (`bnd2td.f` / `tddiag.f`, named in
  lanczos.go:15). The projected matrix stays block-tridiagonal because there is
  no reorthogonalization to fill it in, so a band solver — not a dense `SymEig` —
  extracts the Ritz values. At production scale this stage is NOT cheap: it is
  O(dim²·band) and ran for weeks on one core before it was parallelized — see
  "Parallelizing the band reduction".
- **Ghost / spurious filtering.** Dropping full reorth is what makes it cheap,
  and the price is *Lanczos ghosts*: spurious duplicate/garbage eigenvalues from
  loss of orthogonality in finite precision. theADCcode filters them with a
  main-space-weight test, `spur_thresh = 1e-9`: a Ritz vector with essentially
  zero weight in the 2h "main" space is a ghost and is discarded.

Trade-off vs. ADCgo's current solver: limited-memory Lanczos is O(few vectors)
in memory but needs the ghost filter and generally more matvecs / careful
convergence bookkeeping; full-reorth `Solve` is numerically clean and
bit-reproducible (which is what made SIP checkpointing easy) but O(full basis)
in memory.

## The hook already in the tree: `Result.Spurious()`

ADCgo already carries the ghost test the low-memory driver needs —
`lanczos.go:559`:

```go
// Spurious reports whether a Ritz vector is a Lanczos ghost: essentially zero
// weight in the main space (the reference's spur_thresh = 1e-9 test on the
// main-block components). k indexes a column of MainVecs.
func (r Result) Spurious(k int, thresh float64) bool {
	for c := range r.MainVecs.Rows {
		if math.Abs(r.MainVecs.At(c, k)) > thresh {
			return false
		}
	}
	return true
}
```

Under full reorth ghosts do not arise, so today `Spurious()` is effectively a
no-op safety check. In a limited-memory driver it becomes load-bearing: it is
exactly the `spur_thresh` filter that makes short-recurrence Lanczos usable.

## Implementation sketch (future)

Add a low-memory driver **alongside** `Solve`, do not modify it — SIP depends on
the full-reorth path and its checkpoint format.

1. **New entry point**, e.g. `func SolveLowMem(op Operator, be backend.Backend,
   opts Options) Result`, selected by a solver flag (e.g. `-solver lanczos-lowmem`
   or auto when the full basis would exceed device memory).
2. **Short recurrence.** Keep only `Q_{j-1}, Q_j, Q_{j+1}` block panels on the
   device (3·`n·main`, ~410 GB for the production system — still too big for one GPU at
   `main≈1700`, so either (a) shrink the block to a few start vectors instead of
   the full main space, streaming the main-space projection, or (b) tile `n`
   across GPUs). This is the real design work: the block width, not the iteration
   count, is now the memory knob.
3. **Accumulate `α_j`, `β_j`** (host-resident `main×main` blocks) into a
   block-tridiagonal `T`; do **not** allocate `n·maxdim`.
4. **Band eigensolver** on `T` (port `bnd2td`/`tddiag`, or use a LAPACK banded
   routine via the existing host BLAS/LAPACK path) instead of dense `SymEig`.
5. **Filter with `Spurious(k, 1e-9)`** before reporting roots / pole strengths;
   optionally add selective reorthogonalization (Simon/Parlett) if ghost rates
   are too high to filter cleanly.
6. **Pole strengths / MainVecs** are recovered from the main-space projection of
   the retained Ritz vectors, same observable as today — only the basis handling
   changes.

Checkpointing this path is *harder* than the full-reorth SIP path: with the basis
discarded, resume is no longer "reload the panel and continue" — you would
checkpoint the short-recurrence state (`Q_{j-1}, Q_j`, current `α/β`, iter) and
accept that the discarded history cannot be regenerated for a from-scratch
reorth. For a first cut, run the low-memory DIP without checkpointing (it should
be fast enough per matvec that a single 120 h allocation covers a useful band),
and add short-recurrence checkpointing only if walltime becomes the limit.

## What was implemented

`func SolveLowMem(op Operator, be backend.Backend, opts Options) Result`
(`internal/adc/lanczos/lowmem.go`), selected by `-solver lanczos-lowmem`, with the
block width set by `-lowmem-block` (`Options.LowMemBlock`; 0 → the main-space size).
Two modes, auto-chosen:

- **Mode B — the faithful port (default, block = main).** Short 3-block recurrence
  (only `[prev|cur]` + a work panel resident), Tarantelli's subspace-iteration gate
  (`dip.Matrix.ApplyBlockSatellite` — the 3h1p↔3h1p sub-operator, applied after the
  first two blocks), the banded eigensolver `bandeig.go` (a pure-Go port of
  Tarantelli's `bnd2td`+`tddiag` that materializes only the eigenvector rows that are
  used — the top `main` and the last block's, via `bandEigOpts`), and the
  `Result.Spurious(1e-9)` ghost filter. Resident cost
  ≈ 3·(n×main) — a fat-memory CPU node. Bit-reproducible against theADCcode. On H2O
  it matches the dense main lines and pole strengths exactly (`TestSolveLowMemModeB_MatchesDense`).

- **Mode A — device-frugal full reorthogonalization (block < main).** Keeps only
  three n×block panels on the compute backend and the *full basis on the host*,
  reorthogonalizing each new block against all of it (streamed a block at a time).
  Numerically exact at sufficient Krylov dimension (`TestSolveLowMemModeA_FullExact`),
  and the eigensolve is a plain dense `SymEig` of the small projected matrix with
  main components recovered from a retained `main×dim` host slice (`Qmain·s`).

The banded eigensolver is validated on random banded matrices against dense `SymEig`
for both eigenvalues and the top/bottom partial-vector slices
(`bandeig_test.go`); the satellite gate against a masked dense operator
(`dip/satelem_test.go`).

## Parallelizing the band reduction

The band→tridiagonal reduction plus QL sweep is one call that at the melanin DIP shape
(`dim` 308000, half-bandwidth 3079) is ~10^15 element updates — **~23 days on one core**.
That is what job 14834817 was doing for 2 d 15 h after its last block line: one core busy,
all eight H200s idle. It is CPU-only and stays that way — cuSOLVER has no band or
tridiagonal routine, and these are hand-written scalar loops rather than BLAS calls, so
neither `-mgpu-device-symeig` nor `-tags openblas` reaches them.

Four changes, all bit-for-bit output-preserving:

- **Accumulate only the rows that are used.** `packLowMem` needs the top `main` rows and
  the last block's rows; the symmetric form carried `2·band` of them. `bandEigOpts`
  decouples the accumulated rows from the bandwidth (which `fillBand` still needs at
  `2b-1`), halving the rotation work and dropping `z` from 15.2 GB to 7.6 GB.
- **Defer the eigenvector writes.** `z` is a write-only accumulator — no scalar in either
  algorithm is ever read back out of it — so the rotations go onto a tape (`zTape`) and
  replay across contiguous, cache-line-aligned row blocks. Per row the ops keep their
  original order and operand order, so this is a loop interchange, not a reassociation.
- **Transpose the band matrix and flatten its walks.** `a` is stored row-major, so the bad
  stride is `mb` = 3080 rather than `n` = 308000, and the four hot inner loops advance by a
  running offset instead of a multiply per access.
- **Defer the band-matrix walks too.** Within one chase only the `l == maxl` iteration of the
  column walk feeds the next step, so it is peeled back onto the spine and the rest is deferred
  (`aTape`) and flushed once per chase. That is ~9.5e8 rounds, which is why `parallel.FixedPool`
  (persistent workers, spin barrier) exists rather than `parallel.Chunks`.

### What it actually bought: 3.3x, and it is not from the threading

Measured at the production half-bandwidth (band 3079, dim 4620, 64-core node,
`scripts/helix/eigensolver_tier2.sbatch`):

| configuration | wall | vs reference |
|---|---|---|
| frozen serial reference | 462.1 s | 1.00× |
| narrowed rows + transposed layout, 1 worker | 139.1 s | **3.32×** |
| the same at 16 workers | 196.9 s | 2.35× |

So the win is entirely the two serial changes — accumulating `main + lastSize` rows instead of
`2·band`, and the row-major band matrix with flattened index arithmetic — and **threading the
replay costs a further 1.4×**. `bandEigDefaultWorkers` is therefore 1, and `-eig-workers` turns
the parallel path on for anyone who wants to re-measure. Candidate causes for the loss: the
workers spinning through the long serial stretches, the NUMA placement of a `z` that one
goroutine first-touched, and lower all-core clocks with 16 busy cores instead of one.

`rows` = 3080 at this band is exactly what production accumulates, so that half of the result
transfers directly. What does **not** transfer is the band-matrix side: at `dim/band` = 1.5 a chase
is one or two steps and never reaches `aGrainDefault`, so the A-side parallel path never executed
in that run, whereas at production's `dim/band` = 100 a chase carries ~1.2e6 updates. No benchmark
reproduces production's chase length *and* its bandwidth without also reproducing its cost, so that
one can only be settled by a production run with `-eig-workers 16`.

At 3.32×, expect the melanin DIP reduction to be **~4 days** rather than the ~23 it was, against a
120 h walltime — it fits a resumed generation, with roughly a day of margin and no checkpoint inside
the stage. `Options.EigenProgress` is what makes that margin observable.

### Where the time actually goes now, and what a GPU can take

Splitting the 139.1 s at band 3079 by asking for two accumulated rows instead of 3080
(`BenchmarkBandSymDiagFastTier2/spine`):

| half | wall | share |
|---|---|---|
| band-matrix reduction (the sequential chase) | 88.9 s | 64% |
| eigenvector rotations | 50.2 s | 36% |

The transposed layout moved the bottleneck: before it the reduction dominated, and now that it is
4× faster the *chase* is what is left. Scaling both halves to production's `dim/band` = 100 (the
chase grows as `band·dim²`, the rotations slightly faster) gives roughly 57% chase / 43%
rotations.

`-eig-device` replays the rotations on a CUDA/HIP GPU (`backend/bandeig_kernels.cu`,
`backend.BandEigKernels`). That is the right half to offload — bandwidth-bound streaming over a
multi-gigabyte accumulator, independent per row, no reduction — but its Amdahl ceiling is therefore
**~1.75×**, not an order of magnitude. The chase stays on the CPU: it carries scalars from step to
step and synchronizes ~9.5e8 times, and a kernel launch is microseconds.

Bit-exactness on the device needs care and gets a dedicated test. Go on amd64 does not contract
`a*b+c` into an FMA; nvcc and hipcc do by default, and a fused product rounds once where Go rounds
twice. The kernel therefore uses explicit `__dmul_rn`/`__dadd_rn`/`__dsub_rn` rather than relying on
`-fmad=false`, and `TestGPUBandEigReplayParity`
(`scripts/gpu/bandeig_device_parity.sbatch`) compares device against host with `!=` on real
hardware — the only place the claim can be checked.

### The replay has far more parallelism than the kernel first claimed

`bandeig_kernels.cu` originally asserted that *"there is no more parallelism available than rows,
because a row's ops are strictly ordered."* **That is false for `bnd2td`'s rotations.** Inside one
chase `j` advances by `m1 >= 2` and each step records a single rotation on columns `(j-1, j)`, so
consecutive ops touch **disjoint column pairs and commute**. The claim is true only of `tddiag`'s
chained QL rotations, which is where it came from — and believing the wrong one is why the kernel ran
~96 warps on a 132-SM GPU and sat at 195 GB/s, 4% of HBM, flat across every block size from 32 to
1024 threads.

So the tape carries run boundaries (`zTape.seg`, `zAccum.mark()`), and `bandeig_replay_z_par` replays
a run with a 2-D grid over `rows × ops`. Bit-exactness is untouched: the arithmetic per element is
unchanged and only the schedule differs, which is licensed exactly by disjointness.
`TestZTapeSegmentsCommute` inspects the **real** tape rather than reasoning about the loop, and
`TestConcurrentRunOpsCommute` checks the implication directly by replaying a run in reverse order and
demanding bit-identical output, with a negative control that shares a column.

That test found a bug it was not looking for: it first reported *one op per segment*, which is what a
per-**step** flush looks like — `at.flush()` and `za.mark()` sat at loop depth 6, inside `for j`
rather than after it. The "per-chase flush" had been per-step since the band-matrix tape was written.
Correctness was never affected (flushing more often is always safe), but the tape never held more than
one step's runs, so the A-side parallel replay had nothing to partition. That, not the `aGrain`
threshold, is why that path never engaged.

**MEASURED (job 15058609, H200 sm_90).** Total work held fixed at 16384 ops x 3080 rows x 3 updates,
swept over how many ops a run holds, seq against par:

| ops/run | seq (G upd/s) | par (G upd/s) | gain |
|---|---|---|---|
| 1 | 4.11 | 4.11 | 1.00 |
| 8 | 17.1 | 30.3 | 1.8 |
| 100 | 30.1 | 194 | 6.5 |
| 400 | 31.8 | **280** | 8.8 |
| 2406 | 18.1 | 98.5 | 5.5 |

Three things fall out, and two of them cut the projection down:

- **The launch floor is 2.25 us**, not the 1.5 us assumed: 16384 single-op launches in 36.86 ms. At band
  3079 that is 9.5e8 chases x 2.25 us = **0.59 h of pure launch overhead**, which is not negligible; after
  stage 1 narrows to `b2` = 128 it is 3.9e7 chases = 0.025 h, which is. The two changes compose.
- **The peak at ops=400 does not transfer.** 400 stride-3 ops span 1200 columns = 29 MB, inside the
  H200's 60 MB L2; 2406 ops span 178 MB and fall out of it. Production is far past that — a chase at
  `b2` = 128 puts its ops 128 columns apart, so 2406 of them span the whole `dim` of a 7.6 GB strip. So
  **98.5 G upd/s is the figure that transfers** and 280 G is an L2 artefact. The cross-check is that
  seq at ops=2406 (18.1 G) reproduces the independently measured 18.3 G sequential baseline.
- `ops=1/par` matches `ops=1/seq` to 0.006%, which is the control: the single-op fallback works, so the
  gains above are the kernel and not the harness.

**So the honest gain is 5.4x, not the ~50x the design pass assumed**, and `bnd2td`'s strip share goes
**7 h -> ~1.3 h**, not the projected 0.5 h. The error was modelling occupancy as the only limit; at the
run lengths production actually issues, the replay is back to being HBM-bound, just at 5.4x more of the
bandwidth than before.

**The run boundaries around the underflow rescale are load-bearing.** `bnd2td` rescales every 64
columns (`bnd2td.f:124-145`), and those `scale` ops touch pairwise-distinct columns — but not
distinct from the *next* chase's rotations, which act on the same columns. Nothing between the last
chase's mark and the rescale closes the run, so without an explicit `mark()` on each side the rescale
shares a run with the following chase and the concurrent kernel applies them simultaneously. The
device then returned 1.5e+109 where the host had -1.4e-16. It passed at `dim` = 40/200/260 **only
because the rescale never fires there** — it needs `dim > 64` and a narrow band to trigger at all
(526 firings at `{512,5}`, none at `{200,20}`). The parity suite and
`TestZTapeSegmentsCommute` both now carry `{512,5}` and `{1000,10}`, and the latter asserts that the
rescale actually fired, so the coverage cannot lapse silently.

### The reduction checkpoints itself

Everything else here checkpoints the Krylov phase. The reduction that follows it was a single
uninterruptible call, and at the production shape it is ~144 h under a 120 h walltime — so the
daisychain restored the Krylov state, ground most of a generation into the reduction, was killed,
and repeated, **never converging**, five days of eight H200s per generation. Job 14834817 spent
4 d 19 h doing exactly that.

`bandeig_checkpoint.go` saves `{a, d, z, k}` at a column boundary of `bnd2td`'s outer loop, to
`<-checkpoint path>.eig`, every 30 minutes and on a stop signal. An interrupted reduction surfaces as
`Result.Interrupted`, so `cmd/adcgo` exits 64 and the successor resumes the reduction instead of
restarting it — the same protocol the Krylov phase already used.

Details that are easy to get wrong, and did:

- **The end of a `k` iteration is the only clean boundary.** There `{a, d, z, k}` is the whole state:
  `g` and `ugl` are per-chase, the band-matrix tape is flushed per chase, and `e`/`e2` are not written
  until the tridiagonal assembly. Inside a chase the rotation scalars live across the `j` walk.
- **The eigenvector tape must be flushed before saving.** It flushes when it fills or at the final
  sort, *not* per chase, so at an arbitrary column boundary it still holds pending rotations. Saving
  without applying them produced correct eigenvalues (which come from `a` and `d`) with wrong
  eigenvectors — caught by `TestBandEigCheckpointResumeIsBitExact`, and the hardest failure to notice
  downstream.
- **The host `z` is a stale mirror while `-eig-device` holds the live one,** so it is downloaded first.
- **The guard is a CRC-64 of the input band matrix,** not just its shape. A well-formed checkpoint from
  a different Krylov state of the same dimensions would otherwise resume into a plausible wrong
  spectrum.

`TestBandEigCheckpointResumeIsBitExact` drives the reduction with the stop flag permanently set, so
it restarts at **every** column boundary and the final result must be bit-identical to one
uninterrupted run. `TestSolveLowMemSurfacesReductionInterrupt` covers the integration: stop during
the reduction, checkpoint written, `Interrupted` surfaced, successor finishes, output identical.

### Two-stage reduction: narrow the band first (`-eig-b2`)

The chase's cost splits into two halves with different scaling laws, and that asymmetry is the whole
lever:

| | rotations | band updates per rotation |
|---|---|---|
| `bnd2td` | ≈ `dim²/2` | `≈ 2·band` |
| `tddiag` | ≈ `dim²` | — |

The rotation count is **independent of the bandwidth**; the band work is **proportional** to it. So
narrowing the band before the chase removes the band half and leaves the strip half untouched. At the
production shape that is ~63 h of band work at bandwidth 3079 against ~1.3 h at 64.

`-eig-b2 N` runs stage 1 (`narrowProjected`), which chains two reductions because they are good at
different things:

- **`blockQRReduce`** exploits the *block*-tridiagonal structure: one QR per off-diagonal block halves
  the bandwidth (`2b-1 → b`) in a handful of large GEMMs, far cheaper per unit of narrowing than
  anything general. It cannot go further — the diagonal blocks stay dense — and it needs
  `beta.Rows == size`, which deflation can break (`blockQRReduceOK`).
- **`sbrSweep`** (successive band reduction) then narrows to the target. It reads only `bandStorage`,
  so it is immune to that deflation mismatch, which also makes it the *fallback* when the guard fires:
  one extra sweep from the full bandwidth rather than losing stage 1 entirely.

Two specification details that cannot be derived from memory and are gated by a dense reference
(`sbrDenseRef`, `TestSBRDenseRefIsASimilarity`) before any band code runs:

- A QR of the `m × d` panel can only **triangularise** it — a `d × d` triangle always survives, and its
  diagonal sits at distance `b'` from the main diagonal. That is what sets the resulting bandwidth, and
  why the window top is sub-diagonal `b'` rather than `b'+1`. With `d = bw − b'` and `d ≤ b'`, one sweep
  can at most **halve** the bandwidth, so `sbrReduce` runs a schedule of sweeps.
- The chase stride is **`bw`, not `m`**. After an update the deepest entry in the window's columns sits
  at distance `mm−1+bw`, so the next window starts `bw` lower. `m == bw` exactly when a sweep halves —
  which is why a halving-only test suite hides the bug, and why non-halving shapes (9→8, 5→4) are in
  the suite.

**Stage 1 must be GEMMs, and this is where it would have been lost.** Stage 1 is ~1.17e15 flops split
evenly between the band update (`2·dim²·beta`) and the strip update (`2·rows·dim²`, independent of both
the block width and the bandwidth). Measured with `blas64.Gemm` on 64 cores (job 15048404):

| shape | GFlop/s |
|---|---|
| band, first sweep — 1540×1540×4620 | 159 |
| strip, first sweep — 3080×1540×1540 | **178** |
| band, last sweep — 192×192×576 | 56 |
| strip, last sweep — 3080×192×192 | 76 |

`stripApplyQ` and `sbrSimilarity`'s products were initially hand-written scalar nests running on one
core; they are `blas64.Gemm` now. Single-threaded they would cost 6.5 h and the stage would be
pointless.

**But the synthetic rate is not the rate the real call gets, which is why
`BenchmarkStripApplyQ` exists.** On 64 cores it returned **68.2 GFlop/s** at the same 3080x1540x1540
shape the synthetic product ran at 158.6 — 43%. The cause is not the strided strip operand (~5%) but
**`blas.Trans`: gonum's pure-Go Dgemm is 2.5x slower on a transposed operand** (9.5 vs 23.7 GFlop/s
isolated). Since `Z <- Z*q` on a column-major strip is `S <- q^T S`, every one of these products had a
transposed left factor. Materializing the transpose instead costs `m*m` copies against `2*rows*m*m`
flops — 0.016% of the work at `m` = 1540 — so `gemmTN` now transposes explicitly and `stripApplyQ`
passes `NoTrans`. **Confirmed on 64 cores (job 15194435): 68.2 -> 134.0 GFlop/s at `m` = 1540 (1.97x)
and 27.3 -> 49.4 at `m` = 192**, which takes the real call from 43% to **87%** of the synthetic
product's rate at the same shape.

At the resulting rates stage 1 costs **~2.2 h** against the ~29 h of band work it removes. `-tags openblas` **does not link** on this cluster (no `-lopenblas`/`-llapacke`), so the
untagged pure-Go rate is the rate that matters — and it is enough.

Two traps that cost real time here, both now loud rather than silent:

- **`bandStorage.set` is unchecked**, so an out-of-band write lands in the *next column* and corrupts a
  real entry. `fillBandNarrowed` and `sbSet` panic above 1e-12 instead. The symptom without them was
  eigenvalues wrong in the first significant figure while a full-width rebuild matched to 4e-15.
- **The strip-feasibility guard in `diagProjected` must keep testing the ORIGINAL bandwidth.** Stage 1
  is hooked in after it; feeding it `b2` would divert every production run to the 759 GB dense path.

End to end on `h2o_dzp` DIP, `b2` ∈ {8, 4, 2} against `b2 = 0`: 64 states each, max relative ΔE 3.3e-15,
Δ`ps_percent` ≤ 1.1e-12, Δ`residue` ≤ 1.2e-13. `b2` is flat between 64 and 128 (within 20%) — SBR's
later sweeps get more expensive per flop as the matrices shrink, while the chase's band cost falls
linearly in `b2` — so the schedule stops at 128.

### The dim scaling is WORSE than either law, and the band half is DRAM-bound

`BenchmarkBandSymDiagFastTier2` at the production band (3079), doubling `dim` (job 15194417):

| variant | dim=4620 | dim=9240 | ratio | the law predicts |
|---|---|---|---|---|
| `ref` (frozen serial) | 464.0 s | 2124.3 s | 4.58 | — |
| `spine` (band work only) | 64.7 s | 490.6 s | **7.58** | ~2, linear in `dim` |
| `narrow-w1` (production strip, serial) | 143.9 s | 828.7 s | 5.76 | 4, as `dim²` |
| `narrow` (same, threaded) | 150.7 s | 893.6 s | 5.93 | — |

**So neither scaling law holds at these shapes, and `spine` is the tell.** The band half should be
linear in `dim` at fixed band (`2·dim·band²` work). The chase count does behave: summing
`min(m1, dim-k)` over the outer loop gives 9.5e6 -> 2.37e7, a factor **2.50**. So the arithmetic grew
2.5x while the time grew 7.58x — per-chase cost roughly tripled. The band matrix is
`dim·(band+1)·8` bytes, i.e. **114 MB -> 228 MB** across that step, which crosses out of L3. The band
half is memory-bound, not arithmetic-bound, at any production-sized `dim`.

Two consequences:

- **Every extrapolation to `dim` = 308000 in this document is a LOWER bound**, by the benchmark
  header's own criterion ("if the larger comes out worse than that, DRAM effects have begun and any
  extrapolation is optimistic"). The ~19 h budget below included.
- **It strengthens the case for `-eig-b2`.** Narrowing to 128 shrinks the band matrix from **7.59 GB
  to 318 MB**, so stage 1 attacks the memory traffic as well as the flop count — a saving the
  arithmetic-only model does not capture at all.

Threading remains a loss at both dims (`narrow` vs `narrow-w1`: +4.7% at 4620, +7.8% at 9240), which
is the third independent confirmation of `bandEigDefaultWorkers = 1`.

### Where the eigensolve stands, and why the Householder stage was NOT built

Production budget at `dim` 308000, band 3079, 200 blocks, with `-eig-b2 128 -eig-device`:

| item | h | note |
|---|---|---|
| stage 1 (SBR, host GEMM) | 2.2 | new cost |
| band chase at `b2` = 128 | 2.6 | was ~31.5 h at 3079 |
| `bnd2td` strip, segmented device replay | 1.3 | was ~7 h |
| **`tddiag`'s QL strip** | **13** | **out of scope, and now 68% of the total** |
| | **~19 h** | from ~52 h |

**The plan gated a Householder stage 2 on 0b's measurement, and the measurement kills it.** Stage 2
existed to turn `bnd2td`'s Givens back-transform into blocked WY GEMMs — but the segmented replay
already took that share from ~7 h to ~1.3 h, bit-exactly and with no new numerics, while the design
pass modelled stage 2 at 5.3–11.5 h (the strip traffic gain saturates at 2x while the band-side spine
grows with the block width and is irreducibly serial). It would now cost more than the 1.3 h + 2.6 h it
could possibly address. So it stays unbuilt, which is the outcome the plan anticipated.

What is actually left is `tddiag`: its QL rotations are **chained**, so they are the one case where the
kernel's original "no parallelism beyond rows" claim is true, and the segmented replay cannot touch
them. Shrinking that means replacing the tridiagonal eigensolver — a `dlaed*`-class divide-and-conquer
port whose failure mode is non-orthogonal eigenvectors that do not crash, on a spectrum maximally
clustered by Lanczos ghosts. It is also the one place `bandeig_blocked.go`'s blocking is the right
medicine, since a 33x33 block gives 3080x33 = 101k-way parallelism on a device.

And `-blocks 100` still divides every number above by 4 — the cost goes as `blocks²`, and it is what
theADCcode's reference DIP runs used. It composes with all of this and remains the cheapest lever by a
wide margin.

### No solver library helps here

Neither cuSOLVER nor hipSOLVER/rocSOLVER has a banded eigensolver or a standalone tridiagonal one;
both bind a single dense routine (`cusolverDnDsyevd` / `hipsolverDsyevd`). A dense route would also
throw away the `O(dim·band)` memory that is the whole point of this driver — though not by as much
as once recorded here: the dense projected matrix at `dim` = 308000 is **759 GB**, not 759 TB, which
does fit across 8×H200 (1128 GB). That makes `cusolverMgSyevd` a genuine alternative rather than an
absurd one, and the only option that would remove the chase as well. It is not bit-exact with this
algorithm, needs the 7.59 GB banded matrix expanded 100-fold, and computes all `dim`×`dim`
eigenvectors where Mode B needs ~3080 rows of them.

### Why the CPU threading cannot be scaled up: the barrier caps at 16 cores

The round count is the binding constraint and it cannot be reduced: chases must run in order
(chase `r-1`'s row walk reads rows chase `r`'s column walk wrote), so there are `dim·band` ≈ 9.5e8
of them. The cost is therefore 9.5e8 × whatever one barrier costs, and that was measured rather
than assumed (`BenchmarkFixedPoolRound`, 64-core Helix node, 2 sockets × 32 cores, GOMAXPROCS=64
throughout so the runtime always had spare Ps):

| pool workers | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| round latency | 2.0 µs | 2.4 µs | 2.5 µs | 10 µs | 1.59 ms | 3.85 ms | 3.95 ms |

`parallel.Chunks` over 64 ranges costs 72 µs for comparison. So:

- **the eigensolve runs on 16 cores** (`parallel.FixedPoolWorkers`), not the node. At 2.5 µs the
  barrier costs ~0.8 h over the whole reduction; at 32 workers it would cost 18 days.
- the plan this came from assumed a 1–2 µs barrier at 64 threads and a ~4 h finish. **Neither
  holds**: 64-way costs 3.95 ms a round, and even at the 16-way optimum the threading loses to
  serial (see above).
- this is not false sharing in the benchmark (each range writes its own cache line) and not starved
  `P`s (48 were free). The knee sits below the 32-core socket. Re-measure before raising the cap;
  it is a property of the hardware and the Go runtime, not of the package.

Threading only the rotations has a hard ceiling. Measured with the accumulator switched off
(zero requested rows makes every rotation vanish), the sequential spine is 27–34% of the
runtime — **2.95x** at the tier-1 shape, whatever the core count. That is reproducible as
`BenchmarkBnd2tdSpine` against `BenchmarkBnd2tdFull`, and it is why the band-matrix side is
threaded as well and not just the rotations.

Bit-exactness is tested, not asserted: `bandeig_test.go` holds a frozen verbatim copy of the
pre-change serial code, and `TestBandSymDiagFastBitExactVsReference` compares against it with
`!=` across worker counts and batch sizes. Nothing else in the tree would catch a changed
rounding — there is no golden file for `bandeig`, and the resume test compares two runs of
the same binary.

`Options.EigenProgress` reports the column reached during the reduction, so the stage is no longer
silent.

Measuring it: `scripts/helix/eigensolver_bench.sbatch` (barrier latency, replay throughput vs block
size, tier-1 solve) and `scripts/helix/eigensolver_tier2.sbatch` (production band 3079, the only
shape that reproduces the real accumulated-row width and per-column footprint). The login node's
cgroup caps this account at 4 CPUs while `nproc` reports 128, so nothing measured there says
anything about scaling — every number above comes from a batch job.

## Findings

The implementation work resolved the note's "open design question" (block width /
`n` tiling for a GPU) with a negative result worth recording:

- **The full main-space start block is not optional — it is what carries pole
  strengths.** A block *smaller* than the main space (the GPU "small-block" idea)
  cannot span every pole-carrying direction, so some DIP main lines have *zero*
  overlap with its Krylov space and never appear — not "converge slowly", never
  (measured on H2O: a block-5 run saturates its reachable invariant subspace with
  ~4 of 14 singlet main lines permanently missing, regardless of iteration count).
  This is exactly why theADCcode seeds with the whole main space. So the small-block
  Mode A is sound (exact on what it reaches) but **structurally incomplete for the
  pole-strength band**; it is a device-frugal solver for medium systems / previews,
  not a production band solver.

- **Therefore the production DIP band needs block = main, which needs a fat-memory CPU
  node (Mode B), not a GPU.** Mode B's ≈3·(n×main) ≈ 0.4–0.6 TB fits a 1 TB+ node;
  no block-width shrink or single-GPU trick delivers the complete band. `n`-tiling
  across GPUs (distributed matvec + allreduce) remains the only untried route to a
  GPU full-band solve, and is a much larger undertaking than Mode B.

## Bottom line

- DIP full-band Lanczos on the production system is blocked by ~25–36 TB of *basis* storage —
  algorithmic, not hardware. The low-memory driver removes that: Mode B keeps only
  three n×main panels (~0.4–0.6 TB), runnable on a fat-memory CPU node.
- Near-term production pipeline unchanged: DIP = block-Davidson (lowest roots),
  SIP = full-reorth Lanczos with checkpointing. `-solver lanczos-lowmem` (Mode B) is
  the path to the *whole* DIP band on a fat CPU node.
- The GPU "small-block" idea does **not** yield the complete pole-strength band (see
  Findings); it survives only as Mode A, a device-frugal solver for smaller cases.
