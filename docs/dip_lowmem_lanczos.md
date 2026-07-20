# DIP-ADC(2) low-memory Lanczos — the gap vs. theADCcode

**Status:** IMPLEMENTED (`-solver lanczos-lowmem`, `internal/adc/lanczos/lowmem.go`
+ `bandeig.go`). This note originally recorded why ADCgo's full-band DIP-ADC(2)
block-Lanczos was not runnable on the melanin problem while theADCcode's was, and
sketched the low-memory variant. That variant now exists; the sections below keep
the original analysis (still the correct explanation of the gap) and the
[Implementation](#what-was-implemented) / [Findings](#findings) sections at the end
record what shipped and what the work discovered.

Written 2026-07 while splitting the melanin pipeline into DIP=Davidson /
SIP=Lanczos (see `scripts/HELIX.md`). The immediate melanin DIP runs use
block-Davidson (lowest `-nroots` roots); the low-memory solver is for whoever
wants the *whole* double-ionization band.

## The gap in one line

The DIP-ADC(2) full-band solve is blocked by an **algorithmic** memory cost, not
a hardware ceiling. A bigger node does not fix it — the current solver stores the
entire Krylov basis, and for melanin that basis is tens of terabytes. theADCcode
runs the same physics because it uses a **limited-memory** Lanczos (short
recurrence + banded tridiagonal solver + ghost filtering) that never stores the
full basis.

## Why ADCgo's `Solve` cannot run melanin DIP

`internal/adc/lanczos/lanczos.go`, `func Solve` (line 321), is a block-Krylov
Rayleigh–Ritz with **full reorthogonalization + deflation** (package comment,
lines 6–12). Full reorth means every Krylov vector must be kept for the life of
the run so each new block can be orthogonalized against all of them. The whole
basis is therefore allocated up front on the device:

```go
bbuf := be.Alloc(n * maxdim)   // lanczos.go:332  — the entire basis panel
```

with `maxdim = blocks · main`. That single allocation is the memory driver.

Measured melanin DIP spaces (real space builders, `-blocks 200`):

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
  extracts the Ritz values.
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
   device (3·`n·main`, ~410 GB for melanin — still too big for one GPU at
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
  Tarantelli's `bnd2td`+`tddiag` that materializes only the `2·band` top/bottom
  eigenvector rows), and the `Result.Spurious(1e-9)` ghost filter. Resident cost
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
(`dip/satellite_test.go`).

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
  not a melanin band solver.

- **Therefore the melanin DIP band needs block = main, which needs a fat-memory CPU
  node (Mode B), not a GPU.** Mode B's ≈3·(n×main) ≈ 0.4–0.6 TB fits a 1 TB+ node;
  no block-width shrink or single-GPU trick delivers the complete band. `n`-tiling
  across GPUs (distributed matvec + allreduce) remains the only untried route to a
  GPU full-band solve, and is a much larger undertaking than Mode B.

## Bottom line

- DIP full-band Lanczos on melanin is blocked by ~25–36 TB of *basis* storage —
  algorithmic, not hardware. The low-memory driver removes that: Mode B keeps only
  three n×main panels (~0.4–0.6 TB), runnable on a fat-memory CPU node.
- Near-term melanin pipeline unchanged: DIP = block-Davidson (lowest roots),
  SIP = full-reorth Lanczos with checkpointing. `-solver lanczos-lowmem` (Mode B) is
  the path to the *whole* DIP band on a fat CPU node.
- The GPU "small-block" idea does **not** yield the complete pole-strength band (see
  Findings); it survives only as Mode A, a device-frugal solver for smaller cases.
