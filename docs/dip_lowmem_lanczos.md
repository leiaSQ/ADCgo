# DIP-ADC(2) low-memory Lanczos — the gap vs. theADCcode

**Status:** design note / future work. Records why ADCgo's full-band DIP-ADC(2)
block-Lanczos is not runnable on the melanin problem while theADCcode's is, and
sketches the low-memory variant (built around the existing `Result.Spurious()`
ghost filter) that would close the gap.

Written 2026-07 while splitting the melanin pipeline into DIP=Davidson /
SIP=Lanczos (see `scripts/HELIX.md`). The immediate melanin DIP runs use
block-Davidson (lowest `-nroots` roots), which sidesteps this entirely; this
note is for whoever later wants the *whole* double-ionization band.

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

## Bottom line

- DIP full-band Lanczos on melanin is blocked by ~25–36 TB of basis storage —
  algorithmic, not hardware. **Do not** file a higher-node project request
  expecting it to run; no node is large enough.
- Near-term: DIP = block-Davidson (lowest roots), SIP = full-reorth Lanczos with
  checkpointing. Already shipped.
- To get the *whole* DIP band: implement `SolveLowMem` (short recurrence + banded
  solver + `Spurious()` filter), mirroring theADCcode. The ghost filter already
  exists (`lanczos.go:559`); the open design question is the block width / `n`
  tiling that keeps 3 panels resident.
