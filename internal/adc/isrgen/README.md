# isrgen — generated ISR matrix elements, SIP through QIP

**Symbolically derived secular-matrix elements for k-fold ionization ADC**, from single (k = 1)
to quadruple (k = 4) ionization. `scripts/codegen/generate_adc.py` derives the intermediate-state
representation (ISR) blocks with [adcgen](https://github.com/jonasleitner/adcgen) and writes
them as Go element evaluators, one package per variant:

| Package | Variant | k | Classes |
|---|---|---|---|
| `isrgen/ip` | single ionization | 1 | 1h \| 2h1p \| 3h2p |
| `isrgen/dip` | double ionization | 2 | 2h \| 3h1p \| 4h2p |
| `isrgen/tip` | triple ionization | 3 | 3h \| 4h1p \| 5h2p |
| `isrgen/qip` | quadruple ionization | 4 | 4h \| 5h1p \| 6h2p |

The packages are **reference evaluators, not production operators**. You call
`Element(sp, r, c)` on the rows of a spin-orbital `khci.Space`. The packages have three uses:

- checking the hand-ported `sip` and `dip` against an independent derivation;
- supplying blocks that have no hand code: tip, qip, and ADC(2,2)-QUIP (`internal/adc/quip`);
- building small dense matrices.

The production solvers (`-sip`, `-dip`, GPU, `-mgpu`, `-matfree`) do not use them. See
[Relation to `sip` and `dip`](#relation-to-sip-and-dip).

## What is committed

`GeneratedOrders` lists the highest order generated per block. The blocks are B00 B01 B11 B02
B12 B22, where 0/1/2 are the main, satellite and double-satellite class (`–` means the block
is not generated).

| Package | Schemes | B00 | B01 | B11 | B02 | B12 | B22 | Slowest block |
|---|---|---|---|---|---|---|---|---|
| `ip` | strict:3, adc22m/x/f | 3 | 2 | 2 | – | 1 | 1 | B00: 43 s |
| `dip` | strict:2, adc2x, ci | 2 | 1 | 1 | 1 | 1 | 1 | B22: 76 s |
| `tip` | strict:2, adc2x, ci | 2 | 1 | 1 | – | 1 | – | B12: 134 s |
| `qip` | strict:2, adc2x, ci | 2 | 1 | 1 | – | – | – | B00: 338 s |

The derivation times per block are in each package's `types.go` header. tip and qip stop
short of the (k+2)h2p/(k+2)h2p block: tip's B22 at first order alone did not finish in
10 min. Configurations in that class are handled by khci's Slater–Condon path instead.

## Quick start

The generator needs Python ≥ 3.10 with adcgen, sympy and numpy, plus `gofmt` in `PATH`. On
HELIX that Python is `~/miniconda3/envs/adcgo/bin/python`, and adcgen is found at
`../adcgen` or via `ADCGEN_DIR`.

```sh
# regenerate one package (writes the Go files, gofmt's them, then the numpy reference)
python scripts/codegen/generate_adc.py --variant dip --schemes adc2x,ci,strict:2 \
    --outdir internal/adc/isrgen/dip

# only some blocks, e.g. a timing probe for one block
python scripts/codegen/generate_adc.py --variant qip --schemes adc2x --blocks B00 \
    --outdir /tmp/qip_probe

# only the numpy/Slater-Condon reference JSON, no Go
python scripts/codegen/generate_adc.py --variant tip --schemes ci --reference-only tip_ref.json

# check the committed packages
go test ./internal/adc/isrgen/...
```

Anything beyond ip or dip should run on a compute node. Each generation is single-threaded
sympy, and the login node is capped at 4 CPUs. `scripts/codegen/isrgen.sbatch` runs the variants
side by side, one log each (`isrgen-<variant>-<job>.log`):

```sh
sbatch scripts/codegen/isrgen.sbatch                  # ip dip tip qip
VARIANTS="tip qip" sbatch scripts/codegen/isrgen.sbatch
```

## Schemes

A scheme sets a **maximum perturbation order for each block**. `--schemes` takes a
comma-separated union, and the package is generated at the highest order any listed scheme
needs. At run time, `New(…, scheme)` truncates to one scheme; a block that scheme leaves out
evaluates to zero.

| Scheme | B00 | B01 | B11 | B02 | B12 | B22 | Meaning |
|---|---|---|---|---|---|---|---|
| `strict:N` | from adcgen's `SecularMatrix.block_order(N)` | | | | | | strict ISR-ADC(N). At N = 2: 2, 1, 0 on the first three blocks |
| `ci` | 1 | 1 | 1 | 1 | 1 | 1 | every block through first order |
| `adc2x` | 2 | 1 | 1 | – | – | – | strict:2 with B11 at first order, the extended ADC(2) |
| `adc22m` | 2 | 1 | 2 | – | 1 | 0 | ADC(2,2) of Kolorenč & Averbukh, Table I |
| `adc22x` | 2 | 1 | 2 | – | 1 | 1 | |
| `adc22f` | 2 | 2 | 2 | – | 1 | 1 | the paper's recommendation |

The adc22 schemes follow `internal/adc/sip/elements22.go`.

**`ci` is not CI in B02.** A kh and a (k+2)h2p configuration differ by a double excitation.
So plain CI couples them at first order: for dip, 0.055 E_h on the reference system. In the
ISR, the first-order ground-state correction cancels that coupling, and B02 starts at second
order (the C13 of IP-ADC). Every other block at first order is exactly CI, and the generator
asserts this.

## Conventions

- **Spin orbitals** over the spatial FCIDUMP integrals.
  - Spin orbital `p` has spatial orbital `p>>1` and spin `p&1`.
  - With `nocc` doubly occupied spatial orbitals, occupied spin orbitals are
    `0 … 2·nocc−1`, and virtual `a` (counted from 0) has spin-orbital index `2·nocc+a`.
  - Antisymmetrized integrals: ⟨pq‖rs⟩ = (pr|qs)δδ − (ps|qr)δδ.
- **Canonical Hartree–Fock only.** The expressions are simplified for a diagonal Fock
  matrix, and `New` takes the orbital energies. Localized or other non-canonical dumps (the
  sidecar flag `canonical: false`) go through the rotated representation of
  [`internal/adc/quip`](../quip/quip.go) (`Rotations`, `Rotation`, `Rotate`).
- **Row phase.** A row is |r⟩ = c†_{a1} c†_{a2} c_{i1} … c_{in} |Φ₀⟩, with ascending indices.
  This is adcgen's precursor-operator order and `khci`'s row convention. The first-order =
  CI check fixes it to ≤ 1.8e-15.
- **No spin adaptation.** A `khci.Space` holds one Ms sector (`TwoMs`) of spin-orbital
  determinants, so singlets and triplets come out together. Point-group blocking is
  available through `khci.Options.OrbSym`/`TargetIrrep`.

## Using a package

```go
import (
    "github.com/leiaSQ/ADCgo/internal/adc/isrgen/dip"
    "github.com/leiaSQ/ADCgo/internal/adc/khci"
)

// Ms = 0 two-hole space through the 3h1p class, symmetry off
sp, err := khci.NewSpace(khci.Options{K: 2, NOcc: nocc, NVir: norb - nocc, MaxClass: 3})
el, err := dip.New(ints, eps, nocc, "adc2x") // ints: *integrals.Store, eps: orbital energies
m := el.BuildDense(sp)                        // backend.Mat, rows in parallel
x := el.Element(sp, r, c)                     // one element <r|M|c>
```

| Symbol | Meaning |
|---|---|
| `Variant`, `K` | the variant name and main-class hole count |
| `GeneratedOrders` | highest generated order per block (−1: not generated) |
| `Schemes` | per-block maximum order of every scheme the package carries |
| `New(ints, eps, nocc, scheme)` | builds the intermediates once. `nocc` counts spatial orbitals; a scheme the package does not carry is an error |
| `(*Elem).Element(sp, r, c)` | ⟨r\|M\|c⟩. Symmetric; the lower triangle is read through the upper |
| `(*Elem).BuildDense(sp)` | the full matrix over `sp`, rows in parallel |

`Elem` is read-only after `New` and safe for concurrent use.

## How it is verified

Every package carries the first three layers of checks; ip and dip also carry the fourth.

1. **Fidelity reference** (`testdata/adcgen_ref.json`, replayed by the generated
   `fidelity_test.go` to 1e-12).
   - The reference system is a random canonical system with 6 orbitals, 3 of them occupied.
   - At probe configurations, every generated block and order is evaluated in numpy from
     adcgen's **fully expanded** expression, with no intermediates. That makes it independent
     of the Go translation.
   - A one-term mutation (×1.001 in tip B11) is caught at 3.6e-5.
2. **ISR(≤1) = CI.**
   - The generator evaluates Slater–Condon CI in the row phase convention.
   - It asserts that the first-order truncation equals CI on every block except B02, and
     refuses to write the reference otherwise.
   - The CI values of every block (`ci_entries`) are stored for khci's own gate.
3. **Symmetry.** `TestSymmetricAndMsDegenerate` checks that the matrix is symmetric and that
   Ms-partner spectra coincide.
4. **Hand-written crosschecks**, in `crosscheck_test.go`, which survives regeneration.

| Test | Pins |
|---|---|
| `ip` `TestMainBlockMatchesSipOrder2` | the 1h/1h block through order 2, element by element, against `sip -order 2` |
| `ip` `TestCouplingMatchesSipOrder3` | with sip's main block swapped in, every `sip -order 3` eigenvalue is in the strict:3 spectrum (C11⁽³⁾ differs by design: sip applies Σ(∞) outside the matrix) |
| `ip` `TestADC22MatchesSip` | every `sip -order 22` doublet (m/x/f) is in the generated adc22 spectrum, to ≤ 1.6e-14. This was the first independent check of the hand-transcribed A9–A22, which have no reference implementation |
| `ip` `TestADC22ElementsMatchSip` | **every** `sip -order 22` element, m/x/f, in every block (1h/1h … 3h2p/3h2p, A9–A22 included), to ≤ 3.6e-15. Each sip row is expanded over determinants with `sip.Space.DetExpansion`. Swapping variants is caught: c12_2 by 2.0e-3, 3h2p/3h2p first order by 0.39 |
| `dip` `TestSpectraMatchDipPackage` | singlet ∪ triplet `dip` spectra are in the generated adc2x Ms = 0 spectrum, to 2.8e-13. strict:2 is required to miss: it misses by 3.5 eV, so `dip` *is* ISR adc2x |
| `dip` `TestSpinAdaptedBlocksMatchDipPackage` | each singlet/triplet block of `dip` between two spatial configurations. Its singular values equal those of the generated adc2x block projected onto the S² eigenspace of the same configurations, to ≤ 2.8e-14, and every configuration has as many `dip` rows as spin functions. These are the invariants of the unknown rotation among Tarantelli's spin functions. strict:2 fails by 5.7, in 3h1p/3h1p only |

## Costs

Derivation cost grows steeply with k and with order. The times below are per order, taken from the generation logs of
jobs 14890980 and 14890982 on HELIX compute nodes. The entries marked * come from earlier login-node probes that hit their
time cap:

| Block | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|
| B00, order 2 | 1.7 s | 4.4 s | 24 s | 331 s |
| B11, order 1 | 0.4 s | 1.3 s | 8.6 s | 192 s |
| B12, order 1 | 2.3 s | 10 s | 124 s | — |
| B22, order 1 | 5.9 s | 49 s | > 10 min* | — |

For k = 1, B00 at order 3 takes 41 s; order 4 did not finish in 8 min*.

adcgen's permutational-symmetry search (`exploit_perm_sym`) is factorial in the number of
target indices. It is skipped above 7! = 5040 target permutations (`PERM_SEARCH_MAX`); that
covers tip B11/B12, qip, and dip B22. Those terms are emitted as they are: more Go, but the
same values.

## Relation to `sip` and `dip`

The hand-ported `sip` and `dip` packages remain the production matrices, for four reasons.

- They are **bit-exact against theADCcode** on matched integrals.
- They apply spin-adapted blocks as GEMMs, CUDA kernels and row-partitioned multi-GPU panels.
  Generated code evaluates one spin-orbital element at a time.
- They cover what an ISR derivation cannot: Dyson CVS IP-ADC(4), and the Σ(∞) treatment of
  sip's C11⁽³⁾.
- They carry the reference's conventions: sip `-order 2` is ADC(2)x.

The generated packages are the **second source** that checks those matrices, and the **only
source** for variants with no hand code. A block-level replacement would need a σ-building
(contraction) back end for the generator.

## Limits

- **Element by element.** There are no σ-builds. Use the packages for dense blocks and
  probes, not for matrix-free solves.
- **Canonical HF only** (see [Conventions](#conventions)).
- **Missing top blocks.** tip and qip have no (k+2)h2p/(k+2)h2p block. qip has no B12, and
  the second-order 5h1p/5h1p block that ADC(2,2)-QUIP needs has not been derived.
- **Generated files are not for editing.** They carry `DO NOT EDIT`. Regenerating replaces
  every `*_generated.go`, `types.go`, `doc.go`, `fidelity_test.go` and the reference.
  `crosscheck_test.go` is hand-written and kept.
- **Target index names.** Names are drawn from fixed pools, bare letters first:
  `i…o, i1…o1` for holes and `a…h` for particles. Numbered names break adcgen's
  intermediate factoring at order 3, so they are used only on overflow.
