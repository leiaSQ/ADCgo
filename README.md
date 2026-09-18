# ADCgo

An exact, hardware-accelerated **ADC(n) ionization** solver in Go.

ADCgo builds and diagonalizes the algebraic-diagrammatic-construction secular problem
for electron removal *exactly* — no reduced-scaling truncations — and reaches larger
systems through acceleration (multicore OpenBLAS, GPU block-Lanczos via **hipBLAS** on
AMD and **cuBLAS** on NVIDIA — and, at very large scale, one sector row-partitioned across
a whole node's GPUs over NVLink, tested to 8×H200) rather than approximation. SCF and molecular integrals are
delegated: ADCgo ingests a standard **FCIDUMP** (e.g. from pyscf) plus an optional MO
sidecar for the properties that FCIDUMP does not carry (populations, dipoles).

The solver is one CLI, `cmd/adcgo`: the method is chosen by flags; output is JSON on
stdout (or `-out FILE`). A companion CLI, `cmd/plotspec`, renders that JSON to a figure
(PNG/SVG/PDF) — decay-channel, single-ionization, and transition-dipole spectra. See
[Plotting](#plotting).

## What it computes

| Capability | Method | Flags |
|---|---|---|
| Double ionization | DIP-ADC(2) | `-dip` |
| Single ionization | non-Dyson IP-ADC(2) / IP-ADC(3) | `-sip -order 2\|3` |
| Core single ionization | CVS Dyson IP-ADC(4) | `-sip -order 4 -core` |
| Auger / ICD / ETMD spectrum | decay-channel classification | `-spectrum` |
| Bare eigenvalue spectrum | one stick per state (energy + pole strength) | `-bare` |
| Transition dipoles | RASSI-like ion→ion emission, Dyson photoionization, core→valence X-ray emission | `-tdm` |

## Quick start

```sh
# 0. Generate integrals: RHF+MP2 on H2O/cc-pVDZ, C2v (needs pyscf; see below).
#    Writes testdata/h2o.fcidump and the sidecar testdata/h2o.mo.json.
python scripts/gen_fcidump.py

# 1. Sanity: reconstructed HF + MP2 energies from the FCIDUMP.
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump

# 2. Single ionization, IP-ADC(3), one sector per irrep.
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -order 3 -sym all
```

## Methods

### Double ionization — DIP-ADC(2)

Dication states: energies, pole strengths, leading two-hole configurations, and — with
the `-mo` sidecar — atom-resolved two-hole populations (Tarantelli U-transform). One
sector per point-group irrep and spin.

```sh
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip \
    -mo testdata/h2o.mo.json -solver lanczos -spin both -sym all
```

### Single ionization — IP-ADC(2) / IP-ADC(3)

Cation (doublet) states: ionization energies, spectroscopic factors, per-orbital
one-hole overlaps. `-order 2` is the reference's extended ADC(2); `-order 3` is the
non-Dyson IP-ADC(3) (1h main / 2h1p satellite).

```sh
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -order 3 -sym all
```

### Core ionization — CVS Dyson IP-ADC(4)

`-order 4` is core-valence-separated Dyson ADC(4); it requires `-core` naming the
occupied core orbital(s) (0-based). Only the core orbital's irrep has a main block, so
pin it with `-sym`.

```sh
# O 1s of water (orbital 0, a1 sector)
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -order 4 -core 0 -sym 0
```

The bare core diagonal is Koopmans-level; use the solver as-is for relative core-state
structure, not absolute core binding energies.

### Decay-channel spectrum — Auger / ICD / ETMD

Solve and classify in one pass, emitting a stick-spectrum JSON (rendered by
[`cmd/plotspec`](#plotting)). DIP needs `-mo` (channels are built from atom-resolved populations).
`-init-atom` picks the core-ionized site; `-group NAME=col,~col` defines composite or
passive sites (a bare `-group` opens an interactive prompt).

```sh
go run ./cmd/adcgo -fcidump testdata/h2o_dzp.fcidump -dip -mo testdata/h2o_dzp.mo.json \
    -solver dense -sym all -spectrum -init-atom O

# treat both H as a passive "water" site: only Auger@wat survives
go run ./cmd/adcgo -fcidump testdata/h2o_dzp.fcidump -dip -mo testdata/h2o_dzp.mo.json \
    -solver dense -sym all -spectrum -group "wat=O,~H1,~H2" -init-atom wat
```

### Decay widths and lifetimes — Fano-ADC(2,2) `-fano`

Electronic decay *rates*, not just channels: Γ and τ = ℏ/Γ for a chosen vacancy, by the
Fano/Feshbach method with Stieltjes imaging
([Kolorenč & Averbukh, *J. Chem. Phys.* **152**, 214107 (2020)](https://doi.org/10.1063/5.0007912)).
Works over any SIP secular matrix: `-order 2` is Fano-ADC(2)x, `-order 22` the new ADC(2,2)
scheme, whose explicit 3h2p class is what makes second-order decay (double Auger, double ICD)
describable at all. `-adc22 m|x|f` selects the variant; `f` is the paper's recommendation.

```sh
# Ne+ (1s^-1) Auger width. The vacancy fixes the target irrep, so -sym is determined.
go run ./cmd/adcgo -fcidump ne.fcidump -sip -order 22 -adc22 f     -fano -fano-init 0 -sym all -solver lanczos -matfree on

# interatomic decay: Q is every configuration with ALL holes on the donor subunit
go run ./cmd/adcgo -fcidump dimer.fcidump -sip -order 22 -fano     -fano-init 2 -fano-q 2,3,4 -fano-rule all -mo dimer.mo.json -init-atom A
```

`-mo` adds approximate partial widths per channel (Auger@A, ICD:A→B, ETMD, and `double` for
the 3h2p/second-order channel), each imaged separately.

**Basis requirement — the thing that decides whether a run is possible at all.** Imaging can
only evaluate Γ(E_Φ) if the 2h1p pseudo-continuum brackets E_Φ, and a 2h1p state sits at
ε_a − ε_k − ε_l. So what matters is the *energy span and level density of the virtual space*
near E_Φ, not diffuseness. For the Ne 1s vacancy (E_Φ ≈ 32 E_h) aug-cc-pVTZ tops out at
14.6 E_h and cannot describe the decay at all; aug-cc-pVQZ reaches 68.9 E_h. The run log
reports how many P configurations actually carry coupling and warns when there are too few
for the imaging to settle:

| Ne⁺(1s⁻¹), Fano-ADC(2)x | coupled channels | Γ (meV) |
|---|---|---|
| aug-cc-pVQZ | 77 | 642 ± 269 |
| aug-cc-pV5Z | 96 | 316 ± 8 |
| aug-cc-pV5Z + 4s4p4d | 165 | 227 ± 10 |
| published (Table VI) | | 244 ± 4 |

Two diagnostics decide whether a width is trustworthy, and both are printed:

- **coupled channels** — how many P configurations carry any coupling at all. Imaging
  reconstructs a density from these alone, so a few tens gives a large Stieltjes spread
  however well the rest of the pipeline works.
- **sum-rule residual** — the pseudo-continuum's total strength against the exact
  2π‖g‖². It is zero for a complete basis, so a large value means the Krylov space was
  truncated before it captured the coupling. An unconverged width can still look
  reasonable with a *small* error bar, because the Stieltjes orders agree with each other
  about the wrong density — so raise `-fano-blocks` until the residual closes.

`scripts/gen_fano_atoms.py` generates the atomic fixtures and
`scripts/fano_table6.sbatch` runs the comparison on a compute node (necessary: the login
node's user slice caps CPU at 4 cores, so `GOMAXPROCS` is 4 there whatever `nproc` says).

### Bare eigenvalue spectrum — `-bare`

The plain solver output is just eigenvalues (energies + pole strengths), like legacy ADC.
`-bare` turns that list directly into a stick-spectrum JSON — one line per state, energy =
ionization energy, intensity = pole strength (ps/100), all on a single `states` channel —
without any decay-channel or per-orbital classification. It works for both `-dip` and
`-sip`, needs no `-mo` sidecar, and renders through [`cmd/plotspec`](#plotting) exactly like
any other spectrum (one broadened curve). DIP `-spectrum` *without* `-mo` falls back to this
same bare spectrum, since decay channels require atom-resolved populations.

```sh
# bare per-state DIP spectrum (no MO sidecar needed)
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip -bare -solver dense -out bare.json
go run ./cmd/plotspec -in bare.json -out bare.png -fwhm 1.0

# bare per-state SIP spectrum (vs. the per-orbital -spectrum decomposition)
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -bare -out sip_bare.json
```

`-convert FILE` post-processes an **already-emitted** solver document (the default
`-dip`/`-sip` JSON, or the `-out` file from an earlier run) into the same bare spectrum,
without re-solving — the document already carries every state's energy and pole strength.
Pass `-dip` or `-sip` to say which kind of document it is. The result is byte-identical to
running `-bare` on the original problem.

```sh
# solve once, keep the full document...
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip -solver dense -out dip.json
# ...then derive the bare spectrum from that file whenever you need it
go run ./cmd/adcgo -convert dip.json -dip -out bare.json
```

### Transition dipole moments — `-tdm` (`-rassi`)

RASSI-like transition properties along the ICD decay chain, from a single-ionization run.
Requires `-sip` and a `-mo` sidecar carrying dipole integrals. Emits three sections:

- **`emissions`** — ion→ion radiative transitions within a sector (μ, oscillator
  strength *f*, Einstein *A* in s⁻¹). Within one sector only the totally-symmetric dipole
  component connects states.
- **`photoionization`** — each cation state's Dyson orbital contracted with the dipole
  integrals into an L² photoionization pseudo-spectrum μ(ε_a), one channel per virtual
  orbital (the ejected-electron proxy). Discrete strengths; a smooth σ_ion(ω) needs
  Stieltjes imaging (future work).
- **`cross_emissions`** — for `-order 4`, core→valence X-ray emission between the CVS core
  sector and companion plain-ADC(3) valence sectors. Each row reports the state overlap
  `overlap`, which is 0 (and the moment gauge-independent) across different irreps.

```sh
# ion->ion emission + per-state Dyson photoionization
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -order 3 \
    -mo testdata/h2o.mo.json -solver dense -tdm

# CVS run: adds core->valence X-ray emission (O 1s -> outer valence ~522 eV for H2O)
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -order 4 -core 0 -sym 0 \
    -mo testdata/h2o.mo.json -solver dense -tdm
```

## Solvers

Every method above builds the same real-symmetric secular matrix; `-solver` only chooses
how it is diagonalized. All three return identical energies and pole strengths on the
states they resolve — pick by problem size and how much of the spectrum you need.

### `-solver dense`

Forms the full matrix and diagonalizes it directly (LAPACK `dsyev`). Exact and returns
every state, but is O(N³) time / O(N²) memory — use it for small sectors, validation, and
as the correctness oracle for the other two. Default for the examples above.

### `-solver lanczos` (default)

Matrix-free **block-Lanczos**: builds a Krylov subspace from the main-block start vectors
and Rayleigh–Ritz-projects onto it, never storing the matrix. It sweeps the *whole*
ionization band at once, so it is the right tool for a broad spectrum (Auger/ICD, full DIP
band). `-blocks N` sets the subspace size (`N` × main-block, == theADCcode's `iter N`);
more blocks = finer resolution. Because it matches spectral *moments* rather than
individual eigenvalues, interior poles at a fixed `-blocks` can sit at a pole-strength
centroid of a cluster rather than on any one true root.

```sh
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip -solver lanczos -blocks 100
```

### `-solver davidson`

Matrix-free **block Davidson–Liu**: root-targets the algebraically lowest `-nroots`
eigenpairs, iterating each to a residual threshold with a diagonal `(θ−D)⁻¹`
preconditioner. When you want a handful of converged interior eigenvalues (e.g. the lowest
~20 core-edge roots) rather than a broad envelope, it hits the exact positions at a
fraction of the Lanczos subspace size — this is what reproduces a legacy `adc4_diag.x`
Davidson run directly. Flags: `-nroots` (roots to converge), `-convthr` (residual 2-norm
threshold, a.u.), `-maxdavsp` (subspace cap before a thick restart), `-maxdavit`
(iteration cap). Works for both `-sip` (including CVS `-order 4`) and `-dip`.

```sh
# lowest 8 O 1s core roots of a CVS-ADC(4) run, converged to 1e-3 a.u.
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -order 4 -core 0 -sym 0 \
    -solver davidson -nroots 8 -convthr 1e-3
```

### `-solver lanczos-lowmem`

The same block-Lanczos band, re-cast to keep only a handful of Krylov panels resident
instead of the whole basis — the memory mode that puts the full DIP band of large systems
within reach of a GPU. `-lowmem-block 0` (the default width) is the faithful theADCcode
short recurrence: block width == the 2h main-space size, a Tarantelli subspace-iteration
gate plus a banded eigensolver, with only ~4 n×main panels live at once. A `-lowmem-block`
*below* `main` selects a device-frugal full-reorthogonalization variant instead (3 blocks on
the GPU, the full basis staged in host RAM) — exact on the states it reaches, but a block
narrower than `main` cannot span every pole-carrying direction.

```sh
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip -solver lanczos-lowmem -sym all
```

### Distributed multi-GPU — `-mgpu`

At production scale one whole-band Mode B Krylov block can dwarf a single GPU (≈137 GB for
the production system, past a 141 GB H200). `-mgpu N` **row-partitions one sector across N GPUs**: the
resident state — the live n×main Krylov panels and the block-sparse operator — is split along
the config (row) dimension, so a block that fits nowhere alone fits spread over a node's pool.
Every solver reduction (the α coefficients, the CGS2 projection, the Gram, dots and norms)
contracts the row dimension, so each becomes a device-local partial plus a tiny all-reduce;
only the mat-vec crosses devices, gathering the remote input band per apply — over **NVLink**
(peer-to-peer copy) when the backend supports it, else staged through the host. It scales to
a full 8×H200 NVLink/NVSwitch node.

`-mgpu` requires `-dip -solver lanczos-lowmem -lowmem-block 0` and a fast inter-GPU link.
Sectors run **serially**, each spanning the whole pool — in contrast to `-gpus`, which runs
*independent* sectors concurrently, one GPU each. Row-partitioning divides the operator by the
pool size (≤ 8), so the multi-TB satellite region of a whole-band production run still overflows the
node — pair `-mgpu` with [`-matfree on`](#matrix-free-operator---matfree), which recomputes that
region instead of storing it. Build the CUDA (or HIP) binary, then:

```sh
# row-partition each DIP sector of the production system across all 8 H200 of an NVLink node
adcgo-cuda -fcidump system.fcidump -dip -order 2 \
    -solver lanczos-lowmem -lowmem-block 0 -mgpu 8 \
    -backend cuda -spin both -sym all -blocks 200
```

See [`scripts/HELIX.md`](scripts/HELIX.md) for a complete SLURM job
(`--gres=gpu:H200:8`).

### Matrix-free operator — `-matfree`

By default the block-sparse ADC operator is **materialized**: every nonzero block is assembled
once and kept resident, so each mat-vec is a batched GEMM. For a large sector the dominant
blocks — the DIP 3h1p↔3h1p satellite region, the SIP ADC(4) 2h1p×3h2p / ADC(3) 2h1p² coupling —
are the resident-memory ceiling: hundreds of GB to several TB, larger than a whole 8×H200 node
for the production system. `-matfree on` **recomputes those blocks on the fly from the MO integrals each
mat-vec and never stores them**, collapsing the resident footprint to the Krylov panels plus the
small main/coupling blocks (the direct-σ approach theADCcode uses). `-matfree auto` decides per
block by dense size against `-maxmem`; `off` (default) keeps everything materialized.

It trades recompute per mat-vec for the removed memory ceiling, and runs everywhere the dense
path does: on the host, on a GPU (`-backend cuda`, a custom recompute kernel reading a
device-resident ERI tensor), and composed with `-mgpu` (the partitioned dense blocks and panels
stay on-device while the satellite region is recomputed). This is what puts a **whole-band**
DIP or SIP run of a large system on a single node without dropping polarization or freezing extra
orbitals.

```sh
# whole-band DIP at production scale: matrix-free satellite region, row-partitioned across 8 GPUs
adcgo-cuda -fcidump system.fcidump -dip -order 2 \
    -solver lanczos-lowmem -lowmem-block 0 -mgpu 8 -matfree on \
    -backend cuda -spin both -sym all -blocks 200
```

## Plotting

> **[`cmd/adcanalysis/README.md`](cmd/adcanalysis/README.md) is the full reference**
> for the analysis and plotting pipeline: every `ADCanalysis` and `plotspec` flag,
> the spectrum JSON schema that joins them, and what each rendering knob actually
> does. The section below is the summary.

`cmd/adcgo` writes JSON; `cmd/plotspec` turns that JSON into a figure. The output format
follows the `-out` extension (`.png` / `.svg` / `.pdf`). By default each channel is
Gaussian-broadened onto a shared grid and drawn as one curve; `-stick` draws bare sticks
instead, `-overlay-broadened` draws the curves on top of those sticks, and `-stick-height F`
scales the sticks (sticks and curves are normalised separately, so `-stick-height 0.6` keeps
the bars under the envelope). Sticks are normalised to the tallest one *in the plotted
window*, so `-xrange` zooms rescale them. The mode is picked with `-mode`:

| `-mode` | Input | Plots |
|---|---|---|
| `spectrum` (default) | a `-spectrum` JSON (`-in`) | one broadened curve per decay channel (DIP) or per orbital (SIP); axis/title switch on `meta.kind` |
| `tdm` | a `-tdm` JSON (`-in`) | the **transition-dipole spectrum** — peaks at each transition energy, height = oscillator strength |
| `ees` | a SIP + a DIP JSON (`-sip`, `-dip`) | electron-emission spectrum σ(ε) = ∫ S_in(E)·S_fin(E−ε)/N(E) dE |
| `panel` | SIP + DIP JSON | 3-panel composite (SIP sticks, DIP sticks, EES) |
| `stack` | several spectrum JSONs (`-stack` / `-stack-manifest`) | **energy-offset waterfall** — one envelope per conformer, baselined at its relative energy |

```sh
# Decay-channel spectrum (Auger/ICD/ETMD)
go run ./cmd/plotspec -in spec.json -out spectrum.png -fwhm 1.2

# Transition-dipole spectrum from a -tdm run
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -order 3 \
    -mo testdata/h2o.mo.json -solver dense -tdm -out tdm.json
go run ./cmd/plotspec -mode tdm -in tdm.json -out tdm.png
go run ./cmd/plotspec -mode tdm -in tdm.json -out tdm_sticks.png -stick -xrange 500-560

# Sticks with the broadened envelope over them, sticks scaled to 70% of the curve
go run ./cmd/plotspec -mode tdm -in tdm.json -out tdm_both.png \
    -stick -overlay-broadened -stick-height 0.7 -fwhm 1.5 -xrange 500-560
```

### `-mode stack`

An energy-offset waterfall across the conformers of one structure search. Each conformer
contributes a single Gaussian-broadened **total** envelope (the channel decomposition is
dropped — that detail belongs to the per-conformer figures), and the *baseline* of that
envelope sits at the conformer's relative electronic energy, so vertical position reads
directly as "how much less stable is this basin". It is the stacked convention of an
NMR/IR series, with relative energy in place of time or temperature.

Traces come from `-stack LABEL=PATH[@OFFSET]` (repeatable) and/or from a run manifest:

```sh
# One trace per run in the manifest whose spectrum exists, 1-water subset only.
go run ./cmd/plotspec -mode stack \
    -stack-manifest sip_manifest.json -stack-dir specs -stack-group 1 \
    -xrange 8-40 -fwhm 0.8 -stack-dedup 0.02 -out conformers_1W.pdf
```

`-stack-manifest` reads a JSON object mapping each run name to its metadata; the run's
total energy comes from `-stack-energy-key` (default `relaxed_energy`, in
`-manifest-unit`, default hartree), is re-referenced to the most stable run **that has a
spectrum**, and is plotted in `-offset-unit` (default `kcal`). `-stack-group-key` /
`-stack-group` restrict the manifest to one subset — one hydration level, say. Spectra
are looked for at `<-stack-dir>/<name><-stack-suffix>`; a run whose calculation has not
landed yet is reported on stderr and omitted rather than failing the figure.

Each trace is named on the y axis, with the conformer's name on a second line
directly under its relative energy, and carries a **structure inset** in its top-right
corner: a 2D ball-and-stick depiction generated from the geometry the calculation ran on,
so it cannot disagree with it. Geometry is looked for beside the spectrum
(`-stack-geom-suffix`, default `.zmat`) as a GAMESS-UK Z-matrix or an XYZ; `-stack-inset`
sets its width as a fraction of the energy axis, `0` turns it off, and a run without a
readable geometry just goes without a picture. `-stack-inset-height` and
`-stack-inset-lift` size it and set how far its bottom edge sits above that trace's
baseline, both in units of the trace amplitude — the default lift keeps each structure
low, floating just over its own curve rather than up near the trace above it.

The molecule is flattened onto its own best-fit plane (the two leading principal axes of
the atoms), which is the undistorted view for these near-planar systems and needs no
per-conformer tuning; solvent fragments are then oriented to the upper right so the same
motif is drawn the same way up in every panel. Atoms are CPK balls — O red, N blue, C dark
grey, H light grey, radius following the covalent radius — joined by sticks in the trace's
own colour, with hydrogen bonds dashed. Bonds come from covalent radii (×1.3) and contacts
from the 1.6–2.2 Å window, both HeidelBIRDS' conventions, so the topology drawn is the
topology the structure search used.

Decluttering, in the order it matters:

- `-stack-dedup TOL` folds traces whose normalised curves agree everywhere to within
  `TOL` into their most stable member, which keeps the rest as aliases and is labelled
  `name (xN)`. A search reaches the same minimum from several starting guesses, so the
  raw basin list holds duplicates that would otherwise draw on top of each other and
  overstate how many distinct basins were found. Default `0` — every basin is kept.
- Isoenergetic traces that survive dedup are nudged apart by `-stack-min-sep` (a fraction
  of the mean spacing). The y ticks stay on the *true* energies; the nudge is a drawing
  device only.
- `-stack-trace-scale` sets trace height in units of the mean baseline spacing; `>1`
  lets neighbours overlap, ridgeline-style. The least stable trace is drawn first, so the
  global minimum is painted last and is never occluded.
- `-stack-norm each` (default) normalises every trace to its own maximum, comparing
  *shape*; `-stack-norm common` puts them on one scale, which also shows relative yield.
- Traces are coloured by a single sequential ramp keyed to relative energy (dark = most
  stable), not by the categorical per-channel palette: they differ by a magnitude, not an
  identity, and twenty categorical hues would be a legend explosion. Labels are drawn
  directly on each trace for the same reason.

### `-mode tdm`

Flattens a `-tdm` document into a stick spectrum: the x-position of each line is the
photon energy `omega_ev`, the height is the oscillator strength `osc`, and the three
transition families become the plotted channels — **`emission`** (ion→ion),
**`cross-emission`** (core→valence X-ray, `-order 4`), and **`photoionization`** (per-virtual
Dyson channels). Dipole-forbidden lines (`osc ≤ 0`) are dropped. All the shared rendering
controls apply: `-fwhm`, `-stick`, `-overlay-broadened`, `-stick-height`, `-xrange`,
`-absolute`, `-colorblind`, and the raster `-width` / `-height` / `-dpi`.

Common `plotspec` flags: `-in` / `-out`, `-fwhm F` (broadening FWHM, eV), `-stick`,
`-overlay-broadened` (curves over the sticks), `-stick-height F` (scale the sticks),
`-xrange LO-HI`, `-absolute` (raw instead of tallest-peak = 1), `-exp FILE` (dotted
reference overlay, `spectrum` mode), `-colorblind` (Okabe–Ito palette). Reference spectra
for overlays live in [`testdata/reference/spectra/`](testdata/reference/spectra).

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `-fcidump PATH` | — | FCIDUMP with MO integrals (required) |
| `-dip` | off | solve DIP-ADC(2) |
| `-sip` | off | solve IP-ADC(n) |
| `-order N` | 3 | SIP order: 2, 3, or 4 (4 = CVS Dyson ADC(4), needs `-core`) |
| `-core LIST` | — | CVS core orbitals for `-order 4`: comma-separated 0-based occupied indices |
| `-sym SEL` | all | target irrep: `all` \| `none` \| 0-based index |
| `-spin SEL` | both | DIP spin sector: `both` \| `singlet` \| `triplet` |
| `-mo PATH` | — | MO/overlap/dipole sidecar (needed by populations, `-spectrum -dip`, `-tdm`) |
| `-solver S` | lanczos | `lanczos` (whole-band) \| `lanczos-lowmem` (memory-frugal band, Mode B) \| `davidson` (root-targeting) \| `dense` (full diagonalization) |
| `-blocks N` | 100 | block-Lanczos iterations; Krylov dim = N × 2h-space size |
| `-lowmem-block N` | 0 | `-solver lanczos-lowmem` block width; 0 = 2h main-space size (faithful short recurrence), `< main` = device-frugal full-reorthogonalization mode |
| `-backend B` | gonum | `gonum` \| `hip` \| `cuda` \| `auto` (build-tag gated) |
| `-gpus N` | 0 | `-backend cuda\|hip`: max GPUs for concurrent per-sector solves (0 = all visible) |
| `-mgpu N` | 0 | `-dip -solver lanczos-lowmem -lowmem-block 0`: row-partition ONE sector across N GPUs (needs NVLink); sectors run serially |
| `-matfree M` | off | recompute the memory-dominant operator blocks each mat-vec instead of storing them: `off` \| `auto` \| `on`. DIP 3h1p↔3h1p satellite region and SIP ADC(4)/ADC(3) coupling; host, GPU (`-backend cuda`), and `-mgpu` |
| `-maxmem GB` | 4 | `-matfree auto` threshold: a block whose dense size exceeds this many GB is applied matrix-free |
| `-ps-thresh P` | 1.0 | drop states with pole strength below P percent |
| `-coeff-thresh C` | 0.1 | drop leading components with \|coeff\| below C |
| `-spectrum` | off | emit a stick spectrum: decay channels (DIP + `-mo`) or per orbital (SIP); DIP without `-mo` falls back to `-bare` |
| `-bare` | off | emit a bare per-state stick spectrum (energy + pole strength, one `states` channel); implies `-spectrum` |
| `-convert FILE` | — | convert an existing `-dip`/`-sip` solver document JSON into its bare spectrum (no re-solve); needs `-dip` or `-sip` |
| `-init-atom A` | O | initial core-ionized site (spectrum) |
| `-group SPEC` | — | decay-site grouping `NAME=col,~col` (repeatable; bare = interactive) |
| `-min-weight` / `-min-fraction` / `-include-zero` | 0 / 0 / off | channel thresholds (spectrum) |
| `-st-ratio R` | 3.0 | singlet:triplet ratio recorded in spectrum meta |
| `-tdm` (`-rassi`) | off | emit transition dipole moments (needs `-sip -mo`) |
| `-tdm-osc-thresh T` | 1e-6 | drop photoionization channels below oscillator strength T |
| `-out PATH` | stdout | write JSON here |
| `-profile` | off | per-sector solver phase timings to stderr |

## Backends

Default is pure-Go (`gonum`); the accelerated backends are build-tag gated and selected
at runtime with `-backend`.

```sh
go run -tags openblas ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip -sym all   # multicore CPU
HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  go run -tags hip ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip -backend hip -sym 0 -spin singlet
go build -tags cuda ./...   # cuBLAS: compiles here, run on an NVIDIA host
```

With `-backend auto` the solver calibrates each available backend once and picks the
predicted-fastest per sector (measuring the real mat-vec cost, not a flop estimate).

On a multi-GPU node there are two independent parallelism axes: `-gpus N` runs *independent*
sectors concurrently (one GPU per DIP spin×irrep or SIP irrep), while `-mgpu N` row-partitions
a *single* sector across N GPUs for a whole-band block too large for one device — see
[Distributed multi-GPU](#distributed-multi-gpu---mgpu).

## Tests

```sh
go test ./...          # full validation (slow gates included)
go test -short ./...   # fast subset
HSA_OVERRIDE_GFX_VERSION=11.0.0 go test -tags hip ./...   # + GPU cross-backend gates
```

Validation is layered: MP2 energy reconstruction (M0); DIP cross-checked against
theADCcode's `adcdip*.out` on matched DZP+diffuse integrals (M4); SIP against pyscf's
`ip_adc` on the same integrals (M5); the CVS ADC(4) blocks bit-exact against theADCcode's
B2 tape; and the transition-dipole machinery against hermetic Slater–Condon determinant
oracles.

## Regenerating fixtures (needs pyscf)

```sh
python scripts/gen_fcidump.py       # h2o.fcidump + h2o.mo.json + h2o.ref.json
python scripts/gen_ref_fcidump.py   # matched DZP+diffuse integrals for the M4 DIP gate
python scripts/gen_sip_ref.py       # pyscf IP-ADC + Dyson reference (M5)
```

The committed ADCgo output fixtures are regenerated with the corresponding `-out` runs;
do **not** regenerate the FCIDUMPs to add a sidecar key (it moves ~110 near-zero
integrals by ~1e-13 and breaks the bit-exact gates — use the scripts' `--sidecar-only`
path instead).
