# ADCgo

An exact, hardware-accelerated **ADC(n) ionization** solver in Go.

ADCgo builds and diagonalizes the algebraic-diagrammatic-construction secular problem
for electron removal *exactly* — no reduced-scaling truncations — and reaches larger
systems through acceleration (multicore OpenBLAS, GPU block-Lanczos via **hipBLAS** on
AMD and **cuBLAS** on NVIDIA) rather than approximation. SCF and molecular integrals are
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

## Plotting

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
| `-solver S` | lanczos | `lanczos` (band) or `dense` (full diagonalization) |
| `-blocks N` | 100 | block-Lanczos iterations; Krylov dim = N × 2h-space size |
| `-backend B` | gonum | `gonum` \| `hip` \| `cuda` \| `auto` (build-tag gated) |
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
