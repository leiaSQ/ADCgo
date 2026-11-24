# ADCgo

An exact, hardware-accelerated **ADC ionization** solver in Go.

Developed by Leia Wertebach (Meyer) & Alexander Kuleff. Derived from theADCcode of TC Group Heidelberg.

ADCgo builds and diagonalizes the algebraic-diagrammatic-construction secular problem for
electron removal *exactly* — no reduced-scaling truncations — and reaches large systems through
acceleration rather than approximation: multicore CPUs, GPU block-Lanczos on NVIDIA (cuBLAS) and
AMD (hipBLAS), and one sector row-partitioned across a whole node's GPUs over NVLink (tested to
8×H200). SCF and integrals are delegated: ADCgo reads a standard **FCIDUMP** (e.g. from pyscf)
plus an optional MO sidecar for what FCIDUMP does not carry (MO coefficients, overlap, dipoles).

## Version 0.1

| Method | Flags | Validated against |
|---|---|---|
| Double ionization, **DIP-ADC(2)** | `-dip` | theADCcode `adcdip*.out` on matched integrals |
| Single ionization, non-Dyson **IP-ADC(3)** with static self-energy Σ(∞) | `-sip` | theADCcode `ADC.out` on matched integrals; pyscf `ip_adc` |

Around them: atom-resolved two-hole populations and Auger / ICD / ETMD decay-channel spectra,
four eigensolvers, matrix-free operators, multi-GPU row partitioning, checkpoint/resume for
multi-day runs, and `plotspec` for figures. Other methods in the development tree (CVS ADC(4),
ADC(2,2), Fano decay widths, transition dipoles) are not part of this release. See
[CHANGELOG.md](CHANGELOG.md).

## Install

**1. Go 1.26 or newer** (`go version`):

```sh
# Linux x86-64; other platforms: https://go.dev/dl
curl -LO https://go.dev/dl/go1.26.5.linux-amd64.tar.gz
mkdir -p ~/.local && tar -C ~/.local -xzf go1.26.5.linux-amd64.tar.gz
echo 'export PATH=$HOME/.local/go/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

**2. Build** (the first build downloads the Go dependencies):

```sh
git clone https://github.com/leiaSQ/ADCgo.git && cd ADCgo
go build -o adcgo ./cmd/adcgo          # the solver (pure Go)
go build -o plotspec ./cmd/plotspec    # JSON -> PNG/SVG/PDF
go build -o sizeprobe ./cmd/sizeprobe  # optional: will a whole-band DIP fit?
go build -o ADCanalysis ./cmd/adcanalysis  # optional: theADCcode output -> spectrum JSON
./adcgo -version
```

For OpenBLAS, CUDA or HIP see [Backends](#backends).

**3. Python + pyscf**, only to make FCIDUMPs for your own molecules (`scripts/fcidump/`). The
water examples below ship in `testdata/`.

```sh
pip install pyscf numpy
```

## Quick start

```sh
./adcgo -fcidump testdata/h2o.fcidump                                  # HF + MP2 ingestion check
./adcgo -fcidump testdata/h2o.fcidump -sip -sym all -out sip.json      # IP-ADC(3), every irrep
./adcgo -fcidump testdata/h2o.fcidump -dip -mo testdata/h2o.mo.json -sym all -out dip.json
./adcgo -convert sip.json -sip -bare -out sip.spec.json                # stick spectrum, no re-solve
./plotspec -in sip.spec.json -out sip.png -fwhm 1.0
```

`adcgo -h` prints the flag overview; `adcgo -h <topic>` (`dip`, `sip`, `solver`, `lowmem`,
`matfree`, `mgpu`, `checkpoint`, `sigma`, `spectrum`, ...) the detail.

## Your own molecule

A deck ties geometry, basis, SCF, active space and output together
([format](scripts/fcidump/adcgo_input.py); GAMESS-UK z-matrices and basis files are read
directly):

```
&geometry
  file   molecule.xyz
&basis
  name   cc-pvdz
&scf
  charge 0
  symmetry auto
&active
  active 2 to 60             # optional: freeze the lowest MO, keep 2..60
&output
  fcidump  molecule.fcidump
  sidecar  molecule.mo.json
&adc
  args   -dip -solver lanczos -spin both
  sym    all
```

```sh
scripts/adcgo_run.sh molecule.in ./adcgo
```

`examples/h2o_dzp/` holds a DIP and a SIP deck over one orbital space. For systems that need a
cluster, see [scripts/README.md](scripts/README.md): SLURM templates for the dump, for DIP and SIP
solves that checkpoint and resubmit themselves, and the spectrum post-processing.

## Methods

### Double ionization — DIP-ADC(2)

Dication states: energies, pole strengths, leading two-hole configurations and, with `-mo`,
atom-resolved two-hole populations (Tarantelli U-transform). One sector per irrep and spin
(`-spin both|singlet|triplet`).

### Single ionization — IP-ADC(3)

Cation states: ionization energies, spectroscopic factors, per-orbital one-hole overlaps. The
non-Dyson IP-ADC(3) matrix (1h main / 2h1p satellite) plus the static self-energy, which
theADCcode keeps in a separate module and ADCgo adds by default: `-sigma infinite` (Σ(∞), the
all-order resolvent resummation, bit-exact against theADCcode); `three | four | fplus | off` for
comparison. Omitting Σ shifts every main line by ~0.2–0.35 eV.

### Decay-channel spectra — Auger / ICD / ETMD

`-spectrum` on a DIP run with `-mo` classifies each dication state by where its two holes sit
relative to the initially ionized site. `-group NAME=col,~col` defines composite or passive sites;
`-init-atom` names the initial one. `-convert` applies this to a saved document without
re-solving.

```sh
./adcgo -fcidump testdata/h2o_dzp.fcidump -dip -mo testdata/h2o_dzp.mo.json \
    -solver dense -sym all -spectrum -init-atom O -out auger.json
./plotspec -in auger.json -out auger.png -fwhm 1.2
```

## Solvers

All solvers diagonalize the same real-symmetric matrix; `-solver` only picks how.

| `-solver` | What it does | Use for |
|---|---|---|
| `lanczos` (default) | block-Lanczos over the whole band; `-blocks N` sets the Krylov size (= theADCcode `iter`) | broad spectra |
| `lanczos-lowmem` | the same band with ~4 Krylov panels resident; `-lowmem-block 0` is theADCcode's short recurrence with its banded eigensolver | whole-band DIP of large systems, on GPUs |
| `davidson` | converges the lowest `-nroots` to `-convthr` | a few exact low-lying states |
| `dense` | full LAPACK diagonalization | small sectors, validation |

Block-Lanczos matches spectral moments: at a fixed `-blocks`, an interior pole can sit at the
pole-strength centroid of a cluster of true states. Raise `-blocks` or use `davidson` when
individual interior roots matter.

## Large systems

| Flag | Effect |
|---|---|
| `-matfree on\|auto` | recompute the memory-dominant blocks (DIP 3h1p×3h1p, SIP 2h1p×2h1p) each mat-vec instead of storing them: TB → 0 resident, on host or GPU |
| `-mgpu N` | DIP Mode B (`-solver lanczos-lowmem -lowmem-block 0`): row-partition one sector across N NVLink GPUs; sectors run in turn |
| `-gpus N` | run independent sectors concurrently, one GPU each |
| `-checkpoint PATH` | save the Krylov state every `-checkpoint-every` blocks and on SIGUSR1, exit 64, resume in the next process |
| `-sigma-cache`, `-mainblock-cache` | SIP: keep Σ(∞) and the 1h main block beside the FCIDUMP (on by default) |

At production scale (a DIP sector of ~1.5×10⁷ configurations) one whole-band sector needs
~0.5 TB of device memory even with `-matfree on`: it runs as `-mgpu 8` on an 8×H200 node,
chained over several 120 h jobs. `sizeprobe system.fcidump` prints those numbers — sector sizes,
panels, matrix-free operator and per-GPU memory for each `-mgpu N` — before anything is submitted.
`scripts/slurm/dip.sbatch` and `sip.sbatch` are those jobs. Background:
[docs/dip_lowmem_lanczos.md](docs/dip_lowmem_lanczos.md),
[docs/dip_operator_memory.md](docs/dip_operator_memory.md),
[backend/README.md](backend/README.md).

## Backends

Pure Go (`gonum`) by default. Accelerated backends are build-tag gated, use cgo (needs `gcc`),
and are selected with `-backend`.

| Backend | Needs | Build |
|---|---|---|
| `openblas` (multicore CPU) | OpenBLAS + LAPACKE dev packages | `go build -tags openblas -o adcgo ./cmd/adcgo` |
| `cuda` (NVIDIA) | CUDA toolkit with `nvcc`, cuBLAS, cuSOLVER | `scripts/build_cuda.sh` → `./adcgo-cuda` |
| `hip` (AMD) | ROCm under `/opt/rocm` (hipBLAS, hipSOLVER) | `go build -tags hip -o adcgo ./cmd/adcgo` |

```sh
./adcgo-cuda -fcidump testdata/h2o.fcidump -dip -backend cuda -sym all
sbatch --gres=gpu:2 scripts/slurm/gpu_tests.sbatch      # validate a CUDA build on real GPUs
```

`go build -tags cuda` alone does **not** rebuild the `.cu` kernels; it links whatever
`backend/*.o` exist. `scripts/build_cuda.sh` does both. `-backend auto` measures each available
backend once and picks the fastest per sector.

## Plotting

`plotspec` renders spectrum JSON: one broadened curve per decay channel (DIP) or orbital (SIP),
sticks, the electron-emission spectrum from a SIP + DIP pair (`-mode ees`), three-panel figures
(`-mode panel`), dimer-minus-monomer difference spectra (`-mode ees-diff`) and conformer
waterfalls (`-mode stack`). Full reference: [cmd/plotspec/README.md](cmd/plotspec/README.md).
`ADCanalysis` writes the same spectrum JSON from theADCcode output files (`adcdip*.out`,
`ADC.out`), so both codes plot and compare side by side:
[cmd/adcanalysis/README.md](cmd/adcanalysis/README.md).

## Output

JSON on stdout or `-out FILE`. `-format ref` writes theADCcode's
`Eigenvalue (eV), ps (%), residue` listing instead, byte-compatible with `adcdip*.out`, so a run
diffs directly against a reference output; `-convert a.json,b.json -format ref` does the same for
saved documents.

## Tests

```sh
go test ./...            # every gate that runs without a GPU (a few minutes)
go test -short ./...     # fast subset
```

Validation is layered: MP2 energy reconstruction; DIP against theADCcode's `adcdip*.out` on
matched integrals; SIP and every Σ scheme against theADCcode dumps on matched integrals and
against pyscf's `ip_adc`; matrix-free, multi-GPU and lowmem paths against the assembled dense
path. GPU kernels have on-hardware parity tests (`-tags cuda`, `scripts/slurm/gpu_tests.sbatch`).
Fixtures regenerate with `scripts/fixtures/` (needs pyscf); [internal/README.md](internal/README.md)
maps every package to its theADCcode counterpart.

## License

LGPL-2.1, see [LICENSE](LICENSE).
