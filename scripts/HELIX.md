# Running ADCgo on bwForCluster Helix

Build and run notes for the multi-GPU CUDA backend on Helix (https://wiki.bwhpc.de/e/Helix).

## Node / partitions (gpu4)

A **gpu4** node has **4 GPUs** (Nvidia A40 48 GB or A100 40 GB), **64 CPU cores**, and
**236 GB usable RAM** (no local disk). GPU partitions:

| Partition | Mode | Max nodes | Walltime | Use |
|-----------|------|-----------|----------|-----|
| `gpu-single` | shared (add `--exclusive` for a whole node) | 1 | 120 h | these scripts |
| `gpu-multi` | node-exclusive, gpu4 only | 8 | 48 h | — |

Request GPUs with `--gres=gpu:<n>` (or `--gres=gpu:A40:<n>` / `gpu:A100:<n>` to pin a model).

`gpu-single` also reaches the newer **gpu8** nodes (8× A100 80 GB, and 8× H200 141 GB with
fp64) — its `NodeSets` are `gpu4_a40, gpu4_a100, gpu8_a100, gpu8_h200`. **Avoid A40** for the
ADC solves: they are fp64 throughout and the A40's fp64 rate is ~1/16 of the A100's, so
landing on an A40 is the likeliest cause of a solve blowing past the walltime. The build now
emits `sm_90`, so H200 is a valid target (used by the SIP job, which needs its 141 GB).

## Modules

```
module load gnu/11.3 cuda/13.2      # host compiler + CUDA toolkit (for building)
module load golang/1.20.5           # NOTE: too old — see below
```

**Go version — the one real snag.** This repo's `go.mod` requires **Go ≥ 1.26** (the code
uses post-1.20 language features), so Helix's `golang/1.20.5` module *cannot build it*.
Provide a newer Go on the build node, any of:

- install Go ≥ 1.26 under `$HOME` and put it first on `PATH` (simplest), or
- `conda install -c conda-forge go` in a build env, or
- with a Go ≥ 1.21 already available, `export GOTOOLCHAIN=auto` lets it fetch the toolchain
  `go.mod` names (needs network on the build node).

`scripts/build_adcgo_cuda_helix` checks the Go version and aborts with a clear message if
it is too old. (`comp/nvhpc/24.3` is also present and bundles a CUDA 12 toolkit, but the
node runtime is CUDA 13.2, so build against `cuda/13.2` for a clean runtime-lib match.)

## Build (two stages)

The GPU backend is `cgo` + a custom CUDA kernel, so it needs `nvcc` and the CUDA dev libs,
then a `-tags cuda` Go build. Both are wrapped in one script:

```
scripts/build_adcgo_cuda_helix        # -> ./adcgo-cuda at the repo root
```

It (1) `nvcc`-compiles `internal/adc/backend/adc4_kernels.cu` for every Helix GPU model
(`-gencode` sm_80 for A100, sm_86 for A40, sm_90 for H200), then (2) `go build -tags cuda`.
It reads the CUDA prefix from `$CUDA_HOME` (set by `module load devel/cuda/13.2`); override
with `CGO_CFLAGS`/`CGO_LDFLAGS` if the headers/libs live elsewhere.

Verify: `./adcgo-cuda -h` should list `-gpus`, and `-backend cuda` is accepted.

## Run

**Single input** (its independent sectors spread across all visible GPUs):

```
sbatch scripts/runADCgo_helix examples/melanin/melanin_dip.in ./adcgo-cuda
```

`runADCgo_helix` reserves a full gpu4 node, loads the modules, and drives the standard
`scripts/adcgo_run.sh` (dump the FCIDUMP, then solve).

**Melanin, split pipeline (recommended):**

```
scripts/submit_melanin.sh                     # needs ./adcgo-cuda prebuilt
```

This chains three jobs so each stage runs where it belongs and gets its own 120 h clock:

1. `dump_melanin.sbatch` — builds the shared FCIDUMP on a **`cpu-single`** node (pyscf is
   CPU-only; no reason to hold GPUs idle for it). Writes `melanin.fcidump` / `melanin.mo.json`
   to `$ADCGO_WS` on the scratch filesystem, not `$HOME` (the FCIDUMP is several GB and `$HOME`
   has a quota).
2. `melanin_dip.sbatch` — DIP-ADC(2) via **block-Davidson**, `--gres=gpu:A100:2` (C1 →
   singlet+triplet = 2 sectors, one per GPU).
3. `melanin_sip.sbatch` — SIP-ADC(3) via **checkpointing block-Lanczos**, `--gres=gpu:H200:1`,
   a **self-resubmitting daisychain**.

DIP and SIP both `--dependency=afterok` on the dump job, then run in parallel. Cores are
requested proportional to GPUs (16/GPU: DIP 32, SIP 16) so the node stays shareable.

**Why the solvers differ.** The two melanin spaces are far apart in size (measured with the
real space builders):

| Solve | dim `n` | full-band Lanczos basis @blocks=200 | verdict |
|-------|---------|-------------------------------------|---------|
| DIP singlet | 10.0 M | ~25 TB | infeasible |
| DIP triplet | 14.8 M | ~36 TB | infeasible |
| SIP | 518 k | ~45 GB | fits an 80 GB+ GPU |

DIP's full-band block-Lanczos cannot even hold one Krylov block (137 GB) on any GPU, so DIP
uses **block-Davidson** instead: it caps the subspace at `-maxdavsp` and returns the lowest
`-nroots` roots (two `n×maxdim` panels, ~28 GB at the defaults — fits a 40 GB A100). The
trade-off is the low-lying states, not the whole double-ionization band; raise `-nroots`
(with `-maxdavsp > 2·nroots`) for more, which needs an 80 GB GPU.

SIP's ~45 GB Lanczos basis needs an **80 GB+ GPU**, so it targets **H200** (`gpu:H200:1` can
only land on the 141 GB cards; needs the sm_90 build). Because it can run for days, it
**checkpoints** its Krylov state to `$ADCGO_WS` (`-checkpoint … -checkpoint-every N`) and
each job **resubmits its successor** (`afterany`), which resumes from the checkpoint after a
walltime kill. SLURM `--signal=B:USR1@600` triggers a clean checkpoint 10 min before the
wall; the binary exits 64 ("resume needed") vs 0 (converged), and the wrapper `scancel`s the
unused successor on convergence. `MAX_GEN` bounds the chain. To use A100-80 instead of H200,
override the gres: `ADCGO_SIP_GRES=... scripts/submit_melanin.sh` (confirm the 80 GB
feature/constraint spelling with Helix support — plain `gpu:A100:1` may hit a 40 GB card).

**Workspace** — set `ADCGO_WS` (default `/gpfs/bwfor/scratch/hd_hh323_o05i14/adcgo/melanin`);
export it before `submit_melanin.sh` so the sbatch jobs inherit it. **Python** — set
`ADCGO_PYTHON` if the `adcgo` conda env is not at `$HOME/miniconda3/envs/adcgo/bin/python`.
**Rebuild** the binary after these notes: the build now also emits `sm_90` for H200.

The older `runADCgo_helix_melanin` (both solves concurrently on one `--exclusive` node via
`CUDA_VISIBLE_DEVICES`) still works but is superseded by the split pipeline above.

_Module names on Helix are prefixed: `compiler/gnu/11.3`, `devel/cuda/13.2` (not bare
`gnu/11.3` / `cuda/13.2`)._

## Multi-GPU behaviour

With `-backend cuda`, independent sectors run **one per GPU** (DIP: spin×irrep; SIP: irrep).
Set `-gpus N` to cap the count (default: all visible). For melanin (C1 symmetry):

- DIP `-spin both` → 2 sectors → GPUs 0,1
- SIP (C1 doublet) → 1 sector → GPU 2 (GPU 3 idle)

A single C1 sector cannot be split across GPUs (that would need intra-sector partitioning,
not implemented). Jobs with more symmetry irreps or more spin sectors fill more GPUs.

**CPU threads:** each GPU worker also drives OpenBLAS on the host-side eigensolve. When
running two processes on one node (the melanin script), each is given
`OPENBLAS_NUM_THREADS=32` so the two pools do not oversubscribe the 64 cores.

## pyscf / FCIDUMP

The dump step (SCF + AO→MO transform) is pyscf on the CPU — no GPU. It needs the `adcgo`
conda env; set `ADCGO_PYTHON` if it is not at `$HOME/miniconda3/envs/adcgo/bin/python`.
For melanin (260 basis functions, `active 25 to 236` → 212 orbitals) the FCIDUMP is several
GB and the transform takes a while; 236 GB RAM is ample.
