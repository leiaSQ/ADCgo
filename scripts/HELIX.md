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

It (1) `nvcc`-compiles `internal/adc/backend/adc4_kernels.cu` for both gpu4 GPU models
(`-gencode` sm_80 for A100 and sm_86 for A40), then (2) `go build -tags cuda`. It reads the
CUDA prefix from `$CUDA_HOME` (set by `module load cuda/13.2`); override with
`CGO_CFLAGS`/`CGO_LDFLAGS` if the headers/libs live elsewhere.

Verify: `./adcgo-cuda -h` should list `-gpus`, and `-backend cuda` is accepted.

## Run

**Single input** (its independent sectors spread across all visible GPUs):

```
sbatch scripts/runADCgo_helix examples/melanin/melanin_dip.in ./adcgo-cuda
```

`runADCgo_helix` reserves a full gpu4 node, loads the modules, and drives the standard
`scripts/adcgo_run.sh` (dump the FCIDUMP, then solve).

**Both melanin calculations on one exclusive node:**

```
sbatch scripts/runADCgo_helix_melanin        # needs ./adcgo-cuda prebuilt
```

It builds the shared FCIDUMP once (DIP and SIP use the identical orbital space), then runs
both solves concurrently on disjoint GPU sets via `CUDA_VISIBLE_DEVICES`.

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
