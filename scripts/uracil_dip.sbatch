#!/bin/bash
# uracil DIP-ADC(2) on bwForCluster Helix. Reads the FCIDUMP built by
# uracil_dump.sbatch from $ADCGO_WS. Parametrised by $MOL (uracil1W | uracil2W).
#
# SOLVER — WHOLE DOUBLE-IONIZATION BAND. theADCcode's `adc2dip ... lanczos iter
# 200` is a short-recurrence block-Lanczos that returns the *entire* DIP band
# (every pole), which the popana analysis then assigns. The faithful ADCgo port
# is `-solver lanczos-lowmem -lowmem-block 0` (block 0 = the full 2h main-space
# width, so every pole-carrying direction is spanned). Unlike melanin (n ~ 10-15 M,
# where the 3-panel resident is ~0.4-0.6 TB and forces a fat CPU node), uracil's
# DIP space is small (n ~ few x 10^5), so the ~3x(n x main) resident is only a few
# GB and fits on the GPU -- hence -backend cuda here, not gonum.
#
# Two spin sectors (singlet, triplet; theADCcode spin "1 3") run one per GPU -> 2
# A100 (fp64: A100/H200, never A40 -- its fp64 rate is ~1/16 of the A100's).
#
#   MOL=uracil1W sbatch scripts/uracil_dip.sbatch      # or via scripts/submit_uracil.sh
#
# Env: MOL (default uracil1W), ADCGO (cuda binary), ADCGO_WS / ADCGO_WS_NAME,
#      BLOCKS (Lanczos iterations, default 200 = theADCcode iter 200),
#      LOWMEM_BLOCK (band width; 0 = faithful whole band, default 0),
#      BACKEND (default cuda; set gonum to fall back to a CPU node if a build
#               restricts lowmem-block 0 to gonum), gres override at submit time.

#SBATCH --account=bw22i008
#SBATCH --job-name=uracil-dip
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=120gb
#SBATCH --time=120:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# Recover the repo root (slurm runs a spool copy, so $0 is not in scripts/).
self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" \
        | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/.." && pwd)"
cd "$root"

if command -v module >/dev/null 2>&1; then
    module load compiler/gnu/11.3 devel/cuda/13.2 || true
fi

MOL="${MOL:-uracil1W}"
ADCGO="${ADCGO:-$root/adcgo-cuda}"
BLOCKS="${BLOCKS:-200}"          # theADCcode iter 200
LOWMEM_BLOCK="${LOWMEM_BLOCK:-0}" # 0 = full 2h main-space block = faithful whole band
BACKEND="${BACKEND:-cuda}"

export ADCGO_WS_NAME="${ADCGO_WS_NAME:-$MOL}"
source "$root/scripts/adcgo_ws.sh"
WS="$ADCGO_WS"
fcidump="$WS/${MOL}.fcidump"
mo="$WS/${MOL}.mo.json"

[ -x "$ADCGO" ]   || { echo "no cuda binary at $ADCGO; build with scripts/build_adcgo_cuda_helix" >&2; exit 1; }
[ -f "$fcidump" ] || { echo "no FCIDUMP at $fcidump; run scripts/uracil_dump.sbatch (MOL=$MOL) first" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "host=$(hostname)  MOL=$MOL  fcidump=$fcidump  blocks=$BLOCKS lowmem-block=$LOWMEM_BLOCK backend=$BACKEND"
nvidia-smi -L 2>/dev/null || true
echo "#begin<adc>"
"$ADCGO" -fcidump "$fcidump" -mo "$mo" \
    -dip -order 2 -solver lanczos-lowmem -lowmem-block "$LOWMEM_BLOCK" \
    -spin both -blocks "$BLOCKS" -backend "$BACKEND" -sym all \
    -out "$WS/${MOL}.dip.json"
echo "#end<adc>"
echo "dip -> $WS/${MOL}.dip.json"
