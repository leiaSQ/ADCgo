#!/bin/bash
# A/B per-mat-vec timing for the DIP -mgpu satellite apply, on uracil2W DZ+fc (a sector that
# fits 4 GPUs), BEFORE letting melanin have an 8×H200 node for 120 h.
#
# Runs the identical solve twice, changing only the binary:
#   A  old  ($OLD_BIN)  — satellite recomputed on the HOST (gather-apply-scatter)
#   B  new  ($NEW_BIN)  — satellite recomputed PER DEVICE over NVLink
# so the wall time difference is the thing this work exists to produce. -profile prints the
# per-sector phase breakdown (the mat-vec share) to stderr.
#
# CAVEAT when reading the result: the new binary silently falls back to the host path when the
# partitions are not fully peered (perDeviceSatelliteOK), in which case A and B are the SAME code
# and should time identically. The topology dump below, plus whether the mgpu parity test skipped,
# is how to tell a genuine null result from a fallback.
#
#SBATCH --account=bw22i008
#SBATCH --job-name=u2w-mgpu-timing
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240gb
#SBATCH --time=02:00:00
#SBATCH --output=u2w-mgpu-timing-%j.out
set -euo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/.." && pwd)"
cd "$root"

if command -v module >/dev/null 2>&1; then
    module load compiler/gnu/11.3 devel/cuda/13.2 || true
fi

MOL=uracil2W_dz
WS=/gpfs/bwfor/work/ws/hd_hh323-uracil2W_dz/adcgo/uracil2W_dz
fcidump="$WS/${MOL}.fcidump"
mo="$WS/${MOL}.mo.json"

# A = the pre-per-device build, B = the per-device build. Both are frozen snapshots so this
# comparison is reproducible after the working tree has moved on.
OLD_BIN="${OLD_BIN:-/gpfs/bwfor/work/ws/hd_hh323-melanin/adcgo/runsrc-20260721/adcgo-cuda}"
NEW_BIN="${NEW_BIN:-/gpfs/bwfor/work/ws/hd_hh323-melanin/adcgo/runsrc-fastpath-20260721/adcgo-cuda}"

BLOCKS="${BLOCKS:-10}"   # a few Lanczos iterations is enough to time a mat-vec; not a converged solve
MGPU="${MGPU:-4}"

for f in "$fcidump" "$mo" "$OLD_BIN" "$NEW_BIN"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "host=$(hostname)  mol=$MOL  mgpu=$MGPU blocks=$BLOCKS"
echo "OLD_BIN=$OLD_BIN"
echo "NEW_BIN=$NEW_BIN"
nvidia-smi -L || true
echo "--- interconnect: the per-device gather is only a win over NVLink/NVSwitch, not PHB ---"
nvidia-smi topo -m 2>/dev/null || true

run() {
    local tag="$1" bin="$2"
    echo "#begin<$tag>"
    local t0=$SECONDS
    # || true: a non-zero exit still leaves the timings above it, and B failing must not hide A.
    "$bin" -fcidump "$fcidump" -mo "$mo" \
        -dip -order 2 -solver lanczos-lowmem -lowmem-block 0 -mgpu "$MGPU" \
        -spin singlet -blocks "$BLOCKS" -backend cuda -sym all -profile -matfree on \
        -out "/dev/null" || echo "$tag: exited non-zero"
    echo "$tag: wall $((SECONDS - t0)) s"
    echo "#end<$tag>"
}

run "A-host-gather-scatter" "$OLD_BIN"
run "B-per-device-nvlink"   "$NEW_BIN"

echo "compare the two 'wall' lines and the mat-vec share in each -profile breakdown"
