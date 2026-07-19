#!/bin/bash
# melanin DIP-ADC(2) on bwForCluster Helix — WHOLE-BAND Mode B block-Lanczos spread
# across all 8 H200 of a gpu8 node (NVLink/NVSwitch). Reads the shared FCIDUMP from
# $ADCGO_WS (build it first: scripts/dump_melanin.sbatch, or scripts/submit_melanin.sh).
#
# WHY THIS EXISTS — the faithful whole-band port. Davidson (scripts/melanin_dip.sbatch)
# only reaches the lowest -nroots roots; Mode B (lowmem-block 0, block == main ≈ 1161)
# returns the ENTIRE double-ionization band, matching theADCcode's `adc2dip … lanczos`.
# The obstacle is memory: a single Krylov block is n·main·8 ≈ 137 GB (triplet n≈14.75 M)
# and Mode B keeps ~4 live panels + the ~50 GB operator ≈ 0.5 TB/sector — past any one
# H200 (141 GB). The -mgpu path ROW-PARTITIONS one sector across the whole pool, so the
# per-GPU footprint is ≈ (4·n·main·8 + operator)/8 ≈ 75 GB — comfortable on 141 GB.
# Cross-partition mat-vec input bands move device-to-device over NVLink (peer copy);
# the small orthogonalization reductions all-reduce a ≈10 MB result.
#
# Spin sectors run SERIALLY, each spanning all 8 GPUs (not one-per-GPU like the concurrent
# per-sector path). Host RAM must hold the resident block-sparse operator copy (~50 GB, in
# distMat.host) plus the integrals — hence --mem below. The GPU side is guarded by ADCgo's
# pre-flight fit check, which refuses cleanly (no cudaMalloc panic) if a sector would not
# fit the chosen partition count.
#
#   sbatch scripts/melanin_dip_mgpu.sbatch
#
# Env: ADCGO (cuda binary), ADCGO_WS (workspace), MGPU (partitions, default 8),
#      BLOCKS (Lanczos blocks / iterations, default 200).

#SBATCH --account=bw22i008
#SBATCH --job-name=melanin-dip-mgpu
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H200:8
#SBATCH --mem=480gb
#SBATCH --time=120:00:00
#SBATCH --output=melanin-dip-mgpu-%j.out
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

ADCGO="${ADCGO:-$root/adcgo-cuda}"
source "$root/scripts/melanin_ws.sh"   # resolves + exports ADCGO_WS (ws_allocate, not mkdir)
WS="$ADCGO_WS"
MGPU="${MGPU:-8}"        # GPUs to row-partition ONE sector across (needs a gpu8 node + NVLink)
BLOCKS="${BLOCKS:-200}"  # Lanczos blocks; whole-band Mode B keeps only ~4 panels resident
fcidump="$WS/melanin.fcidump"
mo="$WS/melanin.mo.json"

[ -x "$ADCGO" ]   || { echo "no cuda binary at $ADCGO; build with scripts/build_adcgo_cuda_helix" >&2; exit 1; }
[ -f "$fcidump" ] || { echo "no FCIDUMP at $fcidump; run scripts/dump_melanin.sbatch first" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

echo "host=$(hostname)  fcidump=$fcidump  solver=lanczos-lowmem lowmem-block=0 (whole band)  mgpu=$MGPU blocks=$BLOCKS"
nvidia-smi -L 2>/dev/null || true
echo "#begin<adc>"
"$ADCGO" -fcidump "$fcidump" -mo "$mo" \
    -dip -order 2 -solver lanczos-lowmem -lowmem-block 0 -mgpu "$MGPU" \
    -spin both -blocks "$BLOCKS" -backend cuda -sym all -profile \
    -out "$WS/melanin.dip.json"
echo "#end<adc>"
echo "dip (whole-band Mode B, $MGPU GPUs) -> $WS/melanin.dip.json"
