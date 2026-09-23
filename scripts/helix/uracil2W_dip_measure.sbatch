#!/bin/bash
# uracil2W whole-band DIP — the measurement fixture for the satellite batched-GEMM work.
#
# WHY URACIL2W AND NOT The production system. docs/sigma_build_contractions.md:142-147 warns that the bundled
# h2o_dzp fixture has jiiLKK blocks only 2..11 on a side, so "any conclusion drawn from a whole-apply
# benchmark on a small symmetric test system is an artefact of the test system". The production system is C1 with
# nvir=154, giving 154x154 blocks — and a 40 h mat-vec, which is unaffordable to iterate on.
#
# uracil2W is C1 with nvir=151. Its blocks are 151x151: the SAME degenerate GEMM geometry as
# the production system, at ~1/8 the sector height (n ~ 1.2 M vs 10.0 M) and ~1/4 the panel width (main=435 vs
# 1711). Since the defects being fixed are functions of block shape and issue COUNT, not of n, this
# reproduces them at a cost that fits one job.
#
# WHAT CHANGED SINCE THE 40 h TRACE (job 14211868, the production system, batched path):
#   * runBatches*/runBatchesOwnedRange no longer scan every member of every batch per fill chunk
#     (binary-search member selection over the now-sorted backend.Batch.Blocks).
#   * The batch plan is built ONCE for the pool instead of once per device.
#   * The per-device loop runs concurrently (wg.Go), as gatherSlabs/syncAll already did.
# GPU parity after those changes is numerically IDENTICAL to before (job 14245529 vs 14150821,
# max|Δ| equal to the last digit on all 8 sectors), so this job measures speed, not correctness.
#
# WHAT TO READ OUT of the sattrace lines:
#   wall vs summed gemm  — summed gemm >> wall proves the devices now overlap. Before the parallel
#                          loop, summed gemm EQUALLED wall exactly (zero overlap).
#   members/call         — batching health. ~1 means computeChunks has shredded the plan's batches
#                          (each fill chunk contributing a single member to many batches, which
#                          GemmMatBatched short-circuits to a plain GemmMat). Thousands means the
#                          batching is intact and the remaining cost is real GEMM work.
#   fill vs gemm         — fill is only the async launch time; the fill's real traffic lands in the
#                          gemm window on the same (default) stream.
#
#   sbatch scripts/helix/uracil2W_dip_measure.sbatch
#
# Env: ADCGO, MGPU (4), BLOCKS (4), SATCHUNK (64).

#SBATCH --account=bw22i008
#SBATCH --job-name=u2w-dip-measure
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=480gb
#SBATCH --time=06:00:00
#SBATCH --output=u2w-dip-measure-%j.out
set -uo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" \
        | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/../.." && pwd)"
cd "$root"

if command -v module >/dev/null 2>&1; then
    module load compiler/gnu/11.3 devel/cuda/13.2 || true
fi

ADCGO="${ADCGO:-$root/adcgo-cuda-dip}"
WS=/gpfs/bwfor/work/ws/hd_hh323-uracil2W/adcgo/uracil2W
MGPU="${MGPU:-4}"
BLOCKS="${BLOCKS:-4}"     # a couple of blocks: enough to see a per-block apply time settle
SATCHUNK="${SATCHUNK:-64}"
fcidump="$WS/uracil2W.fcidump"
mo="$WS/uracil2W.mo.json"

[ -x "$ADCGO" ]   || { echo "no cuda binary at $ADCGO" >&2; exit 1; }
[ -f "$fcidump" ] || { echo "no FCIDUMP at $fcidump" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "host=$(hostname)  mgpu=$MGPU blocks=$BLOCKS satchunk=$SATCHUNK"
echo "start=$(date -Is)"
nvidia-smi -L 2>/dev/null || true

# -profile prints the per-sector phase split; -sat-trace prints the per-column-chunk breakdown;
# Options.Progress (wired to both) prints one line per accepted Lanczos block, so per-block apply
# time is readable as the run proceeds rather than only at the end.
echo "#begin<adc>"
"$ADCGO" -fcidump "$fcidump" -mo "$mo" \
    -dip -order 2 -solver lanczos-lowmem -lowmem-block 0 -mgpu "$MGPU" \
    -spin singlet -blocks "$BLOCKS" -backend cuda -sym all -profile -matfree on \
    -satchunk "$SATCHUNK" -sat-trace \
    -out /dev/null
rc=$?
echo "#end<adc>"
echo "end=$(date -Is)  rc=$rc"

echo "### blocks completed ###"
grep -c "^progress dip" "u2w-dip-measure-${SLURM_JOB_ID:-x}.out" 2>/dev/null || true
echo "(see 'progress dip' lines above for per-block apply time, 'sattrace' for the chunk breakdown)"
exit 0
