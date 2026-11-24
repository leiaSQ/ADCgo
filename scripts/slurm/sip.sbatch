#!/bin/bash
# SIP-ADC(3) on a GPU node: block-Lanczos with checkpoint/resume daisy-chain.
#
#   FCIDUMP=/path/sys.fcidump sbatch --gres=gpu:1 --mem=500G scripts/slurm/sip.sbatch
#
# What decides the resources:
#   * -matfree auto recomputes the 2h1p×2h1p satellite block on the GPU instead of storing it
#     (dense it is nSat²·8 bytes — terabytes for a large system). The GPU then holds the norb⁴
#     ERI tensor plus the Krylov basis (n · BLOCKS · main · 8 bytes, linear in BLOCKS); adcgo
#     refuses up front, before any allocation, if that does not fit.
#   * HOST memory is set by the static self-energy Σ(∞), built before the solver starts and not
#     covered by -checkpoint. For a large C1 system give the job the node's RAM.
#   * Σ(∞) and the 1h/1h main block are cached beside the FCIDUMP (adcgo -h checkpoint), so a
#     successor generation skips both and goes straight back to the Krylov checkpoint.
#
# The checkpoint grows with the block count (saveKrylov writes n · dim), so its total write
# volume is quadratic in BLOCKS; CHECKPOINT_EVERY trades that against crash exposure. A walltime
# kill is covered regardless: SIGUSR1 checkpoints at the next block boundary.
#
# A finished solve cannot be extended: the checkpoint is deleted on exit 0 and pins -blocks.
# More blocks means a fresh run (cheap with the caches; Lanczos is deterministic).
#
# Environment:
#   FCIDUMP            input
#   OUT                result document (default: ${FCIDUMP%.fcidump}.sip.json)
#   ADCGO              CUDA binary (default: <repo>/adcgo-cuda, built by scripts/build_cuda.sh)
#   BLOCKS             Lanczos blocks (default 200)
#   CHECKPOINT_EVERY   blocks between crash-resilience checkpoints (default 50)
#   SIGMA              static self-energy scheme (default infinite; adcgo -h sigma)
#   EXTRA              extra adcgo flags, appended verbatim
#   GEN, MAX_GEN       chain generation / cap (chain.sh)
#   MODULES            environment modules to load (e.g. "devel/cuda/13.2")
#
#SBATCH --job-name=adcgo-sip
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=adcgo-sip-%j.out
set -euo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
# shellcheck source=chain.sh
source "$(cd "$(dirname "$self")" && pwd)/chain.sh"
chain_init "$self"
if [ -n "${MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $MODULES
fi

ADCGO="${ADCGO:-$root/adcgo-cuda}"
: "${FCIDUMP:?set FCIDUMP=/path/to/system.fcidump}"
BLOCKS="${BLOCKS:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
SIGMA="${SIGMA:-infinite}"
OUT="${OUT:-${FCIDUMP%.fcidump}.sip.json}"
ckpt="${OUT%.json}.ckpt" # adcgo appends .i<irrep>

[ -x "$ADCGO" ] || { echo "no binary at $ADCGO; build it with scripts/build_cuda.sh" >&2; exit 1; }
[ -f "$FCIDUMP" ] || { echo "no FCIDUMP at $FCIDUMP" >&2; exit 1; }

args=(-fcidump "$FCIDUMP" -sip -sigma "$SIGMA" -solver lanczos -blocks "$BLOCKS"
    -sym all -backend cuda -matfree auto -profile
    -checkpoint "$ckpt" -checkpoint-every "$CHECKPOINT_EVERY" -out "$OUT")
# shellcheck disable=SC2206
[ -n "${EXTRA:-}" ] && args+=($EXTRA)

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "host=$(hostname) gen=$GEN binary=$ADCGO"
command -v nvidia-smi >/dev/null && nvidia-smi -L
echo "adcgo ${args[*]}"

chain_queue
chain_run "$ADCGO" "${args[@]}"
chain_finish "$rc"
echo "result: $OUT"
