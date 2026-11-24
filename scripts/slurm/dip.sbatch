#!/bin/bash
# DIP-ADC(2) on a GPU node: whole-band block-Lanczos with checkpoint/resume daisy-chain.
#
#   FCIDUMP=/path/sys.fcidump MO=/path/sys.mo.json sbatch --gres=gpu:8 scripts/slurm/dip.sbatch
#
# The default is the large-system path: -solver lanczos-lowmem -lowmem-block 0 ("Mode B", the
# whole double-ionization band with only ~4 Krylov panels resident), -matfree on (the 3h1p×3h1p
# satellite region is recomputed, never stored) and -mgpu $MGPU (ONE sector row-partitioned across
# the node's GPUs; needs NVLink). Sectors run one after another. Mode B checkpoints are a constant
# ~2·n·main·8 bytes, so CHECKPOINT_EVERY=1 is affordable when a block takes hours.
#
# Set MGPU=0 for a system that fits one GPU (sectors then run concurrently, one per GPU), or
# SOLVER=davidson to converge only the lowest NROOTS states (not resumable).
#
# Environment:
#   FCIDUMP, MO        inputs (MO optional; needed for populations / decay-channel spectra)
#   OUT                result document (default: ${FCIDUMP%.fcidump}.dip[.SPIN].json)
#   ADCGO              CUDA binary (default: <repo>/adcgo-cuda, built by scripts/build_cuda.sh)
#   SOLVER             lanczos-lowmem (default) | lanczos | davidson
#   SPIN               both (default) | singlet | triplet. On -mgpu, one spin per job keeps the
#                      peak to one sector's footprint; submit two jobs for both spins.
#   MGPU               GPUs per sector (default 8; 0 = off)
#   BLOCKS             Lanczos blocks (default 200)
#   CHECKPOINT_EVERY   blocks between crash-resilience checkpoints (default 1)
#   NROOTS, MAXDAVSP   -solver davidson only (defaults 30, 90)
#   EIG_B2, EIG_DEVICE Mode B eigensolve accelerations (adcgo -h lowmem); off by default
#   EXTRA              extra adcgo flags, appended verbatim
#   GEN, MAX_GEN       chain generation / cap (chain.sh)
#   MODULES            environment modules to load (e.g. "devel/cuda/13.2")
#
#SBATCH --job-name=adcgo-dip
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=adcgo-dip-%j.out
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
MO="${MO:-}"
SOLVER="${SOLVER:-lanczos-lowmem}"
SPIN="${SPIN:-both}"
MGPU="${MGPU:-8}"
BLOCKS="${BLOCKS:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
base="${FCIDUMP%.fcidump}"
OUT="${OUT:-$base.dip$([ "$SPIN" != both ] && echo ".$SPIN").json}"
ckpt="${OUT%.json}.ckpt" # adcgo appends .s<spin>.i<irrep>

[ -x "$ADCGO" ] || { echo "no binary at $ADCGO; build it with scripts/build_cuda.sh" >&2; exit 1; }
[ -f "$FCIDUMP" ] || { echo "no FCIDUMP at $FCIDUMP" >&2; exit 1; }

args=(-fcidump "$FCIDUMP" -dip -spin "$SPIN" -sym all -backend cuda -profile -out "$OUT")
[ -n "$MO" ] && args+=(-mo "$MO")
case "$SOLVER" in
    lanczos-lowmem)
        args+=(-solver lanczos-lowmem -lowmem-block 0 -blocks "$BLOCKS" -matfree on
            -checkpoint "$ckpt" -checkpoint-every "$CHECKPOINT_EVERY")
        [ "$MGPU" -gt 0 ] && args+=(-mgpu "$MGPU")
        [ "${EIG_B2:-0}" -gt 0 ] && args+=(-eig-b2 "$EIG_B2")
        [ "${EIG_DEVICE:-0}" = 1 ] && args+=(-eig-device)
        ;;
    lanczos)
        args+=(-solver lanczos -blocks "$BLOCKS" -matfree auto
            -checkpoint "$ckpt" -checkpoint-every "$CHECKPOINT_EVERY")
        ;;
    davidson)
        args+=(-solver davidson -nroots "${NROOTS:-30}" -maxdavsp "${MAXDAVSP:-90}")
        MAX_GEN=0 # nothing to resume
        ;;
    *) echo "unknown SOLVER=$SOLVER" >&2; exit 2 ;;
esac
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
