#!/bin/bash
# melanin SIP-ADC(3) on bwForCluster Helix — checkpointing, self-resubmitting daisychain.
# Reads the shared FCIDUMP from $ADCGO_WS (scripts/dump_melanin.sbatch, or submit_melanin.sh).
#
# MEMORY / GPU: the SIP block-Lanczos basis is ~45 GB (n≈518 k, blocks=200), which does NOT
# fit a 40 GB A100. It needs an 80 GB+ GPU. H200 is the unambiguous target — there is no
# small-memory H200, so --gres=gpu:H200:1 can only land on the 141 GB cards (and H200 has
# fast fp64). Needs the sm_90 build (scripts/build_adcgo_cuda_helix). To use A100-80 instead,
# override the gres at submit time — `sbatch --gres=... melanin_sip.sbatch` (or set
# ADCGO_SIP_GRES for scripts/submit_melanin.sh, which passes it through). Note plain
# "gpu:A100:1" may schedule a 40 GB gpu4 A100 and OOM, so a node feature/constraint is
# required for A100-80; confirm the correct spelling with Helix support.
#
# DAISYCHAIN: the solve checkpoints its Krylov state to $ADCGO_WS every -checkpoint-every
# blocks and on a SIGUSR1 (raised by SLURM --signal=B:USR1@600, 10 min before the walltime).
# This job submits its successor up-front with afterany, so even a hard kill (OOM / walltime
# past the grace window) is resumed from the last checkpoint. On clean convergence (exit 0)
# the queued successor is cancelled; on "resume needed" (exit 64) it takes over; on a real
# error it is cancelled and the chain stops. GEN/MAX_GEN bound the chain length.
#
#   sbatch --export=ALL,GEN=0 scripts/melanin_sip.sbatch     # or via scripts/submit_melanin.sh
#
# Env: ADCGO, ADCGO_WS, GEN, MAX_GEN, CHECKPOINT_EVERY (gres override is a submit-time flag).

#SBATCH --account=bw22i008
#SBATCH --job-name=melanin-sip
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=120gb
#SBATCH --time=120:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=melanin-sip-g%j.out
set -euo pipefail

# Recover the submitted script's real path (needed both for the repo root and to resubmit
# this same file as the successor). Under slurm $0 is a spool copy.
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
GEN="${GEN:-0}"
MAX_GEN="${MAX_GEN:-20}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
fcidump="$WS/melanin.fcidump"
mo="$WS/melanin.mo.json"
ckpt="$WS/melanin.sip.ckpt"      # adcgo appends a per-irrep suffix (.i<irrep>)

[ -x "$ADCGO" ]   || { echo "no cuda binary at $ADCGO; build with scripts/build_adcgo_cuda_helix" >&2; exit 1; }
[ -f "$fcidump" ] || { echo "no FCIDUMP at $fcidump; run scripts/dump_melanin.sbatch first" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Queue the successor before doing any work: an afterany dependency means it runs no matter
# how this job ends (clean exit, error, or hard kill), and we cancel it below if we finish.
next=""
if [ "$GEN" -lt "$MAX_GEN" ]; then
    next="$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
        --export=ALL,GEN="$((GEN + 1))" "$self")"
    echo "gen $GEN: queued successor gen $((GEN + 1)) as job $next (afterany:$SLURM_JOB_ID)"
fi

echo "host=$(hostname)  gen=$GEN  ckpt=$ckpt  fcidump=$fcidump"
nvidia-smi -L 2>/dev/null || true

# Run adcgo in the background and forward SLURM's USR1 (batch-shell only, hence B:) to it, so
# it checkpoints at the next block boundary and exits 64. Re-wait after the trap fires.
echo "#begin<adc>"
"$ADCGO" -fcidump "$fcidump" -mo "$mo" \
    -sip -order 3 -sigma infinite -solver lanczos -blocks 200 -backend cuda -sym all \
    -checkpoint "$ckpt" -checkpoint-every "$CHECKPOINT_EVERY" \
    -out "$WS/melanin.sip.json" &
adc_pid=$!
trap 'kill -USR1 "$adc_pid" 2>/dev/null || true' USR1
rc=0; wait "$adc_pid" || rc=$?
while [ "$rc" -gt 128 ]; do rc=0; wait "$adc_pid" || rc=$?; done   # wait interrupted by trap
echo "#end<adc>"

case "$rc" in
    0)
        echo "gen $GEN: converged -> $WS/melanin.sip.json"
        [ -n "$next" ] && { scancel "$next" && echo "cancelled unused successor $next"; }
        ;;
    64)
        echo "gen $GEN: checkpoint written; successor (gen $((GEN + 1))) will resume"
        ;;
    *)
        echo "gen $GEN: solver failed rc=$rc; stopping the chain" >&2
        [ -n "$next" ] && scancel "$next"
        exit "$rc"
        ;;
esac
