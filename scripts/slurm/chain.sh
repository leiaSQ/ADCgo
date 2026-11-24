# chain.sh — self-resubmitting SLURM daisy-chain for long adcgo solves. Sourced by
# dip.sbatch and sip.sbatch; not run on its own.
#
# A whole-band solve of a large system outlasts any one walltime. adcgo checkpoints its Krylov
# state (-checkpoint) every -checkpoint-every blocks and, on SIGUSR1, at the next block boundary,
# then exits 64 ("resume needed"). This helper wires that to SLURM:
#
#   1. chain_init       record the real script path (for resubmission) and the repo root.
#   2. chain_queue      submit the successor UP FRONT with --dependency=afterany, so even a hard
#                       kill (OOM, walltime past the grace window) resumes from the last checkpoint.
#   3. chain_run CMD..  run adcgo in the background and forward SLURM's USR1 to it
#                       (#SBATCH --signal=B:USR1@600 sends it 10 min before the walltime).
#   4. chain_finish RC  exit 0 → cancel the unused successor; 64 → let it resume; else stop.
#
# Environment: GEN (generation, default 0), MAX_GEN (default 20).

# chain_init SCRIPT_PATH
chain_init() {
    self="$1"
    # sbatch records the command as typed; a relative path would be resolved against the
    # successor's working directory and break the chain at generation 1.
    case "$self" in
        /*) ;;
        *) self="$(cd "$(dirname "$self")" && pwd)/$(basename "$self")" ;;
    esac
    root="$(cd "$(dirname "$self")/../.." && pwd)"
    GEN="${GEN:-0}"
    MAX_GEN="${MAX_GEN:-20}"
    next=""
}

chain_queue() {
    [ -n "${SLURM_JOB_ID:-}" ] || return 0
    if [ "$GEN" -lt "$MAX_GEN" ]; then
        # Carry the job name and log pattern so every generation lands in one log series.
        next="$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
            --job-name="$SLURM_JOB_NAME" --output="${SLURM_JOB_NAME}-g%j.out" \
            --export=ALL,GEN="$((GEN + 1))" "$self")"
        echo "gen $GEN: queued successor gen $((GEN + 1)) as job $next"
    fi
}

chain_run() {
    "$@" &
    local pid=$!
    trap 'kill -USR1 "'"$pid"'" 2>/dev/null || true' USR1
    rc=0
    wait "$pid" || rc=$?
    # wait returns >128 both when the USR1 trap interrupts it and when the child was killed by a
    # signal (137 = SIGKILL, usually the cgroup OOM killer). Re-wait only while the child is
    # genuinely alive; looping on rc alone spins forever on a dead child.
    while [ "$rc" -gt 128 ] && kill -0 "$pid" 2>/dev/null; do
        rc=0
        wait "$pid" || rc=$?
    done
    if [ "$rc" -gt 128 ]; then
        echo "gen $GEN: adcgo killed by signal $((rc - 128))$([ "$rc" = 137 ] && echo ' (SIGKILL: check MaxRSS against --mem)')" >&2
    fi
}

chain_finish() {
    case "$1" in
        0)
            echo "gen $GEN: converged"
            if [ -n "$next" ]; then scancel "$next" && echo "cancelled unused successor $next"; fi
            ;;
        64)
            echo "gen $GEN: checkpoint written; successor will resume"
            ;;
        *)
            echo "gen $GEN: adcgo failed rc=$1; stopping the chain" >&2
            if [ -n "$next" ]; then scancel "$next"; fi
            exit "$1"
            ;;
    esac
}
