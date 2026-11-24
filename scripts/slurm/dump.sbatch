#!/bin/bash
# Build the FCIDUMP + MO sidecar + manifest for a deck on a CPU node (pyscf RHF + integral
# transformation). A DIP and a SIP run over the same orbital space share one dump.
#
#   DECK=examples/h2o_dzp/dip.in sbatch --cpus-per-task=32 --mem=200G scripts/slurm/dump.sbatch
#
# Outputs land where the deck's &output section says, relative to the deck's directory.
#
# Environment:
#   DECK           input deck (scripts/fcidump/adcgo_input.py documents the format)
#   ADCGO_PYTHON   Python with pyscf + numpy (default: python3 on PATH)
#   MODULES        environment modules to load first
#
#SBATCH --job-name=adcgo-dump
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=adcgo-dump-%j.out
set -euo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/../.." && pwd)"
if [ -n "${MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $MODULES
fi

: "${DECK:?set DECK=/path/to/system.in}"
PY="${ADCGO_PYTHON:-python3}"
# numpy/pyscf size their BLAS pool from these at import; unset, they take every visible core.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "host=$(hostname) deck=$DECK python=$PY threads=$OMP_NUM_THREADS"
"$PY" "$root/scripts/fcidump/dump_fcidump.py" --input "$DECK"
