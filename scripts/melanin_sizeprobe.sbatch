#!/bin/bash
# Melanin DIP operator sizing for the -mgpu per-device (matrix-free satellite) budget.
#
# Reports, per spin sector: n, main, the Krylov panels, and — the number this exists for — the
# MATRIX-FREE resident operator, i.e. the still-dense 2h main + 2h↔3h1p coupling blocks that
# remain after the satellite region is recomputed instead of stored. docs/dip_operator_memory.md
# only ever asserted that residual was small; the 8×H200 triplet budget has ~34 GB of headroom,
# so it needs measuring rather than assuming.
#
# The matrix-free walk is cheap (OperatorResidentBytes skips satelliteResidentBytes entirely when
# matrix-free), so this is a short CPU job — unlike the DENSE walk, which is what was abandoned at
# melanin scale. Set SIZEPROBE_DENSE=1 to include the dense number too, and expect hours.
#
#   sbatch scripts/melanin_sizeprobe.sbatch
#
#SBATCH --account=bw22i008
#SBATCH --job-name=melanin-sizeprobe
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240gb
#SBATCH --time=04:00:00
#SBATCH --output=melanin-sizeprobe-%j.out
set -euo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/.." && pwd)"
cd "$root"

export PATH="/home/hd/hd_hd/hd_hh323/opt/go/bin:$PATH"
source "$root/scripts/melanin_ws.sh"
fcidump="$ADCGO_WS/melanin.fcidump"
[ -f "$fcidump" ] || { echo "no FCIDUMP at $fcidump" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-32}"

echo "host=$(hostname)  go=$(go version)"
echo "fcidump=$fcidump  SIZEPROBE_DENSE=${SIZEPROBE_DENSE:-<unset, matrix-free only>}"

echo "#begin<sizeprobe>"
go run ./cmd/sizeprobe "$fcidump"
echo "#end<sizeprobe>"
