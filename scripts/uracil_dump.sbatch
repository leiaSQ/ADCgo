#!/bin/bash
# Build a uracil-DIP FCIDUMP on a CPU node (bwForCluster Helix).
#
# pyscf on the CPU (RHF + AO->MO transform, no GPU), so it runs on cpu-single and
# never holds a GPU idle. The DIP GPU job (uracil_dip.sbatch) then reads the one
# FCIDUMP this produces. Parametrised by $MOL (uracil1W | uracil2W): the molecule
# selects examples/$MOL/$MOL_dip.in and the $MOL workspace.
#
# Outputs land in $ADCGO_WS (a Helix workspace on the parallel FS, NOT $HOME:
# $HOME has a quota and the FCIDUMP is ~GB). dump_fcidump.py resolves &output
# paths relative to the input file's directory, so we stage a copy of the .in
# (geometry/basis symlinked in) into $WS and the outputs land there.
#
#   MOL=uracil1W sbatch scripts/uracil_dump.sbatch     # or via scripts/submit_uracil.sh
#
# Env: MOL (molecule, default uracil1W), ADCGO_PYTHON (python with pyscf),
#      ADCGO_WS / ADCGO_WS_NAME (workspace; default name = $MOL).

#SBATCH --account=bw22i008
#SBATCH --job-name=uracil-dump
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=180gb
#SBATCH --time=24:00:00
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

MOL="${MOL:-uracil1W}"
ex="$root/examples/$MOL"
[ -f "$ex/${MOL}_dip.in" ] || { echo "no input at $ex/${MOL}_dip.in (bad MOL=$MOL?)" >&2; exit 1; }

PY="${ADCGO_PYTHON:-$HOME/miniconda3/envs/adcgo/bin/python}"
[ -x "$PY" ] || { echo "no python at $PY; set ADCGO_PYTHON to the adcgo conda env" >&2; exit 1; }

export ADCGO_WS_NAME="${ADCGO_WS_NAME:-$MOL}"
source "$root/scripts/adcgo_ws.sh"     # resolves + exports ADCGO_WS (ws_allocate, not mkdir)
WS="$ADCGO_WS"

# Stage inputs into the workspace so the big outputs land there too.
ln -sf "$ex/zmat"          "$WS/zmat"
ln -sf "$ex/_basisset.dat" "$WS/_basisset.dat"
cp -f  "$ex/${MOL}_dip.in" "$WS/${MOL}.in"

echo "host=$(hostname)  MOL=$MOL  WS=$WS  PY=$PY"
echo "#begin<dump>"
OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 "$PY" scripts/dump_fcidump.py --input "$WS/${MOL}.in"
echo "#end<dump>"
echo "wrote: $WS/${MOL}.fcidump  $WS/${MOL}.mo.json"
