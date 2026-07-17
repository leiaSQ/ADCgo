#!/bin/bash
# Build the shared melanin FCIDUMP on a CPU node (bwForCluster Helix).
#
# The dump is pyscf on the CPU (SCF + AO->MO transform, several GB, no GPU), so it does not
# belong inside a GPU allocation. This runs it on cpu-single, then the DIP and SIP GPU jobs
# (melanin_dip.sbatch / melanin_sip.sbatch) read the one FCIDUMP it produces -- DIP and SIP
# share the identical orbital space (active 25 to 236 -> 212 orbitals).
#
# Outputs go to $ADCGO_WS, a Helix workspace on the parallel FS (NOT $HOME): the FCIDUMP is
# several GB and $HOME has a quota. dump_fcidump.py resolves &output paths relative to the
# input file's directory, so we stage a copy of the .in (with geometry/basis symlinked in)
# into $WS and the outputs land there.
#
#   sbatch scripts/dump_melanin.sbatch          # or via scripts/submit_melanin.sh
#
# Env: ADCGO_PYTHON (python with pyscf), ADCGO_WS (workspace dir).

#SBATCH --account=bw22i008
#SBATCH --job-name=melanin-dump
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=180gb
#SBATCH --time=120:00:00
#SBATCH --output=melanin-dump-%j.out
set -euo pipefail

# Recover the repo root (slurm runs a spool copy, so $0 is not in scripts/).
self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" \
        | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/.." && pwd)"
cd "$root"

PY="${ADCGO_PYTHON:-$HOME/miniconda3/envs/adcgo/bin/python}"
source "$root/scripts/melanin_ws.sh"   # resolves + exports ADCGO_WS (ws_allocate, not mkdir)
WS="$ADCGO_WS"
ex="$root/examples/melanin"

[ -x "$PY" ] || { echo "no python at $PY; set ADCGO_PYTHON to the adcgo conda env" >&2; exit 1; }

# Stage inputs into the workspace so the big outputs land there too.
ln -sf "$ex/zmat"           "$WS/zmat"
ln -sf "$ex/_basisset.dat"  "$WS/_basisset.dat"
cp -f  "$ex/melanin_dip.in" "$WS/melanin.in"

echo "host=$(hostname)  WS=$WS  PY=$PY"
echo "#begin<dump>"
OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 "$PY" scripts/dump_fcidump.py --input "$WS/melanin.in"
echo "#end<dump>"
echo "wrote: $WS/melanin.fcidump  $WS/melanin.mo.json"
