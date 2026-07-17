#!/bin/bash
# melanin DIP-ADC(2) on bwForCluster Helix. Reads the shared FCIDUMP from $ADCGO_WS
# (build it first: scripts/dump_melanin.sbatch, or use scripts/submit_melanin.sh).
#
# WHY DAVIDSON, NOT LANCZOS: the DIP-ADC(2) space here is enormous — n ≈ 10 M (singlet) /
# 15 M (triplet) configurations. A full-band block-Lanczos would allocate an n×(blocks·main)
# basis of ~25–36 TB and cannot even hold a single Krylov block (137 GB) on any GPU. Block
# Davidson caps the subspace at -maxdavsp and targets the -nroots lowest roots, so it holds
# two n×maxdim panels (~28 GB at the defaults below) and fits a 40 GB A100. The trade-off is
# that you get the lowest -nroots states, not the whole double-ionization band; raise -nroots
# (and -maxdavsp > 2·nroots) for more, which needs an 80 GB GPU (A100-80 / H200).
#
# Two spin sectors (singlet, triplet) run one per GPU -> 2 A100 (fp64: A100, not A40).
#
# The defaults below (NROOTS=30, MAXDAVSP=90) fit a 40 GB A100 — widely available, so the job
# schedules and makes progress now. For more states on an 80 GB GPU, submit with an 80 GB gres
# and larger counts, e.g.:
#   ADCGO_DIP_GRES=gpu:H200:2 NROOTS=64 MAXDAVSP=192 scripts/submit_melanin.sh
# The binding sector is the triplet (n≈14.8 M); GPU memory ≈ 2·n·(MAXDAVSP+NROOTS)·8 bytes, so
# ~60 GB at 64/192 and a practical ceiling of ~78 roots on an 80 GB card (keep MAXDAVSP≈3·NROOTS).
#
#   sbatch scripts/melanin_dip.sbatch
#
# Env: ADCGO (cuda binary), ADCGO_WS (workspace), NROOTS, MAXDAVSP (gres override: submit-time).

#SBATCH --account=bw22i008
#SBATCH --job-name=melanin-dip
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=120gb
#SBATCH --time=120:00:00
#SBATCH --output=melanin-dip-%j.out
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
NROOTS="${NROOTS:-30}"       # lowest DIP roots per spin sector (the spectral knob)
MAXDAVSP="${MAXDAVSP:-90}"   # Davidson subspace cap; keep > 2·NROOTS. Memory ~ 2·n·MAXDAVSP·8
fcidump="$WS/melanin.fcidump"
mo="$WS/melanin.mo.json"

[ -x "$ADCGO" ]   || { echo "no cuda binary at $ADCGO; build with scripts/build_adcgo_cuda_helix" >&2; exit 1; }
[ -f "$fcidump" ] || { echo "no FCIDUMP at $fcidump; run scripts/dump_melanin.sbatch first" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "host=$(hostname)  fcidump=$fcidump  nroots=$NROOTS maxdavsp=$MAXDAVSP"
nvidia-smi -L 2>/dev/null || true
echo "#begin<adc>"
"$ADCGO" -fcidump "$fcidump" -mo "$mo" \
    -dip -order 2 -solver davidson -spin both -nroots "$NROOTS" -maxdavsp "$MAXDAVSP" \
    -backend cuda -sym all \
    -out "$WS/melanin.dip.json"
echo "#end<adc>"
echo "dip -> $WS/melanin.dip.json"
