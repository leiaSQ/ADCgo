#!/bin/bash
#SBATCH --account=bw22i008
#SBATCH --job-name=isrgen
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=isrgen-%j.log
#
# Regenerate the adcgen ISR evaluator packages internal/adc/isrgen/{ip,dip,tip,qip}
# Each generation is single-threaded sympy, so the four run
# side by side on one node, each with its own log. The login node is capped at 4 CPUs
# and the fidelity reference alone takes minutes per block there.
#
#   sbatch scripts/codegen/isrgen.sbatch                 # all four
#   VARIANTS="tip qip" sbatch scripts/codegen/isrgen.sbatch
#
# What each package carries (see scripts/codegen/generate_adc.py for the schemes):
#   ip   strict:3 and ADC(2,2)m/x/f, every block (checks sip's ADC(3) and A9-A22)
#   dip  strict:2, adc2x and ci, every block (checks the dip package; CI reference)
#   tip  strict:2, adc2x and ci on B00 B01 B11 B12; 5h2p/5h2p (B22) is left to khci's
#        Slater-Condon path: its first order alone did not finish in 10 min
#   qip  strict:2, adc2x and ci on B00 B01 B11 (the qip spike)
set -uo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" \
        | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/../.." && pwd)"
cd "$root"

PY="${ADCGO_PYTHON:-$HOME/miniconda3/envs/adcgo/bin/python}"
[ -x "$PY" ] || { echo "no python at $PY; set ADCGO_PYTHON" >&2; exit 1; }
export OMP_NUM_THREADS=1

declare -A ARGS=(
    [ip]="--schemes strict:3,adc22m,adc22x,adc22f"
    [dip]="--schemes strict:2,adc2x,ci"
    [tip]="--schemes strict:2,adc2x,ci --blocks B00,B01,B11,B12"
    [qip]="--schemes strict:2,adc2x,ci --blocks B00,B01,B11"
)
VARIANTS="${VARIANTS:-ip dip tip qip}"

echo "host=$(hostname) root=$root variants=$VARIANTS"
pids=()
for v in $VARIANTS; do
    [ -n "${ARGS[$v]:-}" ] || { echo "unknown variant $v" >&2; exit 1; }
    ( "$PY" scripts/codegen/generate_adc.py --variant "$v" ${ARGS[$v]} \
          --outdir "internal/adc/isrgen/$v" > "isrgen-$v-${SLURM_JOB_ID:-local}.log" 2>&1
      st=$?
      echo "variant $v exit $st"
      exit $st ) &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
for v in $VARIANTS; do
    echo "===== $v"
    tail -25 "isrgen-$v-${SLURM_JOB_ID:-local}.log"
done
exit $rc
