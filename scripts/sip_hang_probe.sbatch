#!/bin/bash
# Gate job for the production SIP resubmit.
#
# Runs the SIP path on uracil2W_dz with the CUDA backend — the production configuration, at a
# size where a healthy run finishes in minutes. -order 2 skips c11_3 so the assemble is
# seconds and the whole job exercises the post-assemble satellite apply, which is where the
# O(n2^2) stall lived (sip/matfree.go c22Buckets, adc4_kernels.cu c22_apply gate).
#
# EXIT CODE IS THE POINT: 0 only if the solve COMPLETES. If it is still running at the
# deadline it dumps goroutine stacks, kills the run and exits 1, so an afterok dependant
# never starts on an unfixed binary.
#SBATCH --account=bw22i008
#SBATCH --job-name=sip-hang-probe
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=120gb
#SBATCH --time=00:45:00
#SBATCH --output=sip-hang-probe-%j.out
set -u
root=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
cd "$root"
module load compiler/gnu/11.3 devel/cuda/13.2 2>/dev/null || true
export CUDA_HOME=${CUDA_INSTALL_PATH:-/opt/bwhpc/common/devel/cuda/13.2.1}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
FC=/gpfs/bwfor/work/ws/hd_hh323-uracil2W_dz/adcgo/uracil2W_dz/uracil2W_dz.fcidump
LOG=$root/sip-hang-probe-$SLURM_JOB_ID.log
DEADLINE=1500   # 25 min; a healthy run is minutes (measured 257 s for 20 blocks on 16 CPU cores)

echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
echo "binary: $(ls -l --time-style=+%F_%T adcgo-cuda | awk '{print $6, $7}')"
start=$(date +%s)
GOTRACEBACK=all ./adcgo-cuda -fcidump "$FC" -sip -order 2 -sigma off \
  -solver lanczos -blocks 20 -backend cuda -sym all -matfree auto \
  -out /dev/null > "$LOG" 2>&1 &
PID=$!

while kill -0 $PID 2>/dev/null; do
  now=$(( $(date +%s) - start ))
  if [ $now -ge $DEADLINE ]; then
    echo "=== STALLED: still running after ${now}s; dumping goroutines ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    kill -QUIT $PID; sleep 20; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
    echo "=== runnable adcgo frames ==="
    grep -A6 "\[runnable\]" "$LOG" | grep "ADCgo/" | head -30
    echo "PROBE RESULT: STALLED"
    exit 1
  fi
  sleep 10
done
wait $PID; rc=$?
el=$(( $(date +%s) - start ))
echo "=== run exited rc=$rc after ${el}s ==="
tail -15 "$LOG"
if [ $rc -ne 0 ]; then echo "PROBE RESULT: FAILED (rc=$rc)"; exit 1; fi
echo "PROBE RESULT: OK — SIP satellite apply completes on the CUDA path"
exit 0
