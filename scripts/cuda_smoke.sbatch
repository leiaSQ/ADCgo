#!/bin/bash
# Short hardware smoke for the CUDA code paths that have never executed on a GPU.
#
# Deliberately small and short so it backfills quickly (bf_window is 5 days, so a 15-minute job
# schedules almost immediately) — this is a confirmation, not an investigation.
#
# WHAT IT CONFIRMS
#  1. The stale-704 fix. cudaGetLastError() both reads and RESETS the per-thread error state.
#     The .cu launchers did not clear it before launching, and EnablePeerAccess tolerates
#     cudaErrorPeerAccessAlreadyEnabled (704) without clearing — so once ckLaunch began checking,
#     the next launcher reported that stale 704 as its own launch failure and
#     TestSatBatchedPerDeviceParity panicked on every node. That test is the direct regression
#     check: it needs peered devices, which is what raises the 704 in the first place.
#  2. ckLaunch itself, on all five launchers, plus the checked devFree.
#  3. The c22 kernel sparsity gate (adc4_kernels.cu).
#  4. The reworked remote-input path in distBackend.gemmMatOne: reusable band scratch instead of
#     a cudaMalloc/cudaFree per block, and one source drain per device per call instead of one
#     per block.
#  5. ADCGO_CUDA_STRICT=1 end to end.
#SBATCH --account=bw22i008
#SBATCH --job-name=cuda-smoke
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=60gb
#SBATCH --time=00:15:00
#SBATCH --output=cuda-smoke-%j.out
set -u
root=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
cd "$root"
module load compiler/gnu/11.3 devel/cuda/13.2 2>/dev/null || true
export CUDA_HOME=${CUDA_INSTALL_PATH:-/opt/bwhpc/common/devel/cuda/13.2.1}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 -lcudart -lcublas -lcusolver"
export PATH=/home/hd/hd_hd/hd_hh323/opt/go/bin:$PATH
fail=0
LOG=$root/cuda-smoke-$SLURM_JOB_ID.log
echo "host=$(hostname)  date=$(date -Is)"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo; echo "#### 1. cuda-tagged dip tests (peer/multi-device parity = the 704 regression check) ####"
# Full output to a log, never through tail: a panic's message precedes its stack.
go test -tags cuda -count=1 -timeout 10m -v ./internal/adc/dip/ > "$LOG" 2>&1
rc=$?
grep -E "^--- (FAIL|PASS): Test(SatBatched|Satellite|JII|C22|DipSat)" "$LOG" | head -20
grep -cE "^--- PASS" "$LOG" | sed 's/^/total PASS: /'
if [ "$rc" != 0 ]; then
  fail=1; echo "FAIL: cuda dip tests rc=$rc"
  echo "---- failure message with context ----"
  first=$(grep -nE "panic:|dip: partition|^--- FAIL" "$LOG" | head -1 | cut -d: -f1)
  [ -n "$first" ] && sed -n "$((first>20?first-20:1)),$((first+30))p" "$LOG"
fi

echo; echo "#### 2. one DIP -mgpu block under ADCGO_CUDA_STRICT=1 (remote-band path + every launch checked) ####"
ADCGO_CUDA_STRICT=1 timeout 500 ./adcgo-cuda \
  -fcidump /gpfs/bwfor/work/ws/hd_hh323-uracil2W_dz/adcgo/uracil2W_dz/uracil2W_dz.fcidump \
  -dip -order 2 -solver lanczos-lowmem -lowmem-block 0 -mgpu 4 \
  -spin singlet -blocks 1 -backend cuda -sym all -profile -matfree on \
  -out /dev/null 2>&1 | grep -E "dispatch|progress|profile|panic|cudaError" | tail -6
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || { echo "FAIL: strict -mgpu rc=$rc"; fail=1; }

echo; echo "#### RESULT ####"
echo "host=$(hostname)"
[ "$fail" = 0 ] && echo "CUDA SMOKE: PASS" || echo "CUDA SMOKE: FAIL"
exit $fail
