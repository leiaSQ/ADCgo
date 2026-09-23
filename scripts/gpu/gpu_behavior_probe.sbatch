#!/bin/bash
# GPU behaviour battery for the DIP -mgpu path.
#
# WHAT IT IS FOR. Job 14040960 died with cudaError_t 719 on o09c01 and 14561251 died with the
# same 719 on o08c02 — two different nodes, so the long-held "bad node" theory is already dead
# on the evidence. What was never possible before is ATTRIBUTION: every kernel launcher returned
# cudaGetLastError() and every call site discarded it, and devFree's C shim returned void, so a
# sticky asynchronous fault only surfaced later at an unrelated cudaDeviceSynchronize (which is
# why 719 was always reported from distBackend.gemmMatOne, where it was never raised).
#
# That is fixed: ckLaunch checks all five launchers, devFree reports, and ADCGO_CUDA_STRICT=1
# synchronizes after every launch so an illegal access is attributed to the launch that caused
# it. This job exercises the whole GPU surface with that instrumentation on, pinned to ONE node,
# so submitting it per node settles node-dependence with a controlled comparison rather than an
# anecdote.
#
# It also exercises code that had never run on hardware: the c22 kernel sparsity gate, the
# reusable remote-input band scratch, the hoisted per-source drain, and the opt-in device SymEig.
#
# Submit one per node:
#   for n in o08c01 o08c02 o09c01 o09c02; do sbatch --nodelist=$n scripts/gpu/gpu_behavior_probe.sbatch; done
#SBATCH --account=bw22i008
#SBATCH --job-name=gpu-probe
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=240gb
#SBATCH --time=01:30:00
#SBATCH --output=gpu-probe-%j.out
set -u
root=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
cd "$root"
module load compiler/gnu/11.3 devel/cuda/13.2 2>/dev/null || true
export CUDA_HOME=${CUDA_INSTALL_PATH:-/opt/bwhpc/common/devel/cuda/13.2.1}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 -lcudart -lcublas -lcusolver"
export PATH=/home/hd/hd_hd/hd_hh323/opt/go/bin:$PATH
FC=/gpfs/bwfor/work/ws/hd_hh323-uracil2W_dz/adcgo/uracil2W_dz/uracil2W_dz.fcidump
fail=0
hdr() { echo; echo "################ $* ################"; }

echo "host=$(hostname)  date=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
hdr "0. peer topology (NV* = NVLink, SYS = PCIe: timings from SYS are not comparable)"
nvidia-smi topo -m 2>&1 | head -12

hdr "1. cuda-tagged unit tests on real hardware"
# Covers the kernel parity suites plus the distributed backend: the c22 sparsity gate, the
# batched fill/apply, the per-device satellite path, and the reworked remote-input band.
# Full output to a log, never piped through tail: a panic's MESSAGE precedes its stack, so
# tailing the stream discards exactly the line that says what went wrong.
T1=$root/gpu-probe-$SLURM_JOB_ID.units.log
go test -tags cuda -count=1 -timeout 40m -v ./backend/ ./internal/adc/dip/ ./internal/adc/sip/ > "$T1" 2>&1
rc=$?
grep -E "^--- FAIL|^--- PASS|^ok |^FAIL" "$T1" | tail -30
if [ "$rc" != 0 ]; then
  echo "FAIL: cuda unit tests rc=$rc"; fail=1
  echo "---- first failure, with its panic message and 40 lines of context ----"
  grep -nE "^--- FAIL|panic:|dip: partition|cudaError|cublasStatus|launch failed" "$T1" | head -20
  first=$(grep -nE "panic:|dip: partition" "$T1" | head -1 | cut -d: -f1)
  [ -n "$first" ] && sed -n "$((first>25?first-25:1)),$((first+40))p" "$T1"
fi

hdr "2. same suites under compute-sanitizer (out-of-bounds / race localization)"
if command -v compute-sanitizer >/dev/null 2>&1; then
  go test -tags cuda -c -o /tmp/dip.$SLURM_JOB_ID.test ./internal/adc/dip/ 2>&1 | tail -3
  T2=$root/gpu-probe-$SLURM_JOB_ID.sanitizer.log
  compute-sanitizer --tool memcheck --error-exitcode 9 /tmp/dip.$SLURM_JOB_ID.test -test.timeout 30m > "$T2" 2>&1
  rc=$?
  grep -E "ERROR SUMMARY|^--- FAIL|^FAIL|^ok " "$T2" | tail -15
  [ "$rc" = 0 ] || { echo "FAIL: sanitizer rc=$rc (note: a Go test failure and a sanitizer finding both land here; ERROR SUMMARY above discriminates)"; fail=1; }
  rm -f /tmp/dip.$SLURM_JOB_ID.test
else
  echo "compute-sanitizer not on PATH; skipped"
fi

hdr "3. DIP -mgpu 4 with ADCGO_CUDA_STRICT=1 (every launch synced and checked)"
# -mgpu is the configuration that produced the 719, and it is the only one that generates
# remote-input blocks, so this is what exercises the reworked band scratch and hoisted drain.
ADCGO_CUDA_STRICT=1 timeout 2400 ./adcgo-cuda -fcidump "$FC" \
  -dip -order 2 -solver lanczos-lowmem -lowmem-block 0 -mgpu 4 \
  -spin singlet -blocks 3 -backend cuda -sym all -profile -matfree on -satchunk 64 \
  -out /dev/null 2>&1 | tail -25
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || { echo "FAIL: strict-mode -mgpu DIP rc=$rc (124 = timeout)"; fail=1; }

hdr "4. same without strict mode, for the timing comparison"
timeout 2400 ./adcgo-cuda -fcidump "$FC" \
  -dip -order 2 -solver lanczos-lowmem -lowmem-block 0 -mgpu 4 \
  -spin singlet -blocks 3 -backend cuda -sym all -profile -matfree on -satchunk 64 \
  -out /dev/null 2>&1 | grep -E "profile|sattrace|progress" | tail -8
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || { echo "FAIL: plain -mgpu DIP rc=$rc"; fail=1; }

hdr "5. opt-in device SymEig (-mgpu-device-symeig): runs, and moves lines only in the last digits"
timeout 1800 ./adcgo-cuda -fcidump "$FC" \
  -dip -order 2 -solver lanczos-lowmem -lowmem-block 0 -mgpu 4 -mgpu-device-symeig \
  -spin singlet -blocks 3 -backend cuda -sym all -matfree on \
  -out /dev/null 2>&1 | tail -6
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || { echo "FAIL: -mgpu-device-symeig rc=$rc"; fail=1; }

hdr "RESULT"
echo "host=$(hostname)"
if [ "$fail" = 0 ]; then echo "GPU PROBE: ALL PASS"; else echo "GPU PROBE: FAILURES ABOVE"; fi
exit $fail
