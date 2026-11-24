#!/bin/bash
# Validate a CUDA build on real GPUs: the device parity tests, then one small DIP -mgpu and one
# SIP -matfree run end to end with every kernel launch checked (ADCGO_CUDA_STRICT=1).
#
#   sbatch --gres=gpu:2 scripts/slurm/gpu_tests.sbatch
#
# Run it after scripts/build_cuda.sh, on each GPU model you intend to use. -count=1 is not
# optional: go test caches by package and flags, not by GPU, so a cached pass from another
# machine would replay as a pass here.
#
# Environment: ADCGO (CUDA binary, default <repo>/adcgo-cuda), MODULES, CUDA_HOME.
#
#SBATCH --job-name=adcgo-gpu-tests
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=adcgo-gpu-tests-%j.out
set -euo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/../.." && pwd)"
cd "$root"
if [ -n "${MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $MODULES
fi
: "${CUDA_HOME:=${CUDA_INSTALL_PATH:-${CUDA_ROOT:-/usr/local/cuda}}}"
export CGO_CFLAGS="-I$CUDA_HOME/include ${CGO_CFLAGS:-}"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 ${CGO_LDFLAGS:-}"
export ADCGO_CUDA_STRICT=1
ADCGO="${ADCGO:-$root/adcgo-cuda}"
ngpu="$(nvidia-smi -L | wc -l)"
nvidia-smi -L

fail=0
echo "#### 1. device parity tests ####"
go test -tags cuda -count=1 -timeout 30m ./backend/ ./internal/adc/... || fail=1

echo "#### 2. DIP-ADC(2) Mode B, matrix-free, row-partitioned over $ngpu GPU(s) ####"
mg=()
[ "$ngpu" -ge 2 ] && mg=(-mgpu "$ngpu")
"$ADCGO" -fcidump testdata/h2o_dzp.fcidump -mo testdata/h2o_dzp.mo.json -dip -spin singlet \
    -sym all -solver lanczos-lowmem -lowmem-block 0 -matfree on ${mg[@]+"${mg[@]}"} \
    -backend cuda -blocks 20 -out /dev/null || fail=1

echo "#### 3. SIP-ADC(3), matrix-free satellite on the GPU ####"
"$ADCGO" -fcidump testdata/h2o.fcidump -sip -sym all -solver lanczos -matfree on \
    -mainblock-cache off -sigma-cache off -backend cuda -out /dev/null || fail=1

[ "$fail" = 0 ] && echo "GPU TESTS PASSED" || { echo "GPU TESTS FAILED" >&2; exit 1; }
