#!/bin/bash
# GPU validation for the jiiLKK contraction path (docs/sigma_build_contractions.md).
#
# Runs, in order of increasing scope:
#   TestJIIFillDeviceMatchesHost      — the fill kernel's blocks vs the host builder, ENTRY BY
#                                       ENTRY. This is the per-element oracle the contraction
#                                       rewrite was warned would be lost; it is recoverable here
#                                       precisely because the fill kernel's output IS the block.
#   TestJIIMatFreeBatchedDeviceParity — fill + batched GEMM vs the host loop applier.
#   TestSatelliteMatFreeDeviceParity  — the pre-existing per-scalar device path, unchanged, to
#                                       prove this work did not disturb it.
#
#SBATCH --account=bw22i008
#SBATCH --job-name=jii-contraction
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --time=00:40:00
#SBATCH --output=jii-contraction-%j.out
set -euo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/.." && pwd)"
cd "$root"

if command -v module >/dev/null 2>&1; then
    module load compiler/gnu/11.3 devel/cuda/13.2 || true
fi
: "${CUDA_HOME:=${CUDA_INSTALL_PATH:-${CUDA_ROOT:-/usr/local/cuda}}}"
export PATH="/home/hd/hd_hd/hd_hh323/opt/go/bin:$CUDA_HOME/bin:$PATH"
export CGO_CFLAGS="-I$CUDA_HOME/include ${CGO_CFLAGS:-}"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 ${CGO_LDFLAGS:-}"

echo "host=$(hostname)  go=$(go version)"
nvidia-smi -L || true

echo "#begin<nvcc>"
for ker in adc4_kernels adc2dip_kernels; do
    echo "nvcc: $ker.cu"
    nvcc -O3 -std=c++14 \
         -gencode arch=compute_80,code=sm_80 \
         -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_90,code=sm_90 \
         -c "internal/adc/backend/$ker.cu" \
         -o "internal/adc/backend/$ker.o"
done
echo "#end<nvcc>"

echo "#begin<gotest>"
go test -tags cuda -v -run 'TestJIIFillDeviceMatchesHost|TestJIIMatFreeBatchedDeviceParity|TestSatelliteMatFreeDeviceParity' ./internal/adc/dip/
echo "#end<gotest>"
