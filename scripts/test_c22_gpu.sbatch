#!/bin/bash
# Throwaway GPU validation: run the CUDA matrix-free parity tests (order-3 c22 + wert2)
# against the dense reference on real hardware. Cheap (h2o, one GPU, minutes).
#SBATCH --account=bw22i008
#SBATCH --job-name=c22-gpu-test
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=00:20:00
#SBATCH --output=c22-gpu-test-%j.out
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
nvcc -O3 -std=c++14 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     -c internal/adc/backend/adc4_kernels.cu \
     -o internal/adc/backend/adc4_kernels.o
echo "#end<nvcc>"

echo "#begin<gotest>"
go test -tags cuda -v -run 'MatFreeO3DeviceParity|Wert2MatFreeDeviceParity' ./internal/adc/sip/
echo "#end<gotest>"
