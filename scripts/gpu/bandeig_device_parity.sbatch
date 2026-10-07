#!/bin/bash
# Confirms the GPU replay of the banded eigensolver's eigenvector rotations is BIT-IDENTICAL to the
# host replay (TestGPUBandEigReplayParity), on real hardware with the real toolchain.
#
# This cannot be checked anywhere else. The hazard is FMA contraction: Go on amd64 does not fuse
# a*b+c, nvcc and hipcc do by default, and a fused product rounds once where Go rounds twice.
# backend/bandeig_kernels.cu uses explicit __dmul_rn/__dadd_rn/__dsub_rn to prevent that, but
# "prevented" is a claim about a specific compiler on a specific architecture, and a mismatch shows
# up as a wrong pole strength rather than as a crash. -eig-device stays off by default until this
# passes on the hardware in question.
#SBATCH --account=bw22i008
#SBATCH --job-name=bandeig-parity
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# H200 pinned, not a bare gpu:1. An unpinned request on this cluster lands on an A40 as often as
# not, and the production target is sm_90; a parity result from a different architecture is a
# different claim. (The first passing run was on an A40 — correct, but not the answer needed.)
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=32gb
#SBATCH --time=00:25:00
#SBATCH --output=bandeig-parity-%j.out
set -u
root=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
cd "$root"
pre_go=$(command -v go)
module load compiler/gnu/11.3 devel/cuda/13.2 2>/dev/null || true
export PATH="$(dirname "$pre_go"):$PATH"
export CUDA_HOME=${CUDA_INSTALL_PATH:-/opt/bwhpc/common/devel/cuda/13.2.1}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 -lcudart -lcublas -lcusolver"
export ADCGO_CUDA_STRICT=1
echo "host=$(hostname) date=$(date -Is)"
# compute_cap, not just the model name: a pass is a claim about an architecture, and this cluster
# hands out A40 (sm_86) and H200 (sm_90) from the same pool unless --gres pins the model.
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader
# The kernel object must be current: `go build -tags cuda` links whatever .o is on disk and will
# NOT recompile the .cu.
if [ backend/bandeig_kernels.cu -nt backend/bandeig_kernels.o ]; then
    echo "recompiling bandeig_kernels.cu"
    # Compile to a job-unique temp and rename. Two jobs that both find the .o stale would otherwise
    # write the same path concurrently, and a link picking up a half-written object fails in a way that
    # looks like a kernel bug. rename(2) is atomic within the filesystem, so a concurrent `go build`
    # sees either the old object or the new one, both of which are correct code.
    nvcc -O3 -std=c++14 -gencode arch=compute_90,code=sm_90 \
        -c backend/bandeig_kernels.cu -o "backend/bandeig_kernels.o.$SLURM_JOB_ID" || exit 1
    mv -f "backend/bandeig_kernels.o.$SLURM_JOB_ID" backend/bandeig_kernels.o || exit 1
fi
echo "#begin<parity>"
# -count=1 is MANDATORY here, not hygiene. go test caches by package and flags, not by which GPU
# the job landed on, so without it a result from an earlier run on different hardware is replayed
# as a pass. That is exactly what happened: job 15022223 did land on an H200 (o09c01) but reported
# "(cached)" — a genuine A40 pass, replayed, for a test that never executed on sm_90.
go test -tags cuda -count=1 ./internal/adc/lanczos/ -run 'TestGPUBandEigReplayParity' -v -timeout 20m
rc=$?
echo "#end<parity> rc=$rc"
exit $rc
