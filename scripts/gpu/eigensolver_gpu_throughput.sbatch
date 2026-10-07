#!/bin/bash
# PHASE 0 GATE for the two-stage band reduction: what does the device eigenvector replay actually
# achieve, and what block size should it use?
#
# The whole case for offloading the strip rotations rests on a throughput figure that has never been
# measured — the device path was only ever checked for bit-exactness. The projected total for the
# two-stage design swings between ~2.6 h and ~14.5 h entirely on this one number.
#
# The block-size sweep is the substance, not tuning noise. The replay's only parallelism is the
# accumulated row count (3080 in production), so the default 256 threads is 12 blocks on a 132-SM
# H200 — about 9% of the machine. Spreading the same work over more, smaller blocks costs nothing
# and may be worth several times.
#
# Gate: >= 1 TB/s effective (~3.1e10 updates/s, ~10x the CPU) -> proceed to Phase 1.
#        < 300 GB/s even after the sweep -> stop and re-plan; the kernel needs redesigning first.
#SBATCH --account=bw22i008
#SBATCH --job-name=eig-gpu-tput
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# H200 pinned: an unpinned request lands on an A40 as often as not, and a throughput number from a
# different architecture answers a different question.
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=48gb
#SBATCH --time=00:30:00
#SBATCH --output=eig-gpu-tput-%j.out
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
echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
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
# -count=1: go test caches by package and flags, NOT by which GPU the job got.
go test -tags cuda -count=1 ./internal/adc/lanczos/ -run 'TestGPUBandEigReplayParity' -timeout 20m
echo "#end<parity> rc=$?"
echo "#begin<throughput>  (MB/s column = eigenvector updates/s)"
# The sweep is over RUN LENGTH, not block size. The block-size sweep was already done and came out
# flat (195 GB/s at every value from 32 to 1024 threads) — that flatness is what identified the
# sequential kernel as latency-bound with no occupancy to hide it, and it is why the 2-D form exists.
# What is unmeasured is how the 2-D form scales with the ops in a run:
#
#   ops=1      the launch-overhead floor. Production issues one launch per chase, 9.5e8 of them at
#              band 3079, so this sets whether the projected 7 h -> 0.5 h is reachable at all.
#   ops=100    a real chase at band 3079 (dim/band).
#   ops=2406   a real chase after a two-stage reduction to b2 = 128, which is what Phase A targets.
#
# Each length is run seq and par, so the gain is a difference at fixed work rather than a comparison
# against a different benchmark. ops=1/par must match ops=1/seq — the implementation falls back for a
# single-op run — so a gap there is measuring the harness.
go test -tags cuda -count=1 ./internal/adc/lanczos/ \
    -run XXX -bench BenchmarkGPUBandEigReplay -benchtime 200x 2>&1 | grep -E 'Benchmark|ok|FAIL|panic'
echo "#end<throughput>"
