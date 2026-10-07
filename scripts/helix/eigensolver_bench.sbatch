#!/bin/bash
# 64-core CPU measurement for the Mode B banded eigensolver parallelization.
#
# WHY IT EXISTS. The login node's cgroup caps this account at 4 CPUs (nproc reports 128), so no
# measurement taken there says anything about how the replay scales or what a barrier costs at
# the width production runs with. Two numbers decided the design and both need a real node:
#
#  1. parallel: FixedPool round latency vs Chunks round latency at 64 workers. The chase-level
#     synchronization in bnd2td fires ~9.5e8 times at the production DIP shape, so a round that
#     costs 2 us is half an hour of overhead and one that costs 30 us is eight hours. On 4 CPUs
#     the pool measured SLOWER than Chunks (11 us vs 1.7 us) purely from oversubscription, which
#     tells us nothing about 64 spinning workers on 64 cores.
#  2. lanczos: BenchmarkZReplayBlock, whose curve sets zRowMin, and the tier-1 solve benchmarks
#     Ref / plain / Narrow, whose ratios are the actual speedup on this hardware.
#
# CPU only, no GPU: the eigensolve never touches a device.
#SBATCH --account=bw22i008
#SBATCH --job-name=eig-bench
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64gb
#SBATCH --time=00:40:00
#SBATCH --output=eig-bench-%j.out
set -u
root=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
cd "$root"
# Save the toolchain's own bin dir BEFORE any module load: compiler/gnu puts gccgo 1.16 ahead of
# Go 1.26 on PATH, which then rejects the go.mod language version.
pre_go=$(command -v go)
export PATH="$(dirname "$pre_go"):$PATH"
export GOMAXPROCS=${SLURM_CPUS_PER_TASK:-64}
# Iteration counts, lowered when this is submitted to the 30-minute devel partition for a quick
# answer. The tier-1 solve is ~4 s per iteration serial, so 5x is a stable mean and 2x is not.
BT_ROUND=${BT_ROUND:-50000x}
BT_REPLAY=${BT_REPLAY:-20x}
BT_SOLVE=${BT_SOLVE:-5x}
SKIP_SLOW_TESTS=${SKIP_SLOW_TESTS:-0}
echo "host=$(hostname)  date=$(date -Is)  cpus=$SLURM_CPUS_PER_TASK  GOMAXPROCS=$GOMAXPROCS"
go version
echo
echo "================ parallel: round latency ================"
go test ./internal/adc/parallel/ -run XXX -bench 'Round' -benchtime $BT_ROUND
echo
echo "================ lanczos: replay throughput vs block ================"
go test ./internal/adc/lanczos/ -run XXX -bench 'BenchmarkZReplayBlock' -benchtime $BT_REPLAY
echo
echo "================ lanczos: tier-1 solve, reference vs parallel ================"
go test ./internal/adc/lanczos/ -run XXX \
    -bench 'BenchmarkBandSymDiagFastRef|BenchmarkBandSymDiagFast$|BenchmarkBandSymDiagFastNarrow|BenchmarkBnd2tdSpine|BenchmarkBnd2tdFull|BenchmarkTddiagZ' \
    -benchtime $BT_SOLVE -timeout 60m
echo
echo "================ correctness at 64 workers ================"
short=""
if [ "$SKIP_SLOW_TESTS" = 1 ]; then short="-short"; fi
go test ./internal/adc/lanczos/ $short -run 'TestBandEig|TestZTape|TestBandSymDiag' -timeout 60m
rc1=$?
go test ./internal/adc/parallel/ -race -count 2
echo "done tests rc=$rc1/$?"
