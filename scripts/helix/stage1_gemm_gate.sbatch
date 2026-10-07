#!/bin/bash
# GATE for SBR stage 1: what GEMM rate does internal/adc/lanczos actually get?
#
# Stage 1 is ~1.17e15 flops (half band update, half strip update), so the achieved rate decides whether
# it is a stage that disappears or one that costs more than the chase it replaces:
#
#   2.5 TFlop/s  ->   8 min      (threaded OpenBLAS)
#   400 GFlop/s  ->  49 min
#   100 GFlop/s  ->  3.3 h
#    20 GFlop/s  -> 16 h         (measured on 4 login cores, untagged pure-Go gonum)
#
# internal/adc/lanczos builds with NO build tag, so blas64.Gemm is gonum's pure-Go implementation
# unless -tags openblas swaps the engine in backend/openblas.go's init. This runs both and compares.
# Read the MB/s column as FLOP/s (SetBytes carries the flop count).
#SBATCH --account=bw22i008
#SBATCH --job-name=gemm-gate
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=gemm-gate-%j.out
set -u
root=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
cd "$root"
export PATH="$(dirname "$(command -v go)"):$PATH"
export GOMAXPROCS=${SLURM_CPUS_PER_TASK:-64}
export OMP_NUM_THREADS=$GOMAXPROCS
echo "host=$(hostname) date=$(date -Is) cpus=$SLURM_CPUS_PER_TASK"
go version
for tags in "" "openblas"; do
    echo
    if [ -z "$tags" ]; then echo "===== untagged (pure-Go gonum) ====="; else echo "===== -tags $tags ====="; fi
    # An openblas build needs netlib's C libraries; report the failure rather than hiding it.
    # StripApplyQ is the REAL call, not a synthetic product of the same shape: the strip is
    # column-major with a leading dimension, so the GEMM is issued transposed and strided and pays a
    # copy back. Whether gonum threads that as well as it threads a contiguous one is the question the
    # synthetic benchmark cannot answer, and stripApplyQ carries half of stage 1's flops.
    if ! go test ${tags:+-tags $tags} -run XXX -bench 'BenchmarkStage1Gemm|BenchmarkStripApplyQ' \
         -benchtime 20x ./internal/adc/lanczos/ 2>&1 | grep -E 'Benchmark|FAIL|cannot|error'; then
        echo "  (build or run failed for tags='${tags}')"
    fi
done
