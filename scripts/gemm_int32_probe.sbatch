#!/bin/bash
# Probe the >2^31-element GEMM output addressing — the one production-scale hazard the mgpu parity
# tests structurally cannot reach, and the leading remaining suspect for the CUDA 719 that killed
# the production DIP mgpu run (job 14040960).
#
# BACKGROUND: job 14150821 ran the whole mgpu satellite + batched + distributed-backend suite under
# compute-sanitizer on 4 H200s and came back with 0 errors. Those tests run at n ~= 120-960; the
# production DIP sector is n = 10,014,483 over 8 partitions with panel width 1711, where the GEMM
# output panel's far column sits at (1711-1)*1,251,810 ~= 2.1418e9 elements — 0.26% below
# INT32_MAX. Nothing at test scale can tell us whether that offset is computed in 32 or 64 bits.
#
# TestGemmLeadingDimensionCrossesInt32 straddles the boundary deliberately (ld = 1,300,000, so
# 2^31 falls between columns 1651 and 1652) and checks values, so a wrapped index fails whether it
# faults or silently writes elsewhere.
#
#   sbatch scripts/gemm_int32_probe.sbatch
#
# 1 GPU, 30 min: the GEMM is tiny (m=64, k=1, n=1711); only the 17.8 GB output allocation is large.

#SBATCH --account=bw22i008
#SBATCH --job-name=gemm-int32-probe
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --output=gemm-int32-probe-%j.out
set -uo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" \
        | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/.." && pwd)"
cd "$root"

pre_go="$(command -v go 2>/dev/null || true)"
module load compiler/gnu/11.3 devel/cuda/13.2 2>/dev/null || true
if [ -n "$pre_go" ] && [ "$(command -v go 2>/dev/null || true)" != "$pre_go" ]; then
    export PATH="$(dirname "$pre_go"):$PATH"
fi

: "${CUDA_HOME:=${CUDA_INSTALL_PATH:-${CUDA_ROOT:-/usr/local/cuda}}}"
export CGO_CFLAGS="-I$CUDA_HOME/include ${CGO_CFLAGS:-}"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 ${CGO_LDFLAGS:-}"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
SAN="$CUDA_HOME/bin/compute-sanitizer"

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "go = $(go version)"

echo "#begin<go test -c>"
go test -tags cuda -c -o backend.test ./backend || { echo "build failed" >&2; exit 1; }
echo "#end<go test -c>"

# Run from the package directory — fixtures are package-relative (see dip_mgpu_sanitize.sbatch).
cd "$root/backend"

echo
echo "################ 1. plain run (value check) ################"
"$root/backend.test" -test.run 'TestGemmLeadingDimensionCrossesInt32' -test.v -test.timeout 20m
rc_plain=$?
echo "---- plain exit=$rc_plain ----"

echo
echo "################ 2. under compute-sanitizer (fault localization) ################"
# If the addressing wraps, memcheck names the kernel and the offending address instead of letting
# it surface later as a 719 at some unrelated synchronization point.
"$SAN" --tool memcheck --launch-timeout 0 --report-api-errors no \
       "$root/backend.test" -test.run 'TestGemmLeadingDimensionCrossesInt32' -test.v -test.timeout 20m
rc_san=$?
echo "---- sanitizer exit=$rc_san ----"

echo
if [ "$rc_plain" -eq 0 ] && [ "$rc_san" -eq 0 ]; then
    echo "RESULT: GEMM output addressing IS 64-bit clean past 2^31 elements."
    echo "        That clears the last code-side suspect reachable without a production-scale run."
    echo "        Remaining: a hardware/Xid fault on o09c01, or something that only appears at the"
    echo "        full sector. Next step is a rerun of the failing sector pinned to a DIFFERENT node."
else
    echo "RESULT: FAILURE — the >2^31 output offset is NOT handled correctly. This is very likely"
    echo "        the root cause of the 719 in job 14040960. Fix by keeping each device's GEMM"
    echo "        output under 2^31 elements (smaller panel width, or more partitions), or by"
    echo "        routing these calls through the 64-bit cuBLAS entry points."
fi
