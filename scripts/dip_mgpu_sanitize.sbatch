#!/bin/bash
# compute-sanitizer sweep of the multi-GPU DIP kernels, to localize the cudaErrorLaunchFailure
# (CUDA 719) that killed the production DIP mgpu run (job 14040960) after 1 d 15 h.
#
# WHY THIS EXISTS: the kernel launchers (adc2dip_kernels.cu adc2_dip_sat_apply / adc2_dip_fill_sat)
# return only cudaGetLastError(), which catches launch-CONFIGURATION errors. A fault DURING kernel
# execution can therefore only surface at the next synchronization point — in 14040960 that was
# distributed.go:523's PeerCopier.Sync(), which is where the error was DETECTED, not where it
# happened. The traceback names the sync, so the faulting kernel is unrecoverable from the log.
#
# compute-sanitizer traps the access at its origin instead, reporting the kernel, the thread and
# the offending address. The mgpu tests exercise the same kernels and the same partition-boundary
# arithmetic (satRowBands, gatherSlabs, the per-device dip_sat_apply bands) at a scale that runs in
# minutes, so an out-of-bounds in that class is caught here without a 39 h reproduction.
#
#   sbatch scripts/dip_mgpu_sanitize.sbatch
#
# Reads nothing from the workspace: the tests build their own fixtures.
#
# 4 GPUs, not 8: TestSatelliteMatFreePerDeviceParity's minDev is 4 ("at 2 the partition-resolution
# and row-band clamping are trivial and a boundary bug can hide"), so 4 exercises every boundary
# path 8 does. It also schedules in hours instead of days — a 3 h job asking for a whole node
# backfills terribly (the first submission of this script, job 14134495, was estimated 4 days out).

#SBATCH --account=bw22i008
#SBATCH --job-name=dip-mgpu-sanitize
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=120gb
#SBATCH --time=03:00:00
#SBATCH --output=dip-mgpu-sanitize-%j.out
set -uo pipefail   # NOT -e: a sanitizer finding is a nonzero exit we want to REPORT, not abort on

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" \
        | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/.." && pwd)"
cd "$root"

# compiler/gnu/11.3 puts gccgo's ancient `go` on PATH ahead of the user's 1.26 — same trap the
# build script documents, so restore the pre-module go directory afterwards.
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
echo "go   = $(go version)"
echo "san  = $("$SAN" --version 2>&1 | head -1)"
nvidia-smi -L

# The committed .o files are prebuilt; recompile so the sanitizer has current line info for
# whatever is in the working tree.
echo "#begin<nvcc>"
for ker in adc4_kernels adc2dip_kernels; do
    nvcc -O3 -std=c++14 -lineinfo \
         -gencode arch=compute_90,code=sm_90 \
         -c "internal/adc/backend/$ker.cu" \
         -o "internal/adc/backend/$ker.o" || { echo "nvcc failed for $ker" >&2; exit 1; }
done
echo "#end<nvcc>"

# Build the test binaries once; running them under the sanitizer separately keeps `go test`'s own
# toolchain launches (compiler, linker) out of the sanitized process.
echo "#begin<go test -c>"
go test -tags cuda -c -o dip.test     ./internal/adc/dip     || { echo "build dip.test failed" >&2; exit 1; }
go test -tags cuda -c -o backend.test ./internal/adc/backend || { echo "build backend.test failed" >&2; exit 1; }
echo "#end<go test -c>"

rc_total=0
run() {   # run <label> <binary> <-run regexp>
    local label="$1" bin="$2" pat="$3"
    echo
    echo "################ $label ################"
    # --tool memcheck catches OOB / misaligned global, shared and local accesses.
    # --launch-timeout 0: the Go runtime can leave a kernel queued behind a long host phase.
    # --report-api-errors no: cudaGetLastError() polling makes benign sticky-error noise here.
    "$SAN" --tool memcheck --launch-timeout 0 --report-api-errors no \
           --print-limit 20 \
           "./$bin" -test.run "$pat" -test.v -test.timeout 60m
    local rc=$?
    echo "---- $label exit=$rc ----"
    [ "$rc" -ne 0 ] && rc_total=1
}

# The mgpu satellite path first: it is what the production DIP mgpu run was running.
run "dip mgpu satellite (per-device bands + NVLink gather)" dip.test 'MatFreePerDevice|MGPU|Mgpu|Dist'
# The batched fill/gemm path and the single-device satellite apply, same kernels, simpler driver.
run "dip matfree batched + device"                          dip.test 'MatFree|Batched|Satellite'
# Distributed backend plumbing: peer copies, row-band slicing, batched GEMM regrouping.
run "backend distributed + gpu"                             backend.test 'Dist|Peer|Gemm|Repro|MultiDevice'

echo
if [ "$rc_total" -eq 0 ]; then
    echo "RESULT: no sanitizer findings and no test failures."
    echo "        The 719 is then NOT an out-of-bounds reachable at test scale — next suspects are"
    echo "        a production-scale-only index path, or a hardware/Xid fault on o09c01 (719 looks the"
    echo "        same for both). Rerun the failing sector on a different node to discriminate."
else
    echo "RESULT: findings above — search this log for '========= Invalid' / '=== RUN' failures."
fi
