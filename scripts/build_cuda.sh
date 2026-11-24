#!/bin/bash
# Two-stage CUDA build of adcgo: compile the custom kernels with nvcc, then
# `go build -tags cuda`. Produces ./adcgo-cuda at the repo root (or $OUT).
#
#   scripts/build_cuda.sh
#
# `go build -tags cuda` alone does NOT rebuild the kernels: it links whatever backend/*.o
# are on disk. Re-run this script after pulling or editing any backend/*.cu.
#
# Environment (all optional):
#   CUDA_HOME     CUDA toolkit prefix          (default: $CUDA_INSTALL_PATH, $CUDA_ROOT, /usr/local/cuda)
#   CUDA_ARCHS    space-separated SM targets   (default: "80 86 90" = A100, A40/RTX 30xx, H100/H200)
#   OUT           output binary                (default: <repo>/adcgo-cuda)
#   MODULES       environment modules to load first, e.g. "compiler/gnu/11.3 devel/cuda/13.2"
#   PTXAS_VERBOSE 1 prints per-kernel register/spill counts (default 0)
#
# Needs: nvcc, a C/C++ host compiler, Go >= the version in go.mod, cuBLAS + cuSOLVER.
set -euo pipefail

if [ -n "${MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    # A compiler module can put an old gccgo `go` ahead of the user's Go on PATH; keep the
    # go that was resolvable before the load in front.
    pre_go="$(command -v go 2>/dev/null || true)"
    # shellcheck disable=SC2086
    module load $MODULES
    if [ -n "$pre_go" ] && [ "$(command -v go 2>/dev/null || true)" != "$pre_go" ]; then
        PATH="$(dirname "$pre_go"):$PATH"
        export PATH
    fi
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
OUT="${OUT:-$root/adcgo-cuda}"

: "${CUDA_HOME:=${CUDA_INSTALL_PATH:-${CUDA_ROOT:-/usr/local/cuda}}}"
export CGO_CFLAGS="-I$CUDA_HOME/include ${CGO_CFLAGS:-}"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 ${CGO_LDFLAGS:-}"

command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found: put the CUDA toolkit on PATH (or set MODULES)" >&2; exit 1; }
command -v go   >/dev/null 2>&1 || { echo "go not found: install Go (see README)" >&2; exit 1; }

need="$(awk '/^go /{print $2; exit}' go.mod)"
have="$(go env GOVERSION 2>/dev/null | sed 's/^go//')"
if [ -n "$need" ] && [ -n "$have" ] &&
    [ "$(printf '%s\n%s\n' "$need" "$have" | sort -V | tail -1)" != "$have" ]; then
    echo "ERROR: go.mod needs Go >= $need but 'go' is $have" >&2
    exit 1
fi

echo "CUDA_HOME = $CUDA_HOME"
echo "nvcc      = $(nvcc --version | tail -1)"
echo "go        = $(go version)"

gencode=()
for sm in ${CUDA_ARCHS:-80 86 90}; do
    gencode+=(-gencode "arch=compute_${sm},code=sm_${sm}")
done
ptxas=()
[ "${PTXAS_VERBOSE:-0}" = "1" ] && ptxas=(-Xptxas -v)

# All three objects are linked by backend/cuda_kernels.go's #cgo LDFLAGS; a missing one
# fails the link with "cannot find …_kernels.o".
for ker in sip_kernels adc2dip_kernels bandeig_kernels; do
    echo "nvcc: backend/$ker.cu"
    nvcc -O3 -std=c++14 ${ptxas[@]+"${ptxas[@]}"} "${gencode[@]}" \
        -c "backend/$ker.cu" -o "backend/$ker.o"
done

echo "go build -tags cuda -> $OUT"
go build -tags cuda -o "$OUT" ./cmd/adcgo
ls -l "$OUT"
