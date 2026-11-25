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
    # A compiler module can put an old gccgo `go` ahead of the user's Go on PATH; keep the
    # go that was resolvable before the load in front (as scripts/build_cuda.sh does).
    pre_go="$(command -v go 2>/dev/null || true)"
    # shellcheck disable=SC2086
    module load $MODULES
    if [ -n "$pre_go" ] && [ "$(command -v go 2>/dev/null || true)" != "$pre_go" ]; then
        PATH="$(dirname "$pre_go"):$PATH"
        export PATH
    fi
fi
command -v go >/dev/null 2>&1 || { echo "go not found: install Go (see README)" >&2; exit 1; }
need="$(awk '/^go /{print $2; exit}' go.mod)"
have="$(go env GOVERSION 2>/dev/null | sed 's/^go//')"
if [ -n "$need" ] && [ -n "$have" ] &&
    [ "$(printf '%s\n%s\n' "$need" "$have" | sort -V | tail -1)" != "$have" ]; then
    echo "ERROR: go.mod needs Go >= $need but 'go' is ${have:-unknown} ($(command -v go))" >&2
    exit 1
fi
: "${CUDA_HOME:=${CUDA_INSTALL_PATH:-${CUDA_ROOT:-/usr/local/cuda}}}"
export CGO_CFLAGS="-I$CUDA_HOME/include ${CGO_CFLAGS:-}"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 ${CGO_LDFLAGS:-}"
export ADCGO_CUDA_STRICT=1
ADCGO="${ADCGO:-$root/adcgo-cuda}"
ngpu="$(nvidia-smi -L | wc -l)"
nvidia-smi -L

echo "go = $(go version)"
[ -x "$ADCGO" ] || { echo "ERROR: $ADCGO not found: run scripts/build_cuda.sh first" >&2; exit 1; }

# -mgpu writes each finished sector beside -out, so the outputs need a writable directory.
out="$(mktemp -d "${TMPDIR:-/tmp}/adcgo-gpu-tests.XXXXXX")"
trap 'rm -rf "$out"' EXIT

status=()
stage() { # stage NAME CMD...: run one stage, record PASS/FAIL, keep going
    local name="$1"; shift
    echo "#### $name ####"
    if "$@"; then status+=("PASS  $name"); else status+=("FAIL  $name (exit $?)"); fi
}

stage "1. device parity tests" \
    go test -tags cuda -count=1 -timeout 30m ./backend/ ./internal/adc/...

mg=()
[ "$ngpu" -ge 2 ] && mg=(-mgpu "$ngpu")
stage "2. DIP-ADC(2) Mode B, matrix-free, row-partitioned over $ngpu GPU(s)" \
    "$ADCGO" -fcidump testdata/h2o_dzp.fcidump -mo testdata/h2o_dzp.mo.json -dip -spin singlet \
    -sym all -solver lanczos-lowmem -lowmem-block 0 -matfree on ${mg[@]+"${mg[@]}"} \
    -backend cuda -blocks 20 -out "$out/dip.json"

stage "3. SIP-ADC(3), matrix-free satellite on the GPU" \
    "$ADCGO" -fcidump testdata/h2o.fcidump -sip -sym all -solver lanczos -matfree on \
    -mainblock-cache off -sigma-cache off -backend cuda -out "$out/sip.json"

echo "#### summary ####"
printf '%s\n' "${status[@]}"
if printf '%s\n' "${status[@]}" | grep -q '^FAIL'; then
    echo "GPU TESTS FAILED" >&2; exit 1
fi
echo "GPU TESTS PASSED"
