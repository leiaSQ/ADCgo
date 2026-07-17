#!/bin/bash
# Resolve $ADCGO_WS to a real bwForCluster Helix workspace and export it.
#
# Generic sibling of melanin_ws.sh: the workspace *name* (and the per-job subdir)
# come from $ADCGO_WS_NAME, so one script serves every molecule (uracil1W,
# uracil2W, ...). See melanin_ws.sh for the full rationale; the short version:
#
# WHY THIS EXISTS: scratch/work roots on Helix are provisioned by the workspace
# tools (ws_allocate), NOT by mkdir -- /gpfs/bwfor/{scratch,work}/... is
# root-owned, so a plain `mkdir -p /gpfs/.../<id>/...` fails with "Permission
# denied" and kills the job. Allocated workspaces live under
# /gpfs/bwfor/work/ws/<user>-<name> and expire, so the path is discovered at run
# time, never hard-coded.
#
# Sourced by uracil_dump.sbatch / uracil_dip.sbatch / submit_uracil.sh. After
# sourcing, $ADCGO_WS points at an existing, writable workspace subdir.
#
# Overridable env:
#   ADCGO_WS        skip discovery, use this path verbatim (must exist/be creatable)
#   ADCGO_WS_NAME   workspace name to find/allocate (default: adcgo)
#   ADCGO_WS_DAYS   duration when allocating a fresh workspace (default: 30, the gpfs cap)

_adcgo_resolve_ws() {
    local name="${ADCGO_WS_NAME:-adcgo}"
    local days="${ADCGO_WS_DAYS:-30}"

    # Explicit override wins -- caller takes responsibility for the path.
    if [ -n "${ADCGO_WS:-}" ]; then
        mkdir -p "$ADCGO_WS"
        return 0
    fi

    if ! command -v ws_find >/dev/null 2>&1; then
        echo "ws_find not on PATH; set ADCGO_WS to a writable path or load the workspace tools" >&2
        return 1
    fi

    local base
    base="$(ws_find "$name" 2>/dev/null || true)"
    if [ -z "$base" ]; then
        echo "no workspace '$name'; allocating one for $days days" >&2
        base="$(ws_allocate "$name" "$days" 2>/dev/null | grep '^/gpfs/' | head -1)"
    fi
    if [ -z "$base" ] || [ ! -d "$base" ]; then
        echo "could not resolve or allocate workspace '$name' (ws_allocate failed?)" >&2
        return 1
    fi

    export ADCGO_WS="$base/adcgo/$name"
    mkdir -p "$ADCGO_WS"
}

_adcgo_resolve_ws
