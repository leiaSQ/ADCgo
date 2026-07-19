#!/bin/bash
# Queue the uracil DIP-ADC(2) pipeline on bwForCluster Helix. For each molecule
# it submits, with the dependency wired:
#
#   1. uracil_dump.sbatch  (cpu-single)                build the FCIDUMP + sidecar
#   2. uracil_dip.sbatch   (gpu:H200:2)  afterok:dump  whole-band lanczos-lowmem
#
# These are the DIP runs that could not finish on theADCcode in a sane walltime
# (../thesis/uracil1W, ../thesis/uracil2W). Build the CUDA binary first:
#     scripts/build_adcgo_cuda_helix        # -> ./adcgo-cuda
#
#   scripts/submit_uracil.sh                # both molecules
#   scripts/submit_uracil.sh uracil2W       # just one
#
# Overridable env (export before running so the jobs inherit it):
#   ADCGO_WS_NAME   force one workspace name for all mols (default: per-mol = $MOL)
#   ADCGO_DIP_GRES  gres for the DIP job (default gpu:H200:2 in the sbatch). The DIP
#                   operator+panels need a ≥80 GB GPU; H200 (141 GB) is the safe target.
#                   Override, e.g. gpu:A100:2, only if you know the sector fits (sbatch --gres).
#   BLOCKS          Lanczos iterations (default 200 = theADCcode iter 200)
#   LOWMEM_BLOCK    band width (0 = faithful whole band, default)
#   BACKEND         cuda (default) | gonum (CPU fallback -- also flip the partition)
#   SKIP_DUMP=1     reuse existing $ADCGO_WS FCIDUMPs: submit DIP with no dump dep
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

mols=("$@")
[ "${#mols[@]}" -eq 0 ] && mols=(uracil1W uracil2W)

dip_gres=()
[ -n "${ADCGO_DIP_GRES:-}" ] && dip_gres=(--gres="$ADCGO_DIP_GRES")

for mol in "${mols[@]}"; do
    [ -f "$here/../examples/$mol/${mol}_dip.in" ] || { echo "unknown molecule '$mol' (no examples/$mol/${mol}_dip.in)" >&2; exit 1; }
    echo "=== $mol ==="

    dep_dip=()
    if [ "${SKIP_DUMP:-0}" != "1" ]; then
        jid="$(sbatch --parsable --job-name="${mol}-dump" \
                 --export=ALL,MOL="$mol" "$here/uracil_dump.sbatch")"
        echo "  dump job $jid"
        dep_dip=(--dependency="afterok:$jid")
    else
        echo "  SKIP_DUMP=1: submitting DIP against the existing \$ADCGO_WS FCIDUMP"
    fi

    dip="$(sbatch --parsable "${dep_dip[@]}" "${dip_gres[@]}" \
             --job-name="${mol}-dip" --export=ALL,MOL="$mol" "$here/uracil_dip.sbatch")"
    echo "  dip  job $dip  ${dep_dip[*]:-(no dep)}${ADCGO_DIP_GRES:+  gres=$ADCGO_DIP_GRES}"
done

echo
echo "watch:  squeue -u \"$USER\" -o '%.10i %.14j %.10T %.12r %.20E %R'"
