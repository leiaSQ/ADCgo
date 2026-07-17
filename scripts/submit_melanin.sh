#!/bin/bash
# Combined melanin queue for bwForCluster Helix. Submits the whole pipeline and wires the
# dependencies:
#
#   1. dump_melanin.sbatch   (cpu-single)        build the shared FCIDUMP + sidecar
#   2. melanin_dip.sbatch    (gpu:A100:2)  afterok:dump   DIP-ADC(2), block-Davidson
#   3. melanin_sip.sbatch    (gpu:H200:1)  afterok:dump   SIP-ADC(3), checkpointing Lanczos,
#                                                         self-resubmitting daisychain (gen 0)
#
# DIP and SIP share the one FCIDUMP (identical orbital space) and run in parallel once the
# dump succeeds. The SIP job then chains its own successors until it converges (see
# melanin_sip.sbatch); DIP is a single job (Davidson fits in one 120 h allocation).
#
#   scripts/submit_melanin.sh
#
# Overridable env (export before running so the jobs inherit it):
#   ADCGO_WS        workspace dir for the FCIDUMP + checkpoints (default in the sbatch files)
#   ADCGO_DIP_GRES  gres for DIP, e.g. gpu:H200:2 to run Davidson on 80 GB GPUs (with larger
#                   NROOTS/MAXDAVSP); sbatch --gres override
#   ADCGO_SIP_GRES  gres for SIP, e.g. to use A100-80 instead of the default H200 (see
#                   melanin_sip.sbatch); sbatch --gres override
#   NROOTS,MAXDAVSP DIP Davidson knobs (inherited by melanin_dip.sbatch)
#   SKIP_DUMP=1     reuse an existing $ADCGO_WS FCIDUMP: submit DIP/SIP with no dump dependency
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

# Resolve the workspace ONCE here and export it, so all three jobs inherit one consistent
# path (SLURM propagates the submit environment) and no two jobs race to ws_allocate.
source "$here/melanin_ws.sh"
echo "workspace: $ADCGO_WS"

dip_gres=(); sip_gres=()
[ -n "${ADCGO_DIP_GRES:-}" ] && dip_gres=(--gres="$ADCGO_DIP_GRES")
[ -n "${ADCGO_SIP_GRES:-}" ] && sip_gres=(--gres="$ADCGO_SIP_GRES")

dep_dip=(); dep_sip=()
if [ "${SKIP_DUMP:-0}" != "1" ]; then
    jid="$(sbatch --parsable --export=ALL "$here/dump_melanin.sbatch")"
    echo "dump job $jid"
    dep_dip=(--dependency="afterok:$jid")
    dep_sip=(--dependency="afterok:$jid")
else
    echo "SKIP_DUMP=1: submitting DIP/SIP against the existing \$ADCGO_WS FCIDUMP"
fi

dip="$(sbatch --parsable "${dep_dip[@]}" "${dip_gres[@]}" --export=ALL "$here/melanin_dip.sbatch")"
echo "dip  job $dip  ${dep_dip[*]:-(no dep)}${ADCGO_DIP_GRES:+  gres=$ADCGO_DIP_GRES}"

sip="$(sbatch --parsable "${dep_sip[@]}" "${sip_gres[@]}" --export=ALL,GEN=0 "$here/melanin_sip.sbatch")"
echo "sip  job $sip  ${dep_sip[*]:-(no dep)}  (daisychain gen 0${ADCGO_SIP_GRES:+, gres=$ADCGO_SIP_GRES})"

echo
echo "watch:  squeue -u \"$USER\" -o '%.10i %.14j %.10T %.12r %.20E %R'"
