#!/bin/bash
#
# End-to-end ADCgo driver: dump the FCIDUMP from a sectioned input file, then run
# the adcgo solver per target irrep. This is ADCgo's analogue of theADCcode's
# ../ADCanalysis/examples/DIP_h2o/dip.in (which drove GAMESS-UK + theADCcode).
#
# Usage:
#   scripts/adcgo_run.sh <input.in> [adcgo-binary]
#
# Environment:
#   ADCGO_PYTHON   python with pyscf (default: the adcgo conda env)
#
# The &output section names the FCIDUMP/sidecar; the &adc section supplies the
# solver flags (`args`) and the target irreps (`sym`, 1-based like theADCcode, or
# `all`). adcgo's -sym is 0-based, so a 1-based irrep N maps to -sym (N-1).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <input.in> [adcgo-binary]" >&2
    exit 2
fi

input="$1"
here="$(cd "$(dirname "$0")" && pwd)"
PY="${ADCGO_PYTHON:-/home/leia/miniconda3/envs/adcgo/bin/python}"
ADCGO="${2:-adcgo}"

# 1. Dump integrals (FCIDUMP + sidecar + manifest) from the input file.
echo "#begin<dump>"
"$PY" "$here/dump_fcidump.py" --input "$input"
echo "#end<dump>"

# 2. Resolve output paths + ADC options from the same input file (reuse the parser).
eval "$(PYTHONPATH="$here" "$PY" - "$input" <<'PYEOF'
import shlex, sys
from adcgo_input import parse_input_file
from orbital_select import parse_index_list
c = parse_input_file(sys.argv[1])
print("FCIDUMP=" + shlex.quote(c.resolve(c.fcidump)))
print("SIDECAR=" + shlex.quote(c.resolve(c.sidecar) or ""))
print("ADC_ARGS=" + shlex.quote(c.adc_args))
sym = c.adc_sym.strip()
if sym.lower() in ("all", ""):
    print("SYMS=all")
else:
    print("SYMS=" + shlex.quote(" ".join(str(x) for x in parse_index_list(sym))))
PYEOF
)"

if [ -z "${ADC_ARGS// }" ]; then
    echo "no &adc section: FCIDUMP written, skipping the solve step."
    exit 0
fi

base="${FCIDUMP%.fcidump}"
mo_flag=()
[ -n "$SIDECAR" ] && mo_flag=(-mo "$SIDECAR")

# 3. Run adcgo. `all` -> one run over every irrep; otherwise loop the 1-based list.
echo "#begin<adc>"
if [ "$SYMS" = "all" ]; then
    # shellcheck disable=SC2086
    "$ADCGO" -fcidump "$FCIDUMP" "${mo_flag[@]}" $ADC_ARGS -sym all -out "${base}.adc.json"
    echo "wrote ${base}.adc.json"
else
    for s in $SYMS; do
        idx=$((s - 1))   # theADCcode 1-based irrep -> adcgo 0-based -sym
        # shellcheck disable=SC2086
        "$ADCGO" -fcidump "$FCIDUMP" "${mo_flag[@]}" $ADC_ARGS -sym "$idx" \
                 -out "${base}.sym${s}.json"
        echo "wrote ${base}.sym${s}.json  (irrep $s, -sym $idx)"
    done
fi
echo "#end<adc>"
