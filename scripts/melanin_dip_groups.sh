#!/bin/bash
# Regroup a solved melanin DIP document into theADCcode's &popana decay sites and emit the
# decay-channel stick spectrum. This is instant post-processing — it reads the per-atom
# populations already stored in melanin.dip.json (written by melanin_dip.sbatch with -mo) and
# reclassifies them; it does NOT re-solve. Run it on a login node after the DIP job finishes.
#
#   scripts/melanin_dip_groups.sh [INIT_SITE]        # default INIT_SITE=o1
#
# THE GROUPING is theADCcode's &popana, translated from AO ranges to atoms. In this DZ basis
# each heavy atom spans 10 AOs and each H spans 2, and pyscf orders AOs by atom in geometry
# order, so the &popana AO blocks map one-to-one onto atoms. The sidecar names atoms by
# element + 1-based order (O1..O4, N1,N2, C1..C18, H1..H10). The resulting sites:
#
#   &popana range                      -> atoms                 -> ADCgo site
#   o1  1..10 21..30                   -> O1,O3                 -> o1=O1,O3
#   o2  11..20 31..40                  -> O2,O4                 -> o2=O2,O4
#   n1  41..50                         -> N1                    -> n1=N1
#   n2  51..60                         -> N2                    -> n2=N2
#   rc1 61..70 81..90 ... 201..210     -> C1,C3,..,C15 (odd)    -> rc1=C1,C3,C5,C7,C9,C11,C13,C15
#   rc2 71..80 91..100 ... 211..220    -> C2,C4,..,C16 (even)   -> rc2=C2,C4,C6,C8,C10,C12,C14,C16
#   mc1 221..230                       -> C17                   -> mc1=C17
#   mc2 231..240                       -> C18                   -> mc2=C18
#   rh1 241..242 / rh2 243..244        -> H1 / H2               -> rh1=H1 / rh2=H2
#   nh1 245..246 / nh2 247..248        -> H3 / H4               -> nh1=H3 / nh2=H4
#   mh1 249..254                       -> H5,H6,H7              -> mh1=H5,H6,H7
#   mh2 255..260                       -> H8,H9,H10             -> mh2=H8,H9,H10
#
# NOTE: -group needs '=' syntax (it is a bool-style flag): -group=NAME=col1,col2.
# INIT_SITE is the initially core-ionized site that seeds the decay channels; it must be one
# of the site names above. Set it to match the physics of the run (default o1).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if command -v module >/dev/null 2>&1; then
    module load devel/cuda/13.2 || true   # adcgo-cuda is cgo-linked; needs the runtime libs to load
fi

ADCGO="${ADCGO:-$root/adcgo-cuda}"
WS="${ADCGO_WS:-/gpfs/bwfor/scratch/hd_hh323_o05i14/adcgo/melanin}"
INIT_SITE="${1:-${INIT_SITE:-o1}}"
doc="$WS/melanin.dip.json"
mo="$WS/melanin.mo.json"
out="$WS/melanin.dip.spectrum.json"

[ -f "$doc" ] || { echo "no DIP document at $doc; run melanin_dip.sbatch first" >&2; exit 1; }
[ -f "$mo" ]  || { echo "no MO sidecar at $mo; run dump_melanin.sbatch first" >&2; exit 1; }

groups=(
    -group=o1=O1,O3
    -group=o2=O2,O4
    -group=n1=N1
    -group=n2=N2
    -group=rc1=C1,C3,C5,C7,C9,C11,C13,C15
    -group=rc2=C2,C4,C6,C8,C10,C12,C14,C16
    -group=mc1=C17
    -group=mc2=C18
    -group=rh1=H1
    -group=rh2=H2
    -group=nh1=H3
    -group=nh2=H4
    -group=mh1=H5,H6,H7
    -group=mh2=H8,H9,H10
)

echo "regrouping $doc -> $out  (init site: $INIT_SITE)"
"$ADCGO" -convert "$doc" -dip -mo "$mo" -spectrum "${groups[@]}" -init-atom "$INIT_SITE" -out "$out"
echo "wrote $out"
