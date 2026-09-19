#!/bin/bash
#SBATCH --job-name=fano-table6
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=fano-table6-%j.log
#
# Fano-ADC decay widths for the atomic core vacancies of Table VI in
#   P. Kolorenc and V. Averbukh, J. Chem. Phys. 152, 214107 (2020).
#
# WHY THIS IS A BATCH JOB. The HELIX login node's user slice caps CPU at 4 cores
# (cpu.cfs_quota_us=400000 / cpu.cfs_period_us=100000), and Go 1.25+ reads the cgroup
# limit, so GOMAXPROCS is 4 there however much `nproc` claims. Every parallel.Chunks in
# the matrix-free appliers then runs 4-wide, and a mat-vec that should take under a second
# takes tens. Only the order-2 baselines are login-node work.
#
# Generate the FCIDUMPs first (needs the adcgo conda env with pyscf):
#   ws_allocate fano 30
#   FANO_OUT=$(ws_find fano)/fcidump python scripts/gen_fano_atoms.py \
#       --basis aug-cc-pV5Z --kbj 4 Ne Ar Kr
#   FANO_OUT=$(ws_find fano)/fcidump python scripts/gen_fano_atoms.py \
#       --basis aug-cc-pVQZ --kbj 4 Mg          # pyscf has no aug-cc-pV5Z for Mg
#
# ONE augmentation for every system, which is what the paper does: aug-cc-pV5Z plus 4s4p4d
# Kaufmann-Baumeister-Jungen Rydberg functions. Do NOT go back to --augment (even-tempered
# exponents chosen per ion to cover the outgoing electron's kinetic window). That set is
# LINEARLY DEPENDENT with the parent basis -- its exponents land on the valence ones -- and
# pyscf's symmetry-adapted orthogonalisation then silently discards orbitals: Ne loses 1 of
# 163 and Ar 4 of 167 at --augment 4, Ne loses 7 of 181 at --augment 6, the smallest overlap
# eigenvalue falls from 1.7e-3 to as low as 6e-8, and the RHF energy RISES on adding
# functions (Ne -128.546786 -> -128.546700 Eh), which is only possible if basis vectors are
# being thrown away. What gets thrown away is the virtual spectrum Stieltjes samples. With
# KBJ nothing is dropped and the energy goes down, for every ion.
#
# The basis has to bracket E_Phi. Imaging evaluates Gamma at E_Phi, and a 2h1p state sits
# at eps_a - eps_k - eps_l, so the VIRTUAL SPECTRUM must reach that high and be dense
# there. Ne 1s needs eps_a ~ 30 Eh: aug-cc-pVTZ tops out at 14.6 Eh and cannot describe
# the decay at all, aug-cc-pVQZ reaches 68.9 Eh, and the --augment 4 even-tempered s/p/d
# shells are what put enough LEVELS in the 25-40 Eh window to image with. Watch the
# "carry coupling" line in the log: below ~40 coupled channels the width is not converged
# whatever else is right.
set -euo pipefail

ROOT="${ADCGO_ROOT:-$SLURM_SUBMIT_DIR}"

# The FCIDUMPs live in a Helix WORKSPACE, not in scratch and not in the repo. Scratch is
# purged without warning — a previous run of this comparison lost 5 GB of generated
# integrals and every result JSON that way — and the dumps are far too large to commit.
# ws_allocate provisions the path; a plain mkdir under /gpfs/bwfor/... fails, since those
# roots are root-owned. See scripts/adcgo_ws.sh for the same discipline.
if [ -z "${FANO_DUMPS:-}" ]; then
  WS="$(ws_find fano 2>/dev/null || true)"
  [ -n "$WS" ] || { echo "no 'fano' workspace: run  ws_allocate fano 30" >&2; exit 1; }
  FANO_DUMPS="$WS/fcidump"
fi
DUMPS="$FANO_DUMPS"
OUT="${FANO_OUT:-${WS:-$ROOT}/out}"
mkdir -p "$OUT"
cd "$ROOT"

export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-64}"
go build -o "$OUT/adcgo" ./cmd/adcgo

# THE SUPPLEMENTARY MATERIAL'S RECIPE (../ADC/ADC22_sup.pdf). Every field below is taken
# from it, and all four atoms differ from a plain aug-cc-pV5Z setup in four ways at once:
# a CORE-VALENCE parent basis, UNCONTRACTED, a per-ion VIRTUAL ENERGY CUTOFF, and a frozen
# core. HF and integrals were done in MOLCAS there; bases come from the Basis Set Exchange,
# which is also where this script gets them (pyscf's bundled tables have neither
# aug-cc-pCV5Z nor cc-pwCV5Z-PP, and Kr needs the Stuttgart-Koeln MCDHF RSC ECP for [Ne]).
#
#   ion         parent basis                augmentation      frozen   virtuals
#   Ne+(1s-1)   uncontracted aug-cc-pCV5Z   10s10p8d even-t.  -        < 460 Ha
#   Mg+(2s-1)   uncontracted cc-pCV5Z       10s10p8d KBJ      1s       <  80 Ha
#   Ar+(2p-1)   uncontracted aug-cc-pwCV5Z  4s3p even-t.      1s       <  30 Ha
#   Kr+(3d-1)   cc-pwCV5Z-PP (+[Ne] ECP)    5s5p5d2f KBJ      3s,3p    <  11 Ha
#                 with H- and I-type functions removed
#
# The augmentation EXPONENTS are the one thing the supplement does not give. It says the
# choice between even-tempered and KBJ "is based on the quality of convergence of the
# Stieltjes imaging procedure", so they are a tuned input there too; ALPHA0 below is scanned
# and the value with the smallest Stieltjes order spread is the one to keep.
#
# Vacancy indices and Q/P rules are 0-based occupied indices AFTER freezing:
#   Ne  no freeze:        1s=0, 2s=1, 2p=2,3,4                     vac 0
#   Mg  freeze 1s:        2s=0, 2p=1,2,3, 3s=4                     vac 0
#   Ar  freeze 1s:        2s=0, 2p=1,2,3, 3s=4, 3p=5,6,7           vac 1
#   Kr  ECP + freeze 3s,3p: 3d=0..4, 4s=5, 4p=6,7,8                vac 0
#
# Mg and Kr need -fano-qp, a PER-EXCITATION-CLASS rule, because the supplement states their
# partitions per class and -fano-q cannot express that. Mg's keeps the CLOSED 2p^-2 channel
# out of P (it lies above the 2s vacancy) by naming only 3s-hole configurations as continuum,
# and puts 3h2p 2p^-1 3s^-2 in P alone, which the paper says it did deliberately and blames
# for Mg being its worst case. Kr's adds the 4s^-2 4p^-1 shake-up family to Q.
# KRYLOV is a LIST: the pseudo-continuum is a Krylov approximation to P, and at ADC(2,2)
# only ~10% of its states fall below the energy ceiling, so a single Krylov size cannot be
# told apart from a converged one. Measured on Mg ADC(2,2)_m, 864 vectors gave 219 meV and
# 3840 gave 768 meV, with the sum-rule residual barely moving (63% -> 58%) because that
# residual is dominated by the deliberate energy cut rather than by Krylov incompleteness.
KRYLOV="${FANO_KRYLOV:-120}"

GEN="$ROOT/scripts/gen_fano_atoms.py"
PYBIN="${FANO_PYTHON:-$HOME/miniconda3/envs/adcgo/bin/python}"
A0="${FANO_ALPHA0:-0.05}"
RAT="${FANO_RATIO:-2.5}"

gen() { FANO_OUT="$DUMPS" OMP_NUM_THREADS="$GOMAXPROCS" "$PYBIN" "$GEN" "$@"; }

if [ "${FANO_GENERATE:-1}" = 1 ]; then
  gen --bse --basis aug-cc-pCV5Z  --uncontract --aug-shells 10,10,8 \
      --aug-alpha0 "$A0" --aug-ratio "$RAT" --virt-max 460 Ne
  gen --bse --basis cc-pCV5Z      --uncontract --aug-shells 10,10,8 --kbj 10 \
      --freeze 1 --virt-max 80 Mg
  gen --bse --basis aug-cc-pwCV5Z --uncontract --aug-shells 4,3 \
      --aug-alpha0 "$A0" --aug-ratio "$RAT" --freeze 1 --virt-max 30 Ar
  gen --bse --basis cc-pwcv5z-pp  --aug-shells 5,5,5,2 --kbj 5 --drop-l 5 \
      --freeze 4 --virt-max 11 --vacancy 4 Kr
fi

# atom : vacancy : dump basename : orders : Q/P spec
SYSTEMS="${FANO_SYSTEMS:-\
ne:0:ne_augccpcv5z_unc_et10s10p8d_v460:2,22:q:0:1|\
mg:0:mg_ccpcv5z_unc_kbj10s10p8d_v80_fc1:2,22:q:0:1;p:2/4:1;p:3/4:2|\
ar:1:ar_augccpwcv5z_unc_et4s3p_v30_fc1:2,22:q:0-3:1|\
kr:0:kr_ccpwcv5zpp_kbj5s5p5d2f_v11_fc4:2:q:0-4:1;q:3/5:2:2&3/6-8:1}"

# FANO_ONLY restricts the sweep to named atoms, e.g. FANO_ONLY="ne mg kr". Use THIS and
# not FANO_SYSTEMS to split the sweep across allocations: sbatch --export splits its value
# on COMMAS, so a system spec like "ne:0:tag:2,22:0" is shredded into separate environment
# assignments and the Q set silently becomes "2". That failure is not visible at submission
# time -- it surfaces as "Q holds N configurations but none of the main class" after the
# integrals have been read.
# Records are separated by '|' and fields by ':', with the Q/P spec LAST because it
# contains both ':' and ';' itself. `read` puts the unsplit remainder in its final
# variable, which is what makes that work.
printf '%s\n' "$SYSTEMS" | tr '|' '\n' | while IFS= read -r sys; do
  [ -n "$sys" ] || continue
  IFS=: read -r atom vac tag orders qp <<EOFSYS
$sys
EOFSYS
  if [ -n "${FANO_ONLY:-}" ]; then
    case " $FANO_ONLY " in *" $atom "*) ;; *) continue;; esac
  fi
  dump="$DUMPS/${tag}.fcidump"
  [ -r "$dump" ] || { echo "missing $dump — run scripts/gen_fano_atoms.py first" >&2; continue; }

  # Order 2 is the Fano-ADC(2)x baseline (Table VI column 4); order 22 m/x/f are the
  # ADC(2,2) variants (columns 5-10), whose relative differences from the baseline are a
  # sharper check than any single width.
  # The ADC(2)x baseline first: its E_Phi is the common energy the ADC(2,2) variants are
  # then ALSO evaluated at, so that a scheme's effect on the coupling density can be
  # separated from its effect on where that density is sampled. Both are reported.
  #
  # FANO_SCHEMES overrides the list, which is how the sweep is split across two
  # allocations: the cheap half (order 2 and ADC(2,2)_m, whose 3h2p/3h2p block is a
  # resident DIAGONAL) asks for a few hours and backfills, while ADC(2,2)_x/f recompute
  # the full first-order 3h2p/3h2p block on every mat-vec and need the long one. Point the
  # two at different FANO_OUT directories so their labels cannot collide.
  base_ephi=""
  schemes="${FANO_SCHEMES:-2:f}"
  if [ -z "${FANO_SCHEMES:-}" ]; then
    case ",$orders," in *,22,*) schemes="2:f 22:m 22:x 22:f";; esac
  else
    # Drop any order the system is not configured for (Kr is order 2 only).
    keep=""
    for scheme in $schemes; do
      case ",$orders," in *,"${scheme%%:*}",*) keep="$keep $scheme";; esac
    done
    schemes="$keep"
  fi
  for scheme in $schemes; do
    order="${scheme%%:*}"; variant="${scheme##*:}"
   for nblk in $KRYLOV; do
    label="${atom}_o${order}${variant}_k${nblk}"
    echo "=== $label ==="
    # -matfree on: the ADC(2,2) 3h2p/3h2p block is n3^2 and cannot be stored.
    # -solver lanczos drives the PMP pseudo-continuum; the QMQ half uses Davidson by
    # default, which is what a few lowest roots want.
    "$OUT/adcgo" -fcidump "$dump" -sip -order "$order" -adc22 "$variant" \
        -fano -fano-init "$vac" -fano-qp "$qp" -sym all \
        -solver lanczos -matfree on -maxmem 8 \
        -fano-qroots 4 -fano-block 64 -fano-blocks "$nblk" \
        -fano-wmin -1 \
        -sigma off -out "$OUT/$label.json" 2>&1 | tee "$OUT/$label.log" || true

    # Record the baseline E_Phi, then re-evaluate every ADC(2,2) variant there as well.
    if [ "$order" = 2 ]; then
      base_ephi=$(python3 -c "import json;print(json.load(open('$OUT/$label.json'))['e_phi_ev'])" 2>/dev/null || true)
    elif [ -n "$base_ephi" ]; then
      "$OUT/adcgo" -fcidump "$dump" -sip -order "$order" -adc22 "$variant" \
          -fano -fano-init "$vac" -fano-qp "$qp" -sym all \
          -solver lanczos -matfree on -maxmem 8 \
          -fano-qroots 4 -fano-block 64 -fano-blocks "$nblk" \
          -fano-wmin -1 -fano-at "$base_ephi" \
          -sigma off -out "$OUT/${label}_atbase.json" 2>&1 \
          | tee "$OUT/${label}_atbase.log" || true
    fi

    # -sigma off deliberately: ADC(2)x and ADC(2,2) share the same 0,2 1h/1h block, so the
    # static self-energy shifts both E_Phi values together and cancels from the Delta ratio
    # of Eq. (43). Leaving it off changes one variable at a time and saves the Sigma(inf)
    # resummation on a 170-orbital space.
    #
    # -fano-wmin -1 disables the reference's weight cut. gamma > -fano-gmin is the same
    # test done directly and exactly, and at ADC(2,2) the weight cut is actively harmful:
    # a 3h2p-dominated state has little 2h1p character by construction, so cutting on it
    # discards the second-order decay channels the scheme exists to describe.
   done
  done
done

echo "=== summary ==="
for f in "$OUT"/*.json; do
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
v = d.get("variant") or "x"
print(f"{sys.argv[1].split('/')[-1]:32s} order {d['order']:2}{v}  "
      f"E_Phi {d['e_phi_ev']:9.3f} eV  Gamma {d['width_mev']:9.3f} +/- {d['width_sigma_mev']:7.3f} meV  "
      f"tau {d['lifetime_fs']:8.4f} fs  channels {d['coupled_channels']:5}  "
      f"states {d['pseudo_states']:5}  max order {d['stieltjes_max_order']}")
PY
done
