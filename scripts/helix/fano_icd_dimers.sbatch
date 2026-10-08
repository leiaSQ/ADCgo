#!/bin/bash
#SBATCH --job-name=fano-icd
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=fano-icd-%j.log
#
# Interatomic / intermolecular decay widths against the literature — the ICD counterpart of
# fano_table6.sbatch, which checks only intra-atomic Auger.
#
#   ICD_CASE=wat2   (H2O)2, 2a1^-1 of the donor and of the acceptor. Reference: Richter et
#                   al., Nat. Commun. 9, 4988 (2018), Fano-CI: tau = 72 fs (donor), 131 fs
#                   (acceptor), cc-pVDZ + 5s5p5d KBJ on each O, equilibrium dimer. Run as
#                   Fano-CI (-khci 1 -khci-maxclass 2: 1h | 2h1p, the same method), then
#                   Fano-ADC(2)x and Fano-ADC(2,2)f.
#   ICD_CASE=ne2x   Ne2 (2s^-1) Gamma(R), 2sigma_g and 2sigma_u, Fano-ADC(2)x, over ICD_R.
#   ICD_CASE=ne2f   the same at Fano-ADC(2,2)f (expensive: 3h2p over ~250 orbitals).
#                   Reference: Kolorenc & Averbukh, J. Chem. Phys. 152, 214107 (2020) Fig. 1
#                   and its supplement: cc-pVTZ + 5s5p5d KBJ on the atoms + 4s4p4d KBJ at the
#                   midpoint and at +-3R/2, 1s frozen. Values are only in the figure, so the
#                   checks are its stated shape: Gamma ~ R^-6 at large R, Gamma_u/Gamma_g ~ 2
#                   at R_e, ADC(2)x / ADC(2,2)f ~ 1.4 at every R; and tau_exp = 150 +- 50 fs
#                   (Schnorr et al., PRL 111, 093402 (2013)), i.e. ~4.4 meV, at R_e.
#
# Q/P is scheme A here (the paper's Ne2 used scheme B, adapted 2h1p ISs). Two partitions
# are run for the water dimer, because the difference between them is the finding:
#   iv       Q = every 1h + every configuration with an inner-valence hole (e<-1.0);
#            P = everything else, ONE-SITE 2h1p included. Richter's Fano-CI space.
#   twosite  as iv, plus every configuration with all its holes on ONE water in Q, so P
#            holds only holes on two molecules: the ICD/ETMD continuum proper.
#   x        as iv, with the one-site outer-valence configurations EXCLUDED from both Q and P
#            (-fano-qp x: clauses): Richter's Fano-CI spaces. The default variant; the first
#            two fail (job 15225236: 0.3-2 eV against ~9 meV) and are kept for the record.
# At CI level the one-site 2h1p (H2O+ shake-up satellites, H2O2+ + e) lie above E_Phi and
# the two agree. At ADC(2)x the satellites drop to the 2a1 energy: in P they couple to
# |Phi> as a fake continuum (an isolated water's 2a1 gets an 0.8 eV "Auger" width), in Q
# they fragment the 2a1 line, so twosite is also run with |Phi> = the QMQ root of LARGEST
# 2a1 weight (-fano-phi-holes) besides the default lowest root above -fano-qmin, and each
# document's c4_audit lists the fragments within 0.5 eV.
#
# Ne2 deck, two settings forced by the basis (measured at R = 2.6 A, 2026-10-08):
#   symmetry C2v   pyscf labels each ghost centre separately (X1..X5), so mirror-image ghosts
#                  look inequivalent and it detects C_inf_v, which ADCgo cannot use; C2v is
#                  the largest D2h subgroup left, and puts 2sigma_g and 2sigma_u both in A1.
#   conv_tol 1e-10, conv_tol_grad 1e-6
#                  the atom-centred KBJ sets, the cc-pVTZ diffuse shells and the midpoint
#                  ghost are nearly dependent (smallest overlap eigenvalue 4e-10; pyscf drops
#                  those below 1e-6). The energy is stable to 1e-10 Eh from the first attempt,
#                  but the default 1e-12 / 1e-9 thresholds are below that noise and never met.
#
# Ne2 runs the iv rule only: canonical Ne2 orbitals are g/u pairs over both atoms, so a
# one-site rule cannot be written with them, and the Ne+(2p^-2 nl) satellites near the 2s
# line stay in P. Their contamination is atomic and does not fall as R^-6, so the large-R
# behaviour of ne2x is the diagnostic.
# KBJ exponents are kbj.py's continuum-like set (zeta = 1; Coccia et al. Table I), which is
# what both references call "continuum-like KBJ Gaussian functions". Atom-centred sets are
# ghost sites at the atom positions.
#
# Submit (pre-build the binary so a job that waits in the queue runs the code that was
# reviewed, not whatever the tree holds when it starts):
#   ws_allocate icd 30
#   go build -o "$(ws_find icd)/adcgo-icd" ./cmd/adcgo
#   ICD_BIN=$(ws_find icd)/adcgo-icd sbatch --export=ALL,ICD_CASE=wat2 scripts/helix/fano_icd_dimers.sbatch
set -euo pipefail

ROOT="${ADCGO_ROOT:-$SLURM_SUBMIT_DIR}"
WS="$(ws_find icd 2>/dev/null || true)"
[ -n "$WS" ] || { echo "no 'icd' workspace: run  ws_allocate icd 30" >&2; exit 1; }
CASE="${ICD_CASE:?set ICD_CASE=wat2|wat2b|wat2frag|ne2x|ne2f}"
PYBIN="${ICD_PYTHON:-$HOME/miniconda3/envs/adcgo/bin/python}"
DUMP="$ROOT/scripts/fcidump/dump_fcidump.py"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-64}"
export OMP_NUM_THREADS="$GOMAXPROCS"
BIN="${ICD_BIN:-}"
if [ -z "$BIN" ]; then
  BIN="$WS/adcgo-icd-${SLURM_JOB_ID:-local}"
  (cd "$ROOT" && go build -o "$BIN" ./cmd/adcgo)
fi
QP_IV='q:1/e<0:1;q:e<-1.0:1'
BLOCKS="${ICD_BLOCKS:-120}"

# width LABEL QP ARGS... — one -fano run, JSON + log beside each other, failures kept.
width() {
  local label=$1 qp=$2; shift 2
  echo "=== $label ==="
  "$BIN" "$@" -fano -fano-qp "$qp" -fano-blocks "$BLOCKS" -fano-wmin -1 \
      -out "$label.json" 2>&1 | tee "$label.log" || true
}

summary() {
  for f in "$@"; do
    [ -s "$f" ] || { echo "$f: no result"; continue; }
    python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
site = d.get("vacancy_site", "")
pop = d.get("vacancy_site_population", 0)
print(f"{sys.argv[1]:34s} {d['family']:4s} o{d['order']}{d.get('variant','')} vac {d['vacancy']} "
      f"{site}{'(%.2f)' % pop if site else ''}  E_Phi {d['e_phi_ev']:8.3f} eV  "
      f"Gamma {d['width_mev']:9.3f} +/- {d['width_sigma_mev']:7.3f} meV  tau {d['lifetime_fs']:9.2f} fs  "
      f"channels {d['coupled_channels']}  phi weight {d.get('phi_weight', 0):.3f}  "
      f"sum-rule residual {d.get('sum_rule_residual', 0):.3f}")
frag = [f"{a['energy_ev']:.3f} eV ({a.get('target_weight', 0):.2f})" for a in d.get("c4_audit", []) or []]
if frag:
    print("    Q states within 0.5 eV (target weight):", ", ".join(frag))
for p in d.get("partial_widths", []) or []:
    print(f"    {p.get('channel', p.get('name','?')):16s} {p.get('width_mev', 0):9.3f} meV")
PY
  done
}

case "$CASE" in
wat2)
  DIR="$WS/wat2"; mkdir -p "$DIR"; cd "$DIR"
  if [ ! -s wat2_vdz_kbj5.fcidump ]; then
    cat > wat2.xyz <<'EOF'
6
water dimer, S22 (CCSD(T)/cc-pVQZ), donor = atoms 1-3, acceptor = 4-6
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
EOF
    cat > wat2_vdz_kbj5.in <<'EOF'
&geometry
  file wat2.xyz
&basis
  name cc-pvdz
&scf
  charge 0
  symmetry auto
  max_cycle 200
&ghost_sites
  KO_D  -1.551007  -0.114520   0.000000  kbj:2:1-5
  KO_A   1.350625   0.111469   0.000000  kbj:2:1-5
&active
  frozen-core 2
&output
  fcidump  wat2_vdz_kbj5.fcidump
  sidecar  wat2_vdz_kbj5.mo.json
  manifest wat2_vdz_kbj5.ref.json
EOF
    "$PYBIN" "$DUMP" --input wat2_vdz_kbj5.in
  fi
  # The canonical sidecar names the ghost centres X1, X2; each belongs to its water.
  SITES=(-group=D=O1,H1,H2,X1 -group=A=O2,H3,H4,X2)
  COMMON=(-fcidump wat2_vdz_kbj5.fcidump -mo wat2_vdz_kbj5.mo.json "${SITES[@]}")
  QP_2S="$QP_IV;q:2/@D:2;q:2/@A:2;q:3/@D:3;q:3/@A:3"
  # x: one-site outer-valence configurations excluded from both Q and P (Richter's Fano-CI
  # spaces). The e<-1.0:0:0 term keeps each water's own 2a1-hole configurations in Q.
  QP_X="$QP_IV;x:2/@D:2&2/e<-1.0:0:0;x:2/@A:2&2/e<-1.0:0:0;x:3/@D:3&3/e<-1.0:0:0;x:3/@A:3&3/e<-1.0:0:0"
  FCI=(-khci 1 -khci-maxclass 2 -sym none)
  O2X=(-sip -order 2 -sym all -solver lanczos)
  O22F=(-sip -order 22 -adc22 f -sym all -solver lanczos -matfree on -maxmem 8)
  # ICD_WAT2_METHODS / ICD_WAT2_VARIANTS select the matrix; the complete partitions (iv, 2s,
  # 2smax) are kept for the record of why x is needed.
  out=()
  for meth in ${ICD_WAT2_METHODS:-fci o2x}; do
    case $meth in fci) M=("${FCI[@]}");; o2x) M=("${O2X[@]}");; o22f) M=("${O22F[@]}");; esac
    for v in D A; do
      init=(-fano-init "@$v.0")
      [ $meth = fci ] || init+=(-init-atom "$v")
      for var in ${ICD_WAT2_VARIANTS:-x}; do
        case $var in
        iv)    width "wat2_${meth}_iv_$v" "$QP_IV" "${COMMON[@]}" "${M[@]}" "${init[@]}";;
        2s)    width "wat2_${meth}_2s_$v" "$QP_2S" "${COMMON[@]}" "${M[@]}" "${init[@]}";;
        2smax) width "wat2_${meth}_2smax_$v" "$QP_2S" "${COMMON[@]}" "${M[@]}" "${init[@]}" -fano-phi-holes "@$v.0";;
        x)     width "wat2_${meth}_x_$v" "$QP_X" "${COMMON[@]}" "${M[@]}" "${init[@]}";;
        *)     echo "unknown variant $var" >&2; exit 2;;
        esac
        out+=("wat2_${meth}_${var}_$v.json")
      done
    done
  done
  echo "=== summary (reference Fano-CI: D 72 fs, A 131 fs) ==="
  summary "${out[@]}"
  ;;
wat2b)
  # Scheme B (-fano-scheme b): the 2h1p configurations of each particle orbital rotated onto
  # adapted states, classified by their two-hole character over water-localized occupied
  # orbitals (inner -> Q, one-site -> excluded, two-site -> P). It removes the leak of the
  # x partition on canonical orbitals that are shared between the waters (cause 1 of the
  # 25x overestimate). 3h2p keeps the x hole rule. ADC(2,2)f is the target method.
  DIR="$WS/wat2"; cd "$DIR"
  [ -s wat2_vdz_kbj5.fcidump ] || { echo "run ICD_CASE=wat2 once first (canonical fixture)" >&2; exit 1; }
  SITES=(-group=D=O1,H1,H2,X1 -group=A=O2,H3,H4,X2)
  QP_X="$QP_IV;x:2/@D:2&2/e<-1.0:0:0;x:2/@A:2&2/e<-1.0:0:0;x:3/@D:3&3/e<-1.0:0:0;x:3/@A:3&3/e<-1.0:0:0"
  B=(-fcidump wat2_vdz_kbj5.fcidump -mo wat2_vdz_kbj5.mo.json "${SITES[@]}" -sip -sym all
     -solver lanczos -fano-scheme b)
  out=()
  for meth in ${ICD_WAT2_METHODS:-o2x o22f}; do
    case $meth in
    o2x)  M=(-order 2);;
    o22f) M=(-order 22 -adc22 f -matfree on -maxmem 8);;
    esac
    # free: P = two-site adapted states with a FREE particle (the L2 analogue; bound D+A*
    # states excluded). any: every two-site state (the paper's rule; the L1 analogue).
    for part in ${ICD_B_PARTICLE:-free any}; do
      for v in D A; do
        lab="wat2_${meth}_b${part}_$v"
        width "$lab" "$QP_X" "${B[@]}" "${M[@]}" -fano-b-particle "$part" -fano-init "@$v.0" -init-atom "$v"
        width "${lab}_gauss" "$QP_X" "${B[@]}" "${M[@]}" -fano-b-particle "$part" -fano-init "@$v.0" \
            -init-atom "$v" -fano-engine gauss
        out+=("$lab.json" "${lab}_gauss.json")
      done
    done
  done
  echo "=== summary (reference Fano-CI: D 72 fs, A 131 fs) ==="
  summary "${out[@]}"
  ;;
wat2frag)
  # Fano-CI over FRAGMENT-localized orbitals (&orbitals scheme fragment): occupied orbitals
  # owned by one water each (1s, 2a1, 1b2, 3a1, 1b1 per water), compact virtuals per water,
  # free virtuals (the KBJ continuum) unowned. CI is invariant to these rotations; ADC is
  # not, which is why this diagnostic is CI. It separates the two suspected causes of the
  # ~25x too large x-partition widths on canonical orbitals:
  #   L1  one-site configurations excluded, holes localized: P = any two-site 2h1p.
  #       Tests cause 1 (canonical outer-valence orbitals shared between the waters).
  #   L2  L1 plus two-site configurations with a COMPACT particle excluded: P = each water
  #       +1 and one free electron. Tests cause 2 (bound D+A* states in P near E_Phi).
  # Nothing is frozen in a rotated basis, so the inner set per water is its 1s and 2a1.
  DIR="$WS/wat2"; mkdir -p "$DIR"; cd "$DIR"
  [ -s wat2.xyz ] || { echo "run ICD_CASE=wat2 once first (it writes wat2.xyz)" >&2; exit 1; }
  if [ ! -s wat2_vdz_kbj5_frag.fcidump ]; then
    cat > wat2_vdz_kbj5_frag.in <<'EOF'
&geometry
  file wat2.xyz
&basis
  name cc-pvdz
&scf
  charge 0
  symmetry off
  max_cycle 200
&ghost_sites
  KO_D  -1.551007  -0.114520   0.000000  kbj:2:1-5
  KO_A   1.350625   0.111469   0.000000  kbj:2:1-5
&orbitals
  scheme fragment
  fragments D:1-3 A:4-6
  compact_basis cc-pvdz
&output
  fcidump  wat2_vdz_kbj5_frag.fcidump
  sidecar  wat2_vdz_kbj5_frag.mo.json
  manifest wat2_vdz_kbj5_frag.ref.json
EOF
    "$PYBIN" "$DUMP" --input wat2_vdz_kbj5_frag.in
  fi
  IN='@D.0,@D.1,@A.0,@A.1'
  QP_L1="q:1/@D,@A:1;q:$IN:1;x:2/@D:2&2/$IN:0:0;x:2/@A:2&2/$IN:0:0"
  QP_L2="$QP_L1;x:2/$IN:0:0&2/free=0"
  F=(-fcidump wat2_vdz_kbj5_frag.fcidump -mo wat2_vdz_kbj5_frag.mo.json
     -group=D=O1,H1,H2 -group=A=O2,H3,H4 -khci 1 -khci-maxclass 2 -sym none)
  out=()
  for v in D A; do
    for L in L1 L2; do
      qp=$([ $L = L1 ] && echo "$QP_L1" || echo "$QP_L2")
      width "wat2_frag_${L}_$v" "$qp" "${F[@]}" -fano-init "@$v.1"
      width "wat2_frag_${L}_${v}_gauss" "$qp" "${F[@]}" -fano-init "@$v.1" -fano-engine gauss
      out+=("wat2_frag_${L}_$v.json" "wat2_frag_${L}_${v}_gauss.json")
    done
  done
  echo "=== summary (reference Fano-CI: D 72 fs, A 131 fs) ==="
  summary "${out[@]}"
  ;;
ne2x|ne2f)
  DIR="$WS/ne2"; mkdir -p "$DIR"; cd "$DIR"
  RS="${ICD_R:-2.6 3.1 3.5 4.0 5.0 6.0}"
  [ "$CASE" = ne2f ] && RS="${ICD_R:-3.1 6.0}"
  files=()
  for R in $RS; do
    tag="ne2_R${R}"
    if [ ! -s "$tag.fcidump" ]; then
      h=$(python3 -c "print($R/2)"); g=$(python3 -c "print(1.5*$R)")
      cat > "$tag.xyz" <<EOF
2
Ne2 at R = $R Angstrom, along z
Ne 0.0 0.0 -$h
Ne 0.0 0.0  $h
EOF
      cat > "$tag.in" <<EOF
&geometry
  file $tag.xyz
&basis
  name cc-pvtz
&scf
  charge 0
  symmetry C2v
  max_cycle 200
  conv_tol 1e-10
  conv_tol_grad 1e-6
&ghost_sites
  KA1  0.0 0.0 -$h  kbj:2:1-5
  KA2  0.0 0.0  $h  kbj:2:1-5
  KMID 0.0 0.0  0.0 kbj:2:1-4
  KL   0.0 0.0 -$g  kbj:2:1-4
  KR   0.0 0.0  $g  kbj:2:1-4
&active
  frozen-core 2
&output
  fcidump  $tag.fcidump
  sidecar  $tag.mo.json
  manifest $tag.ref.json
EOF
      # One separation failing must not cost the scan.
      "$PYBIN" "$DUMP" --input "$tag.in" || { echo "dump failed at R = $R; skipped" >&2; continue; }
    fi
    # Occupied after freezing the 1s pair: 0 = 2sigma_g, 1 = 2sigma_u (the energy window
    # e<-1.0 is exactly these two). Both are A1 here: see the &scf notes above.
    for v in 0 1; do
      sym=$([ $v = 0 ] && echo g || echo u)
      if [ "$CASE" = ne2x ]; then
        width "${tag}_o2x_$sym" "$QP_IV" -fcidump "$tag.fcidump" -sip -order 2 -sym all \
            -solver lanczos -fano-init $v
        files+=("${tag}_o2x_$sym.json")
      else
        width "${tag}_o22f_$sym" "$QP_IV" -fcidump "$tag.fcidump" -sip -order 22 -adc22 f -sym all \
            -solver lanczos -matfree on -maxmem 8 -fano-init $v
        files+=("${tag}_o22f_$sym.json")
      fi
    done
  done
  echo "=== summary ==="
  summary "${files[@]}"
  ;;
*)
  echo "unknown ICD_CASE=$CASE (want wat2, wat2b, wat2frag, ne2x or ne2f)" >&2; exit 2;;
esac
