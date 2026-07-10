#!/usr/bin/env python
"""Generate the M4 external-validation reference: reproduce theADCcode's h2o DIP
run (../ADCanalysis/examples/DIP_h2o) in pyscf so ADCgo can be compared to
adcdip{1..4}.out on *matched* integrals.

The reference is a GAMESS-UK RHF in the "DZP + Diffuse (Dunning)" basis (spelled
out in _basisset.dat), at the geometry in _zmatrix.dat, with active space "2 to
30" (freeze the O 1s core MO, drop the top virtual -> 29 active orbitals, 8 active
electrons). Its total SCF energy is -76.0498071428 Ha; we gate on that number to
prove the basis + geometry are transcribed correctly before trusting the FCIDUMP.

Run with the `adcgo` conda env:
    /home/leia/miniconda3/envs/adcgo/bin/python scripts/gen_ref_fcidump.py

With --sidecar-only, only the dipole/geometry keys of the existing sidecar are
refreshed; the matched FCIDUMP that the DIP and ADC(4) gates compare against is left
alone.
"""
import json
import os
import sys

import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.tools import fcidump

import fcidump_common
from gamess_orbsym import gamess_orbsym

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(os.path.dirname(HERE), "testdata")
os.makedirs(TESTDATA, exist_ok=True)

SIDECAR_ONLY = "--sidecar-only" in sys.argv[1:]
MO_PATH = os.path.join(TESTDATA, "h2o_dzp.mo.json")

# --- reference constants (from ../ADCanalysis/examples/DIP_h2o) ---------------
REF_SCF_ENERGY = -76.0498071428  # gamess_scf.out "total energy"
NCAS, NELECAS = 29, 8            # active space "2 to 30": ncore=1, drop top virtual

# DZP + Diffuse (Dunning) basis, transcribed from _basisset.dat. The .dat lists
# each primitive as "coeff  exponent"; pyscf/NWChem wants "exponent  coeff", so
# the columns are swapped here. GAMESS-UK uses cartesian GTOs, so cart=True below
# reproduces the 31-function basis (spherical d would give 30).
H_BASIS = gto.basis.parse("""
H    S
     19.24060000   0.03282800
      2.89920000   0.23120800
      0.65340000   0.81723800
H    S
      0.17760000   1.00000000
H    P
      1.00000000   1.00000000
H    S
      0.04827300   1.00000000
""")
O_BASIS = gto.basis.parse("""
O    S
   7816.54000000   0.00203100
   1175.82000000   0.01543600
    273.18800000   0.07377100
     81.16960000   0.24760600
     27.18360000   0.61183200
      3.41360000   0.24120500
O    S
      9.53220000   1.00000000
O    S
      0.93980000   1.00000000
O    S
      0.28460000   1.00000000
O    P
     35.18320000   0.01958000
      7.90400000   0.12418900
      2.30510000   0.39472700
      0.71710000   0.62737500
O    P
      0.21370000   1.00000000
O    D
      0.85000000   1.00000000
O    P
      0.05900000   1.00000000
""")

# Geometry from _zmatrix.dat: O-H 0.9440686 A, HOH 107.0715130 deg. Placed with
# the C2 axis along z (O at origin), molecule in the yz-plane, so pyscf detects
# C2v and the FCIDUMP carries real ORBSYM.
r, half = 0.9440686, np.deg2rad(107.0715130) / 2
sy, sz = r * np.sin(half), r * np.cos(half)
mol = gto.M(
    atom=[["O", (0.0, 0.0, 0.0)],
          ["H", (0.0, sy, -sz)],
          ["H", (0.0, -sy, -sz)]],
    basis={"H": H_BASIS, "O": O_BASIS},
    cart=True,
    symmetry=True,
    unit="Angstrom",
)

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-9
mf.run()

# Gate: the transcribed basis + geometry must reproduce the reference SCF energy.
if abs(mf.e_tot - REF_SCF_ENERGY) > 1e-4:
    raise SystemExit(
        f"SCF gate FAILED: E(SCF)={mf.e_tot:.10f} Ha, reference "
        f"{REF_SCF_ENERGY:.10f} Ha (|d|={abs(mf.e_tot - REF_SCF_ENERGY):.2e} > 1e-4). "
        "Check the basis transcription / geometry / cart setting.")
print(f"SCF gate OK: E(SCF)={mf.e_tot:.10f} Ha  (ref {REF_SCF_ENERGY:.10f}, "
      f"nAO={mol.nao_cart()})")

if SIDECAR_ONLY:
    doc, err = fcidump_common.augment_sidecar(MO_PATH, mol, dm=mf.make_rdm1())
    print(f"AO-basis gate OK: stored overlap matches this mol to {err:.2e}")
    print(f"augmented {MO_PATH}  dip_origin={doc['dip_origin']} scf_dip={doc['scf_dip']}")
    raise SystemExit(0)

# Frozen-core active space (MOs 2..30). CASCI defaults ncore=(10-8)/2=1 and, with
# ncas=29, leaves MO 31 out -> exactly "2 to 30".
mc = mcscf.CASCI(mf, NCAS, NELECAS)
ncore = mc.ncore
active = slice(ncore, ncore + NCAS)
mo_act = mf.mo_coeff[:, active]

h1e, ecore = mc.get_h1eff()
h2e = mc.get_h2eff(mo_act)

# ORBSYM: 1-based GAMESS-UK / theADCcode irrep labels for the active MOs (C2v:
# A1->1, A2->2, B1->3, B2->4), so ADCgo sector N corresponds to reference symmetry
# N. gamess_orbsym generalises this to any supported D2h-subgroup point group.
orbsym = gamess_orbsym(mol, mo_act)

fcidump_path = os.path.join(TESTDATA, "h2o_dzp.fcidump")
fcidump.from_integrals(fcidump_path, h1e, h2e, NCAS, NELECAS, nuc=ecore,
                       ms=0, orbsym=orbsym, tol=1e-18, float_format="% .17g")
print(f"wrote {fcidump_path}  NORB={NCAS} NELEC={NELECAS} nocc={NELECAS // 2}")

# Sidecar for the atom-resolved two-hole populations: active-space MO coeffs
# (nAO x 29), the full AO overlap, the AO->atom map named O/H1/H2 to match the
# reference popana columns, plus dipole integrals and geometry. scf_dip is the
# whole-molecule RHF dipole; it is *not* reproducible from these frozen-core MOs.
mo = fcidump_common.write_sidecar(MO_PATH, mol, mo_act, mf.get_ovlp(),
                                  dm=mf.make_rdm1())
print(f"wrote {MO_PATH}  nAO={mo['nao']} atoms={mo['atom_names']}")

# Small manifest recording provenance + the gate value.
ref = {
    "molecule": "h2o",
    "basis": "DZP+diffuse (Dunning), cartesian",
    "reference": "../ADCanalysis/examples/DIP_h2o/adcdip{1..4}.out (theADCcode)",
    "norb": NCAS, "nelec": NELECAS, "ncore_frozen": int(ncore),
    "e_scf": float(mf.e_tot), "e_scf_reference": REF_SCF_ENERGY,
    "e_core_fcidump": float(ecore),
    "active_orbsym_gamess": [int(x) for x in orbsym],
}
with open(os.path.join(TESTDATA, "h2o_dzp.ref.json"), "w") as fh:
    json.dump(ref, fh, indent=2)
    fh.write("\n")
print("active-space GAMESS-UK ORBSYM:", orbsym)
