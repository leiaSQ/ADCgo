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
"""
import json
import os
from collections import Counter

import numpy as np
from pyscf import gto, scf, mcscf, symm
from pyscf.tools import fcidump

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(os.path.dirname(HERE), "testdata")
os.makedirs(TESTDATA, exist_ok=True)

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

# Frozen-core active space (MOs 2..30). CASCI defaults ncore=(10-8)/2=1 and, with
# ncas=29, leaves MO 31 out -> exactly "2 to 30".
mc = mcscf.CASCI(mf, NCAS, NELECAS)
ncore = mc.ncore
active = slice(ncore, ncore + NCAS)
mo_act = mf.mo_coeff[:, active]

h1e, ecore = mc.get_h1eff()
h2e = mc.get_h2eff(mo_act)

# ORBSYM: map pyscf irrep ids of the active MOs to 1-based Molpro labels (C2v:
# A1->1, A2->4, B1->2, B2->3), the convention ADCgo's reader + XOR product expect.
orbsym_ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_act)
orbsym = [fcidump.ORBSYM_MAP[mol.groupname][i] for i in orbsym_ids]

fcidump_path = os.path.join(TESTDATA, "h2o_dzp.fcidump")
fcidump.from_integrals(fcidump_path, h1e, h2e, NCAS, NELECAS, nuc=ecore,
                       ms=0, orbsym=orbsym, tol=1e-18, float_format="% .17g")
print(f"wrote {fcidump_path}  NORB={NCAS} NELEC={NELECAS} nocc={NELECAS // 2}")

# C/S sidecar for the atom-resolved two-hole populations: active-space MO coeffs
# (nAO x 29), the full AO overlap, and the AO->atom map named O/H1/H2 to match the
# reference popana columns.
C = mo_act
S = mf.get_ovlp()

elem_counts = Counter(mol.atom_symbol(a) for a in range(mol.natm))
seen = Counter()
atom_names = []
for a in range(mol.natm):
    s = mol.atom_symbol(a)
    if elem_counts[s] > 1:
        seen[s] += 1
        atom_names.append(f"{s}{seen[s]}")
    else:
        atom_names.append(s)

ao_atom = [0] * C.shape[0]
for a, (_, _, ao0, ao1) in enumerate(mol.aoslice_by_atom()):
    for p in range(ao0, ao1):
        ao_atom[p] = a

mo = {
    "nao": int(C.shape[0]),
    "nmo": int(C.shape[1]),
    "mo_coeff": [[float(x) for x in row] for row in C],
    "overlap": [[float(x) for x in row] for row in S],
    "ao_atom": [int(x) for x in ao_atom],
    "atom_names": atom_names,
}
mo_path = os.path.join(TESTDATA, "h2o_dzp.mo.json")
with open(mo_path, "w") as fh:
    json.dump(mo, fh)
    fh.write("\n")
print(f"wrote {mo_path}  nAO={mo['nao']} atoms={atom_names}")

# Small manifest recording provenance + the gate value.
ref = {
    "molecule": "h2o",
    "basis": "DZP+diffuse (Dunning), cartesian",
    "reference": "../ADCanalysis/examples/DIP_h2o/adcdip{1..4}.out (theADCcode)",
    "norb": NCAS, "nelec": NELECAS, "ncore_frozen": int(ncore),
    "e_scf": float(mf.e_tot), "e_scf_reference": REF_SCF_ENERGY,
    "e_core_fcidump": float(ecore),
    "active_orbsym_molpro": [int(x) for x in orbsym],
}
with open(os.path.join(TESTDATA, "h2o_dzp.ref.json"), "w") as fh:
    json.dump(ref, fh, indent=2)
    fh.write("\n")
print("active-space Molpro ORBSYM:", orbsym)
