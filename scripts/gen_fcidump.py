#!/usr/bin/env python
"""Generate the M0 reference: RHF + MP2 on H2O/cc-pVDZ with pyscf, dump the
MO integrals as an FCIDUMP, and write a golden reference JSON that the Go tests
validate against.

Run with the `adcgo` conda env:
    /home/leia/miniconda3/envs/adcgo/bin/python scripts/gen_fcidump.py
"""
import json
import os

from pyscf import gto, scf, mp
from pyscf.tools import fcidump

from gamess_orbsym import gamess_orbsym, rewrite_fcidump_orbsym

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(os.path.dirname(HERE), "testdata")
os.makedirs(TESTDATA, exist_ok=True)

BASIS = "cc-pvdz"

# Standard equilibrium-ish water geometry (Angstrom).
# C2v symmetry is enabled so the FCIDUMP carries real ORBSYM labels: this is what
# exercises the per-irrep symmetry blocking in package dip (M2). The RHF/MP2
# energies are symmetry-invariant, so the M0 reference is unchanged.
mol = gto.M(
    atom="""
        O   0.000000   0.000000   0.117790
        H   0.000000   0.755450  -0.471160
        H   0.000000  -0.755450  -0.471160
    """,
    basis=BASIS,
    symmetry=True,
    unit="Angstrom",
)

mf = scf.RHF(mol)
# Tight convergence so the MO-basis Fock is diagonal to ~1e-10; otherwise the
# reported mo_energy and a Fock diagonal reconstructed from the dumped integrals
# disagree at the ~1e-5 gradient tolerance (a convergence artifact, not a bug).
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-9
mf.run()
mp2 = mp.MP2(mf).run()

fcidump_path = os.path.join(TESTDATA, "h2o.fcidump")
# %.17g round-trips a float64 exactly; keep tol tiny so nothing physical is
# dropped. This keeps the reconstructed Fock diagonal within FP path noise of
# pyscf's own mo_energy.
# Write the integrals via from_scf, then relabel ORBSYM with 1-based GAMESS-UK /
# theADCcode irrep numbers (C2v: A1→1, A2→2, B1→3, B2→4) so ADCgo sector N matches
# theADCcode symmetry N. Generalises to any supported D2h subgroup via gamess_orbsym;
# the Go reader's 1-based→0-based + XOR direct product stays consistent with the
# symmetry-off case (all labels 1).
fcidump.from_scf(mf, fcidump_path, tol=1e-18, float_format="% .17g",
                 molpro_orbsym=True)
rewrite_fcidump_orbsym(fcidump_path, gamess_orbsym(mol, mf.mo_coeff))

ref = {
    "molecule": "h2o",
    "basis": BASIS,
    "norb": int(mf.mo_coeff.shape[1]),
    "nelec": int(mol.nelectron),
    "e_nuc": float(mol.energy_nuc()),
    "e_scf": float(mf.e_tot),
    "e_mp2_corr": float(mp2.e_corr),
    "e_mp2_tot": float(mf.e_tot + mp2.e_corr),
    "mo_energy": [float(x) for x in mf.mo_energy],
}
with open(os.path.join(TESTDATA, "h2o.ref.json"), "w") as fh:
    json.dump(ref, fh, indent=2)
    fh.write("\n")

# C/S sidecar: MO coefficients, AO overlap, and AO->atom map. Needed for the
# atom-resolved two-hole population (Tarantelli U-transform) — FCIDUMP carries
# none of this. mo_coeff is (nAO, nMO); overlap is (nAO, nAO), both row-major.
C = mf.mo_coeff
S = mf.get_ovlp()

# Distinct atom labels: element symbol, suffixed with a 1-based per-element index
# when that element occurs more than once (O, H1, H2), matching the popana style.
from collections import Counter
elem_counts = Counter(mol.atom_symbol(a) for a in range(mol.natm))
seen = Counter()
atom_names = []
for a in range(mol.natm):
    sym = mol.atom_symbol(a)
    if elem_counts[sym] > 1:
        seen[sym] += 1
        atom_names.append(f"{sym}{seen[sym]}")
    else:
        atom_names.append(sym)

# AO -> atom index via the per-atom AO slices.
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
with open(os.path.join(TESTDATA, "h2o.mo.json"), "w") as fh:
    json.dump(mo, fh)
    fh.write("\n")
print(f"wrote {os.path.join(TESTDATA, 'h2o.mo.json')}  nAO={mo['nao']} atoms={atom_names}")

print(f"wrote {fcidump_path}")
print(f"NORB={ref['norb']} NELEC={ref['nelec']}")
print(f"E(SCF)      = {ref['e_scf']:.10f} Ha")
print(f"E(MP2 corr) = {ref['e_mp2_corr']:.10f} Ha")
print(f"E(MP2 tot)  = {ref['e_mp2_tot']:.10f} Ha")
