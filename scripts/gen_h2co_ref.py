#!/usr/bin/env python
"""Matched FCIDUMP for the formaldehyde (H2CO) CVS-ADC(4) generality check.

Reproduces the GAMESS-UK RHF/cc-pVDZ (cartesian) formaldehyde used for the reference
ADC4CVS tapes, freezes 1 core (active 2..40 -> 39 orbitals), and writes h2co.fcidump.
Gates on the GUK SCF total energy -113.8760163547 Ha so the integrals are matched.
See testdata/reference/adc4_h2co/README.md. Run with the adcgo conda env.
"""
import sys, os
sys.path.insert(0, "/home/leia/Documents/ADCgo/scripts")
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.tools import fcidump
from gamess_orbsym import gamess_orbsym

REF_SCF = -113.8760163547
B = 0.52917721092  # not used; coords already in bohr
# GUK oriented coordinates (a.u./bohr), C2 along z, molecule in yz-plane
mol = gto.M(
    atom=[["C", (0.0, 0.0, 1.0020893)],
          ["O", (0.0, 0.0, -1.2806997)],
          ["H", (0.0, 1.7904254, 2.1165310)],
          ["H", (0.0, -1.7904254, 2.1165310)]],
    basis="cc-pvdz", cart=True, symmetry=True, unit="Bohr",
)
mf = scf.RHF(mol); mf.conv_tol=1e-12; mf.conv_tol_grad=1e-9; mf.run()
print("nao_cart=", mol.nao_cart(), "E(SCF)=", mf.e_tot)
if abs(mf.e_tot - REF_SCF) > 1e-4:
    raise SystemExit("SCF gate FAILED d=%.2e"%abs(mf.e_tot-REF_SCF))
print("SCF gate OK")
NCAS, NELECAS = 39, 14
mc = mcscf.CASCI(mf, NCAS, NELECAS)
print("ncore=", mc.ncore)
active = slice(mc.ncore, mc.ncore+NCAS)
mo_act = mf.mo_coeff[:, active]
h1e, ecore = mc.get_h1eff(); h2e = mc.get_h2eff(mo_act)
orbsym = gamess_orbsym(mol, mo_act)
fcidump.from_integrals("h2co.fcidump", h1e, h2e, NCAS, NELECAS, nuc=ecore,
                       ms=0, orbsym=orbsym, tol=1e-18, float_format="% .17g")
print("wrote h2co.fcidump NORB=%d NELEC=%d"%(NCAS,NELECAS))
print("orbsym:", orbsym)
