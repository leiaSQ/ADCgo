#!/usr/bin/env python3
"""Generate the M5 SIP oracle: pyscf IP-ADC(2)/(3) energies + spectroscopic
factors on the *same* H2O/cc-pVDZ integrals ADCgo reads from testdata/h2o.fcidump.

Because both codes see identical MO integrals (same geometry, basis, tight SCF),
the comparison is a matched-integral one: any residual is ADC method, not basis.
pyscf implements the standard ISR IP-ADC; theADCcode ndadc3_ip is the same
non-Dyson ADC. In practice the 2h1p satellite roots and the spectroscopic factors
agree to ~1e-5 / ~5e-4, while the strong 1h main lines sit ~0.03-0.07 eV apart
(a small self-energy-formulation difference), so the validation asserts the main
lines within a documented band and the factors tightly.

pyscf's spectroscopic factor p is normalised to 2 (two spin channels) for a
closed-shell RHF reference, so the pole strength (fraction) is p/2 — that is what
ADCgo's F·Y amplitude norm reproduces.

Writes testdata/h2o_sip.pyscf.json. Run in the `adcgo` conda env (pyscf 2.13):
    /home/leia/miniconda3/envs/adcgo/bin/python scripts/gen_sip_ref.py
"""
import json
import os

import numpy as np
from pyscf import gto, scf, adc

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(os.path.dirname(HERE), "testdata")

# Must match scripts/gen_fcidump.py exactly (same mol => same integrals as
# testdata/h2o.fcidump).
BASIS = "cc-pvdz"
mol = gto.M(
    atom="""
        O   0.000000   0.000000   0.117790
        H   0.000000   0.755450  -0.471160
        H   0.000000  -0.755450  -0.471160
    """,
    basis=BASIS,
    symmetry=True,
    unit="Angstrom",
    verbose=0,
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
assert mf.converged, "SCF did not converge"

NROOTS = 12
ref = {"basis": BASIS, "e_scf": float(mf.e_tot), "nroots": NROOTS, "roots": {}}

for order in (2, 3):
    myadc = adc.ADC(mf)
    myadc.method = f"adc({order})"
    myadc.method_type = "ip"
    e, v, p, x = myadc.kernel(nroots=NROOTS)
    e = np.atleast_1d(e)
    p = np.atleast_1d(p)
    ref["roots"][str(order)] = [
        {"e_ha": float(e[i]), "sf": float(p[i]) / 2.0} for i in range(len(e))
    ]
    strong = [(float(e[i]), float(p[i]) / 2.0) for i in range(len(e)) if p[i] / 2.0 > 0.5]
    print(f"ip-adc({order}): {len(e)} roots, {len(strong)} strong (SF>0.5)")
    for en, sf in strong:
        print(f"   {en:.6f} Ha ({en*27.211386:.3f} eV)  SF {sf:.4f}")

out = os.path.join(TESTDATA, "h2o_sip.pyscf.json")
with open(out, "w") as fh:
    json.dump(ref, fh, indent=2)
    fh.write("\n")
print("wrote", out)
