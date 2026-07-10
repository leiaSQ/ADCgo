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

# ---------------------------------------------------------------------------
# Dyson orbitals, term-matched to ADCgo's O(1) virtual block.
#
# pyscf's Dyson amplitude for a *virtual* orbital carries two pieces: the first-order
# 2h1p term (the MP1 doubles, f(1) ~ V/dE) and the second-order singles t1_2 -- plus
# the second-order doubles t2_2 at adc(3). ADCgo implements only f(1) (Chunk 4 of
# docs/adc4_rassi_plan.md; t1_2/t2_2 stay on the deferred order-consistency list), so
# comparing against pyscf's default compute_dyson_mo() would compare different
# quantities. Setting approx_trans_moments=True on adc(2)-x drops t1_2 and t2_2 and
# leaves pyscf's virtual block equal to exactly the one term ADCgo has.
#
# adc(2)-x is the extended ADC(2) that ADCgo's `-order 2` corresponds to (it carries the
# first-order 2h1p/2h1p block). The eigenvectors still differ -- pyscf is Dyson/ISR ADC,
# ndadc3_ip is non-Dyson -- so the Go-side gate is a similarity check, not an equality.
#
# X is (nmo, nroots) in the MO basis of this same mol/SCF, i.e. the basis
# testdata/h2o.fcidump is written in. Store it as-is; compute_dyson_mo() would only
# multiply by mo_coeff.
myadc = adc.ADC(mf)
myadc.method = "adc(2)-x"
myadc.method_type = "ip"
myadc.approx_trans_moments = True
e, v, p, x = myadc.kernel(nroots=NROOTS)
e = np.atleast_1d(e)
p = np.atleast_1d(p)
X = np.asarray(x)  # kernel's 4th return value: the spectroscopic amplitudes, (nmo, nroots)
assert X.shape == (mol.nao_nr(), len(e)), X.shape
ref["dyson_o1"] = {
    "method": "adc(2)-x",
    "approx_trans_moments": True,
    "note": "virtual block = first-order 2h1p term only; term-matched to ADCgo -order 2",
    "roots": [
        {
            "e_ha": float(e[i]),
            "sf": float(p[i]) / 2.0,
            "d": [float(t) for t in X[:, i]],
        }
        for i in range(len(e))
    ],
}
print(f"dyson_o1 (adc(2)-x, approx_trans_moments): {X.shape[0]} MOs x {len(e)} roots")
for i in range(len(e)):
    if p[i] / 2.0 > 0.5:
        occ = np.linalg.norm(X[: mol.nelectron // 2, i]) ** 2
        vir = np.linalg.norm(X[mol.nelectron // 2 :, i]) ** 2
        print(f"   {e[i]:.6f} Ha  SF {p[i]/2:.4f}  |d_occ|^2 {occ:.4f}  |d_vir|^2 {vir:.4f}")

out = os.path.join(TESTDATA, "h2o_sip.pyscf.json")
with open(out, "w") as fh:
    json.dump(ref, fh, indent=2)
    fh.write("\n")
print("wrote", out)
