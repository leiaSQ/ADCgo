"""GAMESS-UK / theADCcode Abelian point-group irrep numbering.

theADCcode numbers the C2v-and-subgroup irreps as (adc3_debug.sh / dip.in):

    C1   a
    Cs   a'   a"
    Ci   ag   au
    C2   a    b
    D2   a    b1   b2   b3
    C2v  a1   a2   b1   b2
    C2h  ag   bg   au   bu
    D2h  ag   b1g  b2g  b3g   au   b1u  b2u  b3u

This differs from pyscf/Molpro ordering. ADCgo's symmetry handling only uses the
irrep integer for the XOR direct product, so any consistent Z2^n automorphism gives
identical physics — but to make ADCgo *sector N* correspond to theADCcode *symmetry
N* (so per-irrep output labels line up), the FCIDUMP ORBSYM must carry this exact
numbering. Emitting these labels for any supported group generalises the earlier
C2v-only fix.
"""

import re

from pyscf import symm

# pyscf Mulliken irrep name -> 1-based GAMESS-UK label, per point group.
GAMESS_LABEL = {
    "C1":  {"A": 1},
    "Cs":  {"A'": 1, 'A"': 2},
    "Ci":  {"Ag": 1, "Au": 2},
    "C2":  {"A": 1, "B": 2},
    "D2":  {"A": 1, "B1": 2, "B2": 3, "B3": 4},
    "C2v": {"A1": 1, "A2": 2, "B1": 3, "B2": 4},
    "C2h": {"Ag": 1, "Bg": 2, "Au": 3, "Bu": 4},
    "D2h": {"Ag": 1, "B1g": 2, "B2g": 3, "B3g": 4,
            "Au": 5, "B1u": 6, "B2u": 7, "B3u": 8},
}


def gamess_orbsym(mol, mo_coeff):
    """1-based GAMESS-UK ORBSYM labels for the columns of ``mo_coeff``.

    Maps each MO's pyscf irrep *name* to theADCcode's numbering, so the resulting
    FCIDUMP is sector-for-sector consistent with theADCcode for any supported
    D2h-subgroup point group. Raises for symmetry-off / unsupported groups.
    """
    group = mol.groupname
    labels = GAMESS_LABEL.get(group)
    if labels is None:
        raise ValueError(
            f"point group {group!r} is not an ADC-supported D2h subgroup "
            f"(expected one of {sorted(GAMESS_LABEL)})")
    names = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
    return [labels[nm] for nm in names]


def rewrite_fcidump_orbsym(path, orbsym):
    """Overwrite the ORBSYM= line of an existing FCIDUMP with ``orbsym``.

    Used when the writer (pyscf ``fcidump.from_scf``) cannot take an explicit
    orbsym: the integrals it wrote are kept byte-for-byte and only the label line
    is replaced, so no energy/golden can shift.
    """
    with open(path) as fh:
        text = fh.read()
    line = "  ORBSYM=" + "".join(f"{s}," for s in orbsym) + "\n"
    text, n = re.subn(r"(?im)^[ \t]*ORBSYM[ \t]*=.*\n", line, text, count=1)
    if n != 1:
        raise RuntimeError(f"no ORBSYM= line found to rewrite in {path}")
    with open(path, "w") as fh:
        fh.write(text)
