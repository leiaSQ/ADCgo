"""Shared FCIDUMP side-artifacts: the MO/overlap sidecar and the run manifest.

FCIDUMP carries neither the MO coefficients nor the AO overlap, but ADCgo's
atom-resolved two-hole population (Tarantelli U-transform) needs both, plus an
AO->atom map. This writer is the single source of that JSON sidecar, factored out
of the previously duplicated blocks in ``gen_fcidump.py`` and
``gen_ref_fcidump.py`` so the generalized ``dump_fcidump.py`` and the legacy
scripts stay byte-compatible.
"""

import json
from collections import Counter


def atom_names(mol):
    """Distinct per-atom labels: element symbol, suffixed with a 1-based index when
    that element occurs more than once (O, H1, H2) -- matches the popana columns."""
    counts = Counter(mol.atom_symbol(a) for a in range(mol.natm))
    seen = Counter()
    names = []
    for a in range(mol.natm):
        sym = mol.atom_symbol(a)
        if counts[sym] > 1:
            seen[sym] += 1
            names.append(f"{sym}{seen[sym]}")
        else:
            names.append(sym)
    return names


def write_sidecar(path, mol, mo_coeff, overlap):
    """Write the C/S sidecar JSON.

    ``mo_coeff`` is (nAO x nMO) for the MOs actually in the FCIDUMP (full space or
    the active subset); ``overlap`` is the (nAO x nAO) AO overlap. Both row-major.
    """
    C = mo_coeff
    ao_atom = [0] * C.shape[0]
    for a, (_, _, ao0, ao1) in enumerate(mol.aoslice_by_atom()):
        for p in range(ao0, ao1):
            ao_atom[p] = a
    doc = {
        "nao": int(C.shape[0]),
        "nmo": int(C.shape[1]),
        "mo_coeff": [[float(x) for x in row] for row in C],
        "overlap": [[float(x) for x in row] for row in overlap],
        "ao_atom": [int(x) for x in ao_atom],
        "atom_names": atom_names(mol),
    }
    with open(path, "w") as fh:
        json.dump(doc, fh)
        fh.write("\n")
    return doc


def write_manifest(path, manifest):
    """Write a small provenance/manifest JSON (indented)."""
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
