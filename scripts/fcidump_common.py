"""Shared FCIDUMP side-artifacts: the MO/overlap sidecar and the run manifest.

FCIDUMP carries neither the MO coefficients nor the AO overlap, but ADCgo's
atom-resolved two-hole population (Tarantelli U-transform) needs both, plus an
AO->atom map. It also carries no dipole integrals and no geometry, which the
transition-moment machinery needs. This writer is the single source of that JSON
sidecar for ``dump_fcidump.py``, ``gen_fcidump.py`` and ``gen_ref_fcidump.py``.
"""

import json
from collections import Counter

import numpy


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


def dipole_keys(mol, dm=None):
    """The dipole/geometry half of the sidecar.

    The dipole integrals are dumped in the AO basis so the sidecar stays independent
    of which MOs the FCIDUMP happens to carry; ADCgo forms C^T D C itself. The gauge
    origin is pinned to the centre of nuclear charge and recorded: ``int1e_r`` is
    origin-at-zero unless told otherwise, and the dipole of the N-1 cation states
    these integrals exist for *is* origin dependent.

    ``dm`` is the SCF AO density matrix. Given one, the whole-molecule RHF dipole is
    recorded as ``scf_dip`` -- a gate value for the AO->MO transform on the reader
    side, reproducible from ``mo_coeff`` only when the MO set spans all the occupied
    orbitals (i.e. not for a frozen-core active space).
    """
    charges = mol.atom_charges()
    coords = mol.atom_coords()  # bohr
    origin = numpy.einsum("a,ax->x", charges, coords) / charges.sum()
    with mol.with_common_orig(origin):
        dip_ao = mol.intor("int1e_r", comp=3)

    keys = {
        "dip_origin": [float(x) for x in origin],
        "dip_ao": [[[float(x) for x in row] for row in comp] for comp in dip_ao],
        "atom_coords": [[float(x) for x in r] for r in coords],
        "atom_charges": [float(z) for z in charges],
    }
    if dm is not None:
        nuc = numpy.einsum("a,ax->x", charges, coords - origin)
        keys["scf_dip"] = [float(x) for x in nuc - numpy.einsum("xpq,pq->x", dip_ao, dm)]
    return keys


def write_sidecar(path, mol, mo_coeff, overlap, dm=None):
    """Write the sidecar JSON from scratch.

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
    doc.update(dipole_keys(mol, dm))
    with open(path, "w") as fh:
        json.dump(doc, fh)
        fh.write("\n")
    return doc


def augment_sidecar(path, mol, dm=None):
    """Add the dipole/geometry keys to an existing sidecar, leaving its other entries
    exactly as they are.

    This exists because the committed reference sidecars pair with committed FCIDUMPs
    that several bit-exact gates depend on. Rewriting them from a newer pyscf moves the
    integrals by ~1e-13 -- harmless physically, fatal to those gates -- so the sidecar
    is extended in place rather than regenerated.

    That is only sound if the freshly computed AO dipole integrals live in the same AO
    basis as the stored MO coefficients. The stored AO overlap is the check: it depends
    on the AO basis and its ordering, and on nothing else (no SCF, no convergence
    noise). If it reproduces this ``mol``'s overlap, the AO conventions agree.
    """
    with open(path) as fh:
        doc = json.load(fh)

    ovlp = mol.intor("int1e_ovlp")
    stored = numpy.asarray(doc["overlap"])
    if stored.shape != ovlp.shape:
        raise SystemExit(f"augment_sidecar: {path} has overlap {stored.shape}, "
                         f"this mol has {ovlp.shape} -- different AO basis")
    err = numpy.abs(stored - ovlp).max()
    if err > 1e-12:
        raise SystemExit(f"augment_sidecar: {path} overlap disagrees with this mol by "
                         f"{err:.3e} -- AO basis or ordering changed, the stored "
                         "mo_coeff cannot be combined with fresh AO integrals")

    doc.update(dipole_keys(mol, dm))
    with open(path, "w") as fh:
        json.dump(doc, fh)
        fh.write("\n")
    return doc, err


def write_manifest(path, manifest):
    """Write a small provenance/manifest JSON (indented)."""
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
