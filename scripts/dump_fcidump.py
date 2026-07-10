#!/usr/bin/env python
"""Generalized FCIDUMP generator for ADCgo: any basis + any z-matrix.

This is the ADCgo analogue of theADCcode's GAMESS-UK front-end. Given a geometry
(GAMESS-UK z-matrix / .xyz / Cartesian), a basis (a GAMESS-UK basis file *or* a
named pyscf basis), and an optional frozen-core/active selection, it runs RHF in
pyscf and writes:

  * a Molpro-format FCIDUMP (MO integrals) with 1-based GAMESS-UK ORBSYM, and
  * the C/S sidecar (for ADCgo's atom-resolved two-hole populations), and
  * a small provenance manifest.

Configuration comes from either a sectioned input file (``--input dump.in``, see
adcgo_input.py) or plain CLI flags. Run with the adcgo conda env:

    /home/leia/miniconda3/envs/adcgo/bin/python scripts/dump_fcidump.py --input dump.in
    /home/leia/miniconda3/envs/adcgo/bin/python scripts/dump_fcidump.py \
        --zmat _zmatrix.dat --basis-file _basisset.dat --cartesian \
        --active "2 to 30" --gate -76.0498071428 --fcidump out.fcidump

Delegating integrals to pyscf keeps ADCgo's "FCIDUMP is the single ingestion path"
contract intact; the ``adcgo`` binary is unchanged.
"""

import argparse
import os
import sys

from pyscf import gto, scf, mp, mcscf
from pyscf.tools import fcidump

import gamess_zmat
import gamess_basis
import orbital_select
import fcidump_common
from adcgo_input import Config, parse_input_file
from gamess_orbsym import gamess_orbsym, rewrite_fcidump_orbsym

# pyscf FCIDUMP writer settings, matched to the existing gen_*.py scripts so
# regenerated dumps stay byte-compatible with the committed testdata.
_TOL = 1e-18
_FLOAT_FMT = "% .17g"


def build_mol(cfg):
    """Build the pyscf Mole from a Config's geometry + basis."""
    atoms, unit = gamess_zmat.read_geometry(cfg.resolve(cfg.geom_file))
    if cfg.unit:
        unit = cfg.unit
    if cfg.basis_file:
        basis = gamess_basis.load_gamess_basis(cfg.resolve(cfg.basis_file))
    else:
        basis = cfg.basis_name
    return gto.M(atom=atoms, basis=basis, charge=cfg.charge, spin=cfg.spin,
                 cart=cfg.cartesian, symmetry=cfg.symmetry, unit=unit)


def run_scf(cfg, mol):
    """Tightly-converged RHF, with an optional reference-energy gate."""
    mf = scf.RHF(mol)
    mf.conv_tol = cfg.conv_tol
    mf.conv_tol_grad = cfg.conv_tol_grad
    mf.run()
    if cfg.gate is not None and abs(mf.e_tot - cfg.gate) > 1e-4:
        raise SystemExit(
            f"SCF gate FAILED: E(SCF)={mf.e_tot:.10f} Ha, reference {cfg.gate:.10f} "
            f"Ha (|d|={abs(mf.e_tot - cfg.gate):.2e} > 1e-4). Check basis/geometry/"
            "cartesian setting.")
    if cfg.gate is not None:
        print(f"SCF gate OK: E(SCF)={mf.e_tot:.10f} Ha (ref {cfg.gate:.10f})")
    return mf


def _orbsym_labels(cfg, mol, mo_coeff):
    """1-based GAMESS-UK ORBSYM for mo_coeff columns; all-1 when symmetry is off."""
    if cfg.symmetry is False:
        return [1] * mo_coeff.shape[1]
    return gamess_orbsym(mol, mo_coeff)


def dump(cfg):
    """Build, solve, select the orbital space, and write FCIDUMP + sidecar + manifest."""
    mol = build_mol(cfg)
    mf = run_scf(cfg, mol)

    nmo = mf.mo_coeff.shape[1]
    sel = orbital_select.resolve(nmo, mol.nelectron, cfg.active, cfg.frozen_core)
    fcidump_path = cfg.resolve(cfg.fcidump)

    if sel.full:
        # Whole MO space: from_scf, then relabel ORBSYM (as gen_fcidump.py does).
        fcidump.from_scf(mf, fcidump_path, tol=_TOL, float_format=_FLOAT_FMT,
                         molpro_orbsym=True)
        orbsym = _orbsym_labels(cfg, mol, mf.mo_coeff)
        rewrite_fcidump_orbsym(fcidump_path, orbsym)
        mo_side = mf.mo_coeff
        norb, nelec, ncore = nmo, mol.nelectron, 0
    else:
        # Frozen-core + active/virtual selection via CASCI on reordered columns
        # (as gen_ref_fcidump.py does for the contiguous case).
        mo_reordered = mf.mo_coeff[:, sel.core0 + sel.active0]
        mc = mcscf.CASCI(mf, sel.ncas, sel.nelecas)
        mc.ncore = sel.ncore
        h1e, ecore = mc.get_h1eff(mo_coeff=mo_reordered)
        mo_act = mo_reordered[:, sel.ncore:]
        h2e = mc.get_h2eff(mo_act)
        orbsym = _orbsym_labels(cfg, mol, mo_act)
        fcidump.from_integrals(fcidump_path, h1e, h2e, sel.ncas, sel.nelecas,
                               nuc=ecore, ms=cfg.spin, orbsym=orbsym, tol=_TOL,
                               float_format=_FLOAT_FMT)
        mo_side = mo_act
        norb, nelec, ncore = sel.ncas, sel.nelecas, sel.ncore
    print(f"wrote {fcidump_path}  NORB={norb} NELEC={nelec} "
          f"nocc={nelec // 2} ncore_frozen={ncore}")

    if cfg.sidecar:
        side = cfg.resolve(cfg.sidecar)
        doc = fcidump_common.write_sidecar(side, mol, mo_side, mf.get_ovlp(),
                                           dm=mf.make_rdm1())
        print(f"wrote {side}  nAO={doc['nao']} atoms={doc['atom_names']} "
              f"dipole_origin={doc['dip_origin']}")

    if cfg.manifest:
        man = cfg.resolve(cfg.manifest)
        manifest = {
            "molecule": os.path.splitext(os.path.basename(cfg.geom_file))[0],
            "basis": cfg.basis_name or os.path.basename(cfg.basis_file),
            "cartesian": bool(cfg.cartesian),
            "norb": int(norb), "nelec": int(nelec), "ncore_frozen": int(ncore),
            "e_scf": float(mf.e_tot),
            "orbsym_gamess": [int(x) for x in orbsym],
        }
        if cfg.gate is not None:
            manifest["e_scf_gate"] = cfg.gate
        if sel.full:
            manifest["e_mp2_corr"] = float(mp.MP2(mf).run().e_corr)
        fcidump_common.write_manifest(man, manifest)
        print(f"wrote {man}")

    print(f"E(SCF) = {mf.e_tot:.10f} Ha  NORB={norb} NELEC={nelec}  "
          f"ORBSYM={orbsym}")
    return mf


def config_from_args(args):
    """Build a Config from CLI flags (the non---input path)."""
    if not args.zmat:
        raise SystemExit("need --zmat (geometry) unless --input is given")
    if bool(args.basis_file) == bool(args.basis):
        raise SystemExit("give exactly one of --basis-file or --basis")
    if not args.fcidump:
        raise SystemExit("need --fcidump (output path)")
    sym = {"auto": True, "off": False}.get(
        (args.sym_group or "auto").lower(), args.sym_group)
    return Config(
        base_dir=".",
        geom_file=args.zmat, unit=args.unit,
        basis_file=args.basis_file, basis_name=args.basis,
        cartesian=args.cartesian,
        charge=args.charge, spin=args.spin, symmetry=sym, gate=args.gate,
        frozen_core=args.frozen_core, active=args.active,
        fcidump=args.fcidump, sidecar=args.sidecar, manifest=args.manifest,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="sectioned input file (see adcgo_input.py)")
    p.add_argument("--zmat", "--geometry", dest="zmat",
                   help="geometry file: GAMESS-UK zmat, .xyz, or Cartesian list")
    p.add_argument("--unit", help="override distance unit (angstrom|bohr)")
    p.add_argument("--basis-file", help="GAMESS-UK basis file")
    p.add_argument("--basis", help="named pyscf basis, e.g. cc-pvdz")
    p.add_argument("--cartesian", action="store_true",
                   help="use cartesian GTOs (needed for GAMESS-UK bases)")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0, help="2S (unpaired electrons)")
    p.add_argument("--sym-group", help="point group: auto|off|C2v|Cs|...")
    p.add_argument("--gate", type=float, help="reference E(SCF) gate (Ha)")
    p.add_argument("--frozen-core", type=int, dest="frozen_core",
                   help="freeze N lowest MOs")
    p.add_argument("--active", help='GAMESS active list, e.g. "2 to 30"')
    p.add_argument("--fcidump", help="output FCIDUMP path")
    p.add_argument("--sidecar", help="output C/S sidecar JSON path")
    p.add_argument("--manifest", help="output manifest JSON path")
    args = p.parse_args(argv)

    cfg = parse_input_file(args.input) if args.input else config_from_args(args)
    dump(cfg)


if __name__ == "__main__":
    main()
