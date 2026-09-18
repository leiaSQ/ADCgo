#!/usr/bin/env python
"""Generate FCIDUMPs for the atomic Auger systems of Table VI in

    P. Kolorenc and V. Averbukh, J. Chem. Phys. 152, 214107 (2020),

so ADCgo's Fano-ADC widths can be compared against published values.

The basis requirement is set by the ENERGY SPAN of the virtual space, not by
diffuseness: Stieltjes imaging can only evaluate Gamma(E_Phi) if the 2h1p
pseudo-continuum brackets E_Phi, and a 2h1p state sits at eps_a - eps_k - eps_l.
For the Ne 1s vacancy E_Phi is 32.8 Eh, so the virtuals must reach ~33 Eh above
the valence double-hole energy; aug-cc-pVTZ tops out at 14.6 Eh and cannot
describe the decay at all, while aug-cc-pVQZ reaches 68.9 Eh. The paper uses
aug-cc-pV5Z plus 4s4p4d Kaufmann-Baumeister-Jungen Rydberg functions.

Run with the adcgo conda env:
    .../envs/adcgo/bin/python scripts/gen_fano_atoms.py [--basis aug-cc-pVQZ] [atoms...]
"""
import os
import sys

from pyscf import gto, scf
from pyscf.tools import fcidump

from gamess_orbsym import gamess_orbsym


def rewrite_orbsym_streaming(path, orbsym):
    """Replace an FCIDUMP's ORBSYM= line without reading the integrals.

    gamess_orbsym.rewrite_fcidump_orbsym slurps the whole file, which is fine for the
    committed fixtures but not here: a 163-orbital FCIDUMP is gigabytes, and the read
    plus the regex copy needs several times that. Only the header line changes, so
    stream the rest through untouched.
    """
    import shutil
    tmp = path + ".tmp"
    line = "  ORBSYM=" + "".join(f"{sym}," for sym in orbsym) + "\n"
    done = False
    with open(path) as src, open(tmp, "w") as dst:
        for raw in src:
            if not done and raw.lstrip().upper().startswith("ORBSYM"):
                dst.write(line)
                done = True
                continue
            dst.write(raw)
            if done and raw.lstrip().startswith("&END"):
                # Header finished; copy the integral block verbatim and in bulk.
                shutil.copyfileobj(src, dst, 1 << 22)
                break
    if not done:
        os.remove(tmp)
        raise RuntimeError(f"no ORBSYM= line found to rewrite in {path}")
    os.replace(tmp, path)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FANO_OUT", os.path.join(os.path.dirname(HERE), "testdata", "fano"))

# atom -> (vacancy label, 0-based occupied MO index of the vacancy)
# atom -> (vacancy label, 0-based occupied MO index), read off the RHF orbital energies:
#   Ne  eps = [-32.77 -1.93 -0.85 x3]                       -> 1s is MO 0
#   Mg  eps = [-49.03 -3.77 -2.28 x3 -0.25]                 -> 2s is MO 1
#   Ar  eps = [-118.61 -12.32 -9.57 x3 -1.28 -0.59 x3]      -> 2p is MO 2 (of 2,3,4)
#   Kr  eps = [-520 -69.9 -63.0 x3 -10.85 -8.33 x3 -3.83 x5 -1.15 -0.52 x3]
#                                                            -> 3d is MO 9 (of 9..13)
# For a degenerate shell any member gives the same width by symmetry, but they sit in
# different irreps (Ar 2p spans b1u/b2u/b3u; Kr 3d spans ag and b1g/b2g/b3g), and the
# vacancy's irrep is the sector that gets solved.
SYSTEMS = {
    "Ne": ("1s", 0),
    "Mg": ("2s", 1),
    "Ar": ("2p", 2),
    "Kr": ("3d", 9),
}


def augment(basis_dict, atom, nfun, lmax, alpha0, ratio):
    """Add an even-tempered set of extra shells to densify the virtual spectrum.

    The paper augments its aug-cc-pV5Z basis with 4s4p4d Kaufmann-Baumeister-Jungen
    Rydberg-like functions. This serves the same PURPOSE with a set we can state
    exactly: Stieltjes imaging samples the coupling density only where the
    pseudo-continuum has states, and for an Auger decay the relevant 2h1p states sit
    at eps_a = E_Phi + eps_k + eps_l, so what matters is the DENSITY OF VIRTUAL LEVELS
    near that energy. A Gaussian of exponent alpha produces virtuals on a kinetic scale
    ~alpha, so alpha has to bracket the OUTGOING ELECTRON'S energy.

    ALPHA0 IS NOT TRANSFERABLE BETWEEN SYSTEMS, and getting it wrong is the single
    largest error in an otherwise correct calculation. The outgoing electron carries

        eps_a = E_Phi - |eps_k| - |eps_l|   (k, l the two final holes)

    so the right window is set by the OPEN channels, not by E_Phi alone:

      Ne+(1s^-1)  E_Phi 32.8 Eh, 2p^-2 channel -> eps_a ~ 31 Eh   -> alpha0 ~ 4  (to 156)
      Ar+(2p^-1)  E_Phi  9.1 Eh, 3p^-2         -> eps_a ~  8 Eh   -> alpha0 ~ 4  works
      Mg+(2s^-1)  E_Phi  3.6 Eh, 2p^-2 CLOSED  -> eps_a ~ 1-3 Eh  -> alpha0 ~ 0.4
                  (2p^-2 sits at 4.6 Eh, ABOVE the 3.6 Eh vacancy, so only 2p^-1 3s^-1
                   and 3s^-2 are open and the electron comes out slow)
      Kr+(3d^-1)  E_Phi  3.5 Eh, 4s/4p channels -> eps_a 1.2-2.4 Eh -> alpha0 ~ 0.4
                  (eps_4s = -1.15, eps_4p = -0.52, so the M4,5-N2,3N2,3 electron leaves
                   at 32-66 eV, where it is measured -- the same slow window as Mg
                   despite a vacancy 26x deeper than Mg's in binding energy terms)

    So the grouping is by the OUTGOING ELECTRON, not by how deep the vacancy is: Ne and Ar
    take the tight set, Mg and Kr the soft one. With the tight set Kr gives 54 meV against
    a published 68 +/- 2, and its P space holds only 367 configurations.

    Measured on Mg with everything else fixed: alpha0 = 4.0 gives 1109 meV, alpha0 = 0.4
    gives 561 meV, against a published 585 +/- 3. A 2x error in the width from the
    exponent range alone.

    nfun functions per l up to lmax, exponents alpha0 * ratio**i.
    """
    from pyscf import gto as _gto
    b = _gto.basis.load(basis_dict, atom) if isinstance(basis_dict, str) else basis_dict
    extra = []
    for l in range(lmax + 1):
        for i in range(nfun):
            extra.append([l, [alpha0 * ratio ** i, 1.0]])
    return b + extra

def kbj(atom, nfun, lmax, charge=1):
    """Kaufmann-Baumeister-Jungen Rydberg/continuum Gaussians (J. Phys. B 22, 2223 (1989)).

    This is what the paper augments aug-cc-pV5Z with ("4s4p4d Rydberg-like Gaussian
    functions", ref. 50 there), and the reason it matters is not diffuseness but LINEAR
    INDEPENDENCE. An even-tempered set built to cover the outgoing electron's kinetic
    window lands on top of the valence exponents it has to coexist with: measured on
    Ne/aug-cc-pV5Z, adding 4s4p4d even-tempered functions at alpha0 = 4.0 drops the
    smallest overlap eigenvalue from 1.7e-3 to 1.6e-7 and costs one MO outright, and a
    6s6p6d set at alpha0 = 2.0 costs seven. The RHF energy then RISES on adding functions
    (-128.546786 -> -128.546700 Eh), which is the signature of orthogonalisation silently
    discarding basis vectors -- and what it discards is the virtual spectrum Stieltjes
    samples.

    KBJ exponents are generated by maximising overlap with Slater functions of constant
    exponent and increasing principal quantum number, which makes them a genuinely
    Rydberg-like sequence rather than a rescaled valence one:

        alpha(n, l) = (Z / (2 n))^2 / (a_l n + b_l)^2

    with n = 1, 2, 3, ... and (a_l, b_l) tabulated per l. Z is the charge seen by the
    outgoing electron, which for decay of a singly ionized species is 1 at the relevant
    distances.
    """
    # KBJ Table 1, the l = 0..4 sequence parameters.
    AB = {0: (0.584342, 0.424483), 1: (0.452663, 0.309805),
          2: (0.382368, 0.251333), 3: (0.337804, 0.215739), 4: (0.306158, 0.191634)}
    extra = []
    for l in range(lmax + 1):
        a, b = AB[l]
        for n in range(1, nfun + 1):
            extra.append([l, [(charge / (2.0 * n)) ** 2 / (a * n + b) ** 2, 1.0]])
    return extra


def bse_basis(name, atom, uncontract=False, drop_l=None):
    """Load a basis (and ECP, if the set has one) from the Basis Set Exchange library.

    The supplementary material says every basis set was taken from BSE, and the ones it
    names are not all in pyscf's bundled tables: aug-cc-pCV5Z (Ne) and cc-pwCV5Z-PP (Kr)
    are missing, and Kr's needs the Stuttgart-Koeln MCDHF RSC ECP for [Ne] that ships with
    it. drop_l removes shells with l >= drop_l, which is how the paper's "H- and I-type
    functions removed" for Kr is applied (drop_l = 5).
    """
    import basis_set_exchange as _bse
    from pyscf import gto as _g
    txt = _bse.get_basis(name, elements=[atom], fmt="nwchem")
    # pyscf's parse() dispatches on the presence of an "ECP" line and then refuses the
    # whole file, so the two sections have to be split before parsing either.
    lines = txt.splitlines()
    cut = next((i for i, ln in enumerate(lines) if ln.strip().upper().startswith("ECP")), None)
    ecp = None
    if cut is None:
        bas = _g.basis.parse(txt, optimize=False)
    else:
        bas = _g.basis.parse("\n".join(lines[:cut]), optimize=False)
        ecp = _g.basis.parse_ecp("\n".join(lines[cut:]))
    if uncontract:
        bas = _g.uncontract(bas)
    if drop_l is not None:
        bas = [sh for sh in bas if sh[0] < drop_l]
    return bas, ecp


def freeze_core(mf, ncore, virt_max=None):
    """Fold the lowest ncore doubly occupied MOs into an effective one-electron term, and
    optionally drop virtual MOs above virt_max hartree.

    Both are "configuration space restrictions" in the supplementary material's language,
    and both have to happen BEFORE the FCIDUMP is written -- freezing changes h1 and the
    core energy, and a virtual cutoff changes which orbitals exist at all. The paper uses
    both for every atom of Table V: 1s frozen and virtuals below 80 Ha for Mg, below 460 Ha
    for Ne (nothing frozen), 1s frozen and below 30 Ha for Ar, 3s/3p frozen and below 11 Ha
    for Kr. The cutoff is not cosmetic: an uncontracted core-valence basis augmented by
    10s10p8d continuum functions puts thousands of virtuals far above any energy the decay
    can reach, and they inflate the 3h2p class and the Stieltjes moments without
    contributing to Gamma(E_Phi).

    Returns (h1, eri, nact, nelec_act, ecore, keep) where keep indexes the retained MOs of
    the parent set, so ORBSYM can be sliced consistently.
    """
    import numpy as np
    from pyscf import ao2mo
    mol = mf.mol
    mo = mf.mo_coeff
    nocc = mol.nelectron // 2
    keep = list(range(ncore, mo.shape[1]))
    if virt_max is not None:
        keep = [i for i in keep if i < nocc or mf.mo_energy[i] < virt_max]
    core, act = mo[:, :ncore], mo[:, keep]
    dm_core = 2.0 * core @ core.T
    hcore = mf.get_hcore()
    vj, vk = mf.get_jk(mol, dm_core)
    veff = vj - 0.5 * vk
    ecore = (np.einsum("ij,ji->", dm_core, hcore)
             + 0.5 * np.einsum("ij,ji->", dm_core, veff) + mol.energy_nuc())
    h1 = act.T @ (hcore + veff) @ act
    eri = ao2mo.kernel(mol, act)
    return h1, eri, act.shape[1], mol.nelectron - 2 * ncore, ecore, keep


def main(argv):
    basis = "aug-cc-pVQZ"
    aug = 0          # --augment N: N even-tempered extra functions per l, up to --aug-lmax
    auglmax = 2      # s, p, d
    alpha0, ratio = 4.0, 2.5
    rydberg = 0      # --kbj N: N Kaufmann-Baumeister-Jungen functions per l instead
    ncore = 0        # --freeze N: fold N doubly occupied MOs into the core energy
    virtmax = None   # --virt-max X: drop virtual MOs at or above X hartree
    uncontract = False  # --uncontract: decontract the parent basis
    auglist = None   # --aug-shells 10,10,8: per-l augmentation counts, overriding --augment
    usebse = False   # --bse: load the parent basis from basis_set_exchange, with its ECP
    dropl = None     # --drop-l N: remove parent shells with l >= N
    atoms = []
    i = 0
    while i < len(argv):
        if argv[i] == "--basis":
            basis = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--augment":
            # Positive: even-tempered augmentation on its own (legacy, ill-conditioned).
            # Negative with --kbj: that many TIGHT continuum shells added to the KBJ set.
            aug = int(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--aug-lmax":
            auglmax = int(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--aug-alpha0":
            alpha0 = float(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--aug-ratio":
            ratio = float(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--kbj":
            rydberg = int(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--freeze":
            ncore = int(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--virt-max":
            virtmax = float(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--uncontract":
            uncontract = True
            i += 1
            continue
        if argv[i] == "--aug-shells":
            auglist = [int(x) for x in argv[i + 1].split(",")]
            i += 2
            continue
        if argv[i] == "--bse":
            usebse = True
            i += 1
            continue
        if argv[i] == "--drop-l":
            dropl = int(argv[i + 1])
            i += 2
            continue
        atoms.append(argv[i])
        i += 1
    if not atoms:
        atoms = ["Ne"]
    os.makedirs(OUT, exist_ok=True)

    for atom in atoms:
        label, vac = SYSTEMS[atom]
        bas = basis
        suffix = ""
        ecp = None
        if auglist is not None:
            # The supplementary material gives per-l counts that are NOT equal across l
            # (Ne/Mg 10s10p8d, Ar 4s3p, Kr 5s5p5d2f), so the augmentation cannot be
            # described by one count and an lmax.
            from pyscf import gto as _g
            if usebse:
                b, ecp = bse_basis(basis, atom, uncontract, dropl)
            else:
                b = _g.basis.load(basis, atom) if isinstance(basis, str) else basis
                if uncontract:
                    b = _g.uncontract(b)
                if dropl is not None:
                    b = [sh for sh in b if sh[0] < dropl]
            extra = []
            for l, n in enumerate(auglist):
                if rydberg > 0:
                    a, bb = {0: (0.584342, 0.424483), 1: (0.452663, 0.309805),
                             2: (0.382368, 0.251333), 3: (0.337804, 0.215739),
                             4: (0.306158, 0.191634)}[l]
                    for m in range(1, n + 1):
                        extra.append([l, [(1.0 / (2.0 * m)) ** 2 / (a * m + bb) ** 2, 1.0]])
                else:
                    for m in range(n):
                        extra.append([l, [alpha0 * ratio ** m, 1.0]])
            bas = {atom: b + extra}
            kind = "kbj" if rydberg > 0 else "et"
            suffix = "_" + kind + "".join(f"{n}{'spdfg'[l]}" for l, n in enumerate(auglist) if n)
            if uncontract:
                suffix = "_unc" + suffix
            if virtmax is not None:
                suffix += f"_v{int(virtmax)}"
            if ncore:
                suffix += f"_fc{ncore}"
        elif aug > 0:
            bas = {atom: augment(basis, atom, aug, auglmax, alpha0, ratio)}
            names = "".join(f"{aug}{'spdfg'[l]}" for l in range(auglmax + 1))
            suffix = f"_plus{names}"
        elif rydberg > 0:
            # KBJ (diffuse, near-threshold) PLUS an optional tight continuum set. Both are
            # needed and they do different jobs. KBJ alone leaves the pseudo-continuum
            # short: for Ne it tops out at 1033 eV against E_Phi = 867, so E_Phi sits at
            # 83% of the sampled range and 325 of 373 P states carry no coupling at all,
            # because diffuse functions have no amplitude in the core region where an
            # Auger matrix element is formed. The tight set supplies continuum-like
            # virtuals; starting it ABOVE the parent basis's valence exponents (alpha0 = 8,
            # ratio 3 rather than 4/2.5) is what keeps it linearly independent -- measured
            # on Ne, KBJ + 4s4p4d tight drops NO orbitals, reaches 2094 Eh and gives the
            # lowest RHF energy of any set tried, where alpha0 = 4 / ratio 2.5 drops one
            # orbital and RAISES the energy.
            from pyscf import gto as _g
            extra = kbj(atom, rydberg, auglmax)
            names = "".join(f"{rydberg}{'spdfg'[l]}" for l in range(auglmax + 1))
            suffix = f"_kbj{names}"
            if aug < 0:
                nt = -aug
                extra = extra + [[l, [alpha0 * ratio ** i, 1.0]]
                                 for l in range(auglmax + 1) for i in range(nt)]
                suffix += "".join(f"_c{nt}{'spdfg'[l]}" for l in range(auglmax + 1))
            bas = {atom: _g.basis.load(basis, atom) + extra}
        if ncore:
            suffix += f"_fc{ncore}"
        kw = {"ecp": {atom: ecp}} if ecp else {}
        mol = gto.M(atom=f"{atom} 0 0 0", basis=bas, verbose=0, symmetry="D2h",
                    spin=0, charge=0, **kw)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-11
        mf.run()
        if not mf.converged:
            raise RuntimeError(f"{atom}/{basis}: RHF did not converge")

        e = mf.mo_energy
        nocc = mol.nelectron // 2
        occ, vir = e[:nocc], e[nocc:]
        outer = occ[vac + 1:]
        if len(outer) < 2:
            outer = occ[vac:]
        lo = vir[0] - outer[-1] - outer[-2]
        hi = vir[-1] - outer[-1] - outer[-2]
        ephi = -occ[vac]

        tag = f"{atom.lower()}_{basis.replace('-', '').lower()}{suffix}"
        path = os.path.join(OUT, tag + ".fcidump")
        orbsym = gamess_orbsym(mol, mf.mo_coeff)
        if ncore or virtmax is not None:
            h1, eri, nact, nelec_act, ecore, keep = freeze_core(mf, ncore, virtmax)
            ksym = [orbsym[i] for i in keep]
            fcidump.from_integrals(path, h1, eri, nact, nelec_act, nuc=ecore,
                                   orbsym=ksym, tol=1e-12)
            vac -= ncore
            orbsym = ksym
        else:
            fcidump.from_scf(mf, path, tol=1e-12)
            orbsym = list(orbsym)
        rewrite_orbsym_streaming(path, orbsym)

        print(f"{atom} {label} / {basis}: E(RHF) = {mf.e_tot:.10f} Eh, "
              f"nao={mol.nao} nocc={nocc} nvir={len(vir)}")
        print(f"  Koopmans E_Phi = {ephi:.4f} Eh = {ephi*27.211386:.2f} eV; "
              f"2h1p span [{lo:.3f}, {hi:.3f}] Eh -> brackets E_Phi: {lo <= ephi <= hi}")
        print(f"  vacancy orbital {vac} (irrep {orbsym[vac]}), wrote {path}")

if __name__ == "__main__":
    main(sys.argv[1:])
