"""Localized orbital basis for net-charge Fano partitions (fano.ParseConfigRule).

The partition needs "a hole on atom A" and "a particle on atom B" to mean something, and
canonical MOs of a rare-gas cluster do not provide that: at 1.3 A two He 1s orbitals
combine to sigma_g/sigma_u. This module builds an orthonormal MO basis in three parts,

    [ occupied, localized ]  one He 1s per real atom, Pipek-Mezey with IAO charges
    [ compact virtuals    ]  per-atom projections of the atom's compact AOs onto the
                             virtual space, Loewdin-orthonormalized across atoms
    [ free virtuals       ]  the orthogonal complement in the virtual space; holds all
                             diffuse (aug) and ghost-centre content; Fock-canonicalized

Occupied and virtual SPACES are the RHF ones, so the occupied-virtual Fock block stays
zero; only rotations within each space are made. The basis is not canonical.

Cs (the molecular plane, z -> -z) is enforced explicitly: every centre lies in the xy
plane, so the reflection acts on each spherical AO by a sign (-1)^(power of z), and all
virtual processing is done per irrep. pyscf's point-group machinery is not used; it would
reorient the molecule and, on the linear chain, mix degenerate pi pairs across irreps.

"Compact" AOs of a real atom are the shells of the compact basis (cc-pVDZ by default)
that occur in its full basis, matched by angular momentum and exponent set; diffuse aug
shells and every ghost-centre shell are never compact.
"""

import json
import sys
from dataclasses import dataclass, field

import threads  # noqa: F401  (caps BLAS/OpenMP threads; must precede numpy)
import numpy as np
from pyscf import gto, lo, scf

A1, A2 = 0, 1  # Cs irreps: A' (even under z -> -z), A'' (odd)
IRREP_NAMES = ("A'", "A''")


@dataclass
class LocalBasis:
    """The rotated MO basis and its per-orbital bookkeeping."""

    C: np.ndarray             # nAO x nMO, columns [occ | compact | free]
    nocc: int
    ncompact: int
    nfree: int
    atom: list                # per MO: real-atom index, or -1 (free)
    kind: list                # per MO: "occ" | "compact" | "free"
    irrep: list               # per MO: 0 (A') or 1 (A'')
    roles: list               # real-atom names, e.g. ["He1","He2","He3"]
    real_atoms: list = field(default_factory=list)  # mol atom index of each role
    metrics: dict = field(default_factory=dict)

    def atom_index(self, role):
        """mol atom index of a real atom by role name."""
        return self.real_atoms[self.roles.index(role)]

    @property
    def nmo(self):
        return self.C.shape[1]

    def orbsym_gamess(self):
        """1-based ORBSYM labels (A' = 1, A'' = 2) for the FCIDUMP header."""
        return [i + 1 for i in self.irrep]


# ---------------------------------------------------------------------------
# molecule and SCF
# ---------------------------------------------------------------------------

def build_mol(geom, basis="aug-cc-pvdz", ghost_basis=None, charge=0, spin=0,
              verbose=0):
    """pyscf Mole for a geometry (atoms as (role, xyz), .ghosts): real atoms, then ghost sites.

    Real atoms are labelled He1..HeN in role order. `basis` is a pyscf basis name for all
    real atoms, or a dict role -> basis with an optional "default" entry (e.g. diffuse
    functions on T only). Ghost site n becomes a basis-only centre "X<n>" (pyscf gives
    X-numbered centres no nuclear charge) whose basis is ghost_basis[site label] (a pyscf
    basis list) if given, else built from the site's own specs (ghost_basis_list), so
    its AOs are attributable to that site.
    """
    atoms, bas = [], {}
    for n, (role, xyz) in enumerate(geom.atoms, start=1):
        label = f"He{n}"
        atoms.append([label, xyz])
        if isinstance(basis, dict):
            if role not in basis and "default" not in basis:
                raise ValueError(f"no basis for role {role} and no default")
            bas[label] = basis.get(role, basis.get("default"))
        else:
            bas[label] = basis
    for n, g in enumerate(geom.ghosts, start=1):
        # pyscf dummy centres are "X<digits>"; "X<letters>" would be read as the ghost
        # of an element named <letters>
        label = f"X{n}"
        if ghost_basis is not None and g.label in ghost_basis:
            bas[label] = ghost_basis[g.label]
        else:
            bas[label] = ghost_basis_list(g.basis)
        atoms.append([label, g.xyz])
    return gto.M(atom=atoms, basis=bas, charge=charge, spin=spin, unit="Angstrom",
                 symmetry=False, verbose=verbose)


def run_rhf(mol, conv_tol=1e-12, conv_tol_grad=1e-9, max_cycle=200, lindep=None):
    """Tight RHF. With lindep set, canonical orthogonalization drops AO combinations
    whose overlap eigenvalue is below it (diffuse KBJ/ghost sets); the number dropped
    is nao - nmo of the result. 1e-6 (pyscf's default) is the smallest that works for
    He clusters with KBJ ghosts: at 1e-7 the IAO Pipek-Mezey step produced NaNs."""
    if lindep is not None:
        # pyscf's built-in canonical orthogonalization (remove_linear_dep_ is
        # deprecated): overlap eigenvalues below the threshold are dropped
        from pyscf.scf import hf as scf_hf
        scf_hf.remove_overlap_zero_eigenvalue = True
        scf_hf.overlap_zero_eigenvalue_threshold = lindep
    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max_cycle
    mf.kernel()
    if not mf.converged:
        raise SystemExit(f"RHF did not converge (E={mf.e_tot:.10f})")
    return mf


# ---------------------------------------------------------------------------
# AO bookkeeping
# ---------------------------------------------------------------------------

def _zpower(ang):
    """Power of z in the angular part of a pyscf real-spherical AO label.

    `ang` is the 4th field of mol.ao_labels(fmt=False): '' for s, 'x'/'y'/'z' for p,
    'xy','yz','z^2','xz','x2-y2' for d, 'y^3','xyz','yz^2','z^3','xz^2','zx^2','x^3'
    for f. The real solid harmonic behind each label has definite z-parity equal to
    the parity of this power (e.g. z^2 = 3z^2 - r^2 is even, yz^2 is even, z^3 odd).
    """
    n, j = 0, 0
    while j < len(ang):
        if ang[j] == "z":
            if ang[j + 1:j + 2] == "^":
                k = j + 2
                while k < len(ang) and ang[k].isdigit():
                    k += 1
                n += int(ang[j + 2:k])
                j = k
                continue
            n += 1
        j += 1
    return n


def reflection_signs(mol):
    """Sign of each AO under z -> -z (all centres must lie in the xy plane)."""
    coords = mol.atom_coords()
    if np.max(np.abs(coords[:, 2])) > 1e-10:
        raise ValueError("Cs handling needs every centre in the xy plane")
    if mol.cart:
        raise ValueError("Cs handling assumes spherical AOs")
    R = np.array([(-1.0) ** _zpower(lab[3]) for lab in mol.ao_labels(fmt=False)])
    # Independent check: evaluate every AO at generic points and at their mirror
    # images. phi(x, y, -z) = R phi(x, y, z) must hold AO by AO. (An overlap-commutation
    # test would not catch an all-even misparse; this does.)
    rng = np.random.default_rng(12345)
    pts = rng.normal(scale=1.5, size=(16, 3)) + mol.atom_coords().mean(axis=0)
    mir = pts * np.array([1.0, 1.0, -1.0])
    a = mol.eval_gto("GTOval_sph", pts)
    b = mol.eval_gto("GTOval_sph", mir)
    err = np.abs(b - a * R[None, :]).max() / max(np.abs(a).max(), 1e-300)
    if err > 1e-10:
        raise ValueError(f"AO parity under z -> -z disagrees with the labels ({err:.2e})")
    return R


def real_atoms(mol):
    """Atoms with a nucleus, in mol order. Ghost centres (pyscf "X..." dummies and the
    "GHOST-" counterpoise centres of dump_fcidump) carry zero nuclear charge."""
    return [a for a in range(mol.natm) if mol.atom_charge(a) > 0]


def ghost_flags(mol):
    return [bool(mol.atom_charge(a) == 0) for a in range(mol.natm)]


def ghost_basis_list(specs):
    """pyscf basis list for a ghost site from its specs: "kbj:<lmax>:<n1>-<n2>" (see
    kbj.py) or "<basis>@<element>" (a named basis taken from an element)."""
    import kbj
    out = []
    for spec in specs:
        if spec.startswith("kbj:"):
            out += kbj.basis(spec)
        elif "@" in spec:
            name, el = spec.split("@", 1)
            out += gto.load(name, el)
        else:
            raise ValueError(f"ghost basis spec {spec!r}: want kbj:... or <basis>@<element>")
    # Merged sites (e.g. the T and centre-of-mass sets on the symmetric chain) can name
    # the same uncontracted shell twice (kbj:2:1-8 and kbj:1:6-10 share n = 6..8),
    # which makes the overlap exactly singular. Keep each (l, exponents, coefficients)
    # once.
    seen, uniq = set(), []
    for sh in out:
        key = (sh[0], tuple((round(p[0], 12), *p[1:]) for p in sh[1:]))
        if key not in seen:
            seen.add(key)
            uniq.append(sh)
    return uniq


def compact_ao_mask(mol, compact_basis="cc-pvdz"):
    """True for AOs of real atoms whose shell occurs in the compact basis."""
    mask = np.zeros(mol.nao, dtype=bool)
    ref = {}
    for ib in range(mol.nbas):
        a = mol.bas_atom(ib)
        if mol.atom_charge(a) == 0:
            continue
        sym = mol.atom_pure_symbol(a)
        if sym not in ref:
            ref[sym] = [(sh[0], tuple(sorted(p[0] for p in sh[1:])))
                        for sh in gto.load(compact_basis, sym)]
        l = mol.bas_angular(ib)
        exps = tuple(sorted(mol.bas_exp(ib)))
        if (l, exps) in ref[sym]:
            p0, p1 = mol.ao_loc_nr()[ib], mol.ao_loc_nr()[ib + 1]
            mask[p0:p1] = True
    return mask


def ao_atoms(mol):
    out = np.zeros(mol.nao, dtype=int)
    for ib in range(mol.nbas):
        p0, p1 = mol.ao_loc_nr()[ib], mol.ao_loc_nr()[ib + 1]
        out[p0:p1] = mol.bas_atom(ib)
    return out


# ---------------------------------------------------------------------------
# linear algebra helpers (S-metric)
# ---------------------------------------------------------------------------

def _orth(X, S, thresh=1e-10):
    """Canonical orthonormalization of the columns of X in the S metric."""
    if X.shape[1] == 0:
        return X
    M = X.T @ S @ X
    w, U = np.linalg.eigh(M)
    keep = w > thresh * max(w.max(), 1.0)
    return X @ (U[:, keep] / np.sqrt(w[keep]))


def _loewdin(X, S):
    """Symmetric orthonormalization (keeps each column as close as possible)."""
    M = X.T @ S @ X
    w, U = np.linalg.eigh(M)
    if w.min() <= 0:
        raise ValueError(f"Loewdin on a singular set (min eig {w.min():.3e})")
    return X @ (U @ np.diag(w ** -0.5) @ U.T), float(w.min())


def _fix_sign(C):
    """Deterministic phase: the largest-|c| AO coefficient of each column positive."""
    C = C.copy()
    for k in range(C.shape[1]):
        i = np.argmax(np.abs(C[:, k]))
        if C[i, k] < 0:
            C[:, k] = -C[:, k]
    return C


def _irrep_of(C, S, R):
    """Irrep of each column (0 A', 1 A''); raises if a column is not Cs-pure."""
    out = []
    for k in range(C.shape[1]):
        c = C[:, k]
        x = c @ S @ (R * c)
        if abs(abs(x) - 1) > 1e-6:
            raise ValueError(f"orbital {k} is not Cs-pure (<phi|sigma|phi> = {x:.6f})")
        out.append(A1 if x > 0 else A2)
    return out


# ---------------------------------------------------------------------------
# the three blocks
# ---------------------------------------------------------------------------

def _iao_charges(mol, S, occ):
    """Orthonormal IAOs of the occupied space and the real atom owning each IAO."""
    reals = real_atoms(mol)
    iao = lo.iao.iao(mol, occ)
    iao = lo.orth.vec_lowdin(iao, S)
    # pyscf's IAO reference molecule drops ghost centres; its atom k is reals[k]
    ref = lo.iao.reference_mol(mol)
    iao_atom = np.zeros(iao.shape[1], dtype=int)
    for ib in range(ref.nbas):
        p0, p1 = ref.ao_loc_nr()[ib], ref.ao_loc_nr()[ib + 1]
        iao_atom[p0:p1] = reals[ref.bas_atom(ib)]
    return iao, iao_atom


def pipek_mezey(C, S, iao, iao_atom, tol=1e-12, max_sweeps=500):
    """Pipek-Mezey localization of the columns of C with IAO atomic charges.

    Maximizes sum_i sum_A Q_A(i,i)^2 with Q_A(i,j) = sum_{mu in A} c_i,mu c_j,mu in the
    orthonormal IAO basis, by 2x2 Jacobi rotations (Pipek and Mezey, J. Chem. Phys. 90,
    4916 (1989), Eqs. 25-27). Written out instead of pyscf.lo.PM, whose CIAH optimizer
    returned NaNs on He clusters with basis-only ghost centres. Only the real atoms enter the
    charges, and the sweep order is fixed, so the result is deterministic.
    """
    X = iao.T @ S @ C           # occupied orbitals in the IAO basis (exact span)
    atoms = sorted(set(iao_atom.tolist()))
    masks = [iao_atom == a for a in atoms]
    n = X.shape[1]
    for _ in range(max_sweeps):
        biggest = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                A = B = 0.0
                for m in masks:
                    qii = X[m, i] @ X[m, i]
                    qjj = X[m, j] @ X[m, j]
                    qij = X[m, i] @ X[m, j]
                    A += qij * qij - 0.25 * (qii - qjj) ** 2
                    B += qij * (qii - qjj)
                if abs(A) < 1e-300 and abs(B) < 1e-300:
                    continue
                g = 0.25 * np.arctan2(B, -A)
                if abs(g) < tol:
                    continue
                biggest = max(biggest, abs(g))
                c, s_ = np.cos(g), np.sin(g)
                xi, xj = X[:, i].copy(), X[:, j].copy()
                X[:, i], X[:, j] = c * xi + s_ * xj, -s_ * xi + c * xj
                ci, cj = C[:, i].copy(), C[:, j].copy()
                C[:, i], C[:, j] = c * ci + s_ * cj, -s_ * ci + c * cj
        if biggest < tol:
            return C
    raise RuntimeError(f"Pipek-Mezey did not converge in {max_sweeps} sweeps")


def localize_occupied(mol, mf, min_pop=0.99, one_per_atom=True):
    """Pipek-Mezey (IAO charges) occupied orbitals, ordered by owning atom.

    With one_per_atom (e.g. He clusters) every real atom must own exactly one
    occupied orbital, and the localization starts from each atom's most compact s AO
    projected onto the occupied space and Loewdin-orthonormalized. On a homonuclear pair
    the canonical sigma_g/sigma_u are a stationary point of the PM functional; a
    symmetric start stays on it (He2 at 0.70 A came back with populations 0.5/0.5 from
    pyscf's 'atomic' guess). This start is already localized and deterministic, and PM
    only refines it. Without one_per_atom the canonical orbitals are the start.

    Returns (C_occ, owning mol atom per orbital, IAO population on the owner).
    """
    occ = mf.mo_coeff[:, mf.mo_occ > 0]
    S = mf.get_ovlp()
    reals = real_atoms(mol)
    iao, iao_atom = _iao_charges(mol, S, occ)
    if one_per_atom:
        if occ.shape[1] != len(reals):
            raise ValueError(f"{occ.shape[1]} occupied orbitals for {len(reals)} real atoms "
                             "(one_per_atom)")
        seeds = []
        for a in reals:
            best = None
            for ib in range(mol.nbas):
                if mol.bas_atom(ib) == a and mol.bas_angular(ib) == 0:
                    e = float(np.max(mol.bas_exp(ib)))
                    if best is None or e > best[0]:
                        best = (e, mol.ao_loc_nr()[ib])
            chi = np.zeros(mol.nao)
            chi[best[1]] = 1.0
            seeds.append(occ @ (occ.T @ S @ chi))
        start, _ = _loewdin(np.array(seeds).T, S)
    else:
        start = occ.copy()
    Cl = pipek_mezey(start.copy(), S, iao, iao_atom)
    pops = np.zeros((Cl.shape[1], mol.natm))
    for k in range(Cl.shape[1]):
        c = iao.T @ S @ Cl[:, k]
        for a in range(mol.natm):
            pops[k, a] = np.sum(c[iao_atom == a] ** 2)
    owner = pops.argmax(axis=1)
    if one_per_atom and sorted(owner.tolist()) != reals:
        raise ValueError(f"localized occupied orbitals do not map one-to-one onto the real "
                         f"atoms {reals}: owners {owner.tolist()}, pops "
                         f"{np.round(pops.max(axis=1), 5).tolist()}")
    order = np.argsort(owner, kind="stable")
    Cl = _fix_sign(Cl[:, order])
    own = owner[order].tolist()
    pop = pops[order, owner[order]]
    if pop.min() < min_pop:
        raise ValueError(f"occupied localization weak: min IAO population {pop.min():.5f}")
    return Cl, own, pop


def split_virtuals(mol, mf, compact_mask, R, thresh=0.02):
    """Compact (per atom, per irrep) and free virtuals.

    For atom A and irrep g, the compact AOs of A with parity g are projected onto the
    virtual space; the eigenvectors of their overlap with eigenvalue > thresh are kept
    (the dropped direction is the part that lived in the occupied space). On a He
    cluster at 1.3 A spacing the spectrum has a wide gap -- dropped <= 1.5e-3, kept >=
    0.35 -- and the default 0.02 sits at its geometric middle, so a 2x threshold change
    (0.01 and 0.04) stays inside it. The union over
    atoms is Loewdin-orthonormalized per irrep, so each vector stays nearest its atom.
    The free virtuals are the orthonormal complement, canonicalized with the Fock matrix
    per irrep.
    """
    S = mf.get_ovlp()
    F = mf.get_fock()
    Cv = mf.mo_coeff[:, mf.mo_occ == 0]
    Pv = Cv @ Cv.T @ S                     # AO-coefficient projector onto the virtuals
    aoat = ao_atoms(mol)
    metrics = {"compact_thresh": thresh, "compact_eig_kept_min": {}, "compact_eig_drop_max": {},
               "compact_loewdin_min_eig": {}}
    comp_cols, comp_atom, comp_irrep = [], [], []
    free_cols, free_irrep = [], []
    for g in (A1, A2):
        par = 1.0 if g == A1 else -1.0
        # virtual space of irrep g
        Vg = _orth((Cv + par * (R[:, None] * Cv)) / 2, S, 1e-8)
        Y, yat = [], []
        for a in real_atoms(mol):
            sel = np.where(compact_mask & (aoat == a) & (R == par))[0]
            if sel.size == 0:
                continue
            X = Pv[:, sel]
            M = X.T @ S @ X
            w, U = np.linalg.eigh(M)
            keep = w > thresh
            key = f"{a}{IRREP_NAMES[g]}"
            metrics["compact_eig_kept_min"][key] = float(w[keep].min()) if keep.any() else None
            metrics["compact_eig_drop_max"][key] = float(w[~keep].max()) if (~keep).any() else None
            for k in np.where(keep)[0]:
                Y.append(X @ U[:, k] / np.sqrt(w[k]))
                yat.append(a)
        if Y:
            Yg, mineig = _loewdin(np.array(Y).T, S)
            metrics["compact_loewdin_min_eig"][IRREP_NAMES[g]] = mineig
            comp_cols.append(Yg)
            comp_atom += yat
            comp_irrep += [g] * len(yat)
        else:
            Yg = np.zeros((mol.nao, 0))
        # free = Vg minus span(Yg)
        Q = Vg - Yg @ (Yg.T @ S @ Vg)
        Fg = _orth(Q, S, 1e-8)
        if Fg.shape[1] != Vg.shape[1] - Yg.shape[1]:
            raise ValueError(f"free space of {IRREP_NAMES[g]} has {Fg.shape[1]} vectors, "
                             f"want {Vg.shape[1]} - {Yg.shape[1]}")
        if Fg.shape[1]:
            e, U = np.linalg.eigh(Fg.T @ F @ Fg)
            Fg = Fg @ U
        free_cols.append(Fg)
        free_irrep += [g] * Fg.shape[1]
    Ccomp = _fix_sign(np.hstack(comp_cols)) if comp_cols else np.zeros((mol.nao, 0))
    Cfree = _fix_sign(np.hstack(free_cols))
    # order compact by (atom, irrep) for readability; free stays per-irrep Fock order
    order = sorted(range(len(comp_atom)), key=lambda k: (comp_atom[k], comp_irrep[k], k))
    Ccomp = Ccomp[:, order]
    comp_atom = [comp_atom[k] for k in order]
    comp_irrep = [comp_irrep[k] for k in order]
    return Ccomp, comp_atom, comp_irrep, Cfree, free_irrep, metrics


def build(mol, mf, roles, compact_basis="cc-pvdz", thresh=0.02, min_pop=0.99,
          one_per_atom=True):
    """The full LocalBasis for a converged RHF. roles names the real atoms in mol
    order (ghost centres excluded)."""
    reals = real_atoms(mol)
    if len(roles) != len(reals):
        raise ValueError(f"{len(roles)} role names for {len(reals)} real atoms")
    S = mf.get_ovlp()
    R = reflection_signs(mol)
    Co, oat, opop = localize_occupied(mol, mf, min_pop, one_per_atom)
    cmask = compact_ao_mask(mol, compact_basis)
    Cc, cat, cirr, Cf, firr, met = split_virtuals(mol, mf, cmask, R, thresh)
    C = np.hstack([Co, Cc, Cf])
    err = np.abs(C.T @ S @ C - np.eye(C.shape[1])).max()
    if err > 1e-9:
        raise ValueError(f"rotated basis not orthonormal: {err:.2e}")
    if C.shape[1] != mf.mo_coeff.shape[1]:
        raise ValueError(f"rotated basis has {C.shape[1]} MOs, RHF has "
                         f"{mf.mo_coeff.shape[1]}")
    irr = _irrep_of(Co, S, R) + cirr + firr
    lb = LocalBasis(C=C, nocc=Co.shape[1], ncompact=Cc.shape[1], nfree=Cf.shape[1],
                    atom=list(oat) + list(cat) + [-1] * Cf.shape[1],
                    kind=["occ"] * Co.shape[1] + ["compact"] * Cc.shape[1]
                    + ["free"] * Cf.shape[1],
                    irrep=irr, roles=list(roles), real_atoms=reals)
    met["occ_iao_pop_min"] = float(opop.min())
    met["occ_iao_pop"] = [float(x) for x in opop]
    met["orthonormality_err"] = float(err)
    met["n_ao"] = int(mol.nao)
    met["n_mo"] = int(C.shape[1])
    met["n_dropped_lindep"] = int(mol.nao - C.shape[1])
    # Occupied weight carried by the ghost functions: the part of each occupied orbital
    # OUTSIDE the span of the real atoms' AOs, 1 - ||P_real phi||^2. (A Loewdin
    # population on ghost AOs is not usable here: the T ghost sits on T's nucleus, and
    # symmetric orthogonalization splits T's own 1s across the concentric ghost shells,
    # reporting 24% "ghost population" for a perfectly atomic 1s.)
    #
    # A ghost centre that sits ON a real nucleus (the T Rydberg/continuum set) is an
    # augmentation of that atom's basis, and any concentric shell improves the
    # three-primitive aug-cc-pVDZ 1s at the 1e-4 level; that is reported separately
    # (occ_concentric_ghost_weight) and not held to the gate. The gate quantity,
    # occ_ghost_pop_max, is the weight on OFF-nuclear ghost centres: an electron must
    # not live where there is no nucleus.
    flags = ghost_flags(mol)
    xyz = mol.atom_coords()
    reals_ = real_atoms(mol)
    concentric = [a for a, g in enumerate(flags) if g and
                  min(np.linalg.norm(xyz[a] - xyz[r]) for r in reals_) < 1e-6]
    offatom = [a for a, g in enumerate(flags) if g and a not in concentric]

    def weight_outside(excluded):
        if not excluded:
            return 0.0
        keep = np.where(~np.isin(ao_atoms(mol), excluded))[0]
        Skk = S[np.ix_(keep, keep)]
        Skc = S[keep, :] @ Co
        inside = np.einsum("ik,ik->k", Skc, np.linalg.solve(Skk, Skc))
        return float(np.max(1.0 - inside))

    met["occ_ghost_pop_max"] = weight_outside(offatom)
    met["occ_concentric_ghost_weight"] = weight_outside(concentric + offatom) \
        - met["occ_ghost_pop_max"] if concentric else 0.0
    met["ghost_centres_concentric"] = len(concentric)
    met["ghost_centres_offatom"] = len(offatom)
    lb.metrics = met
    return lb


def sidecar_labels(mol, lb, extra=None):
    """The mo-sidecar orbital-label group (internal/adc/mo readLabels)."""
    doc = {"orb_kind": list(lb.kind), "orb_atom": [int(a) for a in lb.atom],
           "ghost_atoms": ghost_flags(mol), "canonical": False,
           "orbital_metrics": lb.metrics}
    if extra:
        doc.update(extra)
    return doc


def mo_integrals(mol, mf, C):
    """(h1, eri(4-fold packed), e_core) in the basis C."""
    from pyscf import ao2mo
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.kernel(mol, C, compact=True)
    return h1, eri, mol.energy_nuc()
