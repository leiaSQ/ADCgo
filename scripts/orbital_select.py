"""Frozen-core + active/virtual orbital selection (GAMESS-UK CORE/ACTIVE).

theADCcode selects the correlated orbital space in the GAMESS-UK integral
transformation, e.g. ``CORE 2 END`` + ``ACTIVE 1 3 to 18 END``
(../ADC/libphis/guk/test/h2o.in) or ``active 2 to 30``
(../ADC/regression_h2o/scf_adc.in). This module reproduces that selection so the
FCIDUMP carries exactly the correlated MOs, with the frozen core folded into the
core energy + one-electron term downstream (via pyscf CASCI).

Semantics (well-defined; documented in the plan):
  * ``active <list>``   -- 1-based MOs written to the FCIDUMP (the correlated space).
                           A GAMESS token list: individual indices and ``A to B``
                           ranges, in order, may skip MOs (drops high/selected
                           virtuals). e.g. "2 to 30", "1 3 to 18", "2 5 to 20".
  * ``frozen-core N``   -- freeze the N lowest MOs into the effective core (not in
                           the FCIDUMP). Must be disjoint from ``active``.
  * ``frozen-list <list>`` -- freeze an *explicit* set of 1-based MOs (need not be the
                           lowest), mirroring theADCcode's ``CORE 2 4 END``. Needed for
                           CVS on a non-lowest core, e.g. pyridine's N K-edge freezes the
                           five C 1s cores (MOs 2..6) while keeping the *lowest* MO (N 1s)
                           active as the CVS core hole. When ``active`` is omitted, every
                           non-frozen MO is correlated, in ascending order.
  * A bare contiguous ``active lo to hi`` with no explicit ``frozen-core`` implies
    ``frozen-core = lo-1`` (so "2 to 30" freezes MO 1 -> the validated DZP case).
"""

from collections import namedtuple

# full:  no selection -> caller uses fcidump.from_scf on the whole MO space.
# core0/active0: 0-based MO column indices (frozen core, then correlated active).
Selection = namedtuple(
    "Selection", ["full", "core0", "active0", "ncore", "ncas", "nelecas"])


def parse_index_list(text):
    """Expand a GAMESS orbital list -> ordered list of 1-based ints.

    Supports individual indices and ``A to B`` inclusive ranges, e.g.
    "2 to 30" -> [2..30]; "1 3 to 18" -> [1, 3, 4, ..., 18]. Order is preserved.
    """
    tok = text.replace(",", " ").split()
    out = []
    i = 0
    while i < len(tok):
        if i + 2 < len(tok) and tok[i + 1].lower() == "to":
            lo, hi = int(tok[i]), int(tok[i + 2])
            if hi < lo:
                raise ValueError(f"reversed range '{tok[i]} to {tok[i + 2]}'")
            out.extend(range(lo, hi + 1))
            i += 3
        else:
            if tok[i].lower() == "to":
                raise ValueError(f"malformed range near 'to' in {text!r}")
            out.append(int(tok[i]))
            i += 1
    if not out:
        raise ValueError(f"empty orbital list: {text!r}")
    if any(x < 1 for x in out):
        raise ValueError(f"orbital indices must be 1-based positive: {text!r}")
    return out


def _is_contiguous(idx):
    return all(b - a == 1 for a, b in zip(idx, idx[1:]))


def resolve(nmo, nelec, active_text=None, frozen_core=None, frozen_list=None):
    """Resolve a selection into frozen-core + active MO columns.

    ``nmo``  total MOs; ``nelec`` total electrons (RHF, so ``nelec`` is even).
    ``frozen_list`` an explicit list of 1-based MOs to freeze (need not be the lowest;
    mutually exclusive with ``frozen_core``). Returns a :class:`Selection`. With no
    ``active_text``, ``frozen_core`` nor ``frozen_list`` the result is ``full`` (caller
    dumps the whole space via from_scf).
    """
    if active_text is None and not frozen_core and not frozen_list:
        return Selection(True, [], [], 0, 0, 0)

    nocc = nelec // 2

    # Explicit frozen list: an arbitrary (not necessarily lowest) core set. The active
    # space is either an explicit `active` list or every remaining MO in ascending order.
    # The frozen columns keep their given order; CASCI folds them via mc.ncore regardless
    # of energy ordering, so freezing a non-lowest block (the C 1s above N 1s) is valid.
    if frozen_list is not None:
        if frozen_core:
            raise ValueError("give either frozen-core or frozen-list, not both")
        core1 = parse_index_list(frozen_list) if isinstance(frozen_list, str) \
            else list(frozen_list)
        core0 = [i - 1 for i in core1]
        if len(set(core0)) != len(core0):
            raise ValueError(f"frozen-list has duplicate MOs: {core1}")
        for i in core0:
            if not 0 <= i < nmo:
                raise ValueError(f"frozen MO {i + 1} out of range 1..{nmo}")
        if active_text is not None:
            active0 = [i - 1 for i in parse_index_list(active_text)]
        else:
            coreset = set(core0)
            active0 = [i for i in range(nmo) if i not in coreset]
        ncore = len(core0)
        overlap = sorted(i + 1 for i in set(core0) & set(active0))
        if overlap:
            raise ValueError(f"frozen-list and active overlap at MO(s) {overlap}")
        dropped_occ = sorted(i + 1 for i in range(nocc)
                             if i not in set(core0) and i not in set(active0))
        if dropped_occ:
            raise ValueError(
                f"occupied MO(s) {dropped_occ} are neither frozen nor active")
        ncas = len(active0)
        nelecas = nelec - 2 * ncore
        if nelecas < 0 or nelecas > 2 * ncas:
            raise ValueError(
                f"active electron count {nelecas} inconsistent with {ncas} active MOs")
        return Selection(False, core0, active0, ncore, ncas, nelecas)

    if active_text is not None:
        active1 = parse_index_list(active_text)
        # Imply frozen core from a bare contiguous range starting above MO 1.
        if frozen_core is None and _is_contiguous(active1) and active1[0] > 1:
            frozen_core = active1[0] - 1
    else:
        # frozen-core only: correlate every non-core MO.
        active1 = list(range(frozen_core + 1, nmo + 1))
    ncore = int(frozen_core or 0)

    active0 = [i - 1 for i in active1]
    core0 = list(range(ncore))

    # Validation.
    for i in active0:
        if not 0 <= i < nmo:
            raise ValueError(f"active MO {i + 1} out of range 1..{nmo}")
    if len(set(active0)) != len(active0):
        raise ValueError("active list has duplicate MOs")
    overlap = sorted(i + 1 for i in set(core0) & set(active0))
    if overlap:
        raise ValueError(
            f"frozen-core and active overlap at MO(s) {overlap}; the active list "
            "must not include the frozen-core MOs")
    if ncore > nocc:
        raise ValueError(f"frozen-core {ncore} exceeds occupied MO count {nocc}")
    # Every occupied MO must be accounted for (frozen or active); otherwise the
    # electron count for the active space is ill-defined.
    dropped_occ = sorted(i + 1 for i in range(nocc)
                         if i not in set(core0) and i not in set(active0))
    if dropped_occ:
        raise ValueError(
            f"occupied MO(s) {dropped_occ} are neither frozen-core nor active; "
            "select them as active or freeze them")

    ncas = len(active0)
    nelecas = nelec - 2 * ncore
    if nelecas < 0 or nelecas > 2 * ncas:
        raise ValueError(
            f"active electron count {nelecas} inconsistent with {ncas} active MOs")
    return Selection(False, core0, active0, ncore, ncas, nelecas)
