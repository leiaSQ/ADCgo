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


def resolve(nmo, nelec, active_text=None, frozen_core=None):
    """Resolve a selection into frozen-core + active MO columns.

    ``nmo``  total MOs; ``nelec`` total electrons (RHF, so ``nelec`` is even).
    Returns a :class:`Selection`. With no ``active_text`` and no ``frozen_core``
    the result is ``full`` (caller dumps the whole space via from_scf).
    """
    if active_text is None and not frozen_core:
        return Selection(True, [], [], 0, 0, 0)

    nocc = nelec // 2

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
