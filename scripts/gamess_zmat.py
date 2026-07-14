"""GAMESS-UK Z-matrix -> pyscf Cartesian geometry.

theADCcode's front-end takes geometry as a GAMESS-UK z-matrix file (`_zmatrix.dat`,
`cat`'d into GAMESS-UK in ../ADCanalysis/examples/DIP_h2o/dip.in). The format is:

    zmat angstroms          # unit line: angstroms | bohr | au
     o
     h   1 0.9440686        # literal bond/angle/dihedral values ...
     h   1 0.9440686   2 107.0715130
    end

or, with a symbolic `variables` block (../ADCanalysis/examples/DIP_FA/_zmatrix.dat):

    zmat angstroms
     c
     o   1 oc2
     o   1 oc3   2 oco3
     ...
    variables
    oc2   1.325472
    ...
    constants              # optional; also name value pairs, held fixed
    end

The atom rows are 1-based internal coordinates (symbol; ref bond; ref angle; ref
dihedral), which is exactly pyscf's inline z-matrix once the variable names are
substituted. We do NOT reimplement internal->Cartesian: after substitution the rows
go straight to ``pyscf.gto.mole.from_zmatrix``. Distances stay in the file's unit
(returned alongside so the caller sets ``gto.M(unit=...)``); angles are in degrees.
"""

import os

from pyscf.gto.mole import from_zmatrix

# GAMESS-UK unit keyword -> pyscf unit string. Distances only; angles are degrees.
_UNITS = {
    "angstrom": "Angstrom", "angstroms": "Angstrom", "angs": "Angstrom",
    "ang": "Angstrom", "a": "Angstrom",
    "bohr": "Bohr", "au": "Bohr", "a.u.": "Bohr", "atomic": "Bohr",
}


def _clean_lines(text):
    """Strip comments (# ...) and blank lines, returning lowercased-keyword-aware rows."""
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def parse_gamess_zmat(text):
    """Parse a GAMESS-UK z-matrix string.

    Returns ``(atom_rows, unit)`` where ``atom_rows`` is a list of pyscf inline
    z-matrix rows (variables substituted, numeric) and ``unit`` is "Angstrom" or
    "Bohr". Raises ValueError on malformed input.
    """
    lines = _clean_lines(text)
    if not lines:
        raise ValueError("empty z-matrix")

    # Header: "zmat <unit>". Tolerate a bare "zmat" (default angstrom) or a missing
    # header if the first token is an element symbol.
    idx = 0
    unit = "Angstrom"
    first = lines[0].split()
    if first[0].lower() == "zmat":
        if len(first) > 1:
            key = first[1].lower()
            if key not in _UNITS:
                raise ValueError(f"unknown z-matrix unit {first[1]!r}")
            unit = _UNITS[key]
        idx = 1

    atom_lines, var_lines = [], []
    section = "atoms"
    for line in lines[idx:]:
        low = line.lower()
        if low == "end":
            break
        if low in ("variables", "variable"):
            section = "vars"
            continue
        if low in ("constants", "constant"):
            section = "vars"  # constants are fixed-value params; treat as variables
            continue
        (atom_lines if section == "atoms" else var_lines).append(line)

    # Build the name -> value table from variables/constants rows.
    variables = {}
    for line in var_lines:
        tok = line.split()
        if len(tok) < 2:
            raise ValueError(f"malformed variable row: {line!r}")
        variables[tok[0].lower()] = float(tok[1])

    def resolve(token):
        """A z-matrix value: a literal float or a (optionally negated) variable name."""
        neg = token.startswith("-")
        name = token[1:] if neg else token
        try:
            val = float(token)
            return val
        except ValueError:
            pass
        if name.lower() not in variables:
            raise ValueError(f"undefined z-matrix variable {token!r}")
        val = variables[name.lower()]
        return -val if neg else val

    rows = []
    for line in atom_lines:
        tok = line.split()
        sym = tok[0]
        rest = tok[1:]
        if len(rest) % 2 != 0:
            raise ValueError(f"malformed z-matrix atom row: {line!r}")
        parts = [sym]
        for i in range(0, len(rest), 2):
            ref = rest[i]            # 1-based reference atom index (integer)
            val = resolve(rest[i + 1])
            parts.append(str(int(ref)))
            parts.append(repr(val))
        rows.append(" ".join(parts))
    if not rows:
        raise ValueError("z-matrix has no atoms")
    return rows, unit


def _looks_like_xyz(lines):
    """True if the first row is a bare integer atom count (standard .xyz)."""
    return bool(lines) and lines[0].split()[0].lstrip("+").isdigit()


# GAMESS-UK dummy centres: pure geometry scaffolding (they carry no charge and no
# basis functions). pyscf normalises a leading-X symbol to its ghost form, so `xx`
# comes back from from_zmatrix as `X-X`; match both spellings. `Xe` must not match.
_DUMMY_SYMBOLS = {"x", "xx", "q", "dummy", "x-x", "x-xx"}


def _is_dummy(sym):
    return sym.strip().lower() in _DUMMY_SYMBOLS


def read_geometry(path):
    """Read a geometry file -> ``(atom_spec, unit)`` for ``gto.M``.

    Supports GAMESS-UK z-matrix (default), ``.xyz`` (count + comment + rows), and a
    plain pyscf Cartesian atom list ("Sym x y z" per line). ``atom_spec`` is a list
    of ``[symbol, (x, y, z)]``; ``unit`` is the pyscf unit string.

    Dummy centres (``xx``/``x``/``q``) are dropped, but only *after* the z-matrix is
    converted to Cartesians, since the atom rows reference them by 1-based index.
    """
    with open(path) as fh:
        text = fh.read()
    lines = _clean_lines(text)
    if not lines:
        raise ValueError(f"empty geometry file: {path}")

    first = lines[0].split()
    is_zmat = first[0].lower() == "zmat"
    is_xyz = path.lower().endswith(".xyz") or _looks_like_xyz(lines)

    if is_zmat:
        rows, unit = parse_gamess_zmat(text)
        atoms = [[sym, tuple(float(x) for x in xyz)]
                 for sym, xyz in from_zmatrix("\n".join(rows))]
    else:
        # Cartesian: skip the count/comment header for .xyz; else every row is an atom.
        body = lines[2:] if is_xyz else lines
        atoms = []
        for line in body:
            tok = line.split()
            if len(tok) < 4:
                raise ValueError(f"malformed Cartesian atom row: {line!r}")
            atoms.append([tok[0], (float(tok[1]), float(tok[2]), float(tok[3]))])
        # .xyz is Angstrom by convention; a bare Cartesian list we also treat as such.
        unit = "Angstrom"

    atoms = [a for a in atoms if not _is_dummy(a[0])]
    if not atoms:
        raise ValueError(f"no atoms parsed from {path}")
    return atoms, unit


if __name__ == "__main__":  # quick manual check
    import sys
    atoms, unit = read_geometry(sys.argv[1])
    print(f"unit={unit}")
    for sym, xyz in atoms:
        print(f"{sym:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")
