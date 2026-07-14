"""GAMESS-UK basis file -> pyscf basis dict.

theADCcode's front-end takes the basis as a GAMESS-UK basis file (`_basisset.dat`,
`cat`'d into GAMESS-UK in ../ADCanalysis/examples/DIP_h2o/dip.in). Its layout is:

    # comment
      S   H                    # shell header: <angular-momentum> <element>
          0.03282800        19.24060000    # primitive rows: <coeff>  <exponent>
          0.23120800         2.89920000
          0.81723800         0.65340000
      P   O
          0.01958000        35.18320000
          ...

pyscf's ``gto.basis.parse`` wants NWChem order: header ``<element> <L>`` and rows
``<exponent>  <coeff>``. So per shell we swap the header tokens and swap the two
number columns. This reproduces the manual transcription that
``scripts/gen_ref_fcidump.py`` did by hand for the DZP+Diffuse basis, for an
arbitrary GAMESS-UK basis file. GAMESS-UK uses cartesian GTOs, so callers should
build the molecule with ``cart=True`` for parity.
"""

from pyscf import gto

# Recognised angular-momentum letters (single-shell). SP/L compound shells (the
# GAMESS-UK spelling of a shared-exponent s+p shell, as in 6-31G) carry an extra
# coefficient column and are split into equivalent separate S and P shells on output.
_SHELL_LETTERS = set("SPDFGHI")
_COMPOUND = {"SP", "L"}


def _is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_gamess_basis(text):
    """Parse a GAMESS-UK basis string into ``{element: nwchem_text}``.

    Groups shells by element and emits, per element, an NWChem-format basis block
    (``gto.basis.parse``-ready). Raises ValueError on malformed input and
    NotImplementedError for SP/L compound shells (not present in the reference
    fixtures).
    """
    # element -> list of (shell_letter, [(exponent, coeff), ...])
    shells = {}
    order = []           # preserve first-seen element order for stable output
    cur = None           # (element, shell_letter, primitives-list)

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()

        # A primitive row is all-numeric; a header row is <letter> <element>.
        if all(_is_number(t) for t in tok):
            if cur is None:
                raise ValueError(f"primitive row before any shell header: {line!r}")
            # Plain shell: '<coeff> <exponent>'. SP/L compound shell: the exponent is
            # shared by the s and p parts, so the row is '<coeff-s> <exponent> <coeff-p>'.
            want = 3 if cur[1] in _COMPOUND else 2
            if len(tok) != want:
                cols = ("'<coeff-s> <exponent> <coeff-p>' (3 columns)" if want == 3
                        else "'<coeff> <exponent>' (2 columns)")
                raise ValueError(f"expected {cols}, got {line!r}")
            if want == 3:
                cur[2].append((float(tok[1]), float(tok[0]), float(tok[2])))
            else:
                cur[2].append((float(tok[1]), float(tok[0])))
            continue

        # Header row.
        if len(tok) != 2:
            raise ValueError(f"malformed shell header: {line!r}")
        shell, elem = tok[0].upper(), tok[1].capitalize()
        if shell not in _SHELL_LETTERS and shell not in _COMPOUND:
            raise ValueError(f"unknown shell type {tok[0]!r} in {line!r}")
        if elem not in shells:
            shells[elem] = []
            order.append(elem)
        cur = (elem, shell, [])
        shells[elem].append(cur)

    if not order:
        raise ValueError("no shells parsed from basis file")

    # Emit NWChem text per element: header "<elem> <L>", rows "<exponent> <coeff>".
    # An SP/L shell becomes two blocks (S and P) over the shared exponents.
    out = {}
    for elem in order:
        blocks = []
        for _, shell, prims in shells[elem]:
            if not prims:
                raise ValueError(f"shell {shell} on {elem} has no primitives")
            parts = ([("S", 1), ("P", 2)] if shell in _COMPOUND else [(shell, 1)])
            for letter, col in parts:
                rows = [f"{elem}    {letter}"]
                for prim in prims:
                    rows.append(f"    {prim[0]:.10f}   {prim[col]:.10f}")
                blocks.append("\n".join(rows))
        out[elem] = "\n".join(blocks)
    return out


def load_gamess_basis(path):
    """Read a GAMESS-UK basis file -> ``{element: parsed_basis}`` for ``gto.M``."""
    with open(path) as fh:
        text = fh.read()
    return {elem: gto.basis.parse(nwchem)
            for elem, nwchem in parse_gamess_basis(text).items()}


if __name__ == "__main__":  # quick manual check
    import sys
    b = load_gamess_basis(sys.argv[1])
    for elem, parsed in b.items():
        print(f"{elem}: {len(parsed)} shells")
