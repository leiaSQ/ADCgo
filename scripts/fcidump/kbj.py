"""Kaufmann-Baumeister-Jungen (KBJ) continuum Gaussian exponents for ghost centres.

K. Kaufmann, W. Baumeister, M. Jungen, J. Phys. B 22, 2223 (1989) fit Gaussian
exponent sequences to Slater functions of fixed exponent zeta = 1 and found them well
represented by

    alpha(l, n) = 1 / (4 (a_l n + b_l)^2),     n = 1, 2, 3, ...

(n is an index, not a principal quantum number). The coefficients are not re-typed
here from memory. They are fitted to the published exponent table reprinted by
E. Coccia et al., "Gaussian continuum basis functions for calculating high-harmonic
generation spectra", arXiv:1602.07202, Table I (n = 1..8, l = 0..2), which is stored
verbatim below. Least squares on y = 1/(2 sqrt(alpha)) = a n + b reproduces all 24
entries within the table's own rounding (<= 4e-5 relative, against the 5.2e-5 that
6-digit rounding allows at alpha = 0.009615). Only l <= 2 is tabulated, so only l <= 2
is offered.

Basis spec grammar (used by dump_fcidump's &ghost_sites):

    kbj:<lmax>:<n1>-<n2>     uncontracted shells l = 0..lmax, n = n1..n2
"""

import math

# Coccia et al. arXiv:1602.07202, Table I: alpha(l, n) for n = 1..8.
PUBLISHED = {
    0: [0.245645, 0.098496, 0.052725, 0.032775, 0.022327, 0.016182, 0.012264, 0.009615],
    1: [0.430082, 0.169341, 0.089894, 0.055611, 0.037766, 0.027312, 0.020666, 0.016181],
    2: [0.622557, 0.242160, 0.127840, 0.078835, 0.053428, 0.038583, 0.029163, 0.022815],
}
TABLE_ROUNDING = 5.3e-5  # max relative error 6-digit rounding permits in PUBLISHED


def _fit(l):
    ys = [1.0 / (2.0 * math.sqrt(x)) for x in PUBLISHED[l]]
    ns = list(range(1, len(ys) + 1))
    nb = sum(ns) / len(ns)
    yb = sum(ys) / len(ys)
    a = sum((n - nb) * (y - yb) for n, y in zip(ns, ys)) / sum((n - nb) ** 2 for n in ns)
    return a, yb - a * nb


COEFF = {l: _fit(l) for l in PUBLISHED}


def exponent(l, n):
    if l not in COEFF:
        raise ValueError(f"KBJ coefficients are tabulated for l <= 2 only, not l = {l}")
    if n < 1:
        raise ValueError(f"KBJ index n must be >= 1, got {n}")
    a, b = COEFF[l]
    return 1.0 / (4.0 * (a * n + b) ** 2)


def parse_spec(spec):
    """'kbj:2:3-8' -> (lmax, n1, n2)."""
    parts = spec.split(":")
    if len(parts) != 3 or parts[0] != "kbj":
        raise ValueError(f"not a KBJ spec: {spec!r} (want kbj:<lmax>:<n1>-<n2>)")
    lmax = int(parts[1])
    n1, n2 = (int(x) for x in parts[2].split("-"))
    if n2 < n1:
        raise ValueError(f"KBJ range {parts[2]} runs backwards")
    return lmax, n1, n2


def basis(spec):
    """pyscf basis list for a KBJ spec: one uncontracted shell per (l, n)."""
    lmax, n1, n2 = parse_spec(spec)
    return [[l, [exponent(l, n), 1.0]] for l in range(lmax + 1)
            for n in range(n1, n2 + 1)]


def describe(spec):
    """Sidecar record: the generator parameters and the exponents they produced."""
    lmax, n1, n2 = parse_spec(spec)
    return {"spec": spec, "formula": "alpha = 1/(4 (a_l n + b_l)^2)",
            "source": "KBJ, J. Phys. B 22, 2223 (1989); coefficients fitted to "
                      "Coccia et al. arXiv:1602.07202 Table I",
            "coefficients": {str(l): list(COEFF[l]) for l in range(lmax + 1)},
            "exponents": {str(l): [exponent(l, n) for n in range(n1, n2 + 1)]
                          for l in range(lmax + 1)}}


def check():
    """Reproduce the published table within its rounding."""
    worst = 0.0
    for l, vals in PUBLISHED.items():
        for n, want in enumerate(vals, start=1):
            worst = max(worst, abs(exponent(l, n) - want) / want)
    if worst > TABLE_ROUNDING:
        raise AssertionError(f"KBJ fit misses the published table by {worst:.2e} relative")
    return worst


if __name__ == "__main__":
    print(f"max relative deviation from the published table: {check():.2e}")
    for l, (a, b) in COEFF.items():
        print(f"l={l}: a={a:.6f} b={b:.6f}")
