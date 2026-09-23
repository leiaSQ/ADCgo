#!/usr/bin/env python3
"""Compare two spectrum JSONs state by state (ADCgo vs theADCcode).

Both files use the ADCanalysis spectrum schema, so this works on any pair of
runs of the same system: an ADCgo `-spectrum` document against an ADCanalysis
document built from theADCcode's adcdip*.out, or two ADCgo runs against each
other. States are matched within (irrep, spin) by nearest energy, since the two
codes may keep different numbers of weak satellites above their pole-strength
cut-offs and a positional match would slip.

    scripts/analysis/compare_spectra.py plots/h2o_dip_adcgo.spec.json \
                               plots/h2o_dip_theadccode.spec.json
    scripts/analysis/compare_spectra.py ... --table 10 --irreps a_1,a_2,b_1,b_2

--table N emits the N lowest matched states as a LaTeX tabular (deviations in
units of 1e-6 eV and 1e-3 percentage points, which is where they live once the
integrals are matched); --irreps gives the point group's labels in the code's
1-based irrep order, e.g. C2v = a_1,a_2,b_1,b_2.
"""
import argparse
import json


def states(path):
    """One entry per state: the spectrum lists one line per decay channel."""
    out = {}
    for line in json.load(open(path))["lines"]:
        out[line["state_ref"]] = (
            line["energy"], line["ps_percent"], line["irrep"], line["spin"])
    return list(out.values())


def match(a, t, tol):
    """Pair each reference state with its nearest unused counterpart in a."""
    rows, unmatched = [], []
    sectors = sorted({(v[2], v[3]) for v in a} | {(v[2], v[3]) for v in t})
    for irrep, spin in sectors:
        ea = sorted(v for v in a if v[2] == irrep and v[3] == spin)
        et = sorted(v for v in t if v[2] == irrep and v[3] == spin)
        used = set()
        for x in et:
            free = [(abs(y[0] - x[0]), i)
                    for i, y in enumerate(ea) if i not in used]
            if free and min(free)[0] < tol:
                _, i = min(free)
                used.add(i)
                rows.append((x[0], ea[i][0], x[1], ea[i][1], irrep, spin))
            else:
                unmatched.append(("reference", x[0], x[1]))
        unmatched += [("test", y[0], y[1])
                      for i, y in enumerate(ea) if i not in used]
    rows.sort()
    return rows, unmatched


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test", help="spectrum JSON under test (e.g. ADCgo)")
    p.add_argument("reference", help="reference spectrum JSON (e.g. theADCcode)")
    p.add_argument("--tol", type=float, default=0.5,
                   help="max energy gap (eV) still counted as the same state")
    p.add_argument("--table", type=int, default=0,
                   help="also emit the N lowest states as a LaTeX tabular")
    p.add_argument("--irreps", default="",
                   help="comma-separated irrep labels in 1-based code order")
    args = p.parse_args()

    rows, unmatched = match(states(args.test), states(args.reference), args.tol)
    if not rows:
        raise SystemExit("no states matched")

    print(f"matched {len(rows)} states, {len(unmatched)} unmatched")
    windows = [10, 20, 30, 50, len(rows)]
    print(f"{'lowest':>8}  {'up to':>9}  {'max |dE|':>10}  {'max |dPS|':>10}")
    for n in windows:
        if n > len(rows):
            continue
        sub = rows[:n]
        print("%8d  %7.2f eV  %8.2e eV  %8.2e %%" % (
            n, sub[-1][0],
            max(abs(r[1] - r[0]) for r in sub),
            max(abs(r[3] - r[2]) for r in sub)))

    if not args.table:
        return

    labels = [s.strip() for s in args.irreps.split(",") if s.strip()]
    print("\n% deviations: dE in 1e-6 eV, dP in 1e-3 percentage points")
    for n, (Et, Ea, Pt, Pa, irrep, spin) in enumerate(rows[:args.table], 1):
        sym = labels[irrep - 1] if 0 < irrep <= len(labels) else str(irrep)
        print("%2d & $^{%d}%s$ & %.6f & %.6f & $%+.1f$ & %.2f & %.2f & $%+.1f$ \\\\\\hline"
              % (n, spin, sym, Et, Ea, (Ea - Et) * 1e6, Pt, Pa, (Pa - Pt) * 1e3))


if __name__ == "__main__":
    main()
