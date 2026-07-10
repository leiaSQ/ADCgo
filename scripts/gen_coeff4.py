#!/usr/bin/env python3
"""Generate internal/adc/sip/coeff4.go from the reference init0/1/2 spin tables.

Parses the fixed-form F77 DATA blocks, the VECTOR(...)=SPINk column->row maps, the
VORFAK/VORF normalization assignments, and (init2) the IESSE compaction, exactly as
INIT0_core/INIT1_core/INIT2_core build them. Emits three Go arrays:

  coeff0 [13][36]float64        (init0, /DREI/,  KOPP4)
  coeff1 [3][13][30]float64     (init1, /VIER/,  AB5 2h1p<->3h2p)
  coeff2 [13][13][42]float64    (init2, /FUENF/, AB5 3h2p<->3h2p, after IESSE compaction)
"""
import re, math, sys

REF = "/home/leia/Documents/ADC/adc4core/adc4_constr"


def logical_lines(path):
    """Yield joined logical F77 lines (fixed form: continuation = nonblank col 6)."""
    out = []
    cur = None
    for raw in open(path):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        c1 = line[0] if line else " "
        if c1 in "C*!c":
            continue
        cont = len(line) > 5 and line[5] not in (" ", "0")
        body = line[6:] if len(line) > 6 else ""
        # strip inline comment
        if cont and cur is not None:
            cur += body
        else:
            if cur is not None:
                out.append(cur)
            cur = (line[6:] if len(line) > 6 else "")
    if cur is not None:
        out.append(cur)
    return out


def parse_data_arrays(lines):
    """Return {name: [ints]} for every DATA <name>/ ... / (name may contain spaces)."""
    text = " ".join(lines)
    arrays = {}
    # DATA SPIN 1/ ... / possibly several names? here each DATA has one name.
    for m in re.finditer(r"DATA\s+([A-Z0-9 ]+?)\s*/([^/]*)/", text):
        name = m.group(1).replace(" ", "")
        vals = []
        for tok in m.group(2).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "*" in tok:  # repeat count, e.g. 4*1
                n, v = tok.split("*")
                vals.extend([int(v)] * int(n))
            else:
                vals.append(int(tok))
        arrays[name] = vals
    return arrays


def parse_vorfak(lines, name):
    """Return dict idx->float for assignments like VORFAK(6) = 1.0D0/DSQRT(2.0D0)."""
    out = {}
    pat = re.compile(rf"{name}\(\s*(\d+)\s*\)\s*=\s*(.+?)\s*$")
    for ln in lines:
        m = pat.search(ln)
        if not m:
            continue
        idx = int(m.group(1))
        expr = m.group(2)
        expr = expr.replace("DSQRT", "math.sqrt").replace("DFLOAT", "float")
        expr = re.sub(r"(\d+\.?\d*)D0", r"\1", expr)
        expr = re.sub(r"(\d+\.?\d*)D([+-]?\d+)", r"\1e\2", expr)
        out[idx] = eval(expr, {"math": math, "float": float})
    return out


def parse_vector_map(lines, ndim):
    """Return {colK: srcname} from VECTOR(...,K)=DFLOAT(SRC(...)) lines.
    ndim in {2,3}: VECTOR(I,K) or VECTOR(I,J,K)."""
    out = {}
    if ndim == 2:
        pat = re.compile(r"VECTOR\(\s*I\s*,\s*(\d+)\s*\)\s*=\s*DFLOAT\(\s*([A-Z0-9]+)")
    else:
        pat = re.compile(r"VECTOR\(\s*I\s*,\s*J\s*,\s*(\d+)\s*\)\s*=\s*DFLOAT\(\s*([A-Z0-9]+)")
    for ln in lines:
        for m in pat.finditer(ln):
            out[int(m.group(1))] = m.group(2)
    return out


def build_init0():
    lines = logical_lines(f"{REF}/init0.F")
    arr = parse_data_arrays(lines)
    vor = parse_vorfak(lines, "VORFAK")
    cmap = parse_vector_map(lines, 2)  # col(1..36) -> SPIN name
    assert len(cmap) == 36, len(cmap)
    # VECTOR(I,K) = SPIN{cmap[K]}(I) * VORFAK(I),  I=1..13
    out = [[0.0] * 36 for _ in range(13)]
    for k in range(1, 37):
        src = arr[cmap[k]]
        for i in range(1, 14):
            out[i - 1][k - 1] = src[i - 1] * vor[i]
    return out


def build_init1():
    lines = logical_lines(f"{REF}/init1.F")
    arr = parse_data_arrays(lines)
    vf1 = parse_vorfak(lines, "VORF1")
    vf2 = parse_vorfak(lines, "VORF2")
    cmap = parse_vector_map(lines, 3)
    assert len(cmap) == 30, len(cmap)
    # IZ = (I-1)*13 + J, I=1..3, J=1..13. VECTOR(I,J,K)=SPIN{cmap[K]}(IZ)*VORF1(I)*VORF2(J)
    out = [[[0.0] * 30 for _ in range(13)] for _ in range(3)]
    for i in range(1, 4):
        for j in range(1, 14):
            iz = (i - 1) * 13 + j
            for k in range(1, 31):
                out[i - 1][j - 1][k - 1] = arr[cmap[k]][iz - 1] * vf1[i] * vf2[j]
    return out


def build_init2():
    lines = logical_lines(f"{REF}/init2.F")
    arr = parse_data_arrays(lines)
    vor = parse_vorfak(lines, "VORFAK")
    iesse = arr["IESSE"]
    assert len(iesse) == 92, len(iesse)
    cmap = parse_vector_map(lines, 3)
    assert len(cmap) == 92, len(cmap)
    # IZ = (I-1)*13 + J, I,J=1..13. VECTOR(I,J,K)=SPIN{cmap[K]}(IZ)*VORFAK(I)*VORFAK(J)
    full = [[[0.0] * 92 for _ in range(13)] for _ in range(13)]
    for i in range(1, 14):
        for j in range(1, 14):
            iz = (i - 1) * 13 + j
            for k in range(1, 93):
                full[i - 1][j - 1][k - 1] = arr[cmap[k]][iz - 1] * vor[i] * vor[j]
    # IESSE compaction: keep columns with IESSE[K]!=0, compacted to the front.
    keep = [k for k in range(92) if iesse[k] != 0]
    nkeep = len(keep)
    out = [[[full[i][j][k] for k in keep] for j in range(13)] for i in range(13)]
    return out, nkeep


def fmt(x):
    if x == 0:
        return "0"
    s = repr(x)
    return s


def emit_2d(name, a, dims):
    lines = [f"var {name} = [{dims[0]}][{dims[1]}]float64{{"]
    for row in a:
        lines.append("\t{" + ", ".join(fmt(v) for v in row) + "},")
    lines.append("}")
    return "\n".join(lines)


def emit_3d(name, a, dims):
    lines = [f"var {name} = [{dims[0]}][{dims[1]}][{dims[2]}]float64{{"]
    for plane in a:
        lines.append("\t{")
        for row in plane:
            lines.append("\t\t{" + ", ".join(fmt(v) for v in row) + "},")
        lines.append("\t},")
    lines.append("}")
    return "\n".join(lines)


c0 = build_init0()
c1 = build_init1()
c2, nkeep = build_init2()

header = '''package sip

// coeff4.go — spin-coupling coefficient tables for CVS IP-ADC(4), transcribed
// verbatim from the reference INIT0_core/INIT1_core/INIT2_core
// (../ADC/adc4core/adc4_constr/init{0,1,2}.F) by scripts/gen_coeff4.py. Do not
// edit by hand; regenerate. See docs/adc4_sip_spec.md §6.
//
//   coeff0: /DREI/  VECTOR(13,36)     — KOPP4 (1h<->3h2p direct coupling)
//   coeff1: /VIER/  VECTOR(3,13,30)   — AB5   (2h1p<->3h2p effective coupling)
//   coeff2: /FUENF/ VECTOR(13,13,%d)  — AB5   (3h2p<->3h2p), after IESSE compaction
//
// Fortran is column-first VECTOR(I,J,K); the Go arrays are indexed [I-1][J-1][K-1]
// (or [I-1][K-1] for coeff0), 0-based.
''' % nkeep

with open("/home/leia/Documents/ADCgo/internal/adc/sip/coeff4.go", "w") as f:
    f.write(header + "\n")
    f.write(emit_2d("coeff0", c0, (13, 36)) + "\n\n")
    f.write(emit_3d("coeff1", c1, (3, 13, 30)) + "\n\n")
    f.write(emit_3d("coeff2", c2, (13, 13, nkeep)) + "\n")

# Spot values for manual verification against the F77.
print("nkeep(IESSE) =", nkeep)
print("coeff0[0][0] = SPIN1(1)*VORFAK(1) = 2*0.5 =", c0[0][0])
print("coeff0[5][0] = SPIN1(6)*VORFAK(6) = 1*(1/sqrt2) =", c0[5][0])
print("coeff1[0][0][0] = SPIN1(1)*VORF1(1)*VORF2(1) =", c1[0][0][0])
print("coeff2[0][0][0] = SPIN1(1)*VORFAK(1)^2 = 4*0.25 =", c2[0][0][0])
print("shapes:", len(c0), len(c0[0]), "|", len(c1), len(c1[0]), len(c1[0][0]),
      "|", len(c2), len(c2[0]), len(c2[0][0]))
