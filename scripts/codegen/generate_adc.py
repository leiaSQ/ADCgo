#!/usr/bin/env python3.12
"""
generate_adc.py — adcgen -> Go code generator for k-fold ionization ADC
========================================================================
Derives the ISR secular-matrix blocks of k-fold ionization ADC with the
`adcgen` library and emits Go element evaluators for them:

    variant  k  classes (main | satellite | double satellite)
    ip       1  1h | 2h1p | 3h2p
    dip      2  2h | 3h1p | 4h2p
    tip      3  3h | 4h1p | 5h2p
    qip      4  4h | 5h1p | 6h2p

The blocks are B00, B01, B11, B02, B12, B22, where 0/1/2 name the three
classes. adcgen registers ip and dip itself. tip and qip use KHoleStates, a
subclass that only changes the minimal excitation space: adcgen's
IntermediateStates is variant-agnostic except for a 'pp' special case, and
the subclass is checked against the native dip variant at start-up.

Usage
-----
    python3.12 scripts/codegen/generate_adc.py --variant dip --schemes strict:2,ci \\
        --outdir internal/adc/isrgen/dip [--blocks B00,B01,B11]

Schemes (per-block maximum perturbation order; several may be unioned, and
the generated package truncates to any of them at run time):

    strict:N  adcgen's own SecularMatrix.block_order(N) (strict ADC(N))
    ci        every block through first order. This is plain CI in the class
              space (H - E_HF) for every block EXCEPT B02 = kh/(k+2)h2p: those
              configurations differ by a double excitation, so CI couples them at
              first order, while in the ISR the first-order ground-state correction
              cancels that coupling and B02 starts at second order (the C13 of IP-ADC).
    adc2x     strict:2 with the (k+1)h1p/(k+1)h1p block through first order
    adc22m/x/f  ADC(2,2) of Kolorenc/Averbukh, Table I, as in
              internal/adc/sip/elements22.go: B00<=2, B01<=1 (f: 2), B11<=2,
              B12<=1, B22 0 (x, f: 1); B02 absent

Pipeline (per block, per order):
    isr_matrix_block -> simplify -> diagonalize_fock -> reduce_expr
    -> factor_intermediates -> exploit_perm_sym -> optimize_contractions
The Contraction objects are translated to Go loop nests directly (einsum
text is never parsed). Every referenced intermediate (t2_1, t2_2, t2eri_*,
p0_*, ...) becomes a dense tensor built in New from adcgen's own definition.

Basis and conventions: spin orbitals over the spatial FCIDUMP integrals
(spin orbital p = spatial p>>1, spin p&1; occupied 0..2*nocc-1; virtual a
has absolute index nocc+a), canonical HF (diagonal Fock). Row phases follow
internal/adc/khci: |r> = c+_{a1} c+_{a2} c_{i1} ... c_{in} |Phi0> with
ascending indices, which is adcgen's precursor operator order; the generated
Element(sp, r, c) evaluates <r|M|c> for rows of a khci.Space.

Fidelity reference (testdata/adcgen_ref.json, skip with --no-reference):
on a random canonical 6-orbital system, every generated block and order is
evaluated at probe configurations from adcgen's FULLY EXPANDED expression
(no intermediates) in numpy, independently of the Go translation. For the
blocks' first-order truncation it also evaluates plain Slater-Condon CI in
the khci phase convention and ASSERTS ISR(<=1) = CI to 1e-12, except for
B02 (see the ci scheme); the CI values are stored for every block,
generated or not, for khci's CI gate.

Measured derivation costs (simplified term counts before permutational
symmetry, 2026-09-22):
    k=2: B12 o1 10 s, B22 o0 4.6 s, B22 o1 30 s
    k=3: B11 o1 6 s, B12 o1 114 s, B22 o0 107 s, B22 o1 > 10 min
    ip:  C11 order 3 47 s; C11 order 4 did not finish in 8 min

Requirements: Python >= 3.10, adcgen (../adcgen or ADCGEN_DIR), sympy,
numpy (reference only), gofmt in PATH.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate adcgen
# ---------------------------------------------------------------------------
_ADCGEN_DIR = os.environ.get(
    "ADCGEN_DIR",
    str(Path(__file__).resolve().parents[3] / "adcgen"),
)
sys.path.insert(0, _ADCGEN_DIR)

from sympy import Add, Mul, Pow, S  # noqa: E402

from adcgen import (  # noqa: E402
    ExprContainer,
    GroundState,
    IntermediateStates,
    Intermediates,
    Operators,
    SecularMatrix,
    factor_intermediates,
    get_symbols,
    optimize_contractions,
    unoptimized_contraction,
    reduce_expr,
    set_log_level,
    simplify,
)
from adcgen.generate_code.contraction import Contraction  # noqa: E402
from adcgen.sort_expr import exploit_perm_sym  # noqa: E402
from adcgen.sympy_objects import SymbolicTensor  # noqa: E402
from adcgen.tensor_names import tensor_names  # noqa: E402

RECV = "el"     # Go receiver name; never a valid adcgen index name
TYPE = "Elem"   # Go evaluator type
PACKAGE = "isrgen"  # set from --package in main


def go_header() -> str:
    return ("// Code generated by adcgen pipeline (scripts/codegen/generate_adc.py). "
            f"DO NOT EDIT.\n\npackage {PACKAGE}\n")


# ---------------------------------------------------------------------------
# Variants, classes, blocks
# ---------------------------------------------------------------------------
VARIANTS = {"ip": 1, "dip": 2, "tip": 3, "qip": 4}
NATIVE = {1: "ip", 2: "dip"}

# Target index names, drawn bra first, then ket, from one pool per space: bare
# letters first, numbered names only on overflow. adcgen generates contracted
# indices from counter 3 upward ("i3", "a3", ...), so neither can collide with a
# contracted index (a per-term guard in emit_term_func checks it anyway). Bare
# letters come first because adcgen's factor_intermediates fails an assertion
# on numbered TARGET names: the IP C11 order-3 block factors with targets "i,j"
# and asserts with "i,i1". Overflow only happens in blocks with more than seven
# holes in bra plus ket (dip B22, tip B11/B12/B22, qip), which this generator
# derives at most to first order or (qip B00, 4+4 holes) where it was verified to
# go through.
OCC_POOL = list("ijklmno") + [f"{c}1" for c in "ijklmno"]
VIR_POOL = list("abcdefgh")

BLOCK_PAIRS = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]
BLOCK_NAMES = ["B00", "B01", "B11", "B02", "B12", "B22"]


def class_space(k: int, c: int) -> str:
    """adcgen space string of class c: h^k, p h^(k+1), pp h^(k+2)."""
    return "p" * c + "h" * (k + c)


@dataclass
class BlockSpec:
    """One secular-matrix block <bra class|M|ket class>."""

    k: int
    cb: int          # bra class 0..2
    ck: int          # ket class 0..2
    go_func: str     # "B01"
    index: int       # position in BLOCK_NAMES

    @property
    def bra(self) -> str:
        return class_space(self.k, self.cb)

    @property
    def ket(self) -> str:
        return class_space(self.k, self.ck)

    @property
    def bra_names(self) -> list[str]:
        return VIR_POOL[:self.cb] + OCC_POOL[:self.k + self.cb]

    @property
    def ket_names(self) -> list[str]:
        nb_occ, nb_vir = self.k + self.cb, self.cb
        return (VIR_POOL[nb_vir:nb_vir + self.ck]
                + OCC_POOL[nb_occ:nb_occ + self.k + self.ck])

    @property
    def block_str(self) -> str:
        return f"{self.bra},{self.ket}"

    @property
    def indices_str(self) -> str:
        return f"{''.join(self.bra_names)},{''.join(self.ket_names)}"

    @property
    def target(self) -> list[str]:
        return self.bra_names + self.ket_names

    @property
    def bra_ket_sym(self) -> int:
        return 1 if self.cb == self.ck else 0


def all_blocks(k: int) -> list[BlockSpec]:
    return [BlockSpec(k, cb, ck, BLOCK_NAMES[n], n)
            for n, (cb, ck) in enumerate(BLOCK_PAIRS)]


def scheme_orders(sm: SecularMatrix, k: int, scheme: str) -> list[int]:
    """Per-block maximum order (index = BLOCK_NAMES position), -1 = absent."""
    if scheme.startswith("strict:"):
        n = int(scheme.split(":", 1)[1])
        table = sm.block_order(n)
        out = []
        for cb, ck in BLOCK_PAIRS:
            key = (class_space(k, cb), class_space(k, ck))
            out.append(int(table.get(key, -1)))
        return [max(o, -1) for o in out]
    if scheme == "ci":
        return [1, 1, 1, 1, 1, 1]
    if scheme == "adc2x":
        o = scheme_orders(sm, k, "strict:2")
        o[BLOCK_NAMES.index("B11")] = 1
        return o
    if scheme in ("adc22m", "adc22x", "adc22f"):
        v = scheme[-1]
        return [2, 2 if v == "f" else 1, 2, -1, 1, 1 if v in "xf" else 0]
    raise ValueError(f"unknown scheme {scheme!r}")


# ---------------------------------------------------------------------------
# adcgen derivation
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def quiet():
    """adcgen prints its progress to stdout; keep it out of the terminal."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class KHoleStates(IntermediateStates):
    """adcgen intermediate states for k-fold ionization, k >= 3.

    adcgen registers variants pp/ea/ip/dip/dea only. IntermediateStates keys
    everything on self.min_space (the precursor validation and the space
    hierarchy) and special-cases only variant == "pp", so setting the minimal
    space to h^k is the whole generalization. check_khole_subclass pins it
    against the native dip variant; if adcgen ever grows variant-specific logic
    that check is where it shows.
    """

    def __init__(self, gs: GroundState, k: int):
        super().__init__(gs, variant="ip")
        self.variant = f"k{k}ip"
        self.min_space = ("h" * k,)


def init_adcgen(k: int, force_subclass: bool = False
                ) -> tuple[GroundState, SecularMatrix]:
    h = Operators(variant="mp")
    gs = GroundState(h, first_order_singles=False)
    if k in NATIVE and not force_subclass:
        isr = IntermediateStates(gs, variant=NATIVE[k])
    else:
        isr = KHoleStates(gs, k)
    return gs, SecularMatrix(isr)


def check_khole_subclass():
    """KHoleStates(k=2) must reproduce adcgen's native dip blocks."""
    _, native = init_adcgen(2)
    _, sub = init_adcgen(2, force_subclass=True)
    for spec, order in ((all_blocks(2)[0], 2), (all_blocks(2)[1], 1),
                        (all_blocks(2)[2], 1)):
        with quiet():
            a = ExprContainer(native.isr_matrix_block(
                order=order, block=spec.block_str, indices=spec.indices_str),
                real=True)
            b = ExprContainer(sub.isr_matrix_block(
                order=order, block=spec.block_str, indices=spec.indices_str),
                real=True)
            diff = simplify(a - b)
        if diff.inner is not S.Zero:
            raise RuntimeError(f"KHoleStates(k=2) differs from adcgen's dip on "
                               f"{spec.block_str} order {order}")


def derive_block(sm: SecularMatrix, spec: BlockSpec, order: int
                 ) -> ExprContainer | None:
    """The simplified, intermediate-factored order-th contribution to spec."""
    with quiet():
        raw = sm.isr_matrix_block(order=order, block=spec.block_str,
                                  indices=spec.indices_str)
        expr = ExprContainer(raw, real=True)
        expr.substitute_contracted()
        expr = simplify(expr)
        if expr.inner is S.Zero:
            return None
        expr.diagonalize_fock()
        expr = reduce_expr(expr)
        if expr.inner is S.Zero:
            return None
        expr = factor_intermediates(expr, max_order=order)
    return expr


# ---------------------------------------------------------------------------
# Intermediate definitions
# ---------------------------------------------------------------------------
_T_AMP = re.compile(r"^t(\d+)_(\d+)$")


@dataclass
class ItmdDef:
    """A dense intermediate tensor and its defining expression."""

    name: str
    target: list[str]   # index names in storage order
    spaces: list[str]   # 'o'/'v' per target index
    expr: ExprContainer


def _itmd_definition(gs: GroundState, name: str) -> ItmdDef:
    """
    Definition of a tensor named in a contraction. Registered adcgen
    intermediates use their own non-recursive definition (in terms of
    lower-order intermediates). A t-amplitude with no registered class is
    taken straight from the MP amplitude equations of the ground state.
    """
    available = Intermediates().available
    if name in available:
        itmd = available[name]
        target = list(itmd.default_idx)
        with quiet():
            expr = itmd.expand_itmd(fully_expand=False)
            # storage order must equal the index order of the tensor object
            # as it appears in contractions
            tensor = itmd.tensor(wrap_result=True)
        obj_idx = [s.name for s in tensor.terms[0].objects[-1].idx]
        if obj_idx != target:
            raise RuntimeError(f"intermediate {name}: tensor index order "
                               f"{obj_idx} differs from default {target}")
    elif m := _T_AMP.match(name):
        rank, order = int(m.group(1)), int(m.group(2))
        occ = "ijklmno"[:rank]
        vir = "abcdefgh"[:rank]
        target = list(occ + vir)
        with quiet():
            expr = ExprContainer(
                gs.amplitude(order, "p" * rank + "h" * rank, occ + vir),
                target_idx=occ + vir,
            )
    else:
        raise KeyError(f"no definition for tensor '{name}'")
    with quiet():
        expr = expr.expand()
    spaces = [s.space[0] for s in get_symbols(target)]
    return ItmdDef(name, target, spaces, expr)


# ---------------------------------------------------------------------------
# Go translation of one term
# ---------------------------------------------------------------------------
def go_float(x) -> str:
    """Exact-as-float64 Go literal for a sympy number."""
    v = float(x)
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v)}.0"
    return repr(v)


def _space_of(idx) -> str:
    sp = idx.space[0]
    if sp not in ("o", "v"):
        raise NotImplementedError(f"index {idx} in space {idx.space}")
    return sp


def abs_so(idx) -> str:
    """Absolute spin-orbital index expression for an adcgen Index."""
    return idx.name if _space_of(idx) == "o" else f"{RECV}.nocc+{idx.name}"


def loop_bound(idx) -> str:
    return f"{RECV}.nocc" if _space_of(idx) == "o" else f"{RECV}.nvir"


def go_itmd_field(name: str) -> str:
    return "x_" + name


def go_itmd_get(name: str) -> str:
    return "get_" + name


def orb_energy_go(x) -> tuple[str, set[str]]:
    """Translate a sympy orbital-energy polynomial/fraction to Go."""
    deps: set[str] = set()

    def walk(y) -> str:
        if y.is_Number:
            return go_float(y)
        if isinstance(y, SymbolicTensor):
            if y.name != tensor_names.orb_energy or len(y.idx) != 1:
                raise NotImplementedError(f"unexpected object {y} in an "
                                          "orbital energy fraction")
            deps.add(y.idx[0].name)
            return f"{RECV}.eps({abs_so(y.idx[0])})"
        if isinstance(y, Add):
            return "(" + " + ".join(walk(a) for a in y.args) + ")"
        if isinstance(y, Mul):
            return "(" + " * ".join(walk(a) for a in y.args) + ")"
        if isinstance(y, Pow):
            base, exp = y.args
            if not exp.is_Integer or exp == 0:
                raise NotImplementedError(f"exponent {exp} in {y}")
            b = walk(base)
            prod = " * ".join([b] * abs(int(exp)))
            return f"(1.0 / ({prod}))" if exp < 0 else f"({prod})"
        raise NotImplementedError(f"cannot translate {y} ({type(y)})")

    return walk(x), deps


@dataclass
class Factor:
    kind: str                 # "delta" | "value" | "inner"
    deps: frozenset[str]
    code: str = ""            # value: Go expression
    pair: tuple[str, str] = ("", "")
    contr: Contraction | None = None


class TermEmitter:
    """Emits the loop nest that evaluates one term for fixed target indices."""

    def __init__(self, itmd_names: set[str]):
        self.itmd_names = itmd_names   # filled while emitting
        self.nvar = 0

    def var(self, prefix: str) -> str:
        self.nvar += 1
        return f"{prefix}{self.nvar}"

    def tensor_code(self, name: str, idxs) -> str:
        if name.startswith(f"{tensor_names.eri}_"):
            if len(idxs) != 4:
                raise NotImplementedError(f"ERI {name} with {len(idxs)} idx")
            args = ", ".join(abs_so(i) for i in idxs)
            return f"{RECV}.asym({args})"
        if name.startswith(f"{tensor_names.fock}_"):
            return f"{RECV}.fock({abs_so(idxs[0])}, {abs_so(idxs[1])})"
        if name.startswith(f"{tensor_names.coulomb}_") or \
                name.startswith(f"{tensor_names.ri_sym}_"):
            raise NotImplementedError(f"tensor {name} is not supported")
        self.itmd_names.add(name)
        args = ", ".join(i.name for i in idxs)
        return f"{RECV}.{go_itmd_get(name)}({args})"

    def factors_of(self, c: Contraction, inner: dict[str, Contraction]
                   ) -> list[Factor]:
        out: list[Factor] = []
        for name, idxs in zip(c.names, c.indices):
            names = [i.name for i in idxs]
            if Contraction.is_contraction(name):
                sub = inner[name]
                out.append(Factor("inner", frozenset(i.name for i in
                                                      sub.target), contr=sub))
            elif name.startswith(f"{tensor_names.operator}_"):
                # Kronecker delta (d_oo / d_vv)
                if names[0] != names[1]:
                    out.append(Factor("delta", frozenset(names),
                                      pair=(names[0], names[1])))
            else:
                out.append(Factor("value", frozenset(names),
                                  code=self.tensor_code(name, idxs)))
        return out

    def emit(self, c: Contraction, inner: dict[str, Contraction],
             bound: set[str], acc: str, extra: list[Factor],
             ind: str) -> list[str]:
        """
        Accumulate the contraction c into Go variable acc. Loops are opened
        for c.contracted. Every factor is evaluated at the shallowest loop
        depth where its indices are bound, and a zero value skips the whole
        subtree. That skip is what makes spin-forbidden integrals free.
        """
        lines: list[str] = []
        pending = self.factors_of(c, inner) + extra
        bound = set(bound)
        loops = {i.name: i for i in c.contracted}
        remaining = [i.name for i in c.contracted]
        product: list[str] = []
        depth = 0

        def pad() -> str:
            return ind + "\t" * depth

        def flush():
            nonlocal depth
            progress = True
            while progress:
                progress = False
                for f in list(pending):
                    if not f.deps <= bound:
                        continue
                    pending.remove(f)
                    progress = True
                    if f.kind == "delta":
                        x, y = f.pair
                        lines.append(f"{pad()}if {x} == {y} {{")
                    elif f.kind == "value":
                        v = self.var("fv")
                        lines.append(f"{pad()}{v} := {f.code}")
                        lines.append(f"{pad()}if {v} != 0 {{")
                        product.append(v)
                    else:
                        v = self.var("sv")
                        lines.append(f"{pad()}var {v} float64")
                        lines.extend(self.emit(f.contr, inner, bound, v, [],
                                               pad()))
                        lines.append(f"{pad()}if {v} != 0 {{")
                        product.append(v)
                    depth += 1

        flush()
        while remaining:
            # a delta tying a loop index to a bound one replaces the loop
            chosen, alias = None, None
            for x in remaining:
                for f in pending:
                    if f.kind == "delta" and x in f.pair:
                        other = f.pair[1] if f.pair[0] == x else f.pair[0]
                        if other in bound:
                            chosen, alias = x, (f, other)
                            break
                if chosen:
                    break
            if chosen is None:
                # open the loop that makes the most factors evaluable
                chosen = max(remaining, key=lambda x: sum(
                    f.deps <= bound | {x} for f in pending))
            remaining.remove(chosen)
            if alias is not None:
                f, other = alias
                pending.remove(f)
                used = any(chosen in g.deps for g in pending)
                if used:
                    lines.append(f"{pad()}{{")
                    depth += 1
                    lines.append(f"{pad()}{chosen} := {other}")
                # an index seen only by the delta sums to exactly 1
            else:
                idx = loops[chosen]
                lines.append(f"{pad()}for {chosen} := 0; {chosen} < "
                             f"{loop_bound(idx)}; {chosen}++ {{")
                depth += 1
            bound.add(chosen)
            flush()
        if pending:
            raise RuntimeError(f"unplaced factors {pending} in {c}")
        lines.append(f"{pad()}{acc} += {' * '.join(product) or '1'}")
        while depth:
            depth -= 1
            lines.append(f"{pad()}}}")
        return lines


def emit_term_func(fname: str, target: list[str], term, emitter: TermEmitter,
                   comment: str) -> str:
    """One Go function evaluating a single adcgen term for fixed target."""
    split = term.split_orb_energy()
    num, den, rem = split["num"], split["denom"], split["remainder"]
    orb = num.inner / den.inner
    orb_code, orb_deps = orb_energy_go(orb)
    tset = set(target)

    rem_term = rem.terms[0] if len(rem) else None
    if rem_term is not None and rem_term.prefactor != S.One:
        # split_orb_energy moves prefactors into the numerator
        orb = orb * rem_term.prefactor
        orb_code, orb_deps = orb_energy_go(orb)
    has_objects = rem_term is not None and any(
        not o.inner.is_number for o in rem_term.objects)
    sig = f"func ({RECV} *{TYPE}) {fname}({', '.join(target)} int) float64"
    body: list[str] = [f"// {comment}", sig + " {"]

    if not has_objects:
        body.append(f"\treturn {orb_code}")
        body.append("}")
        return "\n".join(body)

    tgt_symbols = get_symbols(target)
    if orb_deps <= tset:
        # denominator over target indices only: factor it out of the sum
        # When no nested scheme exists under the intermediate-dimension cap (an outer
        # product of three deltas and an ERI in dip B12), the single
        # hyper-contraction is always valid. The cap itself stays: without it the
        # scheme search is exhaustive and far slower.
        with quiet():
            try:
                contrs = optimize_contractions(
                    term=rem_term, target_indices="".join(target), max_itmd_dim=4)
            except RuntimeError:
                contrs = unoptimized_contraction(
                    term=rem_term, target_indices="".join(target))
        extra: list[Factor] = []
        scale = orb_code
    else:
        # the fraction carries contracted indices: one hyper-contraction with
        # the fraction as a factor in the loop nest
        names, idxs = [], []
        for o in rem_term.objects:
            if o.inner.is_number:
                continue
            base, exp = o.base_and_exponent
            if not exp.is_Integer or exp < 1:
                raise NotImplementedError(f"object {o} with exponent {exp}")
            names.extend([o.longname()] * int(exp))
            idxs.extend([o.idx] * int(exp))
        all_idx = {i for ix in idxs for i in ix} | set(
            s for s in term.idx)
        contracted = sorted((i for i in all_idx if i.name not in tset),
                            key=lambda i: i.name)
        contrs = [Contraction(indices=idxs, names=names,
                              term_target_indices=tgt_symbols,
                              contracted=contracted, target=tgt_symbols)]
        extra = [Factor("value", frozenset(orb_deps), code=orb_code)]
        scale = ""

    for c in contrs:
        clash = {i.name for i in c.contracted} & tset
        if clash:
            raise RuntimeError(f"{fname}: contracted indices {sorted(clash)} collide "
                               "with target names")
    inner = {c.contraction_name: c for c in contrs[:-1]}
    outer = contrs[-1]
    body.append("\tvar acc float64")
    body.extend(emitter.emit(outer, inner, tset, "acc", extra, "\t"))
    if scale and scale != "1.0":
        body.append(f"\treturn {scale} * acc")
    else:
        body.append("\treturn acc")
    body.append("}")
    return "\n".join(body)


def _perm_args(target: list[str], perm_product) -> list[str]:
    """Arguments that evaluate P X at the target, for a product of transpositions.

    adcgen's ExprContainer.permute applies the transpositions one after another in
    the order given, each as a simultaneous relabelling of index symbols (its
    substitution dict composes them that way): P_ij P_ik maps X(i,j,k) to
    X(j,k,i). Relabelling the argument list in the same order reproduces that,
    including overlapping products, which the 3-hole antisymmetrizers of dip and
    higher produce.
    """
    args = list(target)
    for p in perm_product:
        a, b = p[0].name, p[1].name
        args = [b if x == a else a if x == b else x for x in args]
    return args


def emit_expr_funcs(fname: str, target: list[str],
                    groups: list[tuple[tuple, list]], emitter: TermEmitter,
                    doc: str) -> str:
    """
    Emit fname(target) = sum over (permutation group, terms) of
    (1 + sum_P sign_P P) sum_terms term. Each term becomes its own function.
    """
    parts: list[str] = []
    calls: list[str] = []
    nterm = 0
    for g, (perm_sym, terms) in enumerate(groups):
        gname = f"{fname}g{g}"
        tcalls = []
        for term in terms:
            tname = f"{fname}t{nterm}"
            nterm += 1
            parts.append(emit_term_func(tname, target, term, emitter,
                                        f"{tname}: {term}"))
            tcalls.append(f"{RECV}.{tname}({', '.join(target)})")
        sig = f"func ({RECV} *{TYPE}) {gname}({', '.join(target)} int) float64"
        parts.append(sig + " {\n\treturn " + " +\n\t\t".join(tcalls) + "\n}")
        perm_desc = " ".join(
            ("+ " if s > 0 else "- ") + "".join(str(p) for p in pp)
            for pp, s in perm_sym)
        calls.append(f"\t// (1 {perm_desc})" if perm_sym else "\t// (1)")
        calls.append(f"\tv += {RECV}.{gname}({', '.join(target)})")
        for pp, sign in perm_sym:
            args = _perm_args(target, pp)
            op = "+=" if sign > 0 else "-="
            calls.append(f"\tv {op} {RECV}.{gname}({', '.join(args)})")
    sig = f"func ({RECV} *{TYPE}) {fname}({', '.join(target)} int) float64"
    parts.append(f"// {doc}\n{sig} {{\n\tvar v float64\n" + "\n".join(calls)
                 + "\n\treturn v\n}")
    return "\n\n".join(parts)


def collect_itmds(gs: GroundState, names: set[str],
                  emitter: TermEmitter) -> list[tuple[ItmdDef, str]]:
    """
    Emit the definition functions of every intermediate reachable from names,
    returned in dependency order (dependencies first).
    """
    defs: dict[str, tuple[ItmdDef, str, set[str]]] = {}
    pending = sorted(names)
    while pending:
        name = pending.pop()
        if name in defs:
            continue
        d = _itmd_definition(gs, name)
        before = set(emitter.itmd_names)
        emitter.itmd_names = set()
        groups = [((), list(d.expr.terms))]
        code = emit_expr_funcs(
            f"def_{name}", d.target, groups, emitter,
            f"def_{name} is adcgen's definition of intermediate {name}"
            f"[{','.join(d.target)}].")
        deps = set(emitter.itmd_names)
        emitter.itmd_names = before | deps
        if name in deps:
            raise RuntimeError(f"intermediate {name} defined recursively")
        defs[name] = (d, code, deps)
        pending.extend(sorted(deps - defs.keys()))

    ordered: list[tuple[ItmdDef, str]] = []
    done: set[str] = set()

    def visit(n: str, stack: tuple[str, ...]):
        if n in done:
            return
        if n in stack:
            raise RuntimeError(f"cyclic intermediates {stack + (n,)}")
        d, code, deps = defs[n]
        for dep in sorted(deps):
            visit(dep, stack + (n,))
        done.add(n)
        ordered.append((d, code))

    for n in sorted(defs):
        visit(n, ())
    return ordered


def emit_itmd_file(outdir: Path, itmds: list[tuple[ItmdDef, str]]) -> None:
    # a first-order-only package has no intermediates and must not import parallel
    parts = [go_header() + ('\nimport "github.com/leiaSQ/ADCgo/internal/adc/'
                            'parallel"' if itmds else "")]
    fields = "\n".join(
        f"\t{go_itmd_field(d.name)} []float64 // {d.name}"
        f"[{','.join(d.target)}]" for d, _ in itmds)
    parts.append("// itmdStore holds the dense intermediate tensors, "
                 "row-major over their\n// indices (occupied extent el.nocc, "
                 "virtual extent el.nvir).\n"
                 f"type itmdStore struct {{\n{fields}\n}}")

    build = ["// buildItmds fills every intermediate in dependency order.",
             f"func ({RECV} *{TYPE}) buildItmds() {{"]
    for d, _ in itmds:
        build.append(f"\t{RECV}.build_{d.name}()")
    build.append("}")
    parts.append("\n".join(build))

    for d, code in itmds:
        ext = [f"{RECV}.nocc" if s == "o" else f"{RECV}.nvir"
               for s in d.spaces]
        # flat index, Horner form
        flat = d.target[0]
        for k in range(1, len(d.target)):
            flat = f"({flat})*{ext[k]}+{d.target[k]}"
        params = ", ".join(d.target)
        get = (f"func ({RECV} *{TYPE}) {go_itmd_get(d.name)}({params} int) "
               f"float64 {{\n\treturn {RECV}.{go_itmd_field(d.name)}"
               f"[{flat}]\n}}")
        size = "*".join(ext)
        loops_open, loops_close = [], []
        for k in range(1, len(d.target)):
            t = "\t" * (k + 1)
            loops_open.append(f"{t}for {d.target[k]} := 0; {d.target[k]} < "
                              f"{ext[k]}; {d.target[k]}++ {{")
            loops_close.insert(0, f"{t}}}")
        t = "\t" * (len(d.target) + 1)
        fill = "\n".join(
            [f"func ({RECV} *{TYPE}) build_{d.name}() {{",
             f"\t{RECV}.{go_itmd_field(d.name)} = make([]float64, {size})",
             f"\tparallel.HeavyRows({ext[0]}, func({d.target[0]} int) {{"]
            + loops_open
            + [f"{t}{RECV}.{go_itmd_field(d.name)}[{flat}] = "
               f"{RECV}.def_{d.name}({params})"]
            + loops_close + ["\t})", "}"])
        parts.append(f"// ---- {d.name} ----\n\n{get}\n\n{fill}\n\n{code}")
    (outdir / "intermediates_generated.go").write_text(
        "\n\n".join(parts) + "\n")



# ---------------------------------------------------------------------------
# File emitters
# ---------------------------------------------------------------------------
# exploit_perm_sym searches permutations of same-space target indices; its cost
# grows as the product over spaces of (count)!. It took > 9 min on tip B11 (8 occupied
# and 2 virtual targets, 80640) and > 1 h on qip B00 order 2 (8 occupied, 40320);
# dip B11 (6 + 2, 1440) is seconds. Above the limit the terms are emitted as is.
PERM_SEARCH_MAX = 5040


def perm_search_size(spec: BlockSpec) -> int:
    from math import factorial
    names = spec.target
    nocc = sum(1 for n in names if n[0] in "ijklmno")
    return factorial(nocc) * factorial(len(names) - nocc)


def emit_block_file(outdir: Path, spec: BlockSpec,
                    orders: dict[int, ExprContainer | None],
                    emitter: TermEmitter) -> list[int]:
    """Write the block's evaluators; returns the non-vanishing orders."""
    parts = [go_header()]
    present = []
    for order, expr in sorted(orders.items()):
        fname = f"{spec.go_func.lower()}o{order}"
        if expr is None:
            parts.append(f"// {spec.block_str} vanishes at order {order}.")
            continue
        if perm_search_size(spec) <= PERM_SEARCH_MAX:
            with quiet():
                sym = exploit_perm_sym(
                    expr=expr, target_indices=spec.indices_str,
                    bra_ket_sym=spec.bra_ket_sym,
                    antisymmetric_result_tensor=True)
            groups = [(perm_sym, list(sub.terms)) for perm_sym, sub in
                      sym.items()]
        else:
            # every term as is: exploit_perm_sym only compresses the (already
            # complete) expression, and its search is factorial in the target size
            groups = [((), list(expr.terms))]
        parts.append(emit_expr_funcs(
            fname, spec.target, groups, emitter,
            f"{fname} is the order-{order} part of block {spec.block_str}, "
            f"<{','.join(spec.bra_names)}|M|{','.join(spec.ket_names)}>."))
        present.append(order)

    params = ", ".join(spec.target) + " int"
    args = ", ".join(spec.target)
    agg = [
        f"// {spec.go_func} is block {spec.block_str} of the secular matrix,",
        f"// <{','.join(spec.bra_names)}|M|{','.join(spec.ket_names)}>, summed "
        "over the orders the scheme keeps.",
        f"func ({RECV} *{TYPE}) {spec.go_func}({params}) float64 {{",
        "\tvar v float64",
    ]
    for order in present:
        agg.append(f"\tif {RECV}.maxOrder[{spec.index}] >= {order} {{")
        agg.append(f"\t\tv += {RECV}.{spec.go_func.lower()}o{order}({args})")
        agg.append("\t}")
    agg += ["\treturn v", "}"]
    parts.append("\n".join(agg))
    path = outdir / f"elements_{spec.go_func.lower()}_generated.go"
    path.write_text("\n\n".join(parts) + "\n")
    return present


def emit_types_go(outdir: Path, variant: str, k: int, gen: list[int],
                  present: dict[str, list[int]], schemes: dict[str, list[int]],
                  timings: dict[str, float]) -> None:
    """types.go: the runtime the generated evaluators build on."""
    def arr(xs):
        return "[6]int{" + ", ".join(str(x) for x in xs) + "}"

    scheme_lines = "\n".join(f'\t"{n}": {arr(o)},' for n, o in sorted(schemes.items()))
    nonzero = "[6]bool{" + ", ".join(
        "true" if present.get(b) else "false" for b in BLOCK_NAMES) + "}"
    cases = []
    for spec in all_blocks(k):
        if not present.get(spec.go_func):
            continue
        n = len(spec.target)
        args = ", ".join(f"a[{i}]" for i in range(n))
        cases.append(f"\tcase {spec.index}:\n\t\treturn {RECV}.{spec.go_func}({args})")
    order_cases = []
    for spec in all_blocks(k):
        args = ", ".join(f"idx[{i}]" for i in range(len(spec.target)))
        for order in present.get(spec.go_func, []):
            order_cases.append(
                f'\tcase block == "{spec.go_func}" && order == {order}:\n'
                f"\t\treturn {RECV}.{spec.go_func.lower()}o{order}({args}), true")
    timing = "\n".join(f"//\t{b:<8s} {t:8.1f} s" for b, t in timings.items())
    code = go_header() + f'''
import (
	"fmt"
	"math/bits"
	"slices"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// Variant is the ionization variant these evaluators were generated for.
const Variant = "{variant}"

// K is the main-class hole count: classes are Kh | (K+1)h1p | (K+2)h2p.
const K = {k}

// GeneratedOrders is the highest order generated per block, indexed as
// B00, B01, B11, B02, B12, B22 (-1: not generated).
var GeneratedOrders = {arr(gen)}

// Schemes maps each scheme name to its per-block maximum order (-1: the block
// is absent in that scheme).
var Schemes = map[string][6]int{{
{scheme_lines}
}}

// Derivation times of this generation (adcgen, seconds per block):
{timing}

// {TYPE} evaluates secular-matrix elements in the spin-orbital basis adcgen
// derives them in: spin orbital p has spatial orbital p>>1 and spin p&1,
// occupied spin orbitals are 0..nocc-1 and virtual a is absolute nocc+a. It
// assumes a CANONICAL Hartree-Fock reference (diagonal Fock matrix), which is
// what the generated expressions were simplified for. Intermediates are built
// once in New; afterwards {TYPE} only reads shared state and is safe for
// concurrent use.
type {TYPE} struct {{
	ints     *integrals.Store
	orbEps   []float64 // spatial orbital energies
	nocc     int       // occupied spin orbitals
	nvir     int       // virtual spin orbitals
	scheme   string
	maxOrder [6]int
	itmdStore
}}

// New builds the evaluator for a closed-shell reference with noccSpatial doubly
// occupied orbitals, truncated to the named scheme.
func New(ints *integrals.Store, eps []float64, noccSpatial int, scheme string) (*{TYPE}, error) {{
	mo, ok := Schemes[scheme]
	if !ok {{
		names := make([]string, 0, len(Schemes))
		for n := range Schemes {{
			names = append(names, n)
		}}
		slices.Sort(names)
		return nil, fmt.Errorf("%s: scheme %q not generated (have %v)", Variant, scheme, names)
	}}
	if 2*noccSpatial < K+2 && mo[5] >= 0 {{
		return nil, fmt.Errorf("%s: %d occupied spin orbitals cannot hold %d holes",
			Variant, 2*noccSpatial, K+2)
	}}
	el := &{TYPE}{{
		ints:     ints,
		orbEps:   eps,
		nocc:     2 * noccSpatial,
		nvir:     2 * ints.NVir(),
		scheme:   scheme,
		maxOrder: mo,
	}}
	el.buildItmds()
	return el, nil
}}

// Scheme is the scheme the evaluator was built for.
func ({RECV} *{TYPE}) Scheme() string {{ return {RECV}.scheme }}

// eps is the orbital energy of absolute spin orbital p.
func ({RECV} *{TYPE}) eps(p int) float64 {{ return {RECV}.orbEps[p>>1] }}

// fock is the canonical Fock matrix f_pq = eps_p delta_pq.
func ({RECV} *{TYPE}) fock(p, q int) float64 {{
	if p != q {{
		return 0
	}}
	return {RECV}.orbEps[p>>1]
}}

// asym is <pq||rs> over absolute spin orbitals: <pq|rs> - <pq|sr>, with
// <pq|rs> = (pr|qs) delta(sp,sr) delta(sq,ss).
func ({RECV} *{TYPE}) asym(p, q, r, s int) float64 {{
	var v float64
	if p&1 == r&1 && q&1 == s&1 {{
		v = {RECV}.ints.Eri(p>>1, r>>1, q>>1, s>>1)
	}}
	if p&1 == s&1 && q&1 == r&1 {{
		v -= {RECV}.ints.Eri(p>>1, s>>1, q>>1, r>>1)
	}}
	return v
}}

// nonzero marks the blocks with at least one non-vanishing generated order. A block a
// scheme enables can still vanish at every generated order (kh/(k+2)h2p starts at
// second order, so the ci scheme's B02 is identically zero).
var nonzero = {nonzero}

// blockOf maps (bra class, ket class), bra <= ket, to the block index.
var blockOf = [3][3]int{{{{0, 1, 3}}, {{-1, 2, 4}}, {{-1, -1, 5}}}}

// Element is <r|M|c> for rows r, c of a khci.Space with the same K. Rows follow
// the khci phase convention, which is adcgen's; the lower triangle uses the
// symmetry of M. A block the scheme leaves out is zero.
func ({RECV} *{TYPE}) Element(sp *khci.Space, r, c int) float64 {{
	cr, cc := sp.Class(r)-K, sp.Class(c)-K
	if cr > cc {{
		r, c = c, r
		cr, cc = cc, cr
	}}
	b := blockOf[cr][cc]
	if {RECV}.maxOrder[b] < 0 || !nonzero[b] {{
		return 0
	}}
	var buf [16]int
	a := buf[:0]
	a = appendRow(a, sp, r)
	a = appendRow(a, sp, c)
	return {RECV}.block(b, a)
}}

// appendRow appends a row's target indices in adcgen order: particles
// (relative virtual spin orbitals), then holes (spin orbitals), ascending.
func appendRow(a []int, sp *khci.Space, r int) []int {{
	a = sp.PartSO(r, a)
	for q := sp.HoleMask(r); q != 0; q &= q - 1 {{
		a = append(a, bits.TrailingZeros64(q))
	}}
	return a
}}

func ({RECV} *{TYPE}) block(b int, a []int) float64 {{
	switch b {{
{chr(10).join(cases)}
	}}
	panic(fmt.Sprintf("%s: block %d requested but not generated", Variant, b))
}}

// BuildDense assembles the full matrix over sp, rows in parallel (each worker
// owns its rows).
func ({RECV} *{TYPE}) BuildDense(sp *khci.Space) backend.Mat {{
	n := sp.Size()
	m := backend.NewMat(n, n)
	parallel.HeavyRows(n, func(r int) {{
		for c := range n {{
			m.Data[r*n+c] = {RECV}.Element(sp, r, c)
		}}
	}})
	return m
}}

// orderPart evaluates the order-th part of a block by name at the block's target
// indices (bra particles, bra holes, ket particles, ket holes). ok is false for
// an order that vanishes identically or was not generated. It serves the adcgen
// fidelity test.
func ({RECV} *{TYPE}) orderPart(block string, order int, idx []int) (v float64, ok bool) {{
	switch {{
{chr(10).join(order_cases)}
	}}
	return 0, false
}}
'''
    (outdir / "types.go").write_text(code)


def emit_fidelity_test(outdir: Path, dense_class: int) -> None:
    """fidelity_test.go: replays testdata/adcgen_ref.json; symmetry and Ms checks."""
    code = f'''// Code generated by adcgen pipeline (scripts/codegen/generate_adc.py). DO NOT EDIT.

package {PACKAGE}

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

type adcgenRef struct {{
	Variant string `json:"variant"`
	Fcidump string `json:"fcidump"`
	Entries []struct {{
		Block string  `json:"block"`
		Order int     `json:"order"`
		Idx   []int   `json:"idx"`
		Value float64 `json:"value"`
	}} `json:"entries"`
}}

type refSystem struct {{
	d    *fcidump.Data
	nocc int
	eps  []float64
	ints *integrals.Store
}}

func loadRef(t *testing.T) (adcgenRef, refSystem) {{
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "adcgen_ref.json"))
	if err != nil {{
		t.Fatalf("read reference (regenerate with scripts/codegen/generate_adc.py): %v", err)
	}}
	var ref adcgenRef
	if err := json.Unmarshal(raw, &ref); err != nil {{
		t.Fatalf("parse reference: %v", err)
	}}
	d, err := fcidump.Read(strings.NewReader(ref.Fcidump))
	if err != nil {{
		t.Fatalf("reference fcidump: %v", err)
	}}
	nocc := mp.NOcc(d)
	return ref, refSystem{{d, nocc, mp.OrbitalEnergies(d, nocc), integrals.New(d, nocc, nil)}}
}}

// widestScheme is the generated scheme that keeps the most orders.
func widestScheme() string {{
	best, bestSum := "", -1
	names := make([]string, 0, len(Schemes))
	for n := range Schemes {{
		names = append(names, n)
	}}
	sort.Strings(names)
	for _, n := range names {{
		s := 0
		for _, o := range Schemes[n] {{
			s += o + 1
		}}
		if s > bestSum {{
			best, bestSum = n, s
		}}
	}}
	return best
}}

// TestMatchesAdcgenReference: every generated block and order reproduces adcgen's
// fully expanded expression, evaluated independently in numpy at probe
// configurations of a random canonical system.
func TestMatchesAdcgenReference(t *testing.T) {{
	ref, sys := loadRef(t)
	if ref.Variant != Variant {{
		t.Fatalf("reference is for %s, code for %s: regenerate", ref.Variant, Variant)
	}}
	el, err := New(sys.ints, sys.eps, sys.nocc, widestScheme())
	if err != nil {{
		t.Fatal(err)
	}}
	type key struct {{
		block string
		order int
	}}
	worst := map[key]float64{{}}
	n := 0
	for _, e := range ref.Entries {{
		got, ok := el.orderPart(e.Block, e.Order, e.Idx)
		if !ok {{
			if math.Abs(e.Value) > 1e-13 {{
				t.Fatalf("%s order %d vanishes in Go but reference %v = %g", e.Block, e.Order, e.Idx, e.Value)
			}}
			continue
		}}
		n++
		k := key{{e.Block, e.Order}}
		worst[k] = max(worst[k], math.Abs(got-e.Value))
	}}
	if n == 0 {{
		t.Fatal("no reference entries matched a generated order")
	}}
	for k, w := range worst {{
		if w > 1e-12 {{
			t.Errorf("%s order %d: max deviation %g from the adcgen reference", k.block, k.order, w)
		}}
	}}
}}

// TestSymmetricAndMsDegenerate: the matrix over the generated classes is symmetric,
// and the spin-flip partner sectors 2Ms = +s and -s have identical spectra.
func TestSymmetricAndMsDegenerate(t *testing.T) {{
	_, sys := loadRef(t)
	be := backend.Gonum{{}}
	for _, scheme := range func() []string {{
		names := make([]string, 0, len(Schemes))
		for n := range Schemes {{
			names = append(names, n)
		}}
		sort.Strings(names)
		return names
	}}() {{
		el, err := New(sys.ints, sys.eps, sys.nocc, scheme)
		if err != nil {{
			t.Fatal(err)
		}}
		ms := 2 - K%2 // 2Ms = 1 for odd K, 2 for even K
		var spectra [2][]float64
		for s, two := range []int{{ms, -ms}} {{
			sp, err := khci.NewSpace(khci.Options{{K: K, NOcc: sys.nocc, NVir: sys.d.NORB - sys.nocc,
				TwoMs: two, MaxClass: {dense_class}}})
			if err != nil {{
				t.Fatal(err)
			}}
			M := el.BuildDense(sp)
			for r := range M.Rows {{
				for c := range r {{
					if d := math.Abs(M.At(r, c) - M.At(c, r)); d > 1e-12 {{
						t.Fatalf("%s 2Ms=%d: M[%d,%d] asymmetric by %g", scheme, two, r, c, d)
					}}
				}}
			}}
			spectra[s], _ = be.SymEig(M)
			sort.Float64s(spectra[s])
		}}
		if len(spectra[0]) != len(spectra[1]) {{
			t.Fatalf("%s: sector sizes differ", scheme)
		}}
		for i := range spectra[0] {{
			if d := math.Abs(spectra[0][i] - spectra[1][i]); d > 1e-10 {{
				t.Fatalf("%s root %d: %g vs %g", scheme, i, spectra[0][i], spectra[1][i])
			}}
		}}
	}}
}}
'''
    (outdir / "fidelity_test.go").write_text(code)


def emit_doc(outdir: Path, variant: str, k: int, schemes: list[str],
             gen: list[int]) -> None:
    names = {1: "single", 2: "double", 3: "triple", 4: "quadruple"}
    cls = f"{k}h | {k + 1}h1p | {k + 2}h2p"
    blocks = ", ".join(f"{b}<={o}" for b, o in zip(BLOCK_NAMES, gen) if o >= 0)
    code = f'''// Code generated by adcgen pipeline (scripts/codegen/generate_adc.py). DO NOT EDIT.

// Package {PACKAGE} holds adcgen-generated ISR secular-matrix element evaluators
// for {names[k]} ionization ({variant}, classes {cls}).
//
// Generated with schemes {", ".join(schemes)}; blocks {blocks}.
// Regenerate with
//
//	python3.12 scripts/codegen/generate_adc.py --variant {variant} --schemes {",".join(schemes)} \\
//	    --outdir internal/adc/isrgen/{variant}
//
// fidelity_test.go (generated) replays testdata/adcgen_ref.json, an independent
// numpy evaluation of adcgen's fully expanded expressions.
package {PACKAGE}
'''
    (outdir / "doc.go").write_text(code)


def collect_all_itmds(gs, emitter, log):
    t0 = time.time()
    itmds = collect_itmds(gs, set(emitter.itmd_names), emitter)
    log(f"  {len(itmds)} intermediates: "
        f"{', '.join(d.name for d, _ in itmds)} ({time.time() - t0:.1f}s)")
    return itmds


# ---------------------------------------------------------------------------
# Fidelity reference
# ---------------------------------------------------------------------------
REF_NORB, REF_NOCC, REF_SEED, REF_PROBES = 6, 3, 7, 300


def random_system(np):
    """Random CANONICAL closed-shell system.

    Random 8-fold symmetric (pq|rs) and chosen orbital energies eps; h is then
    set to h_pq = eps_p delta_pq - sum_i [2(pq|ii) - (pi|iq)], which makes the
    Fock matrix exactly diagonal with diagonal eps. The generated code assumes a
    canonical reference and mp.OrbitalEnergies rebuilds eps from h and the ERIs,
    so the Go side and numpy see the same eps and the same diagonal Fock.
    Returns (FCIDUMP text, spin-orbital h, spin-orbital <pq||rs>, spin-orbital
    eps, E_HF electronic).
    """
    norb, nocc = REF_NORB, REF_NOCC
    rng = np.random.default_rng(REF_SEED)
    eri = rng.normal(scale=0.05, size=(norb,) * 4)
    perms = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
             (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]
    eri = sum(eri.transpose(p) for p in perms) / 8
    eps = np.concatenate([np.linspace(-2.0, -1.0, nocc),
                          np.linspace(0.4, 1.6, norb - nocc)])
    h = np.diag(eps).astype(float)
    for i in range(nocc):
        h -= 2 * eri[:, :, i, i] - eri[:, i, i, :]

    lines = [f" &FCI NORB={norb},NELEC={2 * nocc},MS2=0,",
             f"  ORBSYM={','.join(['1'] * norb)},", "  ISYM=1,", " &END"]
    for p in range(norb):
        for q in range(p + 1):
            for r in range(norb):
                for s in range(r + 1):
                    if p * norb + q >= r * norb + s:
                        lines.append(f"{eri[p, q, r, s]:.17e} {p + 1} {q + 1} "
                                     f"{r + 1} {s + 1}")
    for p in range(norb):
        for q in range(p + 1):
            lines.append(f"{h[p, q]:.17e} {p + 1} {q + 1} 0 0")
    lines.append(f"{0.0:.17e} 0 0 0 0")

    so = np.arange(2 * norb)
    sp, spin = so >> 1, so & 1
    chem = eri[np.ix_(sp, sp, sp, sp)]
    same_pr = spin[:, None, None, None] == spin[None, None, :, None]
    same_qs = spin[None, :, None, None] == spin[None, None, None, :]
    phys = chem.transpose(0, 2, 1, 3) * same_pr * same_qs  # <pq|rs>
    asym = phys - phys.transpose(0, 1, 3, 2)
    hso = h[np.ix_(sp, sp)] * (spin[:, None] == spin[None, :])
    no = 2 * nocc
    ehf = sum(hso[i, i] for i in range(no)) + 0.5 * sum(
        asym[i, j, i, j] for i in range(no) for j in range(no))
    # the construction must have made F diagonal
    F = hso + np.einsum("piqi->pq", asym[:, :no, :, :no])
    assert np.abs(F - np.diag(np.diag(F))).max() < 1e-12
    assert np.abs(np.diag(F) - eps[sp]).max() < 1e-12
    return "\n".join(lines) + "\n", hso, asym, eps[sp], float(ehf)


def expanded_block(sm: SecularMatrix, spec: BlockSpec, order: int
                   ) -> ExprContainer | None:
    """The block with every amplitude expanded to ERIs and orbital energies."""
    with quiet():
        expr = ExprContainer(sm.isr_matrix_block(
            order=order, block=spec.block_str, indices=spec.indices_str),
            real=True)
        expr.substitute_contracted()
        expr = simplify(expr)
        if expr.inner is S.Zero:
            return None
        expr.diagonalize_fock()
        expr = expr.expand_intermediates(fully_expand=True).expand()
    return None if expr.inner is S.Zero else expr


def eval_at(np, expr, values, asym, e_so, no):
    """expr at fixed target index values (absolute spin orbitals), in numpy.

    Every term is contracted over its non-target indices only, with each tensor
    sliced at the fixed target positions: the cost is independent of the size of
    the full target tensor (2e9 elements for the qip 5h1p/5h1p block).
    """
    from adcgen.sympy_objects import KroneckerDelta
    nso = len(e_so)
    ranges = {"o": np.arange(no), "v": np.arange(no, nso)}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    total = 0.0
    for term in expr.terms:
        split = term.split_orb_energy()
        frac = split["num"].inner / split["denom"].inner
        free = sorted({i for i in term.idx if i.name not in values},
                      key=lambda i: i.name)
        L = {ix: letters[n] for n, ix in enumerate(free)}
        pos = {ix: n for n, ix in enumerate(free)}

        def walk(y):
            if y.is_Number:
                return float(y)
            if isinstance(y, SymbolicTensor):
                ix = y.idx[0]
                if ix.name in values:
                    return e_so[values[ix.name]]
                shape = [1] * len(free)
                shape[pos[ix]] = -1
                return e_so[ranges[_space_of(ix)]].reshape(shape)
            if isinstance(y, Add):
                return sum(walk(a) for a in y.args)
            if isinstance(y, Mul):
                r = 1.0
                for a in y.args:
                    r = r * walk(a)
                return r
            if isinstance(y, Pow):
                return walk(y.args[0]) ** int(y.args[1])
            raise NotImplementedError(f"cannot evaluate {y}")

        fshape = [len(ranges[_space_of(i)]) for i in free]
        ops = [np.broadcast_to(np.asarray(walk(frac), dtype=float), fshape)]
        subs = ["".join(L[i] for i in free)]
        rem = split["remainder"]
        for o in (rem.terms[0].objects if len(rem) else ()):
            if o.inner.is_number:
                ops.append(np.array(float(o.inner)))
                subs.append("")
                continue
            base, exp = o.base_and_exponent
            sel = tuple(values[i.name] if i.name in values else ranges[_space_of(i)]
                        for i in o.idx)
            if isinstance(base, KroneckerDelta):
                full = np.eye(nso)
            elif base.name == tensor_names.eri:
                full = asym
            elif base.name == tensor_names.fock:
                full = np.diag(e_so)
            else:
                raise NotImplementedError(f"object {o} in an expanded block")
            arr = full[np.ix_(*[np.atleast_1d(s) for s in sel])]
            arr = arr.reshape([len(np.atleast_1d(s)) for s, i in zip(sel, o.idx)
                               if i.name not in values])
            sub = "".join(L[i] for i in o.idx if i.name not in values)
            for _ in range(int(exp)):
                ops.append(arr)
                subs.append(sub)
        total += float(np.einsum(",".join(subs) + "->", *ops, optimize=True))
    return total


def det_of(nso_occ, holes, parts):
    """Canonical occupation list and sign of c+_{a1} c+_{a2} c_{i1}...c_{in}|Phi0>.

    holes: occupied spin orbitals, parts: absolute virtual spin orbitals, both
    ascending. Operators act right to left. An independent re-derivation of
    khci.Space.Det, used by the Slater-Condon reference.
    """
    occ = list(range(nso_occ))
    sign = 1
    for p in reversed(holes):
        pos = occ.index(p)
        if pos % 2:
            sign = -sign
        occ.pop(pos)
    for x in reversed(parts):
        below = sum(1 for y in occ if y < x)
        if below % 2:
            sign = -sign
        occ.insert(below, x)
    return occ, sign


def _apply_string(occ, creators, annihilators):
    """Apply the operator string c+_{creators...} c_{annihilators...}, written
    left to right, to the canonical determinant with sorted occupation `occ`.

    The rightmost operator acts first. Returns (sorted result, sign), or
    (None, 0) when the string annihilates the determinant. For the double
    excitation c+_{t1} c+_{t2} c_{f2} c_{f1} pass creators=[t1, t2],
    annihilators=[f2, f1].
    """
    cur = list(occ)
    sign = 1
    for p in reversed(annihilators):
        if p not in cur:
            return None, 0
        pos = cur.index(p)
        if pos % 2:
            sign = -sign
        cur.pop(pos)
    for x in reversed(creators):
        if x in cur:
            return None, 0
        below = sum(1 for y in cur if y < x)
        if below % 2:
            sign = -sign
        cur.insert(below, x)
    return cur, sign


def slater_condon(A, B, hso, asym):
    """<A|H|B> for sorted occupation lists (electronic, no nuclear repulsion)."""
    sa, sb = set(A), set(B)
    to = sorted(sa - sb)
    frm = sorted(sb - sa)
    if len(to) != len(frm) or len(to) > 2:
        return 0.0
    if not to:
        return (sum(hso[i, i] for i in A)
                + 0.5 * sum(asym[i, j, i, j] for i in A for j in A))
    # phase: <A| c+_{t...} c_{f...} |B> with the string applied to B
    if len(to) == 1:
        res, sg = _apply_string(B, to, frm)
        if res != list(A):
            raise AssertionError("single excitation did not map B onto A")
        t, f = to[0], frm[0]
        common = [j for j in A if j != t]
        return sg * (hso[t, f] + sum(asym[t, j, f, j] for j in common))
    t1, t2 = to
    f1, f2 = frm
    # c+_{t1} c+_{t2} c_{f2} c_{f1}: c_{f1} acts first
    res, sg = _apply_string(B, [t1, t2], [f2, f1])
    if res != list(A):
        raise AssertionError("double excitation did not map B onto A")
    return sg * asym[t1, t2, f1, f2]


def sample_probes(np, rng, spec: BlockSpec, no: int, nv: int, n: int):
    """Probe (bra, ket) configurations for a block, as target index values.

    A bra configuration is drawn at random; the ket is built from it by adding
    the extra hole/particle pairs its class needs (same spin, so Ms matches)
    and then 0-2 random spin-preserving substitutions. Random independent pairs
    would almost all differ by more than two spin orbitals and test only zeros.
    """
    k = spec.k
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        nh, npart = k + spec.cb, spec.cb
        if nh > no or npart > nv:
            break
        holes = sorted(rng.choice(no, nh, replace=False).tolist())
        parts = sorted(rng.choice(nv, npart, replace=False).tolist())
        kh, kp = list(holes), list(parts)
        ok = True
        for _ in range(spec.ck - spec.cb):
            spin = int(rng.integers(2))
            fh = [p for p in range(no) if p % 2 == spin and p not in kh]
            fp = [a for a in range(nv) if a % 2 == spin and a not in kp]
            if not fh or not fp:
                ok = False
                break
            kh.append(int(rng.choice(fh)))
            kp.append(int(rng.choice(fp)))
        if not ok:
            continue
        for _ in range(int(rng.integers(3))):
            if rng.integers(2) == 0 and kh:
                x = int(rng.choice(kh))
                alt = [p for p in range(no) if p % 2 == x % 2 and p not in kh]
                if alt:
                    kh[kh.index(x)] = int(rng.choice(alt))
            elif kp:
                x = int(rng.choice(kp))
                alt = [a for a in range(nv) if a % 2 == x % 2 and a not in kp]
                if alt:
                    kp[kp.index(x)] = int(rng.choice(alt))
        kh.sort()
        kp.sort()
        out.append((parts, holes, kp, kh))
    return out


def emit_reference(path: Path, sm: SecularMatrix, variant: str, k: int,
                   gen_orders: list[int], log) -> None:
    """Write the fidelity reference JSON (see module docstring)."""
    import numpy as np

    fcidump, hso, asym, e_so, ehf = random_system(np)
    no, nso = 2 * REF_NOCC, 2 * REF_NORB
    nv = nso - no
    rng = np.random.default_rng(REF_SEED + 1)
    entries, ci = [], []
    for spec in all_blocks(k):
        probes = sample_probes(np, rng, spec, no, nv, REF_PROBES)
        if not probes:
            log(f"  reference {spec.go_func}: class does not fit the "
                f"{REF_NORB}-orbital reference system, skipped")
            continue
        names = spec.target
        t0 = time.time()
        # adcgen values for the generated orders plus orders 0 and 1 (what the
        # Slater-Condon check compares against). A block that is not generated
        # gets Slater-Condon values only: deriving it would be the expensive
        # part (k=3 5h2p/5h2p order 1 > 10 min), and khci's CI gate needs only CI.
        generated = gen_orders[spec.index] >= 0
        orders = (sorted(set(range(gen_orders[spec.index] + 1)) | {0, 1})
                  if generated else [])
        per_order = {}
        for order in orders:
            expr = expanded_block(sm, spec, order)
            vals = []
            for bp, bh, kp, kh in probes:
                if expr is None:
                    vals.append(0.0)
                    continue
                idx = bp + bh + kp + kh
                values = {}
                for nme, v in zip(names, idx):
                    values[nme] = v if nme[0] in "ijklmno" else no + v
                vals.append(eval_at(np, expr, values, asym, e_so, no))
            per_order[order] = vals
            if order <= gen_orders[spec.index]:
                for (bp, bh, kp, kh), v in zip(probes, vals):
                    entries.append({"block": spec.go_func, "order": order,
                                    "idx": bp + bh + kp + kh, "value": v})
        # Slater-Condon CI in the khci phase convention
        worst = 0.0
        for n_, (bp, bh, kp, kh) in enumerate(probes):
            A, sa = det_of(no, bh, [no + a for a in bp])
            B, sb = det_of(no, kh, [no + a for a in kp])
            v = sa * sb * slater_condon(A, B, hso, asym)
            if A == B:
                v -= ehf
            if generated and spec.go_func != "B02":
                isr = per_order[0][n_] + per_order[1][n_]
                worst = max(worst, abs(isr - v))
            ci.append({"block": spec.go_func, "idx": bp + bh + kp + kh, "value": v})
        if spec.go_func == "B02":
            check = ("kh/(k+2)h2p: CI couples at first order, the ISR only from second "
                     "order (no equality asserted)")
        elif generated:
            check = f"ISR(<=1) vs Slater-Condon CI max |diff| {worst:.2e}"
        else:
            check = "not generated: Slater-Condon CI values only"
        log(f"  reference {spec.go_func}: {len(probes)} probes, orders {orders}, "
            f"{check} ({time.time() - t0:.1f}s)")
        if worst > 1e-12:
            raise RuntimeError(f"{spec.go_func}: first-order ISR differs from CI in "
                               f"the khci phase convention by {worst:.2e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "variant": variant, "k": k, "norb": REF_NORB, "nocc": REF_NOCC,
        "e_hf": ehf, "fcidump": fcidump,
        "index_convention": "bra particles, bra holes, ket particles, ket holes; "
                            "holes are occupied spin orbitals, particles relative "
                            "virtual spin orbitals, each ascending",
        "entries": entries, "ci_entries": ci}, indent=0) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global PACKAGE
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--schemes", required=True,
                   help="comma-separated: strict:N, ci, adc2x, adc22m, adc22x, adc22f")
    p.add_argument("--outdir", help="output directory for the Go package")
    p.add_argument("--package", help="Go package name (default: the variant)")
    p.add_argument("--blocks", help="comma-separated subset of B00,B01,B11,B02,B12,B22")
    p.add_argument("--no-reference", action="store_true",
                   help="skip testdata/adcgen_ref.json (needs numpy)")
    p.add_argument("--reference-only", metavar="PATH",
                   help="write only the reference JSON to PATH; no Go")
    a = p.parse_args()
    if not a.outdir and not a.reference_only:
        p.error("--outdir is required unless --reference-only is given")
    k = VARIANTS[a.variant]
    PACKAGE = a.package or a.variant
    log = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    set_log_level("WARNING")

    if k >= 3:
        log("checking KHoleStates against adcgen's native dip ...")
        check_khole_subclass()
    gs, sm = init_adcgen(k)
    schemes = {s: scheme_orders(sm, k, s) for s in a.schemes.split(",")}
    gen = [max(o[i] for o in schemes.values()) for i in range(6)]
    if a.blocks:
        keep = set(a.blocks.split(","))
        bad = keep - set(BLOCK_NAMES)
        if bad:
            p.error(f"unknown blocks {sorted(bad)}")
        gen = [o if BLOCK_NAMES[i] in keep else -1 for i, o in enumerate(gen)]
        for s in schemes:
            schemes[s] = [o if gen[i] >= 0 else -1 for i, o in enumerate(schemes[s])]
    log(f"{a.variant} (k={k}): per-block max orders "
        f"{dict(zip(BLOCK_NAMES, gen))}; schemes {schemes}")

    if a.reference_only:
        emit_reference(Path(a.reference_only), sm, a.variant, k, gen, log)
        log("Done.")
        return

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*_generated.go"):
        old.unlink()
    emitter = TermEmitter(set())
    present: dict[str, list[int]] = {}
    timings: dict[str, float] = {}
    for spec in all_blocks(k):
        if gen[spec.index] < 0:
            continue
        t_block = time.time()
        orders: dict[int, ExprContainer | None] = {}
        for order in range(gen[spec.index] + 1):
            t0 = time.time()
            orders[order] = derive_block(sm, spec, order)
            nterms = len(orders[order]) if orders[order] is not None else 0
            log(f"  {spec.go_func} {spec.block_str:<18} order {order}: {nterms:5d} terms "
                f"({time.time() - t0:.1f}s)")
        present[spec.go_func] = emit_block_file(outdir, spec, orders, emitter)
        timings[spec.go_func] = time.time() - t_block
    itmds = collect_all_itmds(gs, emitter, log)
    emit_itmd_file(outdir, itmds)
    emit_types_go(outdir, a.variant, k, gen, present, schemes, timings)
    # dense self-test classes: the largest class whose blocks are all generated
    dense = k
    for top in (1, 2):
        need = [BLOCK_PAIRS.index((x, top)) for x in range(top + 1)]
        if all(gen[i] >= 0 for i in need):
            dense = k + top
        else:
            break
    emit_fidelity_test(outdir, dense)
    emit_doc(outdir, a.variant, k, sorted(schemes), gen)
    gofmt = shutil.which("gofmt")
    if gofmt is None:
        sys.exit("gofmt not found in PATH; generated files are unformatted")
    subprocess.run([gofmt, "-w", str(outdir)], check=True)
    if not a.no_reference:
        emit_reference(outdir / "testdata" / "adcgen_ref.json", sm, a.variant, k,
                       gen, log)
    log("Done.")


if __name__ == "__main__":
    main()
