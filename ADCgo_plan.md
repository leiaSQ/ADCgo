# Plan: **ADCgo** — an exact, hardware-accelerated ADC(n) ionization solver

## Context

Today `ADCanalysis` is a **post-processing** tool: it parses the text output of
`theADCcode` (the C++/F77 Green's-function ADC program in `../ADC`, Golubev/
Cederbaum lineage), classifies dicationic/cationic states into decay channels
(Auger / ICD / ETMD), and renders stick spectra with `gonum/plot`. It contains
**no solver core** — no integrals, no matrices, no eigensolver (a grep for
`gonum/mat` returns nothing; the only gonum use is `plot`). See
`decay_analyzer_sketch.md` and `internal/model/types.go`.

The question this plan answers: **what is the lift to compute ADC(n) in Go**,
rather than only parse someone else's ADC output — and how to carve a niche the
Dreuw group has left open.

### The angle: exact treatment of *larger* systems

The Dreuw world reaches big molecules by **approximating** (RI/density fitting,
CVS truncation, reduced-scaling schemes). ADCgo takes the opposite bet: keep the
ionization ADC secular problem **exact — no reduced-scaling truncations** — and
reach larger systems by **throwing modern hardware at it** instead of throwing
away terms:

- **GPU Lanczos** — the ADC matrix–vector product (the hotspot) runs as batched
  `cuBLAS` GEMM/contractions on the GPU; the Lanczos recurrence is orchestrated
  on the CPU. The matrix `M` is real-symmetric, so Lanczos (band/block) recovers
  the *whole* photoionization band, which is exactly what a DIP/SIP spectrum
  needs — not just a few extremal roots.
- **Multicore CPU BLAS** — back Gonum with a threaded BLAS (OpenBLAS/MKL via
  `netlib` cgo) and goroutine-parallel contraction loops, so the mat-vec uses
  all cores. `theADCcode` leans on archaic, effectively single-threaded
  LAPACK/Lanczos; simply using every core is a real, measurable win before any
  GPU work.

"Exact" here means **no further physical approximation on top of the chosen ADC
order** (no RI, no CVS, no frozen truncation of the satellite space) — the full
secular problem, *accelerated* rather than *truncated*. It does not claim ADC(2)
itself is exact quantum chemistry.

### The competitive landscape (why this is a real niche, not a clone)

| | Dreuw / **adcc** | Cederbaum / **theADCcode** (`../ADC`) | **ADCgo** |
|---|---|---|---|
| Propagator | Polarization (neutral exc.) | **Ionization: IP / DIP / EA**, CVS | **DIP first, then SIP** |
| Reaching big systems | **approximate** (RI, CVS, reduced-scaling) | small/medium, exact | **exact + GPU/multicore** |
| Decay widths | ✗ | ✓ Fano / CAP / Stieltjes | later milestone |
| SCF + integrals | delegated (pyscf/psi4) | delegated (GAMESS-UK/Molcas, libphis) | **delegated (FCIDUMP/pyscf)** |
| Parallelism | C++ tensor lib | archaic ~single-thread BLAS | **cuBLAS GPU + threaded BLAS** |

**Key insight:** adcc *deliberately omits* the entire IP/DIP-ADC and decay
branch — that branch lives only in hard-to-build legacy code (`../ADC` needs
conda gcc8 + libphis + a GAMESS-UK frontend) that does not exploit GPUs or many
cores. Both mature codes already **delegate SCF and integrals**. So the value is
not in re-deriving integrals — it is in a **fast, exact, GPU-capable ionization
solver** that lets you push DIP/ICD physics to systems the legacy code stalls on.

### Decisions locked in

1. **Integrals/SCF: delegate via FCIDUMP** from pyscf
   (`pyscf.tools.fcidump.from_scf`). ADCgo implements *only* ADC. Smallest lift,
   fully portable, mirrors both reference codes.
2. **First method: DIP-ADC(2)** (double ionization). Its 2h / **3h1p** satellite
   space is far larger and more compute-heavy than SIP even at non-Dyson
   ADC(3) — so it is the honest stress-test that justifies (and validates) the
   GPU/multicore performance thesis from day one, and it feeds your existing
   FA_DIP decay-channel analyzer directly.
3. **Performance-first, not single-binary.** The old "one static binary" goal is
   dropped: reaching larger systems needs cgo (OpenBLAS, cuBLAS). Keep a pure-Go
   fallback behind a build tag for portability/CI, but the default build links a
   threaded BLAS and optionally CUDA.

---

## Architecture

Extend the existing `internal/` layout (do **not** fork a new repo). The current
`model` types and golden parser tests become the **validation oracle**: ADCgo's
states / pole strengths / two-hole populations must reproduce the parsed
reference tables (the "v2" idea in `decay_analyzer_sketch.md` §3, §5).

```
internal/
├── model/        # EXISTING — reuse MO, State, SIPState, PopRow, Config …
├── parse/        # EXISTING — reference oracle for validation
├── classify/     # EXISTING — solver output feeds straight into this
├── spectrum/     # EXISTING
└── adc/          # NEW — the ADCgo engine
    ├── fcidump/      # FCIDUMP reader → h_pq, (pq|rs), ε_p, nocc, symmetry
    ├── integrals/    # MO-integral store: occ/vir-blocked, permutation-symmetric
    │                 #   (design borrowed from ../ADC integral_table.hpp)
    ├── mp/           # Møller–Plesset ground state: t2 amplitudes, MP2 energy
    ├── isr/          # DIP config spaces (2h, 3h1p) + indexing; later 1h/2h1p (IP)
    ├── matvec/       # σ = M·x contractions per order — backend-agnostic
    ├── backend/      # linear-algebra backend interface + impls:
    │                 #   gonum (pure-Go) | openblas (netlib cgo) |
    │                 #   cuda (cuBLAS cgo, NVIDIA) | hip (hipBLAS cgo, AMD Instinct)
    ├── lanczos/       # band/block Lanczos driver (spectrum + pole strengths)
    └── analyze/      # eigenvectors → pole strengths, 2h-populations (Eqs 3/4/8)
cmd/
└── adcgo/        # NEW CLI: FCIDUMP + config → states JSON (same schema)
```

The **`backend` interface** is the load-bearing abstraction: `matvec` expresses
the ADC mat-vec as a sequence of GEMM/contraction calls against an interface, and
the same Lanczos driver runs unchanged on pure-Go Gonum, threaded OpenBLAS, or
cuBLAS. This is what makes "CPU now, GPU later" a swap, not a rewrite.

**Deliberate non-goals** (same scope discipline adcc/theADCcode enforce): no AO
integral evaluation, no SCF, no AO→MO transform, no basis-set library. FCIDUMP is
the single ingestion path; the `vfile`/`dfile` GAMESS-UK dumps stay unused.

### Dependencies to add

- `gonum.org/v1/gonum/mat`, `blas/blas64` — dense algebra + the subspace/Ritz
  work; pure-Go fallback backend.
- `gonum.org/v1/netlib` (cgo) → **OpenBLAS/MKL** — the multicore CPU backend.
- **cuBLAS** via a thin cgo wrapper (or an existing Go CUDA binding) — the NVIDIA
  GPU backend. Isolated behind the `backend` interface and a `cuda` build tag.
- **hipBLAS** (ROCm) via an analogous cgo wrapper — the **AMD Instinct** GPU
  backend, behind the same interface and a `hip` build tag. Because hipBLAS mirrors
  the cuBLAS API (`hipblasDgemv`/`hipblasDaxpy` ↔ `cublasDgemv`/`cublasDaxpy`), the
  two GPU backends share the real-GEMV/AXPY contraction surface; only the runtime,
  memory-management, and library-linkage cgo shims differ.

---

## The ADC building blocks (DIP branch, concrete)

For **DIP-ADC** the target is the two-electron-removal secular problem
`M X = ω X`, ω = double-ionization energies, pole strengths from the 2h part of
X. Configuration spaces: **2h** (main) and **3h1p** (satellite — the large one).

1. **FCIDUMP ingest** → `h_pq`, `(pq|rs)` in MO basis, `ε_p`, `nocc`/`nvir`,
   `ORBSYM`. Store ERIs occ/vir-blocked and permutation-symmetric.
2. **MP2 ground state** → `t2_{ijab}`, MP2 energy as the first correctness gate
   (compare to pyscf to ~1e-8 Ha). Needed regardless of branch.
3. **DIP-ADC(2) mat-vec** `σ = M·x` (never build M densely), as contractions
   routed through `backend`:
   - 2h/2h block: 1st/2nd-order terms in the two-hole space.
   - 2h ↔ 3h1p coupling (1st order).
   - 3h1p/3h1p diagonal (zeroth order = orbital-energy differences). This block
     dominates dimension and cost — the reason DIP is the performance target.
4. **Lanczos eigensolver**: band/block Lanczos to sweep the full DIP band; pole
   strengths from the 2h weight of each Ritz vector. Mirrors `../ADC`'s
   `libLanczos` / `adc2_dip`, but with the mat-vec on GPU/threaded BLAS.
5. **Analysis**: two-hole population via the Tarantelli–Cederbaum U-transform +
   overlap metric (`decay_analyzer_sketch.md` §3, Eqs 3/4/8). Emit the **same
   JSON schema** the analyzer already consumes, so `classify`/`spectrum`/
   `plotspec` work unchanged.

SIP (IP-ADC(2) → nD-ADC(3), 1h/2h1p, `../ADC/ndadc3_ip`) is the same machinery on
a smaller config space — a later milestone, validated against your `ndadc3ip`
`ADC.out` files.

---

## Milestones & honest lift estimate

Solo, part-time, integrals delegated. Large but bounded; delegation is what
keeps it bounded, and the `backend` interface is what keeps the GPU work from
being a second project.

| # | Deliverable | Rough lift |
|---|---|---|
| **M0** | FCIDUMP reader + MO-integral store + **MP2 energy** matching pyscf | 1–2 wks — correctness foundation |
| **M1** | **DIP-ADC(2)** mat-vec + Lanczos on the **pure-Go/threaded-BLAS** backend → dication states + pole strengths + 2h populations; validate vs `../ADC` h2o DIP & your FA_DIP | 6–10 wks — the real solver core |
| **M2** | **OpenBLAS backend + goroutine-parallel contractions**; scaling benchmark vs theADCcode on a mid-size DIP case | 2–4 wks |
| **M3** | **GPU backend** for the mat-vec (**cuBLAS** on NVIDIA *and* **hipBLAS** on AMD Instinct); large-DIP case theADCcode can't reach in reasonable time | 4–8 wks — the headline win |
| **M4** | Wire eigenvectors → 2h populations → existing `classify` → JSON → `plotspec` end-to-end | 2–4 wks |
| **M5** | **SIP / nD-ADC(3)** (IP branch) reusing the engine | 3–6 wks |
| **M6** | Decay widths: Stieltjes imaging, then CAP/Fano | open-ended — eventual physics differentiator |

Building integrals + SCF in Go instead of delegating would add ~3–5× to M0 for
near-zero differentiation — rejected.

**Status (2026-07-07):** M0, M1 done. The realized track then took **per-irrep
point-group symmetry blocking** as M2 (the union of per-irrep spectra reproduces
the full spectrum to <1e-8); the OpenBLAS/multicore work above was folded into
M3 as a build-tag over the Gonum backend rather than a separate milestone. **M3
is done**: the backend interface is now handle-based (device-resident
`Vector`/`DeviceMat`), the DIP mat-vec is an assembled block-sparse operator
uploaded once and applied as resident GEMVs, and the Lanczos vectors stay on the
device across iterations. Backends behind build tags (default pure-Go): multicore
**gonum+OpenBLAS** (`openblas`), **hipBLAS** (`hip`, built and validated on a
Radeon 890M / gfx1150), and **cuBLAS** (`cuda`, compiled against CUDA 13.x here,
run on NVIDIA hardware). Gates: hipBLAS ops match gonum to ~1e-11 and a full
device Lanczos solve reproduces the pure-Go dense spectrum to <1e-7; OpenBLAS
matches to ~1e-13. cuBLAS parity is validated on the NVIDIA host.

**M4 is done**: external cross-validation against theADCcode's h2o DIP reference
(`adcdip{1..4}.out`) on *matched* integrals. `scripts/gen_ref_fcidump.py`
reproduces the reference's DZP+diffuse basis + geometry + frozen-core "2 to 30"
active space (gated on SCF = −76.0498071428 Ha; the 29 active-MO symmetries match
the reference orbital-for-orbital). `internal/adc/refout` parses the reference and
`internal/adc/validate` compares a committed ADCgo fixture to it: **every strong
line matches in irrep and leading two-hole configuration, pole strengths within
~3%, the ¹A₁ ground state to 0.15 eV, O/H populations to table rounding**. ADCgo
(strict ADC(2)) sits 0.04–3.2 eV *above* the reference, whose run used a
higher-order ("Order: 4+") static self-energy — a documented method difference,
not an error; a strict-ADC(2) reference would need re-running theADCcode at order
2.

**M5 is done**: the **single-ionization** branch — a **non-Dyson IP-ADC(3)**
solver (1h main / 2h1p satellite), ported from `../ADC/ndadc3_ip` (Breidbach, JCP
109 (1998) 4734) into `internal/adc/sip`, reusing the M1–M3 engine unchanged
(backend / integrals / mp / lanczos, the assembled block-sparse operator: dense
1h/1h main + 1h↔2h1p coupling + symmetric 2h1p/2h1p satellite). `-order 2|3`
gates the 3rd-order main self-energy (`calc_c11_3`) and 2nd-order coupling
(`calc_c12_2`); order 2 is the reference's *extended* ADC(2) (1st-order satellite
kept). Spectroscopic factors use the ND-ADC F-matrix (`FMatrix`, F = 1 + F⁽²⁾ +
F⁽³⁾). Gates: `ApplyFull == BuildMatrix`, matrix symmetric, per-irrep union ==
symmetry-off spectrum <1e-8 (both orders). **Validated on matched integrals vs
pyscf `ip_adc`** (pyscf computes IP-ADC natively, unlike DIP): 2h1p satellite
roots and spectroscopic factors to ~1e-5 / ~5e-3; strong 1h main lines within a
one-sided ~0.03–0.16 eV self-energy-formulation band vs pyscf's ISR IP-ADC.
Committed fixture `testdata/h2o_sip.adcgo.json` pinned to the in-process solver to
1e-8. Definitive reproduction of theADCcode's own `ndadc3ip` main lines (matched
DZP integrals + a SIP `ADC.out` parser) is the remaining follow-on.

**M6 is done** (the polish round, JSON-sticks scope): the decay-channel
interpretation layer — ADCanalysis's `classify` + `spectrum` — is absorbed into
`internal/adc/spectrum`, consuming the in-memory `analyze` sectors directly (no
text round-trip). `classify.go` routes each dicationic state's atom-resolved
two-hole population onto Auger@A / ICD:A→B / ETMD(2) / ETMD(3) channels relative
to a chosen initial site (with site regrouping + passive-column discounting);
`build.go` flattens the solved DIP sectors into the stick-spectrum JSON
(`BuildDIP`) and the SIP sectors per orbital (`BuildSIP`). `cmd/adcgo -spectrum`
runs solve→classify→emit in one call; sites come from `-group NAME=col1,~col2`
(repeatable; a bare `-group` opens an interactive dialogue via the
`IsBoolFlag`/`promptGrouping` idiom) and `-init-atom`. The emitted JSON matches
ADCanalysis's schema field-for-field — **rendering stays in ADCanalysis's
`plotspec`** (verified: it renders the ADCgo fixture directly, 5 channels / 414
sticks), so no `gonum/plot` dependency and no EES/panel port here. Gates: the
ported `classify` unit tests; a committed stick-spectrum fixture
(`testdata/h2o_dzp.spec.json`) pinned to the in-process solve→classify to 1e-8; a
JSON key-contract test.

Remaining (deferred, user-run or later): cuBLAS full-scale run on the NVIDIA box;
optional M3b batched-GEMV perf on Instinct; the definitive theADCcode `ndadc3ip`
`ADC.out` SIP main-line reproduction (M5 follow-on); plotspec/EES rendering, the
EA branch, and Fano/CAP decay widths remain shelved.

---

## Verification (how each milestone is proven)

- **M0**: MP2 energy vs `pyscf.mp.MP2` on H₂O/dzp (same FCIDUMP) to ~1e-8 Ha.
- **M1**: DIP energies + pole strengths vs (a) `pyscf.adc` DIP-ADC(2) where
  available and (b) the `../ADC` H₂O DIP reference and your parsed `FA_DIP.json`.
  Reuse the **existing golden parser tests** (`internal/parse/dipfile_test.go`)
  as the oracle: parse the reference, assert ADCgo reproduces states (energies
  ~1e-3 eV — tables are rounded; pole strengths ~1e-2).
- **M2/M3**: correctness identical across backends (pure-Go == OpenBLAS ==
  cuBLAS == hipBLAS within tolerance) is the primary test; wall-clock + core/GPU
  scaling curves vs theADCcode are the differentiation evidence. For M3 both GPU
  vendors (NVIDIA cuBLAS, AMD Instinct hipBLAS) must reproduce the pure-Go
  reference spectrum.
- **M4**: recomputed two-hole populations must reproduce the parsed `PopRow`
  tables for H₂O (`decay_analyzer_sketch.md` §3 oracle).
- **End-to-end**: `pyscf → FCIDUMP → adcgo → JSON → plotspec` reproduces
  `FADIP.pdf` from a first-principles run rather than a parsed one.

Every step is runnable and self-checking against artifacts already in the repo.

---

## Risks / open questions

1. **Backend abstraction must come first.** If M1 hard-codes Gonum calls, the
   GPU port becomes a rewrite. Design `matvec` against the `backend` interface
   from the start, even while only the pure-Go impl exists.
2. **GPU mat-vec formulation** — decide sparse-SpMV vs batched-GEMM contraction.
   The contraction form (integrals × vector as batched GEMM) maps cleanly to
   both cuBLAS and hipBLAS and avoids materializing M; recommend that. Host↔device
   transfer per Lanczos iteration is the likely bottleneck — keep vectors resident
   on device. Target NVIDIA (cuBLAS) and AMD Instinct (hipBLAS/ROCm) from the same
   GEMV/AXPY contraction surface; the two differ only in the cgo shim.
3. **3h1p dimension / memory** — the DIP satellite space is what makes this hard;
   validate the mat-vec on tiny H₂O first, then watch memory as the case grows.
4. **Lanczos spurious/ghost roots** — band-Lanczos produces spurious eigenvalues;
   need the standard purge/convergence tests. `../ADC` already handles this
   (spurious roots dropped) — a porting reference, and the parser already copes
   with index gaps.
5. **cgo + reproducibility** — threaded BLAS and GPU can perturb reproducibility
   (summation order); keep the deterministic pure-Go backend as the reference in
   CI and assert cross-backend agreement to a set tolerance.
6. **Sign/normalization & spin conventions** — singlet/triplet phases (Eq. 2a)
   and pole-strength normalization must match the reference tables; golden tests
   catch drift.

---

## Immediate first step (M0 spike)

`pyscf.tools.fcidump.from_scf(mf, 'h2o.fcidump')` on the existing H₂O/dzp
example, then implement `internal/adc/fcidump` + `internal/adc/mp` and a
throwaway `cmd/adcgo` that prints the MP2 energy. If that matches pyscf, the
integral-ingestion contract is proven and the DIP-ADC(2) core (M1) can start —
written against the `backend` interface from line one.
