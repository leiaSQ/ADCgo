# Plan: **ADCgo** — an exact, hardware-accelerated ADC(n) ionization solver

## Context

ADCgo is a fast, exact, GPU-accelerated ionization-ADC solver written in Go. It
**delegates SCF + integrals** (FCIDUMP + sidecars, produced by pyscf) and implements
*only* ADC, reaching larger systems by **accelerating** the exact secular problem on
modern hardware rather than **truncating** it (no RI, no reduced-scaling). Where the
Dreuw world (adcc) omits the entire IP/DIP/decay branch and the Cederbaum legacy code
(`../ADC`, "theADCcode") exists only as hard-to-build, effectively single-threaded
C++/F77, ADCgo occupies the open niche: a **fast, exact, GPU-capable ionization solver**
for DIP/SIP/ICD physics.

The foundation (M0–M6, all done) delivers DIP-ADC(2), non-Dyson SIP-ADC(3), device-
resident block-Lanczos across four backends, and a decay-channel classifier that emits
stick spectra. **This document now describes the next arc**: turning ADCgo from a tool
that says *where* decay goes into one that says *how fast* (Track W — ICD/ETMD decay
widths and lifetimes) and *how accurately* (Track A — a modern ADC(4) exchange
implementation). The two tracks are **parallel and independent**, deliberately kept
alongside each other for generality.

---

## Completed foundation (M0–M6)

| # | Deliverable |
|---|---|
| **M0** | FCIDUMP reader + MO-integral store + **MP2 energy** matching pyscf |
| **M1** | **DIP-ADC(2)** mat-vec + block-Lanczos → dication states + pole strengths + 2h populations |
| **M2** | Per-irrep point-group symmetry blocking (union of per-irrep spectra == full spectrum) |
| **M3** | Device-resident backends: **gonum / OpenBLAS / hipBLAS / cuBLAS** behind build tags |
| **M4** | External DIP validation vs theADCcode on **matched integrals** |
| **M5** | **SIP / non-Dyson IP-ADC(3)** (1h/2h1p), validated vs pyscf `ip_adc` |
| **M6** | Decay-channel spectrum: classify (Auger/ICD/ETMD) + stick-spectrum JSON |

**Status (2026-07-08):** M0–M6 done.

**M3** is done: the backend interface is handle-based (device-resident `Vector`/`DeviceMat`);
the DIP/SIP mat-vec is an assembled block-sparse operator uploaded once and applied as
resident GEMVs/batched GEMMs; Lanczos vectors stay on device across iterations. Gates:
hipBLAS matches gonum to ~1e-11 and a full device Lanczos solve reproduces the pure-Go dense
spectrum to <1e-7; OpenBLAS to ~1e-13; cuBLAS parity validated on the NVIDIA host.

**M4** is done: cross-validation vs theADCcode's h2o DIP reference (`adcdip{1..4}.out`) on
*matched* integrals (`scripts/gen_ref_fcidump.py`, gated on SCF = −76.0498071428 Ha; 29
active-MO symmetries match the reference orbital-for-orbital). `internal/adc/refout` parses
the reference; `internal/adc/validate` compares. On matched integrals ADCgo is bit-exact vs
theADCcode (the earlier 0.04–3.2 eV gap was a `backend.AddSubDiagConst` bug, fixed
2026-07-07, not the reference's inert "Order: 4+" self-energy).

**M5** is done: non-Dyson IP-ADC(3) (`internal/adc/sip`, ported from `../ADC/ndadc3_ip`),
reusing the M1–M3 engine. `-order 2|3` gates the 3rd-order main self-energy and 2nd-order
coupling; spectroscopic factors via the ND-ADC F-matrix. Validated on matched integrals vs
pyscf `ip_adc` (satellites/SF ~1e-5/5e-4; mains within ~0.03–0.16 eV nD-gap). Committed
fixture `testdata/h2o_sip.adcgo.json` pinned to 1e-8.

**M6** is done: the decay-channel interpretation layer absorbed into `internal/adc/spectrum`,
consuming the in-memory `analyze` sectors directly (no text round-trip). `classify.go` routes
each dicationic state's atom-resolved two-hole population onto Auger@A / ICD:A→B / ETMD(2) /
ETMD(3); `build.go` flattens sectors into stick-spectrum JSON. `cmd/adcgo -spectrum` runs
solve→classify→emit; the emitted JSON matches ADCanalysis's `plotspec` schema field-for-field
(rendering stays in ADCanalysis).

**Solver-parameter note (2026-07-08):** `-blocks N` == theADCcode's `iter N`; the Krylov
subspace is `N × MainBlockSize()` (start block included). Pinned by
`internal/adc/lanczos/blockcount_test.go`; default is 100 (the reference DIP runs' `iter`).

---

## The next arc: two parallel tracks

The gap M0–M6 leaves open for the ICD/ETMD field: ADCgo computes **energies, pole
strengths, and channel classification** but no **rates**. Every serious ICD/ETMD study
reports lifetimes τ = ℏ/Γ and branching ratios (partial Γ per channel). No open,
GPU-accelerated, exact ionization-ADC code produces them.

### Reference map (where the physics lives in `../ADC`)

- **Fano/Stieltjes (Track W primary):** `adc2_pol/master_fano_new.f90` (golden rule
  `Γ_f = 2π|⟨Φ_i|Ĥ|χ_f⟩|²`); `adc2_pol/partgammas.f90` (`getf_HIJ_adc2/adc2e` — coupling
  vector = (Ĥ−E_i)Φ_i over final configs); `adc2_pol/select_fano.f90` (subspace selection);
  `adc2_pol/stieltjes_phi1.f` (Averbukh quad-precision Stieltjes imaging) + `tql2.f`.
  *Improve on:* the `stieltjes_phi` calls in `fspace.f90` are commented out — the reference
  stops at discrete (E,γ) pairs; our pipeline runs live and automated.
- **CAP-ADC (Track W validator):** `subspaceCAP/cap_main.f90` `docap` — builds
  `H = diag(E_ADC) − iη·W` in the *selected ADC-eigenvector window* and calls `ZGEEV` per η;
  `CAP.f90` (`CAP_SINGLY`, IP) / `CAP_doubly.f90` (DIP + triplet) project the MO-CAP;
  box-CAP AO integrals in `scf_data/r2capmat.f90` + `cap_Sajeev.f`. The complex solve is over
  a small selected window — **no complex Lanczos needed**. *Improve on:* the reference reads
  the resonance off the trajectory by eye — we automate it.
- **ADC(4) exchange (Track A):** legacy **F77** `adc4core/adc4_constr/*.F` — `calcsi.F` (Σ
  with Coulomb + exchange Fock parameters), `sec198_mp.F` (2nd + O(3) + O(4) Σ), `kopp1..4.F`
  and `k1p3h/k2p2h/k3p1h.F` (4th-order coupling/satellite blocks). `adc4cvs_matrix.cpp` is
  **only a C++ driver** that calls that F77 (`ndriver_core_`, `adc_`, `sec198_mp_`) — it holds
  none of the exchange physics; CVS-ADC(4) = the same diagrams + a core-valence-separation
  restriction. Literature: von Niessen/Schirmer/Cederbaum, Comp. Phys. Rep. 1 (1984) 57.

### Reusable ADCgo infrastructure

- **Channel taxonomy & site routing** — `internal/adc/spectrum/classify.go`
  (`Site`/`Regroup`/`Classify`/`Discount`): Track W attaches Γ per channel using this unchanged.
- **Localized-population engine** — `internal/adc/analyze/populations.go` (Tarantelli U/O).
- **Block-Krylov + GPU matvec** — `internal/adc/lanczos`, `internal/adc/{dip,sip}/matvec.go`.
- **MO/overlap sidecar** — `internal/adc/mo/mo.go`, `scripts/gen_fcidump.py` (extended in W5).
- **Config enumeration** — `internal/adc/dip/config.go`, `internal/adc/sip` (P/Q partition, CVS).

### Net-new to build

Complex128 dense algebra + a small non-Hermitian eigensolver (ZGEEV-equivalent, selected
window only); a Stieltjes-imaging engine; a Feshbach P/Q partition + decaying-state selector;
box-CAP MO integrals in the sidecar (delegated from pyscf); width observables in the JSON
schema; the 4th-order exchange blocks for SIP and DIP (+ a CVS filter).

---

## Track W — Decay widths (Fano-Stieltjes primary, CAP validator; SIP + DIP)

**W1 — Feshbach–Fano partition & decaying-state selection (generic over SIP & DIP).**
P (bound/decaying) / Q (continuum) partition over a config space, generic across `dip.Space`
and the SIP space. Select |Φ_d⟩ — the inner-valence 1h (SIP) or 2h (DIP) main configuration —
via a bound-subspace-projected diagonalization (reuse `MainBlockSize` + Lanczos on the
P-block). New `internal/adc/fano`. Mirror `adc2_pol/select_fano.f90`. *Verify:* bound-projected
energy reproduces the inner-valence main line; |Φ_d⟩ has the expected hole character; dense
oracle; both spaces.

**W2 — Per-channel continuum-coupling spectrum (on the GPU engine).**
Compute {E_j, γ_j = 2π|⟨Φ_d|Ĥ−E_d|Ψ_j⟩|²} over the Q-space pseudo-continuum eigenstates
(2h1p for SIP-ICD; 3h1p for DIP) — the projection of (Ĥ−E_d)Φ_d onto the Q-eigenbasis,
reusing the existing matvec + block-Lanczos band sweep. Split the coupling per decay channel
with the existing `spectrum` Site/Regroup/Classify routing. Mirror `adc2_pol/partgammas.f90`.
*Verify:* sum rule Σγ_j = 2π‖Q(Ĥ−E_d)Φ_d‖²; dense vs Lanczos; per-irrep union; both spaces.

**W3 — Stieltjes imaging engine.**
Port `adc2_pol/stieltjes_phi1.f`: spectral moments S₋ₖ = Σ γ_j E_j⁻ᵏ → orthogonal-polynomial
3-term recurrence → tridiagonal → eigen (Gauss nodes/weights) → cumulative γ(E) → numerical
derivative → Γ at E_d, with order-convergence. `math/big.Float` for the moment recurrence,
float64 for the small tridiagonal eig. New `internal/adc/stieltjes`, wired live + automated.
*Verify:* analytic Lorentzian-coupling model with known Γ; order-convergence curve; a published
ICD Γ (Ne dimer literature).

**W4 — Widths / lifetimes / branching ratios → JSON (tied to classify).**
Per-channel Stieltjes → partial Γ_channel; Γ_total = Σ; τ = ℏ/Γ (0.6582 fs·eV / Γ[eV]);
branching = Γ_channel/Γ_total. Extend `analyze.State` and `spectrum.Line`/`Meta` with
`width_ev`, `lifetime_fs`, and per-channel partial widths. New `cmd/adcgo -widths` flow.
*Verify:* Σ partial = total; channel labels match classify; committed fixture + regeneration
guard + schema test.

**W5 — Complex backend + subspace CAP-ADC + automated η-trajectory (independent validator).**
Minimal complex128 in `backend`: a complex dense Mat + a non-Hermitian eigensolver over the
*selected* window only (ZGEEV via netlib/OpenBLAS cgo, or a Go complex-QR — no complex
Lanczos). Extend the MO sidecar + `gen_fcidump.py` with a delegated box-CAP MO matrix W_pq
(pyscf grid box-CAP, occupied MOs zeroed, per `cap_Sajeev.f`/`r2capmat.f90`). Project W into
the selected window (port `CAP_SINGLY`/`CAP_doubly`). Sweep H = diag(E) − iηW → ZGEEV →
trajectory; extract the resonance automatically via min|η·dE/dη| (Riß–Meyer optimal η).
Mirror `subspaceCAP/cap_main.f90`. *Verify:* reproduce `subspaceCAP` on a matched case; dense
complex oracle; **Γ_CAP agrees with Γ_Fano-Stieltjes within method spread**; cross-backend parity.

---

## Track A — Modern ADC(4) exchange terms (SIP + DIP, CVS as a restriction)

**A1 — ADC(4) exchange-diagram spec (derivation / de-risk; no code).**
Extract the exact 4th-order exchange diagrams from the F77 (`calcsi.F`, `sec198_mp.F`,
`kopp*.F`, `k*p*h.F`) and the literature (von Niessen/Schirmer/Cederbaum 1984;
Schirmer/Trofimov), in the ADCgo block-matrix idiom (like `dip/singlet.go` / `sip/elements.go`)
with coefficient tables. A markdown spec in the repo. *Verify:* peer vs literature; a
hand-checked term vs the F77 numeric output.

**A2 — SIP-ADC(4): 4th-order self-energy + coupling + satellite exchange blocks.**
In `internal/adc/sip`, gated by `-order 4`: the O(4) static self-energy (extends `calc_c11_*`),
the 4th-order 1h↔2h1p coupling, and the 2h1p/2h1p exchange terms (`kopp*`/`k2p2h`), as new
order-gated blocks in the assembled operator (GPU matvec unchanged). *Verify:* `ApplyFull ==
BuildMatrix`; symmetric; per-irrep union; **matched-integral vs theADCcode ADC(4)**.

**A3 — DIP-ADC(4) exchange terms.**
The same 4th-order exchange blocks for the DIP 2h/3h1p problem in `internal/adc/dip`, gated by
`-order 4`. *Verify:* matched integrals vs reference; symmetry; per-irrep union; dense oracle.

**A4 — CVS-ADC(4) restriction (core-level; feeds Track W).**
A core-valence-separation projector as a config-space filter on the ADC(4) spaces (drop configs
lacking a core hole), reproducing `adc4cvs_matrix`'s CVS behaviour. Yields core-ionized ADC(4)
states — the Auger/ICD *initial* states — so W1 can select its decaying state from a CVS-ADC(4)
solve. *Verify:* vs `adc4cvs_matrix` on a matched core case; the CVS spectrum is a clean
sub-block of the full ADC(4) spectrum.

---

## Verification (how each milestone is proven)

The four testing pillars proven in M0–M6 are load-bearing and every new milestone keeps them:

1. **Matched integrals** — run ADCgo on theADCcode's *own* exported integrals
   (`testdata/reference/*.matched.fcidump` via `../ADC/fcidump_export`) so any residual is
   pure ADC-method difference. For Track A, export ADC(4) reference matrices via
   `../ADC/matrix_dump`; for Track W, a matched CAP/Fano reference Γ.
2. **Dense/exact oracle** — an in-process exact path the iterative/approximate path is checked
   against (`SolveDense`; a dense complex eigensolver for CAP; the analytic Lorentzian for
   Stieltjes). The dense `SymEig` path is the DIP/SIP correctness oracle.
3. **Committed fixture + regeneration guard** — commit a JSON fixture *and* a test that
   re-solves in-process and asserts reproduction to ~1e-8, so contracts can't drift (pattern of
   `validate_spec_test.go`, `validate_sip_test.go`, `TestFixtureMatchesSolver`).
4. **Cross-backend parity** — every accelerated/complex backend reproduces pure-Go Gonum to
   tight tolerance (`gpu_parity_test.go`, `backend/symeig_test.go`;
   `TestSymEigMatchesGonum` handles degenerate-eigenvector sign/basis ambiguity).

Plus per-irrep/per-spin union tests, the `refout` parser as an independent oracle, and (from
M0) MP2 vs `pyscf.mp.MP2` to ~1e-8 Ha as the integral-ingestion contract.

**Physics gates / differentiation demo for the new arc:**
- Ne₂ **ICD lifetime** reproduced vs the literature (both Fano-Stieltjes and CAP), with
  branching ratios.
- A larger rare-gas / hydrogen-bonded cluster the legacy code can't reach in reasonable time —
  full ICD/ETMD widths + branching — timed GPU vs CPU.
- ADC(4) matched-integral eigenvalues vs theADCcode ADC(4) (SIP, DIP, CVS) to print precision.
- End-to-end: `pyscf → FCIDUMP + sidecars → adcgo (-order 4 / -widths) → JSON` yields
  lifetimes and branching ratios from a first-principles run.

Building integrals + SCF in Go instead of delegating would add ~3–5× for near-zero
differentiation — rejected. FCIDUMP + sidecars remain the single ingestion path.

---

## Risks / open questions

1. **Complex-algebra scope.** Keep the complex path confined to the small *selected*
   ADC-eigenvector window (CAP) — no complex Lanczos. If a full complex diagonalization is ever
   needed it is a separate project.
2. **Stieltjes conditioning.** The moment recurrence is ill-conditioned (the reference uses
   real*16); use `math/big.Float` for it and validate order-convergence against the analytic
   model before trusting any Γ.
3. **CAP delegated integrals.** The box-CAP MO matrix must come from pyscf via the sidecar
   (geometry + AO basis are not in FCIDUMP); occupied MOs zeroed, per the reference.
4. **ADC(4) transcription risk.** The F77 exchange diagrams are opaque; A1 (a written spec,
   no code) de-risks the port, and matched-integral bit-exactness vs theADCcode is the gate.
5. **cgo + reproducibility.** Keep the deterministic pure-Go backend as the CI reference and
   assert cross-backend agreement to a set tolerance (including the new complex ops).
6. **Sign/normalization & spin conventions.** Golden/matched tests catch drift, as in M4/M5.
