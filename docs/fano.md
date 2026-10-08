# Fano decay widths — `-fano`

Γ and τ = ℏ/Γ for a chosen vacancy: Fano/Feshbach method with Stieltjes imaging
([Kolorenč & Averbukh, *J. Chem. Phys.* **152**, 214107 (2020)](https://doi.org/10.1063/5.0007912)).
`adcgo -h fano` lists every flag. ADC(2,2) validation: [`adc22_verification.md`](adc22_verification.md).

## Matrices

| Family | Flags | Matrix | Use |
|---|---|---|---|
| single ionization | `-sip -order 2\|22 -fano` | Fano-ADC(2)x, or ADC(2,2) (`-adc22 m\|x\|f`, `f` recommended) | Auger/ICD widths; ADC(2,2)'s 3h2p class adds second-order decay (double Auger, double ICD) |
| double ionization | `-dip -fano -spin singlet\|triplet -sym none\|IRREP` | DIP-ADC(2), one sector | dicationic widths; no partial widths (the 3h1p decay class has no channel routing) |
| k-hole CI | `-khci K -fano -sym none\|IRREP` | H − E_HF over kh \| (k+1)h1p \| (k+2)h2p, any orbitals | multiply ionized initial states (K = 1..4) |

```sh
# Ne+ (1s^-1) Auger width. The vacancy fixes the target irrep, so -sym is determined.
adcgo -fcidump ne.fcidump -sip -order 22 -adc22 f -fano -fano-init 0 -sym all -solver lanczos -matfree on

# interatomic decay: Q is every configuration with ALL holes on the donor subunit
adcgo -fcidump dimer.fcidump -sip -order 22 -fano -fano-init 2 -fano-q 2,3,4 -fano-rule all \
    -mo dimer.mo.json -init-atom A

# a four-hole initial state (two doubly emptied orbitals), labelled localized dump
adcgo -fcidump sys.fcidump -mo sys.mo.json -khci 4 -khci-maxfree 1 -sym none -fano \
    -fano-init 0 -fano-phi-holes 0,0,4,4 -fano-engine gauss -fano-decompose \
    -fano-qp 'p: charge *=1 & free=1'
```

- `-mo` adds approximate partial widths per channel (Auger@A, ICD:A→B, ETMD, `double` for the
  3h2p/second-order channel), each imaged separately.

## Basis requirement

Decides whether a run is possible at all.

- Imaging evaluates Γ(E_Φ) only if the 2h1p pseudo-continuum brackets E_Φ; a 2h1p state sits at
  ε_a − ε_k − ε_l.
- What matters is the energy span and level density of the virtual space near E_Φ, not
  diffuseness.
- Ne 1s (E_Φ ≈ 32 E_h): aug-cc-pVTZ tops out at 14.6 E_h and cannot describe the decay;
  aug-cc-pVQZ reaches 68.9 E_h.

| Ne⁺(1s⁻¹), Fano-ADC(2)x | coupled channels | Γ (meV) |
|---|---|---|
| aug-cc-pVQZ | 77 | 642 ± 269 |
| aug-cc-pV5Z | 96 | 316 ± 8 |
| aug-cc-pV5Z + 4s4p4d | 165 | 227 ± 10 |
| published (Table VI) | | 244 ± 4 |

## Is the width trustworthy?

The run log prints both diagnostics:

- **coupled channels** — P configurations that carry any coupling. Imaging reconstructs the
  density from these alone; a few tens gives a large Stieltjes spread.
- **sum-rule residual** — pseudo-continuum total strength vs. the exact 2π‖g‖². Zero for a
  complete basis; large means the Krylov space was truncated. An unconverged width can still
  show a *small* error bar (the Stieltjes orders agree about the wrong density) — raise
  `-fano-blocks` until the residual closes.

## Intermolecular decay in clusters (ICD)

- **Name orbitals, not indices.** Every `-fano` orbital list takes `@SITE` (occupied orbitals
  whose largest population is on a `-group` site), `@SITE.k` (the k-th lowest of them) and
  `e<X` / `e>X` (energy window, hartree). The log prints what each resolved to; the document
  records `vacancy_site_population` (below ~0.8 the "site" vacancy is delocalized).
- **Exclude one-site configurations from both Q and P** (`x:` clauses, below). P must hold
  only holes on two or more molecules:
  ```
  -fano-init @W1.0 -fano-qp 'q:1/e<0:1;q:e<-1.0:1;x:2/@W1:2&2/e<-1.0:0:0;x:2/@W2:2&2/e<-1.0:0:0'
  ```
  The `&2/e<-1.0:0:0` term matters: `x:` is tested first and `@W1` includes W1's own 2a1,
  so without it the 2a1⁻¹ outer⁻¹ configurations of the same water (part of the decaying
  state) would be excluded too. For ADC(2,2) add the same pair with class 3.
  The two complete partitions both fail on the water dimer (cc-pVDZ + KBJ, Fano-CI and
  ADC(2)x alike, Γ ≈ 0.3–2 eV against ~9 meV):
  - **One-site 2h1p in P:** H2O⁺ shake-up satellites act as a fake continuum. An isolated
    water's 2a1 gets an 0.8 eV "Auger" width.
  - **One-site 2h1p in Q:** they dress the 2a1 state until it is no longer one. The
    selected root holds 1–35% 2a1 weight, and Γ depends on which fragment is picked
    (`c4_audit` lists them).
- **Orbital assignment is by largest population.** In clusters, canonical outer-valence
  orbitals can be shared between waters, so check `vacancy_site_population` and the
  orbital log lines.
- `-group` is a bool-style flag: `-group NAME=cols` and `-group=NAME=cols` both work from the
  2026-10-08 build; older builds silently dropped every flag after the spaced form.
- Point-charge embedding: `&charges` in the dump deck (`adcgo_input.py`).
- Verification against the literature: `scripts/helix/fano_icd_dimers.sbatch` — water dimer
  (Richter et al. 2018, Fano-CI 72 / 131 fs) and Ne₂ Γ(R) (Kolorenč & Averbukh 2020 Fig. 1).

## Excluding configurations (`x:` clauses)

- `-fano-qp` clauses are `q:` (bound), `p:` (continuum) and `x:` (excluded). `x:` is tested
  first, and a matched configuration is in neither subspace.
- Without `x:`, P = everything not in Q (Q + P = 1). With it, Q + P + X = 1 and the width is
  that of the Hamiltonian restricted to Q ⊕ P. QP = 0 still holds, so the coupling is
  unchanged (one parent mat-vec gathered on P; gated against the explicit cross block).
- **Standard practice, and a model choice.** Fano-CI and the reference Fano-ADC code select
  the initial and final spaces separately, dropping what belongs to neither.
- **Use it only for configurations that are neither part of the decaying state nor an open
  channel.** Excluded configurations no longer relax |Φ⟩ and receive no flux. An excluded
  open channel lowers Γ; excluded decaying-state strength shifts E_Φ.
- **Check it:** compare against the same class in P and in Q. The log prints
  `excluded=N` per class and the document carries `x_size`.
- Works in both grammars: hole terms (`x:2/2,4,6:2`) and net-charge terms
  (`x: 6/charge A1=2`).

## Localized orbitals and tiny widths (`-khci`)

- With a labelled sidecar (`dump_fcidump`'s `&orbitals localized` scheme), `-fano-qp` takes a
  net-charge rule, e.g. `'p: charge *=1 & free=1'` (every atom singly charged, one free electron).
- `-fano-phi-holes 0,0,4,4` picks the decaying state as the interior QMQ root heaviest on that
  configuration, polished to `-fano-phi-tol` (default 1e-10 Eh).
- Widths far below the coupling scale (10⁻¹⁵ Eh): use an engine with no PMP eigenvectors and no
  final-state cut — `-fano-engine gauss` (Gauss rules from a Lanczos run seeded by the coupling
  vector), `shiftinvert` or `kpm`.
- Noise floor is measured, not assumed: `-fano-lambda 'C:D=0'` switches one electron transfer
  off exactly; `-fano-save-g` / `-fano-image-g` image the double difference of such runs.

## Reproducing Table VI

- Fixtures: `scripts/fixtures/gen_fano_atoms.py`.
- Run: `scripts/helix/fano_table6.sbatch` on a compute node. The login node caps the user slice
  at 4 cores, so `GOMAXPROCS` is 4 there whatever `nproc` says.
