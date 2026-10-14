# scripts

| Folder | Contents |
|---|---|
| `fcidump/` | Integral generation: `dump_fcidump.py` (sectioned input, see `adcgo_input.py`), the GAMESS-UK readers, orbital selection, the localized-orbital basis (`orbitals.py`) and KBJ ghost bases (`kbj.py`), `threads.py` (import before numpy on the login node) |
| `fixtures/` | Regenerate the committed test fixtures in `testdata/` |
| `codegen/` | Code generators: adcgen ISR elements (`generate_adc.py`, `isrgen.sbatch`, see `internal/adc/isrgen/README.md`), the ADC(4) spin tables (`gen_coeff4.py`) |
| `helix/` | bwForCluster Helix: `HELIX.md`, the dump-and-run driver (`adcgo_run.sh`, `runADCgo*`), the CUDA build, workspaces, production and benchmark jobs |
| `gpu/` | GPU diagnostics and parity jobs (CUDA smoke, compute-sanitizer, multi-GPU parity, int32 addressing, timing A/B) |
| `analysis/` | Spectrum comparison and plotting |

Scripts find the repository root from their own location, so run them from any directory:

```sh
python scripts/fcidump/dump_fcidump.py --input examples/DIP_h2o/dump.in
sbatch scripts/helix/runADCgo examples/DIP_h2o/dump.in
```

## Auger-Meitner difference spectrum (dimer minus 2 x monomer)

Isolates what a dimer does that two isolated monomers cannot. Needs SIP **and** DIP, whole band, for
both systems; the monomer must be a **counterpoise** fragment (partner ghosted) or the difference
carries basis-set superposition error instead of physics.

```sh
# 1. integrals: dimer (exists) + counterpoise monomer
SYS=naph2 DECK=naph2_cp_mono.in sbatch scripts/dump_melanin.sbatch

# 2. whole-band solves. PREFIX picks the file set, SYS the workspace, so the fragment
#    lives beside its dimer. EIG_B2/EIG_DEVICE are the eigensolve accelerations.
SYS=naph2 PREFIX=naph2_cp_mono EIG_B2=128 EIG_DEVICE=1 sbatch scripts/melanin_dip_mgpu.sbatch
SYS=naph2 PREFIX=naph2_cp_mono sbatch scripts/melanin_sip.sbatch

# 3. solver documents -> stick spectra
adcgo -convert naph2.dip.json -dip -bare -out naph2.dip.spec.json      # and .sip

# 4. the difference. -xrange is REQUIRED (common electron grid) and 2 is exact only for a
#    homodimer whose halves are congruent.
plotspec -mode ees-diff \
  -sip naph2.sip.spec.json         -dip naph2.dip.spec.json \
  -ref-sip naph2_cp_mono.sip.spec.json -ref-dip naph2_cp_mono.dip.spec.json \
  -ref-scale 2 -xrange 0-16 -points 600 -fwhm 1.0 \
  -out naph2_am_diff.png -data naph2_am_diff.tsv
```

- `-xrange` upper bound = `max(SIP energy) - min(DIP energy)`; above it the electron spectrum is zero.
  naph2: 33.17 - 17.09 = **16.1 eV**.
- Intensities are always absolute in `ees-diff`; peak normalisation would make the factor 2 meaningless.
- Channels are summed before subtracting: ghosting breaks the dimer's C2, so the fragment has fewer
  irreps and the decompositions do not correspond.
### Which reference, and why the residual means different things

A dimer's non-interacting double-ionization manifold has **three** families: both holes on A, both on
B, and **one hole on each**. A monomer DIP contains only the one-site kind, so `2 x monomer`
reproduces two families and omits the third entirely.

| reference | flags | residual contains |
|---|---|---|
| one-site only | (default) | the missing two-site family **+** interaction effects, conflated |
| one- and two-site | `-ref-twosite` | interaction effects alone (shifts, splittings, intensity transfer) |

`-ref-twosite` completes the reference by self-convolving `-ref-sip`: each ordered pair of monomer
ionizations (i on A, j on B) becomes a final state at `E_i + E_j + shift` with the independent-hole
weight `s_i * s_j`. Run both and compare — the first says how much of the spectrum the monomers cannot
account for at all, the second how much of *that* is interaction rather than bookkeeping.

```sh
plotspec -mode ees-diff ... -ref-twosite -twosite-shift 4.2
```

- `-twosite-shift` is the hole-hole Coulomb repulsion `e^2/R`, which the strict R->inf reference omits
  but a dimer at contact does not: `14.4/R(Angstrom)` eV, i.e. **4.20 eV** at naph2's 3.425 A
  interplanar separation. Default 0 = the strict non-interacting limit.
- `-twosite-weight` defaults to 0.5, the non-interacting state count, chosen so `-ref-scale 2` stays
  correct: an overall factor on the final-state spectrum cancels between sigma's numerator and its
  `N(E)` denominator, so only the one-site:two-site ratio and the initial-state scale matter.
- It is an **independent-hole construction**, not an ab-initio two-site DIP, and it applies no spin
  multiplicity (each spatial two-site configuration supports a singlet and a triplet). Use
  `-twosite-weight` if a different convention is wanted.
- Cost: `N^2` synthetic states. 573 SIP lines -> 328k states, convolved in ~8 s.
