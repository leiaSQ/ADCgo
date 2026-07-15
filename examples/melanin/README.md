# melanin — DIP-ADC(2) and SIP-ADC(3), DZ basis

Double- and single-ionization spectra of a melanin oligomer building block (34 atoms:
4 O, 2 N, 18 C, 10 H), reproducing the theADCcode drivers in `/home/tcstud21/melanin`.

## Files

| File | Source |
|------|--------|
| `zmat` | copy of `melanin/zmat` — GAMESS-UK z-matrix, Angstrom |
| `_basisset.dat` | copy of `melanin/_basisset.dat` — DZ (Dunning-Hay), GAMESS-UK format |
| `melanin_dip.in` | DIP-ADC(2), C1, singlet+triplet ← `melanin/dip_with_popana.sh` |
| `melanin_sip.in` | SIP-ADC(3), C1, Σ(∞) ← `melanin/ndad3ip_guk.sh` |

Reference RHF total energy (GAMESS-UK, DZ): **−1095.4068317755 Ha** — the SCF `gate`.
260 basis functions, 164 electrons, 82 occupied MOs. Correlated window `active 25 to 236`
freezes the lowest 24 MOs and drops virtuals above 236 → 212 active orbitals.

## theADCcode → ADCgo mapping

| theADCcode (here-doc) | ADCgo |
|-----------------------|-------|
| `adc2dip` | `-dip -order 2` |
| `ndadc3ip` | `-sip -order 3` (non-Dyson IP-ADC(3)) |
| `spin 1 3` (DIP) | `-spin both` (singlet+triplet) |
| `spin 2` (SIP doublet) | implicit for `-sip` |
| `self-energy fplus` / `infinite` | DIP: built in; SIP: `-sigma infinite` |
| `lanczos / iter 200` | `-solver lanczos -blocks 200` |
| `active 25 to 236` | `&active active 25 to 236` |
| `SYMGRP CS` / `C1` | `symmetry auto` (→ **C1** for this geometry) |

**Symmetry — C1, not Cs.** The DIP driver declared `SYMGRP CS`, but pyscf detects **C1**
for this z-matrix even at a loose tolerance, and forcing `symmetry Cs` fails. This molecule
is genuinely three-dimensional (folded), not a planar ring with a mirror: projecting every
atom onto its best-fit plane — the only route to Cs for an asymmetric fragment — collapses
two atoms to 0.001 Å apart (from 1.008 Å) and drops the RHF energy from −1095.4 Ha to a
nonsensical −332 Ha. So Cs is not physically attainable here; both runs use C1.

## Running

DIP and SIP share an **identical orbital space**, hence the same `melanin.fcidump`. The
`&output` section names it in both `.in` files, so the second run reuses the first dump.

Local / single node:
```
scripts/adcgo_run.sh examples/melanin/melanin_dip.in ./adcgo-cuda
scripts/adcgo_run.sh examples/melanin/melanin_sip.in ./adcgo-cuda
```

Helix, both calculations on one exclusive 4-GPU node (DIP → GPUs 0,1; SIP → GPUs 2,3):
```
sbatch scripts/runADCgo_helix_melanin
```

## GPU mapping

With `-backend cuda` and ≥2 visible GPUs, independent sectors run concurrently, one per
GPU (per-sector multi-device support in `internal/adc/backend`):

- DIP `-spin both` → 2 sectors (singlet, triplet) → **2 GPUs**.
- SIP (C1, doublet) → 1 sector → **1 GPU**.

Because the geometry is C1 (see above), SIP has a single symmetry sector, so per-sector
parallelism cannot spread it across GPUs — that would need intra-sector splitting, which is
out of scope for now. The exclusive-node script (`scripts/runADCgo_helix_melanin`) runs DIP
on GPUs 0,1 and SIP on GPU 2 (GPU 3 idle). In true Cs the SIP a′/a″ split would fill it.
