# CVS IP-ADC(4) reference tape — A1 sector of H2O/DZP

Companion to `../adc4_b2_tape`. Same matched DZP integrals
(`../h2o_dzp.matched.fcidump`), but the **A1** sector (`sym 1`), which contains the
core hole — so this tape has the **1h main block** (dim 1712 = 1 [1h] + 46 [2h1p] +
1665 [3h2p]). It therefore validates the 1h couplings that B2 could not: KOPP1/KOPP2/KOPP3
(1h↔2h1p) and KOPP4 (1h↔3h2p).

Files:

- **`FT21F001.ADC`** — off-diagonals; **`FT18F001.ADC`** — header + 1h/2h1p diagonal
  (here `nh12=47`; diagonal[0]=1.37063 is the core-hole 1h diagonal = −ε_core − Σ).
  Formats as in the B2 README. Row/col index 1 = the 1h main.
- **`FT19F001.ADC`** — the **3h2p effective diagonal** (EIGAB = 0th-order orbital-energy
  sum + the 5th-order WERT3 3h2p-CI correction). Record 1 = `IDIM NCOL NECORE N3H2P`
  (4× int32), record 2 = `N3H2P` × float64. Columns carry the same `ab5.F` pam/ELIM
  permutation as FT21's 3h2p columns, so compare it as a multiset.
  Gated by `TestADC4EigabGate`.
- **`SIGMA_STATIC.dat`** — the **static self-energy** actually applied to the 1h block, in
  a.u.: `NECORE`, then `i j ni nj value` per lower-triangle element. Gated by
  `TestADC4StaticSigmaGate`. Note this is *not* the eV value the C++ caller hands `adc_()`:
  `egf.F` converts it with FAKTOR (1 a.u. = 27.211606 eV, **not** the 27.211396 used on the
  way in) and adds SIGMPH, so reconstructing it from the C++ input leaves an ~9e-8 residual.

Neither FT19 nor SIGMA_STATIC.dat existed before: `RSCRT1` rewrites the diagonal tape with
only the first `nh12` entries (discarding EIGAB), and Σ is an *input* to `adc_()` that no
tape records. `../ADC` now dumps both (`ab5.F`, `egf.F`).

## Regeneration

theADCcode rebuilds correctly from source again (two bugs fixed 2026-07-13: `COMMON/BLKIN/`
had been widened off the GAMESS-UK dumpfile record layout, corrupting the per-symmetry
counts `MAXB3`; and `getv_()` was called before `IXSET_core` built the integral index
tables, so the **first** symmetry computed got an all-zero integral array and an empty
matrix — which is exactly why a `sym 1` run used to come out blank). Use
`/home/leia/Documents/ADC_build_fixed/theADCcode`; the bundled cluster binary in
`../../../../ADCanalysis/examples/DIP_h2o/` is no longer needed.

1. Integrals — `gamess` (GUK6, `~/opt/guk6/guk`; it needs the `libgfortran`/`libgcc_s`/
   `libquadmath` bundled beside it): SCF deck (`vectors extguess` / `enter 1`) then the
   transform deck (`restart`, `bypass hf`, `runtype transform`, `active 2 to 30`), with
   `ed3=dfile ed6=vfile`. Gives SCF −76.0498071428, dfile 892928 B, vfile 319488 B.
2. `theADCcode < adc4_a1.in` (ADC4CVS, `spin 2`, `SYMGRP C2v`, `sym 1`, `MCORE 1`,
   `&self-energy infinite`, `&diagonalizer full`). `sym 1` restricts to A1 so the A1 matrix
   survives on the tapes (otherwise the last symmetry overwrites them).

`adc4_a1.in` is committed here.
