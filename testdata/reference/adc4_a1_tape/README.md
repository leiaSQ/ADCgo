# CVS IP-ADC(4) reference tape — A1 sector of H2O/DZP

Companion to `../adc4_b2_tape`. Same matched DZP integrals
(`../h2o_dzp.matched.fcidump`), but the **A1** sector (`sym 1`), which contains the
core hole — so this tape has the **1h main block** (dim 1712 = 1 [1h] + 46 [2h1p] +
1665 [3h2p]). It therefore validates the 1h couplings that B2 could not: KOPP1/KOPP2
(1h↔2h1p) and KOPP4 (1h↔3h2p).

Tape format identical to the B2 fixture (see its README): `FT21F001.ADC` off-diagonals,
`FT18F001.ADC` header + diagonal (here `nh12=47` = 1h + 46 2h1p; diagonal[0]=1.37063 is
the core-hole 1h diagonal = −ε_core − Σ). Row/col index 1 = the 1h main.

## Regeneration (this is how the tape was produced)

The original `theADCcode`/`gamess` binaries could read the shared dfile; a *rebuilt*
theADCcode misreads per-symmetry counts (MAXB3) — so use the **bundled** binaries from
`../../../ADCanalysis/examples/DIP_h2o/` (`gamess`, `theADCcode`) with a boost shim:

1. Regenerate integrals with `gamess` (bundled libgfortran/quadmath/gcc_s on
   `LD_LIBRARY_PATH`): SCF deck (`vectors extguess` / `enter 1`, `ed3=dfile`) then the
   transform deck (`restart`, `bypass hf`, `runtype transform`, `active 2 to 30`,
   `ed3=dfile ed6=vfile`). Gives SCF −76.0498, dfile 892928 B, vfile 319488 B.
2. `theADCcode` needs `libboost_regex.so.1.85.0`; shim it with the system boost
   (`ln -sf /usr/lib64/libboost_regex.so.1.90.0 libboost_regex.so.1.85.0`).
3. Run `theADCcode` with `adc4_a1.in` (ADC4CVS, `spin 2`, `SYMGRP C2v`, `sym 1`,
   `MCORE 1`, `&self-energy infinite`, `&diagonalizer full`). `sym 1` restricts to A1 so
   the A1 matrix survives on the tapes (otherwise the last symmetry overwrites them).

`adc4_a1.in` is committed here. The 3h2p columns are `pam`/ELIM-permuted (see B2 README).
