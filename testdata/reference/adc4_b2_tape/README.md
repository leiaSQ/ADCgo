# CVS IP-ADC(4) reference tape — B2 sector of H2O/DZP

Matched-integral reference for the SIP-ADC(4) matrix (Track A / milestone A2),
produced by theADCcode (`../ADC/adc4core`, `adc4cvs_matrix`) on the same DZP
integrals exported as `testdata/reference/h2o_dzp.matched.fcidump` (NORB=29,
NELEC=8, core = O 1s = orbital 0, `MCORE 1`).

These are FORTRAN unformatted tapes (`adc4.in` = the run's keyword input):

- **`FT21F001.ADC`** — off-diagonal matrix elements. Records of NBUF=1000:
  `AMATRX(1000) f8 | IOI(1000) i4 | JOJ(1000) i4 | count i4 | jdum i4` (16008 bytes
  each; last record's `count` < 1000). 1-based row/col indices. Written by
  `ab3.F`/`ab5.F`; the last symmetry computed (B2, `nrsym=4`) is what survives.
- **`FT18F001.ADC`** — diagonal. Record 1 = header `nh12 i4 | mcore i4 | nrsym i4 |
  nbuf i4 | mtxid char*4` (20 bytes); record 2 = `nh12` f8 = the **2h1p** diagonal.
  (`RSCRT1` rewrites this tape with only the first `nh12` entries, discarding the 3h2p
  `EIGAB` diagonal — which is why FT19 below exists.)
- **`FT19F001.ADC`** — the 3h2p effective diagonal (EIGAB = 0th-order orbital-energy sum +
  the 5th-order WERT3 correction), dumped by `ab5.F` before RSCRT1 can truncate it.
  Record 1 = `IDIM NCOL NECORE N3H2P` (4× i4); record 2 = `N3H2P` f8. Columns carry the
  same pam/ELIM permutation as FT21's, so compare as a multiset. See `TestADC4EigabGate`
  and `../adc4_a1_tape/README.md` for regeneration.

B2 has no core hole in its own irrep, so there is **no 1h main block** — the matrix
is 42 (2h1p) + 1646 (3h2p) = 1688. This tape therefore validates the 2h1p/2h1p block
(WERT1, 3rd+4th order) and the 2h1p↔3h2p coupling (WERT2), but not the KOPP 1h
couplings (those need an A1 tape). See `internal/adc/sip/adc4_gate_test.go`.

The reference permutes 3h2p columns internally (`ab5.F` `pam`/ELIM reordering), which
does not change eigenvalues; the coupling block is therefore compared
permutation-invariantly (per-row sorted multiset).
