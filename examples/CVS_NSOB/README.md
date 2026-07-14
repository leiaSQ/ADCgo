# CVS-ADC(4) core ionization of nitrosobenzene (NSOB)

Recreation, as ADCgo inputs, of the 2015 theADCcode run housed in
`/mnt/worka/alexk/NSOB` — the N 1s and O 1s core-ionization (XPS) spectra of
nitrosobenzene in 6-31G, originally GAMESS-UK + `adc4_constr.x` / `adc4_diag.x`.

| file | what |
|---|---|
| `_zmatrix.dat` | GAMESS-UK z-matrix, verbatim from `nsob.631G.in` (3 dummy centres, planar C<sub>s</sub>) |
| `_basisset.dat` | GAMESS-UK basis, verbatim from `nsob.631G.in`: 6-31G + d on N/O + p on H (C's d stays commented out) |
| `n1s.in` | N K-edge run — recreates `adc_1holes` → `NSOB.N1s.res` / `dav.NSOB.N1s.out` |
| `o1s.in` | O K-edge run — the O-edge twin → `NSOB.O1s.res` / `dav.NSOB.O1s.out` |

## How the legacy input maps onto ADCgo

The legacy deck spread one calculation over four files; ADCgo takes it in one.

- **`MCORE=1`** (one core hole, "CORE-NAEHERUNG") → `-order 4 -core 0`. ADCgo's
  `-order 4` *is* CVS Dyson ADC(4), and `-core` names the core orbital by its
  0-based index in the correlated space.
- **The frozen space.** The legacy run correlated 102 orbitals with 21 occupied
  (`NSOB.N1s.res`: "77 25" functions and "17 4" occupied per symmetry) out of 28
  occupied MOs — i.e. it froze 7: the six C 1s (MOs 3-8) *and the other
  heteroatom's* 1s. So the N-edge run freezes the O 1s and the O-edge run freezes
  the N 1s; the surviving 1s is the CVS core hole. That is the `frozen-list` in
  each `.in`, and it is why the core hole lands at active index 0 in both.
- **`7 1`** (the "SYMMETRIE GRUPPE" line) → `sym 1`. The core orbital is `1 A'`, so
  only the a′ sector carries the core-ionized states. (`Constanti`'s `1 2` covers
  both irreps because the constant diagrams need the whole ground state.)
- **`NROOT=15` / `nroots=20`** (Davidson) → ADCgo's block-Lanczos `-blocks`
  (default 100); it returns the full band rather than a fixed root count.
- **`Constanti` / `cst.x` / `FT05F001.CST`** have no analogue: ADCgo builds Σ
  in-process (`-sigma auto`, the all-order resolvent resummation).

## Validation against the GAMESS-UK reference

Reproduced by `dump_fcidump.py` from these files, versus `nsob.631G.out` /
`NSOB.*.res`:

| quantity | GAMESS-UK / theADCcode | ADCgo front-end |
|---|---|---|
| basis functions (cartesian) | 109 | 109 |
| point group | C<sub>s</sub> | C<sub>s</sub> |
| RHF total energy | −359.2734223925 Ha | −359.2734243159 Ha |
| correlated orbitals (a′ / a″) | 102 (77 / 25) | 102 (77 / 25) |
| occupied in the correlated space | 21 (17 / 4) | 21 (17 / 4) |
| N 1s / O 1s orbital energy | −427.4777 / −561.3252 eV | −427.47 / −561.32 eV |

The SCF energy is the `gate` in each input, so a botched geometry or basis fails
the run rather than quietly producing a wrong spectrum.

## Run

```sh
go build -o adcgo ./cmd/adcgo
scripts/adcgo_run.sh examples/CVS_NSOB/n1s.in ./adcgo    # N K-edge
scripts/adcgo_run.sh examples/CVS_NSOB/o1s.in ./adcgo    # O K-edge
```

This is a large calculation, not a smoke test: 102 orbitals give a ~540 MB
FCIDUMP, and the legacy 3h2p space was `MAXDIM=1753921`. The inputs therefore
pass `-matfree auto`, which applies the big 2h1p×3h2p coupling blocks by
recomputation instead of storing them; add `-backend cuda`/`-hip` (build-tag
gated) and raise `-maxmem` on a GPU host. Generated outputs (`*.fcidump`,
`*.mo.json`, `*.adc.json`, …) are git-ignored.

The `sidecar` line also writes MO coefficients, the AO overlap and dipole
integrals, which is what `-tdm` needs if you want core→valence X-ray *emission*
on top of the ionization spectrum.
