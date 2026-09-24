# internal/ — implementer's guide

This guide is for people who know theADCcode (C++/C/Fortran) and need to work in ADCgo. It
covers how a run flows, which Go package corresponds to which reference module, and the
conventions carried over from the reference. For flags and usage, see the top-level
[README](../README.md) and `adcgo -h <topic>`.

Paths such as `adc2_dip/singlet.cpp` are relative to the theADCcode root. In Go comments they
are written as `../ADC/...`.

**Finding the Go counterpart of a reference file.** Every ported function names its source in
a comment, so grep for the reference file name:

```sh
grep -rn 'calc_c12_2.c' internal/      # → internal/adc/sip/elements.go  c12_2
grep -rn 'singlet.cpp'  internal/      # → internal/adc/dip/singlet.go, blocks.go
grep -rn 'davidson.F'   internal/      # → internal/adc/lanczos/davidson.go
```

## Three layers

```
cmd/adcgo/        driver: flags → per-sector loop → output     ≈ main.cpp, adc_selector.cpp, input_data.cpp
internal/adc/     physics: configuration spaces, matrix elements, self-energy, solvers, analysis
backend/          linear algebra: Gonum/OpenBLAS (host), cuBLAS/hipBLAS (GPU), multi-GPU  ≈ blas_matrix.cpp
```

The layout follows three rules:

- **Physics packages know nothing about flags, files or output formats.** They take a space,
  integrals and orbital energies, and return values. `cmd/adcgo` does all the I/O.
- **Only `backend/` calls BLAS, LAPACK or CUDA.** Element code builds `backend.Mat` blocks, and
  solvers call `be.Gemm`, `be.SymEig` and so on. That is why the same DIP code runs on a laptop
  and on 8×H200: only the `Backend` behind it changes.
- **Each method is one package; the shared infrastructure is shared.** `dip`, `sip` and `khci`
  each own a configuration space and a matrix. All of them use `integrals`, `lanczos` and
  `backend`.

## One run, side by side

theADCcode, `main.cpp`:

```
selector = Adc_selector()                 # parse stdin input, SCF_data_reader, Integral_table
self     = selector->new_self_energy()
for spin, sym:
    mat = selector->new_adc_matrix(sym, spin, self)   # Adc2_matrix / ND_ADC3_matrix / ADC4CVS_matrix
    mat->accept_self_energy(self)
    for an in analyzers:                  # ADC_diagonalizer, ADC2_DIP_analyzer, ADC_saver, ...
        mat->accept_analyzer(an)
```

ADCgo, `cmd/adcgo/main.go` (`runDIP` and `runSIP`; `runFano` and `runISR` have the same shape):

```
d     = fcidump.ReadFile(path)                        # replaces the guk/gus/molcas front end
nocc  = mp.NOcc(d); eps = mp.OrbitalEnergies(d, nocc) # ε from the Fock diagonal
ints  = integrals.New(d, nocc, orbSym)                # Integral_table
Σ     = buildSigma(...)                   (SIP only)  # Self_energy, built once for all sectors
for (spin,) irrep:
    sp  = dip.NewSpace / sip.NewSpace{,4,22}          # add_*_configs
    be  = chooser.pickLanczos / pickDense             # which backend runs this sector
    mx  = dip.New / sip.New(sp, ints, eps, [order,] be)   # the ADC_matrix
    mx.SetStaticSelfEnergy(Σ)                         # accept_self_energy
    res = lanczos.Solve | SolveLowMem | SolveDavidson | SolveDense   # ADC_diagonalizer
    sec = analyze.BuildSector / BuildSIPSector        # ADC2_DIP_analyzer, adc_analyzer
emitJSON(doc) | emitRef(...) | spectrum.Build*(...)   # ADC_saver / printed output
```

**The main structural difference.** The reference uses a visitor: the matrix accepts
analyzers, and the diagonalizer is one of them. ADCgo writes the same pipeline as plain,
sequential code: space → matrix → solve → analyze → emit. Each stage passes a plain value to the
next (`*dip.Space`, `lanczos.Result`, `analyze.Sector`), so a test or a new feature can stop
after any stage and inspect the result.

### The `ADC_matrix` interface

`adc_matrix.hpp` corresponds to `lanczos.Operator` (`internal/adc/lanczos/lanczos.go`):

| theADCcode `ADC_matrix` | ADCgo | Notes |
|---|---|---|
| `size()` | `Size()` | |
| `main_block_size()` | `MainBlockSize()` | 2h (DIP) or 1h (SIP) rows come first, same as the reference |
| `operator()(block_out, block_in, count)` | `ApplyBlock(out, in)` | block mat-vec σ = M·X |
| `build_matrix(mat)` | `BuildMatrix()` (`DenseOperator`) | for `-solver dense` and tests |
| `build_transition_amplitudes` | `sip.Matrix.FMatrix()`, `sip/amplitudes.go` | spectroscopic amplitudes |
| `get_conf(i)` | `Space.Configs[i]`, `Holes`, `Particles` | the configuration lives on the space, not the matrix |
| `accept_self_energy` | `sip.Matrix.SetStaticSelfEnergy` | same sign convention: `main -= Σ` |
| `accept_analyzer` | none; the driver calls `analyze.*` after solving | |

ADCgo keeps the **space** (`Space`: which configurations exist and in what order) separate
from the **matrix** (`Matrix`: elements and the mat-vec). In the reference both live on one
class. Keeping them apart lets Fano restrict a space to a subset of rows (`Space.Restrict`)
and build an ordinary matrix over it.

## Reference module → ADCgo package

### Methods

| theADCcode | ADCgo | What corresponds to what |
|---|---|---|
| `adc2_dip/config.hpp`, `Adc2_matrix::add_{ii,ij,jiir,ijkr}_configs` | `adc/dip/config.go` | `NewSpace`; `addII`, `addIJ`, `addJIIR`, `addIJKR`; same row order |
| `adc2_dip/singlet.cpp`, `triplet.cpp` | `adc/dip/singlet.go`, `triplet.go` | one method per `build_submatrix_*`: `iiJJ`↔`ii_jj`, `ijKL`↔`ij_kl`, `lkkII`↔`lkk_ii`, `jiiLKK`↔`jii_lkk`, `ijkLMN`↔`ijk_lmn`, … |
| `adc2_dip/adc2_dip_blocks.hpp` | `adc/dip/blocks.go` | V/A/B blocks, `vplus`/`vminus`, second-order W/U sums |
| `Adc2_matrix::operator()`, `multiply_submatrix_*` | `adc/dip/matvec.go` (assembled), `matfree*.go` (recomputed), `satelem.go` | see [Performance layers](#performance-layers) |
| `ndadc3_ip/nd_adc3_matrix.cpp::add_configs` | `adc/sip/config.go` | `NewSpace`: 1h main, 2h1p satellites |
| `ndadc3_ip/calc_c11_2.c`, `calc_c11_3.c` | `adc/sip/elements.go` `c11_2`, `c11_3` | main block |
| `ndadc3_ip/calc_c12_1.c`, `calc_c12_2.c` | `adc/sip/elements.go` `c12_1`, `c12_2` (`vv1`/`vv2` = the `VV1`/`VV2` macros) | main↔satellite coupling |
| `ndadc3_ip/calc_c22_1_{dia,off}.c`, `calc_k1/k2.c` | `adc/sip/elements.go` `c22diag`, `c22off` | satellite block |
| `ND_ADC3_matrix::build_main_block` etc. | `adc/sip/matvec.go` `mainBlock`, `coupling`, `satBlock` | |
| method `adc2ipx` / `ndadc3ip` | `sip -order 2` / `-order 3` | same `ND_ADC3_matrix`, different order |
| `adc4core/adc4_constr/state.F` | `adc/sip/config4.go` | CVS 1h / 2h1p / 3h2p spaces, reference tape column order |
| `adc4_constr/kopp{1,2,3,4}.F`, `wert{1,2,3}.F`, `sum{1,3,4}.F` | `adc/sip/elements4.go`, `kopp3.go` | `kopp1`…`kopp4`, `wert2elem4`, `wert3elem`, `sum1_4`… |
| `adc4_constr/egf.F`, `ab3.F`, `ab5.F`, `selec.F` | `adc/sip/matvec4.go` | ADC(4) block assembly |
| `adc4_constr/init*.F` coefficient tables | `adc/sip/coeff4.go` (generated by `scripts/codegen/gen_coeff4.py`) | |
| `ADC4CVS_matrix` + the guk6 tapes | none needed | integrals come from FCIDUMP, so ADC(4) no longer depends on guk6 |
| `ndadc3_prop/my_calc_d{11,12,22_diag,_null}.c` | `adc/sip/isrdipole*.go` | ISR dipole / property matrix for `-tdm` |

### Self-energy, integrals, input

| theADCcode | ADCgo | Notes |
|---|---|---|
| `self_energy/original/original_self_energy.cpp` | `adc/selfenergy/density2.go`, `density3.go`, `dynamic3.go`, `linear.go` | orders `three`, `four`, `fplus` |
| `self_energy/constanti/` (`aufbau1/2.f`, `ab3.f`, `masop.f`, `inversion.f`) | `adc/selfenergy/coupling.go`, `satmatrix.go`, `satspace.go`, `resolvent.go` | Σ(∞), `-sigma infinite` (the default) |
| `&self-energy` keyword | `selfenergy.ParseScheme`, `-sigma` | |
| `integral_table.cpp`, `integral_blocks.hpp` | `adc/integrals` | same V/A/B accessors, per virtual-symmetry group |
| `scf_data/`, `libphis` (guk, gus, molcas readers) | `adc/fcidump`, `adc/mo`, `adc/mp` | FCIDUMP plus a JSON sidecar (MO coefficients, AO overlap, dipoles), written by `scripts/fcidump/` |
| `input_data.cpp`, `adc_selector.cpp` | `cmd/adcgo/main.go` flags | see the input table below |

### Solvers and analysis

| theADCcode | ADCgo | Notes |
|---|---|---|
| `libLanczos/lanczos.h`, `lanczos_engine.h` | `adc/lanczos/lanczos.go` (`Solve`), `lowmem.go` (`SolveLowMem`) | same start block: the main-space unit vectors |
| `libLanczos/bnd2td.f`, `tddiag.f` | `adc/lanczos/bandeig.go` | line-by-line port, used by `SolveLowMem` |
| `adc4core/adc4_diag/davidson.F` | `adc/lanczos/davidson.go` | same preconditioner and restart |
| `analysis/adc_diagonalizer.cpp` | `lanczos.Solve*` plus `solveDIPSector` / `solveSIPSpace` in `cmd/adcgo` | |
| `analysis/adc2_dip_analyzer.cpp` (popana) | `adc/analyze/analyze.go`, `populations.go` | two-hole populations; `au2eV` identical |
| `analysis/adc_analyzer.cpp` | `adc/analyze/analyze.go`, `sip.go` | pole strengths, leading configurations |
| `analysis/adc_saver.cpp`, printed `adcdip*.out` / `ADC.out` | `emitJSON`, `-format ref` | `-format ref` reproduces `adcdip*.out` byte for byte |
| `adc2_pol/master_fano_new.f90`, `select_fano.f90`, `partgammas.f90`, `fspace.f90` | `adc/fano` | the algorithm, applied to IP/DIP instead of PP |
| `adc2_pol/stieltjes_phi1.f` | `adc/stieltjes` | |

### Not ported

`adc2_pol` (the polarization propagator itself; only its Fano algorithm was ported),
`adc2_prop`, `subspaceCAP` and the full/subspace-CAP analyzers, `propagation_analyzer`, the
`ndadc3ap` affinity propagator, `adc_from_file` (`ADCLOAD`), and the `&cma` orbital
analyzer.

### No reference counterpart

| Package | What it is |
|---|---|
| `adc/sip` `*22.go` | ADC(2,2) for single ionization (1h / 2h1p / 3h2p), `-order 22` |
| `adc/khci` | k-hole CI (k = 1…4) in spin-orbital determinants; the exact reference for `isrgen` |
| `adc/isrgen/{ip,dip,tip,qip}` | matrix elements generated by adcgen. **Generated, do not edit**; see [isrgen/README](adc/isrgen/README.md) |
| `adc/isrgen/sigma` | σ = M·Y for the generated matrices, written as tensor contractions |
| `adc/quip` | ADC(2,2) for quadruple ionization |
| `adc/spectrum` | decay-channel classification (Auger/ICD/ETMD), absorbed from the ADCanalysis tool |
| `adc/refout`, `adcanalysis/` | parsers for theADCcode's text output, used for validation and `cmd/adcanalysis` |
| `adc/matfree`, `adc/parallel` | shared memory and worker-pool policy |

## Input: reference sections → flags

| theADCcode input | adcgo |
|---|---|
| `&frontend guk\|gus\|molcas` | `-fcidump FILE` (+ `-mo FILE`) |
| `&propagator adc2dip` | `-dip` |
| `&propagator adc2ipx \| ndadc3ip` | `-sip -order 2 \| 3` |
| `&propagator adc4cvs` + `&adc4` | `-sip -order 4 -core i,j` |
| `&propagator sym … spin …` | `-sym all\|none\|k` (0-based), `-spin both\|singlet\|triplet` |
| `&self-energy` | `-sigma auto\|off\|three\|four\|fplus\|infinite`, `-sigma-akrit`, `-sigma-maxit` |
| `&diagonalizer full \| lanczos \| davi` | `-solver dense \| lanczos \| lanczos-lowmem \| davidson` |
| `&diagonalizer iter N` | `-blocks N` (same meaning; see `lanczos.Options`) |
| `&diagonalizer nroots / maxdavsp / convthr` | `-nroots`, `-maxdavsp`, `-convthr` |
| `&eigen ps / thresh` | `-ps-thresh` (percent), `-coeff-thresh` |
| `&popana` groups | `-mo` sidecar + `-group NAME=…` |
| `&dip` | `-tdm` |

## Conventions carried over

- **Where the paper and the reference code disagree, the code wins.** Package docs say so
  (`dip/config.go`, `sip/config.go`). Matrix elements were ported from the `.cpp`/`.c`/`.F`
  files, not rederived.
- **Names follow the reference.** Configuration families (`ii`, `ij`, `jiir`, `ijkr`),
  element routines (`c11_2`, `kopp2`, `wert3elem`) and integral accessors (`V`, `A`, `B`,
  `vplus`, `vminus`) keep their reference names so the two can be read side by side.
- **Integrals:** chemist notation `(pq|rs)` in storage. `v(a,b,c,d)` in `sip`/`selfenergy`
  is the reference `V1212(a,b,c,d)` = `<ab|cd>` = `Eri(a,c,b,d)`.
- **Indices are 0-based everywhere.** Orbital indices are absolute (virtual `v` is orbital
  `nocc+v`). Irreps are 0-based internally and printed 1-based (`irrep=1`), as in the
  reference output.
- **Spin:** `dip.Singlet = 0`, `dip.Triplet = 2`, matching the reference `spin()`. SIP is
  a single doublet channel.
- **Symmetry:** FCIDUMP `ORBSYM`; the direct product is XOR (D2h and its subgroups).
- **Units:** Hartree internally; eV only at output (`analyze.au2eV = 27.211396`, the
  reference value).
- **Orbital energies** are not stored in FCIDUMP. `mp.OrbitalEnergies` rebuilds them from
  the Fock diagonal, which assumes a canonical RHF reference.

## Performance layers

The physics and the performance code are in separate files, so the element code reads like
the reference and the machinery can change without touching it:

| Layer | Files | Touch it when… |
|---|---|---|
| Configuration space | `dip/config.go`, `sip/config*.go` | adding or reordering configurations |
| Matrix elements | `dip/singlet.go`, `triplet.go`, `blocks.go`; `sip/elements*.go`, `kopp3.go` | changing the physics |
| Block assembly (dense, per block) | `dip/matvec.go`, `sip/matvec*.go` | adding a new block |
| Matrix-free apply (recompute each mat-vec) | `dip/matfree*.go`, `sip/matfree*.go` | a block too large to store; `-matfree` |
| GPU kernels | `backend/*.cu`, `backend/cuda_kernels.go` | only for the matrix-free device path |
| Multi-GPU | `backend/distributed.go`, `dip/matfree_dist.go` | see [backend/README](../backend/README.md) |

The matrix-free and GPU paths have parity tests (`*Parity`, `*Gate`) against the assembled
path. Change the physics in the element code, then run those tests to confirm the faster
paths still agree.

## Where ADCgo deliberately differs numerically

These differences are documented at their use sites and bounded by tests:

- `dip/blocks.go` precomputes the second-order W/U sums as tables, which reassociates the sum
  (~1 ulp per term). `TestSecondOrderTablesMatchDirectSums` covers it.
- `sip/elements.go` `c11_3` changes the summation order and reassociates the divisions
  (~1e-16 relative). `TestSIPMatchedReference` covers it.
- `lanczos.Solve` diagonalizes the projected matrix densely instead of with the banded
  solver. `SolveLowMem` uses the banded port, as the reference does.
- `lanczos/davidson.go` orthogonalizes with two-pass CGS2 instead of single-pass Gram–Schmidt.
  The converged roots are the same to the residual threshold.

## Validating a change against the reference

Reference outputs are committed under `testdata/reference/`: `adcdip*.out`, ADC(4) matrix
tapes, and `h2o_dzp.sip.ADC.out`.

```sh
go test ./internal/adc/validate/ -run 'Matched|Reference'   # DIP-ADC(2) and SIP vs theADCcode output
go test ./internal/adc/sip/ -run 'Gate'                     # element-level gates, incl. ADC(4) vs the FT19 tape
go test ./...                                               # everything that runs without a GPU
```

When porting a new reference routine:

1. Put it in the package that owns the method, next to its siblings, and name it after the
   reference routine.
2. Add a comment naming the source file (and lines) so the grep above finds it.
3. Gate it element by element against a reference dump or tape before wiring it into the
   mat-vec.
