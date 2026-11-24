# scripts

| Path | What it is |
|---|---|
| `fcidump/` | Integral generation: `dump_fcidump.py` (deck format in `adcgo_input.py`), GAMESS-UK basis/z-matrix readers, orbital selection, `threads.py` |
| `adcgo_run.sh` | Deck → dump → `adcgo` in one step, for systems that run interactively |
| `build_cuda.sh` | Build the CUDA binary `adcgo-cuda` (kernels + `go build -tags cuda`) |
| `slurm/` | Batch templates: `dump.sbatch`, `dip.sbatch`, `sip.sbatch` (self-resubmitting checkpoint chains, `chain.sh`), `gpu_tests.sbatch` |
| `fixtures/` | Regenerate the committed test fixtures in `testdata/` (needs pyscf) |

Scripts locate the repository from their own path, so run them from any directory.

## Small systems: one command

```sh
scripts/adcgo_run.sh examples/h2o_dzp/dip.in ./adcgo     # dump + DIP-ADC(2)
scripts/adcgo_run.sh examples/h2o_dzp/sip.in ./adcgo     # dump + SIP-ADC(3)
```

The deck's `&output` names the FCIDUMP and MO sidecar; its `&adc` line supplies the `adcgo` flags.
`ADCGO_PYTHON` points at a Python with pyscf if `python3` is not it.

## Large systems: the batch pipeline

A DIP and a SIP run over the same orbital space share one FCIDUMP. Each solve outlasts a walltime,
so `dip.sbatch` and `sip.sbatch` checkpoint and resubmit themselves until converged (exit 0 =
done, 64 = resumed by the queued successor, anything else stops the chain).

```sh
scripts/build_cuda.sh                                          # once per checkout / GPU arch
go run ./cmd/sizeprobe sys/sys.fcidump                         # sector sizes, per-GPU memory per -mgpu N
sbatch --gres=gpu:2 scripts/slurm/gpu_tests.sbatch             # once per GPU model

# 1. integrals (CPU node; the SCF and the 4-index transform need memory, not GPUs)
DECK=sys/sys.in sbatch --cpus-per-task=64 --mem=500G scripts/slurm/dump.sbatch

# 2. solves. DIP: whole band, matrix-free, one sector across 8 NVLink GPUs, one spin per job.
for s in singlet triplet; do
  FCIDUMP=sys/sys.fcidump MO=sys/sys.mo.json SPIN=$s \
    sbatch --gres=gpu:8 --cpus-per-task=64 --mem=1000G --time=120:00:00 scripts/slurm/dip.sbatch
done
#    SIP: one GPU does the solve; the host memory is for the Σ(∞) build.
FCIDUMP=sys/sys.fcidump sbatch --gres=gpu:1 --cpus-per-task=64 --mem=1000G --time=120:00:00 \
  scripts/slurm/sip.sbatch
```

- Add your site's `--account`/`--partition`, and pin the GPU model (e.g. `--gres=gpu:H200:8`) if
  the cluster mixes models; a run's timings are only comparable on the same model.
- The variables each template reads are listed in its header (`BLOCKS`, `MGPU`, `SOLVER`,
  `CHECKPOINT_EVERY`, `EIG_B2`, `EIG_DEVICE`, `EXTRA`, ...).
- `adcgo` sizes the device footprint before allocating and refuses a sector that does not fit, with
  the term (operator or Krylov basis) that overflowed.
- A DIP `-mgpu` run writes each finished sector beside `-out` (`*.s<spin>.i<irrep>.json`), so a
  failure in a later sector does not lose an earlier one.
- SIP caches Σ(∞) and the 1h main block beside the FCIDUMP; later generations and reruns skip both.
  A finished SIP solve cannot be extended to more blocks: rerun with the higher `BLOCKS` (the caches
  make that cheap, and Lanczos is deterministic).

## Spectra from the solver documents

The solver documents carry energies, pole strengths and (with `-mo`) two-hole populations;
`-convert` turns them into spectra without re-solving.

```sh
adcgo -convert sys.sip.json -sip -bare -out sys.sip.spec.json
adcgo -convert sys.dip.json -dip -bare -out sys.dip.spec.json           # one "states" channel
adcgo -convert sys.dip.json -dip -spectrum -mo sys.mo.json \
      -group="A=C1,C2,...,H8" -group="B=C9,...,H16" -init-atom A \
      -out sys.dip.chan.spec.json                                       # decay channels
plotspec -mode panel -sip sys.sip.spec.json -dip sys.dip.chan.spec.json -out panel.png
```

`-format ref` re-emits saved documents as theADCcode's state listing:
`adcgo -convert sys.sip.json,sys.dip.json -format ref -out sys.ref.out`.

### Decay channels: define the sites, or the channels are meaningless

The classifier routes each dication state by where its two holes sit relative to the initially
ionized site: both on it = Auger, one on it and one elsewhere = ICD, both on one other site =
ETMD(2), on two other sites = ETMD(3). Without `-group`, every atom (population column) is its own
site, so on a molecule with tens of atoms "two different sites" holds for almost every state and
ETMD(3) swallows the spectrum. Fold the atoms into the chemical units the question is about.

- `-group` takes `=` (`-group=NAME=cols`); `~col` makes a column passive. A bare `-group` prompts.
- `-init-atom` defaults to `O`; with `-group` it takes a site name.
- On a homodimer, `Auger@A` and `ETMD(2)` must carry equal weight by symmetry: a free check.
- `plotspec -dip-family` collapses the channels to Auger / ICD / ETMD for the figure.

## Interaction effects: dimer minus 2 × monomer

Isolates what a dimer does that two isolated monomers cannot. It needs SIP **and** DIP, whole band,
for both systems, and the monomer must be a **counterpoise** fragment — the partner kept as ghost
basis centres (`&geometry ghost 13 to 24` in the deck) — or the difference measures basis-set
superposition error instead of physics.

```sh
plotspec -mode ees-diff \
  -sip dimer.sip.spec.json -dip dimer.dip.spec.json \
  -ref-sip mono.sip.spec.json -ref-dip mono.dip.spec.json \
  -ref-scale 2 -xrange 0-16 -points 600 -fwhm 1.0 -out diff.png -data diff.tsv
```

- `-xrange` is required (a common electron grid); its useful upper bound is
  `max(SIP energy) - min(DIP energy)`.
- Intensities are absolute in `ees-diff`: peak normalisation would make the factor 2 meaningless,
  and the 2 is exact only for a homodimer whose halves are congruent.
- Channels are summed before subtracting: ghosting lowers the point group, so the fragment's
  irrep decomposition does not correspond to the dimer's.
- A monomer DIP has only one-site final states; the dimer also has one hole on each monomer.
  `-ref-twosite` adds that family to the reference by self-convolving `-ref-sip` (independent
  holes, weight `s_i·s_j`, `N²` synthetic states), so the residual is interaction alone.
  `-twosite-shift` adds the hole–hole repulsion `14.4/R[Å]` eV, with R the monomer centroid
  distance; 0 is the strict non-interacting limit.
- With `-ref-sip/-ref-dip`, `-mode panel` draws panel (c) as the stacked one-site (Auger + ETMD(2))
  and two-site (ICD) background with the dimer over it.

## Looking at the orbitals

The `.mo.json` sidecar has MO coefficients but no basis set, so no viewer can draw it. `--molden`
writes a Molden file (GaussView, Molden, Avogadro, Jmol, Multiwfn) from the same MOs:

```sh
python scripts/fcidump/dump_fcidump.py --input sys.in --molden sys.molden
```

Orbital numbering matches ADCgo's (both count the active space), so `<55|` in a state listing is
MO 55 in the viewer. Orbital energies are written as zeros; isosurfaces are unaffected.
