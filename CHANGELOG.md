# Changelog

## v0.1.0 — first release

Scope: double ionization by **DIP-ADC(2)** and single ionization by non-Dyson **IP-ADC(3)**.

### Methods
- `-dip`: DIP-ADC(2), singlet and triplet, one sector per irrep; atom-resolved two-hole
  populations with `-mo`.
- `-sip`: IP-ADC(3) (1h / 2h1p) with the static self-energy on the main block — Σ(∞) by default,
  `three | four | fplus | off` selectable.
- `-spectrum`, `-bare`, `-convert`: Auger / ICD / ETMD decay-channel and per-orbital stick
  spectra, site grouping with `-group`, re-derived from saved documents without re-solving.
- `-format ref`: theADCcode's state listing, byte-compatible with `adcdip*.out`.

### Solvers and scale
- `-solver lanczos | lanczos-lowmem | davidson | dense`.
- `-matfree off|auto|on`: DIP 3h1p×3h1p and SIP 2h1p×2h1p blocks recomputed each mat-vec, on
  host and GPU.
- `-mgpu N`: one DIP sector row-partitioned across N NVLink GPUs; `-gpus N`: concurrent sectors.
- `-checkpoint`: Krylov checkpoint / resume across processes (SIGUSR1 → checkpoint → exit 64),
  including the Mode B banded eigensolve; Σ(∞) and SIP main-block caches.
- Backends: Go (default), OpenBLAS, CUDA, HIP.

### Tools
- `plotspec`: spectrum, electron-emission, panel, difference (`ees-diff`) and stack figures.
- `ADCanalysis`: theADCcode `adcdip*.out` / `ADC.out` → the same spectrum JSON.
- `sizeprobe`: DIP sector sizes and per-GPU memory for each `-mgpu N`, before submitting.
- `scripts/fcidump/`: FCIDUMP + MO sidecar + Molden from a deck via pyscf, including
  GAMESS-UK inputs, active spaces, counterpoise ghosts and point-charge embedding.
- `scripts/slurm/`: dump, self-resubmitting DIP and SIP jobs, GPU validation job.
- `scripts/build_cuda.sh`: kernels + CUDA binary.

### Not in this release
CVS ADC(4), ADC(2,2), Fano/Stieltjes decay widths, k-hole CI, generated ISR matrices,
transition dipole moments, and the SIP `-order 2` entry point remain on the development branch.
