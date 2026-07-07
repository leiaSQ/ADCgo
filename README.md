# ADCgo

An exact, hardware-accelerated ADC(n) **ionization** solver in Go.

ADCgo computes the algebraic-diagrammatic-construction (ADC) secular problem for
electron removal (DIP first, then SIP/IP) *exactly* — no reduced-scaling
truncations — and reaches larger systems by acceleration (multicore OpenBLAS,
GPU Lanczos via **hipBLAS** on AMD and **cuBLAS** on NVIDIA) rather than
approximation. SCF + molecular integrals are delegated: ADCgo ingests a standard
**FCIDUMP** (e.g. from pyscf).

See [`ADCgo_plan.md`](ADCgo_plan.md) for the full design and milestones.

## Status: M6 — the decay-channel spectrum

The pipeline now runs end to end: FCIDUMP → solve → populations → **decay
channels → stick spectrum** in one `adcgo` call. The DIP-ADC(2) double-ionization
solver (M1) is implemented, **blocked by point-group irrep** (M2, C₂ᵥ for H₂O),
runs on **accelerated backends** (M3), and is **cross-validated against an
independent DIP-ADC(2)** (M4). **M5 added single ionization** (SIP): a **non-Dyson
IP-ADC(3)** solver (1h main / 2h1p satellite, ported from theADCcode's
`ndadc3_ip`, J. Breidbach, JCP 109 (1998) 4734) reusing the same engine —
configuration space + assembled block-sparse mat-vec + Lanczos — one branch
smaller than DIP. `-order 2|3` selects the ADC order (order 2 is the reference's
*extended* ADC(2): 1st-order satellite block kept, not the bare standard ADC(2)).

**M6 absorbs the decay-channel interpretation layer** (`classify` + `spectrum`,
ported from the sibling tool ADCanalysis). Each dicationic state's atom-resolved
two-hole population is routed onto named channels relative to a chosen
core-ionized site — **Auger@A** (both holes on A), **ICD:A→B** (one on A, one on
a neighbour), **ETMD(2)/ETMD(3)** (holes off A) — and flattened into a stick
spectrum. `-spectrum` emits that spectrum (SIP flattens per orbital instead).
Decay sites are set with `-group NAME=col1,~col2` (a `~`-prefixed column is
passive: its holes are discounted); a **bare `-group` drops into an interactive
site dialogue**. The JSON matches ADCanalysis's schema field-for-field, so its
`plotspec` renders ADCgo output unchanged (rendering stays there — ADCgo is the
solver).

SIP is validated on *matched integrals* against **pyscf's IP-ADC** (`ip_adc`),
which — unlike DIP-ADC — pyscf computes natively: ADCgo and pyscf read the same
H₂O/cc-pVDZ MO integrals, so any residual is method, not basis. The 2h1p satellite
roots and the spectroscopic factors agree with pyscf to **~1e-5 / ~5e-3**; the
strong 1h main lines sit ~0.03–0.16 eV above pyscf's ISR IP-ADC, a small
self-energy-formulation difference (documented, one-sided). The exact ADCgo
numbers are pinned by an in-process regeneration guard to 1e-8.

The DIP M4 external check remains: theADCcode's h2o DIP reference
(`adcdip{1..4}.out`) on matched integrals via `scripts/gen_ref_fcidump.py`
(reproducing the DZP+diffuse basis + frozen-core active space, gated on SCF =
−76.0498071428 Ha). Result: **every strong DIP line matches in irrep and leading
two-hole configuration, pole strengths within ~3%**; ADCgo's DIP energies sit
0.04–3.2 eV *above* that reference, which used a higher-order ("Order: 4+") static
self-energy — a documented method difference, not an error.

M3 backends (build-tag gated; default is pure-Go):

- **gonum** (default) — pure-Go reference; with `-tags openblas` its BLAS/LAPACK
  engine is swapped for multicore OpenBLAS (same code path).
- **hip** (`-tags hip`) — hipBLAS/ROCm; built and validated here on a Radeon 890M
  (gfx1150), deployment target AMD Instinct.
- **cuda** (`-tags cuda`) — cuBLAS; a structural twin of the hip shim, compiled
  against CUDA 13.x here and run on NVIDIA hardware.

The pipeline is:

1. `scripts/gen_fcidump.py` runs RHF + MP2 on H₂O/cc-pVDZ with **pyscf** in C₂ᵥ
   symmetry, writing `testdata/h2o.fcidump` (with 1-based Molpro `ORBSYM`), the
   golden `testdata/h2o.ref.json`, and the C/S sidecar `testdata/h2o.mo.json`
   (MO coefficients + AO overlap + AO→atom map).
2. `internal/adc/fcidump`, `mp` — integral ingestion + MP2 (M0).
3. `internal/adc/backend` — the handle-based linear-algebra interface (resident
   `Vector`/`DeviceMat`) plus the gonum, OpenBLAS, hipBLAS, and cuBLAS backends.
4. `internal/adc/integrals` — occ/vir-blocked, **per-virtual-symmetry** V/A/B
   integral accessors.
5. `internal/adc/dip` — DIP configuration spaces, the singlet/triplet building
   blocks (ported from `../ADC/adc2_dip`, Tarantelli Chem. Phys. 329 (2005) 11),
   the matrix-free mat-vec, and a dense `BuildMatrix` for validation.
   `internal/adc/sip` — the SIP (IP-ADC(n)) analogue: 1h/2h1p spaces + block
   elements ported from `../ADC/ndadc3_ip`, plus the F-matrix (spectroscopic
   amplitudes), reusing the same assembled-operator / Lanczos path.
6. `internal/adc/lanczos` — block-Lanczos band solver + a dense reference path.
7. `internal/adc/analyze` — pole strengths, leading 2h configs, spurious-root
   purge, and atom-resolved 2h populations (Tarantelli U-transform).
8. `internal/adc/spectrum` — the decay-channel layer (M6): routes each state's
   two-hole population onto Auger/ICD/ETMD channels (`classify`) and flattens the
   solved sectors into the stick-spectrum JSON (`BuildDIP`/`BuildSIP`), consuming
   the in-memory `analyze` sectors directly — no text round-trip. Emits the same
   schema ADCanalysis's `plotspec` renders.
9. `internal/adc/refout` + `internal/adc/validate` — parse theADCcode's
   `adcdip*.out` and cross-validate ADCgo against it on matched integrals (M4).

### Validation gates (all green)

- **`Apply == BuildMatrix`**: the matrix-free mat-vec reproduces the dense matrix.
- **dense == Lanczos**: the block-Lanczos spectrum reproduces the dense one.
- **symmetry union == full spectrum**: because the H₂O integrals genuinely carry
  C₂ᵥ symmetry, the symmetry-off matrix is exactly block-diagonal by irrep, so the
  union of the per-irrep spectra reproduces the full spectrum to <1e-8 (M2 gate).
- **physics**: lowest singlet DIP energy 39.25 eV (¹A₁; experiment ≈ 40.7 eV),
  pole strengths 83–88% for main states; ground state localized on O (Auger@O).
- **PopSum ≈ ps/100**: two-hole populations sum to the pole strength, per sector.
- **cross-backend agreement** (M3): each hipBLAS BLAS-1/2 kernel matches gonum to
  ~1e-11, and a full device-resident Lanczos solve reproduces the pure-Go dense
  spectrum to <1e-7 (`-tags hip`, on the GPU). OpenBLAS reproduces the pure-Go
  spectrum to ~1e-13.
- **external reference** (M4): on matched DZP+diffuse integrals (SCF gate
  −76.0498 Ha), ADCgo reproduces theADCcode's `adcdip*.out` DIP structure — every
  strong line's irrep + leading 2h config, pole strengths within ~3%, the ¹A₁
  ground state to 0.15 eV, O/H populations to table rounding. Energies sit 0.04–3.2
  eV above (the reference's 4+ static self-energy). A committed fixture
  (`testdata/h2o_dzp.adcgo.json`) is pinned to the in-process solver to 1e-8.
- **SIP vs pyscf** (M5): on matched H₂O/cc-pVDZ integrals, ADCgo's IP-ADC(2)/(3)
  reproduces pyscf's `ip_adc` — 2h1p satellite roots and spectroscopic factors to
  ~1e-5 / ~5e-3; strong 1h main lines within a one-sided ~0.03–0.16 eV
  self-energy-formulation band. The per-irrep union equals the symmetry-off
  spectrum to <1e-8, and a committed fixture (`testdata/h2o_sip.adcgo.json`) is
  pinned to the in-process solver to 1e-8. Definitive reproduction of theADCcode's
  own `ndadc3ip` main lines (matched DZP integrals) is the remaining follow-on.
- **decay-channel spectrum** (M6): the ported `classify` unit tests pin the
  Auger/ICD/ETMD routing, passive discounting, and site-folding exactly as
  ADCanalysis proved them; a committed stick-spectrum fixture
  (`testdata/h2o_dzp.spec.json`) is pinned to the in-process solve→classify to
  1e-8; a schema key-contract test guards the JSON tags; and ADCanalysis's
  `plotspec` renders the ADCgo fixture directly (plug-compatibility).

### Regenerate the reference (needs the `adcgo` conda env with pyscf)

```sh
/home/leia/miniconda3/envs/adcgo/bin/python scripts/gen_fcidump.py
```

### Run

```sh
# M0 sanity: reconstructed HF + MP2 energies
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump

# DIP-ADC(2): dication states as JSON (energies, pole strengths, leading 2h
# configs, and — with the sidecar — atom-resolved two-hole populations),
# one sector per point-group irrep (-sym all | none | <0-based index>)
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip \
    -mo testdata/h2o.mo.json -solver lanczos -spin both -sym all

# SIP: single-ionization (cation) states as JSON — ionization energies,
# spectroscopic factors, per-orbital one-hole overlaps. -order 2|3 (non-Dyson
# IP-ADC; order 2 is the reference's extended ADC(2)).
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -order 3 -sym all

# Decay-channel stick spectrum (M6): solve + classify in one run. DIP needs -mo
# (channels need atom-resolved populations). -init-atom picks the core-ionized
# site; -group NAME=col1,~col2 defines composite/passive sites; a bare -group
# opens an interactive dialogue. The JSON feeds ADCanalysis's plotspec unchanged.
go run ./cmd/adcgo -fcidump testdata/h2o_dzp.fcidump -dip -mo testdata/h2o_dzp.mo.json \
    -solver dense -sym all -spectrum -init-atom O
# passive-hydrogen water site: only Auger@wat survives, O holes credited in full
go run ./cmd/adcgo -fcidump testdata/h2o_dzp.fcidump -dip -mo testdata/h2o_dzp.mo.json \
    -solver dense -sym all -spectrum -group "wat=O,~H1,~H2" -init-atom wat

# Accelerated backends (build-tag gated; -backend selects at runtime).
go run -tags openblas ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip -sym all   # multicore CPU
HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  go run -tags hip ./cmd/adcgo -fcidump testdata/h2o.fcidump -dip -backend hip -sym 0 -spin singlet
go build -tags cuda ./...   # cuBLAS: compiles here, run on an NVIDIA host

go test ./...          # full validation (slow gates included)
go test -short ./...   # fast subset
HSA_OVERRIDE_GFX_VERSION=11.0.0 go test -tags hip ./...   # + GPU cross-backend gates
```

### Regenerate the M4 reference (matched DZP+diffuse integrals + fixture)

```sh
# 1. matched FCIDUMP + sidecar (asserts SCF == -76.0498071428 Ha)
/home/leia/miniconda3/envs/adcgo/bin/python scripts/gen_ref_fcidump.py
# 2. committed ADCgo fixture the validation test compares to adcdip*.out
go run ./cmd/adcgo -fcidump testdata/h2o_dzp.fcidump -dip -mo testdata/h2o_dzp.mo.json \
    -solver dense -spin both -sym all -out testdata/h2o_dzp.adcgo.json
```

### Regenerate the M5 SIP reference (pyscf oracle + fixture)

```sh
# pyscf IP-ADC(2)/(3) energies + spectroscopic factors on the same cc-pVDZ
# integrals ADCgo reads (writes testdata/h2o_sip.pyscf.json)
/home/leia/miniconda3/envs/adcgo/bin/python scripts/gen_sip_ref.py
# committed ADCgo SIP fixture the regen guard pins
go run ./cmd/adcgo -fcidump testdata/h2o.fcidump -sip -order 3 -sym all \
    -solver dense -out testdata/h2o_sip.adcgo.json
```

### Regenerate the M6 decay-channel spectrum fixture

```sh
# committed stick-spectrum fixture the regen guard pins to the in-process solve
go run ./cmd/adcgo -fcidump testdata/h2o_dzp.fcidump -dip -mo testdata/h2o_dzp.mo.json \
    -solver dense -sym all -spectrum -init-atom O -out testdata/h2o_dzp.spec.json
```

