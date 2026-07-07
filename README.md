# ADCgo

An exact, hardware-accelerated ADC(n) **ionization** solver in Go.

ADCgo computes the algebraic-diagrammatic-construction (ADC) secular problem for
electron removal exactly* — no reduced-scaling truncations — and reaches larger 
systems by acceleration (multicore OpenBLAS, GPU Lanczos via **hipBLAS** on AMD 
and **cuBLAS** on NVIDIA) rather than approximation. SCF + molecular integrals 
are delegated: ADCgo ingests a standard **FCIDUMP** (e.g. from pyscf).

See [`ADCgo_plan.md`](ADCgo_plan.md) for the full design and milestones.

M3 backends (build-tag gated; default is pure-Go):

- **gonum** (default) — pure-Go reference; with `-tags openblas` its BLAS/LAPACK
  engine is swapped for multicore OpenBLAS (same code path).
- **hip** (`-tags hip`) — hipBLAS/ROCm; built and validated here on a Radeon 890M
  (gfx1150), deployment target AMD Instinct.
- **cuda** (`-tags cuda`) — cuBLAS; a structural twin of the hip shim, compiled
  against CUDA 13.x here and run on NVIDIA hardware.

The pipeline is:

1. `scripts/gen_fcidump.py` runs RHF + MP2 on H₂O/cc-pVDZ with **pyscf** in C₂ᵥ
   symmetry, writing `testdata/h2o.fcidump` (with 1-based GAMESS-UK `ORBSYM`), the
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

