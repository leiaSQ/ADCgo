# ADC(2,2)f verification against Kolorenč & Averbukh

- **Method:** `adcgo -sip -order 22 -adc22 f -sigma off`, aug-cc-pVDZ, frozen 1s cores, C2v.
- **References:**
  - Paper: Kolorenč & Averbukh, J. Chem. Phys. **152**, 214107 (2020), Table IV.
  - Code: Prema's 1h-ADC(2,2) variant F (rev. 3391b433251, Mar 2026), H2O run in `examples/ADC22.H2O.tar.bz2`.
- **Result:** all main lines match the paper to its printed precision (0.01 eV). Every oracle state, main and satellite, matches the code to ≤ 0.03 meV.
- Requires the 2h1p hole-order phase fix (`holeOrderPhase`, `internal/adc/sip/spinadapt.go`). Without it, C2v main lines were off by up to 0.27 eV, while C1 was already correct.

## Paper, Table IV (main IPs, eV)

| Molecule | Vacancy | Paper | ADCgo | Δ |
|---|---|---|---|---|
| HF | 1π  | 15.73 | 15.731 | +0.001 |
| HF | 3σ  | 19.72 | 19.723 | +0.003 |
| N2 | 3σg | 15.78 | 15.777 | −0.003 |
| N2 | 1πu | 17.17 | 17.177 | +0.007 |
| N2 | 2σu | 18.76 | 18.762 | +0.002 |
| CO | 5σ  | 14.02 | 14.026 | +0.006 |
| CO | 1π  | 16.82 | 16.819 | −0.001 |
| CO | 4σ  | 19.42 | 19.420 | 0.000 |

- **Geometries:** experimental bond lengths, HF 0.9168 Å, N2 1.0977 Å and CO 1.1283 Å.
- **Frozen cores:** HF 1, N2 and CO 2.
- **Runs:** `examples/ADC22_TableIV/{hf,n2,co}/adc22f_fixed.json`.
- Not run: F2 and C2H4.

## Prema's code, H2O

- **Inputs:**
  - Geometry from the oracle's `seward.inp`: O (0, 0, 0.1173), H (0, ±0.7572, −0.4692) Å.
  - E(SCF) = −76.0413935200 Eh, the same as MOLCAS.
  - MP2 = −0.2193897 Eh, the same as the oracle's E2.
- **Irrep map** (MOLCAS → FCIDUMP sym): a1 1→0, b1 2→2, b2 3→3, a2 4→1.

### Main lines

| Irrep | Oracle E (eV) | ADCgo E (eV) | ΔE | Oracle PS | ADCgo PS |
|---|---|---|---|---|---|
| 1b1 | 12.3412 | 12.3412 | +0.004 meV | 0.9124 | 0.8976 |
| 3a1 | 14.5931 | 14.5931 | +0.005 meV | 0.9151 | 0.8994 |
| 1b2 | 18.7718 | 18.7718 | +0.007 meV | 0.9302 | 0.9135 |

### Every oracle state

| Irrep | Oracle states | Range (eV) | \|ΔE\| median | \|ΔE\| max |
|---|---|---|---|---|
| a1 | 114 | 14.6–74.6 | 0.02 meV | 0.03 meV |
| b1 | 88  | 12.3–73.8 | 0.02 meV | 0.03 meV |
| b2 | 96  | 18.8–76.6 | 0.02 meV | 0.03 meV |
| a2 | 73  | 28.0–70.4 | < 0.01 meV | < 0.01 meV |

- **Matching:** each oracle state is paired with the nearest ADCgo root in the same irrep.
- **Runs:** dense per irrep, `examples/ADC22_H2O/fixed_sym{0..3}.json`.
  - a2 has no 1h space, so adcgo skips it; it came from a dense throwaway build.

## Known difference: pole strengths

- The oracle's PS is larger by a factor that depends only on the 1h orbital: ours/oracle = 0.9838 in every b1 state and 0.982 in every b2 state.
- ADCgo reports the squared 1h eigenvector weight. The oracle's PS + 2h1p weight is 0.993, not 1.
- So the oracle uses a different spectroscopic-factor definition. Energies, and hence the matrix, agree.
