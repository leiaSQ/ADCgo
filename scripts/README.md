# scripts

| Folder | Contents |
|---|---|
| `fcidump/` | Integral generation: `dump_fcidump.py` (sectioned input, see `adcgo_input.py`), the GAMESS-UK readers, orbital selection, the localized-orbital basis (`orbitals.py`) and KBJ ghost bases (`kbj.py`), `threads.py` (import before numpy on the login node) |
| `fixtures/` | Regenerate the committed test fixtures in `testdata/` |
| `codegen/` | Code generators: adcgen ISR elements (`generate_adc.py`, `isrgen.sbatch`, see `internal/adc/isrgen/README.md`), the ADC(4) spin tables (`gen_coeff4.py`) |
| `helix/` | bwForCluster Helix: `HELIX.md`, the dump-and-run driver (`adcgo_run.sh`, `runADCgo*`), the CUDA build, workspaces, production and benchmark jobs |
| `gpu/` | GPU diagnostics and parity jobs (CUDA smoke, compute-sanitizer, multi-GPU parity, int32 addressing, timing A/B) |
| `analysis/` | Spectrum comparison and plotting |

Scripts find the repository root from their own location, so run them from any directory:

```sh
python scripts/fcidump/dump_fcidump.py --input examples/DIP_h2o/dump.in
sbatch scripts/helix/runADCgo examples/DIP_h2o/dump.in
```
