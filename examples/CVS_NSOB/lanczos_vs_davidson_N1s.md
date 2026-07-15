# CVS-ADC(4) N 1s of nitrosobenzene: block-Lanczos (ADCgo) vs. Davidson (legacy)

## Summary

The ADCgo block-Lanczos spectrum and the legacy Davidson spectrum are the **same
calculation** — CVS-ADC(4), N 1s core hole, matrix dimension **1,753,921**, same
molecule/basis/active space. The only variable is the eigensolver. At `-blocks 200`
the ADCgo peaks are noticeably shifted from the Davidson roots, and the pole strength
is redistributed. **This is a solver-convergence artifact, not a difference in the
physics or the matrix.**

- Legacy: `adc4_diag.x`, Davidson, `nroots=20`, `convthr=1.0E-03`
  (`/mnt/worka/alexk/NSOB/N1s.data/dav.NSOB.N1s.out`)
- ADCgo: `-sip -order 4 -core 0 -solver lanczos -matfree auto -blocks 200`
  (`examples/CVS_NSOB/nsob_n1s.sym1.json`)

## Side-by-side (a′, irrep 1)

| Legacy Davidson (eV, ps%) | ADCgo Lanczos (eV, ps%) | Δ energy |
|---|---|---|
| 408.393, 39.3 | 408.603, 46.4 | +0.21 |
| 409.700,  1.9 | 410.708,  1.0 | +1.01 |
| 411.442, 16.5 | 412.402, 10.9 | +0.96 |
| 413.742,  0.8 | 417.464,  1.3 | (no clean match) |
| 418.945,  2.7 | 422.97 / 423.44 … | (no clean match) |

Davidson targeted only 20 roots, all in 408–419 eV. ADCgo's poles above ~419 eV
(422, 423, 444 … 457 eV) are coarse Lanczos pseudo-poles with no Davidson counterpart.

**Key pattern:** the leading pole is nearly exact (**+0.21 eV**); the shift grows to
**~1 eV** by the second intense pole and the correspondence breaks down entirely
higher up. The error increases monotonically as you move into the interior of the
spectrum.

## Why the peaks shift

1. **Lanczos matches spectral moments, not individual eigenvalues.** 200 blocks
   reproduce only the leading moments of the spectral density of a 1.75M-dimensional
   matrix. That pins the near-edge dominant pole and the gross band shape but cannot
   resolve individual interior poles. Each unconverged Ritz value sits at the
   **pole-strength-weighted centroid of a cluster of true eigenstates**, not at any one
   of them — so the real 411.44 (16.5%) pole and its neighbors collapse into a single
   pseudo-pole at ~412.4 carrying the merged weight (16.5 → 10.9%).

2. **Convergence is edge-in and start-vector-weighted.** The dominant pole (46%)
   dominates the Lanczos start vector, so line 1 converges fastest (0.2 eV). States
   deeper in the interior converge progressively slower — exactly the observed gradient.

3. **Davidson root-targets.** It iterates 20 *specific* eigenpairs to error → 0
   (threshold 1e-3 eV), independent of matrix dimension. Its positions are effectively
   exact; the Lanczos positions are approximate. The two methods are not solving the
   same problem, even at "200."

4. **Pole strength redistributes** for the same reason: a Lanczos pseudo-state spans
   several true eigenstates, so its intensity is the aggregate weight of that Krylov
   region. The top-two weight is similar in total (ADCgo 57%, Davidson 56%) but split
   differently — the intense edge pole absorbs weight from its unconverged neighbors.
   Note `residue` is always 0 here, so all weight lives in the reported pole strengths;
   there is no hidden residual channel.

## Suggested fix (legacy-side / cross-check)

The block count needed for sub-eV interior positions scales with spectral density and
how deep you probe — not an absolute number. `-blocks 200` is adequate only for the
main edge and overall envelope.

1. **Convergence study:** rerun ADCgo N1s at `-blocks` 500, 1000, 2000 and confirm the
   two intense poles (408.6, 412.4) drift monotonically toward the Davidson positions
   (408.39, 411.44). The 200 → 800 delta quantifies the residual error at 200.
2. **If only the lowest ~20 roots are wanted**, use ADCgo's root-targeting Davidson
   solver — `-solver davidson -nroots 20 -convthr 1e-3` — which matches the legacy
   positions directly at a fraction of the Krylov size (it converges each requested
   eigenpair to a residual threshold, exactly as `adc4_diag.x` does). Block-Lanczos is
   the wrong tool when you want a handful of converged interior eigenvalues rather than a
   broad spectrum. The Davidson driver (`internal/adc/lanczos/davidson.go`) mirrors
   theADCcode's block Davidson–Liu (diagonal `(θ−D)⁻¹` preconditioner, thick restart at
   `-maxdavsp`), with two robustness upgrades over the reference: it seeds on the smallest
   diagonal entries (so no low root is stranded in an unspanned symmetry block) and drives
   a few buffer roots beyond `-nroots` (so the requested roots do not swap out at the
   window boundary).
3. When comparing against `dav.NSOB.N1s.out`, only compare within the 408–419 eV window
   Davidson actually converged; discard ADCgo pseudo-poles above ~419 eV.

## Reproduce

```
# ADCgo
scripts/adcgo_run.sh examples/CVS_NSOB/n1s.in ./adcgo
# legacy (where the legacy codebase is available)
#   adc4_diag.x with nroots=20, convthr=1.0E-03  -> dav.NSOB.N1s.out
```
