# plots/

Figures generated from ADCgo output, and the spectrum JSONs they were drawn
from. Regenerate with the commands below rather than editing the artefacts.

## adcgo_vs_theadccode.{pdf,png} — ADCgo vs theADCcode verification

Gas-phase H2O DIP-ADC(2), DZP + diffuse (Dunning), active space 2–30, C2v.
Both codes read the *same* molecular-orbital integrals (`testdata/h2o_dzp.fcidump`,
dumped from the GAMESS-UK run in `examples/DIP_h2o/dump.in`, which reproduces the
reference calculation in `../ADCanalysis/examples/DIP_h2o`) and use equivalent
Lanczos settings: theADCcode's `iter 100` / `ps 0.5` map onto ADCgo's `-blocks 100`
/ `-ps-thresh 0.5`.

```sh
# ADCgo spectrum
./adcgo -fcidump testdata/h2o_dzp.fcidump -mo testdata/h2o_dzp.mo.json \
    -dip -order 2 -solver lanczos -spin both -sym all \
    -blocks 100 -ps-thresh 0.5 -spectrum -init-atom O \
    -molecule H2O -basis dzp -point-group C2v \
    -out plots/h2o_dip_adcgo.spec.json

# theADCcode spectrum, from its adcdip{1..4}.out
./adcanalysis -mode dip -in ../ADCanalysis/examples/DIP_h2o -init-atom O \
    -molecule H2O -basis dzp -point-group C2v \
    -out plots/h2o_dip_theadccode.spec.json </dev/null

# overlay: ADCgo solid, theADCcode dotted, one total curve each, common scale
./plotspec -mode spectrum -in plots/h2o_dip_adcgo.spec.json \
    -exp plots/h2o_dip_theadccode.spec.json \
    -total ADCgo -exp-total theADCcode -exp-scale 1 \
    -fwhm 1.5 -xrange 35-160 -width 8 -height 5 -dpi 600 \
    -out plots/adcgo_vs_theadccode.pdf
```

`-exp-scale 1` is essential: the default (0) rescales the overlay to match the
theory's tallest peak, which would hide any intensity disagreement — exactly what
this figure exists to test.

The deviations quoted in the thesis, and the LaTeX rows of its ten-lowest-poles
table, come from

```sh
scripts/analysis/compare_spectra.py plots/h2o_dip_adcgo.spec.json \
    plots/h2o_dip_theadccode.spec.json --table 10 --irreps a_1,a_2,b_1,b_2
```

Agreement over the 181 states the two runs have in common (both above the 0.5 %
pole-strength cut-off): across the 50 lowest, up to 89 eV, max |ΔE| = 3.8e-6 eV
and max |Δ pole strength| = 5.0e-3 percentage points, i.e. the precision at which
theADCcode prints them. Above 100 eV, among the weak satellites at the top of the
Lanczos pseudo-spectrum, the largest deviations are 0.37 eV and 1.1 percentage
points.

The figure is used as Fig. `codeverif` in `~/Documents/Thesis/skeleton.tex`
(installed as `ADCanalysis/Images/adcgo_vs_theadccode.pdf`).

## comparison.{pdf,png}, adcgo.pdf, theADCcode.pdf, *_spec.json

Earlier H2O DIP comparison, run with ADCgo's default `-ps-thresh 1` (so fewer
states than the reference); superseded by the pair above.
