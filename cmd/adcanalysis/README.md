# ADCanalysis — theADCcode output to spectrum JSON

The analysis half of ADCgo: `ADCanalysis` turns a finished ADC run into a
channel-resolved **stick spectrum as JSON**, and `plotspec` turns that JSON into
a **figure**. They are two commands rather than one because the expensive part
(parsing, classifying) is done once and the cheap part (broadening, colours,
axes, layout) is re-tuned many times without touching the run output.

```
 theADCcode / ADCgo output          spectrum JSON              figure
 ─────────────────────────   ──►   ─────────────   ──►   ────────────────
   adcdip*.out  (DIP)              meta            plotspec   .png / .svg / .pdf
   ADC.out      (SIP)              channels[]
                                   lines[]
        ADCanalysis
```

Everything downstream of the JSON is presentation. Broadening width, spin
weighting, normalisation, colour and mode all live in `plotspec`, so a
re-render never re-reads a 30 000-line output file.

- [Quick start](#quick-start)
- [ADCanalysis](#adcanalysis)
  - [Modes and inputs](#modes-and-inputs)
  - [Decay channels](#decay-channels-dip-mode)
  - [Sites and `-group`](#sites-and--group)
  - [Thresholds](#thresholds)
  - [Metadata](#metadata)
  - [Flag reference](#adcanalysis-flag-reference)
- [Gotchas](#gotchas)

The spectrum JSON schema and every `plotspec` flag: [`cmd/plotspec/README.md`](../plotspec/README.md).
`adcgo -spectrum` / `-convert` write the same JSON straight from an ADCgo run; ADCanalysis is
for theADCcode output files.

## Quick start

```sh
go build -o bin/ADCanalysis ./cmd/adcanalysis
go build -o bin/plotspec    ./cmd/plotspec

# Double ionization: classify decay channels against a core hole on O
# (testdata/reference holds theADCcode's adcdip1..4.out for water).
bin/ADCanalysis -in testdata/reference -init-atom O -out h2o.dip.json
bin/plotspec -in h2o.dip.json -out h2o.png -fwhm 1.2

# Single ionization: per-orbital spectrum, no decay-site machinery.
bin/ADCanalysis -mode sip -in internal/adcanalysis/parse/testdata -out h2o.sip.json
bin/plotspec -in h2o.sip.json -out h2o.sip.png -total "SIP" -xrange 8-40
```

---

# ADCanalysis

## Modes and inputs

`-in` is a **directory**, not a file. What is read from it depends on `-mode`:

| `-mode` | Reads | Optional metadata | Channels are |
|---|---|---|---|
| `dip` (default) | every `adcdip*.out` in the directory, sorted | `dip.in` | decay channels (Auger / ICD / ETMD) |
| `sip` | `ADC.out` | `scf_adc.in` | molecular orbitals |

A DIP run writes one file per irrep, so all of them are globbed and merged into
one spectrum; the irrep each state came from is kept on every line. A SIP run
writes a single `ADC.out` holding every symmetry block, so there is one file to
read. Both are theADCcode output; ADCgo's own `-spectrum` JSON already matches
the schema below and goes straight to `plotspec`.

> **The `ADC.out` has to be the ADC section alone.** A combined pipeline log
> (SCF + transformation + ADC concatenated) carries the SCF program's own
> orbital listing, which shadows theADCcode's `m.o. sym energy` table and makes
> the parse pick up the wrong orbitals. Truncate to the ADC section first.

`-mode sip` bypasses the decay-site and initial-atom machinery entirely:
`-group`, `-init-atom`, `-init-orbital`, `-min-weight`, `-min-fraction`,
`-include-zero` and `-st-ratio` are all DIP-only and are ignored.

## Decay channels (`dip` mode)

A core hole on site A decays to a dicationic state with two valence holes.
**Where those two holes sit relative to A is the channel:**

| Both holes | Channel | Label |
|---|---|---|
| on A | local Auger | `Auger@A` |
| one on A, one on B | ICD | `ICD:A->B` |
| both on one site B ≠ A | ETMD(2) | `ETMD(2)` |
| on two sites B, C ≠ A | ETMD(3) | `ETMD(3)` |

The `&popana` table already decomposes each final state into one-site and
two-site atomic weights, so classification is a *routing* of named population
columns onto these labels — no eigenvector work. That is also why the
ETMD(2)/ETMD(3) split only becomes meaningful once you have said what a "site"
is: whether two holes sit on one unit or two depends entirely on the grouping.

## Sites and `-group`

By default **every population column is its own site**, which for a water
cluster means every atom. That is rarely what you want: a hole on a water's O
and a hole on that same water's H are not two sites, they are one water.

`-group NAME=col1,col2,...` folds columns into one decay unit. It is repeatable,
and the declaration order fixes the canonical channel ordering in the output.

```sh
# Each atom its own site (the default): O, H1, H2 are three decay units.
ADCanalysis -in examples/h2o -init-atom O

# One water as a single unit named "wat".
ADCanalysis -in examples/h2o -group "wat=O,H1,H2" -init-atom wat
```

Folding has a real physical consequence: two-site weight *between two columns of
the same site* becomes one-site weight on that site. Two holes on one water is
then an ETMD(2) candidate rather than an ICD between O and H.

### Passive columns (`~`)

Prefixing a member with `~` makes it **passive**: it belongs to the site, but
holes landing on it are discounted rather than counted toward the site's weight.

```sh
# Treat both H as passive: only the population actually on O survives.
ADCanalysis -in examples/h2o -group "wat=O,~H1,~H2" -init-atom wat
```

Use it when a column carries population you do not want to attribute — a
basis-set artefact, or a hydrogen whose Mulliken-like share is an accounting
convention rather than a decay site.

### Interactive selection

With neither `-group` nor `-init-atom` given **and stdin a terminal**,
ADCanalysis lists the population columns it found and prompts for the grouping
and the initial site. Prompts go to **stderr**, so `-out -` still writes clean
JSON to stdout. Flags always win over the prompt; passing either flag
non-interactively suppresses it. In a script, pass both — a pipeline with no
terminal gets the `-init-atom` default (`O`) rather than a prompt.

## Thresholds

Three flags control what survives into the JSON. They compose: a channel must
clear both thresholds to be emitted.

| Flag | Default | Drops a channel when |
|---|---|---|
| `-min-weight W` | `0` | its absolute weight `<= W` |
| `-min-fraction F` | `0` | its weight is below fraction `F` of that **state's** total 2h population |
| `-include-zero` | off | *(inverse)* emit the full canonical channel set per state, even at zero |

At the defaults, only exactly-zero and rounding-noise channels are dropped —
which is what you want, since a state that cannot decay into a channel should
not put a zero-height stick in the plot's legend.

`-min-fraction` is the one to reach for when a spectrum is cluttered: it is
state-relative, so it prunes negligible channels from strong and weak states
alike, where an absolute `-min-weight` would prune a weak state entirely.

`-include-zero` goes the other way and is for **tables, not plots**: it
guarantees every state has an entry for every channel, so downstream code can
index without checking for absence.

## Metadata

`-molecule`, `-basis` and `-point-group` are recorded in `meta` and used by
`plotspec` for titles. `-molecule` and `-basis` are read automatically from
`dip.in` (DIP) or `scf_adc.in` (SIP) when not given; the flags win when set.
`-point-group` is never inferred — pass it if you want it.

`-init-orbital` (e.g. `"1s"`) is **metadata only**. It records which core
orbital was ionized for the reader's benefit; it changes no classification.

`-st-ratio` (default `3`) is likewise **recorded, never applied**. Intensities
in the JSON are always unweighted; the singlet:triplet weighting happens at plot
time under `plotspec -spin-weight`, which reads this number out of `meta`.
Keeping it out of the stored intensities is what lets you re-render with and
without the weighting from one JSON.

## ADCanalysis flag reference

| Flag | Default | Mode | Meaning |
|---|---|---|---|
| `-mode dip\|sip` | `dip` | — | double ionization (`adcdip*.out`) or single ionization (`ADC.out`) |
| `-in DIR` | — | both | **required.** Input directory |
| `-out PATH` | `spec.json` | both | output JSON, or `-` for stdout |
| `-pretty` | `true` | both | indent the JSON; `-pretty=false` for a compact file |
| `-molecule S` | from `dip.in` | both | molecule label for `meta` |
| `-basis S` | from `dip.in` | both | basis label for `meta` |
| `-point-group S` | — | both | point-group label for `meta`, e.g. `C2v` |
| `-group NAME=a,b` | each column its own site | dip | decay-site grouping, repeatable; `~col` makes a member passive |
| `-init-atom NAME` | `O` | dip | initial core-ionized site; prompted on a terminal if omitted |
| `-init-orbital S` | — | dip | initial orbital label, metadata only |
| `-min-weight F` | `0` | dip | drop channels with weight `<=` this |
| `-min-fraction F` | `0` | dip | drop channels below this fraction of a state's total 2h population |
| `-include-zero` | off | dip | emit the full canonical channel set per state, even at zero |
| `-st-ratio F` | `3` | dip | singlet:triplet ratio recorded in `meta` for the plotting layer |

## Gotchas

**`-in` is a directory.** `ADCanalysis -in` takes a directory of theADCcode outputs;
`plotspec -in` takes a file.

**A truncated `ADC.out` is required for `-mode sip`.** Feed it a full pipeline
log and the SCF section's orbital listing shadows theADCcode's MO table.

**Interactive prompts need a terminal.** In a script with neither `-group` nor
`-init-atom`, you silently get the `-init-atom` default (`O`) and the identity
grouping — every column its own site. Pass both explicitly in a pipeline.
