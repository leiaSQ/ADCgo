# plotspec — rendering ADCgo spectra

`adcgo -spectrum` (or `-bare`, or `-convert`) writes a spectrum JSON; `plotspec` turns it into a
figure. Build it with `go build -o plotspec ./cmd/plotspec`.

## The spectrum JSON

The contract between `adcgo` and `plotspec`. Both `-dip` and `-sip` runs produce the
same shape, so `plotspec` needs no special case (`meta.kind` switches the axis
and title wording).

```json
{
  "meta": {
    "kind": "sip",
    "molecule": "H2O",
    "basis": "DZP",
    "point_group": "C1",
    "initial_ionization": { "atom": "O", "orbital": "1s" },
    "irreps": ["1"],
    "energy_unit": "eV",
    "singlet_triplet_ratio": 3,
    "source_files": ["ADC.out"]
  },
  "channels": ["MO 24 (sym 1)", "MO 25 (sym 1)"],
  "lines": [
    {
      "energy": 9.037854,
      "intensity": 0.0816502,
      "channel": "MO 25 (sym 1)",
      "spin": 2,
      "irrep": 1,
      "state_ref": "irrep1/s2/#2",
      "ps_percent": 8.26
    }
  ]
}
```

- **`channels`** is the canonical order; `plotspec` plots in this order so a
  channel keeps its colour between figures. Channels found only in `lines` are
  appended, sorted.
- **`lines`** is one stick per *(state, channel)* pair — a state with weight in
  three channels contributes three lines at the same `energy`.
- **`intensity`** is the channel's share of that state. For SIP it is the
  orbital's squared one-hole amplitude, so a state's intensities sum to its pole
  strength (`ps_percent`/100). For DIP it is the channel's share of the state's
  two-hole population.
- **`state_ref`** identifies the state across channels. `plotspec -total` uses
  it to sum a state's channels back into one stick.
- Intensities are **never** pre-weighted by `singlet_triplet_ratio`.

---

## Modes

`plotspec` reads that JSON, Gaussian-broadens each channel's sticks onto a
shared energy grid, and draws one curve per channel. The output format follows
the `-out` extension: `.png` (raster, honours `-dpi`), `.svg` or `.pdf` (vector,
`-dpi` ignored). Use a vector format for anything going into a paper — the
structure insets and curves stay resolution-independent.

| `-mode` | Input | Draws |
|---|---|---|
| `spectrum` (default) | one spectrum JSON (`-in`) | one curve per decay channel (DIP) or per orbital (SIP) |
| `ees` | a SIP + a DIP JSON (`-sip`, `-dip`) | electron-emission spectrum σ(ε) |
| `ees-diff` | SIP + DIP JSON for a system and a reference (`-ref-sip`, `-ref-dip`) | one EES minus a scaled reference EES, e.g. dimer − 2 × counterpoise monomer ([workflow](../../scripts/README.md#interaction-effects-dimer-minus-2--monomer)) |
| `panel` | SIP + DIP JSON | 3-panel composite: SIP sticks, DIP sticks, EES curve |
| `stack` | several spectrum JSONs | energy-offset waterfall, one envelope per conformer |

## Shared rendering flags

### Broadening — `-fwhm`, `-points`

`-fwhm` is the Gaussian **full width at half maximum in eV** (σ = FWHM /
2√(2 ln 2)). It is the single most consequential knob: too small and the figure
is a comb of individual roots, too large and real structure is washed out.
Match it to the experiment you are comparing against, or to the physical
lifetime broadening — not to what looks tidiest.

`-points` (default `1000`) is the grid resolution. Raise it when `-fwhm` is
small relative to the plotted window, or narrow peaks will be sampled unevenly
and their apparent heights will jitter. A useful floor is ~10 grid points per
FWHM: over a 100 eV window at `-fwhm 0.5`, that is 2000 points.

### Energy window — `-xrange`, `-pad`

Without `-xrange`, the window is the full range of the sticks (theory *and*
`-exp` overlay) padded by `-pad` eV on each side. `-xrange LO-HI` fixes it.

`-xrange` does more than crop: **normalisation is computed inside the plotted
window only.** Zooming into a weak region rescales it to fill the axis rather
than leaving it flat against a tall peak outside the view. That is usually what
you want, and it is why two `-xrange` crops of the same spectrum are not
directly comparable in height.

### Normalisation — `-absolute`

By default the tallest displayed feature is 1 (relative intensity). `-absolute`
plots the raw broadened intensity instead. Use `-absolute` when the absolute
scale carries meaning — comparing two spectra rendered with the same `-fwhm`,
or checking that a channel really is negligible rather than merely normalised
into invisibility.

Sticks and broadened curves are normalised **independently**, because they carry
different units (a stick is a weight, a curve is a weight density). This is what
makes `-stick-height` well-defined.

### Sticks — `-stick`, `-overlay-broadened`, `-stick-height`

- `-stick` draws one bar per state instead of the broadened curve — the honest
  view of what was actually computed, with no broadening choice baked in.
- `-overlay-broadened` draws the curves **on top of** the sticks, so you see
  both the roots and the envelope they produce.
- `-stick-height F` scales the sticks (curves stay at 1). Use `F < 1` so the
  bars sit under the envelope instead of poking through it.

All three work in `spectrum` and `panel` modes.

### Reference overlay — `-exp`, `-exp-scale`, `-exp-label`, `-exp-total`

`-exp FILE` overlays a second spectrum in the same schema as **dotted** lines,
drawn in the same per-channel colours as the theory, so each measured channel
sits directly on its calculated counterpart.

`-exp-scale` defaults to `0` = auto, matching the overlay's tallest peak to the
theory's. A single global factor is used, so relative heights *within* the
overlay are preserved. Set it explicitly when the overlay is already on a
calibrated scale.

`-exp-label` (default `exp`) is the bracketed legend tag. Set it to
`theADCcode` when the overlay is a reference *calculation* rather than a
measurement, so the legend does not claim experimental data.

> **`-spin-weight` is never applied to `-exp`.** Measured rates already embed the
> true singlet:triplet ratio; weighting them again would double-count it.

### Collapsing channels — `-total`, `-exp-total`

`-total NAME` sums every channel of `-in` into one series called `NAME` — the
bare total spectrum with the decomposition dropped. This is what a
code-against-code comparison wants: without it the same comparison is repeated
once per channel, in colours the `-exp` overlay has to share.

`-exp-total NAME` does the same to the overlay; the name is then used verbatim
in the legend, without the `-exp-label` tag.

### Spin weighting — `-spin-weight`

Scales singlet sticks (`spin == 1`) by `meta.singlet_triplet_ratio`; triplets
stay at 1. Theory only. The ratio comes from the JSON (`adcgo -st-ratio`),
so switching it on and off is a re-render, not a re-analysis.

### Figure size and colour — `-width`, `-height`, `-dpi`, `-colorblind`

`-width`/`-height` are **inches** (defaults 8 × 5). For a two-column journal
figure, 3.4 in wide is a single column and ~7 in is full width; set the size at
render time rather than scaling the file afterwards, or the fonts scale with it.

`-colorblind` swaps in the Okabe–Ito palette and applies to **every** mode.

## `-mode spectrum`

The default. One broadened curve per channel from a single JSON. Axis labels and
title switch on `meta.kind`: double- vs single-ionization energy, decay channels
vs orbitals.

```sh
plotspec -in spec.json -out spectrum.png -fwhm 1.2
plotspec -in spec.json -exp h2o_auger_experimental.json -out cmp.png
plotspec -in sip.json  -out sip.png -total "total SIP" -xrange 8-40
```

## `-mode ees`

The electron-emission spectrum: the kinetic-energy distribution of the secondary
electron, in the equilibrium-geometry envelope-convolution approximation.

An intermediate singly-ionized state at `E_in` decays to a final dicationic state
at `E_fin`, emitting an electron of kinetic energy ε = E_in − E_fin (open only
for ε > 0):

```
σ(ε) = ∫ dE · S_in(E) · S_fin_num(E−ε) / N(E),    N(E) = ∫₀^E S_fin_tot(E′) dE′
```

`S_in` is the broadened SIP envelope, `S_fin` the broadened DIP envelope, and
`N(E)` normalises to the *open* final-state population — the cumulative **total**
DIP envelope, so always-open channels are not over-weighted.

`-fin-channels` restricts the **numerator** only (comma-separated names or
prefixes, e.g. `ICD`). `N(E)` always uses every channel. That asymmetry is the
point: it gives a channel's partial electron spectrum still correctly weighted by
the full branching, which a plain channel filter would not.

Three FWHM flags let the two envelopes be broadened differently, because they
are physically different widths — the SIP envelope carries the intermediate
state's lifetime, the DIP envelope the final state's:

| Flag | At its default `0`, falls back to |
|---|---|
| `-fwhm-sip` | `-fwhm` |
| `-fwhm-dip` | `-fwhm` |
| `-fwhm-ees` | `-fwhm-sip` / `-fwhm-dip` |

`-fwhm-ees` sets the width used for **both** envelopes inside the convolution,
decoupling it from the widths used to draw the panel (a)/(b) overlays. Use it
when the EES curve should be smoothed more than the stick panels above it.

```sh
plotspec -mode ees -sip sip.json -dip dip.json -out ees.png -fwhm 1.0
plotspec -mode ees -sip sip.json -dip dip.json -out icd.png -fin-channels ICD
```

## `-mode panel`

The composite figure: **(a)** SIP sticks, **(b)** DIP sticks, **(c)** the
electron-emission curve. Panels (a) and (b) are full-width, stacked, sharing one
x-axis with no gap (only (b) is labelled); panel (c) sits below with a gap,
narrower and horizontally inset, on its own independent axis.

- `-xrange-ab` (default `0-40`) — shared window for (a) and (b)
- `-xrange-c` (default `0-14`) — window for (c), the kinetic-energy axis
- `-overlay-broadened` here draws **one total envelope** over the (a)/(b) sticks,
  not one curve per channel

### `-sip-group`

Groups panel (a)'s orbitals, which would otherwise be one series per MO.

```sh
# Explicit: MO numbers, or symN for a whole symmetry. Repeatable.
plotspec -mode panel -sip sip.json -dip dip.json \
    -sip-group=core=1,2 -sip-group=valence=sym2 -out panel.png

# With a fixed colour for the group.
plotspec ... -sip-group='core#e41a1c=1,2'

# Bare flag: opens an interactive dialogue listing every MO.
plotspec -mode panel -sip sip.json -dip dip.json -sip-group
```

## `-mode stack`

An energy-offset waterfall across the conformers of one structure search. Each
conformer contributes a single broadened **total** envelope, and the *baseline*
of that envelope sits at its relative electronic energy — so vertical position
reads directly as "how much less stable is this basin". It is the stacked
convention of an NMR/IR series with relative energy in place of time.

Twenty hydrogen-bonded minima cannot each have a three-panel figure, and a table
of them is not read. This is the figure that puts a whole search on one axis
pair.

### Selecting traces

Either explicitly, or from a run manifest, or both:

```sh
# Explicit: LABEL=PATH[@OFFSET], offset in -offset-unit. Repeatable.
plotspec -mode stack -stack "global min=a.json@0" -stack "alt=b.json@2.1" -out w.pdf

# From a manifest: one trace per run whose spectrum exists.
plotspec -mode stack \
    -stack-manifest sip_manifest.json -stack-dir specs -stack-group 1 \
    -xrange 8-40 -fwhm 0.8 -stack-dedup 0.02 -out conformers_1W.pdf
```

The manifest is a JSON object mapping a run name to its metadata:

```json
{
  "W1_siteA": { "n_waters": 1, "relaxed_energy": -152.104311207 },
  "W1_siteB": { "n_waters": 1, "relaxed_energy": -152.101717917 }
}
```

| Flag | Default | Meaning |
|---|---|---|
| `-stack-energy-key` | `relaxed_energy` | manifest field holding the total energy |
| `-manifest-unit` | `hartree` | unit of that field (`hartree\|ev\|kcal\|kj`) |
| `-offset-unit` | `kcal` | unit the offsets are *plotted* in |
| `-stack-group-key` | `n_waters` | field `-stack-group` selects on |
| `-stack-group` | — | plot only runs whose key equals this (one hydration level, say) |
| `-stack-dir` | manifest's directory | where the spectrum JSONs live |
| `-stack-suffix` | `.sip.json` | appended to each run name to find its spectrum |

Energies are re-referenced to **the most stable run that actually has a
spectrum**, and that run is named on stderr. This matters: if a more stable
basin is still running, the whole ladder shifts once it lands. A run with no
spectrum yet is reported and omitted rather than failing the figure.

### Decluttering

A structure search reaches the same minimum from several starting guesses, so
the raw basin list holds duplicates that draw exactly on top of one another and
overstate how many distinct basins were found.

- **`-stack-dedup TOL`** folds traces whose normalised curves agree everywhere
  to within `TOL` into their most stable member, which keeps the rest as aliases
  and is labelled `name (xN)`. Default `0` — every basin is kept, so nothing
  disappears unless you ask for it.
- **`-stack-min-sep`** (default `0.12`, a fraction of the mean spacing) nudges
  isoenergetic traces that survive dedup apart. The y ticks stay on the **true**
  energies; the nudge is a drawing device only.
- **`-stack-trace-scale`** (default `2`) sets trace height in units of the mean
  baseline spacing. `>1` lets neighbours overlap, ridgeline-style. The least
  stable trace is drawn first, so the global minimum is painted last and is
  never occluded.
- **`-stack-norm each`** (default) normalises every trace to its own maximum,
  comparing *shape*; **`common`** puts them on one scale, which also shows
  relative yield.

Traces are coloured by a single sequential ramp keyed to relative energy (dark =
most stable), not by the categorical per-channel palette: they differ by a
magnitude, not an identity. Each trace is named on the y axis, with its name on
a second line directly under its relative energy.

### Structure insets

Each trace carries a 2D ball-and-stick depiction of its conformer, generated
from the geometry the calculation ran on, so it cannot disagree with it. A name
like `O2-N1H` only means something to a reader holding the site-numbering
convention in their head.

Geometry is looked for beside the spectrum, with `-stack-suffix` swapped for
`-stack-geom-suffix` (default `.zmat`) — a GAMESS-UK Z-matrix or an XYZ. A run
without a readable geometry simply goes without a picture.

The molecule is flattened onto its own best-fit plane (the two leading principal
axes), which is the undistorted view for near-planar systems and needs no
per-conformer tuning; solvent fragments are then oriented to the upper right so
the same motif is drawn the same way up in every panel. Atoms are CPK balls — O
red, N blue, C dark grey, H light grey, radius from the covalent radius — joined
by sticks in the trace's own colour, with hydrogen bonds dashed. Bonds come from
covalent radii (×1.3) and contacts from the 1.6–2.2 Å window, both HeidelBIRDS'
conventions, so the topology drawn is the topology the structure search used.

| Flag | Default | Meaning |
|---|---|---|
| `-stack-inset` | `0.2` | inset width as a fraction of the energy axis; `0` draws none |
| `-stack-inset-height` | `0.5` | inset height, in units of the trace amplitude |
| `-stack-inset-lift` | `0.05` | height of the inset's bottom edge above its trace's baseline |
| `-stack-geom-suffix` | `.zmat` | geometry filename suffix; empty disables insets |

The default lift keeps each structure low, floating just over its own curve
rather than up near the trace above it. Raise it if a spectrum has real
structure at the high-energy end.

## plotspec flag reference

### All modes

| Flag | Default | Meaning |
|---|---|---|
| `-mode S` | `spectrum` | `spectrum` \| `ees` \| `ees-diff` \| `panel` \| `stack` |
| `-in PATH` | `spec.json` | input spectrum JSON |
| `-out PATH` | `spectrum.png` | output image; format from the extension |
| `-fwhm F` | `1` | Gaussian FWHM for broadening, eV |
| `-points N` | `1000` | energy-grid resolution |
| `-pad F` | `5` | eV of padding each side of the data range |
| `-xrange LO-HI` | data range ± pad | fix the x window; also fixes normalisation to it |
| `-absolute` | off | raw broadened intensity instead of tallest peak = 1 |
| `-spin-weight` | off | weight singlet sticks by `meta.singlet_triplet_ratio` |
| `-stick` | off | bars per state instead of curves |
| `-overlay-broadened` | off | draw curves on top of the sticks |
| `-stick-height F` | `1` | scale the normalised stick heights |
| `-colorblind` | off | Okabe–Ito palette |
| `-width F` | `8` | figure width, inches |
| `-height F` | `5` | figure height, inches |
| `-dpi N` | `192` | raster resolution; ignored for SVG/PDF |

### `spectrum`

| Flag | Default | Meaning |
|---|---|---|
| `-total NAME` | — | collapse every channel of `-in` into one series |
| `-exp PATH` | — | reference/experimental overlay, dotted, same schema |
| `-exp-scale F` | `0` (auto) | scale factor for the overlay |
| `-exp-label S` | `exp` | bracketed legend tag for the overlay |
| `-exp-total NAME` | — | collapse the overlay into one series, named verbatim |

### `ees` / `panel`

| Flag | Default | Meaning |
|---|---|---|
| `-sip PATH` | — | single-ionization spectrum JSON (S_in) |
| `-dip PATH` | — | double-ionization spectrum JSON (S_fin) |
| `-fwhm-sip F` | `0` (= `-fwhm`) | FWHM for the SIP envelope |
| `-fwhm-dip F` | `0` (= `-fwhm`) | FWHM for the DIP envelope |
| `-fwhm-ees F` | `0` (= `-fwhm-sip`/`-fwhm-dip`) | FWHM for both envelopes inside the convolution |
| `-fin-channels S` | all | restrict the numerator's DIP channels; `N(E)` still uses all |
| `-xrange-ab LO-HI` | `0-40` | panel mode: shared window for (a) and (b) |
| `-xrange-c LO-HI` | `0-14` | panel mode: window for (c) |
| `-sip-group SPEC` | — | panel mode: group panel (a) MOs; bare flag prompts |

### `stack`

| Flag | Default | Meaning |
|---|---|---|
| `-stack LABEL=PATH[@OFFSET]` | — | add one trace explicitly; repeatable |
| `-stack-manifest PATH` | — | run manifest JSON; one trace per run with a spectrum |
| `-stack-dir DIR` | manifest's dir | where the spectrum JSONs live |
| `-stack-suffix S` | `.sip.json` | spectrum filename suffix |
| `-stack-energy-key S` | `relaxed_energy` | manifest field holding the total energy |
| `-stack-group-key S` | `n_waters` | manifest field `-stack-group` selects on |
| `-stack-group S` | all | plot only runs whose key equals this |
| `-manifest-unit S` | `hartree` | unit of the manifest energies |
| `-offset-unit S` | `kcal` | unit for the plotted offsets |
| `-stack-norm S` | `each` | `each` (own maximum) or `common` (one scale) |
| `-stack-trace-scale F` | `2` | trace height in units of the mean baseline spacing |
| `-stack-min-sep F` | `0.12` | minimum baseline separation, fraction of mean spacing |
| `-stack-dedup F` | `0` | fold traces agreeing to within this tolerance |
| `-stack-title S` | `Conformer-resolved single-ionization spectra` | figure title |
| `-stack-inset F` | `0.2` | structure-inset width, fraction of the energy axis |
| `-stack-inset-height F` | `0.5` | inset height, units of trace amplitude |
| `-stack-inset-lift F` | `0.05` | inset bottom above the trace baseline |
| `-stack-geom-suffix S` | `.zmat` | geometry suffix; empty disables insets |

## Gotchas

**`-xrange` rescales, it does not just crop.** Normalisation is computed in the
plotted window, so two crops of one spectrum are not height-comparable.

**Interactive prompts need a terminal.** In a script with neither `-group` nor
`-init-atom`, you silently get the `-init-atom` default (`O`) and the identity
grouping — every column its own site. Pass both explicitly in a pipeline.

**Channel order comes from the JSON.** Re-running `adcgo -spectrum` with a different
`-group` declaration order reorders `meta.channels`, which repaints the figure.
Keep the grouping fixed across a figure set.

**`.jpg` / `.jpeg` output is PNG data under the wrong name.** `savePlot` routes
those extensions through the raster path but encodes PNG regardless, so the file
is a PNG called `.jpg`. Viewers that sniff the header cope; strict consumers
(some LaTeX driver rules, some submission systems) will not. Write `.png`.

**Translucent fills do not survive the PDF backend.** Colours that need to look
identical in `.png` and `.pdf` must be pre-blended against the paper rather than
drawn with an alpha channel — this is why `stack` mode's fills are opaque.

**`-stack-dedup` is off by default** and drops nothing unless asked. When it is
on, read the stderr summary: it names every basin folded into a representative.
