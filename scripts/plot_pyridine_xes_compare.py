#!/usr/bin/env python
"""Pyridine N K-edge valence-to-core XES: ADCgo CVS-ADC(4) vs the experimental
valence-orbital ladder.

The cross_emissions in the -tdm document pair CVS core-hole states (init) with
valence-hole states (mid); the physical N Kalpha band is the main N1s core hole
(lowest init) decaying to genuine valence holes (mid). Its band positions are
E_core - E(valence hole), so they trace the occupied valence manifold. We overlay
the experimental pyridine vertical ionization potentials mapped through the same
computed core energy (E_core_main - VIP) as vertical markers: if the computed
emission bands line up with them, the spectrum reproduces the valence structure.
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TDM = sys.argv[1] if len(sys.argv) > 1 else "pyridine_N_tdm.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "pyridine_N_xes_compare.png"

d = json.load(open(TDM))
ce = d["cross_emissions"]

# Main N1s core-hole initial state = lowest core-sector energy.
e_core = min(c["init_ev"] for c in ce)
# Physical N Kalpha: init = main core hole, mid = valence hole (exclude the core-hole
# state that the valence companion sector also carries near ~e_core).
lines = [(c["omega_ev"], c["osc"]) for c in ce
         if abs(c["init_ev"] - e_core) < 1e-3 and c["mid_ev"] < 60 and c["osc"] > 0]
om = np.array([l[0] for l in lines])
osc = np.array([l[1] for l in lines])
print(f"main core hole E={e_core:.2f} eV; {len(lines)} physical valence->core lines")

# Gaussian broaden onto a grid.
fwhm = 0.8
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
grid = np.linspace(om.min() - 3, om.max() + 3, 2000)
env = np.zeros_like(grid)
for w, f in zip(om, osc):
    env += f * np.exp(-0.5 * ((grid - w) / sigma) ** 2)
env /= env.max()

# Experimental pyridine vertical ionization potentials (eV) and orbital labels
# (outer valence; standard He-I/synchrotron PES assignment).
vips = [
    (9.7, "1a$_2$/7a$_1$\n($\\pi$, n$_N$)"), (10.5, "2b$_1$ ($\\pi$)"),
    (12.4, "6a$_1$/4b$_2$"), (13.1, "1b$_1$ ($\\pi$)"), (13.8, "5a$_1$"),
    (14.4, "3b$_2$"), (15.6, "4a$_1$"),
]

# Focus on the outer-valence N Kalpha region — the cleanly measurable band where the
# main emission lines lie. (Inner-valence / shake-up structure at lower emission
# energy, mid_ev > ~25 eV, is real but its ADC(3) intensities are less reliable.)
LO, HI = 386.0, 398.5

# ---- figure ----
ACCENT = "#2563eb"   # computed emission (single series)
REF = "#9ca3af"      # experimental orbital markers (recessive)
INK = "#1f2937"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#d1d5db",
                     "axes.linewidth": 0.8, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(9, 5.2))

win = (grid >= LO) & (grid <= HI)
envw = env[win] / env[win].max()
ax.fill_between(grid[win], envw, color=ACCENT, alpha=0.15, zorder=2)
ax.plot(grid[win], envw, color=ACCENT, lw=2, zorder=3,
        label="ADCgo CVS-ADC(4) N K$\\alpha$ (FWHM 0.8 eV)")
inb = (om >= LO) & (om <= HI)
ax.vlines(om[inb], 0, osc[inb] / osc[inb].max(), color=ACCENT, alpha=0.35, lw=0.9, zorder=1)

# Experimental valence orbitals mapped to emission energy through the computed core.
for i, (vip, lab) in enumerate(vips):
    x = e_core - vip
    if LO <= x <= HI:
        ax.axvline(x, color=REF, ls=(0, (4, 3)), lw=1, zorder=1)
        ytxt = 1.15 if i % 2 == 0 else 1.05
        ax.text(x, ytxt, lab, va="bottom", ha="center", fontsize=8, color="#6b7280")

ax.set_xlim(HI, LO)  # emission: high energy (HOMO) on the left
ax.set_ylim(0, 1.28)
ax.set_xlabel("Emission energy (eV)")
ax.set_ylabel("Relative intensity")
ax.set_title("Pyridine N K$\\alpha$ valence-to-core emission — ADCgo vs. valence-orbital ladder",
             fontsize=12.5, pad=32)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=3)
ax.legend(loc="center left", frameon=False, fontsize=9, bbox_to_anchor=(0.02, 0.62))
ax.text(0.98, 0.60,
        f"main N1s core hole = {e_core:.1f} eV\n(expt. N1s BE $\\approx$ 404.9 eV)\n"
        "gray = expt. valence VIP\nmapped as $E_{core}-$VIP",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#6b7280")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
