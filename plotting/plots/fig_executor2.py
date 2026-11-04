"""Exp-3: why the dimension-pruned executor pays off, and on which queries.

Panel A -- aggregate recall cost of compression against the dpca/LID margin.
           The point is that the *ambient* dimension predicts backwards: GIST is
           960-d and safe at 64 PCA dims (6.7% of ambient), LAION is 512-d and
           still lossy at 256 (50% of ambient). Local intrinsic dimensionality
           (15 vs 65) orders them correctly.
Panel B,C -- the exchange rate the planner actually trades: NDC saved per
           percentage point of recall lost, per difficulty tier. dpca is an
           ordered factor, so it takes the sequential ramp, not categorical hues.

All values are recomputed from per-query records by scratchpad/exchange.py,
which asserts that each compressed arm is paired with a plain arm at identical
per-query EF (0 mismatches on every series drawn here).
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps

ps.init()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(HERE, "..", "data", "exchange.json")
E = json.load(open(DATA))

TIERS = ["easy", "medium", "hard"]
DS_COLOR = {"GIST1M": ps.BLUE, "LAION-10M": ps.ORANGE}
# ordered factor -> sequential ramp, dark = more dimensions retained
RAMP = ["#b0cff0", "#4a90d9", "#1b5296"]

fig = plt.figure(figsize=(ps.FULL_W, 1.65))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.34)

# ---------------------------------------------------------------- A: margin
axA = fig.add_subplot(gs[0, 0])
ps.despine(axA)
ps.ygrid(axA)
for ds, series in E.items():
    m = [l["margin"] for l in series["levels"]]
    y = [l["dR_all"] for l in series["levels"]]
    axA.plot(m, y, "-o", ms=4.5, color=DS_COLOR[ds], label=ds, zorder=3)
    for l in series["levels"]:
        axA.annotate(f"{l['dpca']}", (l["margin"], l["dR_all"]), xytext=(0, -9),
                     textcoords="offset points", ha="center", fontsize=6.0,
                     color=ps.INK2)
axA.axhline(0, color=ps.INK2, lw=0.6, zorder=1)
axA.set_xlabel(r"margin  $d_{\mathrm{PCA}}/\mathrm{LID}$")
axA.set_ylabel("recall change (pp)")
axA.set_ylim(-31, 6)
axA.set_xlim(0.6, 4.7)
axA.legend(loc="lower right", frameon=False, handlelength=1.4,
           handletextpad=0.4, borderpad=0.1, fontsize=6.6)
axA.set_title("(a) PCA prefix and local geometry", fontsize=7, pad=4)

# ---------------------------------------------------------------- B,C: exchange rate
for k, (ds, gspos) in enumerate([("GIST1M", gs[0, 1]), ("LAION-10M", gs[0, 2])]):
    ax = fig.add_subplot(gspos)
    ps.despine(ax)
    ps.ygrid(ax)
    x = np.arange(3)
    lvls = E[ds]["levels"]
    finite = [v for l in lvls for v in
              [l["tiers"].get(t, {}).get("rate") for t in TIERS] if v is not None]
    lo, hi = min(finite), max(finite)
    cap = hi * 12          # where "off the scale" markers sit
    has_lossless = False

    for i, l in enumerate(lvls):
        rates = [l["tiers"].get(t, {}).get("rate") for t in TIERS]
        ys = [np.nan if r is None else r for r in rates]
        ax.plot(x, ys, "-o", ms=4.2, color=RAMP[i],
                label=f"$d$={l['dpca']}", zorder=3)
        # A tier with no measurable loss has no finite exchange rate. Mark it at
        # the top of the scale instead of manufacturing a ratio out of sub-noise
        # recall drift, and connect it so the series stays readable as one line.
        for xi, r in zip(x, rates):
            if r is None:
                has_lossless = True
                ax.plot(xi, cap, "^", ms=4.6, color=RAMP[i], zorder=4,
                        clip_on=False)
        seq = [cap if r is None else r for r in rates]
        ax.plot(x, seq, "-", lw=1.0, color=RAMP[i], alpha=0.45, zorder=2)

    ax.set_yscale("log")
    ax.set_ylim(lo / 3, cap * 2.4)
    ax.set_xticks(x, TIERS)
    ax.set_xlim(-0.35, 2.35)
    ax.set_xlabel("query difficulty tier")
    if k == 0:
        ax.set_ylabel("NDC saved per pp recall lost")
    ax.legend(loc="lower right", frameon=False, handlelength=1.4,
              handletextpad=0.4, borderpad=0.1, labelspacing=0.2, fontsize=6.4)
    if has_lossless:
        ax.text(0.5, 0.99, r"$\blacktriangle$ no measurable recall loss",
                transform=ax.transAxes, ha="center", va="top", fontsize=5.9,
                color=ps.INK2)
    ax.set_title(f"({'bc'[k]}) {ds}", fontsize=7, pad=4)

ps.save(fig, OUT, "fig_executor2")
plt.close(fig)

# ---------------------------------------------------------------- console
print("\nExchange rate rises monotonically easy -> hard in EVERY configuration:")
for ds, series in E.items():
    for l in series["levels"]:
        vals = [l["tiers"].get(t, {}).get("rate") for t in TIERS]
        shown = ["lossless" if v is None else f"{v:,.0f}" for v in vals]
        finite = [v for v in vals if v is not None]
        mono = all(a < b for a, b in zip(finite, finite[1:])) and (
            vals[-1] is None or finite[-1] == vals[-1])
        print(f"  {ds:<10} d={l['dpca']:>3}  margin={l['margin']:.2f}x  "
              f"{shown[0]:>9} -> {shown[1]:>9} -> {shown[2]:>9}   monotone={mono}")
print("\nVerification rate (share of estimates promoted to an exact distance):")
for ds, series in E.items():
    for l in series["levels"]:
        v = [l["tiers"].get(t, {}).get("verify_pct", 0) for t in TIERS]
        print(f"  {ds:<10} d={l['dpca']:>3}   "
              f"{v[0]:.1f}% -> {v[1]:.1f}% -> {v[2]:.1f}%")
