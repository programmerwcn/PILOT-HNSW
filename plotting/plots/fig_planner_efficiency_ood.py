"""Training cost, sample efficiency, and OOD robustness for E4.

All plotted values come from data/planner_efficiency_ood.json.  The JSON also
records the raw artifact provenance and values that remain tabular in the paper.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps

ps.init()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "data", "planner_efficiency_ood.json")
with open(DATA, "r", encoding="utf-8") as f:
    data = json.load(f)

methods = ["Pilot", "LAET", "DARTH"]
colors = {"Pilot": ps.BLUE, "LAET": ps.AQUA, "DARTH": ps.ORANGE}
markers = {"Pilot": "o", "LAET": "s", "DARTH": "^"}

fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.05),
                         gridspec_kw={"width_ratios": [0.82, 1.35, 1.05]})

# (a) Fit time: logarithmic because the point is the order-of-magnitude gap.
ax = axes[0]
fit = [data["training_cost"][m]["fit_seconds"] for m in methods]
bars = ax.bar(np.arange(3), fit, color=[colors[m] for m in methods], width=0.62)
ax.set_yscale("log")
ax.set_ylim(0.035, 5.0)
ax.set_xticks(np.arange(3), methods, rotation=20)
ax.set_ylabel("fit time (s, log scale)")
ax.set_title("(a) Training cost", loc="left", fontweight="bold")
ps.ygrid(ax)
ps.despine(ax, left=False)
for b, value in zip(bars, fit):
    ax.text(b.get_x() + b.get_width() / 2, value * 1.13, f"{value:.3f}",
            ha="center", va="bottom", fontsize=6.7)

# (b) Normalize only within each method: their raw learning objectives differ.
ax = axes[1]
for method in methods:
    series = data["sample_efficiency"][method]
    pts = series["points"]
    x = np.array([p["learner_rows"] for p in pts])
    y = np.array([p["score"] / series["full_score"] for p in pts])
    ax.plot(x, y, marker=markers[method], ms=3.8, color=colors[method],
            label=method, markeredgewidth=0.0)
ax.axhline(0.95, color=ps.GRAY_D, lw=0.7)
ax.text(29, 0.954, "95% of full", fontsize=6.4, color=ps.GRAY_D, va="bottom")
ax.set_xscale("log")
ax.set_xlim(20, 4.2e6)
ax.set_ylim(0.40, 1.025)
ax.set_xlabel("learner rows (log scale)")
ax.set_ylabel("fraction of own full score")
ax.set_title("(b) Sample efficiency", loc="left", fontweight="bold")
ps.ygrid(ax)
ps.despine(ax, left=False)
ax.legend(frameon=False, loc="lower right", handlelength=1.4,
          borderpad=0.1, labelspacing=0.25)

# (c) The desirable OOD direction is upper-left: higher recall at lower work.
ax = axes[2]
for method in methods:
    p = data["ood_zero_shot"][method]
    x, y = p["end_to_end_ndc"], p["recall_at_100"]
    ax.scatter(x, y, s=34, marker=markers[method], color=colors[method], zorder=3)
    offsets = {"Pilot": (5, -1), "LAET": (5, -1), "DARTH": (-5, 4)}
    align = "right" if method == "DARTH" else "left"
    ax.annotate(f"{method}\natt. {p['target_attainment']:.2f}", (x, y),
                xytext=offsets[method], textcoords="offset points",
                ha=align, va="center", fontsize=6.5)
ax.set_xlim(5500, 15500)
ax.set_ylim(0.855, 0.952)
ax.set_xlabel("end-to-end NDC / query")
ax.set_ylabel("OOD Recall@100")
ax.set_title("(c) Zero-shot OOD", loc="left", fontweight="bold")
ps.ygrid(ax)
ps.despine(ax, left=False)
ax.annotate("better", xy=(6100, 0.947), xytext=(8200, 0.917),
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": ps.GRAY_D},
            color=ps.GRAY_D, fontsize=6.5, ha="center")

fig.subplots_adjust(wspace=0.48)
ps.save(fig, OUT, "fig_planner_efficiency_ood")
plt.close(fig)

