"""Exp-2: the search-extent classifier (Stage-1 executor).

Two figures:
  fig_classifier_cm.pdf   -- row-normalized confusion matrices, 4 datasets
  fig_classifier_cost.pdf -- accuracy per feature block vs. its online NDC cost

Every number is transcribed from a results/*/summary.md table or the stored
model artifact it was produced from; provenance is in the dict comments.
"""
import argparse
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps

ps.init()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--only-cm", action="store_true",
                    help="Regenerate only the confusion-matrix figure.")
args = parser.parse_args()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data
# rows = true tier, cols = predicted, order [easy, medium, hard]
DATASETS = [
    # name, confusion matrix, accuracy, majority-class baseline, eval population
    ("GIST1M",
     [[2841, 487, 5], [409, 2403, 522], [9, 553, 2771]],
     0.8015, 0.333, "10,000 held-out"),          # bridge/labels/k100_classifier_50000_v2_50k_result.json
    ("SIFT100M",
     [[1140, 164, 0], [178, 1004, 206], [3, 209, 1096]],
     0.810, 0.333, "4,000 untouched"),           # sift100m_M16efc100_summary.md 2.1
    # probe-trajectory features replaced the 100K-anchor priors: 0.638 -> 0.831
    ("Deep100M",
     [[294, 50, 0], [38, 272, 36], [5, 40, 265]],
     0.831, 0.346, "1,000 public"),              # PROBE_FEATURES_FINDINGS.md
    ("LAION-10M",
     [[260, 59, 8], [74, 225, 56], [10, 53, 255]],
     0.740, 0.355, "1,000 held-out"),            # ef3_classifier_report_50k_routing.json
]
TIERS = ["easy", "med", "hard"]

# GIST feature-group ablation: bridge/labels/feature_group_ablation_50000_result.json.
# online_ndc = distance evaluations the block costs per query at serving time.
# The probe is a resumed prefix of the search itself, so routing features are free;
# the anchor signature needs distances to 100,000 anchor vectors.
BLOCKS = [
    ("routing (probe)",      24,  0.7802, 0),
    ("static prior (anchor)", 31, 0.7620, 100_000),
    ("expert composites",     41, 0.7523, 100_000),
    ("query PCA",            128, 0.7388, 0),
    ("routing + static",      55, 0.7958, 100_000),
    ("all blocks",           226, 0.8021, 100_000),
]

# ---------------------------------------------------------------- fig 1: matrices
def truncated_percentage(count, total):
    """Format a nonnegative count ratio as percent, truncated to one decimal."""
    tenths = int(count) * 1000 // int(total)
    return f"{tenths // 10}.{tenths % 10}"


fig, grid = plt.subplots(2, 2, figsize=(ps.COL_W, 2.65))
axes = grid.ravel()
fig.subplots_adjust(left=.13, right=.985, bottom=.14, top=.92,
                    hspace=.50, wspace=.10)
cmap = ps.seq_cmap()

for panel, (ax, (name, cm, acc, maj, pop)) in enumerate(zip(axes, DATASETS)):
    M = np.array(cm, float)
    Mn = M / M.sum(axis=1, keepdims=True)          # row-normalized: read as recall per true tier
    ax.imshow(Mn, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    # Equal-aspect images leave unused width in each subplot slot. Anchor
    # both columns inward so that this slack does not widen the center gutter.
    ax.set_anchor("E" if panel % 2 == 0 else "W")
    for i in range(3):
        for j in range(3):
            # text wears ink tokens, not the series color; flip to white on dark cells
            ax.text(j, i, truncated_percentage(M[i, j], M[i].sum()),
                    ha="center", va="center",
                    fontsize=6.8, color="white" if Mn[i, j] > 0.55 else ps.INK)
    ax.set_xticks(range(3), TIERS, fontsize=6.8)
    ax.set_yticks(range(3), TIERS if panel % 2 == 0 else ["", "", ""],
                 fontsize=6.8)
    ax.set_title(f"({chr(97+panel)}) {name}", fontsize=7.5, pad=3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

fig.text(axes[0].get_position().x0 - .15, .53, "true tier", va="center", rotation=90,
         fontsize=7.2, color=ps.INK2)
fig.text(.55, .025, "predicted tier", ha="center", fontsize=7.5, color=ps.INK2)
# A native single-column 2x2 layout keeps the cell annotations readable.
with plt.rc_context({"savefig.bbox": None}):
    fig.savefig(os.path.join(OUT, "fig_classifier_cm.pdf"))
    os.makedirs(ps.PREVIEW, exist_ok=True)
    fig.savefig(os.path.join(ps.PREVIEW, "fig_classifier_cm.png"), dpi=240)
print("wrote fig_classifier_cm.pdf (+ preview)")
plt.close(fig)
if args.only_cm:
    sys.exit(0)

# ---------------------------------------------------------------- fig 2: accuracy vs cost
# Dot plot, not bars: the interesting range is 0.73-0.81, and position encoding
# can be zoomed honestly where a truncated bar length cannot. Online cost is
# carried by marker shape as well as hue, so identity is never colour-alone.
fig, ax = plt.subplots(figsize=(ps.COL_W, 1.85))
ps.despine(ax, left=False)
ax.set_axisbelow(True)
ax.xaxis.grid(True, color=ps.MUTED, alpha=0.55)
ax.yaxis.grid(False)

y = np.arange(len(BLOCKS))[::-1]
for yi, (lab, nfeat, acc, cost) in zip(y, BLOCKS):
    isfree = cost == 0
    ax.plot([0.715, acc], [yi, yi], color=ps.MUTED, lw=0.8, zorder=1)
    ax.plot(acc, yi, "o" if isfree else "s", ms=5.5 if isfree else 4.8,
            color=ps.BLUE if isfree else "white",
            markeredgecolor=ps.BLUE if isfree else ps.GRAY_D,
            markeredgewidth=0 if isfree else 1.0, zorder=3)

ax.set_yticks(y, [f"{l} [{n}]" for l, n, _, _ in BLOCKS], fontsize=6.8)
ax.set_xlim(0.715, 0.822)
ax.set_ylim(-0.7, len(BLOCKS) - 0.3)
ax.set_xlabel("three-tier accuracy (GIST1M, 10k held-out)")

# solid hairline reference, not dashed: this is a real threshold, and dashing
# reads as "projection" per the anti-pattern list
ax.axvline(0.8021, color=ps.INK2, lw=0.7, zorder=0)

# selective direct labels only: the free-feature result and the ceiling it nears
ax.annotate(f"{BLOCKS[0][2]:.3f}", (BLOCKS[0][2], y[0]), xytext=(5, 0),
            textcoords="offset points", va="center", fontsize=6.8, color=ps.INK,
            fontweight="bold")
ax.annotate(f"{BLOCKS[-1][2]:.3f}", (BLOCKS[-1][2], y[-1]), xytext=(5, 0),
            textcoords="offset points", va="center", fontsize=6.8, color=ps.INK2)

h_free = plt.Line2D([], [], marker="o", ms=5.5, ls="none", color=ps.BLUE)
h_paid = plt.Line2D([], [], marker="s", ms=4.8, ls="none", color="white",
                    markeredgecolor=ps.GRAY_D, markeredgewidth=1.0)
ax.legend([h_free, h_paid], ["probe-derived (free)", "anchor (+100k NDC/q)"],
          loc="lower left", bbox_to_anchor=(-0.01, 1.0, 1.0, 0.12), mode="expand",
          ncol=2, frameon=False, handletextpad=0.35, borderpad=0.0,
          columnspacing=1.0, fontsize=6.4)
ps.save(fig, OUT, "fig_classifier_cost")
plt.close(fig)

# ---------------------------------------------------------------- console summary
print("\nPer-tier precision/recall, and the operationally critical error rate:")
for name, cm, acc, maj, pop in DATASETS:
    M = np.array(cm, float)
    rec = M.diagonal() / M.sum(1)
    prec = M.diagonal() / M.sum(0)
    exposed = M[2, 0] / M[:, 0].sum()      # predicted easy but truly hard
    print(f"  {name:<10} acc={acc:.3f}  recall(e/m/h)="
          f"{rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}  "
          f"prec={prec[0]:.2f}/{prec[1]:.2f}/{prec[2]:.2f}  "
          f"P(truly hard | predicted easy)={exposed*100:.2f}%")
