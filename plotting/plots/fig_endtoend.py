"""Exp-1: cost to serve a mean Recall@100 = 0.90 workload.

Methods are normally placed at the cheapest configuration measured to reach
0.90 on that dataset. SIFT Ours uses the exact EF1024 point because native
timing shows that it dominates the lower-NDC PCA point in latency, tail latency,
recall and attainment. See build_operating_points.py for all candidates and
selection overrides.

Emits two single-column figures, cost first:
  fig_endtoend_cost.pdf -- NDC/query at that point
  fig_endtoend_sla.pdf  -- the per-query attainment the same point delivers,
                           i.e. how the identical mean recall is distributed

POPULATIONS DIFFER BY DATASET: GIST1M and LAION-10M are natural query
distributions; Deep100M and SIFT100M are difficulty-shifted stress sets built by
their own summaries to overrepresent hard queries. Within a dataset all methods
share one population.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps

ps.init()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True)

OP = json.load(open(os.path.join(DATA, "exp1_operating_points.json")))
ORDER = ["GIST1M", "SIFT100M", "Deep100M", "LAION-10M"]
SERIES = ["plain", "DARTH", "LAET", "Ours"]
LABEL = {"plain": "plain HNSW", "DARTH": "DARTH", "LAET": "LAET", "Ours": "Ours"}
CLR = {"plain": ps.GRAY, "DARTH": ps.ORANGE, "LAET": ps.AQUA, "Ours": ps.BLUE}

sel = {(r["dataset"], r["method"]): r for r in OP["rows"] if r.get("selected")}
x = np.arange(len(ORDER))
w = 0.2

XT = [d.replace("-10M", "\\n-10M").replace("100M", "100M").replace("1M", "1M")
      for d in ORDER]

# ---- figure 1: cost -------------------------------------------------------
fig, axA = plt.subplots(figsize=(ps.COL_W, 1.95))
ps.despine(axA); ps.ygrid(axA)
for si, s in enumerate(SERIES):
    axA.bar(x + (si - 1.5) * w, [sel[(d, s)]["ndc"] for d in ORDER], w * 0.88,
            color=CLR[s], label=LABEL[s], zorder=3)
# direct-label our bar with the saving against the plain control. A white halo
# keeps the ratio attached to our (short) bar instead of reading as a label on
# the taller neighbour.
for xi, d in zip(x, ORDER):
    ours = sel[(d, "Ours")]["ndc"]
    axA.annotate(f"{sel[(d, 'plain')]['ndc']/ours:.1f}$\\times$",
                 (xi + 1.5 * w, ours), xytext=(0, 2.5), textcoords="offset points",
                 ha="center", va="bottom", fontsize=6.3, color=ps.BLUE,
                 bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
axA.set_xticks(x, ORDER, fontsize=6.6)
axA.set_ylabel("NDC / query")
axA.set_ylim(0, max(r["ndc"] for r in sel.values()) * 1.26)
axA.legend(loc="upper left", frameon=False, fontsize=6.0, ncol=2, handlelength=1.0,
           handletextpad=0.35, columnspacing=0.7, borderpad=0.0, labelspacing=0.15)
ps.save(fig, OUT, "fig_endtoend_cost")
plt.close(fig)

# ---- figure 2: service level ----------------------------------------------
fig, axB = plt.subplots(figsize=(ps.COL_W, 1.95))
ps.despine(axB); ps.ygrid(axB)
for si, s in enumerate(SERIES):
    att = [sel[(d, s)]["attainment"] for d in ORDER]
    # a point whose per-query records were never kept has no attainment; mark
    # the slot rather than drawing a zero-height bar, which reads as "0"
    axB.bar(x + (si - 1.5) * w, [0 if a is None else a for a in att], w * 0.88,
            color=CLR[s], label=LABEL[s], zorder=3)
    for xi, a in zip(x, att):
        if a is None:
            axB.annotate("n/a", (xi + (si - 1.5) * w, 0.02), ha="center",
                         va="bottom", rotation=90, fontsize=5.6, color=ps.INK2)
axB.set_xticks(x, ORDER, fontsize=6.6)
axB.set_ylabel("attainment\n(Recall@100 $\\geq$ 0.90)", fontsize=7.0, linespacing=1.1)
axB.set_ylim(0, 1.14)
axB.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axB.legend(loc="upper left", frameon=False, fontsize=6.0, ncol=2, handlelength=1.0,
           handletextpad=0.35, columnspacing=0.7, borderpad=0.0, labelspacing=0.15)
ps.save(fig, OUT, "fig_endtoend_sla")
plt.close(fig)

print(f"{'dataset':<11}{'method':<7}{'config':<32}{'recall':>8}{'NDC/q':>10}{'att':>8}")
print("-" * 78)
for d in ORDER:
    for s in SERIES:
        r = sel[(d, s)]
        a = f"{r['attainment']:.3f}" if r["attainment"] is not None else "n/a"
        print(f"{d if s=='plain' else '':<11}{s:<7}{r['config'][:31]:<32}"
              f"{r['recall']:>8.4f}{r['ndc']:>10,.0f}{a:>8}")
    ours = sel[(d, "Ours")]
    best = min((sel[(d, m)] for m in ("plain", "DARTH", "LAET")), key=lambda r: r["ndc"])
    pl = sel[(d, "plain")]
    if ours["attainment"] is None:
        att_txt = "attainment not recorded for this configuration"
    else:
        da = ours["attainment"] - max(sel[(d, m)]["attainment"]
                                      for m in ("plain", "DARTH", "LAET"))
        att_txt = (f"att {ours['attainment']:.3f} vs plain {pl['attainment']:.3f}; "
                   f"{da*100:+.1f}pp vs best baseline")
    print(f"{'':<11}-> {pl['ndc']/ours['ndc']:.2f}x cheaper than conservative plain, "
          f"{best['ndc']/ours['ndc']:.2f}x vs best learned ({best['method']}); {att_txt}\n")
