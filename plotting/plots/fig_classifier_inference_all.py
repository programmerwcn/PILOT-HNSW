"""Single-column estimated per-query classifier inference cost."""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps

ps.init()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "figures")
SRC = os.path.join(HERE, "..", "data", "classifier_inference_all_datasets.json")
data = json.load(open(SRC))["datasets"]

datasets = ["GIST1M", "SIFT100M", "Deep100M", "LAION-10M"]
methods = ["DARTH", "LAET", "Pilot"]
colors = {"Pilot": ps.BLUE, "LAET": ps.AQUA, "DARTH": ps.ORANGE}
labels = {"Pilot": "PILOT", "LAET": "LAET", "DARTH": "DARTH"}
hatches = {"DARTH": "\\\\", "LAET": "xx", "Pilot": ".."}
plt.rcParams["hatch.linewidth"] = .45
x = np.arange(len(datasets)); width = .24
fig, ax = plt.subplots(figsize=(ps.COL_W, 1.90))
ps.despine(ax, left=False)
ps.ygrid(ax)
for j, method in enumerate(methods):
    vals = [data[d][method]["microseconds_per_query"] for d in datasets]
    pos = x + (j - 1) * width
    ax.bar(pos, vals, width * .9, color=colors[method],
           label=labels[method], hatch=hatches[method],
           edgecolor="#333333", linewidth=.4, zorder=3)

ax.set_yscale("log")
ax.set_ylim(25, 1500)
ax.set_yticks([40, 100, 400, 1000], ["40", "100", "400", "1000"])
ax.minorticks_off()
ax.set_xticks(x, datasets, fontsize=6.5)
ax.set_ylabel(r"Inference time ($\mu$s/query, log)", labelpad=2)

handles, legend_labels = ax.get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.57, .99),
           ncol=3, frameon=False, handlelength=1.6, handleheight=1.1,
           handletextpad=.35, columnspacing=1.1, borderpad=0)
fig.subplots_adjust(top=.81, bottom=.17, left=.17, right=.99)
with plt.rc_context({"savefig.bbox": None}):
    fig.savefig(os.path.join(OUT, "fig_classifier_inference_all.pdf"))
    os.makedirs(ps.PREVIEW, exist_ok=True)
    fig.savefig(os.path.join(ps.PREVIEW, "fig_classifier_inference_all.png"), dpi=240)
print("wrote fig_classifier_inference_all.pdf (+ preview)")
plt.close(fig)
