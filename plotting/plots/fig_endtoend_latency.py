"""Search cost and native mean latency at the reported operating points.

The left panel uses the selected NDC operating points. The right panel uses
five-repeat single-thread timings of the measured query paths. Deep100M uses
the complete monitor+restart harness (3.392 ms/query); its system_replay row
(2.425 ms/query) omits the monitor and is therefore not end-to-end.
"""
import argparse
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pilot-figures-matplotlib')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps


ps.init()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--monochrome', action='store_true',
                    help='Render a separate black-and-white print proof.')
args = parser.parse_args()
plt.rcParams['hatch.linewidth'] = .45
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True)

ORDER = ["GIST1M", "SIFT100M", "Deep100M", "LAION-10M"]
SERIES = ["plain", "DARTH", "LAET", "ours"]
LABEL = {"plain": "plain HNSW", "DARTH": "DARTH", "LAET": "LAET", "ours": "PILOT"}
COLOR = {"plain": ps.GRAY, "DARTH": ps.ORANGE, "LAET": ps.AQUA, "ours": ps.BLUE}
HATCH = {"plain": "///", "DARTH": "\\\\", "LAET": "xx", "ours": ".."}
if args.monochrome:
    COLOR = dict.fromkeys(SERIES, '#eeeeee')

# GIST controls retain the paper's 5,000-query population while PILOT uses the
# 4,527-query non-training subset. Every other dataset shares one population.
# SIFT executes the intended reusable-probe path directly. GIST and LAION
# replay the selected route. Deep uses the measured monitor+restart path because
# its integrated probe-resume executor has not yet been implemented.
METHOD = {
    "GIST1M": {"plain": "plain_paper5000", "DARTH": "darth_paper5000",
                "LAET": "laet_paper5000", "ours": "system_replay"},
    "SIFT100M": {"plain": "plain", "DARTH": "darth", "LAET": "laet",
                  "ours": "system"},
    "Deep100M": {"plain": "plain", "DARTH": "darth", "LAET": "laet",
                  "ours": "system_monitor_restart"},
    "LAION-10M": {"plain": "plain", "DARTH": "darth", "LAET": "laet",
                   "ours": "system_replay"},
}

ops = json.load(open(os.path.join(DATA, "exp1_operating_points.json")))
op = {(row["dataset"], row["method"].lower()): row
      for row in ops["rows"] if row.get("selected")}

rows = json.load(open(os.path.join(DATA, "timing_summary.json")))
lookup = {(row["dataset"], row["method"]): row for row in rows}
timing = {dataset: {series: lookup[(dataset, method)]
                    for series, method in methods.items()}
          for dataset, methods in METHOD.items()}

for dataset in ORDER:
    for series in SERIES:
        assert timing[dataset][series]["repeats"] == 5

assert abs(timing["SIFT100M"]["ours"]["mean_ms_median"] - 2.4120788125) < 1e-9
assert abs(timing["Deep100M"]["ours"]["mean_ms_median"] - 3.392372162) < 1e-9

x = np.arange(len(ORDER))
w = 0.19
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.05))

panels = [
    (axes[0], "(a) Distance computations", "NDC / query",
     lambda d, s: op[(d, s.lower())]["ndc"]),
    (axes[1], "(b) Mean latency", "latency (ms/query)",
     lambda d, s: timing[d][s]["mean_ms_median"]),
]

for ax, title, ylabel, value_of in panels:
    ps.despine(ax)
    ps.ygrid(ax)
    for si, series in enumerate(SERIES):
        values = [value_of(dataset, series) for dataset in ORDER]
        pos = x + (si - 1.5) * w
        bars = ax.bar(pos, values, w * .90, color=COLOR[series],
                      hatch=HATCH[series], edgecolor='#333333', linewidth=.4,
                      label=LABEL[series], zorder=3)
        if series == "ours":
            for dataset, bar, value in zip(ORDER, bars, values):
                ratio = value_of(dataset, "plain") / value
                ax.annotate(f"{ratio:.1f}$\\times$",
                            (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 2.5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.3,
                            color=ps.INK if args.monochrome else ps.BLUE,
                            bbox=dict(boxstyle="round,pad=0.08", fc="white",
                                      ec="none", alpha=.85))
    ax.set_xticks(x, ORDER, fontsize=6.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_ylim(0, max(value_of(d, s) for d in ORDER for s in SERIES) * 1.22)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .995),
           ncol=4, frameon=False, handlelength=1.6, handleheight=1.1, handletextpad=.35,
           columnspacing=1.1, borderpad=0)
fig.subplots_adjust(top=.78, bottom=.20, left=.09, right=.99, wspace=.32)
name = 'fig_endtoend_performance' + ('_bw' if args.monochrome else '')
# Preserve the intended physical size; LaTeX includes the figure at 0.93 textwidth.
fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches=None)
os.makedirs(ps.PREVIEW, exist_ok=True)
fig.savefig(os.path.join(ps.PREVIEW, name + '.png'), dpi=240, bbox_inches=None)
print(f'wrote {name}.pdf (+ preview)')
plt.close(fig)

print(f"{'dataset':<11}{'method':<12}{'NDC':>10}{'mean ms':>10}")
print("-" * 43)
for dataset in ORDER:
    for series in SERIES:
        print(f"{dataset if series == 'plain' else '':<11}{LABEL[series]:<12}"
              f"{op[(dataset, series.lower())]['ndc']:>10.0f}"
              f"{timing[dataset][series]['mean_ms_median']:>10.3f}")
