"""Standalone controller sample-efficiency plot for E2.

Read only the sample_efficiency block of the archived, provenance-linked
planner data; its legacy inference timings are not used by this figure.
Normalize within each task, not across prediction objectives.
"""
import json
from pathlib import Path

import paperstyle as ps
import matplotlib.pyplot as plt
import numpy as np


def main():
    ps.init()
    paper = Path(__file__).resolve().parents[1]
    with (paper / "data/planner_efficiency_ood.json").open() as handle:
        data = json.load(handle)["sample_efficiency"]

    fig, ax = plt.subplots(figsize=(ps.COL_W, 1.70))
    styles = [
        ("Pilot", "PILOT", ps.BLUE, "o", "-"),
        ("LAET", "LAET", ps.AQUA, "s", "--"),
        ("DARTH", "DARTH", ps.ORANGE, "^", "-."),
    ]
    for key, label, color, marker, line in styles:
        series = data[key]
        points = series["points"]
        full = series["full_score"]
        x = np.array([p["learner_rows"] for p in points])
        y = np.array([p["score"] for p in points]) / full
        err = np.array([p["score_std"] for p in points]) / full
        ax.errorbar(x, y, yerr=err, color=color, marker=marker,
                    linestyle=line, markersize=3.5, linewidth=1.2,
                    elinewidth=0.65, capsize=1.5, label=label)

    ax.axhline(0.95, color=ps.GRAY_D, linewidth=0.7, linestyle=":")
    ax.text(30, 0.965, "95% of full score", color=ps.GRAY_D, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(20, 4.2e6)
    ax.set_ylim(0.40, 1.04)
    ax.set_xlabel("Learner rows (log scale)")
    ax.set_ylabel("Relative score", labelpad=2)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ps.ygrid(ax)
    ps.despine(ax, left=False)
    ax.legend(frameon=False, loc="lower right", ncol=3,
              handlelength=1.7, handletextpad=0.4, columnspacing=0.8,
              borderaxespad=0.3, fontsize=6.7)
    fig.tight_layout(pad=0.25)
    ps.save(fig, str(paper / "figures"), "fig_planner_sample_efficiency")
    plt.close(fig)


if __name__ == "__main__":
    main()
