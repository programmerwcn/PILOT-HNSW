"""Shared matplotlib style for the PVLDB two-column figures.

Column geometry: acmart sigconf gives ~3.33in per column and ~7.0in full width.
Figures are emitted as vector PDF at those exact widths so LaTeX never rescales
them -- rescaling is what makes fonts in one figure disagree with fonts in the
next.

Palette roles (dataviz reference instance, categorical slots in fixed order):
the plain-HNSW control is the reference series and wears neutral gray, so the
three *methods* occupy slots 1-3, which are the three that clear the all-pairs
CVD floors in both modes. Nothing is ever colored by rank.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_W = 3.33      # single column, inches
FULL_W = 7.0      # both columns

# categorical slots 1-3 + neutral control
BLUE = "#2a78d6"    # slot 1 -- our system
ORANGE = "#eb6834"  # slot 2 -- DARTH
AQUA = "#1baf7a"    # slot 3 -- LAET
GRAY = "#8a8985"    # neutral -- plain HNSW control
GRAY_D = "#52514e"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#c3c2b7"

METHOD_COLOR = {
    "plain": GRAY,
    "DARTH": ORANGE,
    "LAET": AQUA,
    "ours": BLUE,
}

# sequential single hue for confusion matrices (light -> dark, one hue)
SEQ_BLUE = ["#f2f7fd", "#d9e8f8", "#b0cff0", "#7db0e4", "#4a90d9", "#2a78d6", "#1b5296"]


def init():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": INK2,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
    })


def despine(ax, left=True, bottom=True):
    """Recessive axes: drop the box, keep only the scales that carry meaning."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def ygrid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=MUTED, alpha=0.55)
    ax.xaxis.grid(False)


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)


import os as _os
PREVIEW = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "..", "figures", "preview")


def save(fig, outdir, name):
    """Write the vector PDF LaTeX includes, plus a raster preview to eyeball.

    The validator checks color, not layout -- label collisions and overflow are
    only visible by looking at the rendered figure.
    """
    import os
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    os.makedirs(PREVIEW, exist_ok=True)
    fig.savefig(os.path.join(PREVIEW, f"{name}.png"), dpi=200)
    print(f"wrote {name}.pdf (+ preview)")

# Create output directories for independently invoked plotting scripts.
_os.makedirs(PREVIEW, exist_ok=True)
