"""
One visual system for every figure in the project.

Colors follow the ENTITY, not the chart: the shipped tree is blue in every figure,
the region-aware baseline is always orange, and so on. The categorical steps were
checked with a colorblind-safety validator (adjacent pairs for lines and bars, all
pairs for dot plots). Aqua and yellow sit below 3:1 contrast on the surface, so every
figure is paired with a printed table of the same numbers in the notebook.

Sizes are specified in CSS pixels and converted to points (1px = 0.75pt).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

PX = 0.75  # points per CSS pixel

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Entity -> color, fixed for the whole project.
MODEL_COLORS = {
    "Decision tree (shipped)": "#2a78d6",  # slot 1 blue
    "Decision tree + region": "#eb6834",   # slot 2 orange
    "Random forest": "#1baf7a",            # slot 3 aqua
    "Logistic regression": "#eda100",      # slot 4 yellow
}

# Class comparison uses emphasis: the class the story is about is slot 1, the
# background mass is gray.
FRAUD_COLOR = "#2a78d6"
CLEAN_COLOR = MUTED

LINE_W = 2 * PX
HAIRLINE = 1 * PX
MARKER = 8 * PX       # diameter, >= 8px
RING = 2 * PX         # surface ring around markers
GAP = 2 * PX          # surface gap between touching bars


def apply():
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": HAIRLINE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": HAIRLINE,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_2,
            "lines.linewidth": LINE_W,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def heading(fig, title, subtitle, left, legend_rows=1):
    """
    Title, subtitle and legend rows above the plot, spaced in inches so the gaps
    are the same whatever the figure height. `left` is the plot's left edge as a
    fraction of the figure; it must leave room for the y tick labels so everything,
    title included, lines up on the figure's left edge.

    Returns the y position (figure fraction) where the legend's top should sit.
    """
    height = fig.get_figheight()
    fig.text(0.015, 1 - 0.08 / height, title, ha="left", va="top", fontsize=13,
             fontweight="bold", color=INK)
    fig.text(0.015, 1 - 0.36 / height, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
    fig.subplots_adjust(left=left, top=1 - (0.80 + 0.24 * legend_rows) / height)
    return 1 - 0.56 / height


def legend(fig, anchor_y, ncol):
    fig.legend(loc="upper left", bbox_to_anchor=(0.005, anchor_y), ncol=ncol,
               handletextpad=0.5, columnspacing=1.6)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
