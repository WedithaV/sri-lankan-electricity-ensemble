"""Shared chart colours and Matplotlib style for notebooks and the dashboard.

Categorical order (teal, blue, orange) was checked with a colour-vision-deficiency
validator: all three pairs pass lightness, chroma, CVD separation, normal-vision
separation and 3:1 contrast on a light surface. Use at most three series per chart;
fold extra categories into "Other" or use small multiples.
Navy is used for text/titles only (it is too dark to work as a data colour).
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

NAVY = "#1B2A4A"        # titles, text
INK_2 = "#52514E"       # secondary text, axis labels
GRID = "#E3E6EA"        # recessive grid
SURFACE = "#FCFCFB"     # chart background
MUTED = "#9AA5B1"       # de-emphasised marks / reference lines

TEAL, BLUE, ORANGE = "#0F9D8F", "#2A78D6", "#EB6834"
SERIES = [TEAL, BLUE, ORANGE]

# Sequential (magnitude): one hue, light -> dark (blue ramp).
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#CDE2FB", "#6DA7EC", "#2A78D6", "#184F95", "#0D366B"])
# Diverging (sign, e.g. correlation): blue <- neutral gray -> orange.
DIVERGING = LinearSegmentedColormap.from_list("div_blue_orange", ["#184F95", "#86B6EF", "#EDEDEA", "#F2A27E", "#B8461A"])


def apply() -> None:
    """Apply the project Matplotlib style (thin marks, recessive axes)."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })
