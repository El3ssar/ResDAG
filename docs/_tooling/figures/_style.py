"""Shared figure style — every value comes from docs/_tooling/theme.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "docs" / "assets" / "figures"
THEME = json.loads((Path(__file__).parents[1] / "theme.json").read_text())

DV = THEME["dataviz"]
PAPER, INK, INK_SOFT = DV["paper"], DV["ink"], DV["ink_soft"]
MUTED, RULE = DV["muted"], DV["rule"]
ACCENT, TRUE = DV["accent"], DV["true"]

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
    }
)

ACCENT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "nb_accent", [PAPER, DV["accent_soft"], ACCENT, DV["accent_deep"]]
)
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "nb_diverging", [TRUE, DV["true_soft"], PAPER, DV["accent_soft"], ACCENT]
)


def save(fig, name, **kw):
    out = FIG / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.15, **kw)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")
