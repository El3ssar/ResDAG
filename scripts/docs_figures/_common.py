"""Shared style + I/O helpers for every documentation figure.

Every other module in :mod:`scripts.docs_figures` imports its palette and
``save()`` helper from here so the site keeps a consistent look across the
~50 figures.  Edit this file to retheme the entire docs in one go; edit a
sibling module to tweak a single example.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Palette — keep in sync with docs/stylesheets/extra.css
# --------------------------------------------------------------------------- #
INDIGO = "#3d4f87"
INDIGO_LIGHT = "#7a8ec8"
INDIGO_DARK = "#2a376e"
AMBER = "#c2410c"
GREY = "#9aa0aa"
SOFT = "#f5f6fa"
RULE = "#e4e7ec"
MUTED = "#6b7280"


def apply_style() -> None:
    """Apply the global Matplotlib style.

    Called once by every entry-point script; safe to call multiple times.
    """
    matplotlib.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 144,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9ca3af",
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def save(fig: plt.Figure, name: str) -> None:
    """Save ``fig`` to ``docs/assets/figures/<name>`` and close it."""
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.relative_to(ROOT)}")
