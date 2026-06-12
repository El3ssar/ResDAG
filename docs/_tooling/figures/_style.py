"""Shared figure style — every value comes from docs/_tooling/theme.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "docs" / "assets" / "figures"
THEME = json.loads((Path(__file__).parents[1] / "theme.json").read_text())

DV = THEME["dataviz"]
PAPER, INK, INK_SOFT = DV["paper"], DV["ink"], DV["ink_soft"]
MUTED, RULE = DV["muted"], DV["rule"]
ACCENT, TRUE = DV["accent"], DV["true"]

plt.rcParams.update({
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
})

ACCENT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "nb_accent", [PAPER, DV["accent_soft"], ACCENT, DV["pine_deep"]])
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "nb_diverging", [TRUE, DV["true_soft"], PAPER, DV["accent_soft"], ACCENT])


def save(fig, name, **kw):
    out = FIG / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.15, **kw)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def lorenz_series(n, dt=0.01, skip=500):
    s = np.array([1.0, 1.0, 25.0])

    def f(s):
        x, y, z = s
        return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z])

    out = np.empty((n, 3))
    for i in range(n + skip):
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= skip:
            out[i - skip] = s
    return torch.tensor(out, dtype=torch.float32).unsqueeze(0)


def train_best_lorenz(horizon=2500, tries=6):
    """Train ott_esn on Lorenz over a small grid; keep the best run."""
    import resdag as rd
    from resdag.training import ESNTrainer

    data = lorenz_series(300 + 6000 + horizon + 1)
    data_n = (data - data.mean(dim=1, keepdim=True)) / data.std(dim=1, keepdim=True)
    warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
        data_n, warmup_steps=300, train_steps=6000, val_steps=horizon)
    best = None
    for seed in range(tries):
        for sr in (0.8, 0.95, 1.1):
            for alpha in (1e-8, 1e-6):
                torch.manual_seed(seed)
                model = rd.models.ott_esn(reservoir_size=900, feedback_size=3,
                                          output_size=3, spectral_radius=sr,
                                          readout_alpha=alpha)
                ESNTrainer(model).fit((warmup,), (train,), targets={"output": target})
                pred = model.forecast(f_warmup, horizon=horizon)
                err = (pred - val).norm(dim=-1).squeeze(0)
                vh = int((err > 0.9).nonzero()[0]) if (err > 0.9).any() else horizon
                if best is None or vh > best[0]:
                    best = (vh, pred.clone(), val.clone())
    print(f"  best valid horizon: {best[0]} steps ({best[0] * 0.01 * 0.906:.1f} Lyapunov times)")
    return best
