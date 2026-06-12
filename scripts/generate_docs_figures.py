"""Regenerate every figure in the docs, in the notebook brand style.

Sections (select with --only a,b,...): forecast, phase, topologies,
initializers, architectures, hpo.

Run from the repo root:  uv run python scripts/generate_docs_figures.py
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import resdag as rd
from resdag.training import ESNTrainer

FIG = Path(__file__).resolve().parent.parent / "docs" / "assets" / "figures"

# ---- Brand -----------------------------------------------------------------
PAPER = "#f8f8f6"
SURFACE = "#ffffff"
INK = "#222826"
INK_SOFT = "#4e5955"
MUTED = "#79847f"
RULE = "#cbd2cc"
PINE = "#1d6f54"
SLATE = "#5a6b9e"  # secondary data-viz hue only

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
    "axes.titleweight": "semibold",
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
})

PINE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "pine", [PAPER, "#9ec6b6", PINE, "#0d3a2c"])
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "pine_slate", [SLATE, "#aebadb", PAPER, "#9ec6b6", PINE])


def save(fig, name, **kw):
    out = FIG / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.15, **kw)
    plt.close(fig)
    print(f"  wrote {out.relative_to(FIG.parent.parent.parent)}")


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
    """Train ott_esn on Lorenz, keep the best of a few seeds."""
    data = lorenz_series(300 + 6000 + horizon + 1)
    mean, std = data.mean(dim=1, keepdim=True), data.std(dim=1, keepdim=True)
    data_n = (data - mean) / std
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
    return best  # (valid_horizon, pred, val)  -- normalized units, dt=0.01


# ---- Section: forecast ------------------------------------------------------
def gen_forecast():
    vh, pred, val = train_best_lorenz()
    lyap = 0.906  # largest Lyapunov exponent of Lorenz-63
    t = np.arange(val.shape[1]) * 0.01 * lyap  # time in Lyapunov times
    show = slice(0, 1800)

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 4.6), sharex=True)
    for i, (ax, comp) in enumerate(zip(axes, "xyz")):
        ax.plot(t[show], val[0, show, i], color=SLATE, lw=1.3, label="system")
        ax.plot(t[show], pred[0, show, i], color=PINE, lw=1.3, label="ESN forecast")
        ax.axvline(vh * 0.01 * lyap, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.set_ylabel(comp, rotation=0, va="center", style="italic")
        ax.margins(x=0)
    axes[0].legend(ncols=2, loc="upper right", fontsize=8)
    axes[0].text(0.012, 0.06, f"valid horizon ≈ {vh * 0.01 * lyap:.1f} Lyapunov times",
                 transform=axes[0].transAxes, fontsize=8, color=MUTED)
    axes[-1].set_xlabel("Lyapunov times  (λ₁ t)")
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.18)
    save(fig, "first_forecast.png")


# ---- Section: phase ---------------------------------------------------------
def gen_phase():
    vh, pred, val = train_best_lorenz(horizon=4000)
    fig = plt.figure(figsize=(8.6, 4.0))
    for k, (title, dat, color) in enumerate([
            ("system", val[0].numpy(), SLATE),
            ("ESN forecast (autonomous)", pred[0].numpy(), PINE)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot(dat[:, 0], dat[:, 1], dat[:, 2], color=color, lw=0.45, alpha=0.85)
        ax.set_title(title, pad=0)
        ax.set_axis_off()
        ax.view_init(elev=12, azim=-60)
        ax.set_facecolor(PAPER)
    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=0.92, bottom=0.02)
    save(fig, "lorenz_phase.png")


# ---- Section: topologies ----------------------------------------------------
def gen_topologies():
    import networkx as nx
    from resdag.init.topology import get_topology
    from resdag.init.topology.registry import _TOPOLOGY_REGISTRY

    SIZE = {"kleinberg_small_world": 49, "spectral_cascade": 45}  # grid / triangular
    for name in sorted(_TOPOLOGY_REGISTRY):
        n = SIZE.get(name, 48)
        topo = get_topology(name)
        w = torch.empty(n, n)
        try:
            topo.initialize(w, spectral_radius=None)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        A = w.numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                                       gridspec_kw={"width_ratios": [1.15, 1]})
        G = nx.from_numpy_array((np.abs(A) > 1e-9).astype(int), create_using=nx.DiGraph)
        pos = nx.circular_layout(G)
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=PINE, alpha=0.25,
                               width=0.7, arrows=False)
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=22, node_color=INK_SOFT,
                               linewidths=0)
        ax1.set_title(f"connectivity (n={n})")
        ax1.set_axis_off()
        vmax = np.abs(A).max() or 1
        ax2.imshow(A, cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax2.set_title("weight matrix")
        ax2.set_xticks([]); ax2.set_yticks([])
        for s in ax2.spines.values():
            s.set_visible(True); s.set_color(RULE)
        save(fig, f"topologies/{name}.png")


# ---- Section: initializers --------------------------------------------------
def gen_initializers():
    from resdag.init.input_feedback import get_input_feedback
    from resdag.init.input_feedback.registry import _INPUT_FEEDBACK_REGISTRY

    rows, cols = 80, 8
    EXTRA = {"chain_of_neurons_input": {"features": cols},
             "dendrocycle_input": {"C": 8},
             "ring_window": {"c": 0.25, "window": 10}}
    for name in sorted(_INPUT_FEEDBACK_REGISTRY):
        w = torch.empty(rows, cols)
        try:
            init = get_input_feedback(name, **EXTRA.get(name, {}))
            torch.manual_seed(3)
            init.initialize(w)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        A = w.numpy()
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        vmax = np.abs(A).max() or 1
        im = ax.imshow(A, cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_xlabel("input dim"); ax.set_ylabel("reservoir unit")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(RULE)
        cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
        cb.outline.set_edgecolor(RULE)
        cb.ax.tick_params(labelsize=7, color=MUTED)
        save(fig, f"initializers/{name}.png")


# ---- Section: architectures -------------------------------------------------
def gen_architectures():
    out = FIG / "architectures"
    out.mkdir(parents=True, exist_ok=True)
    factories = {
        "classic_esn": lambda: rd.models.classic_esn(64, 3, 3),
        "ott_esn": lambda: rd.models.ott_esn(64, 3, 3),
        "power_augmented": lambda: rd.models.power_augmented(64, 3, 3),
        "linear_esn": lambda: rd.models.linear_esn(64, 3),
        "headless_esn": lambda: rd.models.headless_esn(64, 3),
    }
    for name, make in factories.items():
        model = make()
        model.plot_model(show_shapes=True, rankdir="LR",
                         save_path=str(out / f"{name}.svg"), format="svg")
        print(f"  wrote architectures/{name}.svg")


# ---- Section: hpo -----------------------------------------------------------
def gen_hpo():
    import optuna
    from optuna.importance import PedAnovaImportanceEvaluator
    from optuna.visualization.matplotlib import (
        plot_optimization_history, plot_parallel_coordinate, plot_contour,
        plot_param_importances, plot_slice)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data = lorenz_series(300 + 2500 + 400 + 1)
    data = (data - data.mean(dim=1, keepdim=True)) / data.std(dim=1, keepdim=True)
    warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
        data, warmup_steps=300, train_steps=2500, val_steps=400)

    def objective(trial):
        torch.manual_seed(trial.number)
        model = rd.models.ott_esn(
            reservoir_size=trial.suggest_int("reservoir_size", 100, 700, step=50),
            feedback_size=3, output_size=3,
            spectral_radius=trial.suggest_float("spectral_radius", 0.5, 1.4),
            leak_rate=trial.suggest_float("leak_rate", 0.3, 1.0),
            readout_alpha=trial.suggest_float("alpha", 1e-9, 1e-3, log=True),
        )
        ESNTrainer(model).fit((warmup,), (train,), targets={"output": target})
        pred = model.forecast(f_warmup, horizon=400)
        err = (pred - val).norm(dim=-1).squeeze(0)
        vh = int((err > 0.9).nonzero()[0]) if (err > 0.9).any() else 400
        return -vh  # maximize valid horizon

    sampler = optuna.samplers.TPESampler(seed=11, multivariate=True)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=70, show_progress_bar=False)
    print(f"  best valid horizon: {-study.best_value:.0f} steps, params {study.best_params}")

    for fname, plot in [
        ("hpo_history", plot_optimization_history),
        ("hpo_importances",
         lambda s: plot_param_importances(s, evaluator=PedAnovaImportanceEvaluator())),
        ("hpo_parallel", plot_parallel_coordinate),
    ]:
        ax = plot(study)
        fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
        for a in fig.axes:
            a.set_facecolor(PAPER)
        fig.set_facecolor(PAPER)
        fig.set_size_inches(8.2, 3.4)
        save(fig, f"{fname}.png")

    ax = plot_contour(study, params=["spectral_radius", "leak_rate"])
    fig = ax.figure
    fig.set_facecolor(PAPER)
    fig.set_size_inches(5.4, 4.2)
    save(fig, "hpo_contour.png")

    axes = plot_slice(study)
    fig = axes[0].figure if hasattr(axes, "__len__") else axes.figure
    for a in fig.axes:
        a.set_facecolor(PAPER)
    fig.set_facecolor(PAPER)
    fig.set_size_inches(9.0, 2.8)
    save(fig, "hpo_slice.png")


SECTIONS = {
    "forecast": gen_forecast, "phase": gen_phase, "topologies": gen_topologies,
    "initializers": gen_initializers, "architectures": gen_architectures, "hpo": gen_hpo,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SECTIONS))
    args = ap.parse_args()
    for sec in args.only.split(","):
        print(f"[{sec}]")
        SECTIONS[sec.strip()]()
