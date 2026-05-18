"""Lorenz-63 forecast — the chaotic cover-page demo.

State of the art for Lorenz-63 reservoir forecasts is roughly 6–10 Lyapunov
times of accurate tracking before drift dominates (Pathak et al., 2018).
With dt=0.02 and a leading Lyapunov exponent ≈ 0.9, that's ~330–550 steps.

The hyperparameters below were tuned to deliver a clean visual lock on the
attractor for the first ~600 steps of the validation horizon, with a
gradual phase drift visible by step ~1200.  Edit and re-run to refresh
both ``predict_lorenz.png`` and the matching docs pages.

Run as a script::

    .venv/bin/python -m scripts.docs_figures.lorenz
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from resdag import ott_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

from ._common import AMBER, GREY, INDIGO, apply_style, save
from .data import lorenz63


# --------------------------------------------------------------------------- #
# Hyperparameters — kept verbatim in docs/getting-started/lorenz-walkthrough.md
# and docs/examples/lorenz.md.
# --------------------------------------------------------------------------- #
N_STEPS = 40_000
WARMUP_STEPS = 2_000
TRAIN_STEPS = 20_000
VAL_STEPS = 1_500          # full horizon used for the tracking metric
PLOT_STEPS = 700           # window shown in the timeseries panels
DISCARD_STEPS = 3_000

RESERVOIR_SIZE = 1_500
SPECTRAL_RADIUS = 1.0
LEAK_RATE = 1.0
READOUT_ALPHA = 1e-7
TOPOLOGY: tuple | None = None     # default uniform random fill
SEED = 1


def _train_and_forecast() -> tuple[torch.Tensor, torch.Tensor]:
    data = lorenz63(N_STEPS, dt=0.02, seed=42)
    # Reseed *after* data generation so reservoir initialisation is governed
    # by SEED, not by the data-generator's internal manual_seed call.
    torch.manual_seed(SEED)
    warmup, train, target, f_warmup, val = prepare_esn_data(
        data,
        warmup_steps=WARMUP_STEPS,
        train_steps=TRAIN_STEPS,
        val_steps=VAL_STEPS,
        discard_steps=DISCARD_STEPS,
        normalize=True,
        norm_method="minmax",
    )

    kwargs: dict = dict(
        reservoir_size=RESERVOIR_SIZE,
        feedback_size=3,
        output_size=3,
        spectral_radius=SPECTRAL_RADIUS,
        leak_rate=LEAK_RATE,
        readout_alpha=READOUT_ALPHA,
    )
    if TOPOLOGY is not None:
        kwargs["topology"] = TOPOLOGY
    model = ott_esn(**kwargs)
    ESNTrainer(model).fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": target},
    )
    model.reset_reservoirs()
    pred = model.forecast(f_warmup, horizon=VAL_STEPS)
    return val[0].detach(), pred[0].detach()


def predict_lorenz() -> None:
    truth, pred = _train_and_forecast()
    err = ((truth - pred) ** 2).mean(dim=-1).sqrt()  # RMSE per timestep
    horizon = int(((err < 0.5).cumsum(dim=0)
                   == torch.arange(1, len(err) + 1)).sum().item())
    lyapunov_steps = horizon * 0.02 / 1.1     # MTU × 0.02 ÷ Lyapunov time ≈ 1.1
    print(f"    lorenz: tracked {horizon} steps "
          f"({lyapunov_steps:.1f} Lyapunov times)  /  "
          f"full val MSE = {float(((truth - pred) ** 2).mean()):.3e}")

    # ----- Timeseries panel: show only the early window where the model
    #        actually tracks the truth.  The tracking metric above is over
    #        the full VAL_STEPS so the figure stays honest.
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 4.2), sharex=True)
    truth_p = truth[:PLOT_STEPS].numpy()
    pred_p = pred[:PLOT_STEPS].numpy()
    for ax, j, label in zip(axes, range(3), ["x", "y", "z"]):
        ax.plot(truth_p[:, j], color=GREY, lw=1.6, label="truth")
        ax.plot(pred_p[:, j], color=AMBER, lw=1.0, label="prediction")
        ax.axvline(horizon, color=INDIGO, lw=0.8, ls="--", alpha=0.7)
        ax.set_ylabel(label, rotation=0, va="center", labelpad=12)
        ax.grid(True, alpha=0.6)
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"Lorenz autoregressive forecast — Ott ESN ({RESERVOIR_SIZE} units).  "
        f"Dashed line marks the {horizon}-step tracking horizon "
        f"(≈ {lyapunov_steps:.1f} Lyapunov times).",
        loc="left", fontsize=10,
    )
    axes[-1].set_xlabel("t after f_warmup")
    save(fig, "predict_lorenz.png")

    # ----- Phase-portrait overlay: 3-D view restricted to the accurate
    #        window. Powerful at-a-glance evidence that the model finds the
    #        right attractor.
    n_phase = min(horizon + 80, len(truth))
    fig2 = plt.figure(figsize=(7.5, 3.6))
    ax_t = fig2.add_subplot(1, 2, 1, projection="3d")
    ax_p = fig2.add_subplot(1, 2, 2, projection="3d")
    for ax, series, color, label in [
        (ax_t, truth[:n_phase].numpy(), GREY, "truth"),
        (ax_p, pred[:n_phase].numpy(), AMBER, "prediction"),
    ]:
        ax.plot(series[:, 0], series[:, 1], series[:, 2], color=color, lw=0.7)
        ax.set_title(label)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = "#e4e7ec"  # type: ignore[attr-defined]
    fig2.suptitle(
        f"Phase portrait — first {n_phase} forecast steps",
        fontsize=10.5, y=1.02,
    )
    save(fig2, "predict_lorenz_phase.png")


def main() -> None:
    apply_style()
    print("Lorenz prediction …")
    predict_lorenz()


if __name__ == "__main__":
    main()
