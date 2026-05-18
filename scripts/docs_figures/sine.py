"""Sine forecasting — the cover-page demo.

The hyperparameters here are chosen so the autoregressive forecast tracks
the truth perfectly across the full validation window. Tweak the file and
re-run to update both the figures *and* the snippet on the home page.

Run as a script::

    .venv/bin/python -m scripts.docs_figures.sine
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch

from resdag import classic_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

from ._common import AMBER, GREY, apply_style, save
from .data import sine_wave


# Public knobs — referenced verbatim in docs/getting-started/your-first-esn.md
# and docs/examples/sine.md.  Keep names and values in sync with those pages.
N_STEPS = 6_000          # length of the synthetic series
PERIODS = 60             # number of full sine cycles across N_STEPS
WARMUP_STEPS = 500
TRAIN_STEPS = 4_500
VAL_STEPS = 800
RESERVOIR_SIZE = 300
SPECTRAL_RADIUS = 0.99
LEAK_RATE = 0.3
READOUT_ALPHA = 1e-8
SEED = 0


def _train_and_forecast() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(truth_val, prediction)``, each of shape ``(VAL_STEPS,)``."""
    torch.manual_seed(SEED)

    data = sine_wave(N_STEPS, periods=PERIODS)
    warmup, train, target, f_warmup, val = prepare_esn_data(
        data,
        warmup_steps=WARMUP_STEPS,
        train_steps=TRAIN_STEPS,
        val_steps=VAL_STEPS,
        normalize=False,
    )

    model = classic_esn(
        reservoir_size=RESERVOIR_SIZE,
        feedback_size=1,
        output_size=1,
        spectral_radius=SPECTRAL_RADIUS,
        leak_rate=LEAK_RATE,
        readout_alpha=READOUT_ALPHA,
    )
    ESNTrainer(model).fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": target},
    )
    model.reset_reservoirs()
    pred = model.forecast(f_warmup, horizon=VAL_STEPS)
    return val[0, :, 0].detach(), pred[0, :, 0].detach()


def predict_sine() -> None:
    truth, pred = _train_and_forecast()
    mse = float(((truth - pred) ** 2).mean())
    print(f"    sine val MSE = {mse:.3e}")

    fig, ax = plt.subplots(figsize=(7.5, 2.7))
    ax.plot(truth.numpy(), color=GREY, lw=1.8, label="truth")
    ax.plot(pred.numpy(), color=AMBER, lw=1.1, label="prediction")
    ax.set_title(
        f"Sine — autoregressive forecast over {VAL_STEPS} steps "
        f"(val MSE = {mse:.1e})",
        loc="left",
    )
    ax.set_xlabel("t after f_warmup")
    ax.set_ylabel("x(t)")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="upper right")
    save(fig, "predict_sine.png")


def main() -> None:
    apply_style()
    print("Sine prediction …")
    predict_sine()


if __name__ == "__main__":
    main()
