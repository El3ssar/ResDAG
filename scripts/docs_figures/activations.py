"""Reservoir activation plot for the sine quickstart."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from resdag.layers import ESNLayer

from ._common import INDIGO, apply_style, save
from .data import sine_wave


def activations_sine() -> None:
    torch.manual_seed(0)
    x = sine_wave(300, periods=3.0)

    layer = ESNLayer(
        reservoir_size=200,
        feedback_size=1,
        spectral_radius=0.9,
        topology="erdos_renyi",
    )
    layer.reset_state()
    with torch.no_grad():
        states = layer(x)[0].numpy()  # (T, 200)

    fig, (ax_signal, ax_states) = plt.subplots(
        2, 1, figsize=(7.5, 3.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.4]},
    )
    ax_signal.plot(x[0, :, 0].numpy(), color=INDIGO, lw=1.4)
    ax_signal.set_ylabel("input")
    ax_signal.grid(True, alpha=0.6)

    n_show = 20
    idx = np.linspace(0, states.shape[1] - 1, n_show).astype(int)
    cmap = plt.get_cmap("twilight_shifted")
    for k, j in enumerate(idx):
        ax_states.plot(states[:, j],
                       color=cmap(k / n_show), lw=0.9, alpha=0.85)
    ax_states.set_ylabel("activation")
    ax_states.set_xlabel("t")
    ax_states.grid(True, alpha=0.6)
    ax_states.set_title(
        f"{n_show} of 200 reservoir neurons reacting to the sine",
        loc="left",
    )
    save(fig, "activations_sine.png")


def main() -> None:
    apply_style()
    print("Activations …")
    activations_sine()


if __name__ == "__main__":
    main()
