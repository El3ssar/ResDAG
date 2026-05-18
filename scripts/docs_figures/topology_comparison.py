"""Side-by-side forecast quality comparison across four topologies."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from resdag import ott_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

from ._common import AMBER, GREY, INDIGO, INDIGO_LIGHT, apply_style, save
from .data import lorenz63


TOPOLOGIES = ["erdos_renyi", "watts_strogatz", "barabasi_albert", "ring_chord"]


def _forecast(topology: str) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    data = lorenz63(20_000, dt=0.02, seed=42)
    warmup, train, target, f_warmup, val = prepare_esn_data(
        data,
        warmup_steps=1_500,
        train_steps=15_000,
        val_steps=800,
        discard_steps=2_000,
        normalize=True,
        norm_method="standard",
    )
    model = ott_esn(
        reservoir_size=600,
        feedback_size=3,
        output_size=3,
        spectral_radius=0.95,
        leak_rate=1.0,
        readout_alpha=1e-7,
        topology=topology,
    )
    ESNTrainer(model).fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": target},
    )
    model.reset_reservoirs()
    pred = model.forecast(f_warmup, horizon=val.shape[1])
    truth = val[0, :, 0].numpy()
    p = pred[0, :, 0].detach().numpy()
    return truth, p


def topology_comparison() -> None:
    fig, axes = plt.subplots(len(TOPOLOGIES), 1, figsize=(7.5, 5.2),
                             sharex=True)
    for ax, name in zip(axes, TOPOLOGIES):
        truth, pred = _forecast(name)
        mse = float(((truth - pred) ** 2).mean())
        ax.plot(truth, color=GREY, lw=1.6, label="truth")
        ax.plot(pred, color=AMBER, lw=0.9, label="prediction")
        ax.set_ylabel(name, rotation=0, ha="right", va="center",
                      labelpad=70, fontsize=9.5)
        ax.grid(True, alpha=0.6)
        ax.text(0.99, 0.05, f"MSE = {mse:.2e}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8,
                color=INDIGO_LIGHT)
    axes[0].set_title("Lorenz x-component — same ESN, four topologies",
                      loc="left")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("t after f_warmup")
    save(fig, "topology_comparison.png")


def main() -> None:
    apply_style()
    print("Topology comparison …")
    topology_comparison()


if __name__ == "__main__":
    main()
