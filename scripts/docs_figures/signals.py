"""Plain timeseries plots (input only, no model)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ._common import AMBER, INDIGO, INDIGO_LIGHT, RULE, apply_style, save
from .data import lorenz63, mackey_glass, sine_wave


def signal_sine() -> None:
    y = sine_wave(400, periods=4.0)[0, :, 0].numpy()
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.plot(y, color=INDIGO, lw=1.4)
    ax.set_title("Sine input — 4 periods over 400 timesteps")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.grid(True, alpha=0.6)
    save(fig, "signal_sine.png")


def signal_lorenz() -> None:
    xyz = lorenz63(3000)[0].numpy()
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 4.2), sharex=True)
    labels = ["x", "y", "z"]
    colors = [INDIGO, INDIGO_LIGHT, AMBER]
    for ax, label, color, series in zip(axes, labels, colors, xyz.T):
        ax.plot(series, color=color, lw=0.9)
        ax.set_ylabel(label, rotation=0, va="center", labelpad=12)
        ax.grid(True, alpha=0.6)
    axes[0].set_title("Lorenz-63 — chaotic 3-D series (normalised)")
    axes[-1].set_xlabel("t")
    save(fig, "signal_lorenz.png")


def signal_lorenz_phase() -> None:
    xyz = lorenz63(3000)[0].numpy()
    fig = plt.figure(figsize=(4.6, 4.0))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=0.5, color=INDIGO)
    ax3d.set_title("Lorenz attractor — phase portrait")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.grid(False)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis._axinfo["grid"]["color"] = RULE  # type: ignore[attr-defined]
    save(fig, "signal_lorenz_phase.png")


def signal_mackey_glass() -> None:
    y = mackey_glass(4000)[0, 1000:3000, 0].numpy()
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.plot(y, color=INDIGO, lw=0.9)
    ax.set_title("Mackey–Glass τ=17 — quasi-periodic chaotic series")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.grid(True, alpha=0.6)
    save(fig, "signal_mackey_glass.png")


def main() -> None:
    apply_style()
    print("Signals …")
    signal_sine()
    signal_lorenz()
    signal_lorenz_phase()
    signal_mackey_glass()


if __name__ == "__main__":
    main()
