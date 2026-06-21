"""13 — Windowed forecast: gap-filling reconstruction of a sparse trajectory.

``ESNModel.windowed_forecast()`` reconstructs a long trajectory that is only
*observed* in short, periodic windows.  Each cycle re-synchronizes the
reservoir on a teacher-forced window of real data, then lets the model
free-run autonomously across the unobserved gap::

    [== warmup ==][~~ gap ~~][= teacher =][~~ gap ~~][= teacher =] ...
      observed       filled     observed      filled      observed
    (teacher-forced)(forecast)(teacher-forced)(forecast)

Because the Echo State Property gives the reservoir fading memory, every
teacher-forced window pulls its state back onto the true trajectory, so
per-window error does not compound across windows — only *within* a gap
(keep ``predict_len`` short relative to the system's predictability horizon).

What it shows
-------------
1. Train an Ott ESN on the chaotic Lorenz-63 attractor
2. Reconstruct a held-out stretch observed only in brief windows (30 of
   every 180 steps in steady state; ~25% overall, incl. the initial sync)
3. Score ONLY the unseen gaps: ``recon[:, ~mask]`` vs ``series[:, ~mask]``
4. Observed segments are copied verbatim; gaps are genuine forecasts

Expected runtime: ~10 s on CPU.

Run with ``--plot`` to visualize the reconstruction (matplotlib optional).
"""

import sys

import torch

import resdag as rd
from resdag.training import ESNTrainer


def lorenz_series(
    n_steps: int,
    dt: float = 0.02,
    transient: int = 1000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> torch.Tensor:
    """Integrate Lorenz-63 with RK4 and return a normalized ``(1, n_steps, 3)`` series."""

    def deriv(state: torch.Tensor) -> torch.Tensor:
        x, y, z = state[0], state[1], state[2]
        return torch.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    state = torch.tensor([1.0, 1.0, 1.0])
    traj = torch.empty(transient + n_steps, 3)
    for i in range(transient + n_steps):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i] = state

    traj = traj[transient:]  # discard the transient onto the attractor
    traj = (traj - traj.mean(dim=0)) / traj.std(dim=0)  # standardize per channel
    return traj.reshape(1, n_steps, 3)


def main() -> None:
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. Train an Ott ESN on Lorenz-63
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. Train an Ott ESN on the Lorenz attractor")
    print("=" * 70)

    data = lorenz_series(n_steps=6000)  # (1, 6000, 3)
    warmup = data[:, :200]  # sync the reservoir
    train_in = data[:, 200:4000]
    train_tgt = data[:, 201:4001]  # next-step target

    model = rd.ott_esn(reservoir_size=400, feedback_size=3, output_size=3, spectral_radius=0.9)
    ESNTrainer(model).fit((warmup,), (train_in,), {"output": train_tgt})
    print(f"trained on {train_in.shape[1]} steps; reservoir_size=400")

    # ------------------------------------------------------------------
    # 2. Gap-filling reconstruction of a held-out stretch
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. windowed_forecast: reconstruct a sparsely-observed stretch")
    print("=" * 70)

    series = data[:, 4000:6000]  # (1, 2000, 3) held out from training
    recon, mask = model.windowed_forecast(
        series,
        warmup_len=200,  # initial re-sync window
        teacher_len=30,  # real steps teacher-forced to re-sync each cycle
        predict_len=150,  # autonomous steps per gap (~3 Lyapunov times)
        return_mask=True,
    )

    observed = int(mask.sum())
    forecast_steps = int((~mask).sum())
    duty = observed / mask.numel()
    print(f"series {tuple(series.shape)}: {observed} observed, {forecast_steps} forecast")
    print(f"only {duty:.0%} of the trajectory was ever shown to the model")

    # ------------------------------------------------------------------
    # 3. Score ONLY the gaps the model never saw
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. Scoring the unseen gaps")
    print("=" * 70)

    # Observed segments are copied verbatim, so recon[:, mask] == series[:, mask].
    assert torch.equal(recon[:, mask, :], series[:, mask, :])

    gap_rmse = ((recon[:, ~mask, :] - series[:, ~mask, :]) ** 2).mean().sqrt().item()
    # Baseline: predict the per-channel mean (std == 1 per channel after standardizing).
    naive_rmse = ((series[:, ~mask, :] - series.mean(dim=(0, 1))) ** 2).mean().sqrt().item()
    print("score the gaps: recon[:, ~mask] vs series[:, ~mask]")
    print(f"gap RMSE over {forecast_steps} reconstructed steps: {gap_rmse:.3f}")
    print(f"   (naive 'predict the mean' baseline RMSE:        {naive_rmse:.3f})")
    print("Phase tracking decays across each gap, but the reconstruction stays on")
    print("the attractor — sparse sampling rebuilds the SHAPE long after the exact")
    print("phase is lost. For driven models, pass one full-timeline driver per")
    print("driver positionally: model.windowed_forecast(series, driver, ...).")

    if "--plot" in sys.argv:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not installed - skipping plot")
        else:
            t = torch.arange(series.shape[1])
            fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
            for c, ax in enumerate(axes):
                ax.plot(t, series[0, :, c], color="0.7", lw=2, label="truth")
                ax.plot(t, torch.where(mask, recon[0, :, c], torch.nan), label="observed", lw=1.2)
                ax.plot(
                    t,
                    torch.where(~mask, recon[0, :, c], torch.nan),
                    "--",
                    label="forecast (gaps)",
                    lw=1.2,
                )
                ax.set_ylabel("xyz"[c])
            axes[0].legend(loc="upper right", ncol=3)
            axes[0].set_title("Lorenz windowed reconstruction: observed windows + forecast gaps")
            axes[-1].set_xlabel("step")
            plt.tight_layout()
            plt.show()

    print("\nDone. Two-phase forecasting basics: 06_forecasting.py")


if __name__ == "__main__":
    main()
