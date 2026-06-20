"""00 — The easiest API: train + forecast a chaotic system in ~10 lines.

What it shows
-------------
1. The headline one-liner: ``rd.ESN(...).fit(series).forecast(horizon=N)``
2. numpy in -> numpy out (the facade mirrors the input array type)
3. Dropping down to the underlying ``ESNModel`` when you want more control

The :class:`resdag.ESN` facade owns the whole reservoir-computing workflow —
building the ``pytorch_symbolic`` graph, slicing the series into a washout +
one-step-ahead training window, driving :class:`~resdag.training.ESNTrainer`,
and running the autoregressive forecast — so the common case is a single
expression.  For the explicit, fully composable pipeline that the facade wraps,
see ``01_quickstart.py``.

Expected runtime: ~3 s on CPU.

Run with ``--plot`` to also display the forecast (requires matplotlib, which is
optional and not needed for the core flow).
"""

import sys

import numpy as np

import resdag as rd


def generate_lorenz(n_steps: int, dt: float = 0.01, seed: int = 42) -> np.ndarray:
    """Integrate the Lorenz-63 system (Euler), returning a normalized (T, 3) array."""
    rng = np.random.default_rng(seed)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    xyz = np.empty((n_steps, 3))
    xyz[0] = np.array([1.0, 1.0, 1.05]) + 1e-3 * rng.standard_normal(3)
    for t in range(1, n_steps):
        x, y, z = xyz[t - 1]
        dxyz = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        xyz[t] = xyz[t - 1] + dt * dxyz
    return (xyz - xyz.mean(0)) / xyz.std(0)  # zero mean, unit std per feature


def main() -> None:
    # ------------------------------------------------------------------
    # The whole workflow, ~10 lines: build -> fit -> forecast.
    # ------------------------------------------------------------------
    series = generate_lorenz(2300)  # (2300, 3) numpy array of (time, features)
    train, truth = series[:1900], series[1900:2100]  # hold out the last 200 steps

    esn = rd.ESN(
        reservoir_size=600,
        spectral_radius=0.8,
        washout=200,  # steps used to synchronize the reservoir before training
        alpha=1e-6,  # ridge regularization of the readout
        seed=42,
    )
    forecast = esn.fit(train).forecast(horizon=truth.shape[0])  # numpy in -> numpy out

    # ------------------------------------------------------------------
    # Inspect the result.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("rd.ESN(...).fit(series).forecast(horizon=N)")
    print("=" * 70)
    print(f"input series : {series.shape}  ({type(series).__name__})")
    print(f"forecast     : {forecast.shape}  ({type(forecast).__name__})")
    mse_short = float(np.mean((forecast[:50] - truth[:50]) ** 2))
    print(f"MSE first 50 steps: {mse_short:.6f}")
    print("(Lorenz is chaotic — short-horizon error is the meaningful metric.)")

    # ------------------------------------------------------------------
    # The facade does not hide anything: the composed ESNModel is reachable
    # via ``esn.model`` for the full building-block API (save_full, custom
    # state management, summary(), plot_model(), ...).
    # ------------------------------------------------------------------
    print("\nUnderlying model stays accessible via `esn.model`:")
    print(f"  type            : {type(esn.model).__name__}")
    print(f"  reservoir states: {list(esn.model.get_reservoir_states())}")

    if "--plot" in sys.argv:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not installed - skipping plot (pip install matplotlib)")
        else:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 6))
            for d, ax in enumerate(axes):
                ax.plot(truth[:, d], label="truth", lw=1.2)
                ax.plot(forecast[:, d], label="forecast", lw=1.2, ls="--")
                ax.set_ylabel(f"x{d}")
            axes[0].legend(loc="upper right")
            axes[-1].set_xlabel("forecast step")
            fig.suptitle("rd.ESN facade forecast vs Lorenz ground truth")
            plt.tight_layout()
            plt.show()

    print("\nDone. Next: 01_quickstart.py for the explicit, composable pipeline.")


if __name__ == "__main__":
    main()
