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


def main() -> None:
    # ------------------------------------------------------------------
    # The whole workflow, ~10 lines: build -> fit -> forecast.
    # ------------------------------------------------------------------
    # rd.lorenz returns a (1, T, 3) tensor; the facade is happy with numpy too,
    # so we drop the batch axis and hand it a plain (T, 3) numpy series.
    series = rd.lorenz(2300, seed=42).squeeze(0).numpy()  # (2300, 3) (time, features)
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
