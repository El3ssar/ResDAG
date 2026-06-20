"""01 — Quickstart: the smallest end-to-end ResDAG workflow.

What it shows
-------------
1. Generate a chaotic time series (Lorenz-63)
2. Split it with ``rd.utils.prepare_esn_data``
3. Build a premade model (``classic_esn``)
4. Fit the readout algebraically with ``ESNTrainer``
5. Autoregressive forecast + MSE against ground truth

Expected runtime: ~5 s on CPU.

Run with ``--plot`` to also display the forecast (requires matplotlib,
which is optional and not needed for the core flow).
"""

import sys
import time

import torch

import resdag as rd
from resdag.training import ESNTrainer


def main() -> None:
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. Generate and split data")
    print("=" * 70)

    # rd.lorenz ships a correct RK4 Lorenz-63 integrator in the public API:
    # a (1, 2300, 3) = (batch, time, features) standardized tensor.
    data = rd.lorenz(2300, seed=42)

    # prepare_esn_data slices one long series into the five segments every
    # ESN workflow needs (val follows train, f_warmup is the tail of train):
    #   [warmup][----------train----------][----val----]
    #                          [f_warmup-]^
    warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
        data,
        warmup_steps=200,
        train_steps=1600,
        val_steps=400,
    )
    print(f"warmup   {tuple(warmup.shape)}   - synchronizes reservoir state")
    print(f"train    {tuple(train.shape)}  - readout fitting input")
    print(f"target   {tuple(target.shape)}  - train shifted by 1 step (next-step prediction)")
    print(f"f_warmup {tuple(f_warmup.shape)}   - warmup for the forecast phase")
    print(f"val      {tuple(val.shape)}   - held-out ground truth")

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Build a premade model")
    print("=" * 70)

    model = rd.classic_esn(
        reservoir_size=600,
        feedback_size=3,  # dimension of the signal fed back into the reservoir
        output_size=3,  # must equal feedback_size for autoregressive forecasting
        spectral_radius=0.8,
        readout_alpha=1e-6,  # ridge regularization of the readout
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"classic_esn: 600 neurons, {n_params} parameters (reservoir frozen)")

    # ------------------------------------------------------------------
    # 3. Train (algebraic ridge regression, no gradient descent)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. Fit the readout")
    print("=" * 70)

    t0 = time.perf_counter()
    trainer = ESNTrainer(model)
    trainer.fit(
        warmup_inputs=(warmup,),  # tuple: (feedback, driver1, ...) — here feedback only
        train_inputs=(train,),
        targets={"output": target},  # key must match the readout's name
    )
    print(f"Readout fitted in {time.perf_counter() - t0:.2f} s (single CG ridge solve)")

    # ------------------------------------------------------------------
    # 4. Forecast
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Autoregressive forecast")
    print("=" * 70)

    horizon = val.shape[1]
    predictions = model.forecast(f_warmup, horizon=horizon)  # (1, horizon, 3)
    print(f"Forecast shape: {tuple(predictions.shape)} (batch, horizon, features)")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Evaluate against ground truth")
    print("=" * 70)

    mse_short = torch.mean((predictions[:, :50] - val[:, :50]) ** 2).item()
    mse_full = torch.mean((predictions - val) ** 2).item()
    print(f"MSE first 50 steps : {mse_short:.6f}")
    print(f"MSE full {horizon} steps: {mse_full:.6f}")
    print("(Lorenz is chaotic: trajectories diverge eventually; short-horizon")
    print(" error is the meaningful metric, long-horizon should stay bounded.)")

    if "--plot" in sys.argv:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not installed - skipping plot (pip install matplotlib)")
        else:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 6))
            for d, ax in enumerate(axes):
                ax.plot(val[0, :, d], label="truth", lw=1.2)
                ax.plot(predictions[0, :, d], label="forecast", lw=1.2, ls="--")
                ax.set_ylabel(f"x{d}")
            axes[0].legend(loc="upper right")
            axes[-1].set_xlabel("forecast step")
            fig.suptitle("classic_esn forecast vs Lorenz ground truth")
            plt.tight_layout()
            plt.show()

    print("\nDone. Next: 02_premade_models.py for the full factory tour.")


if __name__ == "__main__":
    main()
