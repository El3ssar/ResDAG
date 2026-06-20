"""02 — Premade models: every factory in ``resdag.models``.

What it shows
-------------
1. The six factories: classic_esn, ott_esn, power_augmented, linear_esn,
   headless_esn, coupled_ensemble_esn
2. What each architecture computes (readout models vs state extractors)
3. A single comparison table: parameters + short-horizon forecast MSE
   on the same Lorenz-63 task

Expected runtime: ~5 s on CPU.
"""

import time

import torch

import resdag as rd
from resdag.training import ESNTrainer


SIZE = 400  # reservoir size shared by all models in the comparison


def main() -> None:
    torch.manual_seed(42)

    data = rd.lorenz(2300, seed=42)  # (1, 2300, 3) = (batch, time, features)
    warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
        data, warmup_steps=200, train_steps=1600, val_steps=400
    )

    # ------------------------------------------------------------------
    # 1. Forecasting factories (reservoir + trained readout)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. Forecasting factories")
    print("=" * 70)
    print(
        """
classic_esn     Input -> Reservoir -> Concat(Input, States) -> CGReadout
ott_esn         ... -> SelectiveExponentiation (square even units) -> ...
                recommended starting point for chaotic systems
power_augmented generalization of ott_esn: states**exponent branch
"""
    )

    # Identical data + readout settings; only the architecture differs.
    forecasters = {
        "classic_esn": lambda: rd.classic_esn(
            SIZE, feedback_size=3, output_size=3, spectral_radius=0.8
        ),
        "ott_esn": lambda: rd.ott_esn(SIZE, feedback_size=3, output_size=3, spectral_radius=0.8),
        "power_augmented": lambda: rd.power_augmented(
            SIZE, feedback_size=3, output_size=3, exponent=3.0, spectral_radius=0.8
        ),
    }

    results: list[tuple[str, int, float, float]] = []  # name, params, fit_s, mse50

    for name, build in forecasters.items():
        torch.manual_seed(42)  # same reservoir draw for a fair comparison
        model = build()
        n_params = sum(p.numel() for p in model.parameters())

        t0 = time.perf_counter()
        ESNTrainer(model).fit((warmup,), (train,), {"output": target})
        fit_s = time.perf_counter() - t0

        preds = model.forecast(f_warmup, horizon=val.shape[1])  # (1, 400, 3)
        mse50 = torch.mean((preds[:, :50] - val[:, :50]) ** 2).item()
        results.append((name, n_params, fit_s, mse50))
        print(f"{name:<16} trained, forecast shape {tuple(preds.shape)}")

    # ------------------------------------------------------------------
    # 2. Coupled ensemble (N sub-models, shared aggregated feedback)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. coupled_ensemble_esn")
    print("=" * 70)
    print("N independently initialized sub-models forecast together; at every")
    print("autoregressive step all of them get the aggregated (mean) output as")
    print("feedback. See 07_ensembles.py for the full treatment.")

    ensemble = rd.coupled_ensemble_esn(
        n_models=4,
        model_factory=rd.classic_esn,
        seed=42,  # deterministic sub-model initialization
        reservoir_size=SIZE,
        feedback_size=3,
        output_size=3,
        spectral_radius=0.8,
    )
    t0 = time.perf_counter()
    ensemble.fit((warmup,), (train,), {"output": target})
    fit_s = time.perf_counter() - t0
    preds = ensemble.forecast(f_warmup, horizon=val.shape[1])
    mse50 = torch.mean((preds[:, :50] - val[:, :50]) ** 2).item()
    n_params = sum(p.numel() for p in ensemble.parameters())
    results.append(("coupled_ensemble (4x)", n_params, fit_s, mse50))
    print(f"\nEnsemble of {ensemble.n_models} sub-models, forecast shape {tuple(preds.shape)}")

    # ------------------------------------------------------------------
    # 3. State extractors (no readout — they output reservoir states)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. State extractors: headless_esn and linear_esn")
    print("=" * 70)

    x = torch.randn(4, 100, 3)  # (batch, time, features)

    headless = rd.headless_esn(SIZE, feedback_size=3)
    states = headless(x)  # (4, 100, SIZE) — raw tanh reservoir states
    print(f"headless_esn : input {tuple(x.shape)} -> states {tuple(states.shape)}")

    linear = rd.linear_esn(SIZE, feedback_size=3)
    lin_states = linear(x)  # identity activation — linear dynamics baseline
    print(f"linear_esn   : input {tuple(x.shape)} -> states {tuple(lin_states.shape)}")
    print("Use cases: reservoir dynamics analysis, custom heads (see 05), ESP checks.")

    # ------------------------------------------------------------------
    # 4. Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Comparison (same data, same reservoir size, same readout alpha)")
    print("=" * 70)
    header = f"{'model':<22} {'params':>10} {'fit [s]':>8} {'MSE@50 steps':>14}"
    print(header)
    print("-" * len(header))
    for name, n_params, fit_s, mse50 in results:
        print(f"{name:<22} {n_params:>10,} {fit_s:>8.2f} {mse50:>14.6f}")
    print("-" * len(header))
    print("MSE@50 = mean squared error over the first 50 forecast steps.")
    print("Rankings shift with hyperparameters — use 10_hpo.py to tune properly.")


if __name__ == "__main__":
    main()
