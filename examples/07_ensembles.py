"""07 — Coupled ensembles: N reservoirs, one shared feedback signal.

``coupled_ensemble_esn`` builds N independently initialized sub-models.
They are trained independently on the same data, but during forecasting
every sub-model receives the same AGGREGATED output (mean / median /
custom) as its next feedback — the ensemble is coupled through that
shared signal, which suppresses the error growth of any single member.

What it shows
-------------
1. Build + fit + forecast a mean-coupled ensemble
2. Single model vs ensemble accuracy on the same task
3. Aggregators: "mean", "median", OutliersFilteredMean
4. return_individuals: inspecting per-member trajectories and spread

Expected runtime: ~10 s on CPU.
"""

import torch

import resdag as rd
from resdag.ensemble.aggregators import OutliersFilteredMean


def mse(pred: torch.Tensor, truth: torch.Tensor, steps: int) -> float:
    return torch.mean((pred[:, :steps] - truth[:, :steps]) ** 2).item()


def main() -> None:
    torch.manual_seed(42)

    data = rd.lorenz(2300, seed=42)  # (1, 2300, 3) = (batch, time, features)
    warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
        data, warmup_steps=200, train_steps=1600, val_steps=400
    )
    horizon = val.shape[1]
    model_kwargs = dict(reservoir_size=300, feedback_size=3, output_size=3, spectral_radius=0.8)

    # ------------------------------------------------------------------
    # 1. Build, fit, forecast
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. Mean-coupled ensemble (5 x classic_esn)")
    print("=" * 70)

    ensemble = rd.coupled_ensemble_esn(
        n_models=5,
        model_factory=rd.classic_esn,  # any premade factory (default: ott_esn)
        aggregate="mean",
        seed=42,  # sub-model i is built under torch.manual_seed(seed + i)
        **model_kwargs,
    )
    print(
        f"{ensemble.n_models} sub-models, "
        f"{sum(p.numel() for p in ensemble.parameters()):,} total parameters"
    )

    # Same fit() signature as ESNTrainer; every sub-model sees the same data.
    # Diversity comes purely from the different reservoir initializations.
    ensemble.fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": target},
        n_workers=1,  # >1 fits sub-models in a thread pool (CPU only)
    )

    ensemble.reset_reservoirs()
    ens_preds = ensemble.forecast(f_warmup, horizon=horizon)  # (1, 400, 3)
    print(f"forecast shape: {tuple(ens_preds.shape)}")

    # ------------------------------------------------------------------
    # 2. Single model vs ensemble
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Single model vs ensemble")
    print("=" * 70)

    torch.manual_seed(42)
    single = rd.classic_esn(**model_kwargs)
    rd.ESNTrainer(single).fit((warmup,), (train,), {"output": target})
    single_preds = single.forecast(f_warmup, horizon=horizon)

    header = f"{'model':<22} {'MSE@50':>12} {'MSE@200':>12}"
    print(header)
    print("-" * len(header))
    print(
        f"{'single classic_esn':<22} {mse(single_preds, val, 50):>12.6f} "
        f"{mse(single_preds, val, 200):>12.6f}"
    )
    print(
        f"{'5x mean ensemble':<22} {mse(ens_preds, val, 50):>12.6f} "
        f"{mse(ens_preds, val, 200):>12.6f}"
    )

    # ------------------------------------------------------------------
    # 3. Aggregators
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. Aggregation strategies")
    print("=" * 70)
    print('"mean" | "median" | any nn.Module mapping (N, B, T, F) -> (B, T, F)')

    aggregators = {
        "median": "median",
        "outliers_filtered": OutliersFilteredMean(method="z_score", threshold=2.0),
    }
    for label, agg in aggregators.items():
        ens = rd.coupled_ensemble_esn(
            n_models=5,
            model_factory=rd.classic_esn,
            aggregate=agg,
            seed=42,
            **model_kwargs,
        )
        ens.fit((warmup,), (train,), {"output": target})
        ens.reset_reservoirs()
        preds = ens.forecast(f_warmup, horizon=horizon)
        print(f"{label:<22} MSE@50 = {mse(preds, val, 50):.6f}")
    print("OutliersFilteredMean drops members whose output norm is an outlier")
    print("at each timestep — robust when a member occasionally diverges.")

    # ------------------------------------------------------------------
    # 4. Individual trajectories
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. return_individuals: per-member trajectories")
    print("=" * 70)

    ensemble.reset_reservoirs()
    agg_preds, individuals = ensemble.forecast(f_warmup, horizon=horizon, return_individuals=True)
    stacked = torch.stack(individuals)  # (N, batch, horizon, features)
    spread = stacked.std(dim=0).mean(dim=(0, 2))  # (horizon,) mean std across members

    print(
        f"aggregated: {tuple(agg_preds.shape)}, individuals: "
        f"{len(individuals)} x {tuple(individuals[0].shape)}"
    )
    print("Ensemble spread (std across members) grows with the horizon and is")
    print("a free uncertainty proxy:")
    for step in (0, 50, 100, 200, horizon - 1):
        print(f"  step {step:>3}: spread = {spread[step].item():.4f}")

    print("\nDone. Persisting models (incl. ensembles): 08_save_load.py")


if __name__ == "__main__":
    main()
