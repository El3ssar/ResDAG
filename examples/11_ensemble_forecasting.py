"""Coupled ensemble ESN: training and forecasting pipeline.

This example demonstrates the CoupledEnsembleESNModel, where N independently
trained ESN sub-models forecast in parallel and all share the same averaged
output as their feedback at each autoregressive step.

Outline
-------
1. Generate synthetic 3-D chaotic-like data (Lorenz-inspired)
2. Train a 5-model mean ensemble and forecast
3. Train a 10-model ensemble with OutliersFilteredMean aggregation
4. Show how to use a different sub-model factory (classic_esn)
5. Recover individual sub-model trajectories for post-hoc analysis
6. return_warmup flag and save/load
"""

import torch

import resdag as rd
from resdag.layers import OutliersFilteredMean
from resdag.models import classic_esn

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_lorenz_like(
    n_steps: int = 2000,
    dt: float = 0.02,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    seed: int = 42,
) -> torch.Tensor:
    """Integrate a discrete Lorenz system via Euler method.

    Returns
    -------
    torch.Tensor
        Shape ``(1, n_steps, 3)``.
    """
    torch.manual_seed(seed)
    xyz = torch.zeros(n_steps, 3)
    xyz[0] = torch.tensor([1.0, 0.0, 0.0])
    for t in range(1, n_steps):
        x, y, z = xyz[t - 1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        xyz[t] = xyz[t - 1] + dt * torch.stack([dx, dy, dz])
    # Normalise to zero mean, unit std
    xyz = (xyz - xyz.mean(0)) / xyz.std(0)
    return xyz.unsqueeze(0)  # (1, n_steps, 3)


def split_data(data: torch.Tensor, warmup: int = 200, train: int = 1000, forecast: int = 300):
    """Split (1, T, F) into warmup / train / forecast-warmup / ground truth."""
    w = data[:, :warmup, :]
    tr = data[:, warmup : warmup + train, :]
    fw = data[:, warmup + train : warmup + train + warmup, :]
    gt = data[:, warmup + train + warmup : warmup + train + warmup + forecast, :]
    return w, tr, fw, gt


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


# ---------------------------------------------------------------------------
# Example 1: simple mean ensemble (5 models, default ott_esn factory)
# ---------------------------------------------------------------------------

def example_mean_ensemble(data: torch.Tensor) -> None:
    print("\n" + "=" * 60)
    print("Example 1: Coupled mean ensemble (5 × ott_esn)")
    print("=" * 60)

    warmup, train, forecast_warmup, ground_truth = split_data(data)
    targets = train.clone()

    # All factory-specific args go through **model_kwargs
    ensemble = rd.coupled_ensemble_esn(
        n_models=5,
        reservoir_size=200,
        feedback_size=3,
        output_size=3,
        spectral_radius=0.9,
        readout_alpha=1e-6,
    )
    print(f"Ensemble: {ensemble.n_models} sub-models")

    # One ESNTrainer.fit() per sub-model, all on the same data
    ensemble.fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": targets},
    )

    ensemble.reset_reservoirs()
    preds = ensemble.forecast(forecast_warmup, horizon=ground_truth.shape[1])

    print(f"Prediction shape : {preds.shape}")
    print(f"MSE              : {mse(preds, ground_truth):.6f}")


# ---------------------------------------------------------------------------
# Example 2: OutliersFilteredMean aggregation (10 models)
# ---------------------------------------------------------------------------

def example_filtered_mean_ensemble(data: torch.Tensor) -> None:
    print("\n" + "=" * 60)
    print("Example 2: Coupled ensemble with OutliersFilteredMean (10 × ott_esn)")
    print("=" * 60)

    warmup, train, forecast_warmup, ground_truth = split_data(data)
    targets = train.clone()

    aggregator = OutliersFilteredMean(method="z_score", threshold=2.0)
    ensemble = rd.coupled_ensemble_esn(
        n_models=10,
        aggregate=aggregator,
        reservoir_size=200,
        feedback_size=3,
        output_size=3,
        spectral_radius=0.9,
        readout_alpha=1e-6,
    )
    print(f"Ensemble: {ensemble.n_models} sub-models, aggregator: OutliersFilteredMean")

    ensemble.fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": targets},
    )

    ensemble.reset_reservoirs()
    preds = ensemble.forecast(forecast_warmup, horizon=ground_truth.shape[1])

    print(f"Prediction shape : {preds.shape}")
    print(f"MSE              : {mse(preds, ground_truth):.6f}")


# ---------------------------------------------------------------------------
# Example 3: different sub-model factory (classic_esn)
# ---------------------------------------------------------------------------

def example_classic_ensemble(data: torch.Tensor) -> None:
    print("\n" + "=" * 60)
    print("Example 3: Coupled ensemble using classic_esn factory (5 models)")
    print("=" * 60)

    warmup, train, forecast_warmup, ground_truth = split_data(data)
    targets = train.clone()

    ensemble = rd.coupled_ensemble_esn(
        n_models=5,
        model_factory=classic_esn,
        reservoir_size=200,
        feedback_size=3,
        output_size=3,
        spectral_radius=0.95,
        readout_alpha=1e-6,
    )
    print(f"Ensemble: {ensemble.n_models} × classic_esn sub-models")

    ensemble.fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": targets},
    )

    ensemble.reset_reservoirs()
    preds = ensemble.forecast(forecast_warmup, horizon=ground_truth.shape[1])

    print(f"Prediction shape : {preds.shape}")
    print(f"MSE              : {mse(preds, ground_truth):.6f}")


# ---------------------------------------------------------------------------
# Example 4: recovering individual sub-model trajectories
# ---------------------------------------------------------------------------

def example_individual_trajectories(data: torch.Tensor) -> None:
    print("\n" + "=" * 60)
    print("Example 4: Individual sub-model trajectories (return_individuals=True)")
    print("=" * 60)

    warmup, train, forecast_warmup, ground_truth = split_data(data)
    targets = train.clone()

    ensemble = rd.coupled_ensemble_esn(
        n_models=5,
        reservoir_size=200,
        feedback_size=3,
        output_size=3,
    )
    ensemble.fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": targets},
    )

    horizon = ground_truth.shape[1]
    ensemble.reset_reservoirs()

    # return_individuals=True: returns (aggregated, list_of_N_tensors)
    # Per-model buffers are only allocated when this flag is set
    preds, individuals = ensemble.forecast(
        forecast_warmup, horizon=horizon, return_individuals=True
    )

    print(f"Aggregated forecast shape : {preds.shape}")
    print(f"Number of individual traj : {len(individuals)}")
    print(f"Individual traj shape     : {individuals[0].shape}")

    # Spread across individual model predictions (std at each timestep)
    stacked = torch.stack(individuals, dim=0)  # (N, 1, horizon, 3)
    spread = stacked.squeeze(1).std(dim=0)     # (horizon, 3)
    print(f"Mean per-dim spread (std) : {spread.mean(dim=0).tolist()}")

    # MSE of each individual model vs ground truth
    for i, traj in enumerate(individuals):
        print(f"  Sub-model {i} MSE: {mse(traj, ground_truth):.6f}")
    print(f"  Ensemble   MSE: {mse(preds, ground_truth):.6f}")


# ---------------------------------------------------------------------------
# Example 5: return_warmup flag and save/load
# ---------------------------------------------------------------------------

def example_return_warmup_and_save(data: torch.Tensor) -> None:
    print("\n" + "=" * 60)
    print("Example 5: return_warmup=True and save/load")
    print("=" * 60)

    warmup, train, forecast_warmup, ground_truth = split_data(data)
    targets = train.clone()

    ensemble = rd.coupled_ensemble_esn(
        n_models=3,
        reservoir_size=150,
        feedback_size=3,
        output_size=3,
    )
    ensemble.fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": targets},
    )

    ensemble.reset_reservoirs()
    full = ensemble.forecast(
        forecast_warmup,
        horizon=ground_truth.shape[1],
        return_warmup=True,
    )
    w = forecast_warmup.shape[1]
    print(f"Full trajectory shape : {full.shape}")  # (1, w + horizon, 3)
    print(f"  Warmup portion      : {full[:, :w, :].shape}")
    print(f"  Forecast portion    : {full[:, w:, :].shape}")

    # return_warmup + return_individuals: warmup is only on the aggregated tensor
    ensemble.reset_reservoirs()
    full2, individuals = ensemble.forecast(
        forecast_warmup,
        horizon=ground_truth.shape[1],
        return_warmup=True,
        return_individuals=True,
    )
    print(f"With individuals — aggregated shape : {full2.shape}")
    print(f"With individuals — each traj shape  : {individuals[0].shape}")

    # Save / load
    ensemble.save("/tmp/ensemble_test.pt", include_states=True, epoch=1)
    ensemble.load("/tmp/ensemble_test.pt", load_states=True)
    print("Save/load completed successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Lorenz-like attractor data...")
    data = generate_lorenz_like(n_steps=2000)
    print(f"Data shape: {data.shape}")  # (1, 2000, 3)

    example_mean_ensemble(data)
    example_filtered_mean_ensemble(data)
    example_classic_ensemble(data)
    example_individual_trajectories(data)
    example_return_warmup_and_save(data)

    print("\n" + "=" * 60)
    print("All ensemble examples completed.")
    print("=" * 60)
