"""10 — HPO: a minimal Optuna study with ``run_hpo``.

The HPO loop is driven by three user callbacks:

    model_creator(**hparams) -> ESNModel   fresh model per trial
    search_space(trial)      -> dict       optuna suggest_* calls
    data_loader(trial)       -> dict       warmup/train/target/f_warmup/val

Each trial: build model -> fit readout -> forecast over the validation
window -> score with the chosen loss (default "efh", Expected Forecast
Horizon — tailored to chaotic systems).

This example is intentionally tiny (8 trials, small reservoir) so it runs
fast; comments point to the knobs you would scale up.

Expected runtime: ~10 s on CPU. Requires the hpo extra:
``pip install resdag[hpo]`` or ``uv sync --extra hpo``.
"""

import torch

import resdag as rd


def main() -> None:
    try:
        from resdag.hpo import get_study_summary, run_hpo
    except ImportError:
        print("optuna is not installed — this example needs the hpo extra:")
        print("  pip install resdag[hpo]   (or: uv sync --extra hpo)")
        return

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. The three callbacks
    # ------------------------------------------------------------------

    def model_creator(reservoir_size: int, spectral_radius: float, leak_rate: float):
        """One fresh model per trial. Must accept every key of search_space."""
        return rd.classic_esn(
            reservoir_size=reservoir_size,
            feedback_size=3,
            output_size=3,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
        )

    def search_space(trial) -> dict:
        """Maps optuna suggestions onto model_creator's parameters."""
        return {
            "reservoir_size": trial.suggest_int("reservoir_size", 100, 300, step=100),
            "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
            "leak_rate": trial.suggest_float("leak_rate", 0.2, 1.0),
        }

    # Data is loaded per trial (lets you subsample / augment per trial).
    data = rd.lorenz(2300, seed=42)  # (1, 2300, 3)
    warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
        data, warmup_steps=200, train_steps=1600, val_steps=300
    )

    def data_loader(trial) -> dict:
        """Required keys: warmup, train, target, f_warmup, val (all (B, T, D))."""
        return {
            "warmup": warmup,
            "train": train,
            "target": target,
            "f_warmup": f_warmup,
            "val": val,
        }

    # ------------------------------------------------------------------
    # 2. Run the study
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Running a tiny study: 8 trials, single worker, in-memory storage")
    print("=" * 70)

    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=8,  # scale up (100+) for real searches
        loss="efh",  # also: "forecast_horizon", "lyapunov", "standard", "soft_horizon"
        seed=42,  # seeds the sampler and per-trial RNG
        verbosity=0,  # 1 prints per-trial progress
        # For real runs:
        #   n_workers=4,           parallel OS processes
        #   storage="study.log",   journal file -> resumable + multi-worker
        #   monitor_losses=["forecast_horizon"],  log extra metrics without optimizing them
    )

    # ------------------------------------------------------------------
    # 3. Results
    # ------------------------------------------------------------------
    print(get_study_summary(study))
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # The best configuration is just a model_creator call away:
    best_model = model_creator(**study.best_params)
    rd.ESNTrainer(best_model).fit((warmup,), (train,), {"output": target})
    preds = best_model.forecast(f_warmup, horizon=val.shape[1])
    mse50 = torch.mean((preds[:, :50] - val[:, :50]) ** 2).item()
    print(f"\nRefit best model -> forecast MSE@50 = {mse50:.6f}")
    print("Note: 'efh' is maximized internally as a negative loss; lower study")
    print("values = longer usable forecast horizons.")


if __name__ == "__main__":
    main()
