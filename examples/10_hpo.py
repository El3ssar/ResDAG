"""10 — HPO: a minimal Optuna study with ``run_hpo``.

The HPO loop is driven by three user callbacks:

    model_creator(**hparams) -> ESNModel   fresh model per trial
    search_space(trial)      -> dict       optuna suggest_* calls
    data_loader(trial)       -> dict       warmup/train/target/f_warmup/val

Each trial: build model -> fit readout -> forecast over the validation
window -> score with the chosen loss (default "efh", Expected Forecast
Horizon — tailored to chaotic systems).

The three callbacks are defined at **module level on purpose**. When you scale
up to ``n_workers > 1`` (or run any CUDA study), ``run_hpo`` distributes trials
with the ``spawn`` start method, which pickles the objective — so the callbacks
must be top-level, importable functions. Closures defined inside ``main()``
would raise ``PicklingError`` there. Single-worker studies have no such
restriction, but writing them top-level from the start keeps the example
copy-paste-ready for parallel runs.

This example is intentionally tiny (8 trials, small reservoir) so it runs
fast; comments point to the knobs you would scale up.

Expected runtime: ~10 s on CPU. Requires the hpo extra:
``pip install resdag[hpo]`` or ``uv sync --extra hpo``.
"""

from functools import partial

import torch

import resdag as rd

SEED = 42


# ----------------------------------------------------------------------
# The three callbacks — top-level so they pickle for the spawn/multi-worker path
# ----------------------------------------------------------------------


def model_creator(reservoir_size: int, spectral_radius: float, leak_rate: float, seed=None):
    """One fresh model per trial. Must accept every key of search_space."""
    return rd.classic_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        seed=seed,
    )


def search_space(trial, *, quick: bool = False) -> dict:
    """Maps optuna suggestions onto model_creator's parameters."""
    return {
        "reservoir_size": trial.suggest_int(
            "reservoir_size", 16 if quick else 100, 32 if quick else 300, step=16 if quick else 100
        ),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
        "leak_rate": trial.suggest_float("leak_rate", 0.2, 1.0),
    }


def load_splits(*, quick: bool = False):
    """Prepare the canonical Lorenz splits (rebuilt per worker under spawn)."""
    data = rd.lorenz(141 if quick else 2300, seed=SEED, normalize=None)  # raw (1, 2300, 3)
    return rd.data.prepare_esn_data(
        data,
        warmup_steps=12 if quick else 200,
        train_steps=100 if quick else 1600,
        val_steps=24 if quick else 300,
        normalize=True,
        norm_method="standard",
    )


def data_loader(trial, *, quick: bool = False) -> dict:
    """Required keys: warmup, train, target, f_warmup, val (all (B, T, D))."""
    warmup, train, target, f_warmup, val = load_splits(quick=quick)
    return {
        "warmup": warmup,
        "train": train,
        "target": target,
        "f_warmup": f_warmup,
        "val": val,
    }


def main() -> None:
    from _example_options import parse_options

    options = parse_options(__doc__)
    n_trials = options.choose(2, 8)
    try:
        from resdag.hpo import get_study_summary, run_hpo
    except ImportError:
        print("optuna is not installed — this example needs the hpo extra:")
        print("  pip install resdag[hpo]   (or: uv sync --extra hpo)")
        return

    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # Run the study
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"Running a tiny study: {n_trials} trials, single worker, in-memory storage")
    print("=" * 70)

    study = run_hpo(
        model_creator=model_creator,
        search_space=partial(search_space, quick=options.quick),
        data_loader=partial(data_loader, quick=options.quick),
        n_trials=n_trials,  # scale up (100+) for real searches
        loss="efh",  # also: "forecast_horizon", "lyapunov", "standard", "soft_horizon"
        seed=SEED,  # seeds the sampler and per-trial RNG
        verbosity=0,  # 1 prints per-trial progress
        # For real runs (the callbacks above are already spawn-safe/top-level):
        #   n_workers=4,           parallel OS processes (spawn if CUDA)
        #   storage="study.log",   journal file -> resumable + multi-worker
        #   monitor_losses=["forecast_horizon"],  log extra metrics without optimizing them
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print(get_study_summary(study))
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # The best configuration is just a model_creator call away:
    assert len(study.trials) == n_trials
    warmup, train, target, f_warmup, val = load_splits(quick=options.quick)
    best_model = model_creator(**study.best_params, seed=study.best_trial.user_attrs["model_seed"])
    rd.ESNTrainer(best_model).fit((warmup,), (train,), {"output": target})
    preds = best_model.forecast(f_warmup, horizon=val.shape[1])
    assert preds.shape == val.shape and torch.isfinite(preds).all()
    mse50 = torch.mean((preds[:, :50] - val[:, :50]) ** 2).item()
    print(f"\nRefit best model -> forecast MSE@50 = {mse50:.6f}")
    print("Note: 'efh' is maximized internally as a negative loss; lower study")
    print("values = longer usable forecast horizons.")


if __name__ == "__main__":
    main()
