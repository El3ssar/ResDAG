"""Compare vanilla, OTT and custom DAG forecasts on a TSDynamics trajectory.

Install ``resdag[dynamics,viz]`` and the Graphviz system executable. For a
dependency-light diagram use ``--diagram-format dot``. The defaults illustrate
the workflow; they are not tuned recipes or a claim of seven Lyapunov times.

    python examples/17_forecast_experiment.py --system lorenz
    python examples/17_forecast_experiment.py --system rossler --reservoir-size 400
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import resdag as rd


def build_models(reservoir_size: int, seed: int, washout: int) -> dict:
    """Keep the total reservoir width equal across three example architectures."""
    if reservoir_size < 2:
        raise ValueError("reservoir_size must be at least two")
    models = {
        "vanilla": rd.ESN(
            reservoir_size=reservoir_size,
            input_size=3,
            washout=washout,
            readout="svd",
            seed=seed,
            dtype=torch.float64,
        ),
        "ott": rd.Model(
            graph=rd.ott_esn(reservoir_size, 3, 3, readout="svd", seed=seed, dtype=torch.float64),
            washout=washout,
        ),
    }
    x = rd.reservoir_input(3, dtype=torch.float64)
    fast_width = reservoir_size // 2
    fast = rd.ESNLayer(
        fast_width,
        feedback_size=3,
        spectral_radius=0.9,
        leak_rate=1.0,
        seed=seed,
        dtype=torch.float64,
    )(x)
    slow = rd.ESNLayer(
        reservoir_size - fast_width,
        feedback_size=3,
        spectral_radius=0.9,
        leak_rate=0.3,
        seed=seed + 1,
        dtype=torch.float64,
    )(x)
    features = rd.Concatenate()(x, fast, slow)
    y = rd.SVDReadoutLayer(reservoir_size + 3, 3, alpha=1e-6).double()(features)
    models["custom"] = rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=washout)
    return models


def forecast_scores(
    prediction: np.ndarray,
    truth: np.ndarray,
    training: np.ndarray,
    *,
    washout: int,
    dt: float,
    lambda_max: float,
    threshold: float = 0.3,
) -> dict:
    """Score the uninterrupted valid prefix using training-only component scales."""
    if prediction.shape != truth.shape or prediction.ndim != 2 or not len(truth):
        raise ValueError("prediction and truth must share a nonempty (time, signals) shape")
    scale = training[washout + 1 :].std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    error = np.sqrt(np.mean(((prediction - truth) / scale) ** 2, axis=-1))
    valid = np.isfinite(error) & (error < threshold)
    steps = len(valid) if valid.all() else int(np.argmax(~valid))
    secondary_valid = np.isfinite(error) & (error < 0.5)
    secondary_steps = len(error) if secondary_valid.all() else int(np.argmax(~secondary_valid))
    return {
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "valid_steps": steps,
        "valid_lyapunov_times": steps * dt * lambda_max,
        "reached_horizon": bool(valid.all()),
        "secondary_valid_lyapunov_times_at_0_5": secondary_steps * dt * lambda_max,
        "secondary_reached_horizon": bool(secondary_valid.all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("lorenz", "rossler"), default="lorenz")
    parser.add_argument("--reservoir-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--training", type=int, default=4000)
    parser.add_argument("--washout", type=int, default=200)
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--diagram-format", choices=("svg", "dot"), default="svg")
    parser.add_argument("--output", type=Path, default=Path("forecast-experiment"))
    args = parser.parse_args()
    if args.training <= args.washout + 2 or not 0 < args.history <= args.training:
        parser.error("training must exceed washout + 2; history must be within training")
    if args.threads < 1:
        parser.error("threads must be positive")
    try:
        import tsdynamics as ts
    except ImportError as exc:
        raise ImportError("Install resdag[dynamics] on Python 3.12 or newer") from exc

    # These constants apply to the canonical parameter values below only.
    if args.system == "lorenz":
        system, default_dt, lambda_max = ts.systems.Lorenz(), 0.02, 0.9056
    else:
        system = ts.systems.Rossler(a=0.2, b=0.2, c=5.7)
        default_dt, lambda_max = 0.05, 0.0714
    dt = default_dt if args.dt is None else args.dt
    if not np.isfinite(dt) or dt <= 0:
        parser.error("dt must be positive and finite")
    horizon = math.ceil(10 / (dt * lambda_max)) if args.horizon is None else args.horizon
    if horizon < 1:
        parser.error("horizon must be positive")
    torch.set_num_threads(args.threads)
    trajectory = system.run(
        final_time=(args.training + horizon) * dt,
        dt=dt,
        transient=100.0,
        solver="dop853",
        rtol=1e-10,
        atol=1e-12,
    )
    training = trajectory[: args.training]
    truth = trajectory[args.training : args.training + horizon]
    actual_dt = float(training.t[1] - training.t[0])
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for name, model in build_models(args.reservoir_size, args.seed, args.washout).items():
        start = perf_counter()
        model.fit(training)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        prediction = model.forecast(history=training[-args.history :], horizon=horizon)
        forecast_seconds = perf_counter() - start
        np.testing.assert_allclose(prediction.t, truth.t, rtol=0, atol=1e-10)
        result = {
            "model": name,
            "fit_seconds": fit_seconds,
            "forecast_seconds": forecast_seconds,
            **forecast_scores(
                prediction.y,
                truth.y,
                training.y,
                washout=args.washout,
                dt=actual_dt,
                lambda_max=lambda_max,
            ),
        }
        report.append(result)
        print(json.dumps(result))
        model.save(args.output / f"{name}.{args.diagram_format}")
        model.save(args.output / f"{name}.pt")
        restored = rd.load(args.output / f"{name}.pt")
        restored_prediction = restored.forecast(history=training[-args.history :], horizon=horizon)
        np.testing.assert_allclose(restored_prediction.y, prediction.y, rtol=0, atol=0)
    (args.output / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Untuned single-seed examples. reached_horizon means a lower bound, not the true limit.")
    print("Compare candidates on validation data; keep a separate final test segment untouched.")


if __name__ == "__main__":
    main()
