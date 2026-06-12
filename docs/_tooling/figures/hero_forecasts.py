"""Generate true-vs-ESN-forecast trajectory pairs for the docs hero animation.

For each attractor in the catalog (systems ported from TSDynamics):
integrate with RK4, normalize, train an ott_esn on the first stretch, then
autoregressively forecast. Writes docs/assets/forecasts/<name>.json with the
true continuation and the model's forecast over the same window — the hero
canvas animates the pair in 3D.

Run from the repo root:  uv run python docs/_tooling/figures/hero_forecasts.py
"""

import json
import math
from pathlib import Path

import torch

import resdag as rd
from resdag.training import ESNTrainer

OUT = Path(__file__).resolve().parents[3] / "docs" / "assets" / "forecasts"

# name -> (rhs, initial_state, dt, transient_skip)
SYSTEMS = {
    "lorenz": (
        lambda s: [10 * (s[1] - s[0]), 28 * s[0] - s[0] * s[2] - s[1], s[0] * s[1] - 8 / 3 * s[2]],
        [1.0, 1.0, 25.0], 0.01, 500),
    "rossler": (
        lambda s: [-s[1] - s[2], s[0] + 0.2 * s[1], 0.2 + s[2] * (s[0] - 5.7)],
        [1.0, 1.0, 1.0], 0.04, 500),
    "thomas": (
        lambda s: [-1.85 * s[0] + 10 * math.sin(s[1]), -1.85 * s[1] + 10 * math.sin(s[2]), -1.85 * s[2] + 10 * math.sin(s[0])],
        [1.0, 0.0, 0.5], 0.02, 500),
    "halvorsen": (
        lambda s: [-1.4 * s[0] - 4 * s[1] - 4 * s[2] - s[1] ** 2, -1.4 * s[1] - 4 * s[2] - 4 * s[0] - s[2] ** 2, -1.4 * s[2] - 4 * s[0] - 4 * s[1] - s[0] ** 2],
        [-5.0, 0.0, 0.0], 0.01, 500),
    "dadras": (
        lambda s: [s[1] - 3 * s[0] + 2.7 * s[1] * s[2], 1.7 * s[1] - s[0] * s[2] + s[2], 2 * s[0] * s[1] - 9 * s[2]],
        [1.0, 1.0, 1.0], 0.01, 500),
    "arneodo": (
        lambda s: [s[1], s[2], 5.5 * s[0] - 4.5 * s[1] - s[2] - s[0] ** 3],
        [0.2, 0.2, 0.2], 0.02, 500),
    "rucklidge": (
        lambda s: [-2 * s[0] + 6.7 * s[1] - s[1] * s[2], s[0], -s[2] + s[1] ** 2],
        [1.0, 0.0, 4.5], 0.02, 500),
    "sprott_b": (
        lambda s: [s[1] * s[2], s[0] - s[1], 1 - s[0] * s[1]],
        [0.1, 0.1, 0.1], 0.02, 500),
}

DISPLAY = {
    "lorenz": "Lorenz-63", "rossler": "Rössler", "thomas": "Thomas",
    "halvorsen": "Halvorsen", "dadras": "Dadras", "arneodo": "Arneodo",
    "rucklidge": "Rucklidge", "sprott_b": "Sprott B",
}

WARMUP, TRAIN, HORIZON = 300, 6000, 1400


def rk4_series(f, s, dt, n, skip):
    out = []
    for i in range(n + skip):
        k1 = f(s)
        k2 = f([s[j] + dt / 2 * k1[j] for j in range(3)])
        k3 = f([s[j] + dt / 2 * k2[j] for j in range(3)])
        k4 = f([s[j] + dt * k3[j] for j in range(3)])
        s = [s[j] + dt / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) for j in range(3)]
        if i >= skip:
            out.append(list(s))
    return torch.tensor(out).unsqueeze(0)  # (1, n, 3)


def valid_horizon(pred, true, tol=1.0):
    err = (pred - true).norm(dim=-1).squeeze(0)
    bad = (err > tol).nonzero()
    return int(bad[0]) if len(bad) else len(err)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    report = []
    for name, (f, s0, dt, skip) in SYSTEMS.items():
        data = rk4_series(f, list(s0), dt, WARMUP + TRAIN + HORIZON + 1, skip)
        mean, std = data.mean(dim=1, keepdim=True), data.std(dim=1, keepdim=True)
        data = (data - mean) / std

        warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
            data, warmup_steps=WARMUP, train_steps=TRAIN, val_steps=HORIZON
        )
        model = rd.models.ott_esn(reservoir_size=500, feedback_size=3, output_size=3)
        ESNTrainer(model).fit((warmup,), (train,), targets={"output": target})
        pred = model.forecast(f_warmup, horizon=HORIZON)

        vh = valid_horizon(pred, val)
        # The animation's story is: track closely, then decorrelate while
        # both keep tracing the attractor. So keep everything as long as
        # the forecast stays bounded (on-attractor in normalized units);
        # clip only where it actually escapes.
        norms = pred[0].norm(dim=-1)
        escaped = (norms > 4.0).nonzero()
        keep = int(escaped[0]) if len(escaped) else HORIZON
        keep = max(keep, 500)
        true_pts = val[0, :keep].tolist()
        pred_pts = pred[0, :keep].tolist()

        # Shared teacher-forced context before the forecast handoff: the
        # animation plays it identically on both curves, so the split is
        # visible exactly where autoregression begins.
        context = data[0, WARMUP + TRAIN - 400 : WARMUP + TRAIN].tolist()
        payload = {
            "name": DISPLAY[name],
            "context": [[round(v, 4) for v in p] for p in context],
            "true": [[round(v, 4) for v in p] for p in true_pts],
            "pred": [[round(v, 4) for v in p] for p in pred_pts],
        }
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        report.append((name, vh, keep, path.stat().st_size // 1024))

    for name, vh, keep, kb in report:
        print(f"{name:12s} valid_horizon={vh:5d}  kept={keep:5d}  {kb:4d} KB")


if __name__ == "__main__":
    main()
