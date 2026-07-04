"""Forecast-accuracy metrics.

The headline metric is the **valid prediction time** (VPT): how many steps a
closed-loop forecast tracks the true trajectory before the normalized error
crosses a threshold. It is the standard way to compare predictive skill on
chaotic systems and is reported both in steps and in Lyapunov times.

All the numeric machinery lives in the library itself
(:mod:`resdag.metrics`) so the benchmark and the library share **one** source
of truth for the Lyapunov exponent and the VPT / Lyapunov-time definitions.
This module is a thin, benchmark-shaped adapter over it: it keys ``dt`` and
``lambda_max`` off the dataset name (Lorenz today) via the public
:data:`resdag.utils.data.DATASET_DT` / :func:`resdag.utils.data.dataset_lambda_max`
tables, so pointing the harness at another built-in generator generalizes for
free.
"""

from __future__ import annotations

import numpy as np
import torch

from resdag.metrics import steps_to_lyapunov
from resdag.metrics import valid_prediction_time as _rd_vpt
from resdag.utils.data import DATASET_DT, dataset_lambda_max

# The generator the benchmark drives everything with (see ``rc_bench.data``).
# Kept as a name so the dt / lambda lookups below generalize to any built-in
# chaotic generator resdag ships.
DATASET = "lorenz"

# Lorenz-63 largest Lyapunov exponent and the integrator step used in data.py,
# sourced from the single library-side tables rather than re-declared here.
_LAMBDA = dataset_lambda_max(DATASET)
assert _LAMBDA is not None, f"{DATASET!r} has no canonical Lyapunov exponent"
LORENZ_LAMBDA_MAX: float = _LAMBDA
LORENZ_DT: float = DATASET_DT[DATASET]


def valid_prediction_time(pred: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> int:
    """Number of steps the forecast stays valid (delegates to :mod:`resdag.metrics`).

    Both arrays are ``(steps, dim)`` for a single trajectory; a singleton batch
    axis is added so :func:`resdag.metrics.valid_prediction_time` (which works on
    ``(batch, time, dim)`` and defines the canonical NRMSE / contiguous-valid
    semantics) can score them. Returns the whole-step VPT.
    """
    vpt = _rd_vpt(
        torch.as_tensor(pred[None]),
        torch.as_tensor(truth[None]),
        threshold=threshold,
    )
    return int(vpt)


def lyapunov_times(steps: int, dt: float = LORENZ_DT, lam: float = LORENZ_LAMBDA_MAX) -> float:
    """Convert a step count to Lyapunov times (Λ·t) via :mod:`resdag.metrics`."""
    return steps_to_lyapunov(steps, dt=dt, lambda_max=lam)
