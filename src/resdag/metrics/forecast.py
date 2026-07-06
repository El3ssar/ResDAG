"""Forecast-accuracy metrics, reportable in steps or Lyapunov times.

The headline metric is the **valid prediction time** (VPT): how many steps a
closed-loop forecast tracks the true trajectory before its normalised error
first crosses a threshold.  It is the standard way to compare predictive skill
on chaotic systems and — via :func:`resdag.metrics.steps_to_lyapunov` — is best
reported in Lyapunov times so it is comparable across systems and integration
steps.

:func:`valid_prediction_time` here is the **canonical** implementation of the
contiguous-valid-horizon / NRMSE logic.  Two other copies exist in the codebase
for dependency-isolation reasons —
:func:`resdag.hpo.losses.forecast_horizon` (kept out of the optional HPO stack's
import path) and ``tests/test_models/golden_forecast.valid_prediction_time`` (so
the base test suite never imports HPO) — and both mirror the exact same NRMSE /
contiguous-count / batch-median semantics.  A later phase can delegate them to
this function; they are intentionally left untouched for now.
"""

from __future__ import annotations

import numpy as np
import torch

from .lyapunov import CANONICAL_LAMBDA_MAX, MACKEY_GLASS_LAMBDA, steps_to_lyapunov

# Integration step of each built-in generator, duplicated from
# ``resdag.utils.data.datasets`` so importing this metrics module never pulls in
# the dataset generators (and vice versa — datasets imports the constant tables
# from here).  Kept in sync by the ``DATASET_DT`` single-source comment in
# ``datasets.py``.
_DATASET_DT: dict[str, float] = {
    "lorenz": 0.02,
    "rossler": 0.05,
    "henon": 1.0,
    "mackey_glass": 1.0,
    "narma": 1.0,
    "sine": 1.0,
    "kuramoto_sivashinsky": 0.25,
}


def _infer_dt_lambda(
    dataset: str | None,
    dt: float | None,
    lambda_max: float | None,
    **kw: float,
) -> tuple[float, float]:
    """Resolve ``(dt, lambda_max)`` from explicit args or a known dataset name.

    Explicit ``dt`` / ``lambda_max`` always win; any left ``None`` are filled in
    from the canonical tables keyed by ``dataset``.

    Raises
    ------
    ValueError
        If, after inference, either value is still missing (or the dataset is
        non-chaotic and so has no Lyapunov exponent).
    """
    if dataset is not None:
        key = dataset.lower()
        if dt is None:
            dt = _DATASET_DT.get(key)
        if lambda_max is None:
            if key in CANONICAL_LAMBDA_MAX:
                lambda_max = CANONICAL_LAMBDA_MAX[key]
            elif key in ("mackey_glass", "mackeyglass", "mackey-glass"):
                tau = int(kw.get("tau", 17))
                lambda_max = MACKEY_GLASS_LAMBDA.get(tau)

    if dt is None:
        raise ValueError(
            "units='lyapunov' needs dt: pass dt=... or a known dataset= name "
            f"(one of {sorted(_DATASET_DT)})."
        )
    if lambda_max is None:
        raise ValueError(
            "units='lyapunov' needs lambda_max: pass lambda_max=... or a known "
            "chaotic dataset= name (lorenz, rossler, mackey_glass)."
        )
    return dt, lambda_max


def valid_prediction_time(
    pred: torch.Tensor,
    truth: torch.Tensor,
    *,
    threshold: float = 0.3,
    metric: str = "nrmse",
    units: str = "steps",
    dt: float | None = None,
    lambda_max: float | None = None,
    dataset: str | None = None,
    **kw: float,
) -> float:
    """Valid Prediction Time: contiguous steps a forecast stays accurate.

    Per-step error is the **NRMSE** — the RMSE across feature channels, each
    channel normalised by the per-dimension standard deviation of ``truth`` over
    the ``(batch, time)`` axes — reduced across the batch with the **median**.
    The VPT is the number of steps, counting from the start, whose error stays
    strictly below ``threshold`` (``0`` if the very first step is already over).

    This matches :func:`resdag.hpo.losses.forecast_horizon` exactly (that copy
    is kept out of the optional-HPO import path); see the module docstring.

    Parameters
    ----------
    pred, truth : torch.Tensor
        Forecast and ground truth of shape ``(batch, time, dim)``.
    threshold : float, default=0.3
        NRMSE threshold below which a step counts as "valid".
    metric : {"nrmse"}, default="nrmse"
        Per-step error metric.  Only NRMSE is implemented (the reservoir-computing
        standard); the argument exists for forward compatibility.
    units : {"steps", "lyapunov"}, default="steps"
        Return the horizon in raw steps, or converted to Lyapunov times
        (``steps * dt * lambda_max``).  ``"lyapunov"`` needs ``dt`` and
        ``lambda_max``, or a known ``dataset`` name to infer them.
    dt : float, optional
        Integration step, required for ``units="lyapunov"`` unless inferable
        from ``dataset``.
    lambda_max : float, optional
        Largest Lyapunov exponent, required for ``units="lyapunov"`` unless
        inferable from a chaotic ``dataset``.
    dataset : str, optional
        A built-in generator name (``"lorenz"``, ``"rossler"``,
        ``"mackey_glass"``, ...) used to look up ``dt`` / ``lambda_max`` when
        they are not given explicitly.
    **kw
        Extra keys forwarded to inference; ``tau=`` selects the Mackey-Glass
        exponent.

    Returns
    -------
    float
        The VPT: an integer step count as a ``float`` when ``units="steps"``, or
        the horizon in Lyapunov times when ``units="lyapunov"``.

    Raises
    ------
    ValueError
        If ``metric`` is not ``"nrmse"``, ``units`` is unknown, or
        ``units="lyapunov"`` cannot resolve ``dt`` / ``lambda_max``.

    Examples
    --------
    >>> import torch
    >>> pred = torch.zeros(1, 100, 3)
    >>> truth = torch.zeros(1, 100, 3)
    >>> valid_prediction_time(pred, truth)  # identical -> full horizon
    100.0
    """
    if metric != "nrmse":
        raise ValueError(f"Unknown metric {metric!r}; only 'nrmse' is implemented.")
    if units not in ("steps", "lyapunov"):
        raise ValueError(f"Unknown units {units!r}; expected 'steps' or 'lyapunov'.")

    p = pred.detach().cpu().double().numpy()
    t = truth.detach().cpu().double().numpy()
    # Compare over the overlapping prefix if the two differ in length.
    n = min(p.shape[1], t.shape[1])
    p, t = p[:, :n], t[:, :n]

    scale = t.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale == 0.0, 1.0, scale)
    err = np.sqrt((((p - t) / scale) ** 2).mean(axis=2))  # (batch, time)
    e_t = np.median(err, axis=0)  # (time,)

    below = e_t < threshold
    if below.size == 0 or not below[0]:
        steps = 0
    elif (~below).any():
        steps = int(np.argmax(~below))
    else:
        steps = int(below.size)

    if units == "steps":
        return float(steps)

    dt_r, lam_r = _infer_dt_lambda(dataset, dt, lambda_max, **kw)
    return steps_to_lyapunov(steps, dt_r, lam_r)
