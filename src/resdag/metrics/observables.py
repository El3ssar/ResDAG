"""Reservoir and attractor observables — pointwise error, memory, and geometry.

This module rounds out :mod:`resdag.metrics` with the standard *observables* a
reservoir-computing study reports beyond the valid-prediction-time / Lyapunov
bookkeeping that already lives here:

Pointwise forecast error
    :func:`rmse`, :func:`nrmse`, :func:`r2` — the elementary error helpers,
    computed exactly like :func:`resdag.metrics.valid_prediction_time`'s
    internal NRMSE (RMSE across feature channels, ``nrmse`` normalised by the
    per-dimension standard deviation of the truth over the ``(batch, time)``
    axes) so a per-step error and the VPT never disagree.

Reservoir capacity / dynamics
    :func:`memory_capacity` — the linear short-term memory capacity of Jaeger
    (2001): drive the reservoir with an i.i.d. signal, fit a linear readout to
    reconstruct the input delayed by ``k`` for ``k = 1..max_delay``, and sum the
    squared correlations :math:`MC = \\sum_k R^2_k`.  Bounded above by the
    reservoir size ``N`` and rising with the spectral radius toward that bound.
    :func:`effective_spectral_radius` — the mean spectral radius of the
    state-dependent recurrent Jacobian over a trajectory, accounting for
    activation saturation and leakage.

Attractor geometry
    :func:`correlation_dimension` — the Grassberger-Procaccia correlation
    dimension of a trajectory (Grassberger & Procaccia 1983): the fractal
    dimension of the reconstructed attractor, the key "did the forecast rebuild
    the right *climate*" diagnostic.  Reference values: Lorenz-63
    :math:`\\approx 2.05`, a limit cycle :math:`\\approx 1.0`, and
    :math:`d`-dimensional noise :math:`\\approx d`.

Everything here is pure PyTorch / NumPy and always available (no optional
dependency).

Examples
--------
>>> import torch
>>> from resdag.metrics import nrmse, r2
>>> truth = torch.randn(1, 200, 3)
>>> nrmse(truth, truth)  # identical -> zero error
0.0
>>> r2(truth, truth)  # perfect fit -> full variance explained
1.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = [
    "rmse",
    "nrmse",
    "r2",
    "memory_capacity",
    "effective_spectral_radius",
    "correlation_dimension",
]


# --------------------------------------------------------------------------- #
# Pointwise forecast-error helpers.
# --------------------------------------------------------------------------- #


def _as_2d_pair(pred: torch.Tensor, truth: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pred, truth)`` as ``float64`` NumPy arrays over a common prefix.

    Both inputs are detached, moved to CPU, promoted to ``float64``, and clipped
    to their overlapping time prefix (mirroring
    :func:`resdag.metrics.valid_prediction_time`), so a shorter forecast is
    scored only over the steps it actually produced.

    Parameters
    ----------
    pred, truth : torch.Tensor
        Forecast and ground truth of shape ``(batch, time, dim)`` (or any shape
        that broadcasts to a common trailing ``dim``); the time axis is taken to
        be axis 1 when 3-D.

    Returns
    -------
    tuple of numpy.ndarray
        ``(p, t)`` as ``float64`` arrays sharing the same shape.
    """
    p = pred.detach().cpu().double().numpy()
    t = truth.detach().cpu().double().numpy()
    if p.ndim == 3 and t.ndim == 3:
        n = min(p.shape[1], t.shape[1])
        p, t = p[:, :n], t[:, :n]
    if p.shape != t.shape:
        raise ValueError(
            f"pred and truth must have matching shapes after prefix alignment; "
            f"got {p.shape} and {t.shape}."
        )
    return p, t


def rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Root-mean-square error between a forecast and the truth.

    The RMSE is taken over **all** elements (batch, time, and feature axes),
    in the raw units of the data.  For a scale-free, threshold-comparable
    error use :func:`nrmse` instead.

    Parameters
    ----------
    pred, truth : torch.Tensor
        Forecast and ground truth of shape ``(batch, time, dim)``.  If the two
        differ in length along the time axis, they are compared over the
        overlapping prefix (as in :func:`valid_prediction_time`).

    Returns
    -------
    float
        :math:`\\sqrt{\\operatorname{mean}((\\hat y - y)^2)}` over every element.

    Examples
    --------
    >>> import torch
    >>> x = torch.zeros(1, 10, 3)
    >>> rmse(x, x)
    0.0
    """
    p, t = _as_2d_pair(pred, truth)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def nrmse(
    pred: torch.Tensor,
    truth: torch.Tensor,
    *,
    normalization: str = "std",
) -> float:
    """Normalised RMSE — scale-free forecast error.

    The RMSE is normalised so the metric is invariant to the absolute scale of
    the data, which is what makes a fixed error *threshold* (the basis of
    :func:`valid_prediction_time`) meaningful across systems.  Two conventions:

    - ``normalization="std"`` (default) — per-feature RMSE, each channel
      divided by the **per-dimension standard deviation** of ``truth`` over the
      ``(batch, time)`` axes, then combined across channels.  This is *exactly*
      the per-step NRMSE that :func:`valid_prediction_time` thresholds (reduced
      here over the whole trajectory rather than per step), so the two never
      disagree.
    - ``normalization="range"`` — the scalar RMSE over all elements divided by
      the peak-to-peak range (``max - min``) of ``truth``.

    Parameters
    ----------
    pred, truth : torch.Tensor
        Forecast and ground truth of shape ``(batch, time, dim)``, compared over
        their overlapping time prefix.
    normalization : {"std", "range"}, default="std"
        Whether to normalise by the per-dimension standard deviation of the
        truth (the reservoir-computing standard, matching
        :func:`valid_prediction_time`) or by its global peak-to-peak range.

    Returns
    -------
    float
        The normalised RMSE (``0`` for a perfect forecast).  A value of ``1``
        under ``"std"`` normalisation corresponds to the error of predicting the
        per-channel mean.

    Raises
    ------
    ValueError
        If ``normalization`` is not ``"std"`` or ``"range"``.

    Examples
    --------
    >>> import torch
    >>> truth = torch.randn(1, 200, 3)
    >>> nrmse(truth, truth)  # identical
    0.0
    """
    if normalization not in ("std", "range"):
        raise ValueError(f"Unknown normalization {normalization!r}; expected 'std' or 'range'.")
    p, t = _as_2d_pair(pred, truth)
    if normalization == "std":
        scale = t.std(axis=(0, 1), keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        # Per-step RMSE across channels of the standardised residual, then a
        # single RMS reduction across (batch, time) — the trajectory-level
        # counterpart of valid_prediction_time's per-step error.
        err = np.sqrt((((p - t) / scale) ** 2).mean(axis=2))  # (batch, time)
        return float(np.sqrt((err**2).mean()))
    # normalization == "range"
    span = float(t.max() - t.min())
    if span == 0.0:
        span = 1.0
    return float(np.sqrt(np.mean((p - t) ** 2)) / span)


def r2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Coefficient of determination :math:`R^2` of a forecast.

    The standard variance-explained score, pooled over all elements:

    .. math::

        R^2 = 1 - \\frac{\\sum (\\hat y - y)^2}{\\sum (y - \\bar y)^2},

    where :math:`\\bar y` is the grand mean of ``truth``.  ``1.0`` is a perfect
    forecast; ``0.0`` matches the skill of predicting the mean; negative values
    are worse than the mean.

    Parameters
    ----------
    pred, truth : torch.Tensor
        Forecast and ground truth of shape ``(batch, time, dim)``, compared over
        their overlapping time prefix.

    Returns
    -------
    float
        The :math:`R^2` score.  Returns ``1.0`` when ``truth`` is constant and
        the forecast matches it exactly (zero residual and zero total variance),
        else ``0.0`` for a constant truth with any residual.

    Examples
    --------
    >>> import torch
    >>> truth = torch.randn(1, 200, 3)
    >>> r2(truth, truth)
    1.0
    """
    p, t = _as_2d_pair(pred, truth)
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot == 0.0:
        # Constant truth: R^2 is conventionally 1 for a perfect fit, else 0.
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# Memory capacity (Jaeger 2001).
# --------------------------------------------------------------------------- #


def memory_capacity(
    source: Any,
    *,
    max_delay: int = 100,
    n_samples: int = 2000,
    washout: int = 200,
    alpha: float = 1e-8,
    input_low: float = -1.0,
    input_high: float = 1.0,
    seed: int = 0,
    return_curve: bool = False,
    test_samples: int | None = None,
) -> float | tuple[float, np.ndarray]:
    r"""Linear short-term memory capacity of a reservoir (Jaeger 2001).

    Drives the reservoir with a scalar i.i.d. uniform input :math:`u_t` and, for
    each delay :math:`k = 1 \dots` ``max_delay``, fits a **linear** readout from
    the reservoir state :math:`h_t` to the delayed input :math:`u_{t-k}`.  The
    memory capacity is the sum of the squared correlations of the reconstruction

    .. math::

        MC = \sum_{k=1}^{k_{\max}} R^2_k, \qquad
        R^2_k = \frac{\operatorname{cov}(\hat u_{t-k},\, u_{t-k})^2}
                     {\operatorname{var}(\hat u_{t-k})\,\operatorname{var}(u_{t-k})},

    the standard forgetting-curve integral (Jaeger, *Short-term memory in echo
    state networks*, 2001). Readouts are fitted on one random input trajectory
    and scored on an independently generated, independently warmed trajectory.
    No evaluation samples enter fitting or centering statistics.

    The theoretical population capacity is at most the number of reservoir
    states, under the usual independent-input assumptions. This finite-sample
    estimator may exceed that bound through sampling noise, particularly for
    many delays and few test samples. It is not guaranteed to increase with
    spectral radius. Increase ``test_samples`` and repeat seeds to assess
    uncertainty; this diagnostic is not an autonomous forecast score.

    The actual cell update runs on a local CPU float64 evaluation copy. Caller
    weights, cached state, device and training mode are preserved. Training-only
    noise is disabled. A shared ridge solve fits all delayed targets together.

    Parameters
    ----------
    source : BaseReservoirLayer or ESNModel
        The reservoir to probe, or a model containing exactly one ESN reservoir.
        The reservoir must accept a **scalar** feedback (``feedback_size == 1``),
        since memory capacity is defined for a one-dimensional i.i.d. drive.
    max_delay : int, default=100
        Largest delay :math:`k_{\max}` in the forgetting sum.  Should comfortably
        exceed the reservoir's memory (a few times ``N`` for a linear reservoir)
        so the tail :math:`R^2_k \approx 0` and the sum has converged.
    n_samples : int, default=2000
        Number of ``(state, delayed-input)`` samples used to fit each linear
        readout after washout and delay history. More samples than reservoir
        states are recommended; positive ridge regularization permits fewer.
    washout : int, default=200
        Leading driven steps discarded so the reservoir state reflects the input
        history rather than the (zero) initial condition.
    alpha : float, default=1e-8
        Non-negative ridge regularization on centered reservoir features. The
        intercept is unpenalized. Zero selects an unregularized least-squares
        fit; positive values stabilize correlated or underdetermined states.
    input_low, input_high : float, default=(-1.0, 1.0)
        Bounds of the uniform input distribution :math:`u_t \sim
        \mathcal{U}(\text{low}, \text{high})`.  The default symmetric ``[-1, 1]``
        is the Jaeger convention; a smaller range keeps a ``tanh`` reservoir
        nearer its linear regime (larger MC).
    seed : int, default=0
        Seed for the input draw, so the estimate is fully deterministic.
    test_samples : int or None, default=None
        Number of independent scoring samples, at least 2. ``None`` uses
        ``max(1000, n_samples)`` to limit chance correlations at long delays.
    return_curve : bool, default=False
        If ``True``, also return the per-delay forgetting curve
        :math:`(R^2_1, \dots, R^2_{k_{\max}})` as a ``(max_delay,)`` array.

    Returns
    -------
    float or tuple of (float, numpy.ndarray)
        The memory capacity :math:`MC`.  When ``return_curve`` is ``True``, a
        ``(MC, curve)`` pair where ``curve[k-1]`` is :math:`R^2_k`.

    Raises
    ------
    ValueError
        If the reservoir needs multiple feedback channels or an external driver,
        counts are invalid, training/scoring samples are fewer than 2, or the
        distribution bounds / regularization are invalid.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> res = ESNLayer(100, feedback_size=1, spectral_radius=0.9, seed=0)
    >>> mc = memory_capacity(res, max_delay=120, n_samples=2000)
    >>> mc >= 0.0
    True

    See Also
    --------
    effective_spectral_radius : Realized contraction rate of the driven reservoir.
    """
    from resdag.data.prepare import _validate_step_count
    from resdag.metrics.lyapunov import _diagnostic_cell, _drive_cell

    max_delay = _validate_step_count("max_delay", max_delay)
    n_samples = _validate_step_count("n_samples", n_samples)
    washout = _validate_step_count("washout", washout, allow_zero=True)
    test_samples = _validate_step_count(
        "test_samples", max(1000, n_samples) if test_samples is None else test_samples
    )
    if n_samples < 2 or test_samples < 2:
        raise ValueError("n_samples and test_samples must both be at least 2.")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative.")
    if not (np.isfinite(input_low) and np.isfinite(input_high) and input_low < input_high):
        raise ValueError("input_low and input_high must be finite with input_low < input_high.")
    cell = _diagnostic_cell(source)
    if getattr(cell, "feedback_size", None) != 1 or getattr(cell, "input_size", None) not in (
        None,
        0,
    ):
        raise ValueError("memory_capacity requires feedback_size == 1 and no external driver.")
    generator = torch.Generator().manual_seed(int(seed))
    start = washout + max_delay

    def samples(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.rand(1, start + count, 1, generator=generator, dtype=torch.float64)
        u = input_low + (input_high - input_low) * u
        states, _ = _drive_cell(cell, [u])
        times = torch.arange(start, start + count)
        delays = torch.arange(1, max_delay + 1)
        targets = u[0, times[:, None] - delays, 0]
        return states[0, start:], targets

    # Independent i.i.d. streams, independently warmed from the reset template:
    # no evaluation target or state belongs to the fitting sequence.
    train_x, train_y = samples(n_samples)
    test_x, test_y = samples(test_samples)
    x_mean, y_mean = train_x.mean(0), train_y.mean(0)
    centered = train_x - x_mean
    gram = centered.T @ centered
    gram.diagonal().add_(alpha)
    coefficients = (
        torch.linalg.solve(gram, centered.T @ (train_y - y_mean))
        if alpha > 0
        else torch.linalg.lstsq(centered, train_y - y_mean).solution
    )
    predictions = (test_x - x_mean) @ coefficients + y_mean
    predictions = predictions - predictions.mean(0)
    targets = test_y - test_y.mean(0)
    denominator = predictions.square().sum(0) * targets.square().sum(0)
    correlations = (predictions * targets).sum(0).square()
    curve_tensor = torch.where(denominator > 0, correlations / denominator, 0.0).clamp(0, 1)
    curve = curve_tensor.numpy()
    capacity = float(curve.sum())
    return (capacity, curve) if return_curve else capacity


# --------------------------------------------------------------------------- #
# Effective (realized) spectral radius.
# --------------------------------------------------------------------------- #


def effective_spectral_radius(
    source: Any,
    states: torch.Tensor,
    *,
    max_states: int = 256,
    seed: int = 0,
) -> float:
    r"""Mean spectral radius of the state-dependent leaky-ESN Jacobian.

    For leak rate :math:`\alpha` and recurrent matrix :math:`W`, the update is
    :math:`h_t = (1-\alpha)h_{t-1} + \alpha\tanh(\mathrm{pre}_t)`, with Jacobian

    .. math::

        J_t = (1-\alpha)I + \alpha\operatorname{diag}
              (1-\tanh^2(\mathrm{pre}_t))W.

    The activation derivative depends on ``tanh(pre)``, which differs from the
    stored hidden state when any neuron has a leak rate below 1. Consecutive states recover
    that activation without re-running the reservoir. Bias and drive affect the
    result through the trajectory's activation saturation. ``W`` here includes
    calibrated recurrent row gains and intrinsic gains. Leak rates may differ
    per neuron. Caller weights, cached state and mode are preserved.

    This returns a mean local eigenvalue radius. Leakage and non-normal weights
    mean it need not be below the weight spectral radius, and it does not bound
    transient amplification. Eigensolves use CPU float64 after one transfer, so
    trajectories from CUDA and MPS reservoirs are both supported.

    Parameters
    ----------
    source : BaseReservoirLayer or ESNModel
        The reservoir whose recurrent Jacobian is analysed, or a model
        containing exactly one standard ``ESNCell`` reservoir with ``tanh``.
        Alternative cell update equations are explicitly unsupported.
    states : torch.Tensor
        Reservoir states sampled along an attractor trajectory, of shape
        ``(batch, time, N)`` or ``(time, N)`` (with ``N`` the reservoir size).
        Supply consecutive states from each trajectory, as returned by the
        reservoir on representative input. For leak rate 1 the activation is
        the state itself. For a leaky reservoir it is recovered from adjacent
        states as ``(h_t - (1-alpha)*h_{t-1}) / alpha``; the first state of each
        trajectory is excluded because its predecessor is unavailable.
    max_states : int, default=256
        Cap on the number of trajectory states the mean is taken over; the
        states are subsampled on an evenly spaced deterministic grid so
        the estimate stays manageable for a long trajectory —
        each state costs one :math:`N \times N` eigenvalue solve.
    seed : int, default=0
        Retained for API compatibility; even subsampling uses no randomness.

    Returns
    -------
    float
        The mean realized spectral radius :math:`\langle \rho(J_t) \rangle` over
        the sampled states, in per-step units. Leakage and non-normal recurrent
        weights mean this need not be below the weight spectral radius.

    Raises
    ------
    ValueError
        If ``source`` has no (or more than one) ESN reservoir, or if ``states``
        does not have a trailing dimension equal to the reservoir size.
    NotImplementedError
        If the cell is not a standard tanh ``ESNCell``.

    Examples
    --------
    >>> import torch
    >>> from resdag.layers import ESNLayer
    >>> res = ESNLayer(100, feedback_size=1, spectral_radius=0.9, seed=0)
    >>> u = torch.rand(1, 1500, 1) * 2 - 1
    >>> res.reset_state(batch_size=1)
    >>> st = res(u)
    >>> esr = effective_spectral_radius(res, st)
    >>> esr < 0.9  # tanh saturation contracts below the nominal weight radius
    True

    See Also
    --------
    resdag.layers.cells.ESNCell.spectral_radius_achieved : The *weight* spectral
        radius including calibration gains, before activation derivatives and leakage.
    resdag.metrics.reservoir_lyapunov : Benettin/QR maximal Lyapunov exponent of
        the driven reservoir (a time-averaged log-contraction, vs this per-step
        eigenvalue mean).
    """
    from resdag.data.prepare import _validate_step_count
    from resdag.metrics.lyapunov import _diagnostic_cell, _esn_recurrent_jacobian_weight

    max_states = _validate_step_count("max_states", max_states)
    cell = _diagnostic_cell(source, tanh_esn=True)
    weight = _esn_recurrent_jacobian_weight(cell)
    n = weight.shape[0]
    h = states.detach().cpu().to(dtype=torch.float64)
    if h.ndim not in (2, 3) or h.shape[-1] != n:
        raise ValueError(f"states must have shape (time,{n}) or (batch,time,{n}).")
    if not bool(torch.isfinite(h).all()):
        raise ValueError("states must be finite.")
    alpha = torch.as_tensor(cell.leak_rate, dtype=torch.float64)
    if alpha.ndim == 0:
        alpha = alpha.expand(n)
    if not bool((alpha == 1).all()):
        if h.shape[-2] < 2:
            raise ValueError(
                "Leaky reservoirs need at least two consecutive states per trajectory."
            )
        # Invert leakage per neuron, preserving trajectory boundaries.
        h = (h[..., 1:, :] - (1 - alpha) * h[..., :-1, :]) / alpha
    h = h.reshape(-1, n)
    if not h.shape[0]:
        raise ValueError("states must contain at least one trajectory state.")
    if h.shape[0] > max_states:
        indices = torch.linspace(0, h.shape[0] - 1, max_states).round().long().unique()
        h = h[indices]
    _ = seed  # The even subsampling grid does not draw randomness.
    identity = torch.eye(n, dtype=torch.float64)
    radii = []
    for activation in h:
        jacobian = alpha[:, None] * (1 - activation.square())[:, None] * weight
        jacobian += (1 - alpha)[:, None] * identity
        radii.append(torch.linalg.eigvals(jacobian).abs().max())
    return float(torch.stack(radii).mean())


# --------------------------------------------------------------------------- #
# Grassberger-Procaccia correlation dimension.
# --------------------------------------------------------------------------- #


def correlation_dimension(
    trajectory: torch.Tensor | np.ndarray,
    *,
    n_bins: int = 20,
    r_min_frac: float = 0.01,
    r_max_frac: float = 0.5,
    fit_window: tuple[float, float] = (0.1, 0.5),
    max_points: int = 2000,
    seed: int = 0,
) -> float:
    r"""Grassberger-Procaccia correlation dimension of an attractor trajectory.

    Estimates the fractal (correlation) dimension :math:`D_2` of the attractor
    a trajectory lives on, via the Grassberger-Procaccia algorithm (Grassberger
    & Procaccia, *Characterization of strange attractors*, 1983).  The
    correlation sum

    .. math::

        C(r) = \frac{2}{M(M-1)} \sum_{i<j} \Theta\!\big(r - \lVert x_i - x_j \rVert\big)

    counts the fraction of point pairs closer than :math:`r`.  On a
    :math:`D_2`-dimensional attractor it scales as :math:`C(r) \sim r^{D_2}` over
    an intermediate ("scaling") range of :math:`r`, so :math:`D_2` is the slope
    of :math:`\log C(r)` against :math:`\log r` fit over that range.

    This is the standard **attractor-fidelity** metric for reservoir forecasts:
    a free-running forecast that reproduces the true system's *climate* (its
    long-run geometry) matches the truth's correlation dimension, even when the
    pointwise trajectory has long since decorrelated.

    Reference values (used to validate this implementation):

    * **Lorenz-63** :math:`D_2 \approx 2.05` (literature; Grassberger & Procaccia
      1983).
    * **A limit cycle / closed curve** :math:`D_2 \approx 1.0` (a 1-D manifold).
    * **:math:`d`-dimensional i.i.d. noise** :math:`D_2 \approx d` (it fills the
      embedding space).

    Parameters
    ----------
    trajectory : torch.Tensor or numpy.ndarray
        The attractor trajectory, of shape ``(time, dim)`` (a 1-D ``(time,)``
        signal is treated as a single channel).  A leading singleton batch axis
        ``(1, time, dim)`` is squeezed.  For a fair comparison of a forecast to
        the truth, embed both the same way and pass equal-length windows.
    n_bins : int, default=20
        Number of log-spaced radii :math:`r` at which :math:`C(r)` is evaluated.
    r_min_frac, r_max_frac : float, default=(0.01, 0.5)
        The radius grid spans ``[r_min_frac, r_max_frac] * d_max`` where
        ``d_max`` is the largest pairwise distance, so the grid adapts to the
        attractor's size.
    fit_window : tuple of float, default=(0.1, 0.5)
        The scaling region over which the :math:`\log C` vs :math:`\log r` slope
        is fit, given as fractions of the (valid) ``log r`` span.  The default
        lower-middle window avoids the noisy small-:math:`r` depletion and the
        large-:math:`r` saturation that bias a full-range fit.
    max_points : int, default=2000
        Cap on the number of trajectory points used (the pair count is
        :math:`O(M^2)`); a longer trajectory is uniformly subsampled.  ``2000``
        points (~2M pairs) already give a stable dimension for Lorenz-63.
    seed : int, default=0
        Seed for the subsampling draw, so the estimate is deterministic.

    Returns
    -------
    float
        The estimated correlation dimension :math:`D_2`.

    Raises
    ------
    ValueError
        If ``trajectory`` is not 1-D/2-D (after squeezing a batch axis), has
        fewer than 2 points, or is a single constant point (zero extent).

    Notes
    -----
    The estimate is a **geometry** metric, not an error: two trajectories on the
    *same* attractor but out of phase share a correlation dimension.  Compare a
    forecast's dimension to the truth's (``|D2_pred - D2_truth|``) to score
    climate reconstruction — this is what
    :func:`resdag.hpo.losses.attractor_dimension_error` optimises.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    >>> circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    >>> abs(correlation_dimension(circle) - 1.0) < 0.15
    True
    """
    if isinstance(trajectory, torch.Tensor):
        x = trajectory.detach().cpu().double().numpy()
    else:
        x = np.asarray(trajectory, dtype=np.float64)

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(
            f"trajectory must be 1-D (time,), 2-D (time, dim), or 3-D (1, time, dim); "
            f"got shape {np.asarray(trajectory).shape}."
        )
    n_points = x.shape[0]
    if n_points < 2:
        raise ValueError(f"trajectory needs at least 2 points, got {n_points}.")

    if n_points > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_points, size=max_points, replace=False)
        idx.sort()
        x = x[idx]
        n_points = max_points

    # Upper-triangular pairwise Euclidean distances.
    diff = x[:, None, :] - x[None, :, :]
    dmat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    iu = np.triu_indices(n_points, k=1)
    dists = dmat[iu]
    dists = dists[dists > 0.0]
    if dists.size == 0:
        raise ValueError(
            "trajectory has zero extent (all points coincide); the correlation "
            "dimension is undefined."
        )
    d_max = float(dists.max())

    radii = np.logspace(
        np.log10(r_min_frac * d_max),
        np.log10(r_max_frac * d_max),
        n_bins,
    )
    # Correlation sum C(r): fraction of pairs closer than r. searchsorted on the
    # sorted distances turns each count into an O(log) lookup.
    dists_sorted = np.sort(dists)
    counts = np.searchsorted(dists_sorted, radii, side="left")
    corr_sum = counts / dists_sorted.size

    valid = corr_sum > 0.0
    log_r = np.log(radii[valid])
    log_c = np.log(corr_sum[valid])
    if log_r.size < 2:
        raise ValueError(
            "not enough non-empty correlation-sum bins to fit a slope; "
            "increase n_bins or the trajectory length."
        )

    lo, hi = fit_window
    r0, r1 = log_r.min(), log_r.max()
    span = r1 - r0
    if span <= 0.0:
        fit_mask = np.ones_like(log_r, dtype=bool)
    else:
        fit_mask = (log_r >= r0 + lo * span) & (log_r <= r0 + hi * span)
        if fit_mask.sum() < 2:
            fit_mask = np.ones_like(log_r, dtype=bool)

    slope = float(np.polyfit(log_r[fit_mask], log_c[fit_mask], 1)[0])
    return slope
