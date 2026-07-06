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
    *state-dependent* recurrent Jacobian over an attractor trajectory: the
    **realized** contraction rate the driven reservoir actually experiences,
    which ``tanh`` saturation pushes strictly below the nominal *weight*
    spectral radius.

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

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from resdag.layers.cells import ESNCell
    from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer

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


def _resolve_reservoir_local(source: Any) -> "BaseReservoirLayer":
    """Resolve ``source`` to a single ESN reservoir layer (shared with lyapunov).

    Thin re-export of :func:`resdag.metrics.lyapunov._resolve_reservoir` so the
    observables here accept the same ``BaseReservoirLayer`` / ``ESNModel``
    dispatch, without importing it at module load (avoids a cycle on the layers
    package).
    """
    from .lyapunov import _resolve_reservoir

    return _resolve_reservoir(source)


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
    state networks*, 2001).  Each :math:`R^2_k \in [0, 1]`, and the total is
    bounded above by the reservoir size :math:`N` — a linear reservoir with an
    i.i.d. input saturates the bound :math:`MC \to N` as the spectral radius
    approaches 1, while a ``tanh`` reservoir sits below it (saturation destroys
    linear memory).  ``MC`` rises with the spectral radius toward that bound.

    The reconstruction uses one shared ridge factorisation (the design matrix is
    the same for every delay; only the target shifts), so the whole forgetting
    curve costs a single :math:`(N{+}1)\times(N{+}1)` solve regardless of
    ``max_delay``.

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
        readout (after the washout).  Must exceed ``N`` for a well-posed fit.
    washout : int, default=200
        Leading driven steps discarded so the reservoir state reflects the input
        history rather than the (zero) initial condition.
    alpha : float, default=1e-8
        Tiny ridge regularisation on the linear readout — large enough to keep
        the :math:`(N{+}1)` normal equations invertible, small enough not to
        bias :math:`R^2` downward.
    input_low, input_high : float, default=(-1.0, 1.0)
        Bounds of the uniform input distribution :math:`u_t \sim
        \mathcal{U}(\text{low}, \text{high})`.  The default symmetric ``[-1, 1]``
        is the Jaeger convention; a smaller range keeps a ``tanh`` reservoir
        nearer its linear regime (larger MC).
    seed : int, default=0
        Seed for the input draw, so the estimate is fully deterministic.
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
        If the reservoir's feedback dimension is not 1, if ``max_delay`` /
        ``n_samples`` / ``washout`` are non-positive, or if ``n_samples`` leaves
        too few samples for the requested delays.

    Notes
    -----
    ``MC`` is an upper-bounded, monotone-in-spectral-radius capacity, **not** a
    forecast score: it measures how much of the input history the reservoir
    linearly retains, independent of any downstream task.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> res = ESNLayer(100, feedback_size=1, spectral_radius=0.9, seed=0)
    >>> mc = memory_capacity(res, max_delay=120, n_samples=2000)
    >>> 0.0 <= mc <= 100.0
    True

    See Also
    --------
    effective_spectral_radius : Realized contraction rate of the driven reservoir.
    """
    if max_delay < 1:
        raise ValueError(f"max_delay must be >= 1, got {max_delay}.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    if washout < 0:
        raise ValueError(f"washout must be >= 0, got {washout}.")

    reservoir = _resolve_reservoir_local(source)
    # ``_resolve_reservoir_local`` guarantees a cell owning ``weight_hh`` (an ESN
    # cell); narrow the type so the analytic weight access below type-checks.
    cell = cast("ESNCell", reservoir.cell)
    feedback_size = int(getattr(cell, "feedback_size", 0))
    if feedback_size != 1:
        raise ValueError(
            f"memory_capacity requires a scalar-input reservoir (feedback_size == 1), "
            f"got feedback_size={feedback_size}. Memory capacity is defined for a "
            f"one-dimensional i.i.d. drive."
        )

    weight_hh: torch.Tensor = cell.weight_hh
    device = weight_hh.device
    native_dtype = weight_hh.dtype
    n_reservoir = int(weight_hh.shape[0])

    # The present index t must be able to reach back max_delay steps *and* start
    # after the washout, so we need washout + max_delay + n_samples driven steps.
    total = washout + max_delay + n_samples
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    u = torch.rand(total, 1, generator=generator, dtype=torch.float64)
    u = input_low + (input_high - input_low) * u  # U(low, high)
    u_native = u.to(device=device, dtype=native_dtype).unsqueeze(0)  # (1, T, 1)

    reservoir.reset_state(batch_size=1)
    with torch.no_grad():
        states = reservoir(u_native)  # (1, T, N)
    states64 = states.squeeze(0).to(torch.float64)  # (T, N)
    u_flat = u.squeeze(-1)  # (T,) on CPU float64
    states_cpu = states64.cpu()

    # The "present" time indices at which we read the state and predict u[t-k].
    # They start at washout + max_delay so t - max_delay >= washout (the earliest
    # delayed target still lies in the driven, post-washout window).
    t0 = washout + max_delay
    t_idx = torch.arange(t0, t0 + n_samples)
    if t_idx[-1].item() >= total:
        raise ValueError(
            "n_samples/max_delay/washout leave too few driven steps; "
            "increase n_samples or reduce max_delay/washout."
        )

    X = states_cpu[t_idx]  # (n_samples, N)
    ones = torch.ones(X.shape[0], 1, dtype=torch.float64)
    Xb = torch.cat([X, ones], dim=1)  # (n_samples, N+1) — bias column

    # One shared ridge factorisation reused for every delay (design matrix is
    # delay-independent; only the target shifts).
    gram = Xb.T @ Xb + alpha * torch.eye(n_reservoir + 1, dtype=torch.float64)
    factor = torch.linalg.cholesky(gram)
    xty_base = Xb.T  # (N+1, n_samples)

    curve = np.empty(max_delay, dtype=np.float64)
    for k in range(1, max_delay + 1):
        target = u_flat[t_idx - k]  # (n_samples,)
        w = torch.cholesky_solve((xty_base @ target).unsqueeze(1), factor).squeeze(1)
        pred = Xb @ w
        # Squared correlation coefficient (Jaeger's covariance form). Robust to
        # the readout's arbitrary scale/offset and bounded in [0, 1].
        p_c = pred - pred.mean()
        t_c = target - target.mean()
        denom = (p_c @ p_c) * (t_c @ t_c)
        r2_k = float((p_c @ t_c) ** 2 / denom) if float(denom) > 0.0 else 0.0
        # Guard tiny negative/over-one drift from round-off.
        curve[k - 1] = min(max(r2_k, 0.0), 1.0)

    mc = float(curve.sum())
    if return_curve:
        return mc, curve
    return mc


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
    r"""Mean spectral radius of the state-dependent recurrent Jacobian.

    The **weight** spectral radius (``ESNLayer(spectral_radius=...)`` /
    :attr:`ESNCell.spectral_radius_achieved`) is a property of the recurrent
    matrix :math:`W_{hh}` *alone* — the largest :math:`|\lambda|` of
    :math:`W_{hh}`.  It ignores the activation, so it over-states the contraction
    the reservoir actually experiences: at every step the leaky-``tanh`` update
    contracts by the **state-dependent** Jacobian

    .. math::

        J_t = (1-\alpha) I + \alpha \operatorname{diag}\!\big(1 - \tanh^2(\mathrm{pre}_t)\big)\, W_{hh},

    whose diagonal gate :math:`1 - \tanh^2 \le 1` is strictly below 1 wherever
    the units are driven away from the origin.  This function returns the mean
    over an attractor trajectory of :math:`\rho(J_t)` — the **realized**
    (effective) spectral radius — which ``tanh`` saturation makes **smaller than
    the nominal weight spectral radius**, and smaller still under a stronger
    drive.  It is the honest per-step contraction rate; do not confuse it with
    the weight spectral radius (see the two references in the return value).

    **Regime dependence — it is the state that saturates, not the input.**
    The gate is set by the *state* magnitude, not directly by the input, so the
    ESR equals the nominal weight radius only in the **zero-state / tiny-state
    (linear) regime**:

    - **Zero state** (:math:`h_t = 0`): the gate is exactly 1, so the ESR is
      *exactly* the nominal weight spectral radius.
    - **Tiny state near the origin**: the gate is :math:`\approx 1`, so the ESR
      is :math:`\approx` nominal.

    A *tiny input* only yields a tiny state — and hence ESR :math:`\approx`
    nominal — when nothing else pushes the state off the origin.  On a default
    ``bias=True`` reservoir the constant ``bias_h`` term drives the state away
    from the origin even under a vanishingly small input, so the state
    self-saturates: an on-attractor (or even a ``1e-4``-scaled) drive reads ESR
    materially *below* nominal (e.g. ``~0.73x`` on a nominal-``0.9`` reservoir),
    not ``~1x``.  With ``bias=False`` a tiny input keeps the state near the
    origin and the ESR sits at :math:`\approx` nominal, dropping below it only as
    the drive grows.  So "tiny drive ⇒ ESR ≈ nominal" holds for the *state*, and
    for the *input* only on an unbiased reservoir; a biased reservoir on its
    attractor always reads ESR < nominal.

    Parameters
    ----------
    source : BaseReservoirLayer or ESNModel
        The reservoir whose recurrent Jacobian is analysed, or a model
        containing exactly one ESN reservoir.  Must use the ``tanh`` activation
        (the only activation whose derivative is wired here).
    states : torch.Tensor
        Reservoir states sampled along an attractor trajectory, of shape
        ``(batch, time, N)`` or ``(time, N)`` (with ``N`` the reservoir size).
        These are the post-activation states :math:`h_t = \tanh(\mathrm{pre}_t)`,
        so the Jacobian gate is evaluated as :math:`1 - h_t^2` directly — no
        need to recover the pre-activation.  Generate them by running the
        reservoir on representative input (e.g. the training feedback).
    max_states : int, default=256
        Cap on the number of trajectory states the mean is taken over; the
        states are subsampled evenly (with a fixed seed for the tie-breaking
        draw when needed) so the estimate stays cheap for a long trajectory —
        each state costs one :math:`N \times N` eigenvalue solve.
    seed : int, default=0
        Seed for the (rare) random subsampling tie-break, for determinism.

    Returns
    -------
    float
        The mean realized spectral radius :math:`\langle \rho(J_t) \rangle` over
        the sampled states, in per-step units.  Strictly below the reservoir's
        weight spectral radius for a saturating ``tanh`` reservoir.

    Raises
    ------
    ValueError
        If ``source`` has no (or more than one) ESN reservoir, or if ``states``
        does not have a trailing dimension equal to the reservoir size.
    NotImplementedError
        If the reservoir activation is not ``tanh``.

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
        radius (of ``weight_hh`` alone), for contrast.
    resdag.metrics.reservoir_lyapunov : Benettin/QR maximal Lyapunov exponent of
        the driven reservoir (a time-averaged log-contraction, vs this per-step
        eigenvalue mean).
    """
    reservoir = _resolve_reservoir_local(source)
    # ``_resolve_reservoir_local`` guarantees a cell owning ``weight_hh`` (an ESN
    # cell); narrow the type so the analytic weight access below type-checks.
    cell = cast("ESNCell", reservoir.cell)

    activation = getattr(cell, "activation", "tanh")
    if activation != "tanh":
        raise NotImplementedError(
            f"effective_spectral_radius only supports the 'tanh' activation "
            f"(Jacobian gate 1 - tanh^2); got {activation!r}."
        )

    weight_hh: torch.Tensor = cell.weight_hh
    W = weight_hh.detach().to(torch.float64)
    n_reservoir = int(W.shape[0])

    h = states.detach().to(torch.float64)
    if h.dim() == 3:
        h = h.reshape(-1, h.shape[-1])
    elif h.dim() != 2:
        raise ValueError(
            f"states must be 2-D (time, N) or 3-D (batch, time, N); got {tuple(states.shape)}."
        )
    if h.shape[-1] != n_reservoir:
        raise ValueError(
            f"states trailing dimension ({h.shape[-1]}) must equal the reservoir size "
            f"({n_reservoir})."
        )
    h = h.to(W.device)

    n_total = h.shape[0]
    if n_total > max_states:
        # Even subsample so the mean covers the whole trajectory, not a prefix.
        idx = torch.linspace(0, n_total - 1, steps=max_states, dtype=torch.float64)
        idx = idx.round().to(torch.long).unique()
        # ``seed`` reserved for a future randomised tie-break; the deterministic
        # even grid needs no RNG, so touching it here keeps the signature honest.
        _ = seed
        h = h[idx]

    alpha = float(getattr(cell, "leak_rate", 1.0))
    identity = torch.eye(n_reservoir, dtype=torch.float64, device=W.device)

    radii = torch.empty(h.shape[0], dtype=torch.float64)
    for i in range(h.shape[0]):
        gate = 1.0 - h[i] ** 2  # f'(pre) = 1 - tanh^2(pre) = 1 - h^2
        jac = alpha * (gate[:, None] * W)
        if alpha != 1.0:
            jac = jac + (1.0 - alpha) * identity
        eigvals = torch.linalg.eigvals(jac)
        radii[i] = eigvals.abs().max()

    return float(radii.mean())


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
