"""Largest-Lyapunov-exponent estimation and Lyapunov-time bookkeeping.

Reservoir forecasts on chaotic systems are best reported in **Lyapunov times**
rather than raw integration steps: one Lyapunov time :math:`t_\\lambda = 1 /
\\lambda_\\mathrm{max}` is the natural clock of a chaotic attractor — the
timescale over which nearby trajectories separate by a factor of :math:`e`.
Expressing a valid-prediction horizon as ``VPT_steps * dt * lambda_max`` makes
it comparable across systems, integration steps, and papers.

This module supplies

* pure, dependency-free conversions between steps and Lyapunov times
  (:func:`lyapunov_time`, :func:`steps_per_lyapunov_time`,
  :func:`steps_to_lyapunov`, :func:`lyapunov_to_steps`);
* a table of **canonical literature** :math:`\\lambda_\\mathrm{max}` values for
  resdag's built-in generators (:data:`CANONICAL_LAMBDA_MAX`,
  :data:`MACKEY_GLASS_LAMBDA`);
* :func:`largest_lyapunov`, a single entry point that dispatches on its argument
  — a known generator *name* (no dependency), a measured *series*, or a
  `tsdynamics` *System* — to a canonical constant or an on-the-fly estimate; and
* :func:`reservoir_lyapunov`, a **native-torch** (no dependency) Benettin/QR
  estimate of the maximal Lyapunov exponent of a trained reservoir's own
  autonomous dynamics — the standard edge-of-chaos / stability diagnostic for
  reservoir computers.

The `tsdynamics <https://pypi.org/project/tsdynamics/>`_ paths
(series-from-data and System-spectrum estimation) are **optional**: every
`tsdynamics` import is lazy and guarded with a friendly :class:`ImportError`
pointing at ``pip install resdag[dynamics]``.  The pure conversions and the
canonical-constant and reservoir-Jacobian paths never touch it.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer

# --------------------------------------------------------------------------- #
# Canonical literature constants.
# --------------------------------------------------------------------------- #

#: Largest Lyapunov exponent :math:`\lambda_\mathrm{max}` for resdag's built-in
#: chaotic generators, keyed by generator name.  These are the accepted
#: literature values for the *standard* parameterisation used by
#: :func:`resdag.lorenz` / :func:`resdag.rossler` (the classic chaotic regime):
#:
#: * ``lorenz`` — Lorenz-63 with ``(sigma, rho, beta) = (10, 28, 8/3)``:
#:   :math:`\lambda_\mathrm{max} \approx 0.9056` (natural-time units, i.e. per
#:   unit :math:`t`, independent of the integration step ``dt``).  Sprott (2003),
#:   *Chaos and Time-Series Analysis*; the widely quoted reservoir-computing
#:   reference value (e.g. Pathak et al. 2018).
#: * ``rossler`` — Rössler with ``(a, b, c) = (0.2, 0.2, 5.7)``:
#:   :math:`\lambda_\mathrm{max} \approx 0.0714`.  Sprott (2003).
#:
#: Both agree with a `tsdynamics` ``lyapunov_spectrum`` estimate to two decimals
#: (verified in ``tests/test_metrics``).  The value is in **inverse natural-time
#: units**; multiply by the integrator step ``dt`` to convert a step count to
#: Lyapunov times.
CANONICAL_LAMBDA_MAX: dict[str, float] = {
    "lorenz": 0.9056,
    "rossler": 0.0714,
    # Kuramoto-Sivashinsky on the canonical L=22 periodic domain — the standard
    # parallel-reservoir benchmark (Pathak et al. 2018, "Model-Free Prediction
    # of Large Spatiotemporally Chaotic Systems from Data").  Its largest
    # Lyapunov exponent is domain-size dependent; ``0.043`` is the literature
    # value for L=22 (Pathak et al.; Edson et al. 2019), so a Lyapunov time is
    # ~23 natural-time units.  Only valid for the default L=22 generator.
    "kuramoto_sivashinsky": 0.043,
}

#: Largest Lyapunov exponent of the Mackey-Glass delay system, keyed by the
#: delay ``tau``.  Mackey-Glass has no single :math:`\lambda_\mathrm{max}`: it
#: depends strongly on ``tau`` (the system is chaotic for ``tau`` above
#: ~16.8 and its exponent grows with the delay).  Values are the standard
#: literature figures for ``(beta, gamma, n) = (0.2, 0.1, 10)``:
#:
#: * ``tau = 17`` — the canonical MG-17 benchmark: :math:`\approx 0.0086`.
#: * ``tau = 23`` — :math:`\approx 0.0097`.
#: * ``tau = 30`` — :math:`\approx 0.0071`.
#:
#: Sources: Farmer (1982), *Chaotic attractors of an infinite-dimensional
#: dynamical system*; Grassberger & Procaccia (1983).  Keyed by the integer
#: delay because the exponent is not a smooth single number across ``tau``.
MACKEY_GLASS_LAMBDA: dict[int, float] = {
    17: 0.0086,
    23: 0.0097,
    30: 0.0071,
}

_DYNAMICS_HINT = (
    "This path needs the optional 'tsdynamics' package. Install it with "
    "`pip install resdag[dynamics]` (requires Python >= 3.12)."
)


def _import_tsdynamics() -> Any:
    """Import :mod:`tsdynamics` lazily, raising a friendly error if missing.

    Returns
    -------
    module
        The imported ``tsdynamics`` module.

    Raises
    ------
    ImportError
        If ``tsdynamics`` is not installed, with a message pointing at the
        ``resdag[dynamics]`` extra.
    """
    try:
        import tsdynamics  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(_DYNAMICS_HINT) from exc
    return tsdynamics


# --------------------------------------------------------------------------- #
# Pure conversions (no dependencies).
# --------------------------------------------------------------------------- #


def lyapunov_time(lambda_max: float) -> float:
    """Return the Lyapunov time :math:`t_\\lambda = 1 / \\lambda_\\mathrm{max}`.

    One Lyapunov time is the characteristic timescale over which infinitesimally
    close trajectories of a chaotic system separate by a factor of :math:`e`.

    Parameters
    ----------
    lambda_max : float
        Largest Lyapunov exponent (must be strictly positive for a chaotic
        system).

    Returns
    -------
    float
        The Lyapunov time :math:`1 / \\lambda_\\mathrm{max}`, in the same time
        units as ``lambda_max`` is inverse to.

    Raises
    ------
    ValueError
        If ``lambda_max`` is not strictly positive.
    """
    if lambda_max <= 0.0:
        raise ValueError(f"lambda_max must be positive to define a Lyapunov time, got {lambda_max}")
    return 1.0 / lambda_max


def steps_per_lyapunov_time(dt: float, lambda_max: float) -> float:
    """Return the number of integration steps in one Lyapunov time.

    A convenience for turning a horizon-in-steps into a horizon-in-Lyapunov-times
    (divide the step count by this) and vice versa.

    Parameters
    ----------
    dt : float
        Integration step (time between consecutive samples).
    lambda_max : float
        Largest Lyapunov exponent (strictly positive).

    Returns
    -------
    float
        ``1 / (dt * lambda_max)`` — the number of samples spanning one Lyapunov
        time :math:`t_\\lambda`.

    Raises
    ------
    ValueError
        If ``dt`` or ``lambda_max`` is not strictly positive.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if lambda_max <= 0.0:
        raise ValueError(f"lambda_max must be positive, got {lambda_max}")
    return 1.0 / (dt * lambda_max)


def steps_to_lyapunov(steps: float, dt: float, lambda_max: float) -> float:
    """Convert a step count to Lyapunov times :math:`\\Lambda t`.

    This is the headline conversion: ``VPT_in_Lyapunov = steps * dt *
    lambda_max``.

    Parameters
    ----------
    steps : float
        Number of integration steps (e.g. a valid-prediction time).
    dt : float
        Integration step.
    lambda_max : float
        Largest Lyapunov exponent (strictly positive).

    Returns
    -------
    float
        The horizon expressed in Lyapunov times, ``steps * dt * lambda_max``.

    Raises
    ------
    ValueError
        If ``dt`` or ``lambda_max`` is not strictly positive.

    Examples
    --------
    A Lorenz-63 valid-prediction time of 357 steps at ``dt = 0.02``:

    >>> round(steps_to_lyapunov(357, dt=0.02, lambda_max=0.9056), 2)
    6.47
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if lambda_max <= 0.0:
        raise ValueError(f"lambda_max must be positive, got {lambda_max}")
    return steps * dt * lambda_max


def lyapunov_to_steps(lyap: float, dt: float, lambda_max: float) -> int:
    """Convert a horizon in Lyapunov times back to a whole number of steps.

    The inverse of :func:`steps_to_lyapunov`, rounded to the nearest integer
    step (horizons are discrete step counts).

    Parameters
    ----------
    lyap : float
        Horizon in Lyapunov times :math:`\\Lambda t`.
    dt : float
        Integration step.
    lambda_max : float
        Largest Lyapunov exponent (strictly positive).

    Returns
    -------
    int
        ``round(lyap / (dt * lambda_max))`` — the equivalent step count.

    Raises
    ------
    ValueError
        If ``dt`` or ``lambda_max`` is not strictly positive.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if lambda_max <= 0.0:
        raise ValueError(f"lambda_max must be positive, got {lambda_max}")
    return int(round(lyap / (dt * lambda_max)))


# --------------------------------------------------------------------------- #
# Dispatching estimator.
# --------------------------------------------------------------------------- #


def _lambda_from_name(name: str, **kw: Any) -> float:
    """Resolve a generator name to a canonical :math:`\\lambda_\\mathrm{max}`.

    Parameters
    ----------
    name : str
        A resdag generator name (case-insensitive), e.g. ``"lorenz"``,
        ``"rossler"``, or ``"mackey_glass"``.
    **kw
        Generator parameters; ``tau`` selects the Mackey-Glass value.

    Returns
    -------
    float
        The canonical largest Lyapunov exponent.

    Raises
    ------
    KeyError
        If the name (or, for Mackey-Glass, the ``tau``) has no canonical value.
    """
    key = name.lower()
    if key in CANONICAL_LAMBDA_MAX:
        return CANONICAL_LAMBDA_MAX[key]
    if key in ("mackey_glass", "mackeyglass", "mackey-glass"):
        tau = int(kw.get("tau", 17))
        if tau not in MACKEY_GLASS_LAMBDA:
            raise KeyError(
                f"No canonical Mackey-Glass Lyapunov exponent for tau={tau}. "
                f"Known values: {sorted(MACKEY_GLASS_LAMBDA)}."
            )
        return MACKEY_GLASS_LAMBDA[tau]
    raise KeyError(
        f"No canonical Lyapunov exponent for generator {name!r}. "
        f"Known chaotic generators: {sorted(CANONICAL_LAMBDA_MAX) + ['mackey_glass']}. "
        f"For a measured series pass the array with from_='data', or a tsdynamics "
        f"System for a high-accuracy spectrum estimate."
    )


def _lambda_from_series(series: Any, *, dt: float, **kw: Any) -> float:
    """Estimate :math:`\\lambda_\\mathrm{max}` from a scalar/vector time series.

    Uses ``tsdynamics.lyapunov_from_data`` (Kantz / Rosenstein slope of the mean
    log-divergence).  This is a **diagnostic** estimate: data-driven exponents
    are noisy and sensitive to the embedding and fit window, so they can differ
    from the canonical value by a large factor.  Prefer :data:`CANONICAL_LAMBDA_MAX`
    (a known generator name) or a `tsdynamics` System when accuracy matters.
    """
    tsdynamics = _import_tsdynamics()
    arr = np.asarray(series, dtype=float)
    # Reduce to a scalar 1-D series: the data-based estimators embed a single
    # 1-D signal, so use the first channel of the first trajectory.  For a 3-D
    # ``(batch, time, dim)`` array take ``arr[0]`` first — concatenating the
    # trajectories would inject a spurious discontinuity at each batch boundary.
    if arr.ndim == 3:
        arr = arr[0]  # (time, dim)
    if arr.ndim == 2:
        arr = arr[:, 0]  # (time,)
    if arr.ndim != 1:
        raise ValueError(
            "series must be 1-D (time,), 2-D (time, dim), or 3-D (batch, time, dim); "
            f"got shape {np.asarray(series).shape}."
        )
    analysis = tsdynamics.analysis if hasattr(tsdynamics, "analysis") else tsdynamics
    result = analysis.lyapunov_from_data(arr, dt=dt, **kw)
    # ``lyapunov_from_data`` returns a small result object exposing ``.lyapunov``
    # (and is float-castable); fall back to float() for forward compatibility.
    value = getattr(result, "lyapunov", None)
    return float(value) if value is not None else float(result)


def _lambda_from_system(system: Any, *, dt: float | None, **kw: Any) -> float:
    """Estimate :math:`\\lambda_\\mathrm{max}` from a `tsdynamics` System.

    Uses ``tsdynamics.analysis.lyapunov_spectrum`` (Benettin QR integration of the
    variational equations) and returns the leading exponent — a high-accuracy
    estimate.  Falls back to the system's own ``known_lyapunov`` spectrum when
    available and no override kwargs force a recompute is *not* done here: the
    spectrum is always computed so custom parameters are honoured.
    """
    tsdynamics = _import_tsdynamics()
    call_kw = dict(kw)
    if dt is not None:
        call_kw.setdefault("dt", dt)
    analysis = tsdynamics.analysis if hasattr(tsdynamics, "analysis") else tsdynamics
    spectrum = analysis.lyapunov_spectrum(system, **call_kw)
    # Current versions return an ArrayResult with .values; older versions
    # returned a bare array. The first entry is the largest exponent.
    values = getattr(spectrum, "values", spectrum)
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("tsdynamics.lyapunov_spectrum returned an empty spectrum.")
    return float(arr[0])


def largest_lyapunov(
    source: Any,
    *,
    dt: float | None = None,
    from_: str | None = None,
    **kw: Any,
) -> float:
    """Estimate the largest Lyapunov exponent of ``source``.

    A single entry point that dispatches on the type of ``source``:

    * **generator name** (:class:`str`, e.g. ``"lorenz"``, ``"mackey_glass"``) —
      returns the canonical literature value from :data:`CANONICAL_LAMBDA_MAX`
      / :data:`MACKEY_GLASS_LAMBDA` (keyed by ``tau=`` for Mackey-Glass).
      **No dependency.**
    * **time series** (:class:`numpy.ndarray` / :class:`torch.Tensor`) — a
      *diagnostic* estimate via ``tsdynamics.lyapunov_from_data``.  Data-driven
      exponents are noisy; you must opt in with ``from_="data"`` so a series is
      never silently mistaken for accurate.  **Needs** ``resdag[dynamics]``.
    * **tsdynamics System** — a high-accuracy Benettin estimate via
      ``tsdynamics.lyapunov_spectrum``.  **Needs** ``resdag[dynamics]``.

    Parameters
    ----------
    source : str, numpy.ndarray, torch.Tensor, or tsdynamics System
        What to estimate the exponent of; see above.
    dt : float, optional
        Integration step.  Required for the series path; forwarded to the
        System path if given.  Ignored for a generator name.
    from_ : {"data", None}, optional
        Set to ``"data"`` to force the noisy data-driven estimate for an array
        source.  Without it, passing a raw array raises to prevent silently
        trusting a diagnostic number.
    **kw
        Extra arguments: ``tau=`` selects the Mackey-Glass canonical value;
        otherwise forwarded to the underlying `tsdynamics` estimator (e.g.
        ``method=``, ``fit=`` for data; ``final_time=``, ``transient=`` for a
        System).

    Returns
    -------
    float
        The estimated largest Lyapunov exponent (inverse natural-time units).

    Raises
    ------
    ImportError
        If a `tsdynamics` path is taken but the package is not installed.
    TypeError
        If an array is passed without ``from_="data"``.
    KeyError
        If a generator name has no canonical value.

    Examples
    --------
    >>> largest_lyapunov("lorenz")
    0.9056
    >>> largest_lyapunov("mackey_glass", tau=17)
    0.0086
    """
    if isinstance(source, str):
        return _lambda_from_name(source, **kw)

    if isinstance(source, torch.Tensor):
        source = source.detach().cpu().numpy()

    if isinstance(source, np.ndarray):
        if from_ != "data":
            raise TypeError(
                "Estimating a Lyapunov exponent from a raw data series is a noisy "
                "diagnostic; pass from_='data' (and dt=...) to opt in explicitly, "
                "or use a generator name / tsdynamics System for an accurate value."
            )
        if dt is None:
            raise ValueError("dt is required to estimate a Lyapunov exponent from data.")
        return _lambda_from_series(source, dt=dt, **kw)

    # Anything else is assumed to be a tsdynamics System (dispatched lazily so we
    # never import tsdynamics just to check the type).
    return _lambda_from_system(source, dt=dt, **kw)


# --------------------------------------------------------------------------- #
# Reservoir Lyapunov exponent (native torch, no dependency).
# --------------------------------------------------------------------------- #


def _resolve_reservoir(source: Any) -> "BaseReservoirLayer":
    """Return the single ESN reservoir layer to analyse from ``source``.

    Accepts a :class:`BaseReservoirLayer` directly, or an :class:`ESNModel`
    (or any ``nn.Module``) containing exactly one tanh-style ESN reservoir.

    Raises
    ------
    ValueError
        If no ESN reservoir, or more than one, is found in a model.
    TypeError
        If the reservoir's cell exposes no ``weight_hh`` (not an ESN cell).
    """
    from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer

    if isinstance(source, BaseReservoirLayer):
        reservoir = source
    else:
        reservoirs = [m for m in source.modules() if isinstance(m, BaseReservoirLayer)]
        # A NG-RC reservoir has no recurrent weight matrix, so it has no
        # reservoir Lyapunov exponent in this sense; keep only ESN-style cells.
        reservoirs = [r for r in reservoirs if getattr(r.cell, "weight_hh", None) is not None]
        if not reservoirs:
            raise ValueError(
                "reservoir_lyapunov needs an ESN reservoir (a recurrent weight "
                "matrix); none was found in the model."
            )
        if len(reservoirs) > 1:
            raise ValueError(
                f"Model has {len(reservoirs)} ESN reservoirs; pass a single "
                "ESNLayer to reservoir_lyapunov to disambiguate."
            )
        reservoir = reservoirs[0]

    if getattr(reservoir.cell, "weight_hh", None) is None:
        raise TypeError("reservoir_lyapunov requires a reservoir whose cell owns 'weight_hh'.")
    return reservoir


def _diagnostic_cell(source: Any, *, tanh_esn: bool = False) -> Any:
    """Copy a cell for CPU float64 diagnostics without mutating live state/mode.

    Jacobian formulas here describe the standard ESN update exactly, including
    its calibration and vector leak. Other update equations need their own
    validated tangent map; a ``weight_hh`` attribute alone is insufficient.
    """
    from resdag.layers.cells import ESNCell

    reservoir = _resolve_reservoir(source)
    if tanh_esn:
        if type(reservoir.cell) is not ESNCell:
            raise NotImplementedError(
                "This Jacobian diagnostic supports the standard ESNCell only; "
                "other reservoir dynamics need a validated tangent map."
            )
        if reservoir.cell.activation != "tanh":
            raise NotImplementedError("This Jacobian diagnostic requires the 'tanh' activation.")
    return copy.deepcopy(reservoir.cell).cpu().to(dtype=torch.float64).eval()


def _drive_cell(cell: Any, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the actual cell update locally, retaining no state on the caller."""
    state = cell.init_state(inputs[0].shape[0], inputs[0].device, inputs[0].dtype)
    projected = cell.project_inputs(inputs)
    outputs = []
    with torch.no_grad():
        for t in range(inputs[0].shape[1]):
            if projected is None:
                output, state = cell([stream[:, t] for stream in inputs], state)
            else:
                output, state = cell.step(projected[:, t], state)
            outputs.append(output)
    return torch.stack(outputs, dim=1), state


def _esn_recurrent_jacobian_weight(cell: Any) -> torch.Tensor:
    """Derivative of calibrated preactivation with respect to previous state."""
    weight = cell.weight_hh.detach()
    if weight.layout != torch.strided:
        weight = weight.to_dense()
    if cell._has_recurrent_gain:
        weight = cell._recurrent_gain[:, None] * weight
    if cell._has_intrinsic:
        weight = cell._intrinsic_gain[:, None] * weight
    return cast(torch.Tensor, weight)


def reservoir_lyapunov(
    source: Any,
    warmup_inputs: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    horizon: int = 1000,
    k: int = 1,
    transient: int = 100,
    seed: int | None = 0,
) -> float | np.ndarray:
    """Largest (or top-``k``) Lyapunov exponent of a driven ESN reservoir.

    Estimates the maximal Lyapunov exponent of the reservoir's own dynamics via
    **Benettin's algorithm with QR reorthonormalisation** of the analytic
    tangent map, iterated along an input-driven reservoir trajectory.  This is a
    sampled local-stability diagnostic. Negative estimates support contraction
    along the supplied driven trajectory; they do not prove a global Echo State
    Property for every input. This is distinct from the forecasted attractor's
    largest Lyapunov exponent.

    A local CPU float64 evaluation copy preserves caller weights, device,
    cached state and training mode. Training-only noise is disabled. The
    validated tangent map supports the standard tanh ``ESNCell``, including
    per-neuron leaks, recurrent calibration gains and intrinsic gains/biases.
    Other update equations are rejected until their tangent maps are validated.

    The reservoir map is the leaky-ESN update

    .. math::

        h_t = (1-\\alpha) h_{t-1} + \\alpha\\, f(W_\\mathrm{fb} x_t
              + W_\\mathrm{in} u_t + W_\\mathrm{hh} h_{t-1} + b),

    whose Jacobian with respect to the state is

    .. math::

        J_t = (1-\\alpha) I + \\alpha\\, \\mathrm{diag}\\!\\big(f'(\\mathrm{pre}_t)\\big)\\, W_\\mathrm{hh},

    with :math:`f' = 1 - \\tanh^2` and calibrated gains included in
    :math:`W_\\mathrm{hh}`. Scalar leakage generalizes row-wise for vector leaks.  The
    driving inputs come from ``warmup_inputs``; the trajectory is looped over
    them if ``horizon`` exceeds their length, so a short synchronisation window
    still yields a long tangent-space average.

    Parameters
    ----------
    source : BaseReservoirLayer or ESNModel
        The reservoir to analyse, or a model containing exactly one ESN
        reservoir.
    warmup_inputs : torch.Tensor or tuple of torch.Tensor
        Driving inputs of shape ``(batch, time, features)``.  A bare tensor is
        the feedback stream; a tuple is ``(feedback, driver, ...)``.  Only the
        first sample of the batch is used (the exponent is a property of the
        dynamics, not the batch).
    horizon : int, default=1000
        Number of tangent-map iterations (after the transient).  Longer gives a
        tighter estimate.
    k : int, default=1
        Number of leading exponents to return.  ``k=1`` returns the scalar
        maximal exponent; ``k>1`` returns the top-``k`` as a ``(k,)`` array.
    transient : int, default=100
        Additional tangent iterations discarded before the ``horizon`` scoring
        iterations. Must be non-negative; it is not subtracted from the horizon.
    seed : int or None, default=0
        Seed for the random initial tangent frame.  Defaults to ``0`` so the
        estimate is fully deterministic (the converged leading exponent is robust
        to the frame regardless).  Pass ``None`` to draw from the process-global
        RNG instead.

    Returns
    -------
    float or numpy.ndarray
        The maximal Lyapunov exponent (``k=1``) or the top-``k`` spectrum as a
        ``(k,)`` array, in **inverse-step units** (per reservoir update).

    Raises
    ------
    ValueError
        If ``source`` has no (or more than one) ESN reservoir, or ``k`` exceeds
        the reservoir size.
    NotImplementedError
        If the cell is not a standard tanh ``ESNCell``.

    Notes
    -----
    Exponents are per **reservoir step**.  Multiply by ``1/dt`` to compare with
    a natural-time attractor exponent from :data:`CANONICAL_LAMBDA_MAX`.

    Examples
    --------
    >>> import torch
    >>> from resdag.layers import ESNLayer
    >>> res = ESNLayer(100, feedback_size=3, spectral_radius=0.9, seed=0)
    >>> feedback = torch.randn(1, 300, 3)
    >>> lam = reservoir_lyapunov(res, feedback, horizon=500)
    >>> lam < 0.0  # contraction along this driven trajectory
    True
    """
    from resdag.data.prepare import _validate_step_count

    horizon = _validate_step_count("horizon", horizon)
    transient = _validate_step_count("transient", transient, allow_zero=True)
    k = _validate_step_count("k", k)
    cell = _diagnostic_cell(source, tanh_esn=True)
    streams = [warmup_inputs] if isinstance(warmup_inputs, torch.Tensor) else list(warmup_inputs)
    if not streams or any(x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 for x in streams):
        raise ValueError("warmup_inputs must contain non-empty (batch, time, features) streams.")
    if any(x.shape[:2] != streams[0].shape[:2] for x in streams):
        raise ValueError("warmup input streams must have matching batch and time dimensions.")
    inputs = [x[:1].detach().cpu().to(dtype=torch.float64) for x in streams]
    if not all(bool(torch.isfinite(x).all()) for x in inputs):
        raise ValueError("warmup inputs must be finite.")
    _, state = _drive_cell(cell, inputs)
    n = int(cell.state_size)
    if k > n:
        raise ValueError(f"k must be in [1, reservoir_size={n}], got {k}.")
    weight = _esn_recurrent_jacobian_weight(cell)
    alpha = torch.as_tensor(cell.leak_rate, dtype=torch.float64)
    if alpha.ndim == 0:
        alpha = alpha.expand(n)
    projected = cell.project_inputs(inputs)
    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    frame = torch.linalg.qr(torch.randn(n, k, dtype=torch.float64, generator=generator))[0]
    log_sums = torch.zeros(k, dtype=torch.float64)
    identity = torch.eye(n, dtype=torch.float64)
    with torch.no_grad():
        for step in range(transient + horizon):
            t = step % inputs[0].shape[1]
            # Match the live cell's calibrated preactivation, then advance via
            # its actual implementation rather than a second update equation.
            pre = projected[:, t] + cell._recurrent(state)
            if cell._has_intrinsic:
                pre = cell._intrinsic_gain * pre + cell._intrinsic_bias
            gate = 1.0 - torch.tanh(pre).reshape(n).square()
            jacobian = alpha[:, None] * gate[:, None] * weight
            jacobian = jacobian + (1.0 - alpha)[:, None] * identity
            _, state = cell.step(projected[:, t], state)
            frame, triangular = torch.linalg.qr(jacobian @ frame)
            if step >= transient:
                growth = triangular.diagonal().abs().clamp_min(1e-300)
                log_sums += growth.log()
    exponents = (log_sums / horizon).numpy()
    return float(exponents[0]) if k == 1 else exponents
