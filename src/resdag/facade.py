"""High-level, opinionated ESN facade — train + forecast in ~10 lines.

This module provides :class:`ESN`, a single object that owns the end-to-end
reservoir-computing workflow.  It hides the symbolic-graph plumbing
(:func:`~resdag.models.classic_esn`), data slicing, and the
:class:`~resdag.training.ESNTrainer` ↔ :meth:`~resdag.core.ESNModel.forecast`
hand-off behind two chainable methods::

    from resdag import ESN

    esn = ESN(reservoir_size=400, spectral_radius=0.9).fit(series)
    prediction = esn.forecast(horizon=500)

It is the friendly front door, *not* a replacement for the composable
``pytorch_symbolic`` path.  The underlying :class:`~resdag.core.ESNModel` is
always reachable via :attr:`ESN.model`, so advanced users can drop down to the
full building-block API at any time without rebuilding anything.

See Also
--------
resdag.models.classic_esn : The builder this facade wraps.
resdag.training.ESNTrainer : The trainer this facade drives.
resdag.core.ESNModel.forecast : The autoregressive forecaster this facade calls.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from resdag.core import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.models import classic_esn
from resdag.training import ESNTrainer


class ESN:
    """Opinionated, end-to-end Echo State Network — ``fit().forecast()``.

    A thin, friendly wrapper that owns the whole reservoir-computing workflow so
    a researcher can train on a time series and roll out an autoregressive
    forecast of a chaotic system in roughly ten lines, without touching the
    ``pytorch_symbolic`` graph, data slicing, or the trainer/forecast hand-off
    by hand.

    Hyperparameters are stored at construction; the underlying
    :class:`~resdag.core.ESNModel` is built lazily on the first :meth:`fit` call
    (once the feature dimension is known from the data).  :meth:`fit` slices off
    the first ``washout`` steps for state synchronisation, builds the
    one-step-ahead training target internally, and drives
    :class:`~resdag.training.ESNTrainer`.  :meth:`forecast` resets the reservoir,
    re-synchronises on the tail of the training series, and runs
    :meth:`~resdag.core.ESNModel.forecast`.

    The wrapper is deliberately non-hiding: the composed model is exposed as
    :attr:`model`, so the full building-block API (custom transforms, multiple
    readouts, :meth:`~resdag.core.ESNModel.save_full`, ...) remains available.

    Parameters
    ----------
    reservoir_size : int, default=500
        Number of reservoir units.
    spectral_radius : float, default=0.9
        Target spectral radius of the recurrent weights (memory / stability).
    leak_rate : float, default=1.0
        Leaky-integration rate (``1.0`` = no leak, the standard ESN).
    washout : int, default=100
        Number of initial steps used to synchronise the reservoir (the "warmup"
        window).  These steps teacher-force the state and are *not* used as
        training targets.  Also the length of the re-synchronisation window
        used at the start of :meth:`forecast`.
    alpha : float, default=1e-6
        Ridge-regression (L2) regularisation strength for the readout.
    topology : TopologySpec, optional
        Recurrent-weight topology.  Accepts a registry name (``"erdos_renyi"``),
        a ``(name, params)`` tuple, a pre-built
        :class:`~resdag.init.topology.TopologyInitializer`, or ``None`` for the
        :func:`~resdag.models.classic_esn` default.
    feedback_initializer : InitializerSpec, optional
        Initializer for the feedback (input) weights.  Same three-way spec as
        ``topology``.
    activation : str, default="tanh"
        Reservoir activation: ``"tanh"``, ``"relu"``, ``"sigmoid"`` or
        ``"identity"``.
    bias : bool, default=True
        Whether the reservoir uses a bias term.
    readout_bias : bool, default=True
        Whether the readout uses a bias term.
    seed : int, optional
        If given, seeds ``torch`` before the model is built so the random
        reservoir weights are reproducible.
    device : torch.device or str, optional
        Device the model and data live on.  If ``None``, the device of the
        series passed to the first :meth:`fit` is adopted (CPU for numpy input),
        so a CUDA tensor "just works" without passing ``device`` explicitly.
    dtype : torch.dtype, optional
        Floating dtype for the model and data.  If ``None``, a floating-point
        series keeps its own dtype (a ``float64`` series is **not** silently
        downcast); an integer/bool series is promoted to
        ``torch.get_default_dtype()``.
    **reservoir_kwargs : Any
        Extra keyword arguments forwarded to
        :class:`~resdag.layers.ESNLayer` via
        :func:`~resdag.models.classic_esn`.

    Attributes
    ----------
    model : resdag.core.ESNModel or None
        The composed model, available after :meth:`fit`.  ``None`` beforehand.
    feature_size : int or None
        Number of features ``D`` inferred from the training series.
    readout_name : str
        Name of the internal readout layer (matches the trainer's targets key).

    Examples
    --------
    Train and forecast a chaotic system in ~10 lines:

    >>> import numpy as np
    >>> from resdag import ESN
    >>> rng = np.random.default_rng(0)
    >>> series = np.cumsum(rng.standard_normal((2000, 3)), axis=0)  # (T, D)
    >>> esn = ESN(reservoir_size=300, spectral_radius=0.9, washout=100)
    >>> prediction = esn.fit(series).forecast(horizon=200)
    >>> prediction.shape  # numpy in -> numpy out, shape (horizon, D)
    (200, 3)

    Drop down to the composed model for advanced use:

    >>> esn.model.summary()                 # doctest: +SKIP
    >>> states = esn.model.get_reservoir_states()   # doctest: +SKIP

    See Also
    --------
    resdag.models.classic_esn : Lower-level builder this facade composes.
    resdag.training.ESNTrainer : Trainer this facade drives.
    resdag.core.ESNModel.forecast : Autoregressive forecaster this facade calls.

    Notes
    -----
    - **Stateful reset.** :meth:`forecast` always resets the reservoir and
      re-synchronises on the training tail, so back-to-back forecasts are
      independent and reproducible — no manual ``reset_reservoirs()`` needed.
    - **Readout name.** The internal readout is named consistently and the
      trainer is fed a matching ``{name: target}`` dict, so the
      "readout name must match the targets key" footgun is handled for you.
    - **Slot-0 forecast.** The first output dimension equals the feedback
      dimension by construction, so :meth:`forecast` is genuinely
      autoregressive from its first emitted step.
    """

    def __init__(
        self,
        reservoir_size: int = 500,
        *,
        spectral_radius: float = 0.9,
        leak_rate: float = 1.0,
        washout: int = 100,
        alpha: float = 1e-6,
        topology: TopologySpec = None,
        feedback_initializer: InitializerSpec = None,
        activation: str = "tanh",
        bias: bool = True,
        readout_bias: bool = True,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **reservoir_kwargs: Any,
    ) -> None:
        if reservoir_size < 1:
            raise ValueError(f"reservoir_size must be >= 1, got {reservoir_size}")
        if washout < 1:
            raise ValueError(f"washout must be >= 1, got {washout}")

        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.washout = washout
        self.alpha = alpha
        self.topology = topology
        self.feedback_initializer = feedback_initializer
        self.activation = activation
        self.bias = bias
        self.readout_bias = readout_bias
        self.seed = seed
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.reservoir_kwargs = reservoir_kwargs

        # Fixed internal readout name so the trainer's targets key always
        # matches — the caller never has to reason about it.
        self.readout_name = "output"

        # Built lazily on first fit(), once the feature dimension is known.
        self.model: ESNModel | None = None
        self.feature_size: int | None = None

        # Remembered across fit -> forecast: the synchronisation tail and the
        # original input type, so forecast() can re-sync the state and return
        # the same array type the caller fitted with.
        self._sync_tail: torch.Tensor | None = None
        self._returns_numpy: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, series: Any) -> "ESN":
        """Build the model (if needed) and fit the readout on ``series``.

        The series is split into a ``washout`` synchronisation prefix and a
        training window; the one-step-ahead target is built internally
        (``x[t] -> x[t + 1]``) and :class:`~resdag.training.ESNTrainer` fits the
        readout in a single forward pass.

        Parameters
        ----------
        series : torch.Tensor or numpy.ndarray
            Time series of shape ``(time, features)`` or
            ``(batch, time, features)``.  A 2-D array is treated as a single
            batch.  Must contain at least ``washout + 2`` timesteps (one
            washout window plus one input/target pair).  numpy input is
            converted to torch and the original type is remembered so
            :meth:`forecast` returns numpy as well.

        Returns
        -------
        ESN
            ``self``, to allow ``ESN(...).fit(series).forecast(...)`` chaining.

        Raises
        ------
        ValueError
            If ``series`` is not 2-D/3-D, or is too short for the configured
            ``washout``, or its feature dimension changes between successive
            :meth:`fit` calls.

        Examples
        --------
        >>> esn = ESN(reservoir_size=200).fit(series)   # doctest: +SKIP
        """
        data, returns_numpy = self._to_3d_tensor(series)
        self._returns_numpy = returns_numpy

        batch, timesteps, feature_size = data.shape
        # Need: washout warmup + at least one (input, shifted-target) pair.
        if timesteps < self.washout + 2:
            raise ValueError(
                f"series has {timesteps} timesteps but at least washout + 2 = "
                f"{self.washout + 2} are required (a {self.washout}-step washout "
                f"plus one input/target step). Lower `washout` or pass a longer series."
            )

        # Lazily build the model on the first fit; reuse it on refits as long as
        # the feature dimension is unchanged (refitting overwrites the readout).
        if self.model is None:
            self.feature_size = feature_size
            self.model = self._build_model(feature_size)
        elif feature_size != self.feature_size:
            raise ValueError(
                f"This ESN was fit on {self.feature_size}-feature data; got "
                f"{feature_size}-feature data. Create a new ESN for a different "
                f"feature dimension."
            )

        # Split into warmup + autoregressive (input, one-step-ahead target).
        #   warmup : [0,             washout)
        #   train  : [washout,       T - 1)   -> predicts...
        #   target : [washout + 1,   T)
        warmup = data[:, : self.washout, :]
        train = data[:, self.washout : timesteps - 1, :]
        target = data[:, self.washout + 1 :, :]

        trainer = ESNTrainer(self.model)
        trainer.fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets={self.readout_name: target},
        )

        # Remember the tail of the *training* series so forecast() can
        # re-synchronise the reservoir from a known, teacher-forced window.
        self._sync_tail = data[:, timesteps - self.washout :, :].clone()
        return self

    def forecast(
        self,
        horizon: int,
        *,
        return_warmup: bool = False,
    ) -> Any:
        """Roll out an autoregressive forecast continuing the fitted series.

        Resets the reservoir, re-synchronises it on the final ``washout`` steps
        of the training series, then autoregressively generates ``horizon``
        steps via :meth:`~resdag.core.ESNModel.forecast`.  Because the state is
        always reset first, successive calls are independent.

        Parameters
        ----------
        horizon : int
            Number of future steps to generate. Must be ``>= 1``.
        return_warmup : bool, default=False
            If ``True``, prepend the (teacher-forced) re-synchronisation outputs
            to the result, giving ``(washout + horizon, features)`` per batch.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            Forecast of shape ``(horizon, features)`` for a single-batch fit, or
            ``(batch, horizon, features)`` when fit on batched data.  The array
            type matches what was passed to :meth:`fit` (numpy in → numpy out).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If ``horizon < 1``.

        Examples
        --------
        >>> prediction = esn.fit(series).forecast(horizon=500)   # doctest: +SKIP
        """
        if self.model is None or self._sync_tail is None:
            raise RuntimeError("forecast() called before fit(). Call fit(series) first.")
        if horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        # forecast() resets the reservoir internally (reset=True), so two
        # successive forecasts are independent — no manual reset needed.
        prediction = self.model.forecast(
            self._sync_tail,
            horizon=horizon,
            return_warmup=return_warmup,
        )
        # classic_esn is single-output, so forecast returns a single tensor.
        assert isinstance(prediction, torch.Tensor)
        return self._from_tensor(prediction)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_model(self, feature_size: int) -> ESNModel:
        """Compose the underlying :class:`~resdag.core.ESNModel`.

        Output size is pinned to ``feature_size`` so the first (only) output can
        seed the autoregressive forecast loop — the slot-0 forecast contract.
        The model is moved onto the working ``dtype``/``device`` (resolved from
        the first fitted series) so its weights match the data exactly — e.g. a
        ``float64`` series gets a ``float64`` reservoir, avoiding a downstream
        "expected Float but found Double" mismatch.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        model = classic_esn(
            reservoir_size=self.reservoir_size,
            feedback_size=feature_size,
            output_size=feature_size,
            topology=self.topology,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            feedback_initializer=self.feedback_initializer,
            activation=self.activation,
            bias=self.bias,
            readout_alpha=self.alpha,
            readout_bias=self.readout_bias,
            readout_name=self.readout_name,
            **self.reservoir_kwargs,
        )
        # ``nn.Module.to`` is annotated to return ``Any`` in the torch stubs;
        # it returns ``self`` in practice, so the cast is purely for mypy.
        return cast(ESNModel, model.to(device=self.device, dtype=self.dtype))

    def _to_3d_tensor(self, series: Any) -> tuple[torch.Tensor, bool]:
        """Coerce numpy/torch input to a ``(batch, time, features)`` tensor.

        Resolves dtype and device once (on the first fit), preserving a
        floating input's own dtype unless the caller pinned one — so a
        ``float64`` chaotic series is not silently downcast — and adopting the
        input's device when none was given.  Returns the tensor together with a
        flag recording whether the caller passed numpy, so :meth:`forecast` can
        mirror the input type.
        """
        returns_numpy = isinstance(series, np.ndarray)
        if returns_numpy:
            tensor = torch.as_tensor(series)
        elif isinstance(series, torch.Tensor):
            tensor = series
        else:
            raise TypeError(
                f"series must be a torch.Tensor or numpy.ndarray, got {type(series).__name__}."
            )

        # Resolve the working dtype/device on the first fit so the model is
        # built to match and later refits stay consistent. Preserve a floating
        # input's own dtype (no silent float64 -> float32 downcast); only an
        # integer/bool series is promoted to a default float, since the
        # reservoir math is float-only.
        if self.dtype is None:
            self.dtype = tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
        if self.device is None:
            self.device = tensor.device

        tensor = tensor.to(dtype=self.dtype, device=self.device)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (T, D) -> (1, T, D)
        elif tensor.dim() != 3:
            raise ValueError(
                f"series must be 2-D (time, features) or 3-D (batch, time, features); "
                f"got a {tensor.dim()}-D array with shape {tuple(tensor.shape)}."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(
                "series contains NaN or infinite values; clean the data before "
                "fitting (the ridge-regression readout would otherwise produce an "
                "all-NaN forecast)."
            )
        return tensor, returns_numpy

    def _from_tensor(self, prediction: torch.Tensor) -> Any:
        """Mirror the caller's input: squeeze single batch, numpy if asked.

        A single-batch fit yields ``(1, horizon, D)``; the leading axis is
        squeezed so the common case returns the intuitive ``(horizon, D)``.
        numpy input round-trips back to numpy.
        """
        if prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
        if self._returns_numpy:
            return prediction.detach().cpu().numpy()
        return prediction

    def __repr__(self) -> str:
        fitted = self.model is not None
        return (
            f"ESN(reservoir_size={self.reservoir_size}, "
            f"spectral_radius={self.spectral_radius}, leak_rate={self.leak_rate}, "
            f"washout={self.washout}, alpha={self.alpha}, fitted={fitted})"
        )


__all__ = ["ESN"]
