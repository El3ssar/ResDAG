r"""
LIF Reservoir Layer (Liquid State Machine)
==========================================

This module provides the public-facing spiking-reservoir sequence layer:

- :class:`LIFReservoirLayer` — stateful RNN layer wrapping a
  :class:`~resdag.layers.cells.lif_cell.LIFCell`, the reservoir of a Liquid State
  Machine (LSM; Maass, Natschläger & Markram, *Neural Computation* 14(11), 2002).

The old name :class:`LIFReservoir` is retained as a deprecated alias.

References
----------
W. Maass, T. Natschläger and H. Markram, "Real-Time Computing Without Stable
States: A New Framework for Neural Computation Based on Perturbations,"
*Neural Computation* 14(11):2531-2560, 2002.

See Also
--------
resdag.layers.reservoirs.base_reservoir : Abstract base (BaseReservoirLayer).
resdag.layers.cells.lif_cell : Underlying single-step LIF cell.
resdag.layers.reservoirs.esn : Standard leaky-ESN layer.
"""

import warnings
from typing import Any

import torch

from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.cells.lif_cell import LIFCell
from resdag.utils.general import SeedLike

from .base_reservoir import BaseReservoirLayer


class LIFReservoirLayer(BaseReservoirLayer):
    r"""
    Stateful leaky integrate-and-fire (LIF) reservoir layer — the LSM liquid.

    Public-facing class.  Internally creates a
    :class:`~resdag.layers.cells.lif_cell.LIFCell` that owns all parameters and
    performs the single-step spiking update; the sequence loop and state
    management live in the parent
    :class:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer`, so this
    layer shares the full state API (``reset_state`` / ``get_state`` /
    ``set_state`` / ``set_random_state``) and the projected fast path with
    :class:`~resdag.layers.reservoirs.esn.ESNLayer`.

    The reservoir is a recurrent network of spiking neurons.  Its per-step output
    is a real-valued **synaptic trace** (a low-pass-filtered spike train), not the
    binary spikes — so a ridge / CG readout can regress on it directly.  The state
    is a single packed 2-D tensor ``(batch, cell.state_size)`` bundling the
    membrane potential, previous spikes, optional synaptic current, and the trace
    (see :class:`~resdag.layers.cells.lif_cell.LIFCell`).

    .. important::
       A Liquid State Machine is an **input-driven** reservoir: it maps a driving
       signal to a filtered liquid state a readout regresses on.  It is *not* a
       closed-loop autoregressive chaos forecaster — the output is a spike trace,
       not the feedback variable.  Use :func:`~resdag.models.ott_esn` /
       :func:`~resdag.models.es2n` for autoregressive chaotic forecasting.

    .. note::
       Drive the liquid hard enough to spike: with default unit-scale weights
       most neurons may never cross ``v_threshold``, leaving the trace all-zero
       and the readout with nothing to fit.  Raise the input scaling and/or lower
       ``v_threshold`` (check the firing rate via ``output="spikes"``).  See
       :func:`~resdag.models.lsm` for guidance.

    Parameters
    ----------
    reservoir_size : int
        Number of spiking units.
    feedback_size : int
        Dimension of the feedback / driving signal.  Required.
    input_size : int, optional
        Dimension of an optional second driving input.  ``None`` (or ``0``)
        builds a single-input reservoir.
    tau_mem : float, default=20.0
        Membrane time constant :math:`\tau_\text{mem}`.
    tau_syn : float, default=10.0
        Synaptic time constant :math:`\tau_\text{syn}` (CuBa-LIF only).
    dt : float, default=1.0
        Integration timestep.
    synaptic_current : bool, default=False
        If ``True``, use the second-order current-based LIF (CuBa-LIF).
    v_threshold : float, default=1.0
        Firing threshold.
    reset : {'subtract', 'zero'}, default='subtract'
        Membrane reset on a spike.
    surrogate_beta : float, default=10.0
        Fast-sigmoid surrogate slope for the straight-through spike gradient.
    trace_decay : float, default=0.9
        Decay of the low-pass readout trace.
    output : {'trace', 'voltage', 'spikes'}, default='trace'
        Readable-output mode; ``'trace'`` is the ridge-regressable default.
    spectral_radius : float, optional
        Target spectral radius for the recurrent spike-coupling weight ``W_rec``.
    bias : bool, default=True
        Whether to include a bias current.
    bias_scaling : float, default=1.0
        Scale of the random bias (``uniform(-bias_scaling, bias_scaling)``).
    trainable : bool, default=False
        If ``True``, all weights are trainable via the surrogate spike gradient.
    feedback_initializer, input_initializer, topology : spec, optional
        Same accepted forms as :class:`~resdag.layers.ESNLayer`.
    seed : int or torch.Generator, optional
        Reproducibility seed fixing the recurrent, feedback, input and bias draws.

    recurrent_initializer : spec, optional
        First-class alias for ``topology``. Supply one spelling only. Names,
        tuples, callables and registered initializer objects are accepted.
    bias_initializer : spec, optional
        Input/feedback initializer grammar applied to a ``(1, reservoir_size)``
        view of the bias. Requires ``bias=True``; overrides ``bias_scaling``.
    initial_state_initializer : spec, optional
        Initializes a persistent ``(1, cell.state_size)`` reset template once.
        Each batch example receives an independent copy at reset. Zero by default.
    device : torch.device or str, optional
        Initial parameter location. Metal construction initializes on CPU before
        transferring to support initialization operations across backends.
    dtype : torch.dtype, optional
        Floating precision used during initialization, not only after conversion.

    Attributes
    ----------
    state : torch.Tensor or None
        Current packed reservoir state of shape ``(batch, cell.state_size)``, or
        ``None`` before the first forward.

    Examples
    --------
    >>> import torch
    >>> reservoir = LIFReservoirLayer(reservoir_size=200, feedback_size=3, seed=0)
    >>> feedback = torch.randn(4, 50, 3)  # (batch, time, features)
    >>> traces = reservoir(feedback)
    >>> traces.shape
    torch.Size([4, 50, 200])

    See Also
    --------
    resdag.layers.cells.lif_cell.LIFCell : Underlying single-step cell.
    resdag.layers.ESNLayer : Standard leaky-ESN layer.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        dt: float = 1.0,
        synaptic_current: bool = False,
        v_threshold: float = 1.0,
        reset: str = "subtract",
        surrogate_beta: float = 10.0,
        trace_decay: float = 0.9,
        output: str = "trace",
        spectral_radius: float | None = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        trainable: bool = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
        seed: SeedLike = None,
        *,
        recurrent_initializer: TopologySpec = None,
        bias_initializer: InitializerSpec = None,
        initial_state_initializer: InitializerSpec = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        cell = LIFCell(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            input_size=input_size,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            dt=dt,
            synaptic_current=synaptic_current,
            v_threshold=v_threshold,
            reset=reset,
            surrogate_beta=surrogate_beta,
            trace_decay=trace_decay,
            output=output,
            spectral_radius=spectral_radius,
            bias=bias,
            bias_scaling=bias_scaling,
            trainable=trainable,
            feedback_initializer=feedback_initializer,
            input_initializer=input_initializer,
            topology=topology,
            seed=seed,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            initial_state_initializer=initial_state_initializer,
            device=device,
            dtype=dtype,
        )
        super().__init__(cell)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute lookups to the wrapped cell."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Reach into _modules directly to avoid recursion during __init__.
        modules = self.__dict__.get("_modules")
        if modules is not None and "cell" in modules:
            try:
                return getattr(modules["cell"], name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        """Return string representation (delegates to the cell)."""
        return repr(self.cell)


class LIFReservoir(LIFReservoirLayer):
    """Deprecated alias for :class:`LIFReservoirLayer`.

    Retained for backward compatibility.  Instantiating this class emits a
    :class:`DeprecationWarning` and constructs a :class:`LIFReservoirLayer`;
    the two are otherwise identical (``LIFReservoir`` is a subclass, so
    ``isinstance(LIFReservoir(...), LIFReservoirLayer)`` holds).

    .. deprecated::
        Use :class:`LIFReservoirLayer` instead.  ``LIFReservoir`` will be
        removed in a future release.

    See Also
    --------
    LIFReservoirLayer : The canonical class this aliases.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "LIFReservoir is deprecated and will be removed in a future release; "
            "use LIFReservoirLayer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
