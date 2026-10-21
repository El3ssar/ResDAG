"""
Gated Reservoir Layer
=====================

This module provides the public-facing sequence layer for the gated
("reservoir GRU") cell:

- :class:`GatedReservoirLayer` — stateful RNN layer wrapping
  :class:`~resdag.layers.cells.gated_cell.GatedReservoirCell`.

The old name :class:`GatedReservoir` is retained as a deprecated alias.

See Also
--------
resdag.layers.cells.gated_cell : Single-step light-gated cell (GatedReservoirCell).
resdag.layers.reservoirs.base_reservoir : Abstract base class (BaseReservoirLayer).
resdag.layers.reservoirs.esn : The ungated ESN sequence layer (ESNLayer).
"""

import warnings
from typing import Any

import torch

from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.cells.gated_cell import GatedReservoirCell
from resdag.utils.general import SeedLike

from .base_reservoir import BaseReservoirLayer


class GatedReservoirLayer(BaseReservoirLayer):
    """
    Stateful RNN reservoir layer for the gated (LiGRU-style) reservoir.

    Public-facing class.  Internally creates a
    :class:`~resdag.layers.cells.gated_cell.GatedReservoirCell` that owns every
    weight and performs the single-step light-gated update; the sequence loop and
    state management live in the parent :class:`BaseReservoirLayer`, so this layer
    shares the full state API (``reset_state`` / ``get_state`` / ``set_state`` /
    ``set_random_state``), the projected fast path, and the compiled
    teacher-forced scan with :class:`~resdag.layers.reservoirs.esn.ESNLayer`.

    The state is a single packed 2-D tensor ``(batch, reservoir_size)`` (the GRU
    hidden state ``h``), following the packed multi-tensor-state convention
    documented in :mod:`resdag.layers.cells.gated_cell`.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (hidden state dimension).
    feedback_size : int
        Dimension of the feedback signal.  Required.
    input_size : int, optional
        Dimension of an optional driving input.  ``None`` (or ``0``) builds a
        feedback-only reservoir.
    spectral_radius : float, optional
        Target spectral radius for the recurrent **candidate** weight.  Controls
        the echo of past state through the candidate path.  ``None`` applies no
        scaling.
    bias : bool, default=True
        Whether to include the gate biases.
    bias_scaling : float, default=1.0
        Scale of the random candidate bias ``b_h``.  The update-gate bias is the
        constant retention bias (see ``update_gate_bias``), not randomised.
    update_gate_bias : float, default=2.0
        Constant update-gate bias biasing ``z`` toward retention (``z≈0.88`` at
        ``2.0``).  Keeps a frozen gated reservoir from washing out its memory.
    activation : {'relu', 'tanh', 'identity', 'sigmoid'}, default='relu'
        Candidate nonlinearity (LiGRU default is ``"relu"``).
    gate_activation : {'sigmoid', ...}, default='sigmoid'
        Update-gate nonlinearity; ``"sigmoid"`` keeps ``z`` a valid gate.
    trainable : bool, default=False
        If ``True``, all weights are trainable via backpropagation.  A frozen
        gated reservoir often underperforms an ESN — gated reservoirs are usually
        best used with ``trainable=True``.
    feedback_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Initializer for the feedback drive weights.  Same accepted forms as
        :class:`~resdag.layers.ESNLayer`.
    input_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Initializer for the driving-input drive weights.  Only used when
        ``input_size`` is given.
    topology : str, callable, tuple, or TopologyInitializer, optional
        Structure of the recurrent candidate weight.  The update-gate recurrent
        weight is always dense-random.
    seed : int or torch.Generator, optional
        Reproducibility seed fixing the entire reservoir.

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

    gate_recurrent_initializer : spec, optional
        Separate recurrent gate matrix. Defaults to small dense random weights.
    gate_spectral_radius : float, optional
        Outer radius target for the gate recurrence, respecting its scaling policy.
    gate_feedback_initializer, gate_input_initializer : spec, optional
        Separate gate projections. Omitted projections share candidate weights.
        A driver gate initializer requires ``input_size``.
    gate_bias_initializer : spec, optional
        Gate-bias initializer on ``(1, reservoir_size)``; overrides
        ``update_gate_bias``. Requires ``bias=True``.

    Attributes
    ----------
    state : torch.Tensor or None
        Current packed reservoir state of shape ``(batch, reservoir_size)``, or
        ``None`` if not yet initialized.

    Examples
    --------
    Basic feedback-only gated reservoir:

    >>> reservoir = GatedReservoirLayer(reservoir_size=200, feedback_size=3)
    >>> feedback = torch.randn(4, 50, 3)     # (batch, time, features)
    >>> states = reservoir(feedback)
    >>> states.shape
    torch.Size([4, 50, 200])

    Trainable gated reservoir (the interesting configuration):

    >>> reservoir = GatedReservoirLayer(200, feedback_size=3, trainable=True)

    See Also
    --------
    resdag.layers.cells.gated_cell.GatedReservoirCell : Underlying single-step cell.
    resdag.layers.reservoirs.esn.ESNLayer : The ungated ESN layer.
    resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer : State-management base.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        spectral_radius: float | None = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        update_gate_bias: float = 2.0,
        activation: str = "relu",
        gate_activation: str = "sigmoid",
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
        gate_recurrent_initializer: TopologySpec = None,
        gate_feedback_initializer: InitializerSpec = None,
        gate_input_initializer: InitializerSpec = None,
        gate_bias_initializer: InitializerSpec = None,
        gate_spectral_radius: float | None = None,
    ) -> None:
        cell = GatedReservoirCell(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            input_size=input_size,
            spectral_radius=spectral_radius,
            bias=bias,
            bias_scaling=bias_scaling,
            update_gate_bias=update_gate_bias,
            activation=activation,
            gate_activation=gate_activation,
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
            gate_recurrent_initializer=gate_recurrent_initializer,
            gate_feedback_initializer=gate_feedback_initializer,
            gate_input_initializer=gate_input_initializer,
            gate_bias_initializer=gate_bias_initializer,
            gate_spectral_radius=gate_spectral_radius,
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


class GatedReservoir(GatedReservoirLayer):
    """Deprecated alias for :class:`GatedReservoirLayer`.

    Retained for backward compatibility.  Instantiating this class emits a
    :class:`DeprecationWarning` and constructs a :class:`GatedReservoirLayer`;
    the two are otherwise identical (``GatedReservoir`` is a subclass, so
    ``isinstance(GatedReservoir(...), GatedReservoirLayer)`` holds).

    .. deprecated::
        Use :class:`GatedReservoirLayer` instead.  ``GatedReservoir`` will be
        removed in a future release.

    See Also
    --------
    GatedReservoirLayer : The canonical class this aliases.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "GatedReservoir is deprecated and will be removed in a future release; "
            "use GatedReservoirLayer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
