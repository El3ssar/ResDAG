"""
ESN Cell and Layer
==================

This module provides the concrete Echo State Network (ESN) implementation of
the cell/layer hierarchy:

- :class:`ESNCell` — single-timestep leaky-ESN update; owns all parameters.
- :class:`ESNLayer` — public-facing sequence layer; drop-in replacement for
  the legacy ``ReservoirLayer``.

See Also
--------
resdag.layers.base : Abstract base classes (ReservoirCell, BaseReservoirLayer).
resdag.layers.ReservoirLayer : Backwards-compatible alias for ESNLayer.
resdag.init.topology : Topology initialization for recurrent weights.
resdag.init.input_feedback : Input/feedback weight initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from resdag.init.utils import InitializerSpec, TopologySpec, resolve_initializer, resolve_topology

from .base import BaseReservoirLayer, ReservoirCell


class ESNCell(ReservoirCell):
    """
    Single-timestep leaky Echo State Network update.

    Owns all weight matrices and bias.  Sequence iteration is delegated to
    the enclosing :class:`ESNLayer`.

    The state update follows:

    .. math::

        h_t = f((1 - \\alpha)\\,h_{t-1} + \\alpha\\,g(W_{fb}\\,x_{fb,t}
               + W_{in}\\,x_{in,t} + W_{rec}\\,h_{t-1} + b))

    where :math:`f` is the activation function, :math:`\\alpha` is the leak
    rate, :math:`W_{fb}` is the feedback weight matrix, :math:`W_{in}` is the
    (optional) input weight matrix, and :math:`W_{rec}` is the recurrent
    weight matrix.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (hidden state dimension).
    feedback_size : int
        Dimension of feedback signal.  Required for all ESN cells.
    input_size : int, optional
        Dimension of driving inputs.  If ``None``, no driving input weight
        matrix is created.
    spectral_radius : float, optional
        Target spectral radius for recurrent weights.  If ``None``, no
        spectral radius scaling is applied.
    bias : bool, default=True
        Whether to include a bias term.
    activation : {'tanh', 'relu', 'identity', 'sigmoid'}, default='tanh'
        Activation function for reservoir dynamics.
    leak_rate : float, default=1.0
        Leaky integration rate in [0, 1].  A value of 1.0 means no leaking
        (standard RNN update); smaller values create slower dynamics.
    trainable : bool, default=False
        If ``True``, reservoir weights are trainable via backpropagation.
        Standard ESNs use frozen (non-trainable) weights.
    feedback_initializer : str, tuple, or InputFeedbackInitializer, optional
        Initializer for the feedback weight matrix.
    input_initializer : str, tuple, or InputFeedbackInitializer, optional
        Initializer for the input weight matrix.  Only used when
        ``input_size`` is provided.
    topology : str, tuple, or TopologyInitializer, optional
        Graph topology for recurrent weights.

    Attributes
    ----------
    weight_feedback : torch.nn.Parameter
        Feedback weight matrix of shape ``(reservoir_size, feedback_size)``.
    weight_input : torch.nn.Parameter or None
        Input weight matrix of shape ``(reservoir_size, input_size)``, or
        ``None`` if ``input_size`` was not provided.
    weight_hh : torch.nn.Parameter
        Recurrent weight matrix of shape ``(reservoir_size, reservoir_size)``.
    bias_h : torch.nn.Parameter or None
        Bias vector of shape ``(reservoir_size,)``, or ``None`` if
        ``bias=False``.

    See Also
    --------
    resdag.layers.esn.ESNLayer : Layer that sequences this cell.
    resdag.layers.base.ReservoirCell : Abstract cell interface.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        spectral_radius: float | None = None,
        bias: bool = True,
        activation: str = "tanh",
        leak_rate: float = 1.0,
        trainable: bool = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
    ) -> None:
        super().__init__()

        # Store configuration
        self.reservoir_size = reservoir_size
        self.feedback_size = feedback_size
        self.input_size = input_size
        self.topology = topology
        self.spectral_radius = spectral_radius
        self.feedback_initializer = feedback_initializer
        self.input_initializer = input_initializer
        self.leak_rate = leak_rate
        self.trainable = trainable

        # Activation function
        self._activation_name = activation
        self.activation_fn = self._get_activation(activation)

        # Store bias flag before initialization
        self._bias = bias

        # Initialize weight matrices
        self._initialize_weights()

        # Freeze weights if not trainable
        if not self.trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # ReservoirCell interface
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        """Dimensionality of the hidden state vector."""
        return self.reservoir_size

    @property
    def activation(self) -> str:
        """
        str : Name of the activation function.
        """
        return self._activation_name

    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the next ESN state for a single timestep.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Per-timestep input slices.  ``inputs[0]`` is the feedback slice
            of shape ``(batch, feedback_size)``.  If a driving input is
            present, ``inputs[1]`` has shape ``(batch, input_size)``.
        state : torch.Tensor
            Current hidden state of shape ``(batch, reservoir_size)``.

        Returns
        -------
        torch.Tensor
            Next hidden state of shape ``(batch, reservoir_size)``.

        Raises
        ------
        ValueError
            If the feedback feature dimension does not match
            ``self.feedback_size``, if a driving input is provided but
            ``self.weight_input`` is ``None``, or if the driving input
            feature dimension does not match ``self.input_size``.
        """
        fb_t = inputs[0]

        if fb_t.shape[-1] != self.feedback_size:
            raise ValueError(
                f"Feedback size mismatch. Expected {self.feedback_size}, got {fb_t.shape[-1]}"
            )

        has_driving = len(inputs) > 1
        if has_driving:
            if self.weight_input is None:
                raise ValueError(
                    "Reservoir was initialized without input_size, "
                    "but driving input was provided in forward pass"
                )
            x_t = inputs[1]
            if x_t.shape[-1] != self.input_size:
                raise ValueError(
                    f"Driving input size mismatch. Expected {self.input_size}, "
                    f"got {x_t.shape[-1]}"
                )

        feedback_contrib = F.linear(fb_t, self.weight_feedback)
        recurrent_contrib = F.linear(state, self.weight_hh)
        pre_activation = feedback_contrib + recurrent_contrib

        if has_driving:
            pre_activation = pre_activation + F.linear(inputs[1], self.weight_input)

        if self.bias_h is not None:
            pre_activation = pre_activation + self.bias_h

        new_state = self.activation_fn(pre_activation)

        if self.leak_rate < 1.0:
            return (1 - self.leak_rate) * state + self.leak_rate * new_state
        return new_state

    # ------------------------------------------------------------------
    # Weight initialization (verbatim from legacy ReservoirLayer)
    # ------------------------------------------------------------------

    def _get_activation(self, activation: str) -> callable:
        """Get activation function by name."""
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "identity": lambda x: x,
            "sigmoid": torch.sigmoid,
        }

        if activation not in activations:
            raise ValueError(
                f"Unknown activation '{activation}'. Supported: {list(activations.keys())}"
            )

        return activations[activation]

    def _initialize_weights(self) -> None:
        """Initialize all weight matrices."""
        self._initialize_feedback_weights()

        if self.input_size is not None:
            self._initialize_input_weights()
        else:
            self.register_parameter("weight_input", None)

        self._initialize_recurrent_weights()

        if self._bias:
            self.bias_h = nn.Parameter(torch.zeros(self.reservoir_size))
        else:
            self.register_parameter("bias_h", None)

    def _initialize_feedback_weights(self) -> None:
        """Initialize feedback weight matrix."""
        self.weight_feedback = nn.Parameter(torch.empty(self.reservoir_size, self.feedback_size))

        resolved = resolve_initializer(self.feedback_initializer)
        if resolved is not None:
            resolved.initialize(self.weight_feedback)
        else:
            nn.init.uniform_(self.weight_feedback, -1, 1)

    def _initialize_input_weights(self) -> None:
        """Initialize driving input weight matrix."""
        assert self.input_size is not None
        self.weight_input = nn.Parameter(torch.empty(self.reservoir_size, self.input_size))

        resolved = resolve_initializer(self.input_initializer)
        if resolved is not None:
            resolved.initialize(self.weight_input)
        else:
            nn.init.uniform_(self.weight_input, -1, 1)

    def _initialize_recurrent_weights(self) -> None:
        """Initialize recurrent weight matrix from topology or random."""
        self.weight_hh = nn.Parameter(torch.empty(self.reservoir_size, self.reservoir_size))

        resolved = resolve_topology(self.topology)
        if resolved is not None:
            resolved.initialize(self.weight_hh, spectral_radius=self.spectral_radius)
        else:
            nn.init.uniform_(self.weight_hh, -1.0, 1.0)
            if self.spectral_radius is not None:
                self._scale_spectral_radius()

    def _scale_spectral_radius(self) -> None:
        """Scale recurrent weights to target spectral radius."""
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(self.weight_hh.data)
            current_spectral_radius = torch.max(torch.abs(eigenvalues)).item()

            if current_spectral_radius > 0:
                scale = self.spectral_radius / current_spectral_radius
                self.weight_hh.data *= scale

    def __repr__(self) -> str:
        """Return string representation."""
        input_str = f", input_size={self.input_size}" if self.input_size is not None else ""
        return (
            f"ESNCell("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"spectral_radius={self.spectral_radius}"
            f")"
        )


class ESNLayer(BaseReservoirLayer):
    """
    Stateful RNN reservoir layer for Echo State Networks.

    Public-facing class and drop-in replacement for the legacy
    ``ReservoirLayer``.  Internally creates an :class:`ESNCell` that owns
    all parameters and performs the single-step update; the sequence loop
    and state management live in the parent :class:`BaseReservoirLayer`.

    The constructor signature and all defaults are identical to the legacy
    ``ReservoirLayer``.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (hidden state dimension).
    feedback_size : int
        Dimension of feedback signal.  Required for all reservoirs.
    input_size : int, optional
        Dimension of driving inputs.  If ``None``, no driving input is
        expected.
    spectral_radius : float, optional
        Target spectral radius for recurrent weights.  Controls the
        "memory" and stability of the reservoir.  If ``None``, no spectral
        radius scaling is applied.
    bias : bool, default=True
        Whether to include a bias term.
    activation : {'tanh', 'relu', 'identity', 'sigmoid'}, default='tanh'
        Activation function for reservoir dynamics.
    leak_rate : float, default=1.0
        Leaky integration rate in [0, 1].  Value of 1.0 means no leaking
        (standard RNN update).  Smaller values create slower dynamics.
    trainable : bool, default=False
        If ``True``, reservoir weights are trainable via backpropagation.
        Standard ESNs use frozen (non-trainable) weights.
    feedback_initializer : str, tuple, or InputFeedbackInitializer, optional
        Initializer for the feedback weight matrix.
    input_initializer : str, tuple, or InputFeedbackInitializer, optional
        Initializer for the input weight matrix.  Only used when
        ``input_size`` is provided.
    topology : str, tuple, or TopologyInitializer, optional
        Graph topology for recurrent weights.

    Attributes
    ----------
    state : torch.Tensor or None
        Current reservoir state of shape ``(batch, reservoir_size)``.
        ``None`` if not yet initialized.

    Examples
    --------
    Basic feedback-only reservoir:

    >>> reservoir = ESNLayer(reservoir_size=500, feedback_size=10)
    >>> feedback = torch.randn(4, 50, 10)  # (batch, time, features)
    >>> output = reservoir(feedback)
    >>> print(output.shape)
    torch.Size([4, 50, 500])

    Reservoir with driving input:

    >>> reservoir = ESNLayer(
    ...     reservoir_size=500,
    ...     feedback_size=10,
    ...     input_size=5,
    ...     spectral_radius=0.95
    ... )
    >>> feedback = torch.randn(4, 50, 10)
    >>> driving = torch.randn(4, 50, 5)
    >>> output = reservoir(feedback, driving)

    Using graph topology by name:

    >>> reservoir = ESNLayer(
    ...     reservoir_size=500,
    ...     feedback_size=10,
    ...     topology="erdos_renyi",
    ...     spectral_radius=0.9
    ... )

    Using topology with custom parameters (tuple format):

    >>> reservoir = ESNLayer(
    ...     reservoir_size=500,
    ...     feedback_size=10,
    ...     topology=("watts_strogatz", {"k": 6, "p": 0.3}),
    ...     feedback_initializer=("pseudo_diagonal", {"input_scaling": 0.5}),
    ...     spectral_radius=0.95
    ... )

    Stateful processing across batches:

    >>> out1 = reservoir(data1)  # State initialized
    >>> out2 = reservoir(data2)  # State carries over
    >>> reservoir.reset_state()  # Manual reset
    >>> out3 = reservoir(data3)  # Fresh state

    See Also
    --------
    resdag.layers.esn.ESNCell : Underlying single-step cell.
    resdag.layers.base.BaseReservoirLayer : Base providing state management.
    resdag.init.topology.TopologyInitializer : Base class for topology init.
    resdag.init.input_feedback.InputFeedbackInitializer : Input init base.
    resdag.composition.ESNModel : Model composition using reservoir layers.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        spectral_radius: float | None = None,
        bias: bool = True,
        activation: str = "tanh",
        leak_rate: float = 1.0,
        trainable: bool = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
    ) -> None:
        cell = ESNCell(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            input_size=input_size,
            spectral_radius=spectral_radius,
            bias=bias,
            activation=activation,
            leak_rate=leak_rate,
            trainable=trainable,
            feedback_initializer=feedback_initializer,
            input_initializer=input_initializer,
            topology=topology,
        )
        super().__init__(cell)
        # Preserve legacy attribute used by existing callsites and tests
        self._initialized = True

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute lookups to the wrapped cell.

        This preserves backwards compatibility: code that accessed attributes
        directly on ``ReservoirLayer`` (e.g. ``layer.reservoir_size``,
        ``layer.weight_hh``) continues to work unchanged on ``ESNLayer``.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Reach into _modules directly to avoid recursion during __init__
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
