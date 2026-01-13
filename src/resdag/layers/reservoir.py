"""
Reservoir Layer
===============

This module provides :class:`ReservoirLayer`, a stateful recurrent neural network
layer with graph-based weight initialization for Echo State Networks (ESN).

The reservoir layer is the core computational component of ESNs, implementing
recurrent dynamics with the Echo State Property.

See Also
--------
resdag.init.topology : Topology initialization for reservoir weights.
resdag.init.input_feedback : Input/feedback weight initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..init.input_feedback import InputFeedbackInitializer, get_input_feedback
from ..init.topology import TopologyInitializer, get_topology


class ReservoirLayer(nn.Module):
    """
    Stateful RNN reservoir layer with graph-based weight initialization.

    A custom RNN implementation designed for Echo State Networks, featuring
    stateful processing, separate feedback and driving inputs, and
    graph-based topology initialization.

    The reservoir state evolves according to:

    .. math::

        h_t = f((1 - \\alpha) h_{t-1} + \\alpha \\cdot g(W_{fb} x_{fb,t} + W_{in} x_{in,t} + W_{rec} h_{t-1} + b))

    where :math:`f` is the activation function, :math:`\\alpha` is the leak rate,
    :math:`W_{fb}` is the feedback weight matrix, :math:`W_{in}` is the input
    weight matrix, and :math:`W_{rec}` is the recurrent weight matrix.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (hidden state dimension).
    feedback_size : int
        Dimension of feedback signal. Required for all reservoirs.
    input_size : int, optional
        Dimension of driving inputs. If None, no driving input is expected.
    topology : str or TopologyInitializer, optional
        Graph topology for recurrent weights. Can be:

        - None: Random uniform initialization (default)
        - str: Registered topology name (e.g., ``"erdos_renyi"``, ``"watts_strogatz"``)
        - TopologyInitializer: Custom topology instance

    spectral_radius : float, optional
        Target spectral radius for recurrent weights. Controls the
        "memory" and stability of the reservoir. Typical values are
        0.9-1.0. If None, no spectral radius scaling is applied.
    bias : bool, default=True
        Whether to include a bias term.
    activation : {'tanh', 'relu', 'identity', 'sigmoid'}, default='tanh'
        Activation function for reservoir dynamics.
    leak_rate : float, default=1.0
        Leaky integration rate in [0, 1]. Value of 1.0 means no leaking
        (standard RNN update). Smaller values create slower dynamics.
    trainable : bool, default=False
        If True, reservoir weights are trainable via backpropagation.
        Standard ESNs use frozen (non-trainable) weights.
    feedback_initializer : str or InputFeedbackInitializer, optional
        Initializer for feedback weight matrix. If None, uses uniform
        random initialization.
    input_initializer : str or InputFeedbackInitializer, optional
        Initializer for input weight matrix. If None, uses uniform
        random initialization. Only used if ``input_size`` is provided.

    Attributes
    ----------
    state : torch.Tensor or None
        Current reservoir state of shape ``(batch, reservoir_size)``.
        None if not yet initialized.
    weight_feedback : torch.nn.Parameter
        Feedback weight matrix of shape ``(reservoir_size, feedback_size)``.
    weight_input : torch.nn.Parameter or None
        Input weight matrix of shape ``(reservoir_size, input_size)``,
        or None if no input_size was specified.
    weight_hh : torch.nn.Parameter
        Recurrent weight matrix of shape ``(reservoir_size, reservoir_size)``.
    bias_h : torch.nn.Parameter or None
        Bias vector of shape ``(reservoir_size,)``, or None if ``bias=False``.

    Examples
    --------
    Basic feedback-only reservoir:

    >>> reservoir = ReservoirLayer(reservoir_size=500, feedback_size=10)
    >>> feedback = torch.randn(4, 50, 10)  # (batch, time, features)
    >>> output = reservoir(feedback)
    >>> print(output.shape)
    torch.Size([4, 50, 500])

    Reservoir with driving input:

    >>> reservoir = ReservoirLayer(
    ...     reservoir_size=500,
    ...     feedback_size=10,
    ...     input_size=5,
    ...     spectral_radius=0.95
    ... )
    >>> feedback = torch.randn(4, 50, 10)
    >>> driving = torch.randn(4, 50, 5)
    >>> output = reservoir(feedback, driving)

    Using graph topology:

    >>> reservoir = ReservoirLayer(
    ...     reservoir_size=500,
    ...     feedback_size=10,
    ...     topology="erdos_renyi",
    ...     spectral_radius=0.9
    ... )

    Stateful processing across batches:

    >>> out1 = reservoir(data1)  # State initialized
    >>> out2 = reservoir(data2)  # State carries over
    >>> reservoir.reset_state()   # Manual reset
    >>> out3 = reservoir(data3)  # Fresh state

    See Also
    --------
    resdag.init.topology.TopologyInitializer : Base class for topology initialization.
    resdag.init.input_feedback.InputFeedbackInitializer : Base class for input initialization.
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
        feedback_initializer: InputFeedbackInitializer | str | None = None,
        input_initializer: InputFeedbackInitializer | str | None = None,
        topology: str | TopologyInitializer | None = None,
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

        # Internal state (initialized on first forward pass)
        self.state: torch.Tensor | None = None

        # Store bias flag before initialization
        self._bias = bias

        # Initialize weight matrices
        self._initialize_weights()

        # Freeze weights if not trainable
        if not self.trainable:
            self._freeze_weights()

        self._initialized = True

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

    def _freeze_weights(self) -> None:
        """Freeze all weights by setting requires_grad=False."""
        for param in self.parameters():
            param.requires_grad_(False)

    def _initialize_weights(self) -> None:
        """Initialize all weight matrices."""
        # Feedback weights: (reservoir_size, feedback_size) - always present
        self._initialize_feedback_weights()

        # Driving input weights: (reservoir_size, input_size) - optional
        if self.input_size is not None:
            self._initialize_input_weights()
        else:
            self.register_parameter("weight_input", None)

        # Recurrent weights: (reservoir_size, reservoir_size)
        self._initialize_recurrent_weights()

        # Bias
        if self._bias:
            self.bias_h = nn.Parameter(torch.zeros(self.reservoir_size))
        else:
            self.register_parameter("bias_h", None)

    def _initialize_feedback_weights(self) -> None:
        """Initialize feedback weight matrix."""
        self.weight_feedback = nn.Parameter(torch.empty(self.reservoir_size, self.feedback_size))

        if self.feedback_initializer is not None:
            if isinstance(self.feedback_initializer, str):
                self.feedback_initializer = get_input_feedback(self.feedback_initializer)
            elif isinstance(self.feedback_initializer, InputFeedbackInitializer):
                self.feedback_initializer.initialize(self.weight_feedback)
            else:
                raise TypeError(
                    f"feedback_initializer must be a string or InputFeedbackInitializer, "
                    f"got {type(self.feedback_initializer).__name__}"
                )
        else:
            nn.init.uniform_(self.weight_feedback, -1, 1)

    def _initialize_input_weights(self) -> None:
        """Initialize driving input weight matrix."""
        assert self.input_size is not None
        self.weight_input = nn.Parameter(torch.empty(self.reservoir_size, self.input_size))

        if self.input_initializer is not None:
            if isinstance(self.input_initializer, str):
                self.input_initializer = get_input_feedback(self.input_initializer)
            elif isinstance(self.input_initializer, InputFeedbackInitializer):
                self.input_initializer.initialize(self.weight_input)
            else:
                raise TypeError(
                    f"input_initializer must be a string or InputFeedbackInitializer, "
                    f"got {type(self.input_initializer).__name__}"
                )
        else:
            nn.init.uniform_(self.weight_input, -1, 1)

    def _initialize_recurrent_weights(self) -> None:
        """Initialize recurrent weight matrix from topology or random."""
        self.weight_hh = nn.Parameter(torch.empty(self.reservoir_size, self.reservoir_size))

        if self.topology is not None:
            if isinstance(self.topology, str):
                topology_initializer = get_topology(self.topology)
            elif isinstance(self.topology, TopologyInitializer):
                topology_initializer = self.topology
            else:
                raise TypeError(
                    f"topology must be a string or TopologyInitializer, "
                    f"got {type(self.topology).__name__}"
                )

            topology_initializer.initialize(self.weight_hh, spectral_radius=self.spectral_radius)
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

    def forward(
        self,
        feedback: torch.Tensor,
        *driving_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process input sequence through the reservoir.

        Computes reservoir states for each timestep using feedback signal
        and optional driving inputs.

        Parameters
        ----------
        feedback : torch.Tensor
            Feedback signal of shape ``(batch, timesteps, feedback_size)``.
            This is the primary input that drives the reservoir dynamics.
        *driving_inputs : torch.Tensor
            Optional driving input tensor of shape ``(batch, timesteps, input_size)``.
            Only one driving input tensor is supported.

        Returns
        -------
        torch.Tensor
            Reservoir states for all timesteps, shape
            ``(batch, timesteps, reservoir_size)``.

        Raises
        ------
        ValueError
            If feedback shape is invalid, if driving input is provided but
            ``input_size`` was not set, or if tensor dimensions don't match.

        Notes
        -----
        The reservoir maintains internal state across forward calls. Use
        :meth:`reset_state` to clear state between independent sequences.

        Examples
        --------
        >>> reservoir = ReservoirLayer(100, feedback_size=10)
        >>> feedback = torch.randn(4, 50, 10)
        >>> states = reservoir(feedback)
        >>> print(states.shape)
        torch.Size([4, 50, 100])
        """
        # Validate feedback
        if feedback.dim() != 3:
            raise ValueError(f"Feedback must be 3D (B, T, F), got shape {feedback.shape}")

        batch_size, seq_len, fb_size = feedback.shape

        if fb_size != self.feedback_size:
            raise ValueError(
                f"Feedback size mismatch. Expected {self.feedback_size}, got {fb_size}"
            )

        # Process driving input if provided
        driving_input = None
        if len(driving_inputs) > 0:
            if len(driving_inputs) > 1:
                raise ValueError("Only one driving input tensor allowed")

            driving_input = driving_inputs[0]

            if driving_input.shape[0] != batch_size or driving_input.shape[1] != seq_len:
                raise ValueError(
                    f"Driving input must match feedback dimensions. "
                    f"Feedback: {feedback.shape}, Driving: {driving_input.shape}"
                )

            if self.input_size is None:
                raise ValueError(
                    "Reservoir was initialized without input_size, "
                    "but driving input was provided in forward pass"
                )
            if driving_input.shape[-1] != self.input_size:
                raise ValueError(
                    f"Driving input size mismatch. Expected {self.input_size}, "
                    f"got {driving_input.shape[-1]}"
                )

        # Initialize state if needed
        if (
            self.state is None
            or self.state.shape[0] != batch_size
            or self.state.device != feedback.device
        ):
            self.state = torch.zeros(
                batch_size, self.reservoir_size, device=feedback.device, dtype=feedback.dtype
            )

        # Process sequence timestep by timestep
        outputs = torch.empty(
            batch_size, seq_len, self.reservoir_size, device=feedback.device, dtype=feedback.dtype
        )
        for t in range(seq_len):
            fb_t = feedback[:, t, :]

            feedback_contrib = F.linear(fb_t, self.weight_feedback)
            recurrent_contrib = F.linear(self.state, self.weight_hh)

            pre_activation = feedback_contrib + recurrent_contrib

            if driving_input is not None:
                x_t = driving_input[:, t, :]
                input_contrib = F.linear(x_t, self.weight_input)
                pre_activation = pre_activation + input_contrib

            if self.bias_h is not None:
                pre_activation = pre_activation + self.bias_h

            new_state = self.activation_fn(pre_activation)

            if self.leak_rate < 1.0:
                self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
            else:
                self.state = new_state

            outputs[:, t, :] = self.state

        return outputs

    def reset_state(self, batch_size: int | None = None) -> None:
        """
        Reset internal state to zero.

        Parameters
        ----------
        batch_size : int, optional
            If provided, initialize state with this batch size on the
            current device. If None, state is set to None and will be
            lazily initialized on next forward pass.

        Examples
        --------
        >>> reservoir.reset_state()  # Lazy initialization
        >>> reservoir.reset_state(batch_size=4)  # Explicit initialization
        """
        if batch_size is not None:
            device = self.weight_hh.device if self._initialized else torch.device("cpu")
            dtype = self.weight_hh.dtype if self._initialized else torch.float32
            self.state = torch.zeros(batch_size, self.reservoir_size, device=device, dtype=dtype)
        else:
            self.state = None

    def set_random_state(self) -> None:
        """
        Set internal state to a random value.

        Examples
        --------
        >>> reservoir.set_random_state()
        >>> print(reservoir.state.shape)
        torch.Size([batch_size, reservoir_size])
        """
        if self._initialized:
            self.state = torch.randn(
                self.state.shape, device=self.state.device, dtype=self.state.dtype
            )
        else:
            raise RuntimeError("Reservoir not initialized")

    def set_state(self, state: torch.Tensor) -> None:
        """
        Set internal state to a specific value.

        Parameters
        ----------
        state : torch.Tensor
            New state tensor of shape ``(batch, reservoir_size)``.

        Raises
        ------
        ValueError
            If state has incorrect reservoir_size dimension.

        Examples
        --------
        >>> saved_state = reservoir.get_state()
        >>> # ... process some data ...
        >>> reservoir.set_state(saved_state)  # Restore
        """
        if state.shape[-1] != self.reservoir_size:
            raise ValueError(
                f"State size mismatch. Expected (..., {self.reservoir_size}), got {state.shape}"
            )
        self.state = state.clone()

    def get_state(self) -> torch.Tensor | None:
        """
        Get current internal state.

        Returns
        -------
        torch.Tensor or None
            Clone of current state tensor of shape ``(batch, reservoir_size)``,
            or None if state has not been initialized.

        Examples
        --------
        >>> state = reservoir.get_state()
        >>> if state is not None:
        ...     print(f"State shape: {state.shape}")
        """
        return self.state.clone() if self.state is not None else None

    @property
    def activation(self) -> str:
        """
        str : Name of the activation function.
        """
        return self._activation_name

    def __repr__(self) -> str:
        """Return string representation."""
        input_str = f", input_size={self.input_size}" if self.input_size is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"spectral_radius={self.spectral_radius}"
            f")"
        )
