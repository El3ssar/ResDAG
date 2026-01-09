"""ReservoirLayer implementation for torch_rc.

This module provides ReservoirLayer, a stateful recurrent neural network layer
with graph-based weight initialization for Echo State Networks (ESN).
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..init.input_feedback import InputFeedbackInitializer
from ..init.topology import TopologyInitializer, get_topology


class ReservoirLayer(nn.Module):
    """Stateful RNN reservoir layer with graph-based weight initialization.

    This is a custom RNN implementation (not using nn.RNNBase) designed for
    Echo State Networks. Key features:

    - Stateful: Internal state persists across forward passes
    - Separate feedback and driving inputs with independent weight matrices
    - Graph topology: Recurrent weights initialized from graph adjacency matrices
    - Per-timestep processing: Input/output shape (B, T, F)

    The reservoir state evolves according to:
        h_t = activation(W_fb @ fb_t + W_in @ x_t + W_rec @ h_{t-1} + b)

    Where:
        - W_fb: Feedback weight matrix (reservoir_size, feedback_size)
        - W_in: Driving input weight matrix (reservoir_size, input_size) - optional
        - W_rec: Recurrent weight matrix (reservoir_size, reservoir_size)
        - h_t: Hidden state at timestep t
        - fb_t: Feedback signal (autoregressive during generation)
        - x_t: Driving inputs (external signals)

    Args:
        reservoir_size: Number of reservoir units
        feedback_size: Size of feedback signal (required)
        input_size: Size of driving inputs (optional, for external signals)
        topology: Graph topology for recurrent weights (Phase 2 feature). Can be:
                  - None: Random initialization (default)
                  - str: Name of topology ("erdos_renyi", "small_world", etc.) - Phase 2
                  - callable: Function returning adjacency matrix - Phase 2
        spectral_radius: Desired spectral radius for W_rec (default: 0.9)
        feedback_scaling: Scale factor for feedback weights (default: 1.0)
        input_scaling: Scale factor for driving input weights (default: 1.0)
        bias: Whether to use bias (default: True)
        activation: Activation function name ("tanh", "relu", "identity")
        leak_rate: Leaky integration rate (1.0 = no leak, default: 1.0)

    Example:
        # Feedback only
        reservoir = ReservoirLayer(reservoir_size=500, feedback_size=10)
        output = reservoir(feedback)  # feedback: (B, T, 10) -> output: (B, T, 500)

        # Feedback + driving inputs
        reservoir = ReservoirLayer(reservoir_size=500, feedback_size=10, input_size=5)
        output = reservoir(feedback, driving)  # (B, T, 10), (B, T, 5) -> (B, T, 500)

        # Stateful processing
        out1 = reservoir(feedback1)  # State initialized
        out2 = reservoir(feedback2)  # State carries over from out1
        reservoir.reset_state()  # Manual reset
        out3 = reservoir(feedback3)  # Fresh state
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        spectral_radius: float = 0.9,
        feedback_scaling: float = 1.0,
        input_scaling: float = 1.0,
        bias: bool = True,
        activation: str = "tanh",
        leak_rate: float = 1.0,
        trainable: bool = False,
        feedback_initializer: InputFeedbackInitializer | None = None,
        input_initializer: InputFeedbackInitializer | None = None,
        topology: str | TopologyInitializer | None = None,
    ) -> None:
        """Initialize the ReservoirLayer."""
        super().__init__()

        # Store configuration
        self.reservoir_size = reservoir_size
        self.feedback_size = feedback_size
        self.input_size = input_size
        self.topology = topology
        self.spectral_radius = spectral_radius
        self.feedback_scaling = feedback_scaling
        self.input_scaling = input_scaling
        self.feedback_initializer = feedback_initializer
        self.input_initializer = input_initializer
        self.leak_rate = leak_rate
        self.trainable = trainable

        # Activation function
        self._activation_name = activation  # Store name for functional forecast
        self.activation = self._get_activation(activation)

        # Internal state (initialized on first forward pass)
        self.state: Optional[torch.Tensor] = None

        # Store bias flag before initialization
        self._bias = bias

        # Initialize weight matrices
        self._initialize_weights()

        # Freeze weights if not trainable
        if not self.trainable:
            self._freeze_weights()

        self._initialized = True

    def _get_activation(self, activation: str) -> callable:
        """Get activation function by name.

        Args:
            activation: Name of activation function

        Returns:
            Activation function callable

        Raises:
            ValueError: If activation name is not recognized
        """
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
        """Initialize all weight matrices.

        Initializes three separate weight matrices:
        - W_fb (feedback weights): Random uniform with feedback_scaling
        - W_in (input weights): Random uniform with input_scaling (optional)
        - W_rec (recurrent weights): From topology or random with spectral radius scaling
        - bias: Zero initialization
        """
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
        """Initialize feedback weight matrix.

        Creates W_fb with either custom initializer or default uniform random.
        """
        self.weight_feedback = nn.Parameter(torch.empty(self.reservoir_size, self.feedback_size))

        if self.feedback_initializer is not None:
            # Use custom initializer
            self.feedback_initializer.initialize(self.weight_feedback)
        else:
            # Default: uniform random scaled by feedback_scaling
            nn.init.uniform_(self.weight_feedback, -self.feedback_scaling, self.feedback_scaling)

    def _initialize_input_weights(self) -> None:
        """Initialize driving input weight matrix.

        Creates W_in with either custom initializer or default uniform random.
        Only called if input_size is provided.
        """
        assert self.input_size is not None
        self.weight_input = nn.Parameter(torch.empty(self.reservoir_size, self.input_size))

        if self.input_initializer is not None:
            # Use custom initializer
            self.input_initializer.initialize(self.weight_input)
        else:
            # Default: uniform random scaled by input_scaling
            nn.init.uniform_(self.weight_input, -self.input_scaling, self.input_scaling)

    def _initialize_recurrent_weights(self) -> None:
        """Initialize recurrent weight matrix.

        Creates W_rec (reservoir_size, reservoir_size) with:
        - Topology-based initialization if topology is provided (Phase 2)
        - Random uniform initialization in [-1, 1] otherwise
        - Spectral radius scaling to achieve desired dynamics
        """
        # Create recurrent weight matrix parameter
        self.weight_hh = nn.Parameter(torch.empty(self.reservoir_size, self.reservoir_size))

        if self.topology is not None:
            # Phase 2: Use graph topology
            if isinstance(self.topology, str):
                # String topology name - look up in registry
                topology_initializer = get_topology(self.topology)
            elif isinstance(self.topology, TopologyInitializer):
                # Direct topology initializer
                topology_initializer = self.topology
            else:
                raise TypeError(
                    f"topology must be a string or TopologyInitializer, "
                    f"got {type(self.topology).__name__}"
                )

            # Initialize using topology (with spectral radius scaling)
            topology_initializer.initialize(self.weight_hh, spectral_radius=self.spectral_radius)
        else:
            # Random initialization
            nn.init.uniform_(self.weight_hh, -1.0, 1.0)
            # Scale to desired spectral radius
            self._scale_spectral_radius()

    def _scale_spectral_radius(self) -> None:
        """Scale recurrent weight matrix to desired spectral radius.

        Computes the largest eigenvalue and scales the matrix so that
        the spectral radius matches the target value.
        """
        with torch.no_grad():
            # Compute spectral radius (largest absolute eigenvalue)
            eigenvalues = torch.linalg.eigvals(self.weight_hh.data)
            current_spectral_radius = torch.max(torch.abs(eigenvalues)).item()

            # Scale to target spectral radius
            if current_spectral_radius > 0:
                scale = self.spectral_radius / current_spectral_radius
                self.weight_hh.data *= scale

    def forward(
        self,
        feedback: torch.Tensor,
        *driving_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the reservoir.

        Processes feedback signal and optional driving inputs through the reservoir.
        Feedback and inputs are processed with separate weight matrices.

        Args:
            feedback: Feedback signal of shape (B, T, feedback_size)
                     Required - provides autoregressive signal
            *driving_inputs: Optional driving input tensors, each of shape (B, T, F_i)
                            Concatenated if multiple provided

        Returns:
            Reservoir states for all timesteps: (B, T, reservoir_size)

        Raises:
            ValueError: If feedback shape is invalid or inputs have inconsistent shapes
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

            # Validate shape
            if driving_input.shape[0] != batch_size or driving_input.shape[1] != seq_len:
                raise ValueError(
                    f"Driving input must match feedback dimensions. "
                    f"Feedback: {feedback.shape}, Driving: {driving_input.shape}"
                )

            # Check input size matches
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

        # Initialize state if needed (or if device changed)
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
            fb_t = feedback[:, t, :]  # (B, feedback_size)

            # Compute contributions: h_t = activation(W_fb @ fb_t + W_in @ x_t + W_rec @ h_{t-1} + b)
            feedback_contrib = F.linear(fb_t, self.weight_feedback)  # (B, H)
            recurrent_contrib = F.linear(self.state, self.weight_hh)  # (B, H)

            pre_activation = feedback_contrib + recurrent_contrib

            # Add driving input contribution if present
            if driving_input is not None:
                x_t = driving_input[:, t, :]  # (B, input_size)
                input_contrib = F.linear(x_t, self.weight_input)  # (B, H)
                pre_activation = pre_activation + input_contrib

            # Add bias
            if self.bias_h is not None:
                pre_activation = pre_activation + self.bias_h

            # Apply activation
            new_state = self.activation(pre_activation)

            # Leaky integration
            if self.leak_rate < 1.0:
                self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
            else:
                self.state = new_state

            outputs[:, t, :] = self.state

        return outputs

    def reset_state(self, batch_size: Optional[int] = None) -> None:
        """Reset internal state to zero.

        Args:
            batch_size: If provided, initialize state with this batch size.
                       If None, state is set to None and will be lazily initialized.
        """
        if batch_size is not None:
            device = self.weight_hh.device if self._initialized else torch.device("cpu")
            dtype = self.weight_hh.dtype if self._initialized else torch.float32
            self.state = torch.zeros(batch_size, self.reservoir_size, device=device, dtype=dtype)
        else:
            self.state = None

    def set_state(self, state: torch.Tensor) -> None:
        """Set internal state to a specific value.

        Args:
            state: New state tensor of shape (B, reservoir_size)

        Raises:
            ValueError: If state shape is invalid
        """
        if state.shape[-1] != self.reservoir_size:
            raise ValueError(
                f"State size mismatch. Expected (..., {self.reservoir_size}), got {state.shape}"
            )
        self.state = state.clone()

    def get_state(self) -> Optional[torch.Tensor]:
        """Get current internal state.

        Returns:
            Current state tensor of shape (B, reservoir_size), or None if not initialized
        """
        return self.state.clone() if self.state is not None else None

    def __repr__(self) -> str:
        """String representation."""
        input_str = f", input_size={self.input_size}" if self.input_size is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"spectral_radius={self.spectral_radius}"
            f")"
        )
