"""
ESN Reservoir Layer
===================

This module provides the concrete Echo State Network (ESN) implementation of
the reservoir layer:

- :class:`ESNLayer` — public-facing sequence layer

See Also
--------
resdag.layers.reservoirs.base_reservoir : Abstract base class (BaseReservoirLayer).
resdag.init.topology : Topology initialization for recurrent weights.
resdag.init.input_feedback : Input/feedback weight initialization.
"""

from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.cells import ESNCell

from .base_reservoir import BaseReservoirLayer


class ESNLayer(BaseReservoirLayer):
    """
    Stateful RNN reservoir layer for Echo State Networks.

    Public-facing class.  Internally creates an :class:`ESNCell` that owns
    all parameters and performs the single-step update; the sequence loop
    and state management live in the parent :class:`BaseReservoirLayer`.

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
        """Delegate unknown attribute lookups to the wrapped cell."""
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
