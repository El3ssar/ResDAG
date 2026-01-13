"""Headless ESN architecture without readout layer."""

from typing import Any

import pytorch_symbolic as ps

from resdag.composition import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec, resolve_initializer, resolve_topology
from resdag.layers import ReservoirLayer


def headless_esn(
    reservoir_size: int,
    feedback_size: int,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    leak_rate: float = 1.0,
    feedback_initializer: InitializerSpec | None = None,
    activation: str = "tanh",
    bias: bool = True,
    trainable: bool = False,
    # Extra reservoir kwargs
    **reservoir_kwargs: Any,
) -> ESNModel:
    """Build an ESN model with no readout layer.

    This model can be used to study the dynamics of the reservoir by applying
    different transformations to the reservoir states without a readout layer.
    Useful for analyzing reservoir dynamics, state space properties, and
    feature extraction.

    Architecture:
        Input -> Reservoir (output)

    The reservoir is not connected to a readout layer, allowing direct
    access to reservoir states for analysis or custom processing.

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features.
    topology : TopologySpec, optional
        Topology for recurrent weights. Accepts:
        - str: Registry name (e.g., "erdos_renyi")
        - tuple: (name, params) like ("watts_strogatz", {"k": 6, "p": 0.1})
        - GraphTopology: Pre-configured object
    spectral_radius : float, default=0.9
        Desired spectral radius for recurrent weights.
    leak_rate : float, default=1.0
        Leaky integration rate (1.0 = no leak).
    feedback_initializer : InitializerSpec, optional
        Initializer for feedback weights.
    activation : str, default="tanh"
        Activation function ("tanh", "relu", "sigmoid", "identity").
    bias : bool, default=True
        Whether to use bias in the reservoir.
    trainable : bool, default=False
        Whether reservoir weights are trainable.
    **reservoir_kwargs
        Additional keyword arguments passed to ReservoirLayer.

    Returns
    -------
    ESNModel
        ESN model with reservoir output only.

    Examples
    --------
    >>> from resdag.models import headless_esn
    >>> model = headless_esn(100, 1)
    >>> reservoir_states = model(input_data)  # Direct reservoir output
    """
    # Resolve topology and initializer specs
    resolved_topology = resolve_topology(topology)
    resolved_feedback_init = resolve_initializer(feedback_initializer)

    # Build model - just input and reservoir
    inp = ps.Input((100, feedback_size))

    reservoir = ReservoirLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        input_size=0,
        topology=resolved_topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        feedback_initializer=resolved_feedback_init,
        activation=activation,
        bias=bias,
        trainable=trainable,
        **reservoir_kwargs,
    )(inp)

    return ESNModel(inp, reservoir)
