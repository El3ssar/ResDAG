"""Linear ESN architecture with identity activation."""

from typing import Any

import pytorch_symbolic as ps

from resdag.composition import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec, resolve_initializer, resolve_topology
from resdag.layers import ReservoirLayer


def linear_esn(
    reservoir_size: int,
    feedback_size: int,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    leak_rate: float = 1.0,
    feedback_initializer: InitializerSpec | None = None,
    bias: bool = True,
    trainable: bool = False,
    # Extra reservoir kwargs
    **reservoir_kwargs: Any,
) -> ESNModel:
    """Build an ESN model with no readout layer and a linear activation function.

    This model uses a linear activation function in the reservoir, which can be
    useful for studying linear dynamics or as a baseline for comparison with
    nonlinear reservoirs.

    Architecture:
        Input -> Reservoir(activation='identity') (output)

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
    bias : bool, default=True
        Whether to use bias in the reservoir.
    trainable : bool, default=False
        Whether reservoir weights are trainable.
    **reservoir_kwargs
        Additional keyword arguments passed to ReservoirLayer.

    Returns
    -------
    ESNModel
        ESN model with linear reservoir activation.

    Examples
    --------
    >>> from resdag.models import linear_esn
    >>> model = linear_esn(100, 1)
    >>> linear_states = model(input_data)
    """
    # Resolve topology and initializer specs
    resolved_topology = resolve_topology(topology)
    resolved_feedback_init = resolve_initializer(feedback_initializer)

    # Build model - just input and reservoir with linear activation
    inp = ps.Input((100, feedback_size))

    reservoir = ReservoirLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        input_size=0,
        topology=resolved_topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        feedback_initializer=resolved_feedback_init,
        activation="identity",  # Force linear activation
        bias=bias,
        trainable=trainable,
        **reservoir_kwargs,
    )(inp)

    return ESNModel(inp, reservoir)
