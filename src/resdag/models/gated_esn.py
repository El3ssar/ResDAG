"""Gated ("reservoir GRU") ESN architecture.

This module provides the :func:`gated_esn` factory, which builds a model whose
reservoir is a frozen (or trainable) light-gated recurrent cell
(:class:`~resdag.layers.reservoirs.gated.GatedReservoirLayer`) instead of the plain
leaky ESN reservoir.  The architecture mirrors :func:`resdag.models.classic_esn`
— the raw input is concatenated with the reservoir output before the readout::

    Input -> GatedReservoirLayer -> Concatenate(Input, Reservoir) -> Readout

The gated reservoir uses the LiGRU update (Ravanelli et al. 2018); see
:mod:`resdag.layers.cells.gated_cell` for the equations and attribution.  A
*frozen* gated reservoir often underperforms a classic ESN, so ``trainable=True``
(a BPTT rollout that updates the gates) is frequently the more useful setting.

See Also
--------
resdag.models.classic_esn : The ungated classic ESN with the same wiring.
resdag.layers.reservoirs.gated.GatedReservoirLayer : The gated reservoir layer.
"""

from typing import Any

from pytorch_symbolic.symbolic_data import SymbolicTensor

from resdag.core import ESNModel, reservoir_input
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers import Concatenate
from resdag.layers.reservoirs.gated import GatedReservoirLayer

from ._builder import ReadoutSpec, _make_readout


def gated_esn(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    input_size: int | None = None,
    input_initializer: InitializerSpec | None = None,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    update_gate_bias: float = 2.0,
    activation: str = "relu",
    gate_activation: str = "sigmoid",
    feedback_initializer: InitializerSpec | None = None,
    bias: bool = True,
    trainable: bool = False,
    concat_input: bool = True,
    # Readout params
    readout: ReadoutSpec = "ridge",
    readout_alpha: float = 1e-6,
    readout_bias: bool = True,
    readout_name: str = "output",
    readout_kwargs: dict[str, Any] | None = None,
    # Extra reservoir kwargs
    **reservoir_kwargs: Any,
) -> ESNModel:
    """Build a gated ("reservoir GRU") ESN model.

    The reservoir is a frozen (or trainable) LiGRU-style gated recurrent cell.
    The raw input is concatenated with the reservoir output before the readout,
    following the classic-ESN wiring::

        Input -> GatedReservoirLayer -> Concatenate(Input, Reservoir) -> Readout

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features (input dimension).
    output_size : int
        Number of output features.
    input_size : int or None, optional
        Dimension of an optional driving (exogenous) input.  When given, the
        model takes two inputs ``(feedback, driver)``; the driver feeds the
        reservoir but is kept out of the input concatenation, so the readout
        ``in_features`` stays ``feedback_size + reservoir_size``.
    input_initializer : InitializerSpec, optional
        Initializer for the driving-input weights.  Only used when ``input_size``
        is given.
    topology : TopologySpec, optional
        Topology for the recurrent **candidate** weight.  The update-gate
        recurrent weight is always dense-random.
    spectral_radius : float, default=0.9
        Desired spectral radius for the recurrent candidate weight.
    update_gate_bias : float, default=2.0
        Retention bias on the update gate (keeps a frozen gated reservoir from
        washing out its memory).  See
        :class:`~resdag.layers.cells.gated_cell.GatedReservoirCell`.
    activation : str, default="relu"
        Candidate nonlinearity (LiGRU default is ``"relu"``).
    gate_activation : str, default="sigmoid"
        Update-gate nonlinearity.
    feedback_initializer : InitializerSpec, optional
        Initializer for feedback weights.
    bias : bool, default=True
        Whether to use gate biases in the reservoir.
    trainable : bool, default=False
        Whether reservoir weights are trainable.  A frozen gated reservoir often
        underperforms an ESN; ``True`` (BPTT through the gates) is frequently the
        more useful setting.  Note the ReLU candidate is unbounded, so a *frozen*
        autoregressive forecast that drifts off the attractor can grow large in
        magnitude (not merely track poorly) — prefer ``trainable=True`` for
        closed-loop forecasting.
    concat_input : bool, default=True
        Whether to concatenate the raw input with the reservoir output before the
        readout.
    readout : str or type[ReadoutLayer], default="ridge"
        A built-in name (``"ridge"``, ``"cholesky"``, ``"cg"``, ``"svd"``,
        ``"pinv"``, ``"incremental"``, ``"rls"``, ``"fista"``) or a
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass. Additional
        solver options go in ``readout_kwargs``.
    readout_alpha : float, default=1e-6
        Ridge regularization for the readout.
    readout_bias : bool, default=True
        Whether to use bias in the readout.
    readout_name : str, default="output"
        Name for the readout layer (used as the target key in training).
    readout_kwargs : dict, optional
        Extra readout options, such as ``trainable=True``, ``solver``,
        ``state_noise`` or ``gram_dtype``. Architecture dimensions are fixed.
    **reservoir_kwargs : Any
        Additional keyword arguments forwarded to
        :class:`~resdag.layers.reservoirs.gated.GatedReservoirLayer`.

    Returns
    -------
    :class:`~resdag.core.ESNModel`
        Configured gated ESN model ready for training and inference.

    Raises
    ------
    ValueError
        If ``output_size`` is not a positive integer, or ``readout`` is an
        unknown string key.
    TypeError
        If ``readout`` is neither a known string key nor a
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass.

    Examples
    --------
    Trainable gated ESN on a 3-D signal:

    >>> from resdag.models import gated_esn
    >>> model = gated_esn(200, feedback_size=3, output_size=3, trainable=True)

    See Also
    --------
    resdag.models.classic_esn : The ungated classic ESN with the same wiring.
    resdag.layers.reservoirs.gated.GatedReservoirLayer : The gated reservoir layer.
    """
    if output_size <= 0:
        raise ValueError(f"output_size must be a positive integer, got {output_size!r}")

    # Resolve the readout up front so an invalid spec fails before construction.

    inp = reservoir_input(feedback_size)
    driver = reservoir_input(input_size) if input_size else None

    reservoir = GatedReservoirLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        input_size=input_size,
        topology=topology,
        spectral_radius=spectral_radius,
        update_gate_bias=update_gate_bias,
        activation=activation,
        gate_activation=gate_activation,
        feedback_initializer=feedback_initializer,
        input_initializer=input_initializer,
        bias=bias,
        trainable=trainable,
        **reservoir_kwargs,
    )

    output: SymbolicTensor = reservoir(inp) if driver is None else reservoir(inp, driver)
    model_inputs = inp if driver is None else [inp, driver]

    if concat_input:
        output = Concatenate()(inp, output)
        readout_in_features = feedback_size + reservoir_size
    else:
        readout_in_features = reservoir_size

    readout_layer = _make_readout(
        readout,
        in_features=readout_in_features,
        out_features=output_size,
        bias=readout_bias,
        alpha=readout_alpha,
        name=readout_name,
        readout_kwargs=readout_kwargs,
    )(output)

    return ESNModel(model_inputs, readout_layer)
