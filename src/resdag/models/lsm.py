r"""
Liquid State Machine (LSM) Architecture
=======================================

This module provides the :func:`lsm` factory, which builds a spiking Liquid
State Machine (Maass, Natschläger & Markram 2002) — a recurrent reservoir of
leaky integrate-and-fire neurons whose real-valued **synaptic-trace** output is
read out by a ridge regression.  The architecture mirrors
:func:`~resdag.models.classic_esn` — the raw input is concatenated with the
reservoir trace before the readout::

    Input -> LIFReservoirLayer -> Concatenate(Input, trace) -> Readout

The reservoir is the flagship *non-ESN* reservoir in resdag: spiking neurons with
a surrogate-gradient spike (so the liquid is SGD-trainable) instead of the
tanh-ESN recurrence.  See :mod:`resdag.layers.cells.lif_cell` and
:mod:`resdag.layers.cells.spiking_cell` for the neuron model and attribution.

.. important::
   **An LSM is for input-driven regression / classification, not closed-loop
   chaos forecasting.** A Liquid State Machine (Maass 2002) maps a *driving*
   signal to a high-dimensional filtered liquid state, and a readout regresses
   that state onto a target — its use is driven tasks (a nonlinear function of
   the input, temporal pattern discrimination, sequence classification). The
   reservoir's output is a low-pass-filtered **spike trace**, *not* the feedback
   variable, so it cannot be fed back to itself: a free-running autoregressive
   forecast of a chaotic attractor is **not** what an LSM does, and its
   valid-prediction-time on such a task will be poor — the unbounded linear
   readout is fed back with no bounding nonlinearity, so the rollout may even
   **diverge to infinity**. For autoregressive chaotic forecasting reach for
   :func:`~resdag.models.ott_esn` / :func:`~resdag.models.classic_esn` or
   :func:`~resdag.models.es2n` instead. Continuous inputs enter the liquid as
   *injected current* (the standard analog-input convention for a
   real-valued-driven LSM).

.. note::
   **Drive the liquid hard enough to spike.** With the default unit-scale
   feedback weights a spiking reservoir is easy to leave *silent* — if most
   neurons never cross ``v_threshold`` the trace is all-zero and the readout has
   nothing to fit (and a frozen ``W_rec`` then receives a zero, uninformative
   gradient). Scale the input up and/or lower the threshold until a healthy
   fraction of neurons fire — e.g. ``feedback_initializer=("normal",
   {"input_scaling": 3.0})``, a smaller ``v_threshold`` (say ``0.5``), or a
   nonzero ``bias_scaling``. A quick ``(reservoir(feedback) > 0).float().mean()``
   on the trace (or watching ``output="spikes"``) tells you the firing rate.

References
----------
W. Maass, T. Natschläger and H. Markram, "Real-Time Computing Without Stable
States: A New Framework for Neural Computation Based on Perturbations,"
*Neural Computation* 14(11):2531-2560, 2002.

See Also
--------
classic_esn : Traditional leaky-ESN architecture with the same wiring.
eusn : Euler State Network (another input-driven reservoir).
resdag.layers.reservoirs.lif.LIFReservoirLayer : The spiking reservoir layer.
"""

from typing import Any

from pytorch_symbolic.symbolic_data import SymbolicTensor

from resdag.core import ESNModel, reservoir_input
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers import Concatenate
from resdag.layers.reservoirs.lif import LIFReservoirLayer

from ._builder import ReadoutSpec, _make_readout


def lsm(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    input_size: int | None = None,
    input_initializer: InitializerSpec | None = None,
    # Reservoir (LIF) params
    tau_mem: float = 20.0,
    tau_syn: float = 10.0,
    dt: float = 1.0,
    synaptic_current: bool = False,
    v_threshold: float = 1.0,
    reset: str = "subtract",
    surrogate_beta: float = 10.0,
    trace_decay: float = 0.9,
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
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
    r"""Build a spiking Liquid State Machine (LSM) model.

    The reservoir is a recurrent network of leaky integrate-and-fire neurons
    (:class:`~resdag.layers.reservoirs.lif.LIFReservoirLayer`); its real-valued
    synaptic-trace output is concatenated with the raw input and read out by a
    ridge regression, following the classic-ESN wiring::

        Input -> LIFReservoirLayer -> Concatenate(Input, trace) -> Readout

    .. important::
       This is an **input-driven** reservoir (Maass 2002), *not* a closed-loop
       chaos forecaster — the reservoir output is a filtered spike trace, not the
       feedback variable, so it cannot autoregress on itself. Its use is driven
       regression / classification, where a readout maps the filtered liquid
       state to a target. For autoregressive chaotic forecasting use
       :func:`~resdag.models.ott_esn`, :func:`~resdag.models.classic_esn`, or
       :func:`~resdag.models.es2n`. Continuous inputs enter as injected current.

    Parameters
    ----------
    reservoir_size : int
        Number of spiking units in the liquid.
    feedback_size : int
        Number of input features driving the reservoir.
    output_size : int
        Number of output (target) features.
    input_size : int or None, optional
        Dimension of an optional second driving input.  When given, the model
        takes two inputs ``(feedback, driver)``; the driver feeds the reservoir
        but is kept out of the input concatenation, so the readout
        ``in_features`` stays ``feedback_size + reservoir_size``.
    input_initializer : InitializerSpec, optional
        Initializer for the driving-input weights.  Only used when ``input_size``
        is given.
    tau_mem : float, default=20.0
        Membrane time constant :math:`\tau_\text{mem}` (in units of ``dt``).
        Sets the membrane decay :math:`\beta = e^{-\Delta t/\tau_\text{mem}}` —
        longer values give longer fading memory.
    tau_syn : float, default=10.0
        Synaptic time constant :math:`\tau_\text{syn}`.  Only used when
        ``synaptic_current=True`` (CuBa-LIF).
    dt : float, default=1.0
        Integration timestep :math:`\Delta t`.
    synaptic_current : bool, default=False
        If ``True``, use the second-order current-based LIF (CuBa-LIF), which
        adds an explicit synaptic-current state and smoother temporal filtering.
    v_threshold : float, default=1.0
        Firing threshold.
    reset : {'subtract', 'zero'}, default='subtract'
        Membrane reset on a spike (soft subtract vs hard zero).
    surrogate_beta : float, default=10.0
        Fast-sigmoid surrogate slope for the straight-through spike gradient
        (only matters when ``trainable=True``).
    trace_decay : float, default=0.9
        Decay :math:`\kappa` of the low-pass readout trace
        :math:`r_t = \kappa r_{t-1} + s_t` — the real-valued signal the readout
        regresses on.
    topology : TopologySpec, optional
        Topology for the recurrent spike-coupling weight ``W_rec``.  Accepts a
        registry name, a ``(name, params)`` tuple, a callable, or a configured
        :class:`~resdag.init.topology.TopologyInitializer`.
    spectral_radius : float, default=0.9
        Target spectral radius for ``W_rec`` (recurrent spike-coupling strength).
    feedback_initializer : InitializerSpec, optional
        Initializer for the feedback weights.
    bias : bool, default=True
        Whether to use a bias current in the reservoir.
    trainable : bool, default=False
        Whether the reservoir weights are trainable (via the surrogate spike
        gradient).  A frozen liquid + fitted readout is the classic LSM; set
        ``True`` to fine-tune the liquid with SGD.
    concat_input : bool, default=True
        Whether to concatenate the raw input with the reservoir trace before the
        readout.
    readout : str or type[ReadoutLayer], default="ridge"
        A built-in name (``"ridge"``, ``"cholesky"``, ``"cg"``, ``"svd"``,
        ``"pinv"``, ``"incremental"``, ``"rls"``, ``"fista"``, ``"lsmr"``) or a
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass. Additional
        solver options go in ``readout_kwargs``.
    readout_alpha : float, default=1e-6
        Ridge regularization for the readout.
    readout_bias : bool, default=True
        Whether to use a bias in the readout.
    readout_name : str, default="output"
        Name for the readout layer (the target key in training).
    readout_kwargs : dict, optional
        Extra readout options, such as ``trainable=True``, ``solver``,
        ``state_noise`` or ``gram_dtype``. Architecture dimensions are fixed.
    **reservoir_kwargs : Any
        Additional keyword arguments forwarded to
        :class:`~resdag.layers.reservoirs.lif.LIFReservoirLayer` (e.g. ``bias_scaling``,
        ``seed``, ``input_initializer``).

    Returns
    -------
    :class:`~resdag.core.ESNModel`
        Configured LSM model ready for driven training and inference.

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
    >>> from resdag.models import lsm
    >>> model = lsm(reservoir_size=300, feedback_size=1, output_size=1, seed=0)

    See Also
    --------
    resdag.models.classic_esn : Traditional leaky ESN with the same wiring.
    resdag.models.eusn : Euler State Network (another input-driven reservoir).
    resdag.layers.reservoirs.lif.LIFReservoirLayer : The spiking reservoir layer.
    """
    if output_size <= 0:
        raise ValueError(f"output_size must be a positive integer, got {output_size!r}")

    # Resolve the readout up front so an invalid spec fails before construction.

    inp = reservoir_input(
        feedback_size, dtype=reservoir_kwargs.get("dtype"), device=reservoir_kwargs.get("device")
    )
    driver = (
        reservoir_input(
            input_size, dtype=reservoir_kwargs.get("dtype"), device=reservoir_kwargs.get("device")
        )
        if input_size
        else None
    )

    reservoir = LIFReservoirLayer(
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
        spectral_radius=spectral_radius,
        topology=topology,
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
    ).to(device=reservoir.weight_feedback.device, dtype=reservoir.weight_feedback.dtype)(output)

    return ESNModel(model_inputs, readout_layer)
