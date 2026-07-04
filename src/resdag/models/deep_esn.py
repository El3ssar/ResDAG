"""
Deep ESN Architecture (stacked reservoirs)
==========================================

This module provides :func:`deep_esn`, which builds a *hierarchical* Echo State
Network by stacking ``N`` reservoir layers so that each layer's state sequence
drives the next.  The readout, by default, sees the **concatenation of every
layer's state** (the standard DeepESN readout), giving it access to the whole
multiple-timescale hierarchy the stack develops rather than just its top layer.

The construction follows Gallicchio & Micheli's Deep Echo State Network: an
untrained stack of reservoirs where depth — not just width — enriches the
temporal features, because successive layers develop progressively longer,
more structured memory timescales.  A per-layer **leak-rate schedule** exposes
that directly: the modern DeepESN literature recommends leak rates that
*decrease with depth* (fast dynamics near the input, slow dynamics deep in the
stack), which :func:`deep_esn` supports by accepting a list of leak rates (one
per layer).

References
----------
C. Gallicchio, A. Micheli and L. Pedrelli, "Deep reservoir computing: A
critical experimental analysis," Neurocomputing 268, 87-99 (2017).

C. Gallicchio and A. Micheli, "Deep Echo State Network (DeepESN): A Brief
Survey," arXiv:1712.04323 (2017).

See Also
--------
classic_esn : Single-reservoir ESN with input concatenation.
hybrid_esn : Physics-prior (knowledge-based) hybrid ESN.
"""

from collections.abc import Sequence
from typing import Any

from resdag.core import ESNModel, reservoir_input
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers import Concatenate, ESNLayer

from ._builder import ReadoutSpec, _resolve_readout_cls


def _resolve_reservoir_sizes(
    reservoir_sizes: Sequence[int] | int,
    n_layers: int | None,
) -> list[int]:
    """Resolve the ``(reservoir_sizes, n_layers)`` pair to a per-layer size list.

    Parameters
    ----------
    reservoir_sizes : sequence of int or int
        Either an explicit per-layer list of reservoir sizes, or a single int
        broadcast to every layer (which then requires ``n_layers``).
    n_layers : int or None
        Number of layers.  Required (and used) only when ``reservoir_sizes`` is a
        scalar; ignored (but validated for consistency) when it is a sequence.

    Returns
    -------
    list of int
        One positive reservoir size per layer.

    Raises
    ------
    ValueError
        If the resolved stack is empty, any size is not a positive integer, a
        scalar size is given without ``n_layers``, or ``n_layers`` contradicts an
        explicit ``reservoir_sizes`` list.
    """
    if isinstance(reservoir_sizes, int):
        if n_layers is None:
            raise ValueError(
                "deep_esn: n_layers is required when reservoir_sizes is a single int "
                "(the size is broadcast to every layer). Pass n_layers, or give "
                "reservoir_sizes as a per-layer list."
            )
        if n_layers < 1:
            raise ValueError(f"deep_esn: n_layers must be >= 1, got {n_layers!r}.")
        sizes = [reservoir_sizes] * n_layers
    else:
        sizes = list(reservoir_sizes)
        if n_layers is not None and n_layers != len(sizes):
            raise ValueError(
                f"deep_esn: n_layers ({n_layers}) contradicts len(reservoir_sizes) "
                f"({len(sizes)}). Pass one or the other consistently, or leave "
                f"n_layers=None when reservoir_sizes is a list."
            )

    if len(sizes) == 0:
        raise ValueError("deep_esn: at least one layer is required (reservoir_sizes is empty).")
    for i, size in enumerate(sizes):
        if not isinstance(size, int) or size <= 0:
            raise ValueError(
                f"deep_esn: every reservoir size must be a positive integer; "
                f"reservoir_sizes[{i}] = {size!r}."
            )
    return sizes


def _broadcast_leak_rates(leak_rate: float | Sequence[float], n_layers: int) -> list[float]:
    """Resolve a scalar-or-per-layer leak rate into a per-layer list.

    Parameters
    ----------
    leak_rate : float or sequence of float
        A scalar applied to every layer, or a per-layer schedule whose length
        must equal ``n_layers``.
    n_layers : int
        Number of layers in the stack.

    Returns
    -------
    list of float
        One leak rate per layer.

    Raises
    ------
    ValueError
        If a per-layer schedule length does not match ``n_layers``.
    """
    if isinstance(leak_rate, (int, float)):
        return [float(leak_rate)] * n_layers
    rates = [float(r) for r in leak_rate]
    if len(rates) != n_layers:
        raise ValueError(
            f"deep_esn: leak_rate schedule length ({len(rates)}) must equal the "
            f"number of layers ({n_layers}). Pass a scalar to share one rate, or a "
            f"list with one entry per layer."
        )
    return rates


def deep_esn(
    reservoir_sizes: Sequence[int] | int,
    n_layers: int | None = None,
    feedback_size: int = 1,
    output_size: int = 1,
    *,
    concat_all_states: bool = True,
    interlayer_scaling: float = 0.5,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    leak_rate: float | Sequence[float] = 1.0,
    feedback_initializer: InitializerSpec | None = None,
    activation: str = "tanh",
    bias: bool = True,
    trainable: bool = False,
    # Readout params
    readout: ReadoutSpec = "ridge",
    readout_alpha: float = 1e-6,
    readout_bias: bool = True,
    readout_name: str = "output",
    # Extra reservoir kwargs
    **reservoir_kwargs: Any,
) -> ESNModel:
    """Build a Deep (hierarchical, stacked-reservoir) Echo State Network.

    Stacks ``N`` reservoir layers so that layer ``i``'s state sequence is the
    feedback input of layer ``i + 1``.  The readout, by default, is fitted on the
    **concatenation of all layers' states** — the standard DeepESN readout — so
    it can exploit the progressively slower, more structured memory timescales
    the deeper layers develop.  Set ``concat_all_states=False`` to read out from
    the last layer alone.

    Architecture (``concat_all_states=True``)::

        Input -> Reservoir_1 -> Reservoir_2 -> ... -> Reservoir_N
                     |              |                     |
                     +--------------+----- Concatenate ---+---> Readout

    Only the first layer receives the raw feedback; every subsequent layer's
    feedback dimension is the previous layer's ``reservoir_size``.

    Parameters
    ----------
    reservoir_sizes : sequence of int or int
        Per-layer reservoir sizes as a list (one entry per layer), or a single
        int broadcast to all ``n_layers`` layers.  When a single int is given,
        ``n_layers`` is required.
    n_layers : int or None, optional
        Number of stacked layers.  Required when ``reservoir_sizes`` is a single
        int; ignored (but must be consistent) when ``reservoir_sizes`` is a list.
    feedback_size : int, default=1
        Number of feedback features driving the *first* layer (the input
        dimension of the model).
    output_size : int, default=1
        Number of output features.  For autoregressive :meth:`forecast`, this
        must equal ``feedback_size`` (the first model output is fed back).
    concat_all_states : bool, default=True
        When ``True`` (the standard DeepESN readout) the readout sees the
        concatenation of every layer's state, so its ``in_features`` is
        ``sum(reservoir_sizes)``.  When ``False`` the readout sees only the last
        layer's state, so ``in_features`` is ``reservoir_sizes[-1]``.
    interlayer_scaling : float, default=0.5
        Magnitude of the inter-layer coupling weights (layers ``2..N``).  Each
        deeper layer's feedback is the *previous layer's high-dimensional state*,
        so the default ``uniform(-1, 1)`` feedback weights would over-saturate the
        ``tanh``; scaling them down (via a ``uniform(-interlayer_scaling,
        interlayer_scaling)`` initializer) keeps the deeper reservoirs in their
        useful dynamic range.  Applied only when ``feedback_initializer`` is left
        as ``None``; an explicit ``feedback_initializer`` is used verbatim on
        every layer and this knob is ignored.  The first layer always uses the raw
        (unscaled) input path.
    topology : TopologySpec, optional
        Topology for the recurrent weights, shared by every layer.  Accepts a
        registry name, a ``(name, params)`` tuple, a callable, or a configured
        :class:`~resdag.init.topology.TopologyInitializer`.
    spectral_radius : float, default=0.9
        Target spectral radius for the recurrent weights of every layer.
    leak_rate : float or sequence of float, default=1.0
        Leaky-integration rate.  A scalar is shared by every layer; a list (one
        entry per layer) sets a per-layer schedule.  The DeepESN literature
        recommends leak rates that **decrease with depth** — fast dynamics near
        the input, slow dynamics deep in the stack — e.g.
        ``leak_rate=[1.0, 0.7, 0.4]`` for a 3-layer network.
    feedback_initializer : InitializerSpec, optional
        Initializer for the feedback weights, shared by every layer.  Same
        accepted forms as ``topology``.
    activation : str, default="tanh"
        Activation function for every layer (``"tanh"``, ``"relu"``,
        ``"sigmoid"``, ``"identity"``).
    bias : bool, default=True
        Whether every reservoir layer uses a bias term.
    trainable : bool, default=False
        Whether the reservoir weights are trainable.
    readout : str or type[ReadoutLayer], default="ridge"
        Which readout solver to attach.  ``"ridge"`` (default) is the exact
        direct Cholesky ridge solve (:class:`~resdag.layers.RidgeReadoutLayer`);
        ``"cg"`` selects the legacy iterative
        :class:`~resdag.layers.CGReadoutLayer`.  A
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass may also be passed.
    readout_alpha : float, default=1e-6
        Ridge regression regularization for the readout.
    readout_bias : bool, default=True
        Whether the readout uses a bias term.
    readout_name : str, default="output"
        Name for the readout layer (used as the target key in training).
    **reservoir_kwargs : Any
        Additional keyword arguments forwarded to **every**
        :class:`~resdag.layers.ESNLayer`.

    Returns
    -------
    :class:`~resdag.core.ESNModel`
        The assembled deep ESN, ready for training and forecasting.

    Raises
    ------
    ValueError
        If the layer stack is empty, a reservoir size is not a positive integer,
        ``reservoir_sizes`` is a scalar without ``n_layers``, the per-layer
        ``leak_rate`` schedule length does not match the number of layers, or
        ``output_size`` is not a positive integer.

    Notes
    -----
    The stack is *untrained* like a standard ESN — depth adds temporal richness
    without any extra gradient training; only the readout is fitted (by ridge
    regression).  Depth is not a free win over a single wide reservoir: on a
    short, single-timescale task (e.g. a brief Lorenz-63 window) a single
    ``classic_esn`` of the combined width usually forecasts *further*, because
    one large reservoir already spans the needed dynamics.  DeepESN pays off on
    **multi-timescale / hierarchical** signals — where a decreasing-with-depth
    leak schedule (see ``leak_rate`` below) lets successive layers specialise on
    progressively slower dynamics (Gallicchio & Micheli 2017; the layer-specific
    time-scale separation of Frontiers in AI 2024).  Reach for it when the signal
    has structure at several timescales, not as a drop-in replacement for a wide
    ESN.

    Examples
    --------
    A 3-layer stack of equal width with a decreasing-with-depth leak schedule::

    >>> from resdag.models import deep_esn
    >>> model = deep_esn(
    ...     reservoir_sizes=300,
    ...     n_layers=3,
    ...     feedback_size=3,
    ...     output_size=3,
    ...     leak_rate=[1.0, 0.6, 0.3],
    ...     spectral_radius=0.9,
    ... )

    Explicit per-layer sizes (narrowing with depth), reading out from the last
    layer only::

    >>> model = deep_esn(
    ...     reservoir_sizes=[400, 200, 100],
    ...     feedback_size=3,
    ...     output_size=3,
    ...     concat_all_states=False,
    ... )

    Training and forecasting are identical to any other premade model::

    >>> from resdag.training import ESNTrainer
    >>> trainer = ESNTrainer(model)
    >>> trainer.fit(
    ...     warmup_inputs=(warmup,),
    ...     train_inputs=(train,),
    ...     targets={"output": target},
    ... )
    >>> predictions = model.forecast(warmup, horizon=500)

    See Also
    --------
    classic_esn : Single-reservoir ESN with input concatenation.
    hybrid_esn : Physics-prior hybrid ESN.
    resdag.training.ESNTrainer : Trainer for fitting the readout.
    """
    sizes = _resolve_reservoir_sizes(reservoir_sizes, n_layers)
    leak_rates = _broadcast_leak_rates(leak_rate, len(sizes))

    if not isinstance(output_size, int) or output_size <= 0:
        raise ValueError(f"deep_esn: output_size must be a positive integer, got {output_size!r}.")

    # Resolve the readout spec up front so an invalid one fails before any
    # partial construction.
    readout_cls = _resolve_readout_cls(readout)

    inp = reservoir_input(feedback_size)

    # Build the stack: layer i takes the previous layer's state as its feedback,
    # so its feedback dimension is the previous reservoir_size (feedback_size for
    # the first layer). Collect every layer's state for the concat-all readout.
    layer_states = []
    prev_output = inp
    prev_feedback_size = feedback_size
    for i, (size, layer_leak) in enumerate(zip(sizes, leak_rates)):
        # First layer sees the raw input (user's feedback_initializer); deeper
        # layers see a high-dimensional reservoir state, so scale their coupling
        # down (unless the user pinned an explicit feedback_initializer).
        if i == 0 or feedback_initializer is not None:
            layer_feedback_init = feedback_initializer
        else:
            layer_feedback_init = (
                "uniform",
                {"low": -interlayer_scaling, "high": interlayer_scaling},
            )
        state = ESNLayer(
            reservoir_size=size,
            feedback_size=prev_feedback_size,
            topology=topology,
            spectral_radius=spectral_radius,
            leak_rate=layer_leak,
            feedback_initializer=layer_feedback_init,
            activation=activation,
            bias=bias,
            trainable=trainable,
            **reservoir_kwargs,
        )(prev_output)
        layer_states.append(state)
        prev_output = state
        prev_feedback_size = size

    # Standard DeepESN readout: concatenate every layer's state. When
    # concat_all_states is False, read out from the last layer alone.
    if concat_all_states and len(layer_states) > 1:
        features = Concatenate()(*layer_states)
        readout_in_features = sum(sizes)
    else:
        features = layer_states[-1]
        readout_in_features = sizes[-1]

    readout_layer = readout_cls(
        in_features=readout_in_features,
        out_features=output_size,
        bias=readout_bias,
        alpha=readout_alpha,  # type: ignore[call-arg]
        name=readout_name,
    )(features)

    return ESNModel(inp, readout_layer)
