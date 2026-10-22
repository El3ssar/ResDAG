"""Shared ESN-model builder behind the premade factories.

This private module hosts :func:`_esn_builder`, the single construction routine
that every premade factory in :mod:`resdag.models` delegates to.  The premade
architectures (``classic_esn``, ``ott_esn``, ``power_augmented``,
``headless_esn``, ``linear_esn``) differ only in a few orthogonal choices:

- which augmentation node (if any) is applied to the reservoir output,
- whether the raw input is concatenated back in before the readout,
- whether a readout layer is attached at all,
- which readout solver the readout uses (a direct Cholesky ridge solve by
  default, automatic selection or an explicit algorithm on request).

Rather than repeat the ``reservoir_input -> ESNLayer -> Concatenate ->
RidgeReadoutLayer -> ESNModel`` chain in five places, those choices are
expressed as arguments to :func:`_esn_builder`.

The default readout is :class:`~resdag.layers.RidgeReadoutLayer` with
``solver='cholesky'``. Normal equations can reduce fitting cost for tall,
well-conditioned data, but forming a Gram matrix squares the design's
condition number. Choose ``readout="auto"`` for precision-aware selection
and structured solve diagnostics, or select an algorithm/class explicitly.
Measure accuracy and runtime on the actual data, dtype and device.

This module is intentionally private; the public surface is the factory
functions exported from :mod:`resdag.models`.
"""

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from pytorch_symbolic.symbolic_data import SymbolicTensor

from resdag.core import ESNModel, reservoir_input
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer, RidgeReadoutLayer
from resdag.layers.readouts import (
    AutoReadoutLayer,
    CholeskyReadoutLayer,
    FISTAReadoutLayer,
    IncrementalRidgeReadoutLayer,
    PinvReadoutLayer,
    ReadoutLayer,
    RLSReadout,
    SVDReadoutLayer,
)
from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer

AugmentFactory = Callable[[], nn.Module]

# A readout spec is either a short string key naming a built-in solver, or a
# :class:`~resdag.layers.readouts.ReadoutLayer` subclass to instantiate.
ReadoutSpec = str | type[ReadoutLayer]

# Built-in string keys -> readout classes. ``"ridge"`` is the direct Cholesky
# ridge solver; ``"auto"`` adds conditioning checks and stable alternatives.
_READOUT_REGISTRY: dict[str, type[ReadoutLayer]] = {
    "auto": AutoReadoutLayer,
    "ridge": RidgeReadoutLayer,
    "cg": CGReadoutLayer,
    "cholesky": CholeskyReadoutLayer,
    "svd": SVDReadoutLayer,
    "pinv": PinvReadoutLayer,
    "incremental": IncrementalRidgeReadoutLayer,
    "rls": RLSReadout,
    "fista": FISTAReadoutLayer,
}


def _resolve_readout_cls(readout: ReadoutSpec) -> type[ReadoutLayer]:
    """Resolve a readout spec to a concrete :class:`ReadoutLayer` subclass.

    Parameters
    ----------
    readout : str or type[ReadoutLayer]
        Either a built-in key (such as ``"auto"``, ``"ridge"`` or ``"svd"``) or a
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass to use directly.

    Returns
    -------
    type[ReadoutLayer]
        The readout class to instantiate.

    Raises
    ------
    ValueError
        If ``readout`` is an unknown string key.
    TypeError
        If ``readout`` is neither a known string nor a ``ReadoutLayer``
        subclass.
    """
    if isinstance(readout, str):
        try:
            return _READOUT_REGISTRY[readout]
        except KeyError:
            valid = ", ".join(repr(k) for k in _READOUT_REGISTRY)
            raise ValueError(
                f"Unknown readout {readout!r}; expected one of {valid}, or a "
                f"ReadoutLayer subclass."
            ) from None
    if isinstance(readout, type) and issubclass(readout, ReadoutLayer):
        return readout
    raise TypeError(
        f"readout must be a string key ({', '.join(repr(k) for k in _READOUT_REGISTRY)}) "
        f"or a ReadoutLayer subclass, got {readout!r}."
    )


def _make_readout(
    readout: ReadoutSpec,
    *,
    in_features: int,
    out_features: int,
    alpha: float,
    bias: bool,
    name: str,
    readout_kwargs: dict[str, Any] | None = None,
) -> ReadoutLayer:
    """Construct a readout with solver options separate from reservoir options.

    Dimensions are fixed by the architecture. Other options can be overridden.
    Pseudoinverse readouts have no ridge penalty. For RLS, an explicit delta
    replaces the default alpha alias; explicitly supplying both remains an error.
    """
    readout_cls = _resolve_readout_cls(readout)
    options = dict(readout_kwargs or {})
    reserved = {"in_features", "out_features"}.intersection(options)
    if reserved:
        raise ValueError(
            f"readout_kwargs cannot override architecture dimensions: {sorted(reserved)}"
        )
    kwargs: dict[str, Any] = {"bias": bias, "name": name}
    if not issubclass(readout_cls, PinvReadoutLayer) and not (
        issubclass(readout_cls, RLSReadout) and "delta" in options
    ):
        kwargs["alpha"] = alpha
    kwargs.update(options)
    return readout_cls(in_features=in_features, out_features=out_features, **kwargs)


def _esn_builder(
    reservoir_size: int,
    feedback_size: int,
    output_size: int | None = None,
    *,
    augment: AugmentFactory | None = None,
    concat_input: bool = True,
    input_size: int | None = None,
    input_initializer: InitializerSpec | None = None,
    reservoir_cls: type[BaseReservoirLayer] = ESNLayer,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    leak_rate: float = 1.0,
    feedback_initializer: InitializerSpec | None = None,
    activation: str = "tanh",
    bias: bool = True,
    trainable: bool = False,
    # Readout params
    readout: ReadoutSpec = "ridge",
    readout_alpha: float = 1e-6,
    readout_bias: bool = True,
    readout_name: str = "output",
    readout_kwargs: dict[str, Any] | None = None,
    # Extra reservoir kwargs
    **reservoir_kwargs: Any,
) -> ESNModel:
    """Build an ESN model from composable, orthogonal building blocks.

    This is the shared construction routine behind every premade factory in
    :mod:`resdag.models`.  It always builds a reservoir, then optionally applies
    an augmentation node, optionally concatenates the raw input back in, and
    optionally attaches a ridge-regression readout.

    The readout defaults to :class:`~resdag.layers.RidgeReadoutLayer` with
    ``solver='cholesky'``. This direct normal-equations route suits
    well-conditioned data. Use ``readout="auto"`` to check conditioning and
    select stable alternatives without changing the L2 penalty. An explicit
    key or :class:`~resdag.layers.readouts.ReadoutLayer` subclass selects a
    custom readout.

    Architecture (each optional stage is skipped when not requested)::

        Input -> Reservoir -> [augment] -> [Concatenate(Input, .)] -> [Readout]

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features (input dimension).
    output_size : int or None, optional
        Number of output features.  Required (non-``None``) when a readout is
        attached; ignored when ``output_size`` is ``None`` (no readout, e.g. the
        headless and linear architectures).
    augment : callable, optional
        Zero-argument factory returning an :class:`~torch.nn.Module` applied to
        the reservoir output (e.g. ``SelectiveExponentiation`` or ``Power``).
        ``None`` (default) leaves the reservoir output untouched.
    concat_input : bool, default=True
        Whether to concatenate the raw input with the (possibly augmented)
        reservoir output before the readout.  Only meaningful when a readout is
        attached.
    input_size : int or None, optional
        Dimension of an optional driving (exogenous) input passed to the
        reservoir.  When given (a positive integer), a second symbolic input is
        built via :func:`~resdag.core.reservoir_input`, the reservoir is called
        as ``ESNLayer(...)(feedback, driver)``, and both inputs are wired into
        the returned :class:`~resdag.core.ESNModel`.  The driver is deliberately
        kept out of the input concatenation, so the readout ``in_features`` stays
        ``feedback_size + reservoir_size``.  ``None`` (default) builds a
        feedback-only reservoir, leaving the non-driven path unchanged.
    input_initializer : InitializerSpec, optional
        Initializer for the driving-input weights.  Same accepted forms as
        ``topology``.  Only used when ``input_size`` is given.
    reservoir_cls : type[BaseReservoirLayer], default=ESNLayer
        Reservoir layer class to instantiate.  Defaults to
        :class:`~resdag.layers.ESNLayer`; the edge-of-stability factories pass
        :class:`~resdag.layers.reservoirs.es2n.ES2NLayer` or
        :class:`~resdag.layers.reservoirs.resesn.ResESNLayer`.  ``leak_rate`` is
        forwarded only to ``ESNLayer`` (and subclasses accepting it); the
        skip layers replace leaky integration with their orthogonal skip and
        receive their own knobs (``beta`` / ``alpha`` / ``orthogonal_scaling``)
        through ``reservoir_kwargs``.
    topology : TopologySpec, optional
        Topology for recurrent weights.  Accepts a registry name, a
        ``(name, params)`` tuple, a callable, or a pre-configured
        :class:`~resdag.init.topology.TopologyInitializer`.
    spectral_radius : float, default=0.9
        Desired spectral radius for recurrent weights.
    leak_rate : float, default=1.0
        Leaky integration rate (1.0 = no leak).
    feedback_initializer : InitializerSpec, optional
        Initializer for feedback weights.  Same accepted forms as ``topology``.
    activation : str, default="tanh"
        Activation function (``"tanh"``, ``"relu"``, ``"sigmoid"``,
        ``"identity"``).
    bias : bool, default=True
        Whether to use bias in the reservoir.
    trainable : bool, default=False
        Whether reservoir weights are trainable.
    readout : str or type[ReadoutLayer], default="ridge"
        A built-in name (``"auto"``, ``"ridge"``, ``"cholesky"``, ``"cg"``, ``"svd"``,
        ``"pinv"``, ``"incremental"``, ``"rls"``, ``"fista"``) or a
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass.
    readout_alpha : float, default=1e-6
        Ridge regression regularization for the readout.
    readout_bias : bool, default=True
        Whether to use bias in the readout.
    readout_name : str, default="output"
        Name for the readout layer (used as the target key in training).
    readout_kwargs : dict, optional
        Solver-specific constructor options, including readout trainability.
        Architecture dimensions cannot be overridden.
    **reservoir_kwargs : Any
        Additional keyword arguments forwarded to
        :class:`~resdag.layers.ESNLayer`.

    Returns
    -------
    :class:`~resdag.core.ESNModel`
        The assembled model.  Its output is the readout when ``output_size`` is
        given, otherwise the (possibly augmented) reservoir output.

    Raises
    ------
    ValueError
        If a readout is requested (``output_size`` is not ``None``) but
        ``output_size`` is not a positive integer, or if ``readout`` is an
        unknown string key.
    TypeError
        If ``readout`` is neither a known string key nor a
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass.

    See Also
    --------
    resdag.models.classic_esn : Classic ESN built on this helper.
    resdag.models.ott_esn : Ott's state-augmented ESN built on this helper.
    """
    # Build the symbolic feedback input.  The time dimension is a placeholder —
    # actual sequence lengths are inferred from the input at call time.
    inp = reservoir_input(
        feedback_size, dtype=reservoir_kwargs.get("dtype"), device=reservoir_kwargs.get("device")
    )

    # An optional driving (exogenous) input.  When present the reservoir is
    # called with ``(feedback, driver)`` and both inputs are wired into the
    # model; the driver is kept out of the input concatenation below so the
    # readout ``in_features`` stays ``feedback_size + reservoir_size``.
    driver = (
        reservoir_input(
            input_size, dtype=reservoir_kwargs.get("dtype"), device=reservoir_kwargs.get("device")
        )
        if input_size
        else None
    )

    # Shared reservoir arguments understood by every ``BaseReservoirLayer``
    # subclass this builder wires (ESNLayer and the orthogonal-skip ES2NLayer /
    # ResESNLayer).  ``leak_rate`` is *not* shared: the skip layers replace the
    # leaky-integrator memory with their orthogonal skip, so it is added only
    # for ``ESNLayer`` (and subclasses that accept it).  Skip-specific knobs
    # (``beta`` / ``alpha`` / ``orthogonal_scaling``) arrive via
    # ``reservoir_kwargs`` from the ES2N / ResESN factories.
    reservoir_args: dict[str, Any] = dict(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        input_size=input_size,
        topology=topology,
        spectral_radius=spectral_radius,
        feedback_initializer=feedback_initializer,
        input_initializer=input_initializer,
        activation=activation,
        bias=bias,
        trainable=trainable,
    )
    if issubclass(reservoir_cls, ESNLayer):
        reservoir_args["leak_rate"] = leak_rate

    reservoir = reservoir_cls(**reservoir_args, **reservoir_kwargs)

    output: SymbolicTensor = reservoir(inp) if driver is None else reservoir(inp, driver)

    # Inputs wired into the final model: feedback alone, or [feedback, driver].
    model_inputs = inp if driver is None else [inp, driver]

    if augment is not None:
        output = augment()(output)

    # Headless / linear variants stop at the (possibly augmented) reservoir.
    if output_size is None:
        return ESNModel(model_inputs, output)

    if output_size <= 0:
        raise ValueError(f"output_size must be a positive integer, got {output_size!r}")

    if concat_input:
        output = Concatenate()(inp, output)
        readout_in_features = feedback_size + reservoir_size
    else:
        readout_in_features = reservoir_size

    reference = next(reservoir.parameters())
    readout_layer = _make_readout(
        readout,
        in_features=readout_in_features,
        out_features=output_size,
        bias=readout_bias,
        alpha=readout_alpha,
        name=readout_name,
        readout_kwargs=readout_kwargs,
    ).to(device=reference.device, dtype=reference.dtype)(output)

    return ESNModel(model_inputs, readout_layer)
