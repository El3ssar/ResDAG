"""
Hybrid (physics-prior / knowledge-based) ESN Architecture
=========================================================

This module provides the hybrid reservoir-computing architecture of Pathak et
al. (2018): a reservoir *combined with an imperfect knowledge-based model* (a
physics prior).  A user-supplied "expert" — for example one integration step of
an approximate ODE — produces a one-step-ahead prediction ``K(u)`` that is fed
into the reservoir **twice**:

1. concatenated with the raw input *before* the reservoir (``[u, K(u)]`` drives
   the reservoir), and
2. concatenated with the reservoir state *before* the readout (the readout sees
   ``[h, K(u)]``).

The reservoir therefore only has to learn the *residual* the expert gets wrong,
which is a much easier target than the full dynamics.  Pathak's headline result
is robustness: even a crude expert lets the hybrid forecast a chaotic system far
longer than either the pure reservoir or the pure expert alone.

The reservoir's input weights use the :class:`~resdag.init.input_feedback.InformedInputInitializer`
(``informed_init``), which splits the reservoir units into a "state" fraction
``gamma`` wired to the raw-input block and a "model" fraction ``1 - gamma`` wired
to the knowledge-model block — exactly the wiring the hybrid needs.

:class:`KnowledgeModel` is the thin :class:`~torch.nn.Module` adapter that slots
an arbitrary callable ``f(u) -> u_next`` into the ``pytorch_symbolic`` DAG so it
is re-evaluated on the model's own output at every autoregressive forecast step.

References
----------
J. Pathak, A. Wikner, R. Fussell, S. Chandra, B. R. Hunt, M. Girvan and E. Ott,
"Hybrid forecasting of chaotic processes: Using machine learning in conjunction
with a knowledge-based model," Chaos 28, 041101 (2018).

See Also
--------
classic_esn : Single-reservoir ESN with input concatenation.
deep_esn : Stacked (hierarchical) reservoirs.
resdag.init.input_feedback.InformedInputInitializer : The informed input split.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from resdag.core import ESNModel, reservoir_input
from resdag.init.utils import TopologySpec
from resdag.layers import Concatenate, ESNLayer

from ._builder import ReadoutSpec, _make_readout

# A knowledge model maps a state tensor to a one-step-ahead prediction of the
# same feature dimension, broadcasting over any leading (batch/time) axes.
KnowledgeFn = Callable[[torch.Tensor], torch.Tensor]


class KnowledgeModel(nn.Module):
    """Wrap a knowledge-based expert ``f(u) -> u_next`` as a DAG layer.

    Adapts an arbitrary one-step-ahead predictor — a plain callable or another
    :class:`~torch.nn.Module` (e.g. one Euler/RK step of an approximate ODE) — so
    it slots into a ``pytorch_symbolic`` model and is re-evaluated on the current
    feedback at every timestep, including the autoregressive forecast loop where
    the reservoir feeds the model its own previous prediction.

    The wrapped callable is applied directly to the ``(batch, time, features)``
    (or single-step ``(batch, 1, features)``) tensor and must **broadcast over the
    leading batch/time axes**, operating feature-wise on the last dimension — the
    standard PyTorch idiom.  Its output feature dimension, ``model_out_size``,
    may differ from the input dimension (though it is typically equal, since the
    expert predicts the next state).

    Parameters
    ----------
    fn : callable or torch.nn.Module
        The knowledge model.  Called as ``fn(u)`` with ``u`` of shape
        ``(batch, time, features)`` and must return a tensor of shape
        ``(batch, time, model_out_size)`` — i.e. it may change only the last
        (feature) dimension.  A crude approximate integrator is enough; it does
        not need to be accurate, only informative.
    model_out_size : int
        Feature dimension of ``fn``'s output.  Used by :func:`hybrid_esn` to size
        the concatenations and the informed input split; validated at call time.

    Attributes
    ----------
    fn : callable or torch.nn.Module
        The wrapped knowledge model.  Registered as a submodule when it is an
        :class:`~torch.nn.Module`, so its parameters/buffers move with ``.to()``.
    model_out_size : int
        The declared output feature dimension.

    Raises
    ------
    ValueError
        At :meth:`forward` time, if ``fn``'s output rank does not match its input
        or its last dimension is not ``model_out_size``.

    Examples
    --------
    A rough Lorenz-63 Euler step used as a physics prior::

    >>> import torch
    >>> from resdag.models import KnowledgeModel
    >>> def lorenz_euler(u, dt=0.02, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    ...     x, y, z = u[..., 0], u[..., 1], u[..., 2]
    ...     dx = sigma * (y - x)
    ...     dy = x * (rho - z) - y
    ...     dz = x * y - beta * z
    ...     return u + dt * torch.stack([dx, dy, dz], dim=-1)
    >>> expert = KnowledgeModel(lorenz_euler, model_out_size=3)
    >>> u = torch.randn(4, 50, 3)
    >>> expert(u).shape
    torch.Size([4, 50, 3])

    See Also
    --------
    hybrid_esn : The factory that wires this expert into a hybrid ESN.
    """

    def __init__(self, fn: KnowledgeFn | nn.Module, model_out_size: int) -> None:
        """Store the callable and its declared output dimension.

        Raises
        ------
        TypeError
            If ``fn`` is not callable.
        ValueError
            If ``model_out_size`` is not a positive integer.
        """
        super().__init__()
        if not callable(fn):
            raise TypeError(f"KnowledgeModel: fn must be callable, got {type(fn).__name__}.")
        if not isinstance(model_out_size, int) or model_out_size <= 0:
            raise ValueError(
                f"KnowledgeModel: model_out_size must be a positive integer, "
                f"got {model_out_size!r}."
            )
        # Registering an nn.Module submodule under ``fn`` means its parameters and
        # buffers are discovered by ``.parameters()`` / moved by ``.to(...)``. A
        # plain callable is stored as a bare attribute (no parameters to track).
        self.fn = fn
        self.model_out_size = model_out_size

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply the knowledge model to the current state ``u``.

        Parameters
        ----------
        u : torch.Tensor
            Current state of shape ``(batch, time, features)`` (or the single-step
            ``(batch, 1, features)`` slice used during autoregressive forecasting).

        Returns
        -------
        torch.Tensor
            The one-step-ahead prediction of shape
            ``(batch, time, model_out_size)``.

        Raises
        ------
        ValueError
            If the wrapped model's output rank differs from ``u`` or its last
            dimension is not ``model_out_size``.
        """
        out = self.fn(u)
        if out.shape[:-1] != u.shape[:-1] or out.shape[-1] != self.model_out_size:
            raise ValueError(
                f"KnowledgeModel: fn(u) must return shape "
                f"{(*u.shape[:-1], self.model_out_size)} (same batch/time as the "
                f"input, last dim = model_out_size={self.model_out_size}); got "
                f"{tuple(out.shape)} for input {tuple(u.shape)}."
            )
        return out

    def extra_repr(self) -> str:
        fn_name = getattr(self.fn, "__name__", type(self.fn).__name__)
        return f"fn={fn_name}, model_out_size={self.model_out_size}"


def hybrid_esn(
    knowledge_model: KnowledgeModel | KnowledgeFn | nn.Module,
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    *,
    model_out_size: int | None = None,
    gamma: float = 0.5,
    input_scaling: float = 0.1,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    leak_rate: float = 1.0,
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
    """Build a hybrid physics-prior ESN (Pathak et al. 2018).

    Combines a reservoir with an imperfect knowledge-based expert ``K``.  The
    expert's one-step-ahead prediction ``K(u)`` is concatenated with the raw
    input to drive the reservoir, **and** concatenated with the reservoir state
    before the readout, so the reservoir only has to learn the residual the
    expert gets wrong.

    Architecture::

                     +---------------- K(u) --------------------+
                     |                                          |
        Input -- u --+--> Concatenate([u, K(u)]) --> Reservoir  |
                                                         |       v
                                                         +--> Concatenate([h, K(u)]) --> Readout

    The reservoir's input weights use the informed split
    (:class:`~resdag.init.input_feedback.InformedInputInitializer`): a fraction
    ``gamma`` of the units are wired to the raw-state block ``u`` and ``1 - gamma``
    to the knowledge-model block ``K(u)``.

    Parameters
    ----------
    knowledge_model : KnowledgeModel, callable, or torch.nn.Module
        The knowledge-based expert.  A bare callable / :class:`~torch.nn.Module`
        ``f(u) -> u_next`` is wrapped in a :class:`KnowledgeModel` automatically
        (using ``model_out_size``, defaulting to ``feedback_size``); pass a
        pre-built :class:`KnowledgeModel` to control its declared output size.
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features (the input dimension of the model, ``u``).
    output_size : int
        Number of output features.  For autoregressive :meth:`forecast` this must
        equal ``feedback_size`` (the first model output is fed back as ``u``).
    model_out_size : int or None, optional
        Feature dimension of the knowledge model's output.  Only used when
        ``knowledge_model`` is a bare callable (to build the wrapper); ignored when
        a :class:`KnowledgeModel` is passed (its own ``model_out_size`` is used).
        Defaults to ``feedback_size`` (the usual case: the expert predicts the
        next full state).
    gamma : float, default=0.5
        Fraction of reservoir units wired to the raw-state block ``u`` by the
        informed input split; ``1 - gamma`` are wired to the ``K(u)`` block.  Must
        lie in ``[0, 1]``.  ``gamma`` near ``1`` weights the reservoir toward the
        raw state, near ``0`` toward the knowledge model.
    input_scaling : float, default=0.1
        Half-width of the informed input weights: each nonzero entry is drawn
        uniformly from ``[-input_scaling, input_scaling]``.
    topology : TopologySpec, optional
        Topology for the recurrent weights.  Accepts a registry name, a
        ``(name, params)`` tuple, a callable, or a configured
        :class:`~resdag.init.topology.TopologyInitializer`.
    spectral_radius : float, default=0.9
        Target spectral radius for the recurrent weights.
    leak_rate : float, default=1.0
        Leaky-integration rate (1.0 = no leak).
    activation : str, default="tanh"
        Activation function (``"tanh"``, ``"relu"``, ``"sigmoid"``,
        ``"identity"``).
    bias : bool, default=True
        Whether the reservoir uses a bias term.
    trainable : bool, default=False
        Whether the reservoir weights are trainable.
    readout : str or type[ReadoutLayer], default="ridge"
        A built-in name (``"ridge"``, ``"cholesky"``, ``"cg"``, ``"svd"``,
        ``"pinv"``, ``"incremental"``, ``"rls"``, ``"fista"``) or a
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass. Additional
        solver options go in ``readout_kwargs``.
    readout_alpha : float, default=1e-6
        Ridge regression regularization for the readout.
    readout_bias : bool, default=True
        Whether the readout uses a bias term.
    readout_name : str, default="output"
        Name for the readout layer (used as the target key in training).
    readout_kwargs : dict, optional
        Extra readout options, such as ``trainable=True``, ``solver``,
        ``state_noise`` or ``gram_dtype``. Architecture dimensions are fixed.
    **reservoir_kwargs : Any
        Additional keyword arguments forwarded to :class:`~resdag.layers.ESNLayer`.

    Returns
    -------
    :class:`~resdag.core.ESNModel`
        The assembled hybrid ESN, ready for training and forecasting.

    Raises
    ------
    ValueError
        If ``output_size`` is not a positive integer, or (via the informed
        initializer) if the wiring dimensions are inconsistent.

    Notes
    -----
    The reservoir's ``feedback_size`` is ``feedback_size + model_out_size``
    because it consumes the concatenation ``[u, K(u)]``; the *model's* input
    dimension stays ``feedback_size`` (``K(u)`` is computed inside the graph).
    The readout ``in_features`` is ``reservoir_size + model_out_size`` (it sees
    ``[h, K(u)]``).  During autoregressive :meth:`forecast`, only the readout
    output is fed back as the next ``u``; ``K`` is re-evaluated on it every step,
    so the physics prior stays live throughout the rollout.

    Examples
    --------
    A rough Lorenz Euler step as the physics prior::

    >>> import torch
    >>> from resdag.models import hybrid_esn
    >>> def lorenz_euler(u, dt=0.02, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    ...     x, y, z = u[..., 0], u[..., 1], u[..., 2]
    ...     dx = sigma * (y - x)
    ...     dy = x * (rho - z) - y
    ...     dz = x * y - beta * z
    ...     return u + dt * torch.stack([dx, dy, dz], dim=-1)
    >>> model = hybrid_esn(
    ...     lorenz_euler,
    ...     reservoir_size=400,
    ...     feedback_size=3,
    ...     output_size=3,
    ...     gamma=0.5,
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
    KnowledgeModel : The expert wrapper used here.
    resdag.init.input_feedback.InformedInputInitializer : The informed input split.
    deep_esn : Stacked (hierarchical) reservoirs.
    """
    if not isinstance(output_size, int) or output_size <= 0:
        raise ValueError(
            f"hybrid_esn: output_size must be a positive integer, got {output_size!r}."
        )

    # Wrap a bare callable/Module in a KnowledgeModel; a pre-built one is used
    # as-is (its own model_out_size wins). The expert predicts the next state, so
    # model_out_size defaults to feedback_size.
    if isinstance(knowledge_model, KnowledgeModel):
        expert = knowledge_model
    else:
        expert = KnowledgeModel(
            knowledge_model,
            model_out_size=model_out_size if model_out_size is not None else feedback_size,
        )
    model_out = expert.model_out_size
    expert.to(device=reservoir_kwargs.get("device"), dtype=reservoir_kwargs.get("dtype"))

    # Resolve the readout spec up front so an invalid one fails before any
    # partial construction.

    # The model input u has dimension feedback_size; K(u) is computed in-graph, so
    # it is re-evaluated on the model's own output each autoregressive step.
    inp = reservoir_input(
        feedback_size, dtype=reservoir_kwargs.get("dtype"), device=reservoir_kwargs.get("device")
    )
    k = expert(inp)  # (batch, time, model_out)

    # Pre-reservoir: drive the reservoir with [u, K(u)].
    reservoir_in = Concatenate()(inp, k)
    reservoir_feedback_size = feedback_size + model_out

    # Informed input split: the first ``feedback_size`` columns are the raw-state
    # block, the trailing ``model_out`` columns the knowledge-model block. gamma is
    # the fraction of units wired to the state block.
    feedback_initializer = (
        "informed_init",
        {"model_in_size": model_out, "gamma": gamma, "input_scaling": input_scaling},
    )

    reservoir_state = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=reservoir_feedback_size,
        topology=topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        feedback_initializer=feedback_initializer,
        activation=activation,
        bias=bias,
        trainable=trainable,
        **reservoir_kwargs,
    )(reservoir_in)

    # Pre-readout: the readout sees [h, K(u)].
    readout_in = Concatenate()(reservoir_state, k)
    readout_in_features = reservoir_size + model_out

    readout_layer = _make_readout(
        readout,
        in_features=readout_in_features,
        out_features=output_size,
        bias=readout_bias,
        alpha=readout_alpha,
        name=readout_name,
        readout_kwargs=readout_kwargs,
    ).to(device=reservoir_kwargs.get("device"), dtype=reservoir_kwargs.get("dtype"))(readout_in)

    return ESNModel(inp, readout_layer)
