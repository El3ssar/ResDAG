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

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.cells import ESNCell
from resdag.layers.cells.esn_cell import SparseSpec
from resdag.utils.general import SeedLike

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
    bias_scaling : float, default=1.0
        Scale of the random bias: entries are drawn from
        ``uniform(-bias_scaling, bias_scaling)``.  Set to ``0.0`` for a
        zero bias (pre-0.5 behaviour).  Ignored when ``bias=False``.
    activation : {'tanh', 'relu', 'identity', 'sigmoid'}, default='tanh'
        Activation function for reservoir dynamics.
    leak_rate : float or 1-D tensor/sequence of shape ``(reservoir_size,)``, default=1.0
        Leaky integration rate in ``(0, 1]``.  A scalar 1.0 means no leaking
        (standard RNN update); smaller values create slower dynamics.  A 1-D
        tensor/sequence of length ``reservoir_size`` assigns a **per-neuron**
        leak (heterogeneous timescales in one reservoir, matching RC.jl's
        ``leak_coefficient`` vector); every entry must be finite and in
        ``(0, 1]``.  See :class:`~resdag.layers.cells.esn_cell.ESNCell` for the
        full contract (a vector leak is stored as a buffer that moves with
        ``.to()`` / ``.double()`` and round-trips through save/load).
    noise : float, default=0.0
        Standard deviation of additive Gaussian state noise injected after the
        activation (the classical ESN readout-conditioning regularizer).
        Active **only in training mode** (like dropout) and a no-op under
        :meth:`~torch.nn.Module.eval`.  The default ``0.0`` disables it, leaving
        outputs bit-identical to the noiseless layer.  Reproducible under the
        layer's ``seed``.  Must be non-negative.
    trainable : bool, default=False
        If ``True``, reservoir weights are trainable via backpropagation.
        Standard ESNs use frozen (non-trainable) weights.
    sparse : bool or {'auto'}, default=False
        Store a CSR copy of the frozen recurrent matrix and use a sparse
        matrix-vector product for the recurrent term (instead of the dense
        ``addmm``).  For RC-typical sparse reservoirs (density ~1-10%) this
        accelerates the recurrent matmul at large sizes and shrinks the
        matmul-operand / serialized-checkpoint footprint.  It does **not** reduce
        live memory: the dense ``weight_hh`` parameter stays resident as the
        canonical form, so a sparse cell's live footprint is ``dense(n²) +
        CSR(nnz)`` — slightly larger than dense-only, not smaller (no
        memory-ceiling lift).  ``False`` (default) keeps the dense path
        byte-for-byte unchanged; ``True`` forces the CSR path; ``"auto"`` enables
        it only for large, sparse reservoirs, with a **device-dependent** size
        gate (``reservoir_size >= 1000`` on CPU, ``>= 5000`` on CUDA, where the
        dense/sparse crossover is higher) and density ``<= 0.2``.  Requires
        ``trainable=False``; the dense math is preserved exactly (forecasts match
        the dense path to floating-point tolerance).  See
        :class:`~resdag.layers.cells.esn_cell.ESNCell` for the full contract,
        including the memory caveat.
    feedback_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Initializer for the feedback weight matrix.  Accepts a registry
        name, ``(name, params)``, any matrix-building callable
        ``fn(rows, cols, **kw) -> matrix`` (or in-place ``fn(tensor)``,
        e.g. ``torch.nn.init.xavier_uniform_``), ``(callable, params)``,
        or a configured initializer object.
    input_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Initializer for the input weight matrix.  Same formats as
        ``feedback_initializer``.  Only used when ``input_size`` is provided.
    topology : str, callable, tuple, or TopologyInitializer, optional
        Structure of the recurrent weight matrix.  Accepts a registry name
        (graph or matrix topology), ``(name, params)``, any matrix-building
        callable ``fn(n, **kw) -> matrix | nx.Graph`` (or in-place
        ``fn(tensor)``), ``(callable, params)``, or a configured topology
        object.
    recurrent_initializer : str, callable, tuple, or TopologyInitializer, optional
        First-class alias for ``topology`` accepting graph and matrix builders,
        registered names and custom callables. Supply only one of the two.
    bias_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Override the default random bias. Uses the same formats as input
        initialization on a ``(1, reservoir_size)`` tensor, then flattens to
        the bias vector. Requires ``bias=True``. Explicit hook values take
        precedence over ``bias_scaling``.
    initial_state_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Initialize a persistent ``(1, reservoir_size)`` state template once
        at construction. Every reset broadcasts an independent copy to each
        batch item. ``None`` keeps the zero-state default. The template moves
        with ``.to()`` and is included in checkpoints.
    device : torch.device or str, optional
        Device on which parameters are constructed. Defaults to the Torch
        default device. CUDA initialization runs on the requested device;
        MPS cells initialize on CPU and move the complete cell afterward,
        preserving compatibility across supported Torch versions.
    dtype : torch.dtype, optional
        Floating-point initialization precision. Defaults to Torch's default
        dtype. ``torch.float64`` constructs in double precision rather than
        rounding float32 weights and converting afterward.
    seed : int or torch.Generator, optional
        Reproducibility seed that deterministically fixes the *entire*
        reservoir — topology (recurrent) matrix, feedback weights, input
        weights, and bias — so two layers built with the same seed are
        identical down to the last entry.  Accepts a plain ``int`` or a
        :class:`torch.Generator` (handy for threading a per-trial generator
        from an HPO loop).  This covers both explicit initializers/topologies
        *and* the default ``uniform(-1, 1)`` draws used when none is given.
        With ``seed`` set, a string-form ``topology='erdos_renyi'`` is
        reproducible without the ``('erdos_renyi', {'seed': ...})`` tuple form;
        an explicit seed in a tuple/object spec always wins.  When ``seed=None``
        (the default), the reservoir is still reproducible under
        ``torch.manual_seed`` because every generator (NumPy and torch) is
        derived from torch's global RNG.

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

    Fully reproducible reservoir via the ``seed`` argument — recurrent matrix,
    feedback/input weights, and bias all match:

    >>> a = ESNLayer(50, feedback_size=3, topology="erdos_renyi", seed=42)
    >>> b = ESNLayer(50, feedback_size=3, topology="erdos_renyi", seed=42)
    >>> torch.equal(a.weight_hh, b.weight_hh)
    True
    >>> torch.equal(a.weight_feedback, b.weight_feedback)
    True
    >>> torch.equal(a.bias_h, b.bias_h)
    True

    A :class:`torch.Generator` works too (e.g. an HPO per-trial generator):

    >>> g1 = torch.Generator().manual_seed(7)
    >>> g2 = torch.Generator().manual_seed(7)
    >>> a = ESNLayer(50, feedback_size=3, seed=g1)
    >>> b = ESNLayer(50, feedback_size=3, seed=g2)
    >>> torch.equal(a.weight_feedback, b.weight_feedback)
    True

    The reservoir is also reproducible under ``torch.manual_seed`` without an
    explicit ``seed``:

    >>> torch.manual_seed(0)
    >>> a = ESNLayer(50, feedback_size=3, topology="erdos_renyi")
    >>> torch.manual_seed(0)
    >>> b = ESNLayer(50, feedback_size=3, topology="erdos_renyi")
    >>> torch.equal(a.weight_hh, b.weight_hh)
    True

    Stateful processing across batches:

    >>> out1 = reservoir(data1)  # State initialized
    >>> out2 = reservoir(data2)  # State carries over
    >>> reservoir.reset_state()  # Manual reset
    >>> out3 = reservoir(data3)  # Fresh state

    See Also
    --------
    resdag.layers.cells.esn_cell.ESNCell : Underlying single-step cell.
    resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer : Base providing state management.
    resdag.init.topology.TopologyInitializer : Base class for topology init.
    resdag.init.input_feedback.InputFeedbackInitializer : Input init base.
    resdag.core.ESNModel : Model composition using reservoir layers.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        spectral_radius: float | None = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        activation: str = "tanh",
        leak_rate: float | Tensor | Sequence[float] = 1.0,
        noise: float = 0.0,
        trainable: bool = False,
        sparse: SparseSpec = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
        seed: SeedLike = None,
        *,
        recurrent_initializer: TopologySpec = None,
        bias_initializer: InitializerSpec = None,
        initial_state_initializer: InitializerSpec = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        cell = ESNCell(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            input_size=input_size,
            spectral_radius=spectral_radius,
            bias=bias,
            bias_scaling=bias_scaling,
            activation=activation,
            leak_rate=leak_rate,
            noise=noise,
            trainable=trainable,
            sparse=sparse,
            feedback_initializer=feedback_initializer,
            input_initializer=input_initializer,
            topology=topology,
            seed=seed,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            initial_state_initializer=initial_state_initializer,
            device=device,
            dtype=dtype,
        )
        super().__init__(cell)

    def __getattr__(self, name: str) -> Any:
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
