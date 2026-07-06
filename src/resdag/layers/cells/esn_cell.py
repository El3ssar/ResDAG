"""
ESN Cell
========

This module provides the concrete Echo State Network (ESN) implementation of
the reservoir cell:

- :class:`ESNCell` — single-timestep leaky-ESN update; owns all parameters.

See Also
--------
resdag.layers.cells.base_cell : Abstract base class (ReservoirCell).
resdag.init.topology : Topology initialization for recurrent weights.
resdag.init.input_feedback : Input/feedback weight initialization.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from resdag.init.topology import estimate_spectral_radius, scale_to_spectral_radius
from resdag.init.utils import InitializerSpec, TopologySpec, resolve_initializer, resolve_topology
from resdag.utils.general import SeedLike, coerce_seed_to_int, create_torch_generator

from .base_cell import ReservoirCell

# Sparse-CSR recurrent-matmul policy.
#
# ``sparse="auto"`` opts a *frozen* reservoir into a CSR ``weight_hh`` spmv only
# when both gates are met: the reservoir is large enough that the dense
# ``addmm`` actually dominates the per-step cost, *and* the recurrent matrix is
# sparse enough that a CSR spmv beats it.  Below either threshold the dense
# ``addmm`` is faster (its BLAS kernel has no per-nonzero overhead), so ``auto``
# stays dense there.  Explicit ``sparse=True`` bypasses these gates (the caller
# has asserted their topology is sparse); ``sparse=False`` (the default) is
# always dense and byte-for-byte unchanged.
#
# The size gate is **device-dependent**, because the dense-``addmm``/CSR-spmv
# crossover is: a GPU's dense BLAS is far faster per FLOP than a CPU's, so the
# reservoir must be much larger before the structural-zero savings of a CSR spmv
# pay off there.  Measured on this repo's hardware (recurrent matmul, batch=1):
#
#   * CPU float64 — sparse wins from N~1000 at RC densities (N=2000/1%: ~14x,
#     N=5000/1%: ~24x); dense wins for N<1000.  ``auto`` fires at N>=1000.
#   * CUDA float32 — dense wins through N=2000 (N=1000/1%: 0.53x, N=2000/1%:
#     0.91x — sparse *slower*); the crossover is ~N=3000, and sparse wins
#     decisively from N=5000 (6.9x at 1%).  ``auto`` therefore fires only at
#     N>=5000 on CUDA, so it never selects the slower sparse path in the
#     1000<=N<5000 GPU regime.
#
# Both thresholds are deliberately conservative so ``auto`` never regresses the
# default path on either device.  The density gate is shared (sparse only helps
# a genuinely sparse matrix on any device).
_SPARSE_AUTO_MIN_SIZE_CPU = 1000
_SPARSE_AUTO_MIN_SIZE_CUDA = 5000
_SPARSE_AUTO_MAX_DENSITY = 0.2

# Back-compat alias: the CPU size gate was the sole threshold before the
# device-aware split.  Kept so existing imports (and tests) that reference
# ``_SPARSE_AUTO_MIN_SIZE`` continue to resolve to the CPU floor.
_SPARSE_AUTO_MIN_SIZE = _SPARSE_AUTO_MIN_SIZE_CPU

# Type of the ``sparse`` construction flag.
SparseSpec = bool | Literal["auto"]

# Golden-ratio offsets used to derive independent, deterministic sub-seeds from
# the single layer-level ``seed``.  ``0x9E3779B9`` (2**32 / phi) is the classic
# Fibonacci-hashing additive constant; doubling it gives a second, distinct
# offset for the input-weight stream.
#
# ``_NOISE_SEED_OFFSET`` separates the train-mode state-noise stream from the
# weight-init draws (historical behaviour, unchanged).  ``_INPUT_SEED_OFFSET``
# separates the *input* weight init from the *feedback* weight init: feeding the
# same named initializer the identical integer for both roles produced
# ``weight_input == weight_feedback``, collapsing feedback and driving input
# into a single projection.  The two offsets differ so the derived integer
# seeds never coincide.
_NOISE_SEED_OFFSET = 0x9E3779B9
_INPUT_SEED_OFFSET = (0x9E3779B9 * 2) % (2**63 - 1)


def _salt_seed(seed: int | None, offset: int) -> int | None:
    """Derive a distinct-but-deterministic integer seed from ``seed``.

    Adds a fixed ``offset`` modulo the 63-bit seed range so a single
    layer-level seed can drive several independent init streams (feedback vs
    input weights, weight init vs state noise).  ``None`` is passed through
    unchanged, keeping the stream tied to torch's global RNG.

    Parameters
    ----------
    seed : int or None
        Base integer seed (typically ``self._seed_int``), or ``None``.
    offset : int
        Additive constant that separates this stream from the base stream.

    Returns
    -------
    int or None
        ``(seed + offset) % (2**63 - 1)`` when ``seed`` is an ``int``,
        otherwise ``None``.
    """
    if seed is None:
        return None
    return (seed + offset) % (2**63 - 1)


def _identity(x: torch.Tensor) -> torch.Tensor:
    """Identity activation.

    Defined at module level (rather than as a ``lambda``) so that cells using
    ``activation="identity"`` — e.g. :func:`resdag.models.linear_esn` — stay
    picklable.  A local lambda would make whole-model serialization
    (``ESNModel.save_full`` / ``torch.save(model)``) fail with a
    ``PicklingError``.
    """
    return x


class ESNCell(ReservoirCell):
    """
    Single-timestep leaky Echo State Network update.

    Owns all weight matrices and bias.  Sequence iteration is delegated to
    the enclosing :class:`ESNLayer`.

    The state update follows the standard leaky-integrator ESN equation
    (Jaeger 2001; Lukoševičius 2012):

    .. math::

        h_t = (1 - \\alpha)\\,h_{t-1} + \\alpha\\,f(W_{fb}\\,x_{fb,t}
               + W_{in}\\,x_{in,t} + W_{rec}\\,h_{t-1} + b)

    where :math:`f` is the activation function, :math:`\\alpha` is the leak
    rate, :math:`W_{fb}` is the feedback weight matrix, :math:`W_{in}` is the
    (optional) input weight matrix, :math:`W_{rec}` is the recurrent
    weight matrix, and :math:`b` is a fixed random bias drawn from
    :math:`\\mathcal{U}(-\\beta, \\beta)` with :math:`\\beta` = ``bias_scaling``.

    The bias breaks the odd symmetry of ``tanh`` dynamics: without it,
    negated inputs produce exactly negated states, which constrains the
    representations the readout can draw from.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (hidden state dimension).
    feedback_size : int
        Dimension of feedback signal.  Required for all ESN cells.
    input_size : int, optional
        Dimension of driving inputs.  If ``None``, no driving input weight
        matrix is created.
    spectral_radius : float, optional
        Target spectral radius for recurrent weights.  If ``None``, no
        spectral radius scaling is applied.
    bias : bool, default=True
        Whether to include a bias term.
    bias_scaling : float, default=1.0
        Scale of the random bias: entries are drawn from
        ``uniform(-bias_scaling, bias_scaling)``, matching the default
        initialization of the feedback/input weights.  Ignored when
        ``bias=False``.  Set to ``0.0`` to keep a zero bias (the historical
        pre-0.5 behaviour, where ``bias=True`` was effectively a no-op
        because the bias was zero-initialized and frozen).
    activation : {'tanh', 'relu', 'identity', 'sigmoid'}, default='tanh'
        Activation function for reservoir dynamics.
    leak_rate : float or 1-D tensor/sequence of shape ``(reservoir_size,)``, default=1.0
        Leaky integration rate in ``(0, 1]``.  A scalar value of 1.0 means no
        leaking (standard RNN update); smaller values create slower dynamics.
        Values outside ``(0, 1]`` are rejected at construction with a
        ``ValueError``: ``0.0`` would freeze the state entirely and values
        ``> 1.0`` (or ``< 0``) produce ill-defined dynamics.

        A **per-neuron** leak rate — a 1-D tensor or sequence of length
        ``reservoir_size`` — assigns each unit its own timescale (heterogeneous
        reservoir, matching RC.jl's ``leak_coefficient`` vector).  Every entry
        must be finite and in ``(0, 1]``; a wrong length or any out-of-range /
        non-finite entry raises ``ValueError``.  A vector leak is stored as a
        registered buffer (``_leak_rate_vec``), so it moves with
        ``.to(device)`` / ``.double()`` and round-trips through save/load; a
        scalar stays a plain float with byte-identical dynamics to the
        pre-vector cell.
    noise : float, default=0.0
        Standard deviation of additive Gaussian state noise injected after the
        activation, following the classical ESN regularizer (Jaeger 2001;
        Lukoševičius 2012):

        .. math::

            \\tilde{h}_t = f(\\cdot) + \\nu\\,\\epsilon_t,
            \\quad \\epsilon_t \\sim \\mathcal{N}(0, 1)

        where :math:`\\nu` = ``noise``.  Noise is applied **only in training
        mode** (``self.training is True``, like dropout) and is a no-op under
        :meth:`~torch.nn.Module.eval`.  The default ``0.0`` disables it
        entirely, leaving outputs bit-identical to the noiseless cell.  The
        noise stream is reproducible: it is seeded deterministically from
        ``seed`` (independently of the weight-init draws), so a given ``seed``
        reproduces the same perturbations on every run.  Must be non-negative.
    trainable : bool, default=False
        If ``True``, reservoir weights are trainable via backpropagation.
        Standard ESNs use frozen (non-trainable) weights.
    sparse : bool or {'auto'}, default=False
        Store a CSR copy of the (frozen) recurrent matrix and use a sparse
        matrix-vector product for the recurrent term, instead of the dense
        ``addmm``.  For an RC-typical sparse reservoir (density ~1-10%) the CSR
        spmv skips the structural zeros a dense BLAS kernel still multiplies, so
        it **accelerates the per-step recurrent matmul** at large sizes (the hot
        inner-loop kernel of warmup / teacher-forcing / forecasting) and
        **shrinks the serialized-on-disk / matmul-operand footprint** (a 20k-unit
        float32 recurrent matrix is ~1.6 GB dense; its CSR form at 1% density is
        ~48 MB — the size the spmv actually touches, and the size a CSR-serialized
        checkpoint would be).

        .. warning::

           This is **not** a live-memory reduction, and does **not** lift the
           dense ``(reservoir_size, reservoir_size)`` memory ceiling.  The dense
           ``weight_hh`` :class:`~torch.nn.Parameter` is *always* materialised —
           it is the canonical, serialized form used for introspection
           (spectral-radius, Lyapunov), ``state_dict`` save/load, and
           ``torch.equal`` comparisons — and the CSR operator is an
           **additional** non-persistent buffer built from it.  A sparse cell's
           resident memory is therefore ``dense(n²) + CSR(nnz)``, i.e. slightly
           *larger* than a dense-only cell, not smaller.  ``sparse=True`` buys
           recurrent-matmul speed (at low density and large ``n``) and a smaller
           matmul operand / potential checkpoint, not a smaller live model.  To
           actually build a reservoir that does not fit dense, a future change
           would need to make the CSR the canonical resident form and free the
           dense ``weight_hh`` (reconstructing dense only for introspection /
           save) — that is deliberately out of scope here.

        Three settings:

        * ``False`` (default) — always dense; byte-for-byte identical to the
          historical cell.
        * ``True`` — force the CSR spmv path (the caller asserts their topology
          is sparse enough to benefit).
        * ``"auto"`` — enable the CSR path only when the reservoir is large *and*
          the initialized recurrent matrix is sparse (density ``<= 0.2``);
          otherwise fall back to dense, where the dense ``addmm`` is faster.  The
          size gate is **device-dependent** — the dense/sparse crossover is much
          higher on a GPU (its dense BLAS is fast), so ``auto`` fires only at
          ``reservoir_size >= 5000`` on CUDA but at ``reservoir_size >= 1000`` on
          CPU (both measured on this repo's hardware).  This keeps ``auto`` from
          selecting the *slower* sparse path in the ``1000 <= n < 5000`` GPU
          regime, where dense still wins.  The decision consults the recurrent
          weight's device, so a cell moved to GPU before the decision is re-taken
          (e.g. on ``load_state_dict``) sees the right gate.

        The sparse path requires a **frozen** reservoir: it is only valid with
        ``trainable=False`` (sparse-CSR autograd for this update is fragile).
        Requesting ``sparse`` together with ``trainable=True`` raises
        ``ValueError``.  Flipping ``trainable=True`` on a live sparse cell
        transparently drops the CSR buffer and reverts to the dense path (with a
        ``RuntimeWarning``).  Because the dense ``weight_hh`` parameter is the
        canonical, serialized form, save/load, spectral-radius introspection, and
        Lyapunov analysis are unchanged, and the forecast ``compile=True`` path
        falls back to eager (``torch.compile`` does not trace sparse tensors).
    feedback_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Initializer for the feedback weight matrix.  Accepts a registry
        name, ``(name, params)``, any matrix-building callable, a
        ``(callable, params)`` tuple, or a configured initializer object.
    input_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Initializer for the input weight matrix.  Same formats as
        ``feedback_initializer``.  Only used when ``input_size`` is provided.
    topology : str, callable, tuple, or TopologyInitializer, optional
        Structure of the recurrent weight matrix: a registry name (graph or
        matrix topology), any matrix-building callable, a
        ``(callable, params)`` tuple, or a configured topology object.
    seed : int or torch.Generator, optional
        Reproducibility seed that deterministically fixes *every* reservoir
        parameter — the recurrent (topology) matrix, the feedback and input
        weights (including the default ``uniform(-1, 1)`` draw used when no
        initializer is given), and the random bias. Accepts a plain ``int`` or
        a :class:`torch.Generator` (an int is extracted from the generator's
        ``initial_seed()`` for the NumPy-backed topology/named-initializer
        path, while the generator itself drives the torch default-init draws).
        With ``seed`` set, a string- or callable-form
        ``topology='erdos_renyi'`` produces the same recurrent matrix on every
        build, without the ``('erdos_renyi', {'seed': ...})`` tuple form. An
        explicit ``seed`` inside a tuple/object spec always wins over this
        argument. When ``seed=None`` (the default), string-form graph
        topologies and the default torch draws are still reproducible under
        ``torch.manual_seed`` because their generators are derived from torch's
        global RNG.

    Attributes
    ----------
    weight_feedback : torch.nn.Parameter
        Feedback weight matrix of shape ``(reservoir_size, feedback_size)``.
    weight_input : torch.nn.Parameter or None
        Input weight matrix of shape ``(reservoir_size, input_size)``, or
        ``None`` if ``input_size`` was not provided.
    weight_hh : torch.nn.Parameter
        Recurrent weight matrix of shape ``(reservoir_size, reservoir_size)``.
    bias_h : torch.nn.Parameter or None
        Bias vector of shape ``(reservoir_size,)``, or ``None`` if
        ``bias=False``.

    See Also
    --------
    resdag.layers.esn.ESNLayer : Layer that sequences this cell.
    resdag.layers.base.ReservoirCell : Abstract cell interface.
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
        leak_rate: float | torch.Tensor | Sequence[float] = 1.0,
        noise: float = 0.0,
        trainable: bool = False,
        sparse: SparseSpec = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
        seed: SeedLike = None,
    ) -> None:
        super().__init__()

        # A sparse-CSR recurrent matmul is only valid for a frozen reservoir;
        # sparse-CSR autograd for this update is fragile, so reject the
        # trainable+sparse combination up front rather than silently degrading.
        if sparse and trainable:
            raise ValueError(
                "sparse=True requires a frozen reservoir (trainable=False): "
                "sparse-CSR autograd for the recurrent update is unsupported. "
                "Use a dense reservoir (sparse=False) for a trainable ESN."
            )

        # leak_rate / noise are validated by their property setters (below), so
        # both the constructor path (``self.leak_rate = ...``) and live writes
        # (``layer.leak_rate = ...``) share one validation.

        # Store configuration
        self.reservoir_size = reservoir_size
        self.feedback_size = feedback_size
        # Treat input_size == 0 the same as input_size is None (no driving
        # input weight matrix is created).  This avoids the historical
        # (reservoir_size, 0) zero-column tensor produced by passing ``0``
        # explicitly from the premade factories.
        self.input_size = input_size if input_size else None
        self.topology = topology
        self.spectral_radius = spectral_radius
        self.feedback_initializer = feedback_initializer
        self.input_initializer = input_initializer
        # Backing fields for the leak-rate property.  A scalar leak lives in the
        # float ``_leak_rate``; a per-neuron leak lives in the *persistent*
        # buffer ``_leak_rate_vec`` (so it moves with ``.to()`` / ``.double()``
        # and round-trips through ``state_dict()`` / save-load) and flips
        # ``_is_vector_leak`` so the ``!= 1.0`` scalar fast path is bypassed.
        # Both are pre-seeded here so the property setter — shared by the
        # constructor and live writes — can branch on ``_is_vector_leak`` and
        # reassign the buffer freely (buffers accept reassignment, incl. back to
        # ``None``).
        self._leak_rate: float = 1.0
        self._is_vector_leak: bool = False
        self.register_buffer("_leak_rate_vec", None, persistent=True)
        # ``register_buffer(..., None)`` leaves mypy inferring ``None``; annotate
        # the true buffer type so the vector-leak paths type-check.
        self._leak_rate_vec: torch.Tensor | None
        self.leak_rate = leak_rate
        self.noise = noise
        # ``trainable`` is a live property (see below) whose setter flips
        # ``requires_grad_`` on every parameter.  During ``__init__`` the
        # parameters do not exist yet, so stash the requested value in the
        # backing field directly; the actual ``requires_grad`` freeze is applied
        # after ``_initialize_weights`` (see the block at the end of __init__).
        self._trainable = trainable
        # Sparse-CSR recurrent-matmul config.  ``_sparse_requested`` records the
        # user's intent (``True`` / ``False`` / ``"auto"``); ``_sparse_active``
        # is the resolved decision (set once the recurrent matrix exists and its
        # density is known — see ``_maybe_build_sparse_weight``).  The derived
        # CSR operator lives in the *non-persistent* buffer ``_weight_hh_csr``
        # (rebuilt from the dense ``weight_hh`` on load / device move), keeping
        # the dense ``weight_hh`` parameter as the single canonical, serialized
        # form so introspection and save/load are unchanged.
        self._sparse_requested: SparseSpec = sparse
        self._sparse_active: bool = False
        self.register_buffer("_weight_hh_csr", None, persistent=False)
        self._weight_hh_csr: torch.Tensor | None
        self.seed = seed
        # Integer form threaded into the NumPy-backed topology builders and the
        # named feedback/input initializers (they take an int/None seed).  A
        # torch.Generator is reduced to its initial_seed() so the topology stays
        # a pure function of the generator.
        self._seed_int = coerce_seed_to_int(seed)
        # Per-device cache of torch Generators driving the train-mode state
        # noise.  Built lazily on first use (and rebuilt after unpickling — see
        # ``__getstate__``) so the noise stream follows the state's device while
        # staying a deterministic function of ``seed``.  Drawn from a stream
        # *independent* of the weight-init generator so that toggling ``noise``
        # never perturbs weight reproducibility.
        self._noise_generators: dict[torch.device, torch.Generator] = {}

        # Activation function
        self._activation_name = activation
        self.activation_fn = self._get_activation(activation)

        # Store bias config before initialization
        self._bias = bias
        self.bias_scaling = bias_scaling

        # Initialize weight matrices
        self._initialize_weights()

        # Apply the requested trainable state to the freshly-built parameters.
        # Going through the property setter (rather than an inline freeze loop)
        # keeps construction and post-init ``cell.trainable = ...`` writes on a
        # single code path: both flip ``requires_grad_`` on *all* parameters.
        self.trainable = trainable

        # Resolve the sparse-CSR decision now that ``weight_hh`` exists and its
        # density is known, and materialise the derived CSR operator if enabled.
        # Done after the ``trainable`` freeze so the guard sees the final state.
        self._maybe_build_sparse_weight()

    # ------------------------------------------------------------------
    # ReservoirCell interface
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        """Dimensionality of the hidden state vector."""
        return self.reservoir_size

    @property
    def output_size(self) -> int:
        """Dimensionality of the per-step output (equals state_size for ESN)."""
        return self.reservoir_size

    @property
    def trainable(self) -> bool:
        """
        bool : Whether the reservoir weights participate in backpropagation.

        This is a *live* switch, not a construction-time-only flag.  Assigning
        to it flips ``requires_grad_`` on **every** cell parameter
        (``weight_hh``, ``weight_feedback``, ``weight_input``, ``bias_h``) so a
        post-construction ``cell.trainable = True`` genuinely unfreezes the
        reservoir for a BPTT rollout — the documented
        ``reservoir.trainable = True`` recipe now takes effect instead of being
        a silent no-op.  The getter reflects the current state.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        was_trainable = self._trainable
        self._trainable = bool(value)
        for p in self.parameters():
            p.requires_grad_(self._trainable)
        # The sparse-CSR recurrent path is frozen-only.  A live flip to trainable
        # must drop it (sparse-CSR autograd for this update is fragile); a flip
        # back to frozen may re-materialise it.  Guarded by ``hasattr`` because
        # the setter also runs during ``__init__`` before the sparse machinery
        # is wired.
        if self._trainable and not was_trainable and getattr(self, "_sparse_active", False):
            warnings.warn(
                "trainable=True on a sparse reservoir: densifying weight_hh and "
                "reverting to the dense recurrent path (sparse-CSR autograd for "
                "the recurrent update is unsupported).",
                RuntimeWarning,
                stacklevel=2,
            )
            self._sparse_active = False
            self._weight_hh_csr = None
        elif not self._trainable and was_trainable and hasattr(self, "_sparse_requested"):
            # Re-freezing: honour the original sparse request again.
            self._maybe_build_sparse_weight()

    @property
    def leak_rate(self) -> float | torch.Tensor:
        """
        float or torch.Tensor : Leaky-integration rate(s) read fresh every step.

        A *live* knob: assigning to it (directly or through
        ``ESNLayer.leak_rate = ...``) validates the value and takes effect on the
        next forward.  Returns the scalar ``float`` for a homogeneous leak
        (``1.0`` means no leaking, standard ESN) or the ``(reservoir_size,)``
        buffer for a per-neuron heterogeneous leak.  Every entry stays in
        ``(0, 1]``.
        """
        if self._is_vector_leak:
            assert self._leak_rate_vec is not None  # invariant when flag is set
            return self._leak_rate_vec
        return self._leak_rate

    @leak_rate.setter
    def leak_rate(self, value: float | torch.Tensor | Sequence[float]) -> None:
        # A tensor/sequence selects the per-neuron vector path; anything scalar
        # keeps the byte-identical float path.  ``torch.Tensor`` and
        # non-str/bytes sequences are treated as vectors; a 0-d tensor is a
        # scalar in disguise, so it falls through to the float branch.
        if (isinstance(value, torch.Tensor) and value.dim() > 0) or (
            not isinstance(value, (str, bytes)) and isinstance(value, Sequence)
        ):
            self._set_vector_leak(value)
        else:
            self._set_scalar_leak(value)

    def _set_scalar_leak(self, value: float | torch.Tensor | Sequence[float]) -> None:
        """Validate and store a homogeneous scalar leak (the byte-identical path).

        Accepts a Python scalar or a 0-d tensor; both collapse to a plain
        ``float`` so the ``!= 1.0`` fast path in :meth:`forward` / :meth:`step`
        stays unchanged.  Clears any previously-set per-neuron buffer.  The
        ``Sequence`` arm of the type is never reached at runtime (the setter
        dispatches sequences to :meth:`_set_vector_leak`); it only widens the
        signature to the shared leak-rate union.
        """
        scalar = float(value)  # type: ignore[arg-type]
        if not 0.0 < scalar <= 1.0:
            raise ValueError(f"leak_rate must be in (0, 1], got {value}")
        self._leak_rate = scalar
        self._is_vector_leak = False
        # Drop any stale per-neuron buffer so a scalar overwrite of a vector leak
        # frees the buffer and keeps ``state_dict()`` clean.
        self._leak_rate_vec = None

    def _set_vector_leak(self, value: float | torch.Tensor | Sequence[float]) -> None:
        """Validate and store a per-neuron leak vector as a persistent buffer.

        The vector must be 1-D of length ``reservoir_size`` with every entry
        finite and in ``(0, 1]``.  It is stored as a buffer (matching the
        cell's current parameter dtype/device when available) so it moves under
        ``.to()`` / ``.double()`` and round-trips through save/load; the
        ``_is_vector_leak`` flag then routes the per-step update through an
        unconditional ``torch.lerp`` (the scalar ``!= 1.0`` shortcut cannot be
        used once the leak is a tensor).
        """
        vec = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if not vec.is_floating_point():
            vec = vec.to(torch.get_default_dtype())
        if vec.dim() != 1 or vec.shape[0] != self.reservoir_size:
            raise ValueError(
                f"vector leak_rate must have shape ({self.reservoir_size},) to match "
                f"reservoir_size, got shape {tuple(vec.shape)}"
            )
        if not torch.isfinite(vec).all():
            raise ValueError("vector leak_rate must be finite (no NaN/Inf entries)")
        if not bool(((vec > 0.0) & (vec <= 1.0)).all()):
            raise ValueError("every leak_rate entry must be in (0, 1]")

        # Align the buffer with the cell's existing float parameters (dtype +
        # device) so a vector passed at construction — before the caller calls
        # ``.to()`` — already sits where the weights are.  Falls back to the
        # freshly-built tensor when no float parameter exists yet.
        ref = next((p for p in self.parameters() if p.is_floating_point()), None)
        if ref is not None:
            vec = vec.to(dtype=ref.dtype, device=ref.device)
        self._leak_rate_vec = vec.contiguous()
        self._is_vector_leak = True

    def _reconcile_leak_buffer(self, state_dict: dict, prefix: str) -> None:
        """Align the optional per-neuron leak buffer with the checkpoint.

        The vector-leak buffer ``_leak_rate_vec`` is present in a checkpoint only
        when the saved cell had a *vector* leak (a scalar-leak cell registers it
        as ``None``, so it is absent from ``state_dict``).  Without this
        reconciliation, loading a vector-leak checkpoint into a scalar-rebuilt
        cell (or vice versa) would raise on the mismatched key or silently drop
        the leak configuration.  We align our buffer's presence — and the
        :attr:`_is_vector_leak` flag — to the incoming checkpoint before the
        normal load, so the loaded leak configuration always wins.

        Called from :meth:`_load_from_state_dict` before the parent copy.
        """
        vec_key = prefix + "_leak_rate_vec"
        if vec_key in state_dict:
            # Checkpoint carries a vector leak: materialise a buffer of the right
            # shape so the parent copy has a target, and flip the flag.
            self._leak_rate_vec = torch.empty_like(state_dict[vec_key])
            self._is_vector_leak = True
        else:
            # Checkpoint has a scalar leak: drop any vector buffer we carry so a
            # strict load neither reports a missing key nor keeps a stale vector.
            self._leak_rate_vec = None
            self._is_vector_leak = False

    @property
    def noise(self) -> float:
        """
        float : Std-dev of additive train-mode state noise (``>= 0``).

        A *live* knob validated on assignment; ``0.0`` disables noise.  Noise is
        applied only when the cell is in training mode (see ``_apply_noise``).
        """
        return self._noise

    @noise.setter
    def noise(self, value: float) -> None:
        if float(value) < 0.0:
            raise ValueError(f"noise must be non-negative, got {value}")
        self._noise = float(value)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a zero hidden state of shape ``(batch_size, reservoir_size)``."""
        return torch.zeros(batch_size, self.state_size, device=device, dtype=dtype)

    @property
    def activation(self) -> str:
        """
        str : Name of the activation function.
        """
        return self._activation_name

    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the next ESN state for a single timestep.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Per-timestep input slices.  ``inputs[0]`` is the feedback slice
            of shape ``(batch, feedback_size)``.  If a driving input is
            present, ``inputs[1]`` has shape ``(batch, input_size)``.
        state : torch.Tensor
            Current hidden state of shape ``(batch, reservoir_size)``.

        Returns
        -------
        output : torch.Tensor
            Next hidden state of shape ``(batch, reservoir_size)``.
        new_state : torch.Tensor
            Same tensor as output (state and output are identical for ESN).

        Raises
        ------
        ValueError
            If the feedback feature dimension does not match
            ``self.feedback_size``, if a driving input is provided but
            ``self.weight_input`` is ``None``, or if the driving input
            feature dimension does not match ``self.input_size``.

        Notes
        -----
        When ``self.noise > 0`` and the cell is in training mode, additive
        Gaussian noise is injected into the post-activation state (see the
        ``noise`` constructor parameter).  This matches the noise applied in
        the :meth:`step` fast path, so the two paths stay consistent.
        """
        projected = self.project_inputs(inputs)
        new_state = self.activation_fn(projected + self._recurrent(state))
        new_state = self._apply_noise(new_state)
        new_state = self._apply_leak(state, new_state)
        return new_state, new_state

    def project_inputs(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Precompute all input-dependent pre-activation terms.

        Computes ``W_fb x_fb + W_in x_in + b`` for every timestep at once.
        Works on full ``(batch, timesteps, features)`` sequences (the layer's
        fast path) and on single-step ``(batch, features)`` slices (reused by
        :meth:`forward`).

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Input streams; ``inputs[0]`` is feedback, ``inputs[1]`` (if
            present) is the driving input.

        Returns
        -------
        torch.Tensor
            Input projection with the same leading dimensions as the inputs
            and trailing dimension ``reservoir_size``.

        Raises
        ------
        ValueError
            If feature dimensions do not match the cell configuration, or a
            driving input is supplied to a cell built without ``input_size``.
        """
        fb = inputs[0]

        if fb.shape[-1] != self.feedback_size:
            raise ValueError(
                f"Feedback size mismatch. Expected {self.feedback_size}, got {fb.shape[-1]}"
            )

        projected = F.linear(fb, self.weight_feedback)

        if len(inputs) > 1:
            if self.weight_input is None:
                raise ValueError(
                    "Reservoir was initialized without input_size, "
                    "but driving input was provided in forward pass"
                )
            x = inputs[1]
            if x.shape[-1] != self.input_size:
                raise ValueError(
                    f"Driving input size mismatch. Expected {self.input_size}, got {x.shape[-1]}"
                )
            projected = projected + F.linear(x, self.weight_input)

        if self.bias_h is not None:
            projected = projected + self.bias_h

        return projected

    def step(
        self,
        projected_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent-only single-step update for the projected fast path.

        ``addmm`` fuses the recurrent matmul with the precomputed input
        projection into one kernel; with activation and leak that is three
        kernel launches per timestep instead of six.

        Parameters
        ----------
        projected_t : torch.Tensor
            Slice of :meth:`project_inputs` output, shape
            ``(batch, reservoir_size)``.
        state : torch.Tensor
            Current hidden state, shape ``(batch, reservoir_size)``.

        Returns
        -------
        output : torch.Tensor
            Next hidden state.
        new_state : torch.Tensor
            Same tensor as output.
        """
        if self._sparse_active:
            # Sparse CSR path: the fused ``addmm`` has no CSR analogue, so add
            # the precomputed projection to the sparse spmv result.  Identical
            # math to the dense branch (``projected_t + state @ weight_hh.T``).
            pre_activation = projected_t + self._recurrent(state)
        else:
            pre_activation = torch.addmm(projected_t, state, self.weight_hh.t())
        new_state = self.activation_fn(pre_activation)
        new_state = self._apply_noise(new_state)
        new_state = self._apply_leak(state, new_state)
        return new_state, new_state

    def _apply_leak(self, state: torch.Tensor, new_state: torch.Tensor) -> torch.Tensor:
        """Blend the previous and freshly-computed state by the leak rate.

        Realises the leaky-integrator mix
        ``h_t = (1 - α) h_{t-1} + α · f(·)`` shared by :meth:`forward` and the
        :meth:`step` fast path, so the two never drift.

        Two paths, chosen by :attr:`_is_vector_leak`:

        * **Scalar** — the historical fast path.  ``leak_rate`` is validated to
          ``(0, 1]`` at construction, so ``!= 1.0`` applies the leaky mix only
          when there is actual leaking (and ``1.0`` returns ``new_state``
          untouched, byte-identical to the pre-vector cell).
        * **Per-neuron vector** — ``leak`` is a ``(reservoir_size,)`` buffer.
          ``leak != 1.0`` would be a *tensor* (unusable as a Python branch), so
          the mix is always taken; ``torch.lerp`` broadcasts the leak vector
          over the ``(batch, reservoir_size)`` state, giving each unit its own
          timescale.

        Parameters
        ----------
        state : torch.Tensor
            Previous hidden state, shape ``(batch, reservoir_size)``.
        new_state : torch.Tensor
            Freshly computed (post-activation, post-noise) state, same shape.

        Returns
        -------
        torch.Tensor
            The leaky-blended state.
        """
        if self._is_vector_leak:
            leak_vec = self._leak_rate_vec
            assert leak_vec is not None  # invariant when the vector flag is set
            return torch.lerp(state, new_state, leak_vec)
        if self._leak_rate != 1.0:
            return torch.lerp(state, new_state, self._leak_rate)
        return new_state

    # ------------------------------------------------------------------
    # Recurrent term (dense addmm / sparse-CSR spmv)
    # ------------------------------------------------------------------

    def _recurrent(self, state: torch.Tensor) -> torch.Tensor:
        r"""Compute the recurrent contribution ``state @ weight_hh.T``.

        The single place the recurrent matmul is expressed, so the dense and
        sparse paths never drift.  Two branches, chosen by
        :attr:`_sparse_active`:

        * **Dense** — ``F.linear(state, weight_hh)`` = ``state @ weight_hh.T``,
          the historical path, byte-for-byte unchanged.
        * **Sparse CSR** — a compressed-row spmv on the derived
          :attr:`_weight_hh_csr` operator.  ``torch.sparse.mm`` computes
          ``W @ state.T``; transposing both sides gives
          ``state @ W.T``, matching the dense result to floating-point
          round-off (the same nonzeros, just skipping the structural zeros).

        Parameters
        ----------
        state : torch.Tensor
            Current hidden state, shape ``(batch, reservoir_size)``.

        Returns
        -------
        torch.Tensor
            The recurrent contribution, shape ``(batch, reservoir_size)``.
        """
        if self._sparse_active:
            csr = self._weight_hh_csr
            assert csr is not None  # invariant when the sparse flag is set
            # ``state @ W.T == (W @ state.T).T``; ``.contiguous()`` on the
            # feedback so the spmv gets a well-formed dense RHS.
            recurrent: torch.Tensor = torch.sparse.mm(csr, state.t().contiguous())
            return recurrent.t()
        return F.linear(state, self.weight_hh)

    # ------------------------------------------------------------------
    # Sparse-CSR recurrent operator (frozen-reservoir opt-in)
    # ------------------------------------------------------------------

    @property
    def sparse(self) -> bool:
        """bool : Whether the CSR sparse recurrent-matmul path is active.

        Reflects the *resolved* decision (after the ``"auto"`` density gate and
        the frozen-reservoir requirement), not the raw construction request.
        ``False`` for a dense cell or when ``sparse="auto"`` fell back to dense.
        """
        return self._sparse_active

    def _recurrent_density(self) -> float:
        """Fraction of nonzero entries in the dense recurrent matrix."""
        w = self.weight_hh.data
        total = w.numel()
        if total == 0:
            return 0.0
        return float((w != 0).sum().item()) / total

    def _auto_min_size(self) -> int:
        """Device-dependent size gate for the ``"auto"`` sparse decision.

        The dense-``addmm``/CSR-spmv crossover moves with the device: a GPU's
        dense BLAS is fast enough that the reservoir must be far larger before a
        CSR spmv pays for its per-nonzero overhead.  Returns the higher CUDA
        floor (:data:`_SPARSE_AUTO_MIN_SIZE_CUDA`) when the recurrent weights live
        on a CUDA device, and the CPU floor (:data:`_SPARSE_AUTO_MIN_SIZE_CPU`)
        otherwise.  Consulting the *weight's* device (not a construction-time
        argument) means a cell moved to GPU via ``.cuda()`` before the ``"auto"``
        decision is re-taken — e.g. on ``load_state_dict`` — sees the right gate.
        """
        if self.weight_hh.device.type == "cuda":
            return _SPARSE_AUTO_MIN_SIZE_CUDA
        return _SPARSE_AUTO_MIN_SIZE_CPU

    def _should_use_sparse(self) -> bool:
        """Resolve the ``sparse`` request against the size/density ``auto`` gates.

        ``True`` / ``False`` are honoured verbatim (explicit opt-in / opt-out);
        ``"auto"`` enables the CSR path only when the reservoir is large enough
        (``reservoir_size >= self._auto_min_size()`` — a **device-dependent**
        floor, higher on CUDA where the dense BLAS is faster) *and* the recurrent
        matrix is sparse enough (density ``<= _SPARSE_AUTO_MAX_DENSITY``) for a
        CSR spmv to beat the dense ``addmm`` on that device.  A ``trainable``
        reservoir is always dense (the constructor rejects ``sparse=True`` +
        ``trainable``; ``"auto"`` and a live re-freeze route here).
        """
        if self._trainable:
            return False
        if self._sparse_requested is True:
            return True
        if self._sparse_requested == "auto":
            return (
                self.reservoir_size >= self._auto_min_size()
                and self._recurrent_density() <= _SPARSE_AUTO_MAX_DENSITY
            )
        return False

    def _maybe_build_sparse_weight(self) -> None:
        """(Re)materialise the derived CSR operator from the dense ``weight_hh``.

        Resolves :meth:`_should_use_sparse` and, when sparse is active, builds
        the non-persistent :attr:`_weight_hh_csr` buffer from the current dense
        ``weight_hh`` (matching its dtype/device).  When sparse is inactive the
        buffer is cleared.  Idempotent and safe to call after any event that
        changes the dense weights or moves them across devices/dtypes
        (construction, ``load_state_dict``, ``.to()`` — see :meth:`_apply`).
        """
        self._sparse_active = self._should_use_sparse()
        if self._sparse_active:
            self._weight_hh_csr = self.weight_hh.data.to_sparse_csr()
        else:
            self._weight_hh_csr = None

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "ESNCell":
        """Rebuild the derived CSR operator after a device/dtype move.

        ``nn.Module._apply`` (driven by ``.to()`` / ``.cuda()`` / ``.double()``)
        moves the dense ``weight_hh`` parameter, but the derived
        :attr:`_weight_hh_csr` buffer is a *separate* CSR tensor that must be
        rebuilt from the moved dense weights so the sparse spmv runs on the right
        device/dtype.

        A device move can also *change the ``"auto"`` decision*: the size gate is
        device-dependent (higher on CUDA), so a cell that ``auto`` sparsified on
        CPU may need to revert to dense once moved to GPU in the
        ``1000 <= n < 5000`` regime (and vice versa).  We therefore re-resolve the
        whole decision via :meth:`_maybe_build_sparse_weight` for an ``"auto"``
        request, and merely rebuild the CSR (an unconditional, device-independent
        decision) for an explicit ``sparse=True``.
        """
        super()._apply(fn, recurse=recurse)
        if getattr(self, "_sparse_requested", False) == "auto":
            # Re-decide against the *new* device's gate, rebuilding/dropping CSR.
            self._maybe_build_sparse_weight()
        elif getattr(self, "_sparse_active", False):
            self._weight_hh_csr = self.weight_hh.data.to_sparse_csr()
        return self

    def _load_from_state_dict(  # type: ignore[override]
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Rebuild the derived CSR operator after loading dense ``weight_hh``.

        Extends the base leak-buffer reconciliation (see the docstring on the
        override just below the leak setters) so that, after the dense
        ``weight_hh`` parameter has been copied in from the checkpoint, the
        derived :attr:`_weight_hh_csr` operator is re-materialised to match.  The
        CSR buffer is non-persistent (never in the checkpoint), so this is the
        only place a load refreshes it.
        """
        self._reconcile_leak_buffer(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        # The dense ``weight_hh`` is now the checkpoint's; refresh the CSR view.
        if hasattr(self, "_sparse_requested"):
            self._maybe_build_sparse_weight()

    # ------------------------------------------------------------------
    # State noise (train-mode regularizer)
    # ------------------------------------------------------------------

    def _noise_generator(self, device: torch.device) -> torch.Generator:
        """Return the cached noise generator for ``device`` (lazily built).

        Each device gets its own :class:`torch.Generator` so the noise can be
        drawn directly on the state's device.  The generators are seeded from
        ``self.seed`` but on a stream *offset* from the weight-init draws, so
        the perturbations are reproducible without disturbing weight
        reproducibility.

        Parameters
        ----------
        device : torch.device
            Device on which the noise tensor will be drawn.

        Returns
        -------
        torch.Generator
            A generator placed on ``device``, deterministic in ``self.seed``.
        """
        generator = self._noise_generators.get(device)
        if generator is None:
            # Offset the integer seed so the noise stream is independent of the
            # weight-init stream; ``None`` keeps it tied to torch's global RNG
            # (so ``torch.manual_seed`` still propagates).  Uses the same
            # golden-ratio salt helper as the input weights (see ``_salt_seed``);
            # the noise generator is a *torch* stream while the input weights are
            # numpy-backed, so sharing the offset does not couple their draws.
            noise_seed = _salt_seed(self._seed_int, _NOISE_SEED_OFFSET)
            generator = create_torch_generator(noise_seed, device=device)
            self._noise_generators[device] = generator
        return generator

    def _apply_noise(self, state: torch.Tensor) -> torch.Tensor:
        """Add Gaussian state noise in training mode (no-op otherwise).

        Mirrors the dropout gating convention: noise is injected only when
        ``self.training`` is ``True`` and ``self.noise > 0``; under
        :meth:`~torch.nn.Module.eval` (or with ``noise == 0.0``) the state is
        returned unchanged, so eval-mode and noiseless outputs stay
        bit-identical to the legacy cell.

        Parameters
        ----------
        state : torch.Tensor
            Post-activation state of shape ``(batch, reservoir_size)``.

        Returns
        -------
        torch.Tensor
            ``state`` perturbed by ``noise * N(0, 1)`` in training mode,
            otherwise ``state`` itself.
        """
        if not self.training or self.noise == 0.0:
            return state
        generator = self._noise_generator(state.device)
        epsilon = torch.randn(
            state.shape,
            generator=generator,
            device=state.device,
            dtype=state.dtype,
        )
        return state + self.noise * epsilon

    def __getstate__(self) -> dict:
        """Drop the transient per-device noise generators before pickling.

        :class:`torch.Generator` objects carry live device/RNG state that does
        not round-trip cleanly across devices; the cache is rebuilt lazily (and
        deterministically from ``seed``) on first use after unpickling.
        """
        state = self.__dict__.copy()
        state["_noise_generators"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state, re-creating the empty noise-generator cache."""
        state.setdefault("_noise_generators", {})
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Weight initialization (verbatim from legacy ReservoirLayer)
    # ------------------------------------------------------------------

    def _get_activation(self, activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get activation function by name."""
        activations: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "identity": _identity,
            "sigmoid": torch.sigmoid,
        }

        if activation not in activations:
            raise ValueError(
                f"Unknown activation '{activation}'. Supported: {list(activations.keys())}"
            )

        return activations[activation]

    def _initialize_weights(self) -> None:
        """Initialize all weight matrices.

        A single torch ``Generator`` (built from ``self.seed``) drives every
        default ``nn.init`` draw — feedback, input, recurrent, then bias, in
        that fixed order — so a given ``seed`` deterministically reproduces all
        parameters.  Named/object initializers and topologies seed themselves
        from the integer seed instead (see ``_seed_int``).
        """
        # One generator for all torch default draws; seeded from self.seed so
        # the parameters are a pure function of the seed and independent of the
        # global RNG state.  When seed is None it is derived from the global RNG
        # so torch.manual_seed still propagates.
        generator = create_torch_generator(self.seed)

        self._initialize_feedback_weights(generator)

        if self.input_size is not None:
            self._initialize_input_weights(generator)
        else:
            self.register_parameter("weight_input", None)

        self._initialize_recurrent_weights(generator)

        if self._bias:
            # Random bias drawn like the feedback/input weights.  A
            # zero-initialized frozen bias would be a no-op and leave the
            # tanh dynamics oddly symmetric (f(-x) = -f(x)).
            self.bias_h = nn.Parameter(torch.empty(self.reservoir_size))
            if self.bias_scaling != 0.0:
                nn.init.uniform_(
                    self.bias_h, -self.bias_scaling, self.bias_scaling, generator=generator
                )
            else:
                nn.init.zeros_(self.bias_h)
        else:
            self.register_parameter("bias_h", None)

    def _initialize_feedback_weights(self, generator: torch.Generator) -> None:
        """Initialize feedback weight matrix."""
        self.weight_feedback = nn.Parameter(torch.empty(self.reservoir_size, self.feedback_size))

        resolved = resolve_initializer(self.feedback_initializer, seed=self._seed_int)
        if resolved is not None:
            resolved.initialize(self.weight_feedback)
        else:
            nn.init.uniform_(self.weight_feedback, -1, 1, generator=generator)

    def _initialize_input_weights(self, generator: torch.Generator) -> None:
        """Initialize driving input weight matrix.

        The input initializer is seeded from a salted copy of ``_seed_int`` (see
        :data:`_INPUT_SEED_OFFSET`) rather than the raw feedback seed.  Feeding
        the same named initializer the identical integer for both roles produced
        ``weight_input == weight_feedback``, silently collapsing feedback and
        driving input into a single projection.  The salt only alters the
        layer-level fallback seed forwarded to ``resolve_initializer``; an
        explicit seed pinned inside a tuple/object spec (e.g.
        ``('random', {'seed': 7})``) still wins via ``resolve_initializer``'s
        ``_inject_seed`` (spec seed beats forwarded seed).
        """
        assert self.input_size is not None
        self.weight_input = nn.Parameter(torch.empty(self.reservoir_size, self.input_size))

        input_seed = _salt_seed(self._seed_int, _INPUT_SEED_OFFSET)
        resolved = resolve_initializer(self.input_initializer, seed=input_seed)
        if resolved is not None:
            resolved.initialize(self.weight_input)
        else:
            nn.init.uniform_(self.weight_input, -1, 1, generator=generator)

    def _initialize_recurrent_weights(self, generator: torch.Generator) -> None:
        """Initialize recurrent weight matrix from topology or random."""
        self.weight_hh = nn.Parameter(torch.empty(self.reservoir_size, self.reservoir_size))

        resolved = resolve_topology(self.topology, seed=self._seed_int)
        if resolved is not None:
            resolved.initialize(self.weight_hh, spectral_radius=self.spectral_radius)
        else:
            nn.init.uniform_(self.weight_hh, -1.0, 1.0, generator=generator)
            if self.spectral_radius is not None:
                self._scale_spectral_radius()

    def _scale_spectral_radius(self) -> None:
        """Scale recurrent weights to the target spectral radius.

        Delegates to the single shared
        :func:`resdag.init.topology.scale_to_spectral_radius` implementation,
        so the cell and the topology base never drift apart.  That routine
        picks the cheapest accurate estimator (power iteration for dense
        matrices, scipy sparse ``eigs`` for sparse ones, a tiny-N dense
        ``eigvals`` fallback) instead of always paying for a full dense
        eigendecomposition.
        """
        assert self.spectral_radius is not None
        with torch.no_grad():
            scaled = scale_to_spectral_radius(self.weight_hh.data, self.spectral_radius)
            self.weight_hh.data.copy_(scaled)

    @property
    def spectral_radius_achieved(self) -> float:
        """float : Realized largest absolute eigenvalue of ``weight_hh``.

        Returns the spectral radius actually present in the recurrent matrix
        after initialization/scaling, as opposed to the *requested*
        :attr:`spectral_radius` target.  Computed lazily via the shared
        :func:`resdag.init.topology.estimate_spectral_radius` (power iteration /
        sparse ``eigs`` / tiny-N dense fallback), so it stays cheap even for
        large reservoirs and is GPU-resident for dense matrices.

        For a freshly built cell with a non-``None`` ``spectral_radius`` this
        sits within the estimator's tolerance of the target; it differs once
        the recurrent weights are trained or otherwise modified.
        """
        with torch.no_grad():
            return estimate_spectral_radius(self.weight_hh.data)

    def __repr__(self) -> str:
        """Return string representation."""
        input_str = f", input_size={self.input_size}" if self.input_size is not None else ""
        noise_str = f", noise={self.noise}" if self.noise != 0.0 else ""
        sparse_str = ", sparse=True" if self._sparse_active else ""
        return (
            f"ESNCell("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"spectral_radius={self.spectral_radius}"
            f"{noise_str}"
            f"{sparse_str}"
            f")"
        )
