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

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from resdag.init.topology import estimate_spectral_radius, scale_to_spectral_radius
from resdag.init.utils import InitializerSpec, TopologySpec, resolve_initializer, resolve_topology
from resdag.utils.general import SeedLike, coerce_seed_to_int, create_torch_generator

from .base_cell import ReservoirCell


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
    leak_rate : float, default=1.0
        Leaky integration rate in [0, 1].  A value of 1.0 means no leaking
        (standard RNN update); smaller values create slower dynamics.
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
        leak_rate: float = 1.0,
        noise: float = 0.0,
        trainable: bool = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
        seed: SeedLike = None,
    ) -> None:
        super().__init__()

        if noise < 0.0:
            raise ValueError(f"noise must be non-negative, got {noise}")

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
        self.leak_rate = leak_rate
        self.noise = noise
        self.trainable = trainable
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

        # Freeze weights if not trainable
        if not self.trainable:
            for p in self.parameters():
                p.requires_grad_(False)

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
        recurrent_contrib = F.linear(state, self.weight_hh)
        new_state = self.activation_fn(projected + recurrent_contrib)
        new_state = self._apply_noise(new_state)

        if self.leak_rate < 1.0:
            new_state = torch.lerp(state, new_state, self.leak_rate)
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
        pre_activation = torch.addmm(projected_t, state, self.weight_hh.t())
        new_state = self.activation_fn(pre_activation)
        new_state = self._apply_noise(new_state)

        if self.leak_rate < 1.0:
            new_state = torch.lerp(state, new_state, self.leak_rate)
        return new_state, new_state

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
            # (so ``torch.manual_seed`` still propagates).
            base = self._seed_int
            noise_seed = None if base is None else (base + 0x9E3779B9) % (2**63 - 1)
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
        """Initialize driving input weight matrix."""
        assert self.input_size is not None
        self.weight_input = nn.Parameter(torch.empty(self.reservoir_size, self.input_size))

        resolved = resolve_initializer(self.input_initializer, seed=self._seed_int)
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
        return (
            f"ESNCell("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"spectral_radius={self.spectral_radius}"
            f"{noise_str}"
            f")"
        )
