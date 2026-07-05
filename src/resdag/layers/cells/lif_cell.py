r"""
Leaky Integrate-and-Fire (LIF) Cell
===================================

This module provides the concrete leaky integrate-and-fire reservoir cell — the
spiking neuron of a Liquid State Machine (LSM; Maass, Natschläger & Markram,
*Neural Computation* 14(11), 2002):

- :class:`LIFCell` — a :class:`~resdag.layers.cells.spiking_cell.SpikingReservoirCell`
  whose sub-threshold dynamics are a discretised leaky integrator; owns the
  frozen recurrent weight ``W_rec``, the feedback / input weights and bias.

Two variants share one class, selected by ``synaptic_current``:

* **First-order LIF** (default): a single membrane state ``V``,

  .. math::

      V_{t+1} = \beta\, V_t + (1 - \beta)\,\big(W_\text{rec} s_t + W_\text{fb} x_{fb}
                + W_\text{in} u + b\big), \qquad \beta = e^{-\Delta t / \tau_\text{mem}},

  with recurrent spike feedback :math:`s_t` (the previous step's spikes).

* **Current-based LIF (CuBa-LIF)** (``synaptic_current=True``): a second-order
  model with an extra synaptic-current state ``I``,

  .. math::

      I_{t+1} &= \alpha\, I_t + \big(W_\text{rec} s_t + W_\text{fb} x_{fb}
                 + W_\text{in} u + b\big), \qquad \alpha = e^{-\Delta t / \tau_\text{syn}}, \\
      V_{t+1} &= \beta\, V_t + (1 - \beta)\, I_{t+1}.

Continuous inputs enter as **injected current** (the drive term above) — the
standard analog-input convention for an LSM driven by a real-valued signal
(rather than encoding the input as a spike train first).  This keeps the cell a
drop-in reservoir for resdag's real-valued time-series pipeline; a spike-encoding
front-end can always be prepended by the user.

State packing
-------------
The packed state carries the previous spikes ``s`` as an extra neuron tensor so
:meth:`neuron_update` can add the recurrent term :math:`W_\text{rec} s` without
the base having to thread spikes separately.  The layouts are

* first-order LIF: ``[V | s | r]`` (:attr:`n_state_tensors` = 3),
* CuBa-LIF:        ``[I | V | s | r]`` (:attr:`n_state_tensors` = 4),

where ``r`` is the base-owned readout trace.  The membrane potential ``V`` is
always the tensor immediately before ``s`` so the ``output="voltage"`` mode and
the base's ``V``-extraction stay correct.

See Also
--------
resdag.layers.cells.spiking_cell.SpikingReservoirCell : Shared spiking base.
resdag.layers.cells.esn_cell.ESNCell : Source of the weight / topology machinery.
resdag.layers.reservoirs.lif.LIFReservoirLayer : Layer that sequences this cell.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from resdag.init.topology import estimate_spectral_radius, scale_to_spectral_radius
from resdag.init.utils import InitializerSpec, TopologySpec, resolve_initializer, resolve_topology
from resdag.utils.general import SeedLike, coerce_seed_to_int, create_torch_generator

from .esn_cell import _INPUT_SEED_OFFSET, _salt_seed
from .spiking_cell import SpikingReservoirCell


class LIFCell(SpikingReservoirCell):
    r"""
    Single-timestep leaky integrate-and-fire (LIF) reservoir cell.

    Owns the frozen recurrent weight ``W_rec`` (topology-initialised and
    spectral-radius scaled like an ESN), the feedback / driving-input weights and
    the bias.  The sub-threshold dynamics are a discretised leaky integrator; the
    surrogate spike, reset, and readout trace are inherited from
    :class:`~resdag.layers.cells.spiking_cell.SpikingReservoirCell`.

    Parameters
    ----------
    reservoir_size : int
        Number of spiking units.
    feedback_size : int
        Dimension of the feedback signal.  Required.
    input_size : int, optional
        Dimension of an optional driving input.  ``None`` (or ``0``) builds a
        feedback-only cell.
    tau_mem : float, default=20.0
        Membrane time constant :math:`\tau_\text{mem} > 0` (same time units as
        ``dt``).  Sets the membrane decay :math:`\beta = e^{-\Delta t/\tau_\text{mem}}`.
    tau_syn : float, default=10.0
        Synaptic time constant :math:`\tau_\text{syn} > 0`.  Only used when
        ``synaptic_current=True`` (CuBa-LIF); sets :math:`\alpha = e^{-\Delta t/\tau_\text{syn}}`.
    dt : float, default=1.0
        Integration timestep :math:`\Delta t > 0`.
    synaptic_current : bool, default=False
        If ``True``, use the second-order CuBa-LIF model with an explicit
        synaptic-current state ``I``; otherwise the first-order LIF.
    v_threshold : float, default=1.0
        Firing threshold (see :class:`SpikingReservoirCell`).
    reset : {'subtract', 'zero'}, default='subtract'
        Membrane reset on a spike (see :class:`SpikingReservoirCell`).
    surrogate_beta : float, default=10.0
        Fast-sigmoid surrogate slope for the straight-through spike gradient.
    trace_decay : float, default=0.9
        Decay of the low-pass readout trace (the cell's real-valued output).
    output : {'trace', 'voltage', 'spikes'}, default='trace'
        Readable-output mode.  ``'trace'`` (default) is the ridge-regressable
        filtered spike response — the LSM-correct choice.
    spectral_radius : float, optional
        Target spectral radius for ``W_rec``.  Scales the recurrent spike-coupling
        strength (how strongly one neuron's spikes drive its neighbours).  ``None``
        leaves ``W_rec`` unscaled.
    bias : bool, default=True
        Whether to include a bias current ``b``.
    bias_scaling : float, default=1.0
        Scale of the random bias (``uniform(-bias_scaling, bias_scaling)``).
    trainable : bool, default=False
        If ``True``, all weights participate in backpropagation (through the
        surrogate spike gradient).  Frozen LSMs use ``False``.
    feedback_initializer, input_initializer : spec, optional
        Initializers for the feedback / driving-input weights.  Same accepted
        forms as :class:`~resdag.layers.cells.esn_cell.ESNCell`.
    topology : spec, optional
        Structure of the recurrent weight ``W_rec``.  Same forms as ``ESNCell``.
    seed : int or torch.Generator, optional
        Reproducibility seed fixing every weight and bias.

    Attributes
    ----------
    beta : float
        Membrane decay :math:`\beta = e^{-\Delta t/\tau_\text{mem}}` (derived).
    alpha : float
        Synaptic-current decay :math:`\alpha = e^{-\Delta t/\tau_\text{syn}}`
        (derived; only meaningful when ``synaptic_current=True``).
    weight_rec : torch.nn.Parameter
        Recurrent spike-coupling weight ``W_rec``, shape
        ``(reservoir_size, reservoir_size)``.
    weight_feedback : torch.nn.Parameter
        Feedback weight ``W_fb``, shape ``(reservoir_size, feedback_size)``.
    weight_input : torch.nn.Parameter or None
        Driving-input weight ``W_in``, shape ``(reservoir_size, input_size)``, or
        ``None``.
    bias_h : torch.nn.Parameter or None
        Bias current ``b``, shape ``(reservoir_size,)``, or ``None``.

    Examples
    --------
    >>> import torch
    >>> cell = LIFCell(reservoir_size=64, feedback_size=3, tau_mem=20.0, seed=0)
    >>> state = cell.init_state(2, torch.device("cpu"), torch.float32)
    >>> out, new_state = cell([torch.randn(2, 3)], state)
    >>> out.shape
    torch.Size([2, 64])

    See Also
    --------
    resdag.layers.cells.spiking_cell.SpikingReservoirCell : Shared spiking base.
    resdag.layers.reservoirs.lif.LIFReservoirLayer : Layer that sequences this cell.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        dt: float = 1.0,
        synaptic_current: bool = False,
        v_threshold: float = 1.0,
        reset: str = "subtract",
        surrogate_beta: float = 10.0,
        trace_decay: float = 0.9,
        output: str = "trace",
        spectral_radius: float | None = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        trainable: bool = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
        seed: SeedLike = None,
    ) -> None:
        # First-order LIF packs [V | s | r] (3 tensors); CuBa-LIF packs
        # [I | V | s | r] (4).  Set the count BEFORE super().__init__ so
        # ``state_size`` is correct the moment the base needs it.
        self.synaptic_current = bool(synaptic_current)
        self.n_state_tensors = 4 if self.synaptic_current else 3

        super().__init__(
            reservoir_size=reservoir_size,
            v_threshold=v_threshold,
            reset=reset,
            surrogate_beta=surrogate_beta,
            trace_decay=trace_decay,
            output=output,
        )

        if tau_mem <= 0.0:
            raise ValueError(f"tau_mem must be positive, got {tau_mem}")
        if tau_syn <= 0.0:
            raise ValueError(f"tau_syn must be positive, got {tau_syn}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.feedback_size = feedback_size
        # Treat input_size == 0 the same as None (no driving-input weights),
        # matching ESNCell and the premade factories.
        self.input_size = input_size if input_size else None
        self.tau_mem = float(tau_mem)
        self.tau_syn = float(tau_syn)
        self.dt = float(dt)
        # Membrane / synaptic decays derived from the time constants.  Exact
        # exponential integration of the leaky ODE over one dt (zero-order hold).
        self.beta = math.exp(-self.dt / self.tau_mem)
        self.alpha = math.exp(-self.dt / self.tau_syn)

        self.topology = topology
        self.spectral_radius = spectral_radius
        self.feedback_initializer = feedback_initializer
        self.input_initializer = input_initializer
        self._bias = bias
        self.bias_scaling = bias_scaling

        self.seed = seed
        self._seed_int = coerce_seed_to_int(seed)
        self._trainable = trainable

        self._initialize_weights()

        # Apply the requested freeze through the base property so construction and
        # post-init ``cell.trainable = ...`` share one code path.
        self.trainable = trainable

    # ------------------------------------------------------------------
    # ReservoirCell interface
    # ------------------------------------------------------------------

    @property
    def trainable(self) -> bool:
        """
        bool : Whether the reservoir weights participate in backpropagation.

        A *live* switch: assigning to it flips ``requires_grad_`` on **every**
        cell parameter (``weight_rec``, ``weight_feedback``, ``weight_input``,
        ``bias_h``).  With the surrogate-gradient spike this genuinely unfreezes
        the LIF reservoir for a BPTT rollout.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        self._trainable = bool(value)
        for p in self.parameters():
            p.requires_grad_(self._trainable)

    def project_inputs(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Precompute the analog input drive ``W_fb x_fb + W_in u + b``.

        This is the *input-dependent* part of the neuron drive — the recurrent
        spike term ``W_rec s`` is added inside :meth:`neuron_update` from the
        previous spikes carried in state.  Works on full
        ``(batch, timesteps, features)`` sequences (the layer fast path) and on
        single-step ``(batch, features)`` slices.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            ``inputs[0]`` feedback; ``inputs[1]`` (optional) driving input.

        Returns
        -------
        torch.Tensor
            Input drive with the inputs' leading dims and trailing dimension
            ``reservoir_size``.

        Raises
        ------
        ValueError
            On a feedback / driving-input feature-dimension mismatch, or if a
            driving input is supplied to a cell built without ``input_size``.
        """
        fb = inputs[0]
        if fb.shape[-1] != self.feedback_size:
            raise ValueError(
                f"Feedback size mismatch. Expected {self.feedback_size}, got {fb.shape[-1]}"
            )
        drive = F.linear(fb, self.weight_feedback)

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
            drive = drive + F.linear(x, self.weight_input)

        if self.bias_h is not None:
            drive = drive + self.bias_h

        return drive

    def neuron_update(
        self,
        neuron_state: tuple[torch.Tensor, ...],
        projected_input: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        r"""Advance the LIF (or CuBa-LIF) membrane one step and emit spikes.

        The recurrent spike term :math:`W_\text{rec} s_t` is added here (from the
        previous spikes ``s`` carried in ``neuron_state``), completing the drive
        ``projected_input + W_rec s``.  Branch-free apart from the static
        ``synaptic_current`` / reset selection, so it traces ``fullgraph=True``.

        Parameters
        ----------
        neuron_state : tuple of torch.Tensor
            ``(V, s)`` for first-order LIF or ``(I, V, s)`` for CuBa-LIF, each of
            shape ``(batch, reservoir_size)``.  ``s`` is the previous step's
            spikes (recurrent input); ``I`` / ``V`` the current / membrane states.
        projected_input : torch.Tensor
            Precomputed analog drive ``W_fb x_fb + W_in u + b``, shape
            ``(batch, reservoir_size)``.

        Returns
        -------
        new_neuron_state : tuple of torch.Tensor
            ``(V_new, s_new)`` or ``(I_new, V_new, s_new)`` in packing order.
        spikes : torch.Tensor
            The spikes emitted this step (``== s_new``), carrying the surrogate
            gradient.
        """
        # Complete the drive with the recurrent spike-coupling term.
        recurrent = F.linear(neuron_state[-1], self.weight_rec)
        drive = projected_input + recurrent

        if self.synaptic_current:
            i_prev, v_prev = neuron_state[0], neuron_state[1]
            i_new = self.alpha * i_prev + drive
            v_pre = self.beta * v_prev + (1.0 - self.beta) * i_new
            spikes = self.spike(v_pre)
            v_new = self.apply_reset(v_pre, spikes)
            return (i_new, v_new, spikes), spikes

        v_prev = neuron_state[0]
        v_pre = self.beta * v_prev + (1.0 - self.beta) * drive
        spikes = self.spike(v_pre)
        v_new = self.apply_reset(v_pre, spikes)
        return (v_new, spikes), spikes

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _initialize_weights(self) -> None:
        """Initialize the recurrent, feedback, input weights and bias.

        A single torch ``Generator`` (seeded from ``self.seed``) drives every
        default draw in a fixed order — feedback, input, recurrent, then bias —
        so a given ``seed`` deterministically reproduces the whole cell.
        """
        generator = create_torch_generator(self.seed)

        # Feedback weight.
        self.weight_feedback = nn.Parameter(torch.empty(self.reservoir_size, self.feedback_size))
        resolved_fb = resolve_initializer(self.feedback_initializer, seed=self._seed_int)
        if resolved_fb is not None:
            resolved_fb.initialize(self.weight_feedback)
        else:
            nn.init.uniform_(self.weight_feedback, -1, 1, generator=generator)

        # Driving-input weight (salted seed so it never coincides with feedback).
        if self.input_size is not None:
            self.weight_input = nn.Parameter(torch.empty(self.reservoir_size, self.input_size))
            input_seed = _salt_seed(self._seed_int, _INPUT_SEED_OFFSET)
            resolved_in = resolve_initializer(self.input_initializer, seed=input_seed)
            if resolved_in is not None:
                resolved_in.initialize(self.weight_input)
            else:
                nn.init.uniform_(self.weight_input, -1, 1, generator=generator)
        else:
            self.register_parameter("weight_input", None)

        # Recurrent spike-coupling weight W_rec: topology-structured + spectral
        # scaled, exactly like the ESN recurrent matrix.
        self.weight_rec = nn.Parameter(torch.empty(self.reservoir_size, self.reservoir_size))
        resolved_topo = resolve_topology(self.topology, seed=self._seed_int)
        if resolved_topo is not None:
            resolved_topo.initialize(self.weight_rec, spectral_radius=self.spectral_radius)
        else:
            nn.init.uniform_(self.weight_rec, -1.0, 1.0, generator=generator)
            if self.spectral_radius is not None:
                self._scale_spectral_radius()

        # Bias current.
        if self._bias:
            self.bias_h = nn.Parameter(torch.empty(self.reservoir_size))
            if self.bias_scaling != 0.0:
                nn.init.uniform_(
                    self.bias_h, -self.bias_scaling, self.bias_scaling, generator=generator
                )
            else:
                nn.init.zeros_(self.bias_h)
        else:
            self.register_parameter("bias_h", None)

    def _scale_spectral_radius(self) -> None:
        """Scale the recurrent weight ``W_rec`` to the target spectral radius."""
        assert self.spectral_radius is not None
        with torch.no_grad():
            scaled = scale_to_spectral_radius(self.weight_rec.data, self.spectral_radius)
            self.weight_rec.data.copy_(scaled)

    @property
    def spectral_radius_achieved(self) -> float:
        """float : Realized largest absolute eigenvalue of ``weight_rec``."""
        with torch.no_grad():
            return estimate_spectral_radius(self.weight_rec.data)

    def __repr__(self) -> str:
        """Return string representation."""
        input_str = f", input_size={self.input_size}" if self.input_size is not None else ""
        model = "CuBa-LIF" if self.synaptic_current else "LIF"
        return (
            f"LIFCell("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"model={model}, "
            f"tau_mem={self.tau_mem}, "
            f"tau_syn={self.tau_syn if self.synaptic_current else None}, "
            f"dt={self.dt}, "
            f"v_threshold={self.v_threshold}, "
            f"reset={self.reset}, "
            f"trainable={self._trainable}"
            f")"
        )
