"""
Gated Reservoir Cell
====================

This module provides a **gated** reservoir cell — a "reservoir GRU":  a
LiGRU-style gated recurrent update (Ravanelli et al. 2018) whose recurrent and
input gate weights are **frozen** random matrices (a reservoir), so the cell is
trained (if at all) only through its readout, exactly like an ESN.

- :class:`GatedReservoirCell` — single-timestep light-gated update; owns all
  weights, frozen by default.

The Light Gated Recurrent Unit (LiGRU) drops the reset gate of the classical
GRU and replaces the ``tanh`` candidate with a ReLU one, leaving a single
update gate:

.. math::

    z_t        &= \\sigma(u_t + W_{hz}\\,h_{t-1} + b_z), \\\\
    \\tilde{h}_t &= \\phi(u_t + W_{hh}\\,h_{t-1} + b_h), \\\\
    h_t        &= z_t \\odot h_{t-1} + (1 - z_t) \\odot \\tilde{h}_t,

where :math:`u_t = W_{fb}\\,x_{fb,t} + W_{in}\\,x_{in,t}` is a **single shared
input drive** (feedback plus optional driving input), added to *both*
pre-activations — a reservoir simplification of the full LiGRU, which uses a
separate input projection per gate.  The gate and candidate keep *distinct*
recurrent weights (:math:`W_{hz} \\ne W_{hh}`), so the gating is not degenerate.
:math:`\\sigma` is the update-gate nonlinearity (sigmoid), :math:`\\phi` the
candidate nonlinearity (ReLU by default), and :math:`\\odot` elementwise product.

References
----------
The gate structure and retention update follow the
Light Gated Recurrent Unit of Ravanelli, Brakel, Omologo & Bengio,
"Light Gated Recurrent Units for Speech Recognition", IEEE TETCI 2018
(`arXiv:1803.10225 <https://arxiv.org/abs/1803.10225>`_).

The packed multi-tensor-state convention
-----------------------------------------
resdag's :class:`~resdag.layers.cells.base_cell.ReservoirCell` contract is a
**single 2-D state tensor** ``(batch, state_size)``.  Multi-tensor cells (a GRU
carries one hidden ``h``; a future LSTM-style cell would carry ``(h, c)``; a
continuous/spiking cell might carry ``(v, w)``) are expressed by *packing* every
state tensor side-by-side into that one 2-D tensor:

    ``state_size = n_state_tensors * hidden_size``

and splitting / concatenating inside the update:

    >>> h, = _split_state(state, hidden_size)          # single-state
    >>> # h, c = _split_state(state, hidden_size)       # 2-state
    >>> new_state = _pack_state(h_new)                  # single-state
    >>> # new_state = _pack_state(h_new, c_new)         # 2-state

``output_size = hidden_size`` (the readout reads ``h``).  The split/cat helpers
(:func:`_split_state`, :func:`_pack_state`) are written generically for any
``n_state_tensors`` so Phases 10/11 (continuous, spiking) reuse this module
verbatim.  For this LiGRU cell ``n_state_tensors == 1`` so the pack/split are
the identity, but they are unit-tested for the hypothetical 2-state layout to
prove the convention is ready.

See Also
--------
resdag.layers.cells.base_cell : Abstract base class (ReservoirCell).
resdag.layers.cells.esn_cell : Concrete ungated ESN cell (ESNCell).
resdag.init.topology : Topology / spectral-radius scaling for recurrent weights.
resdag.init.input_feedback : Input/feedback weight initialization.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from resdag.init.recurrent import select_recurrent_spec
from resdag.init.topology import estimate_spectral_radius, scale_to_spectral_radius
from resdag.init.utils import InitializerSpec, TopologySpec, resolve_initializer, resolve_topology
from resdag.utils.general import SeedLike, coerce_seed_to_int, create_torch_generator

from ._initialization import (
    initialize_vector,
    initialized_state,
    install_initial_state,
    restore_initial_state_support,
)
from .base_cell import ReservoirCell
from .esn_cell import _INPUT_SEED_OFFSET, _identity, _salt_seed

# Existing gate/input/bias/reset roles occupy 2C and 4C through 9C. Keep
# candidate recurrence on its own stream without changing those named roles.
_CANDIDATE_RECURRENT_SEED_OFFSET = 0x9E3779B9 * 10

# ----------------------------------------------------------------------------
# Packed multi-tensor-state helpers (the reusable convention)
# ----------------------------------------------------------------------------


def _split_state(state: torch.Tensor, hidden_size: int) -> tuple[torch.Tensor, ...]:
    """Split a packed 2-D state into its constituent per-tensor slices.

    Inverse of :func:`_pack_state`.  A state packed as
    ``(batch, n_state_tensors * hidden_size)`` is split along the last axis into
    ``n_state_tensors`` contiguous ``(batch, hidden_size)`` blocks.  For a
    single-state cell (``n_state_tensors == 1``) this returns a length-1 tuple
    whose only element *is* the input (an identity split), so single-state cells
    pay nothing.

    Parameters
    ----------
    state : torch.Tensor
        Packed state of shape ``(batch, n_state_tensors * hidden_size)``.
    hidden_size : int
        Width of each individual state tensor.

    Returns
    -------
    tuple of torch.Tensor
        ``n_state_tensors`` tensors, each of shape ``(batch, hidden_size)``, in
        packing order.

    Raises
    ------
    ValueError
        If ``state``'s last dimension is not a positive integer multiple of
        ``hidden_size``.
    """
    total = state.shape[-1]
    if hidden_size <= 0 or total % hidden_size != 0:
        raise ValueError(
            f"_split_state: packed state width {total} is not a positive multiple "
            f"of hidden_size {hidden_size}."
        )
    n = total // hidden_size
    if n == 1:
        # Identity fast path: a single-state cell's packed state *is* the tensor.
        return (state,)
    return tuple(state.split(hidden_size, dim=-1))


def _pack_state(*tensors: torch.Tensor) -> torch.Tensor:
    """Concatenate per-tensor state slices back into one packed 2-D state.

    Inverse of :func:`_split_state`.  A single tensor is returned unchanged
    (identity pack); two or more are concatenated along the last axis in the
    order given, which must match the order :func:`_split_state` unpacks.

    Parameters
    ----------
    *tensors : torch.Tensor
        One or more ``(batch, hidden_size)`` state slices in packing order.

    Returns
    -------
    torch.Tensor
        Packed state of shape ``(batch, n_state_tensors * hidden_size)``.

    Raises
    ------
    ValueError
        If no tensors are given.
    """
    if not tensors:
        raise ValueError("_pack_state: expected at least one state tensor.")
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=-1)


class GatedReservoirCell(ReservoirCell):
    """
    Single-timestep frozen light-gated (LiGRU) reservoir update — a reservoir GRU.

    Owns all weight matrices and biases; **frozen by default**
    (``trainable=False``), like a classical ESN, so only a downstream readout is
    fitted.  Sequence iteration is delegated to the enclosing
    :class:`~resdag.layers.reservoirs.gated.GatedReservoirLayer`.

    The update is a reservoir adaptation of the Light Gated Recurrent Unit
    (Ravanelli et al. 2018):

    .. math::

        z_t        &= \\sigma(W_{iz}\\,u_t + W_{hz}\\,h_{t-1} + b_z), \\\\
        \\tilde{h}_t &= \\phi(W_{ih}\\,u_t + W_{hh}\\,h_{t-1} + b_h), \\\\
        h_t        &= z_t \\odot h_{t-1} + (1 - z_t) \\odot \\tilde{h}_t,

    where :math:`u_t = W_{fb}\\,x_{fb,t} + W_{in}\\,x_{in,t}` is the *drive*
    (feedback plus optional driving input, projected exactly as in
    :class:`ESNCell`), :math:`\\sigma` is the sigmoid update-gate nonlinearity,
    and :math:`\\phi` is the candidate nonlinearity (``"relu"`` by default, the
    LiGRU choice).  ``z`` gates *retention*: ``z→1`` keeps ``h``, ``z→0`` writes
    the candidate.

    Stability of a frozen gated reservoir
    -------------------------------------
    A frozen random GRU can saturate — a random update-gate pre-activation
    centred at ``0`` gives ``z ≈ 0.5``, which over-writes the state every step
    and washes out memory (or, with an unfortunate draw, drives ``h`` to a
    constant).  Two knobs keep it well-behaved:

    * ``update_gate_bias`` (default ``2.0``) biases the update gate toward
      **retention**: ``b_z = update_gate_bias`` makes ``z`` start near
      ``σ(2) ≈ 0.88``, so a fresh reservoir *keeps* most of its state each step
      and the dynamics do not collapse.  This mirrors the classic "initialise
      the forget-gate bias positive" trick for trainable gated RNNs.
    * ``spectral_radius`` scales the recurrent **candidate** weight
      :math:`W_{hh}` (via the shared resdag machinery), controlling the echo of
      past state through the candidate path just as it does for an ESN.

    Even so, a *frozen* gated reservoir often underperforms a plain ESN; the
    gated cell is usually most useful with ``trainable=True`` (a BPTT rollout
    that updates the gates), which this cell fully supports via the base
    :attr:`trainable` property.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (hidden state dimension ``hidden_size``).
    feedback_size : int
        Dimension of the feedback signal.  Required.
    input_size : int, optional
        Dimension of an optional driving input.  If ``None`` (or ``0``), no
        driving-input weights are created.
    spectral_radius : float, optional
        Target spectral radius for the recurrent **candidate** weight
        :math:`W_{hh}`.  If ``None``, no scaling is applied.
    bias : bool, default=True
        Whether to include gate biases (``b_z``, ``b_h``).
    bias_scaling : float, default=1.0
        Scale of the random candidate bias ``b_h``: entries are drawn from
        ``uniform(-bias_scaling, bias_scaling)``.  The update-gate bias ``b_z``
        is set to the constant :attr:`update_gate_bias` (retention bias), *not*
        randomised.  Ignored when ``bias=False`` (both biases are then ``None``
        and the update gate starts centred at ``z ≈ 0.5``).
    update_gate_bias : float, default=2.0
        Constant added to the update-gate pre-activation (the value of ``b_z``
        when ``bias=True``).  A positive value biases ``z`` toward ``1``
        (retention).  ``2.0`` gives ``z ≈ 0.88`` at init.  Set ``0.0`` for an
        unbiased update gate.
    activation : {'relu', 'tanh', 'identity', 'sigmoid'}, default='relu'
        Candidate nonlinearity :math:`\\phi`.  ``"relu"`` is the LiGRU default;
        ``"tanh"`` recovers a classical-GRU-like bounded candidate.
    gate_activation : {'sigmoid', ...}, default='sigmoid'
        Update-gate nonlinearity :math:`\\sigma`.  ``"sigmoid"`` (the gate
        convention) keeps ``z`` in ``(0, 1)``; other names are accepted for
        experimentation but only ``"sigmoid"`` guarantees a valid gate.
    trainable : bool, default=False
        If ``True``, all weights participate in backpropagation.  Frozen ESN-style
        reservoirs use ``False``; gated reservoirs are often best ``True``.
    feedback_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Initializer for the feedback weight matrix (drive path).  Same accepted
        forms as :class:`ESNCell`.
    input_initializer : str, callable, tuple, or InputFeedbackInitializer, optional
        Initializer for the driving-input weight matrix.  Only used when
        ``input_size`` is given.
    topology : str, callable, tuple, or TopologyInitializer, optional
        Structure of the recurrent **candidate** weight :math:`W_{hh}`.  The
        recurrent gate weights ``W_hz`` are always dense-random (a graph
        topology would make the update gate structurally sparse, which is not
        the intent).
    seed : int or torch.Generator, optional
        Reproducibility seed for independent weight, bias and reset-state roles.
        Integers and a generator's ``initial_seed()`` do not advance the caller's
        RNG. ``None`` draws one master seed from Torch's CPU global RNG. Explicit
        initializer seeds and callback-owned generators retain their own policy.

    recurrent_initializer : spec, optional
        First-class alias for ``topology``. Supply one spelling only. Names,
        tuples, callables and registered initializer objects are accepted.
    bias_initializer : spec, optional
        Input/feedback initializer grammar applied to a ``(1, reservoir_size)``
        view of the bias. Requires ``bias=True``; overrides ``bias_scaling``.
    initial_state_initializer : spec, optional
        Initializes a persistent ``(1, cell.state_size)`` reset template once.
        Each batch example receives an independent copy at reset. Zero by default.
    device : torch.device or str, optional
        Initial parameter location. Metal construction initializes on CPU before
        transferring to support initialization operations across backends.
    dtype : torch.dtype, optional
        Floating precision used during initialization, not only after conversion.

    gate_recurrent_initializer : spec, optional
        Separate recurrent gate matrix. Defaults to small dense random weights.
    gate_spectral_radius : float, optional
        Outer radius target for the gate recurrence, respecting its scaling policy.
    gate_feedback_initializer, gate_input_initializer : spec, optional
        Separate gate projections. Omitted projections share candidate weights.
        A driver gate initializer requires ``input_size``.
    gate_bias_initializer : spec, optional
        Gate-bias initializer on ``(1, reservoir_size)``; overrides
        ``update_gate_bias``. Requires ``bias=True``.

    Attributes
    ----------
    weight_feedback : torch.nn.Parameter
        Feedback drive weight ``W_fb`` of shape ``(reservoir_size, feedback_size)``.
    weight_input : torch.nn.Parameter or None
        Driving-input drive weight ``W_in`` of shape
        ``(reservoir_size, input_size)``, or ``None``.
    weight_hh : torch.nn.Parameter
        Recurrent **candidate** weight :math:`W_{hh}` of shape
        ``(reservoir_size, reservoir_size)`` (spectral-radius scaled).
    weight_hz : torch.nn.Parameter
        Recurrent **update-gate** weight :math:`W_{hz}` of shape
        ``(reservoir_size, reservoir_size)``.
    bias_h : torch.nn.Parameter or None
        Candidate bias ``b_h``, shape ``(reservoir_size,)``, or ``None``.
    bias_z : torch.nn.Parameter or None
        Update-gate bias ``b_z`` (initialised to ``update_gate_bias``), shape
        ``(reservoir_size,)``, or ``None``.

    See Also
    --------
    resdag.layers.reservoirs.gated.GatedReservoirLayer : Layer that sequences this cell.
    resdag.layers.cells.esn_cell.ESNCell : Ungated ESN cell.
    resdag.layers.cells.base_cell.ReservoirCell : Abstract cell interface.
    """

    # Number of tensors packed into the single 2-D state.  A GRU carries one
    # hidden state, so the pack/split are identity; kept as a named class
    # attribute so Phase 10/11 multi-state cells set it to 2 and reuse the same
    # ``state_size = n_state_tensors * hidden_size`` machinery.
    n_state_tensors: int = 1
    weight_gate_feedback: nn.Parameter | None
    weight_gate_input: nn.Parameter | None

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        spectral_radius: float | None = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        update_gate_bias: float = 2.0,
        activation: str = "relu",
        gate_activation: str = "sigmoid",
        trainable: bool = False,
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
        gate_recurrent_initializer: TopologySpec = None,
        gate_feedback_initializer: InitializerSpec = None,
        gate_input_initializer: InitializerSpec = None,
        gate_bias_initializer: InitializerSpec = None,
        gate_spectral_radius: float | None = None,
    ) -> None:
        super().__init__()

        self.reservoir_size = reservoir_size
        self.feedback_size = feedback_size
        # Treat input_size == 0 the same as None (no driving-input weights),
        # matching ESNCell and the ott_esn/power_augmented factories.
        self.input_size = input_size if input_size else None
        self.topology = select_recurrent_spec(
            topology=topology, recurrent_initializer=recurrent_initializer
        )
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError("Reservoir dtype must be floating point")
        if not bias and bias_initializer is not None:
            raise ValueError("bias_initializer requires bias=True")
        self.bias_initializer = bias_initializer
        self.initial_state_initializer = initial_state_initializer
        target_device = torch.device(device) if device is not None else torch.empty(0).device
        if target_device.type == "mps" and (dtype or torch.get_default_dtype()) == torch.float64:
            raise TypeError("MPS does not support float64; use float32 or CPU")
        # Initialize on CPU before a Metal transfer for portable RNG/linalg support.
        init_device = "cpu" if target_device is not None and target_device.type == "mps" else device
        self._factory_kwargs: dict[str, Any] = {"device": init_device, "dtype": dtype}
        if not bias and gate_bias_initializer is not None:
            raise ValueError("gate_bias_initializer requires bias=True")
        if not self.input_size and gate_input_initializer is not None:
            raise ValueError("gate_input_initializer requires input_size")
        if gate_spectral_radius is not None and (
            not torch.isfinite(torch.tensor(gate_spectral_radius)) or gate_spectral_radius < 0
        ):
            raise ValueError("gate_spectral_radius must be finite and non-negative")
        self.gate_recurrent_initializer = gate_recurrent_initializer
        self.gate_feedback_initializer = gate_feedback_initializer
        self.gate_input_initializer = gate_input_initializer
        self.gate_bias_initializer = gate_bias_initializer
        self.gate_spectral_radius = gate_spectral_radius
        self._separate_gate_drive = (
            gate_feedback_initializer is not None or gate_input_initializer is not None
        )
        self.spectral_radius = spectral_radius
        self.feedback_initializer = feedback_initializer
        self.input_initializer = input_initializer
        self.update_gate_bias = float(update_gate_bias)

        self._trainable = trainable
        self.seed = seed
        self._seed_int = coerce_seed_to_int(seed)
        if self._seed_int is None:
            self._seed_int = int(torch.randint(0, 2**63 - 1, (1,), device="cpu").item())

        self._activation_name = activation
        self.activation_fn = self._get_activation(activation)
        self._gate_activation_name = gate_activation
        self.gate_activation_fn = self._get_activation(gate_activation)

        self._bias = bias
        self.bias_scaling = bias_scaling

        self._initialize_weights()
        install_initial_state(
            self,
            self.initial_state_initializer,
            seed=_salt_seed(self._seed_int, 0x9E3779B9 * 7),
            state_size=self.state_size,
            reference=self.weight_feedback,
        )

        # Apply the requested freeze through the base property so construction
        # and post-init ``cell.trainable = ...`` share one code path.
        self.trainable = trainable
        if target_device is not None and target_device.type == "mps":
            self.to(target_device)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore optional initialization defaults in historical full-model pickles."""
        super().__setstate__(state)
        self.__dict__.setdefault("_separate_gate_drive", False)
        self.__dict__.setdefault("initial_state_initializer", None)
        restore_initial_state_support(self)

    # ------------------------------------------------------------------
    # ReservoirCell interface
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        """Width of a single packed state tensor (== reservoir_size)."""
        return self.reservoir_size

    @property
    def state_size(self) -> int:
        """Packed state width: ``n_state_tensors * hidden_size``.

        For this single-state (GRU) cell that is just ``hidden_size``; the
        multiplication documents the packed convention for future multi-state
        cells (see the module docstring).
        """
        return self.n_state_tensors * self.reservoir_size

    @property
    def output_size(self) -> int:
        """Dimensionality of the per-step output (the hidden state ``h``)."""
        return self.reservoir_size

    @property
    def trainable(self) -> bool:
        """
        bool : Whether the reservoir weights participate in backpropagation.

        A *live* switch: assigning to it flips ``requires_grad_`` on **every**
        cell parameter, so ``cell.trainable = True`` genuinely unfreezes the
        gated reservoir for a BPTT rollout (identical semantics to
        :class:`ESNCell`).
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        self._trainable = bool(value)
        for p in self.parameters():
            p.requires_grad_(self._trainable)

    @property
    def activation(self) -> str:
        """str : Name of the candidate activation function :math:`\\phi`."""
        return self._activation_name

    @property
    def gate_activation(self) -> str:
        """str : Name of the update-gate activation function :math:`\\sigma`."""
        return self._gate_activation_name

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a zero packed state of shape ``(batch_size, state_size)``."""
        return initialized_state(
            self,
            batch_size,
            self.state_size,
            device,
            dtype,
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the next gated state for a single timestep.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            ``inputs[0]`` is the feedback slice ``(batch, feedback_size)``;
            ``inputs[1]`` (if present) is the driving input
            ``(batch, input_size)``.
        state : torch.Tensor
            Current packed state of shape ``(batch, state_size)``.

        Returns
        -------
        output : torch.Tensor
            Next hidden state ``h`` of shape ``(batch, reservoir_size)``.
        new_state : torch.Tensor
            Next packed state (identical to ``output`` for this single-state cell).
        """
        drive = self.project_inputs(inputs)
        return self.step(drive, state)

    def project_inputs(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Precompute the input *drive* ``u = W_fb x_fb + W_in x_in`` for the sequence.

        This is the sequence-precomputable part of the gated update — it does
        **not** include the gate biases or the recurrent terms, which depend on
        the (per-step) state.  Works on both full
        ``(batch, timesteps, features)`` sequences (the layer fast path) and
        single-step ``(batch, features)`` slices (reused by :meth:`forward`).

        Parameters
        ----------
        inputs : list[torch.Tensor]
            ``inputs[0]`` feedback, ``inputs[1]`` (optional) driving input.

        Returns
        -------
        torch.Tensor
            Drive ``u`` with the inputs' leading dims and trailing dimension
            ``reservoir_size``.

        Raises
        ------
        ValueError
            On a feedback/driving-input feature-dimension mismatch, or if a
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

        if self._separate_gate_drive:
            gate_feedback = (
                self.weight_gate_feedback
                if self.weight_gate_feedback is not None
                else self.weight_feedback
            )
            gate_drive = F.linear(fb, gate_feedback)
            if len(inputs) > 1:
                gate_input = (
                    self.weight_gate_input
                    if self.weight_gate_input is not None
                    else self.weight_input
                )
                assert gate_input is not None
                gate_drive = gate_drive + F.linear(inputs[1], gate_input)
            return torch.cat((drive, gate_drive), dim=-1)
        return drive

    def step(
        self,
        projected_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step gated update consuming a slice of :meth:`project_inputs`.

        Shape-static and branch-free (both gates are ``σ`` / ``φ`` of tensors),
        so it traces cleanly under ``torch.compile(fullgraph=True)`` for the
        compiled forecast engine.

        Parameters
        ----------
        projected_t : torch.Tensor
            Per-timestep drive ``u_t`` of shape ``(batch, reservoir_size)``.
        state : torch.Tensor
            Current packed state ``(batch, state_size)``.

        Returns
        -------
        output : torch.Tensor
            Next hidden state ``h`` of shape ``(batch, reservoir_size)``.
        new_state : torch.Tensor
            Next packed state (identical to ``output`` here).
        """
        (h,) = _split_state(state, self.reservoir_size)
        if self._separate_gate_drive:
            candidate_drive, gate_drive = projected_t.split(self.reservoir_size, dim=-1)
        else:
            candidate_drive = gate_drive = projected_t

        # Update-gate pre-activation: drive + recurrent gate term + retention bias.
        z_pre = gate_drive + F.linear(h, self.weight_hz)
        if self.bias_z is not None:
            z_pre = z_pre + self.bias_z
        z = self.gate_activation_fn(z_pre)

        # Candidate pre-activation: drive + recurrent candidate term + bias.
        h_pre = candidate_drive + F.linear(h, self.weight_hh)
        if self.bias_h is not None:
            h_pre = h_pre + self.bias_h
        h_tilde = self.activation_fn(h_pre)

        # Retention convention: z→1 keeps h, z→0 writes the candidate.
        new_h = z * h + (1.0 - z) * h_tilde
        new_state = _pack_state(new_h)
        return new_h, new_state

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _get_activation(self, activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Resolve an activation function by name (shared by candidate + gate)."""
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
        """Initialize every drive, recurrent, and gate weight plus biases.

        Each role uses a distinct local stream, shared by its default draw and
        initializer fallback. Optional or overridden roles cannot shift another
        role's draws; explicitly seeded initializers retain precedence.
        """
        target_device = self._factory_kwargs["device"]
        feedback_generator = create_torch_generator(self._seed_int, device=target_device)
        input_seed = _salt_seed(self._seed_int, _INPUT_SEED_OFFSET)
        input_generator = create_torch_generator(input_seed, device=target_device)
        candidate_seed = _salt_seed(self._seed_int, _CANDIDATE_RECURRENT_SEED_OFFSET)
        candidate_generator = create_torch_generator(candidate_seed, device=target_device)
        gate_seed = _salt_seed(self._seed_int, 0x9E3779B9 * 4)
        gate_generator = create_torch_generator(gate_seed, device=target_device)
        bias_generator = create_torch_generator(
            _salt_seed(self._seed_int, 0x9E3779B9 * 5), device=target_device
        )

        # Feedback drive weight.
        self.weight_feedback = nn.Parameter(
            torch.empty(self.reservoir_size, self.feedback_size, **self._factory_kwargs)
        )
        resolved_fb = resolve_initializer(self.feedback_initializer, seed=self._seed_int)
        if resolved_fb is not None:
            resolved_fb.initialize(self.weight_feedback)
        else:
            nn.init.uniform_(self.weight_feedback, -1, 1, generator=feedback_generator)

        # Driving-input drive weight (salted seed so it never coincides with the
        # feedback stream — same rationale as ESNCell).
        if self.input_size is not None:
            self.weight_input = nn.Parameter(
                torch.empty(self.reservoir_size, self.input_size, **self._factory_kwargs)
            )
            resolved_in = resolve_initializer(self.input_initializer, seed=input_seed)
            if resolved_in is not None:
                resolved_in.initialize(self.weight_input)
            else:
                nn.init.uniform_(self.weight_input, -1, 1, generator=input_generator)
        else:
            self.register_parameter("weight_input", None)

        for role, spec, width, offset in (
            ("feedback", self.gate_feedback_initializer, self.feedback_size, 8),
            ("input", self.gate_input_initializer, self.input_size, 9),
        ):
            if spec is None:
                self.register_parameter(f"weight_gate_{role}", None)
                continue
            assert width is not None
            weight = nn.Parameter(torch.empty(self.reservoir_size, width, **self._factory_kwargs))
            initializer = resolve_initializer(
                spec, seed=_salt_seed(self._seed_int, 0x9E3779B9 * offset)
            )
            assert initializer is not None
            initializer.initialize(weight)
            if not torch.isfinite(weight).all():
                raise ValueError(f"gate_{role}_initializer must produce finite values")
            self.register_parameter(f"weight_gate_{role}", weight)

        # Recurrent candidate weight W_hh: topology-structured + spectral scaled.
        self.weight_hh = nn.Parameter(
            torch.empty(self.reservoir_size, self.reservoir_size, **self._factory_kwargs)
        )
        resolved_topo = resolve_topology(self.topology, seed=candidate_seed)
        if resolved_topo is not None:
            resolved_topo.initialize(self.weight_hh, spectral_radius=self.spectral_radius)
        else:
            nn.init.uniform_(self.weight_hh, -1.0, 1.0, generator=candidate_generator)
            if self.spectral_radius is not None:
                self._scale_spectral_radius()

        # Recurrent update-gate weight W_hz: always dense-random.  A graph
        # topology here would make the *gate* structurally sparse (wrong intent);
        # the gate is scaled small so a frozen gate does not immediately
        # saturate.  1/sqrt(N) keeps the recurrent gate pre-activation O(1).
        self.weight_hz = nn.Parameter(
            torch.empty(self.reservoir_size, self.reservoir_size, **self._factory_kwargs)
        )
        gate_scale = self.reservoir_size**-0.5
        nn.init.uniform_(self.weight_hz, -gate_scale, gate_scale, generator=gate_generator)
        gate_initializer = resolve_topology(self.gate_recurrent_initializer, seed=gate_seed)
        if gate_initializer is not None:
            gate_initializer.initialize(self.weight_hz, spectral_radius=self.gate_spectral_radius)
        elif self.gate_spectral_radius is not None:
            with torch.no_grad():
                self.weight_hz.copy_(
                    scale_to_spectral_radius(self.weight_hz, self.gate_spectral_radius)
                )
        if not torch.isfinite(self.weight_hz).all():
            raise ValueError("gate_recurrent_initializer must produce finite values")

        # Biases.
        if self._bias:
            self.bias_h = nn.Parameter(torch.empty(self.reservoir_size, **self._factory_kwargs))
            if self.bias_scaling != 0.0:
                nn.init.uniform_(
                    self.bias_h, -self.bias_scaling, self.bias_scaling, generator=bias_generator
                )
            else:
                nn.init.zeros_(self.bias_h)
            # Update-gate bias is the constant retention bias, not randomised.
            self.bias_z = nn.Parameter(
                torch.full((self.reservoir_size,), self.update_gate_bias, **self._factory_kwargs)
            )
            initialize_vector(
                self.bias_z,
                self.gate_bias_initializer,
                seed=_salt_seed(self._seed_int, 0x9E3779B9 * 6),
            )
            initialize_vector(
                self.bias_h,
                self.bias_initializer,
                seed=_salt_seed(self._seed_int, 0x9E3779B9 * 5),
            )
        else:
            self.register_parameter("bias_h", None)
            self.register_parameter("bias_z", None)

    def _scale_spectral_radius(self) -> None:
        """Scale the recurrent candidate weight ``W_hh`` to the target radius."""
        assert self.spectral_radius is not None
        with torch.no_grad():
            scaled = scale_to_spectral_radius(self.weight_hh.data, self.spectral_radius)
            self.weight_hh.data.copy_(scaled)

    @property
    def spectral_radius_achieved(self) -> float:
        """float : Realized largest absolute eigenvalue of the candidate ``W_hh``."""
        with torch.no_grad():
            return estimate_spectral_radius(self.weight_hh.data)

    def __repr__(self) -> str:
        """Return string representation."""
        input_str = f", input_size={self.input_size}" if self.input_size is not None else ""
        return (
            f"GatedReservoirCell("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"spectral_radius={self.spectral_radius}, "
            f"activation={self._activation_name}, "
            f"update_gate_bias={self.update_gate_bias}, "
            f"trainable={self._trainable}"
            f")"
        )
