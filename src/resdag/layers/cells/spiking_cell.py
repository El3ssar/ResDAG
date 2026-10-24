r"""
Spiking Reservoir Cell (shared base)
====================================

This module provides `SpikingReservoirCell`, an intermediate abstract
base for **spiking** reservoir cells — the substrate of a Liquid State Machine
(LSM; Maass, Natschläger & Markram, *Neural Computation* 14(11), 2002).  It is
the groundwork for the spiking family; the first concrete subclass is the leaky
integrate-and-fire cell (`resdag.layers.cells.lif_cell.LIFCell`).

Unlike the ESN / continuous cells, a spiking reservoir carries **multiple**
per-neuron state tensors — a membrane potential `V`, an optional synaptic
current `I`, and a low-pass **synaptic trace** of the spike train — and emits
*binary spikes* internally.  The base owns:

* the **packed multi-tensor state** (the Phase-9 convention re-used verbatim from
  `resdag.layers.cells.gated_cell`): every `(batch, hidden)` state tensor
  is packed side-by-side into one 2-D `(batch, n_state_tensors * hidden)`
  tensor, so the single-tensor `resdag.layers.cells.base_cell.ReservoirCell`
  contract is honoured;
* the **surrogate-gradient spike** — a straight-through estimator whose forward
  is the exact Heaviside threshold and whose backward flows through a smooth
  surrogate, so the cell is `torch.compile(fullgraph=True)`-clean *and*
  trainable by SGD (no custom `torch.autograd.Function`);
* the **spike reset** (subtract or zero);
* the **readout trace** — a real-valued low-pass filter of the spike train, the
  cell's per-step *output*.  A ridge / CG readout cannot regress on raw binary
  spikes, so the readable output is this filtered response, **not** the spikes.

A concrete subclass supplies only the neuron's sub-threshold dynamics via
`neuron_update`; this base wires the spike, reset, trace, packing, and the
`forward` / `project_inputs` / `step` plumbing (so the compiled forecast
engine applies unchanged).

Why a straight-through surrogate (and not `autograd.Function`)
---------------------------------------------------------------
The spike is

$$
s = \operatorname{surrogate} + (\operatorname{hard} - \operatorname{surrogate})
    .\mathrm{detach}(),
$$

with `hard = (V - V_\text{th} > 0)` the exact Heaviside forward and
`surrogate = σ(β (V - V_\text{th}))` (a fast sigmoid) the smooth backward.  In
the forward pass the detached bracket cancels the surrogate, so `s == hard`
*exactly* (binary spikes).  In the backward pass the detached bracket has zero
gradient, so `∂s/∂V == ∂surrogate/∂V = β σ'` — nonzero near threshold,
vanishing far away.  This is the standard *surrogate-gradient* trick (Neftci,
Mostafa & Zenke, *IEEE Signal Processing Magazine* 36(6), 2019).  Expressing it
with `torch.Tensor.detach` (rather than a custom
`torch.autograd.Function`) keeps the whole step a pure composition of
tensor ops, which `torch.compile` can trace `fullgraph=True`.

See Also
--------
resdag.layers.cells.base_cell.ReservoirCell : Abstract single-step cell interface.
resdag.layers.cells.gated_cell : Source of the packed multi-tensor-state helpers.
resdag.layers.cells.lif_cell.LIFCell : Concrete leaky integrate-and-fire cell.
resdag.layers.reservoirs.lif.LIFReservoirLayer : Layer that sequences the LIF cell.
"""

from abc import ABC, abstractmethod

import torch

from .base_cell import ReservoirCell
from .gated_cell import _pack_state, _split_state

# Readout-output modes.  ``"trace"`` (the default and the LSM-correct choice) is
# the real-valued low-pass-filtered spike train a ridge/CG readout can regress
# on; ``"voltage"`` exposes the raw membrane potential; ``"spikes"`` exposes the
# binary spike train (useful for analysis / spike-based readouts, but not
# directly ridge-regressable).
_VALID_OUTPUTS = frozenset({"trace", "voltage", "spikes"})

# Spike-reset modes.  ``"subtract"`` (default) subtracts the threshold on a
# spike (soft reset, preserves supra-threshold overshoot — the classic LIF
# reset); ``"zero"`` clamps the membrane to 0 on a spike (hard reset).
_VALID_RESETS = frozenset({"subtract", "zero"})


class SpikingReservoirCell(ReservoirCell, ABC):
    r"""
    Abstract base for spiking reservoir cells (the LSM substrate).

    Owns the machinery shared by every spiking neuron model — the packed
    multi-tensor state, the surrogate-gradient spike, the spike reset, and the
    low-pass **readout trace** — and delegates only the sub-threshold membrane
    dynamics to a concrete subclass through `neuron_update`.

    Packed state layout
    -------------------
    A spiking cell carries several `(batch, reservoir_size)` state tensors,
    packed side-by-side into one 2-D `(batch, n_state_tensors * reservoir_size)`
    tensor (the convention documented in
    `resdag.layers.cells.gated_cell`).  The layout is **fixed by the
    subclass** via `n_state_tensors`; the last packed block is *always* the
    synaptic trace `r` (owned by this base), and the leading blocks are the
    subclass's neuron state, which **ends with the previous step's spikes `s`**
    (the recurrent input) — e.g. `V` then `s`, or `I`, `V` then `s`.
    Concretely:

    * first-order LIF (`n_state_tensors` = 3): `[V | s | r]`;
    * CuBa-LIF (`n_state_tensors` = 4): `[I | V | s | r]`.

    `neuron_update` receives / returns only the *neuron* tensors (all but
    the trace); the base peels the trace off before the call and re-appends the
    updated trace after, so a subclass never handles the trace.  Because those
    neuron tensors end with `s`, the membrane potential `V` is the
    second-to-last neuron tensor — this is what `output="voltage"` returns.

    The surrogate-gradient spike
    ----------------------------
    `spike` implements the straight-through estimator: an exact Heaviside
    forward with a fast-sigmoid surrogate backward (slope `surrogate_beta`).
    Subclasses call it from `neuron_update` to emit spikes from the
    (updated) membrane potential.

    Parameters
    ----------
    reservoir_size : int
        Number of spiking units (hidden dimension of each packed state tensor).
    v_threshold : float, default=1.0
        Firing threshold $V_\text{th}$.  A neuron spikes when its membrane
        potential crosses this value from below.
    reset : {'subtract', 'zero'}, default='subtract'
        Membrane reset on a spike.  `'subtract'` does the soft reset
        $V \leftarrow V - V_\text{th}\,s$ (preserves overshoot);
        `'zero'` does the hard reset $V \leftarrow V (1 - s)$.
    surrogate_beta : float, default=10.0
        Slope $\beta$ of the fast-sigmoid surrogate
        $\sigma(\beta (V - V_\text{th}))$ used for the straight-through
        backward pass.  Larger values sharpen the surrogate toward the true
        (non-differentiable) step; the default follows common SNN practice.
    trace_decay : float, default=0.9
        Decay $\kappa \in [0, 1)$ of the low-pass readout trace
        $r_t = \kappa\, r_{t-1} + s_t$.  The trace is the cell's readable
        output (a real-valued filtered spike response).  `0.0` makes the trace
        the instantaneous spike; values near `1` give long synaptic memory.
    output : {'trace', 'voltage', 'spikes'}, default='trace'
        Which quantity `forward` / `step` return as the readable
        output.  `'trace'` (default) is the ridge-regressable filtered spike
        response — **the LSM-correct choice**; `'voltage'` exposes the raw
        membrane potential; `'spikes'` exposes the binary spike train.

    Attributes
    ----------
    reservoir_size : int
        Hidden dimension of each packed state tensor.
    v_threshold : float
        Firing threshold.
    surrogate_beta : float
        Surrogate slope (live knob).
    trace_decay : float
        Readout-trace decay (live knob).

    Notes
    -----
    Refractoriness is intentionally **not** modelled in this first version: a
    refractory counter is integer-valued and its "is the neuron clamped?" branch
    is data-dependent, which fights `torch.compile(fullgraph=True)` and the
    surrogate-gradient flow.  It is deferred as future work rather than shipped
    fragile; the soft/hard reset already yields the regular-spiking f–I behaviour
    a Liquid State Machine relies on.

    See Also
    --------
    resdag.layers.cells.lif_cell.LIFCell : Concrete leaky integrate-and-fire cell.
    resdag.layers.cells.base_cell.ReservoirCell : Abstract cell interface.
    """

    # Total number of packed state tensors, INCLUDING the trace ``r`` this base
    # owns.  A subclass sets it: first-order LIF -> 2 ([V | r]); CuBa-LIF -> 3
    # ([I | V | r]).  ``n_neuron_tensors == n_state_tensors - 1`` is the count of
    # tensors handed to / returned from ``neuron_update``.
    n_state_tensors: int = 2

    def __init__(
        self,
        reservoir_size: int,
        v_threshold: float = 1.0,
        reset: str = "subtract",
        surrogate_beta: float = 10.0,
        trace_decay: float = 0.9,
        output: str = "trace",
    ) -> None:
        super().__init__()

        if reservoir_size <= 0:
            raise ValueError(f"reservoir_size must be a positive integer, got {reservoir_size}")
        if reset not in _VALID_RESETS:
            raise ValueError(f"Unknown reset {reset!r}. Supported: {sorted(_VALID_RESETS)}.")
        if output not in _VALID_OUTPUTS:
            raise ValueError(f"Unknown output {output!r}. Supported: {sorted(_VALID_OUTPUTS)}.")

        self.reservoir_size = reservoir_size
        self.v_threshold = float(v_threshold)
        self.reset = reset
        self._output = output

        # ``surrogate_beta`` / ``trace_decay`` are live knobs validated by their
        # property setters (shared by the constructor and post-init writes).
        self._surrogate_beta: float = 10.0
        self._trace_decay: float = 0.0
        self.surrogate_beta = surrogate_beta
        self.trace_decay = trace_decay

    # ------------------------------------------------------------------
    # ReservoirCell interface (sizes / state)
    # ------------------------------------------------------------------

    @property
    def n_neuron_tensors(self) -> int:
        """Number of *neuron* state tensors (packed state minus the trace)."""
        return self.n_state_tensors - 1

    @property
    def state_size(self) -> int:
        """Packed state width: `n_state_tensors * reservoir_size`.

        Includes the trailing synaptic-trace block owned by this base, so a
        first-order LIF (`n_state_tensors == 2`) has `state_size == 2 * reservoir_size` and a CuBa-LIF (`3`) has `3 * reservoir_size`.
        """
        return self.n_state_tensors * self.reservoir_size

    @property
    def output_size(self) -> int:
        """Dimensionality of the per-step readable output (`reservoir_size`)."""
        return self.reservoir_size

    @property
    def output(self) -> str:
        """str : Readable-output mode (`'trace'` | `'voltage'` | `'spikes'`)."""
        return self._output

    @property
    def surrogate_beta(self) -> float:
        r"""float : Fast-sigmoid surrogate slope $\beta > 0$ (live knob)."""
        return self._surrogate_beta

    @surrogate_beta.setter
    def surrogate_beta(self, value: float) -> None:
        scalar = float(value)
        if scalar <= 0.0:
            raise ValueError(f"surrogate_beta must be positive, got {value}")
        self._surrogate_beta = scalar

    @property
    def trace_decay(self) -> float:
        r"""float : Readout-trace decay $\kappa \in [0, 1)$ (live knob)."""
        return self._trace_decay

    @trace_decay.setter
    def trace_decay(self, value: float) -> None:
        scalar = float(value)
        if not 0.0 <= scalar < 1.0:
            raise ValueError(f"trace_decay must be in [0, 1), got {value}")
        self._trace_decay = scalar

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a zero packed state of shape `(batch_size, state_size)`."""
        return torch.zeros(batch_size, self.state_size, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Shared spiking machinery
    # ------------------------------------------------------------------

    def spike(self, v: torch.Tensor) -> torch.Tensor:
        r"""Surrogate-gradient spike from a membrane potential (straight-through).

        Forward is the exact Heaviside threshold `(v - V_th > 0)`; backward
        flows through the fast-sigmoid surrogate
        $\sigma(\beta (v - V_\text{th}))$.  The
        `surrogate + (hard - surrogate).detach()` construction makes the
        returned tensor *equal* `hard` in value while carrying the surrogate's
        gradient — a pure composition of tensor ops (no custom
        `torch.autograd.Function`), so it traces `fullgraph=True`.

        Parameters
        ----------
        v : torch.Tensor
            Membrane potential, shape `(batch, reservoir_size)`.

        Returns
        -------
        torch.Tensor
            Binary spike tensor (values in `{0, 1}`), same shape as `v`,
            with a surrogate gradient w.r.t. `v`.
        """
        shifted = v - self.v_threshold
        hard = (shifted > 0).to(v.dtype)
        surrogate = torch.sigmoid(self._surrogate_beta * shifted)
        return surrogate + (hard - surrogate).detach()

    def apply_reset(self, v: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
        r"""Reset the membrane potential on the neurons that just spiked.

        `'subtract'` (soft): $V \leftarrow V - V_\text{th}\,s$ — keeps the
        supra-threshold overshoot, the standard LIF reset.  `'zero'` (hard):
        $V \leftarrow V (1 - s)$ — clamps the membrane to zero on a spike.
        The branch is a Python attribute compare fixed at construction, so it is
        static under `torch.compile`.

        Parameters
        ----------
        v : torch.Tensor
            Post-integration membrane potential, shape `(batch, reservoir_size)`.
        spikes : torch.Tensor
            Spike tensor from `spike`, same shape.

        Returns
        -------
        torch.Tensor
            The reset membrane potential.
        """
        if self.reset == "zero":
            return v * (1.0 - spikes)
        return v - self.v_threshold * spikes

    def update_trace(self, trace: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
        r"""Low-pass the spike train into the readout trace.

        $r_t = \kappa\, r_{t-1} + s_t$ with $\kappa$ = `trace_decay`.
        The trace is the real-valued synaptic response a readout regresses on.

        Parameters
        ----------
        trace : torch.Tensor
            Previous trace `r_{t-1}`, shape `(batch, reservoir_size)`.
        spikes : torch.Tensor
            Current spikes `s_t`, same shape.

        Returns
        -------
        torch.Tensor
            Updated trace `r_t`.
        """
        return self._trace_decay * trace + spikes

    def _select_output(
        self, trace: torch.Tensor, v: torch.Tensor, spikes: torch.Tensor
    ) -> torch.Tensor:
        """Pick the readable output tensor per `output` (static branch)."""
        if self._output == "voltage":
            return v
        if self._output == "spikes":
            return spikes
        return trace

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def neuron_update(
        self,
        neuron_state: tuple[torch.Tensor, ...],
        projected_input: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        r"""Advance the sub-threshold dynamics one step and emit spikes.

        Called by `step` with the *neuron* state tensors only (the trailing
        trace is peeled off by the base) and the precomputed input drive.  The
        subclass integrates its membrane / current dynamics, emits spikes via the
        shared `spike` helper, applies the reset via `apply_reset`,
        and returns the updated neuron tensors together with the spikes.

        Parameters
        ----------
        neuron_state : tuple of torch.Tensor
            The `n_neuron_tensors` neuron state tensors in packing order —
            `(V,)` for first-order LIF, `(I, V)` for CuBa-LIF — each of shape
            `(batch, reservoir_size)`.
        projected_input : torch.Tensor
            Precomputed input drive (recurrent-spike + feedback + driver + bias
            *excluding* the recurrent term, which the subclass adds from the
            *previous* spikes it keeps in state — see `LIFCell`), shape
            `(batch, reservoir_size)`.

        Returns
        -------
        new_neuron_state : tuple of torch.Tensor
            The updated neuron tensors in the same packing order.
        spikes : torch.Tensor
            The binary spike tensor emitted this step, shape
            `(batch, reservoir_size)` (carries the surrogate gradient).
        """
        ...

    @abstractmethod
    def project_inputs(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Precompute the per-step input drive for the sequence.

        Concrete spiking cells own the feedback / input / bias weights, so the
        projection lives on the subclass (mirrors
        `resdag.layers.cells.esn_cell.ESNCell.project_inputs`).
        """
        ...

    # ------------------------------------------------------------------
    # ReservoirCell interface (forward / step)
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one timestep: project inputs, then `step`.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            `inputs[0]` feedback `(batch, feedback_size)`; `inputs[1]`
            (optional) driving input `(batch, input_size)`.
        state : torch.Tensor
            Current packed state `(batch, state_size)`.

        Returns
        -------
        output : torch.Tensor
            Readable per-step output `(batch, reservoir_size)` (see
            `output`).
        new_state : torch.Tensor
            Updated packed state `(batch, state_size)`.
        """
        projected = self.project_inputs(inputs)
        return self.step(projected, state)

    def step(
        self,
        projected_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Single-step spiking update from a precomputed input drive.

        Peels the trailing trace `r` off the packed state, hands the neuron
        tensors + drive to `neuron_update`, low-passes the returned spikes
        into the new trace, and re-packs.  Branch-free apart from the static
        reset / output selection, so it traces `fullgraph=True` for the
        compiled forecast / teacher-forced engines.

        Parameters
        ----------
        projected_t : torch.Tensor
            Precomputed input drive for this step, shape
            `(batch, reservoir_size)`.
        state : torch.Tensor
            Current packed state `(batch, state_size)`.

        Returns
        -------
        output : torch.Tensor
            Readable per-step output `(batch, reservoir_size)`.
        new_state : torch.Tensor
            Updated packed state `(batch, state_size)`.
        """
        tensors = _split_state(state, self.reservoir_size)
        neuron_state = tensors[:-1]
        trace = tensors[-1]

        new_neuron_state, spikes = self.neuron_update(neuron_state, projected_t)
        new_trace = self.update_trace(trace, spikes)

        new_state = _pack_state(*new_neuron_state, new_trace)
        # The neuron tensors always END with the current spikes ``s``
        # (``[V | s]`` or ``[I | V | s]``), so the membrane potential ``V`` is the
        # *second-to-last* neuron tensor — that is what the "voltage" mode reads.
        v = new_neuron_state[-2]
        output = self._select_output(new_trace, v, spikes)
        return output, new_state
