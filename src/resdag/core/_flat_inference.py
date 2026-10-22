"""
Flattened single-step inference engine for autoregressive forecasting.

:meth:`resdag.core.ESNModel.forecast` rolls the model forward one timestep at a
time.  Driving that loop with the full symbolic ``__call__`` re-walks the entire
``pytorch_symbolic`` graph on *every* step: it re-dispatches
``nn.Module.__call__`` for the model and each layer and pays the reservoir
sequence-loop bookkeeping (``torch.stack`` of a one-element list, ``shape[1]``
reads, lazy-init/detach checks) just to process a single frame. Flattening the
graph removes this repeated Python dispatch and sequence bookkeeping.

This module compiles the model's shared execution plan **once** into a flat,
straight-line Python step function that captures layer references directly and
threads reservoir states explicitly (rather than mutating module buffers
mid-loop). The plan preserves the symbolic graph's operation order, so the result is
numerically identical to the per-step ``self(*step_inputs)`` path while shedding
the per-step dispatch and wrapper overhead.

The same flat function is what the opt-in ``compile=True`` path wraps in
:func:`torch.compile` — it is built to be fullgraph-clean (no Python-level
data-dependent control flow; every branch is resolved at build time), with a
transparent eager fallback when compilation is unavailable or fails.

Notes
-----
The engine feeds every non-reservoir layer a 3-D ``(batch, 1, features)``
slice. Such layers must be framewise: splitting a sequence along time must
preserve their behavior. Whole-sequence custom transforms require the common
Model's explicit history replay mode instead. Only the reservoir update is
special-cased: its parents are squeezed to
2-D for the single-step
:meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.step_stateless`
and the output is unsqueezed back.  Reservoir *states* are threaded separately and
keep their native shape (2-D for an ESN, 3-D delay buffer for NG-RC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch

from resdag.core._forecast_capabilities import validate_fast_plan
from resdag.layers.reservoirs import BaseReservoirLayer

if TYPE_CHECKING:
    from resdag.core.model import ESNModel

# A step maps (per-input (batch, 1, F) slices, reservoir states) to
# (per-output (batch, 1, F) slices, updated reservoir states).
StepFn = Callable[
    [tuple[torch.Tensor, ...], list[torch.Tensor]],
    tuple[tuple[torch.Tensor, ...], list[torch.Tensor]],
]

# A chunk step maps ((batch, 1, F) feedback, (batch, chunk, F) driver chunks,
# states) to ((batch, chunk, F) per-output blocks, updated states, last feedback).
ChunkStepFn = Callable[
    [torch.Tensor, tuple[torch.Tensor, ...], list[torch.Tensor]],
    tuple[tuple[torch.Tensor, ...], list[torch.Tensor], torch.Tensor],
]

# torch.compile gained the maturity the reduce-overhead/cudagraph forecast path
# relies on in 2.10; below that we silently use the eager flat step.
_MIN_COMPILE_TORCH = "2.10.0"

# Steps unrolled per compiled call. Wrapping the autoregressive loop in
# ``mode="reduce-overhead"`` only pays off when a single cudagraph replay covers
# many steps: a one-step graph spends as much time copying the fed-back tensors
# in/out as it saves. Unrolling ~64 steps amortises that into one replay per
# chunk and is where the large GPU long-horizon speedups come from.
DEFAULT_COMPILE_CHUNK = 64


@dataclass
class FlatStep:
    """Compiled single-step executor for a symbolic graph.

    Attributes
    ----------
    step : StepFn
        Eager straight-line step function.  Takes ``(inputs, states)`` and
        returns ``(outputs, new_states)`` where ``inputs``/``outputs`` are
        tuples of ``(batch, 1, features)`` slices (model-input / model-output
        order) and ``states``/``new_states`` are lists of reservoir states in
        :attr:`reservoir_layers` order.
    reservoir_layers : list[BaseReservoirLayer]
        Reservoir layers in graph invocation order. Repeated use of one
        reservoir instance is rejected to keep state independent per node.
    source : str
        The generated Python source of :attr:`step` (kept for debugging /
        introspection, mirroring ``pytorch_symbolic``'s
        ``_generated_forward_source``).
    n_inputs : int
        Number of model inputs the step expects.
    n_outputs : int
        Number of model outputs the step returns.
    """

    step: StepFn
    reservoir_layers: list[BaseReservoirLayer]
    source: str
    n_inputs: int
    n_outputs: int
    _compiled_chunks: dict[tuple[Any, ...], ChunkStepFn] = field(default_factory=dict, repr=False)


def build_flat_step(model: ESNModel) -> FlatStep:
    """
    Compile a model's symbolic graph into a flat single-step executor.

    Consumes the same immutable execution plan as full-sequence explicit-state
    inference and emits straight-line assignments into a generated function.
    Reservoir instances must be distinct at each graph position so sequence
    and stepwise execution have the same causal recurrent-state semantics.
    Reservoir nodes call :meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.step_stateless`
    with their state threaded in and out; every other node calls its layer's
    ``forward`` directly on its parents' values, bypassing the
    ``nn.Module.__call__`` dispatch.

    Parameters
    ----------
    model : ESNModel
        The model whose graph to flatten.  Read-only — no state is mutated.

    Returns
    -------
    FlatStep
        The compiled step, the reservoir layers in threading order, and the
        generated source.

    Notes
    -----
    The generated function captures the layers' bound methods in a closure, so
    the executor stays valid after ``.to(device)`` / ``.to(dtype)`` (the layer
    objects are unchanged; only their tensors move).  It must be rebuilt if the
    graph topology changes — e.g. after
    :meth:`pytorch_symbolic.SymbolicModel.add_output` — which the model's cache
    detects via the identity of ``model.outputs``.
    """
    plan = model._get_execution_plan()
    validate_fast_plan(plan)
    callables: list[Any] = []
    lines = [f"    v{i} = _in[{i}]" for i in range(len(plan.input_features))]
    lines += [f"    r{i} = _st[{i}]" for i in range(len(plan.reservoirs))]
    for op in plan.operations:
        parents = [f"v{i}" for i in op.parents]
        outputs = [f"v{i}" for i in op.outputs]
        index = len(callables)
        if op.reservoir_index is not None:
            state = f"r{op.reservoir_index}"
            reservoir = plan.reservoirs[op.reservoir_index]
            callables.append(reservoir.step_stateless)
            squeezed = ", ".join(f"{parent}.select(1, 0)" for parent in parents)
            lines.append(f"    u, {state} = _C[{index}]([{squeezed}], {state})")
            lines.append(f"    {outputs[0]} = u.unsqueeze(1)")
        else:
            callables.append(op.layer.forward)
            if len(outputs) > 1:
                assert len(parents) == 1, "multi-output node must have one parent"
                lines.append(f"    {', '.join(outputs)} = _C[{index}](*{parents[0]})")
            else:
                lines.append(f"    {outputs[0]} = _C[{index}]({', '.join(parents)})")
    out_src = ", ".join(f"v{i}" for i in plan.output_slots)
    state_src = ", ".join(f"r{i}" for i in range(len(plan.reservoirs)))
    lines.append(f"    return ({out_src},), [{state_src}]")
    source = "def _flat_step(_in, _st):\n" + "\n".join(lines) + "\n"
    namespace: dict[str, Any] = {"_C": tuple(callables)}
    exec(source, namespace)  # noqa: S102 - structure comes from the model, not user text.
    step_fn: StepFn = namespace["_flat_step"]
    return FlatStep(
        step=step_fn,
        reservoir_layers=list(plan.reservoirs),
        source=source,
        n_inputs=len(plan.input_features),
        n_outputs=len(plan.output_slots),
    )


def build_chunk_step(flat: FlatStep, chunk: int, n_outputs: int) -> ChunkStepFn:
    """
    Wrap :attr:`FlatStep.step` into a function that runs ``chunk`` steps in one call.

    The returned function threads the feedback and reservoir states through
    ``chunk`` autoregressive steps *internally* (a Python loop that
    :func:`torch.compile` unrolls into a single graph) and concatenates the
    per-step outputs along the time axis.  Driving the forecast with this —
    instead of one compiled call per step — is what lets ``mode="reduce-overhead"``
    amortise a cudagraph replay over many steps.

    Parameters
    ----------
    flat : FlatStep
        The single-step executor to unroll.
    chunk : int
        Number of steps per call.
    n_outputs : int
        Number of model outputs (so the result is concatenated per output).

    Returns
    -------
    ChunkStepFn
        ``f(feedback, drivers, states) -> (blocks, new_states, last_feedback)``
        where ``feedback`` is a ``(batch, 1, feedback_dim)`` slice, ``drivers`` is
        a tuple of ``(batch, chunk, driver_dim)`` tensors (empty for feedback-only
        models), and ``blocks`` is a tuple of ``(batch, chunk, out_dim)`` tensors
        in model-output order.  Numerically identical to calling
        :attr:`FlatStep.step` ``chunk`` times.
    """
    step = flat.step

    def chunk_step(
        feedback: torch.Tensor,
        drivers: tuple[torch.Tensor, ...],
        states: list[torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor], torch.Tensor]:
        acc: list[list[torch.Tensor]] = [[] for _ in range(n_outputs)]
        fb = feedback
        for k in range(chunk):
            outs, states = step((fb, *(driver[:, k : k + 1, :] for driver in drivers)), states)
            for i in range(n_outputs):
                acc[i].append(outs[i])
            fb = outs[0]
        # Each per-step output is (batch, 1, out_dim); concatenate along time.
        blocks = tuple(torch.cat(acc[i], dim=1) for i in range(n_outputs))
        return blocks, states, fb

    return chunk_step


def resolve_chunk_step(
    flat: FlatStep,
    *,
    chunk: int,
    n_outputs: int,
    mode: str,
    trial_feedback: torch.Tensor,
    trial_drivers: tuple[torch.Tensor, ...],
    trial_states: list[torch.Tensor],
) -> ChunkStepFn | None:
    """
    Build and compile a chunked step, or return ``None`` to fall back to eager.

    Wraps :func:`build_chunk_step` in :func:`torch.compile` (``fullgraph=True``)
    and *trial-traces it once* on cloned seed inputs so any compilation failure
    surfaces here.  Returns ``None`` — with a :class:`RuntimeWarning` — when
    ``torch < 2.10`` or compilation fails, signalling the caller to drive the
    forecast with the eager single step instead of aborting.

    Parameters
    ----------
    flat : FlatStep
        The single-step executor to unroll and compile.
    chunk : int
        Steps per compiled call.
    n_outputs : int
        Number of model outputs.
    mode : str
        ``torch.compile`` ``mode`` (e.g. ``"reduce-overhead"``).
    trial_feedback : torch.Tensor
        Seed feedback used to trial-trace the compiled function.
    trial_drivers : tuple[torch.Tensor, ...]
        First driver chunk (``(batch, chunk, driver_dim)`` tensors), empty for
        feedback-only models.
    trial_states : list[torch.Tensor]
        Current reservoir states; cloned internally for the trial call so the
        real loop's states are never advanced.

    Returns
    -------
    ChunkStepFn or None
        The warm compiled chunk step, or ``None`` to fall back to eager.
    """
    import warnings

    if torch.__version__ < _MIN_COMPILE_TORCH:
        warnings.warn(
            f"forecast(compile=True) needs torch>={_MIN_COMPILE_TORCH} for the "
            f"reduce-overhead step path (found {torch.__version__}); using the "
            f"eager flat step instead.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    # Reuse the warmed executor across repeated forecasts. Re-wrapping a new
    # closure on every call repeats compilation/trial capture even when all
    # shapes and devices are unchanged. The parent model drops FlatStep when
    # copied/pickled or when its graph changes, so closures remain model-local.
    def tensor_key(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (tuple(tensor.shape), tensor.dtype, tensor.device, tuple(tensor.stride()))

    key = (
        chunk,
        n_outputs,
        mode,
        tensor_key(trial_feedback),
        tuple(tensor_key(driver) for driver in trial_drivers),
        tuple(tensor_key(state) for state in trial_states),
    )
    cached = flat._compiled_chunks.get(key)
    if cached is not None:
        return cached

    try:
        compiled: ChunkStepFn = torch.compile(
            build_chunk_step(flat, chunk, n_outputs), fullgraph=True, mode=mode
        )
        # Trial-trace once on cloned states so a graph break / capture failure
        # surfaces here, where we can still fall back cleanly.
        torch.compiler.cudagraph_mark_step_begin()
        compiled(trial_feedback, trial_drivers, [state.clone() for state in trial_states])
        flat._compiled_chunks[key] = compiled
        return compiled
    except Exception as exc:
        # Any compilation/capture failure degrades to the eager flat step rather
        # than aborting the forecast (e.g. an inductor backend error).
        warnings.warn(
            f"forecast(compile=True): torch.compile failed ({type(exc).__name__}: {exc}); "
            f"falling back to the eager flat step.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def rollout(
    flat: FlatStep,
    seed_feedback: torch.Tensor,
    drivers: tuple[torch.Tensor, ...],
    states: list[torch.Tensor],
    horizon: int,
    out_dims: list[int],
    *,
    no_grad: bool,
    compile: bool = False,
    output_templates: tuple[torch.Tensor, ...] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Roll a flattened single-step engine forward autoregressively.

    Drives ``horizon`` genuinely autoregressive steps: slot ``0`` is produced
    from ``seed_feedback`` (already the model's own prediction — the last warmup
    output or a caller-supplied seed), and every later slot feeds back the
    previous step's first output.  No teacher-forced frame is emitted, so
    ``horizon=1`` yields one real forecast step.  Driver alignment matches
    training: step ``t`` consumes ``drivers[i][:, t : t + 1, :]`` alongside the
    current feedback and writes its prediction to slot ``t``.

    The rollout is the shared core of :meth:`resdag.core.ESNModel.forecast`
    (and, through it, :meth:`~resdag.core.ESNModel.windowed_forecast` and the
    ensemble forecast paths): it owns the compile/eager engine choice, the two
    output-assembly strategies, and threading reservoir states explicitly rather
    than mutating module buffers.  Callers are responsible for warmup, seed and
    initial-state resolution, and persisting the returned final states.

    Parameters
    ----------
    flat : FlatStep
        The single-step executor for the model graph (see :func:`build_flat_step`).
    seed_feedback : torch.Tensor
        Feedback of shape ``(batch, 1, feedback_dim)`` seeding the first step.
    drivers : tuple of torch.Tensor
        One ``(batch, >=horizon, driver_dim)`` tensor per driver input, empty for
        feedback-only models.  Only the first ``horizon`` timesteps are consumed;
        step ``t`` uses ``drivers[i][:, t : t + 1, :]``.
    states : list of torch.Tensor
        Initial reservoir states in :attr:`FlatStep.reservoir_layers` order.
        Threaded through the loop and returned (advanced) — not mutated in place
        beyond the compiled path's internal cloning.
    horizon : int
        Number of autoregressive steps to generate (``>= 1``).
    out_dims : list of int
        Feature dimension of each model output, in model-output order.  Its
        length is the number of outputs the step returns.
    no_grad : bool, keyword-only
        Selects the output-assembly strategy.  When ``True`` (the historical,
        graph-free default) each step is written into a preallocated
        ``torch.empty`` buffer.  When ``False`` (a differentiable BPTT rollout)
        the per-step ``(batch, 1, dim)`` slices are collected and ``torch.cat``'d
        at the end, so no in-place ``CopySlices`` grafts autograd-tracked values
        onto an uninitialised buffer.  This function does **not** open a grad
        context — the caller wraps the call (and its own seed/warmup work) in the
        matching ``torch.no_grad()`` / ``nullcontext`` so the whole two-phase
        forecast is consistent.
    compile : bool, keyword-only, default=False
        If ``True``, wrap the step in a compiled chunked executor
        (``mode="reduce-overhead"``, ``fullgraph=True``) via
        :func:`resolve_chunk_step`, driving ``DEFAULT_COMPILE_CHUNK``-step chunks
        per cudagraph replay and filling the ``< chunk`` remainder eagerly.  On
        ``torch < 2.10`` or a compilation failure it transparently falls back to
        the eager single-step loop (with a :class:`RuntimeWarning`).  The compile
        path is inference-only, so callers must gate it to ``no_grad=True`` (the
        ``forecast`` public method raises otherwise); here it only ever writes
        into the preallocated buffers.
    output_templates : tuple of torch.Tensor, optional
        One warmup output per head, used to preserve each head's dtype and
        device when allocating inference buffers. Omitted templates retain the
        feedback placement, for callers with homogeneous outputs.

    Returns
    -------
    forecast_buffers : list of torch.Tensor
        One ``(batch, horizon, out_dim)`` tensor per model output, in
        model-output order.  With ``no_grad=False`` these carry the BPTT
        ``grad_fn``.
    final_states : list of torch.Tensor
        The reservoir states after the last step, in
        :attr:`FlatStep.reservoir_layers` order.  The caller persists them
        (honouring ``detach_state_between_calls``) so a follow-up
        ``forecast(reset=False)`` continues from here.

    Notes
    -----
    The compiled and eager paths are numerically identical to calling
    :attr:`FlatStep.step` ``horizon`` times, and both are bit-for-bit identical
    to the per-step symbolic re-execution the engine replaces.
    """
    batch_size = seed_feedback.shape[0]
    dtype = seed_feedback.dtype
    device = seed_feedback.device
    n_out = len(out_dims)
    has_drivers = len(drivers) > 0
    current_feedback = seed_feedback  # (batch, 1, feedback_dim)

    # Two assembly strategies share the same per-step engine:
    #
    # * default (``no_grad=True``): write each step into a preallocated
    #   buffer — minimal allocation, no graph anyway;
    # * differentiable (``no_grad=False``): collect per-step slices and
    #   ``torch.cat`` them at the end.  In-place index writes into a
    #   ``torch.empty`` buffer would graft autograd-tracked values onto an
    #   uninitialised tensor (a ``CopySlices`` per step); accumulating in a
    #   list and concatenating keeps the BPTT graph clean and standard.
    forecast_buffers: list[torch.Tensor] = []
    step_accumulators: list[list[torch.Tensor]] = []
    if no_grad:
        if output_templates is None:
            forecast_buffers = [
                torch.empty(batch_size, horizon, dim, dtype=dtype, device=device)
                for dim in out_dims
            ]
        else:
            if len(output_templates) != n_out:
                raise ValueError("Expected one output template per forecast head")
            forecast_buffers = [
                template.new_empty(batch_size, horizon, dim)
                for template, dim in zip(output_templates, out_dims)
            ]
    else:
        step_accumulators = [[] for _ in range(n_out)]

    # Optionally compile a chunked step. ``done`` tracks how many steps the
    # compiled path has produced; the eager single-step loop below fills the
    # rest (the whole horizon when not compiling, or the < chunk remainder).
    # The compile path is gated to ``no_grad=True`` by the caller, so it only
    # ever writes into ``forecast_buffers``.
    done = 0
    if compile:
        chunk = min(DEFAULT_COMPILE_CHUNK, horizon)
        compiled_chunk = resolve_chunk_step(
            flat,
            chunk=chunk,
            n_outputs=n_out,
            mode="reduce-overhead",
            trial_feedback=current_feedback,
            trial_drivers=tuple(d[:, 0:chunk, :] for d in drivers),
            trial_states=states,
        )
        if compiled_chunk is not None:
            for _ in range(horizon // chunk):
                driver_chunk = tuple(d[:, done : done + chunk, :] for d in drivers)
                # cudagraph trees reuse static output buffers; mark a new
                # step so this replay's outputs may feed the next, and clone
                # the tensors carried across the chunk boundary.
                torch.compiler.cudagraph_mark_step_begin()
                blocks, states, current_feedback = compiled_chunk(
                    current_feedback, driver_chunk, states
                )
                for i in range(n_out):
                    forecast_buffers[i][:, done : done + chunk, :] = blocks[i]
                current_feedback = current_feedback.clone()
                states = [state.clone() for state in states]
                done += chunk

    # Eager single-step loop: the whole horizon when not compiling, or the
    # trailing remainder the chunked path could not cover.
    # One view per output exposes (time, batch, 1, features). Selecting a frame
    # then copying avoids repeated three-axis indexing and squeeze operations.
    # Copy immediately: custom inference layers may reuse their output buffers.
    output_views = (
        [buffer.permute(1, 0, 2).unsqueeze(2) for buffer in forecast_buffers] if no_grad else []
    )
    for t in range(done, horizon):
        if has_drivers:
            step_inputs = (current_feedback, *(d[:, t : t + 1, :] for d in drivers))
        else:
            step_inputs = (current_feedback,)
        step_outputs, states = flat.step(step_inputs, states)
        if no_grad:
            for i in range(n_out):
                output_views[i][t].copy_(step_outputs[i])
        else:
            for i in range(n_out):
                step_accumulators[i].append(step_outputs[i])
        current_feedback = step_outputs[0]

    if not no_grad:
        # Concatenate the per-step (batch, 1, dim) slices into the full
        # (batch, horizon, dim) forecast for each output.
        forecast_buffers = [torch.cat(acc, dim=1) for acc in step_accumulators]

    return forecast_buffers, states
