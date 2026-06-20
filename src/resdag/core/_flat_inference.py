"""
Flattened single-step inference engine for autoregressive forecasting.

:meth:`resdag.core.ESNModel.forecast` rolls the model forward one timestep at a
time.  Driving that loop with the full symbolic ``__call__`` re-walks the entire
``pytorch_symbolic`` graph on *every* step: it re-dispatches
``nn.Module.__call__`` for the model and each layer and pays the reservoir
sequence-loop bookkeeping (``torch.stack`` of a one-element list, ``shape[1]``
reads, lazy-init/detach checks) just to process a single frame.  Profiled, that
is ~3x the cost of the raw per-step cell update — the reason forecasting lost to
NumPy baselines at every horizon despite training winning by a wide margin.

This module compiles the symbolic graph **once** into a flat, straight-line
Python step function that captures the layer references directly and threads
reservoir states explicitly (rather than mutating module buffers mid-loop).
The generated function mirrors the order ``pytorch_symbolic`` itself uses for
its codegen'd ``forward`` (``model._execution_order_nodes``), so the result is
numerically identical to the per-step ``self(*step_inputs)`` path while shedding
the per-step dispatch and wrapper overhead.

The same flat function is what the opt-in ``compile=True`` path wraps in
:func:`torch.compile` — it is built to be fullgraph-clean (no Python-level
data-dependent control flow; every branch is resolved at build time), with a
transparent eager fallback when compilation is unavailable or fails.

Notes
-----
The engine feeds every non-reservoir layer the same 3-D ``(batch, 1, features)``
slice the graph path would, so the result is identical for *any* layer — including
ones that inspect input rank or operate along the time axis — not just feature-wise
transforms.  (The built-in transforms are additionally rank-agnostic, so they would
also work on a squeezed 2-D slice; keeping the engine 3-D makes that independent of
the layer.)  Only the reservoir update is special-cased: its parents are squeezed to
2-D for the single-step
:meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.step_stateless`
and the output is unsqueezed back.  Reservoir *states* are threaded separately and
keep their native shape (2-D for an ESN, 3-D delay buffer for NG-RC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

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
        Reservoir layers in graph execution (topological) order — the order in
        which states are threaded through :attr:`step`.
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


def build_flat_step(model: ESNModel) -> FlatStep:
    """
    Compile a model's symbolic graph into a flat single-step executor.

    Walks ``model._execution_order_nodes`` (the same topologically ordered node
    list ``pytorch_symbolic`` replays in its ``forward``) and emits one
    straight-line assignment per node into a generated Python function.
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
    inputs = list(model.inputs)
    outputs = list(model.outputs)
    exec_nodes = list(model._execution_order_nodes)

    # id(node) -> the variable name holding its per-step value.
    var: dict[int, str] = {id(node): f"a{i}" for i, node in enumerate(inputs)}

    callables: list[Any] = []  # referenced as _C[k] in the generated source
    reservoir_layers: list[BaseReservoirLayer] = []
    new_state_vars: list[str] = []
    node_lines: list[str] = []
    vcount = 0

    for node in exec_nodes:
        layer = node.layer
        parent_vars = [var[id(parent)] for parent in node._parents]
        siblings = node._layer_full_siblings

        if isinstance(layer, BaseReservoirLayer):
            ridx = len(reservoir_layers)
            reservoir_layers.append(layer)
            k = len(callables)
            callables.append(layer.step_stateless)
            tmp_var = f"u{ridx}"
            value_var = f"v{vcount}"
            vcount += 1
            var[id(node)] = value_var
            state_var = f"nr{ridx}"
            new_state_vars.append(state_var)
            # step_stateless works in 2-D; squeeze the singleton time axis off
            # each parent slice, run the single-step update, and restore it.
            squeezed = ", ".join(f"{p}[:, 0, :]" for p in parent_vars)
            node_lines.append(f"    {tmp_var}, {state_var} = _C[{k}]([{squeezed}], r{ridx})")
            node_lines.append(f"    {value_var} = {tmp_var}.unsqueeze(1)")
        elif len(siblings) > 1:
            # Multi-output (unpack) node: pytorch_symbolic's ``_launch`` calls the
            # layer as ``layer(*parent_value)`` (a single, iterable parent), so the
            # parent must be splatted. Every sibling value is bound, even siblings
            # that are unused downstream.
            assert len(parent_vars) == 1, "multi-output node must have exactly one parent"
            k = len(callables)
            callables.append(layer.forward)
            sibling_vars = []
            for sibling in siblings:
                sib_var = f"v{vcount}"
                vcount += 1
                var[id(sibling)] = sib_var
                sibling_vars.append(sib_var)
            node_lines.append(f"    {', '.join(sibling_vars)} = _C[{k}](*{parent_vars[0]})")
        else:
            k = len(callables)
            callables.append(layer.forward)
            value_var = f"v{vcount}"
            vcount += 1
            var[id(node)] = value_var
            node_lines.append(f"    {value_var} = _C[{k}]({', '.join(parent_vars)})")

    header = [f"    a{i} = _in[{i}]" for i in range(len(inputs))]
    header += [f"    r{i} = _st[{i}]" for i in range(len(reservoir_layers))]
    out_src = ", ".join(var[id(node)] for node in outputs)
    ns_src = ", ".join(new_state_vars)
    return_line = f"    return ({out_src},), [{ns_src}]"

    source = "def _flat_step(_in, _st):\n" + "\n".join(header + node_lines + [return_line]) + "\n"

    namespace: dict[str, Any] = {"_C": tuple(callables)}
    # Codegen + exec mirrors pytorch_symbolic's own forward codegen; the source
    # is fully determined by the model graph (no user input is interpolated).
    exec(source, namespace)  # noqa: S102
    step_fn: StepFn = namespace["_flat_step"]

    return FlatStep(
        step=step_fn,
        reservoir_layers=reservoir_layers,
        source=source,
        n_inputs=len(inputs),
        n_outputs=len(outputs),
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

    try:
        compiled: ChunkStepFn = torch.compile(
            build_chunk_step(flat, chunk, n_outputs), fullgraph=True, mode=mode
        )
        # Trial-trace once on cloned states so a graph break / capture failure
        # surfaces here, where we can still fall back cleanly.
        torch.compiler.cudagraph_mark_step_begin()
        compiled(trial_feedback, trial_drivers, [state.clone() for state in trial_states])
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
