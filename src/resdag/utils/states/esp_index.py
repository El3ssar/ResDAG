"""
Echo State Property Index Calculation
======================================

This module provides the :func:`esp_index` function for computing the Echo State
Property (ESP) index of reservoir layers in ESN models.

The implementation is **cell-agnostic**: it discovers every
:class:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer` in the model
(not only :class:`~resdag.layers.reservoirs.esn.ESNLayer`), routes all state
mutation through the public, validating state-management API
(:meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.reset_state`
and
:meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.set_random_state`),
and supports states of any rank (the 2-D ESN hidden state as well as the 3-D
NG-RC delay buffer).  Reservoirs whose ESP is undefined — those carrying no
randomisable state, e.g. a ``k=1`` NG-RC reservoir — are skipped with an
explicit message instead of silently producing a meaningless value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import torch

if TYPE_CHECKING:
    from resdag.core import ESNModel
    from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer


def esp_index(
    model: "ESNModel",
    feedback_seq: torch.Tensor,
    *driving_seqs: torch.Tensor,
    history: bool = False,
    iterations: int = 10,
    transient: int = 0,
    window: int | None = None,
    relative: bool = False,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> Union[
    dict[str, list[torch.Tensor]],
    Tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]]],
]:
    """
    Compute the Echo State Property (ESP) index for reservoir layers.

    The ESP index measures how quickly trajectories from different initial
    states converge when driven by the same input.  All
    :class:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer`
    reservoirs are discovered automatically, so the diagnostic applies equally
    to ESN, NG-RC, and any future custom-cell reservoir.

    Random restarts are drawn through each reservoir's public
    :meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.set_random_state`,
    which samples a **standard-normal** state — the same convention as
    :meth:`~resdag.core.ESNModel.set_random_reservoir_states`.  The index is
    therefore directly comparable to states produced elsewhere in the library.

    Parameters
    ----------
    model : ESNModel
        Model containing reservoir layers.
    feedback_seq : torch.Tensor
        Feedback sequence, shape ``(batch, timesteps, features)``.
    *driving_seqs : torch.Tensor
        Optional driving sequences in model input order.
    history : bool, default=False
        If True, return full distance history over time.
    iterations : int, default=10
        Number of random initial states to average over.
    transient : int, default=0
        Timesteps to discard from sequence start.
    window : int or None, default=None
        Length of the trailing window over which the index is averaged.  By
        default (``None``) the index is the mean over *every* post-transient
        timestep, which for a healthy reservoir is dominated by the early
        convergence transient rather than the asymptotic distance.  Setting
        ``window=k`` averages only the **last** ``k`` post-transient steps, so
        the index reflects the asymptotic regime — a far more diagnostic
        statistic.  Must be a positive integer no larger than the number of
        post-transient timesteps.  Does not affect the returned ``history``
        tensors, which always span the full post-transient sequence.
    relative : bool, default=False
        If True, normalise each per-step distance by the corresponding
        base-state norm, ``||base - rand|| / (||base|| + eps)``, yielding a
        scale-free relative distance that is comparable across reservoir sizes
        and spectral radii.  If False (default), the raw absolute distance is
        used.  The normalisation is applied to both the returned index and the
        ``history`` tensors.
    chunk_size : int or None, default=None
        Maximum number of random restarts run together in a single batched
        forward pass.  All restarts are statistically independent and differ
        only in their initial state, so they are folded into the batch
        dimension and evaluated in one forward pass instead of ``iterations``
        sequential passes (typically a ~3x speed-up).  A large
        ``reservoir_size * iterations`` product can exhaust device memory,
        however; set ``chunk_size`` to cap the per-pass batch at
        ``batch * chunk_size`` and process the restarts in chunks.  ``None``
        (default) runs all ``iterations`` restarts in one pass, transparently
        falling back to chunking (halving the chunk each time) if a CUDA
        out-of-memory error is raised.
    verbose : bool, default=True
        Print progress and skip messages.

    Returns
    -------
    dict or tuple
        If ``history=False``: dict mapping layer names to single-element
        lists containing the ESP index scalar tensor.
        If ``history=True``: tuple of (ESP indices dict, history dict).
        History tensors have shape ``(iterations, timesteps, batch)`` and span
        the full post-transient sequence regardless of ``window``.

        Reservoirs whose ESP is undefined (no randomisable state) are omitted
        from both dictionaries.

    Raises
    ------
    ValueError
        If ``transient`` is not smaller than the number of timesteps, if
        ``window`` is not a positive integer no larger than the number of
        post-transient timesteps, if the model contains no reservoir layers,
        or if no reservoir has a well-defined ESP.

    Notes
    -----
    For LINEAR systems (identity activation), input does NOT affect ESP
    because the input contribution cancels out in the state difference.
    This is mathematically correct behavior.

    State is never mutated by direct assignment.  The zero base orbit is set
    via ``res.reset_state(batch_size=...)`` and the random restarts via
    ``res.set_random_state()``, so each cell's own shape contract (2-D for
    ESN, 3-D delay buffer for NG-RC) is honoured and validated by the public
    state-management API.

    The caller's reservoir states are saved before the computation and
    restored afterwards (even on error), so calling ``esp_index`` never leaves
    the model's reservoirs in a mutated (random) state.

    **Batched restarts.**  The ``iterations`` random restarts are statistically
    independent and differ only in their initial state, so they are folded into
    the batch dimension: the feedback (and any drivers) are tiled ``iterations``
    times, one random initial state is composed per restart into a single
    ``(batch * iterations, ...)`` state, and the whole ensemble is evaluated in
    **one** forward pass rather than ``iterations`` sequential passes.  The
    random restarts are drawn in exactly the same order as the sequential loop
    (outer over restarts, inner over reservoirs), so the *initial states* are
    bit-identical to the sequential computation.  The reservoir *outputs* — and
    hence the returned index — match the sequential value to floating-point
    rounding (typically ``< 1e-6`` relative), the only difference being the
    reassociation of a wider batched matmul; the result is fully deterministic
    under a fixed seed.  (If a reservoir cell injects per-step state noise in
    training mode, ``noise > 0``, the batched noise draws differ from the
    sequential ones; the index remains a valid, seed-deterministic stochastic
    diagnostic but is no longer numerically identical.  Run the model in eval
    mode, the default for a diagnostic, to disable noise entirely.)

    Examples
    --------
    Score only the asymptotic regime with a scale-free statistic::

        idx = esp_index(model, feedback, window=50, relative=True)

    See Also
    --------
    resdag.core.ESNModel.set_random_reservoir_states :
        Source of the standard-normal restart convention used here.
    """
    from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer

    device = feedback_seq.device
    dtype = feedback_seq.dtype
    batch_size, total_timesteps, _ = feedback_seq.shape

    if transient >= total_timesteps:
        raise ValueError(f"transient ({transient}) >= timesteps ({total_timesteps})")

    timesteps = total_timesteps - transient

    if window is not None and not (0 < window <= timesteps):
        raise ValueError(
            f"window ({window}) must be a positive integer no larger than the "
            f"number of post-transient timesteps ({timesteps})."
        )

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size ({chunk_size}) must be a positive integer or None.")

    inputs = (feedback_seq,) + driving_seqs

    # Save the caller's reservoir states *before* any probing/mutation so they
    # can be restored on the way out (including on error).  ``esp_index`` is a
    # read-only diagnostic; it must not leave the model's reservoirs in the
    # random state it uses internally.  ``_has_well_defined_esp`` and the base
    # orbit below both mutate state, so the snapshot has to be taken here first.
    # ``get_reservoir_states`` omits reservoirs whose state is ``None`` (never
    # warmed up), so those names are tracked separately and reset back to
    # ``None`` on restore instead of being left holding an internal state.
    saved_states = model.get_reservoir_states()
    reservoirs_with_none_state = [
        module
        for module in model.modules()
        if isinstance(module, BaseReservoirLayer) and module.state is None
    ]

    try:
        # Discover every reservoir layer (cell-agnostic): any
        # BaseReservoirLayer, not just ESNLayer.  This includes NGReservoirLayer and
        # any custom-cell layer.
        all_reservoirs: list[tuple[str, "BaseReservoirLayer"]] = []
        for name, module in model.named_modules():
            if isinstance(module, BaseReservoirLayer):
                all_reservoirs.append((name, module))

        if not all_reservoirs:
            raise ValueError("No reservoir layers found in model.")

        # Partition reservoirs into those with a well-defined ESP (a non-empty,
        # randomisable state) and those without.  A reservoir with no per-sample
        # state — e.g. a ``k=1`` NG-RC delay buffer of size 0 — has no notion of
        # "different initial states converging", so its ESP is undefined.
        reservoirs: list[tuple[str, "BaseReservoirLayer"]] = []
        for name, res in all_reservoirs:
            if _has_well_defined_esp(res, batch_size):
                reservoirs.append((name, res))
            elif verbose:
                label = name or type(res).__name__
                print(
                    f"Skipping reservoir '{label}': ESP is undefined "
                    f"(reservoir carries no randomisable state)."
                )

        if not reservoirs:
            raise ValueError(
                "No reservoir with a well-defined ESP found in model "
                "(every reservoir carries an empty state)."
            )

        # Run base orbit from the zero initial state.  Routed through the
        # public, validating, cloning API instead of assigning ``res.state``.
        for _, res in reservoirs:
            res.reset_state(batch_size=batch_size)

        base_states = _run_and_collect(model, reservoirs, inputs)
        if transient > 0:
            base_states = {name: s[:, transient:, :] for name, s in base_states.items()}

        # Evaluate all random restarts.  They are statistically independent and
        # differ only in their initial state, so they fold into the batch
        # dimension and run in one (or a few chunked) forward passes instead of
        # ``iterations`` sequential ones.
        random_states_by_iter = _collect_random_restarts(
            model,
            reservoirs,
            inputs,
            batch_size=batch_size,
            iterations=iterations,
            transient=transient,
            device=device,
            dtype=dtype,
            chunk_size=chunk_size,
            verbose=verbose,
        )

        esp_sums = {name: torch.tensor(0.0, device=device, dtype=dtype) for name, _ in reservoirs}
        if history:
            esp_history = {
                name: torch.zeros(iterations, timesteps, batch_size, device=device, dtype=dtype)
                for name, _ in reservoirs
            }

        # Reduce per restart.  Splitting the batched outputs back into
        # per-restart blocks and reducing them one at a time reproduces the
        # sequential arithmetic exactly (same per-iteration mean, then mean over
        # iterations), so ``window`` / ``relative`` / ``history`` behave
        # identically to the sequential path.
        for i in range(iterations):
            random_states = random_states_by_iter[i]
            for name, base in base_states.items():
                rand = random_states[name]
                # Distance at each (batch, timestep): norm over the feature
                # dimension.  Works for any output rank the reservoir produces.
                dist = torch.norm(base - rand, dim=-1)  # (batch, timesteps)

                if relative:
                    # Scale-free relative distance: divide by the base-state
                    # norm so the index is comparable across sizes / radii.
                    base_norm = torch.norm(base, dim=-1)  # (batch, timesteps)
                    dist = dist / (base_norm + 1e-12)

                # Mean distance for this restart.  With ``window`` set, average
                # only over the trailing window (the asymptotic regime);
                # otherwise over every post-transient timestep.
                dist_for_index = dist[:, -window:] if window is not None else dist
                esp_sums[name] += dist_for_index.mean()

                if history:
                    # Store the full post-transient sequence regardless of
                    # window: (batch, timesteps) -> row i of
                    # (iterations, timesteps, batch).
                    esp_history[name][i] = dist.T  # Transpose to (timesteps, batch)

        # Average over iterations
        esp_indices = {name: [val / iterations] for name, val in esp_sums.items()}

        if history:
            # Wrap history values in lists for API consistency
            esp_history_out = {name: [h] for name, h in esp_history.items()}
            return esp_indices, esp_history_out

        return esp_indices
    finally:
        # Restore the caller's reservoir states, whatever happened above.  Use
        # ``strict=False`` because reservoirs skipped as ESP-undefined (or any
        # that had no state to begin with) are simply absent from the snapshot.
        model.set_reservoir_states(saved_states, strict=False)
        # Reservoirs that started with a ``None`` state (never warmed up) are
        # absent from the snapshot, so ``set_reservoir_states`` would leave them
        # holding an internal state.  Reset them back to ``None`` to fully
        # restore the caller's pre-call condition.
        for module in reservoirs_with_none_state:
            module.reset_state()


def _draw_restart_states(
    reservoirs: list[tuple[str, "BaseReservoirLayer"]],
    batch_size: int,
    iterations: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, torch.Tensor]]:
    """
    Pre-draw every random restart state in the sequential RNG draw order.

    The sequential ESP loop draws, for each restart ``i``, one standard-normal
    state per reservoir *in reservoir order* via
    :meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.set_random_state`.
    Reproducing that exact nested order (restart-major, reservoir-minor) here —
    once, up front — makes the composed batched initial states **bit-identical**
    to the sequential ones, and decouples the RNG stream from how the forward
    passes are later chunked (chunking never changes which numbers are drawn).

    Parameters
    ----------
    reservoirs : list of (str, BaseReservoirLayer)
        Reservoirs with a well-defined ESP, in discovery order.
    batch_size : int
        Batch size of the driving sequence.
    iterations : int
        Number of random restarts to draw.
    device : torch.device
        Target device for the drawn states.
    dtype : torch.dtype
        Target dtype for the drawn states.

    Returns
    -------
    list of dict
        One dict per restart, mapping reservoir name to its cloned
        ``(batch_size, ...)`` standard-normal initial state.
    """
    per_iter: list[dict[str, torch.Tensor]] = []
    for _ in range(iterations):
        states: dict[str, torch.Tensor] = {}
        for name, res in reservoirs:
            # Route through the public API so each cell's state layout (2-D ESN
            # hidden state, 3-D NG-RC delay buffer, ...) is materialised and
            # validated correctly, with the same randn draw the sequential path
            # would make.
            res.set_random_state(batch_size=batch_size, device=device, dtype=dtype)
            state = res.get_state()
            assert state is not None  # set_random_state just materialised it
            states[name] = state
        per_iter.append(states)
    return per_iter


def _run_chunk(
    model: "ESNModel",
    reservoirs: list[tuple[str, "BaseReservoirLayer"]],
    inputs: tuple[torch.Tensor, ...],
    chunk_states: list[dict[str, torch.Tensor]],
    batch_size: int,
    transient: int,
) -> list[dict[str, torch.Tensor]]:
    """
    Evaluate a contiguous block of restarts in a single batched forward pass.

    The ``len(chunk_states)`` restarts are folded into the batch dimension: the
    inputs are tiled once per restart and each reservoir's per-restart states
    are concatenated into a single ``(chunk * batch_size, ...)`` tensor set via
    :meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.set_state`.
    One forward pass then produces every restart's output, which is reshaped
    back to ``(chunk, batch_size, timesteps, ...)`` and split per restart.

    Parameters
    ----------
    model : ESNModel
        Model to run.
    reservoirs : list of (str, BaseReservoirLayer)
        Reservoirs to hook and seed.
    inputs : tuple of torch.Tensor
        Driving sequences, each ``(batch_size, timesteps, features)``.
    chunk_states : list of dict
        Per-restart initial states (one dict per restart in this chunk).
    batch_size : int
        Batch size of a single restart.
    transient : int
        Leading timesteps to discard from each collected output.

    Returns
    -------
    list of dict
        One dict per restart in the chunk, mapping reservoir name to its
        post-transient output of shape ``(batch_size, timesteps, output_size)``.
    """
    chunk = len(chunk_states)

    # Compose the batched initial state per reservoir: block i is restart i.
    for name, res in reservoirs:
        stacked = torch.cat([chunk_states[i][name] for i in range(chunk)], dim=0)
        res.set_state(stacked)

    # Tile each driving sequence once per restart so every block sees the same
    # inputs the sequential path fed to that restart.
    tiled_inputs = tuple(x.repeat(chunk, *([1] * (x.dim() - 1))) for x in inputs)

    collected = _run_and_collect(model, reservoirs, tiled_inputs)

    out: list[dict[str, torch.Tensor]] = [{} for _ in range(chunk)]
    for name, batched in collected.items():
        # (chunk * batch_size, T, ...) -> (chunk, batch_size, T, ...)
        reshaped = batched.reshape(chunk, batch_size, *batched.shape[1:])
        if transient > 0:
            reshaped = reshaped[:, :, transient:, :]
        for i in range(chunk):
            out[i][name] = reshaped[i]
    return out


def _collect_random_restarts(
    model: "ESNModel",
    reservoirs: list[tuple[str, "BaseReservoirLayer"]],
    inputs: tuple[torch.Tensor, ...],
    batch_size: int,
    iterations: int,
    transient: int,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int | None,
    verbose: bool,
) -> list[dict[str, torch.Tensor]]:
    """
    Evaluate all random restarts, batching them into the batch dimension.

    All restart states are drawn once up front in the sequential RNG order (see
    :func:`_draw_restart_states`), then the forward passes are batched: with
    ``chunk_size`` restarts share each forward pass.  When ``chunk_size`` is
    ``None`` the whole ensemble runs in one pass, transparently retried with a
    halved chunk on a CUDA out-of-memory error — a large
    ``reservoir_size * iterations`` product can otherwise exhaust device memory.
    Because the states are pre-drawn, the chosen chunk size affects only memory
    and speed, never which random states are used.

    Parameters
    ----------
    model : ESNModel
        Model to run.
    reservoirs : list of (str, BaseReservoirLayer)
        Reservoirs with a well-defined ESP.
    inputs : tuple of torch.Tensor
        Driving sequences, each ``(batch_size, timesteps, features)``.
    batch_size : int
        Batch size of a single restart.
    iterations : int
        Number of random restarts.
    transient : int
        Leading timesteps discarded from each collected output.
    device : torch.device
        Device the restart states are drawn on.
    dtype : torch.dtype
        Dtype the restart states are drawn in.
    chunk_size : int or None
        Maximum restarts per batched forward pass; ``None`` runs all at once
        with an OOM-triggered fallback.
    verbose : bool
        Print per-chunk progress.

    Returns
    -------
    list of dict
        One dict per restart, mapping reservoir name to its post-transient
        output of shape ``(batch_size, timesteps, output_size)``.
    """
    restart_states = _draw_restart_states(reservoirs, batch_size, iterations, device, dtype)

    current_chunk = chunk_size if chunk_size is not None else iterations
    while True:
        current_chunk = max(1, min(current_chunk, iterations))
        try:
            results: list[dict[str, torch.Tensor]] = []
            for start in range(0, iterations, current_chunk):
                stop = min(start + current_chunk, iterations)
                if verbose:
                    print(f"\rRestarts {stop}/{iterations}", end="", flush=True)
                results.extend(
                    _run_chunk(
                        model,
                        reservoirs,
                        inputs,
                        restart_states[start:stop],
                        batch_size,
                        transient,
                    )
                )
            if verbose:
                print()
            return results
        except torch.cuda.OutOfMemoryError:
            # Only auto-shrink when the caller left the chunk size to us; an
            # explicit chunk_size is honoured and its OOM propagates.
            if chunk_size is not None or current_chunk <= 1:
                raise
            torch.cuda.empty_cache()
            current_chunk = max(1, current_chunk // 2)
            if verbose:
                print(f"\rCUDA OOM — retrying with chunk size {current_chunk}", flush=True)


def _has_well_defined_esp(reservoir: "BaseReservoirLayer", batch_size: int) -> bool:
    """
    Report whether a reservoir has a well-defined Echo State Property.

    The ESP index quantifies how quickly two trajectories started from
    *different* initial states converge under identical driving.  That is only
    meaningful when the reservoir actually carries a randomisable state: a
    reservoir whose materialised state has zero elements per sample (e.g. a
    ``k=1`` NG-RC reservoir, whose delay buffer has ``state_size == 0``) has no
    initial state to perturb, so its ESP is undefined.

    State is materialised through the public ``reset_state`` API and inspected
    without any direct attribute assignment, keeping the check cell-agnostic.
    The materialised state inherits the cell's own device/dtype, so no device
    or dtype argument is needed here.

    Parameters
    ----------
    reservoir : BaseReservoirLayer
        Reservoir whose ESP definedness is being probed.
    batch_size : int
        Batch size used to materialise the zero state.

    Returns
    -------
    bool
        ``True`` if the reservoir carries a non-empty per-sample state and its
        ESP is therefore well-defined; ``False`` otherwise.
    """
    reservoir.reset_state(batch_size=batch_size)
    state = reservoir.get_state()
    if state is None or state.shape[0] == 0:
        return False
    return state[0].numel() > 0


def _run_and_collect(
    model: "ESNModel",
    reservoirs: list[tuple[str, "BaseReservoirLayer"]],
    inputs: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    """
    Run model forward pass and collect reservoir output histories.

    Parameters
    ----------
    model : ESNModel
        Model to run.
    reservoirs : list of (str, BaseReservoirLayer)
        Reservoir layers to hook, paired with their module names.
    inputs : tuple of torch.Tensor
        Positional inputs forwarded to ``model``.

    Returns
    -------
    dict
        Mapping of layer name to its output tensor of shape
        ``(batch, timesteps, output_size)``.
    """
    collected: dict[str, torch.Tensor] = {}

    def make_hook(name: str):  # type: ignore[no-untyped-def]
        def hook(module: torch.nn.Module, inp: object, output: torch.Tensor) -> None:
            collected[name] = output

        return hook

    handles = []
    for name, res in reservoirs:
        h = res.register_forward_hook(make_hook(name))
        handles.append(h)

    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for h in handles:
            h.remove()

    return collected
