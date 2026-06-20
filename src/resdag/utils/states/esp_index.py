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
    verbose : bool, default=True
        Print progress and skip messages.

    Returns
    -------
    dict or tuple
        If ``history=False``: dict mapping layer names to single-element
        lists containing the ESP index scalar tensor.
        If ``history=True``: tuple of (ESP indices dict, history dict).
        History tensors have shape ``(iterations, timesteps, batch)``.

        Reservoirs whose ESP is undefined (no randomisable state) are omitted
        from both dictionaries.

    Raises
    ------
    ValueError
        If ``transient`` is not smaller than the number of timesteps, if the
        model contains no reservoir layers, or if no reservoir has a
        well-defined ESP.

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
    """
    from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer

    device = feedback_seq.device
    dtype = feedback_seq.dtype
    batch_size, total_timesteps, _ = feedback_seq.shape

    if transient >= total_timesteps:
        raise ValueError(f"transient ({transient}) >= timesteps ({total_timesteps})")

    timesteps = total_timesteps - transient
    inputs = (feedback_seq,) + driving_seqs

    # Discover every reservoir layer (cell-agnostic): any BaseReservoirLayer,
    # not just ESNLayer.  This includes NGReservoir and any custom-cell layer.
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

    # Run base orbit from the zero initial state.  Routed through the public,
    # validating, cloning API instead of assigning ``res.state`` directly.
    for _, res in reservoirs:
        res.reset_state(batch_size=batch_size)

    base_states = _run_and_collect(model, reservoirs, inputs)

    # Apply transient
    if transient > 0:
        base_states = {name: s[:, transient:, :] for name, s in base_states.items()}

    # Initialize accumulators
    esp_sums = {name: torch.tensor(0.0, device=device, dtype=dtype) for name, _ in reservoirs}

    if history:
        esp_history = {
            name: torch.zeros(iterations, timesteps, batch_size, device=device, dtype=dtype)
            for name, _ in reservoirs
        }

    # Run iterations with random initial states
    for i in range(iterations):
        if verbose:
            print(f"\rIteration {i + 1}/{iterations}", end="", flush=True)

        # Set random initial states through the public API so each cell's
        # state layout (2-D ESN hidden state, 3-D NG-RC delay buffer, ...) is
        # materialised and validated correctly.
        for _, res in reservoirs:
            res.set_random_state(batch_size=batch_size, device=device, dtype=dtype)

        # Run forward pass
        random_states = _run_and_collect(model, reservoirs, inputs)

        # Apply transient
        if transient > 0:
            random_states = {name: s[:, transient:, :] for name, s in random_states.items()}

        # Compute distances
        for name, base in base_states.items():
            rand = random_states[name]
            # Distance at each (batch, timestep): norm over the feature
            # dimension.  Works for any output rank produced by the reservoir.
            dist = torch.norm(base - rand, dim=-1)  # (batch, timesteps)

            # Mean distance for this iteration
            esp_sums[name] += dist.mean()

            if history:
                # Store: (batch, timesteps) -> row i of (iterations, timesteps, batch)
                esp_history[name][i] = dist.T  # Transpose to (timesteps, batch)

    if verbose:
        print()

    # Average over iterations
    esp_indices = {name: [val / iterations] for name, val in esp_sums.items()}

    if history:
        # Wrap history values in lists for API consistency
        esp_history_out = {name: [h] for name, h in esp_history.items()}
        return esp_indices, esp_history_out

    return esp_indices


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
