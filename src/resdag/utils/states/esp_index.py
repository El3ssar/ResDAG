"""
Echo State Property Index Calculation
======================================

This module provides the :func:`esp_index` function for computing the Echo State
Property (ESP) index of reservoir layers in ESN models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import torch

if TYPE_CHECKING:
    from resdag.composition import ESNModel
    from resdag.layers import ESNLayer


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
    states converge when driven by the same input.

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
        Print progress.

    Returns
    -------
    dict or tuple
        If ``history=False``: dict mapping layer names to ESP index scalars.
        If ``history=True``: tuple of (ESP indices dict, history dict).
        History tensors have shape ``(iterations, timesteps, batch)``.

    Notes
    -----
    For LINEAR systems (identity activation), input does NOT affect ESP
    because the input contribution cancels out in the state difference.
    This is mathematically correct behavior.
    """
    from resdag.layers import ESNLayer

    device = feedback_seq.device
    dtype = feedback_seq.dtype
    batch_size, total_timesteps, _ = feedback_seq.shape

    if transient >= total_timesteps:
        raise ValueError(f"transient ({transient}) >= timesteps ({total_timesteps})")

    timesteps = total_timesteps - transient
    inputs = (feedback_seq,) + driving_seqs

    # Find all reservoir layers
    reservoirs = []
    for name, module in model.named_modules():
        if isinstance(module, ESNLayer):
            reservoirs.append((name, module))

    if not reservoirs:
        raise ValueError("No reservoir layers found in model.")

    # Run base orbit from zero initial state
    for _, res in reservoirs:
        res.state = torch.zeros(batch_size, res.reservoir_size, device=device, dtype=dtype)

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

        # Set random initial states
        for _, res in reservoirs:
            res.state = (
                torch.rand(batch_size, res.reservoir_size, device=device, dtype=dtype) * 2 - 1
            )

        # Run forward pass
        random_states = _run_and_collect(model, reservoirs, inputs)

        # Apply transient
        if transient > 0:
            random_states = {name: s[:, transient:, :] for name, s in random_states.items()}

        # Compute distances
        for name, base in base_states.items():
            rand = random_states[name]
            # Distance at each (batch, timestep): norm over state dimension
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


def _run_and_collect(
    model: "ESNModel",
    reservoirs: list[tuple[str, "ESNLayer"]],
    inputs: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    """
    Run model forward pass and collect reservoir state histories.

    Returns dict mapping layer name to state tensor (batch, timesteps, state_size).
    """
    collected = {}

    def make_hook(name: str):
        def hook(module, inp, output):
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
