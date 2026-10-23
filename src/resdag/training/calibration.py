"""Unsupervised ESN calibration on explicitly supplied training sequences.

Calibration precedes readout fitting. It never reads model validation data,
changes readout weights, or enables gradient descent on the reservoir.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from resdag.init.topology import estimate_spectral_radius
from resdag.layers.cells.esn_cell import ESNCell
from resdag.layers.reservoirs.esn import ESNLayer


@dataclass(frozen=True)
class CalibrationReport:
    """Small immutable summary of calibration, measured on the training inputs.

    ``steps`` counts parameter updates (batch-averaged, not individual samples).
    Activity statistics pool all neurons and samples after ``warmup`` and are
    descriptive; a Gaussian target or a requested radius is not a guarantee of
    task accuracy. Radius fields are populated only for flow control.
    """

    method: str
    epochs: int
    steps: int
    samples: int
    activity_mean_before: float
    activity_std_before: float
    activity_mean_after: float
    activity_std_after: float
    spectral_radius_before: float | None = None
    spectral_radius_after: float | None = None


@dataclass(frozen=True)
class _TrainingData:
    cell: ESNCell
    projected: Tensor
    initial_state: Tensor
    warmup: int
    epochs: int
    rate: float


def _positive_int(name: str, value: int, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'a non-negative' if zero else 'a positive'} integer")


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def _prepare(
    reservoir: ESNLayer,
    training_inputs: Tensor | Sequence[Tensor],
    driver: Tensor | None,
    warmup: int,
    epochs: int,
    learning_rate: float,
    min_gain: float,
    max_gain: float,
) -> _TrainingData:
    if not isinstance(reservoir, ESNLayer) or not isinstance(reservoir.cell, ESNCell):
        raise TypeError("calibration requires an ESNLayer")
    cell = reservoir.cell
    if cell.activation != "tanh":
        raise ValueError("calibration requires activation='tanh'")
    _positive_int("warmup", warmup, zero=True)
    _positive_int("epochs", epochs)
    rate = _finite("learning_rate", learning_rate, positive=True)
    lower = _finite("min_gain", min_gain, positive=True)
    upper = _finite("max_gain", max_gain, positive=True)
    if lower >= upper:
        raise ValueError("min_gain must be smaller than max_gain")
    if isinstance(training_inputs, Tensor):
        inputs = [training_inputs] if driver is None else [training_inputs, driver]
    else:
        if driver is not None:
            raise ValueError("supply driver separately only when training_inputs is a Tensor")
        inputs = list(training_inputs)
    expected = 2 if cell.input_size is not None else 1
    if len(inputs) != expected:
        raise ValueError(f"expected {expected} training input stream(s), got {len(inputs)}")
    prepared: list[Tensor] = []
    sizes = [cell.feedback_size] + ([cell.input_size] if expected == 2 else [])
    leading: tuple[int, int] | None = None
    rank: int | None = None
    for index, (value, size) in enumerate(zip(inputs, sizes, strict=True)):
        if not isinstance(value, Tensor):
            raise TypeError("training input streams must be torch.Tensor instances")
        if value.ndim not in (2, 3):
            raise ValueError(
                "training inputs must have shape (time, features) or (batch, time, features)"
            )
        if rank is not None and value.ndim != rank:
            raise ValueError("all training input streams must have the same rank")
        rank = value.ndim
        sequence = value.unsqueeze(0) if value.ndim == 2 else value
        if sequence.shape[-1] != size:
            raise ValueError(f"training stream {index} expects {size} features")
        shape = (sequence.shape[0], sequence.shape[1])
        if 0 in shape or (leading is not None and shape != leading):
            raise ValueError("training streams must have matching nonempty batch/time dimensions")
        if sequence.device != cell.weight_hh.device or sequence.dtype != cell.weight_hh.dtype:
            raise ValueError(
                "training inputs must match reservoir dtype and device; use .to(...) explicitly"
            )
        if not bool(torch.isfinite(sequence).all()):
            raise ValueError("training inputs must contain only finite values")
        leading = shape
        prepared.append(sequence.detach())
    assert leading is not None
    if warmup >= leading[1]:
        raise ValueError("warmup must leave at least one calibration timestep")
    if any(not bool(torch.isfinite(p).all()) for p in cell.parameters()):
        raise ValueError("reservoir parameters must be finite")
    compact_values = cell._buffers.get("_hh_values")
    if compact_values is not None and not bool(torch.isfinite(compact_values).all()):
        raise ValueError("reservoir parameters must be finite")
    for name in ("_intrinsic_gain", "_intrinsic_bias", "_recurrent_gain", "_initial_state"):
        value = getattr(cell, name)
        if value is not None and not bool(torch.isfinite(value).all()):
            raise ValueError("existing calibration and initial-state buffers must be finite")
    initial = cell.init_state(leading[0], cell.weight_hh.device, cell.weight_hh.dtype)
    return _TrainingData(
        cell, cell.project_inputs(prepared).detach(), initial, warmup, epochs, rate
    )


def _buffer(cell: ESNCell, name: str, default: float) -> Tensor:
    value = getattr(cell, name)
    return (
        cell.weight_feedback.new_full((cell.reservoir_size,), default)
        if value is None
        else value.clone()
    )


def _stats(
    data: _TrainingData, gain: Tensor, bias: Tensor, recurrent_gain: Tensor
) -> tuple[float, float]:
    # Merge centered block moments (Chan/Welford), never E[x²] - E[x]²:
    # the latter destroys small activity variances near tanh saturation.
    # CPU/CUDA accumulate in float64; MPS does not support that dtype. Centering
    # around an observed value also preserves low-variance precision on MPS.
    state = data.initial_state.clone()
    accumulator_dtype = torch.float32 if state.device.type == "mps" else torch.float64
    mean = torch.zeros((), device=state.device, dtype=accumulator_dtype)
    m2 = torch.zeros_like(mean)
    origin: Tensor | None = None
    pending: list[Tensor] = []
    count = 0
    for time in range(data.projected.shape[1]):
        raw = data.projected[:, time] + recurrent_gain * F.linear(state, data.cell.weight_hh)
        activated = torch.tanh(gain * raw + bias)
        state = data.cell._apply_leak(state, activated)
        if time < data.warmup:
            continue
        if origin is None:
            origin = state.reshape(-1)[0].to(accumulator_dtype).clone()
        pending.append(state)
        # Fixed-size blocks bound memory and amortize accelerator reductions.
        if len(pending) == 64 or time == data.projected.shape[1] - 1:
            values = torch.stack(pending).to(accumulator_dtype) - origin
            block_variance, block_mean = torch.var_mean(values, correction=0)
            block_count = values.numel()
            combined_count = count + block_count
            delta = block_mean - mean
            m2 = (
                m2
                + block_variance * block_count
                + delta.square() * (count * block_count / combined_count)
            )
            mean = mean + delta * (block_count / combined_count)
            count = combined_count
            pending.clear()
    assert origin is not None
    variance = (m2 / count).clamp_min(0)
    if not bool(torch.isfinite(origin) & torch.isfinite(mean) & torch.isfinite(variance)):
        raise RuntimeError(
            "calibration produced non-finite activity; reduce learning_rate or input scale"
        )
    # Add the origin in Python double precision, even for MPS accumulation.
    return float(origin) + float(mean), float(variance.sqrt())


def _report(
    data: _TrainingData,
    method: str,
    before: tuple[float, float],
    after: tuple[float, float],
    radius_before: float | None = None,
    radius_after: float | None = None,
) -> CalibrationReport:
    steps = data.epochs * (data.projected.shape[1] - data.warmup)
    return CalibrationReport(
        method,
        data.epochs,
        steps,
        steps * data.projected.shape[0],
        *before,
        *after,
        radius_before,
        radius_after,
    )


@torch.no_grad()
def calibrate_intrinsic_plasticity(
    reservoir: ESNLayer,
    training_inputs: Tensor | Sequence[Tensor],
    *,
    driver: Tensor | None = None,
    target_mean: float = 0.0,
    target_std: float = 0.1,
    learning_rate: float = 1e-3,
    epochs: int = 1,
    warmup: int = 0,
    min_gain: float = 1e-3,
    max_gain: float = 100.0,
) -> CalibrationReport:
    r"""Adapt tanh gains and offsets before fitting a readout.

    Pass only training data, shaped ``(T, F)`` or ``(B, T, F)``. Supply a driver
    with ``driver=...`` or pass ``(feedback, driver)``. Each batch item starts
    from the configured initial state each epoch; ``warmup`` evolves the state
    without adapting. Noise is not applied, and there is no random sampling.
    Repeating the call on identically initialized layers is deterministic.

    This is Gaussian-target intrinsic plasticity, with neuron output
    ``y=tanh(a*x+b)`` and the stochastic updates

    .. math::
        \Delta b=-\eta[-\mu/\sigma^2+(y/\sigma^2)
          (2\sigma^2+1-y^2+\mu y)],\quad
        \Delta a=\eta/a+\Delta b\,x.

    For batched inputs updates are averaged over batch items. Gain clipping to
    ``[min_gain, max_gain]`` bounds unsafe updates. The target is approximate:
    bounded tanh outputs cannot realize an arbitrary Gaussian distribution.
    Reservoir weights remain fixed. Changes are committed only after a finite
    rollout succeeds; failure leaves weights, buffers, mode and state intact.
    Success clears the old recurrent state so the new dynamics start cleanly.

    Reference: Verstraeten, Schrauwen & Stroobandt, *Adapting reservoirs to get
    Gaussian distributions*, ESANN 2007, equations 10–11.
    """
    mean = _finite("target_mean", target_mean)
    std = _finite("target_std", target_std, positive=True)
    if abs(mean) >= 1 or std * std >= 1 - mean * mean:
        raise ValueError(
            "target_mean/target_std must describe feasible moments inside tanh's (-1, 1) range"
        )
    data = _prepare(
        reservoir, training_inputs, driver, warmup, epochs, learning_rate, min_gain, max_gain
    )
    if std * std < torch.finfo(data.cell.weight_hh.dtype).tiny:
        raise ValueError("target_std is too small for the reservoir dtype")
    gain = _buffer(data.cell, "_intrinsic_gain", 1.0)
    bias = _buffer(data.cell, "_intrinsic_bias", 0.0)
    recurrent_gain = _buffer(data.cell, "_recurrent_gain", 1.0)
    if not bool((gain > 0).all()):
        raise ValueError("intrinsic gains must be positive")
    before = _stats(data, gain, bias, recurrent_gain)
    finite = gain.new_tensor(True, dtype=torch.bool)
    variance = std * std
    for _ in range(epochs):
        state = data.initial_state.clone()
        for time in range(data.projected.shape[1]):
            raw = data.projected[:, time] + recurrent_gain * F.linear(state, data.cell.weight_hh)
            output = torch.tanh(gain * raw + bias)
            state = data.cell._apply_leak(state, output)
            if time < warmup:
                continue
            delta_bias = -data.rate * (
                -mean / variance
                + output / variance * (2 * variance + 1 - output.square() + mean * output)
            )
            next_gain = gain + data.rate / gain + (delta_bias * raw).mean(dim=0)
            next_bias = bias + delta_bias.mean(dim=0)
            finite &= torch.isfinite(next_gain).all() & torch.isfinite(next_bias).all()
            gain = next_gain.clamp(min_gain, max_gain)
            bias = next_bias
    if not bool(finite):
        raise RuntimeError("intrinsic plasticity diverged; reduce learning_rate or input scale")
    after = _stats(data, gain, bias, recurrent_gain)
    report = _report(data, "intrinsic_plasticity", before, after)
    data.cell._intrinsic_gain = gain.detach()
    data.cell._intrinsic_bias = bias.detach()
    data.cell._has_intrinsic = True
    reservoir.reset_state()
    return report


@torch.no_grad()
def calibrate_flow_control(
    reservoir: ESNLayer,
    training_inputs: Tensor | Sequence[Tensor],
    *,
    driver: Tensor | None = None,
    target_radius: float = 0.9,
    learning_rate: float = 1e-3,
    epochs: int = 1,
    warmup: int = 0,
    mode: Literal["local", "global"] = "local",
    min_gain: float = 1e-3,
    max_gain: float = 100.0,
) -> CalibrationReport:
    r"""Adapt recurrent row gains using local or global flow control.

    ``a <- a * (1 + learning_rate * (target_radius**2 * h_prev**2 - x_rec**2))``
    where ``x_rec=a*(W@h_prev)``. Updates average over batch items; ``global``
    also averages over neurons, applying a shared multiplicative update.
    Input/feedback weights and bias are fixed. Gain clipping supplies numerical
    bounds. No eigenvalue calculation occurs in the adaptation loop; measured
    matrix radii are reported before/after as diagnostics.

    This rule assumes unit-leak tanh dynamics with no intrinsic-plasticity
    transform. Unsupported combinations fail before mutation. Local control is
    approximate and can miss the requested radius for strongly correlated
    inputs or structured weights. A target radius is not a stability guarantee.
    Input shapes, warmup, state cleanup and transactional behavior match
    :func:`calibrate_intrinsic_plasticity`.

    Reference: Schubert & Gros (2021), *Local homeostatic regulation of the
    spectral radius of echo-state networks*, equations 5–6,
    https://arxiv.org/abs/2101.10665.
    """
    radius = _finite("target_radius", target_radius, positive=True)
    if mode not in ("local", "global"):
        raise ValueError("mode must be 'local' or 'global'")
    data = _prepare(
        reservoir, training_inputs, driver, warmup, epochs, learning_rate, min_gain, max_gain
    )
    leak = torch.as_tensor(data.cell.leak_rate)
    if not bool((leak == 1).all()):
        raise ValueError("flow control requires leak_rate=1")
    if data.cell._intrinsic_gain is not None:
        raise ValueError(
            "apply flow control before intrinsic plasticity, using an unadapted tanh activation"
        )
    weights = data.cell.weight_hh
    stored_values = weights.values() if weights.layout == torch.sparse_csr else weights
    if not bool((stored_values != 0).any()):
        raise ValueError("flow control cannot adapt an all-zero recurrent matrix")
    gain = _buffer(data.cell, "_recurrent_gain", 1.0)
    if not bool((gain > 0).all()):
        raise ValueError("recurrent gains must be positive")
    ones = gain.new_ones(gain.shape)
    zeros = gain.new_zeros(gain.shape)
    before = _stats(data, ones, zeros, gain)
    from resdag.layers.cells._compact_recurrent import _scale_recurrent_rows

    radius_before = estimate_spectral_radius(_scale_recurrent_rows(data.cell.weight_hh, gain))
    finite = gain.new_tensor(True, dtype=torch.bool)
    for _ in range(epochs):
        state = data.initial_state.clone()
        for time in range(data.projected.shape[1]):
            recurrent = gain * F.linear(state, data.cell.weight_hh)
            output = torch.tanh(data.projected[:, time] + recurrent)
            if time >= warmup:
                delta = (radius * radius * state.square() - recurrent.square()).mean(dim=0)
                if mode == "global":
                    delta = delta.mean()
                next_gain = gain * (1 + data.rate * delta)
                finite &= torch.isfinite(next_gain).all()
                gain = next_gain.clamp(min_gain, max_gain)
            state = output
    if not bool(finite):
        raise RuntimeError("flow control diverged; reduce learning_rate or input scale")
    after = _stats(data, ones, zeros, gain)
    radius_after = estimate_spectral_radius(_scale_recurrent_rows(data.cell.weight_hh, gain))
    report = _report(data, "flow_control", before, after, radius_before, radius_after)
    data.cell._recurrent_gain = gain.detach()
    data.cell._has_recurrent_gain = True
    reservoir.reset_state()
    return report
