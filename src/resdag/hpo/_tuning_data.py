"""Leakage-safe, fixed data views for the common model tuning workflow.

Tensor targets are origin-aligned; trajectory targets carry physical clocks.
Only this boundary prepares clocks and score scales. Trials never recompute
validation windows or fit preprocessing on held-out observations.
"""

from __future__ import annotations

import copy
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from resdag._contracts import module_reference
from resdag.data._trajectory import is_trajectory, trajectory_type
from resdag.data._trajectory_ports import (
    PreparedTrajectoryPorts,
    _named_values,
    _read_port,
    _sample,
    has_trajectory_port,
    prepare_trajectory_port_drivers,
    prepare_trajectory_port_history,
    prepare_trajectory_ports,
)

if TYPE_CHECKING:
    from resdag.core.workflow import Model, Streams

TensorPorts = dict[str, torch.Tensor]


def _integer(value: Any, label: str) -> int:
    try:
        count = operator.index(value)
    except TypeError:
        raise ValueError(f"{label} must be a positive integer") from None
    if isinstance(value, (bool, np.bool_)) or count < 1:
        raise ValueError(f"{label} must be a positive integer")
    return count


def model_signature(model: Model) -> tuple[Any, ...]:
    """The fixed task interface; hidden widths and washout can vary per trial."""
    model._validate_ports()
    plan = model._feedback_plan()
    return (
        model.input_names,
        model.output_names,
        model._input_widths,
        model._output_widths,
        model._prediction_leads,
        tuple((route.output_index, route.input_index, route.delay) for route in plan.routes),
        model.resample_dt,
        model.resample_method,
    )


def _on_model(values: TensorPorts, model: Model) -> TensorPorts:
    """Give each fit/forecast its own tensors, including for in-place Torch layers."""
    reference = module_reference(model.graph)
    with torch.inference_mode(False):
        return {
            name: (
                value.detach().clone()
                if reference is None
                else value.detach().to(device=reference.device, dtype=reference.dtype).clone()
            )
            for name, value in values.items()
        }


def _layout(values: tuple[torch.Tensor, ...], names: tuple[str, ...], was_2d: bool) -> TensorPorts:
    return {name: value.squeeze(0) if was_2d else value for name, value in zip(names, values)}


def _time(value: torch.Tensor) -> int:
    return value.shape[-2]


def _slice(values: TensorPorts, start: int | None, stop: int | None) -> TensorPorts:
    return {name: value[..., start:stop, :] for name, value in values.items()}


@dataclass(frozen=True)
class PreparedFit:
    """Admitted labels and complete observations, in their original tensor layout."""

    inputs: TensorPorts
    targets: TensorPorts
    supervised_start: int
    supervised_stop: int
    implicit: bool
    trajectory_context: dict[str, Any] | None = None
    trajectory_system: tuple[Any, ...] | None = None

    def fit(self, model: Model) -> Model:
        return model._fit_prepared(
            _on_model(self.inputs, model),
            _on_model(self.targets, model),
            supervised_start=self.supervised_start,
            supervised_stop=self.supervised_stop,
            implicit=self.implicit,
            trajectory_context=copy.deepcopy(self.trajectory_context),
            trajectory_system=self.trajectory_system,
        )


@dataclass(frozen=True)
class ForecastOrigin:
    """One fixed forecast; discard ``skip`` unscored physical-clock preroll rows.

    ``horizon`` includes preroll, whereas truth/times contain scored rows only.
    Histories, predictions and truth preserve the input's 2-D/3-D layout.
    """

    history: TensorPorts
    future_inputs: TensorPorts | None
    truth: TensorPorts
    times: torch.Tensor
    horizon: int
    skip: int = 0

    def forecast(self, model: Model, **kwargs: Any) -> TensorPorts:
        output = model.forecast(
            _on_model(self.history, model),
            horizon=self.horizon,
            future_inputs=(
                None if self.future_inputs is None else _on_model(self.future_inputs, model)
            ),
            return_type="tensor",
            **kwargs,
        )
        values = output if isinstance(output, dict) else {model.output_names[0]: output}
        return {
            name: cast(torch.Tensor, values[name])[..., self.skip :, :]
            for name in model.output_names
        }


@dataclass(frozen=True)
class TuningData:
    """One immutable protocol shared by every candidate and reservoir seed."""

    train: PreparedFit
    refit: PreparedFit
    origins: tuple[ForecastOrigin, ...]
    scales: TensorPorts
    report: dict[str, Any]
    signature: tuple[Any, ...]

    def validate_model(self, model: Model) -> None:
        if model_signature(model) != self.signature:
            raise ValueError(
                "Tuning candidates must retain input/output names, widths, prediction leads, "
                "feedback connections and resampling settings"
            )

    def fit(self, model: Model, *, refit: bool = False) -> Model:
        self.validate_model(model)
        return (self.refit if refit else self.train).fit(model)

    def scales_for(self, model: Model) -> TensorPorts:
        return _on_model(self.scales, model)


def _score_scales(prepared: PreparedFit, washout: int) -> TensorPorts:
    start = max(washout, prepared.supervised_start) - prepared.supervised_start
    if start >= prepared.supervised_stop - prepared.supervised_start:
        raise ValueError("Training must contain washout plus at least one supervised sample")
    scales = {}
    for name, value in prepared.targets.items():
        admitted = value[..., start:, :].detach()
        if not bool(torch.isfinite(admitted).all()):
            raise ValueError("Admitted training targets must contain only finite values")
        calculation = (
            admitted.float() if admitted.dtype in (torch.float16, torch.bfloat16) else admitted
        )
        axes = tuple(range(calculation.ndim - 1))
        scale = calculation.std(dim=axes, correction=0)
        # Constant signals retain a meaningful original-unit error. Near-constant
        # signals have a documented dtype floor rather than division by zero.
        scale = torch.where(
            scale == 0, torch.ones_like(scale), scale.clamp_min(torch.finfo(scale.dtype).eps)
        )
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("Training score scales are not finite; rescale extreme input units")
        scales[name] = scale
    return scales


def _choose(indices: list[int], count: int) -> list[int]:
    count = min(count, len(indices))
    if count == 0:
        raise ValueError("Validation cannot support a complete forecast with the requested horizon")
    if count == 1:
        return [indices[0]]
    return [indices[(len(indices) - 1) * i // (count - 1)] for i in range(count)]


def _tensor_fit(model: Model, inputs: TensorPorts, labels: TensorPorts | None) -> PreparedFit:
    length = _time(next(iter(inputs.values())))
    stop = length - max(model._prediction_leads)
    if stop <= 0:
        raise ValueError("Training contains too few samples for its prediction leads")
    if labels is None:
        if (
            len(model.input_names) != 1
            or len(model.output_names) != 1
            or model._input_widths != model._output_widths
        ):
            raise ValueError(
                "Implicit fitting requires one input/output of equal width; provide named targets"
            )
        lead = model._prediction_leads[0]
        admitted = {model.output_names[0]: inputs[model.input_names[0]][..., lead : lead + stop, :]}
    else:
        admitted = _slice(labels, 0, stop)
    if any(not bool(torch.isfinite(value).all()) for value in admitted.values()):
        raise ValueError("Admitted training targets must contain only finite values")
    return PreparedFit(inputs, admitted, 0, stop, labels is None)


def _tensor_ports(model: Model, value: Any, *, targets: bool = False) -> TensorPorts:
    names, widths = (
        (model.output_names, model._output_widths)
        if targets
        else (model.input_names, model._input_widths)
    )
    streams, was_2d = model._streams(
        cast("Streams", value),
        names,
        widths,
        "targets" if targets else "training",
        finite=not targets,
    )
    return _layout(streams, names, was_2d)


def _tensor_pair(model: Model, inputs: Any, targets: Any) -> tuple[TensorPorts, TensorPorts | None]:
    values = _tensor_ports(model, inputs)
    labels = None if targets is None else _tensor_ports(model, targets, targets=True)
    reference = next(iter(values.values()))
    if labels is not None and any(
        value.ndim != reference.ndim or value.shape[:-1] != reference.shape[:-1]
        for value in labels.values()
    ):
        raise ValueError("Explicit targets must share training batch/time axes and layout")
    return values, labels


def _tensor_data(
    model: Model,
    training: Any,
    targets: Any,
    validation: tuple[Any, Any] | None,
    fraction: float,
    horizon: int | None,
    n_origins: int,
) -> tuple[PreparedFit, PreparedFit, tuple[ForecastOrigin, ...], dict[str, Any]]:
    inputs, labels = _tensor_pair(model, training, targets)
    n_train = _time(next(iter(inputs.values())))
    if validation is None:
        length = n_train
        cutoff = math.floor((1 - fraction) * length)
        fitting_inputs = _slice(inputs, 0, cutoff)
        fitting_labels = None if labels is None else _slice(labels, 0, cutoff)
        train = _tensor_fit(model, fitting_inputs, fitting_labels)
        refit = _tensor_fit(model, inputs, labels)
        first = cutoff
        combined, truth_labels = inputs, labels
    else:
        val_inputs, val_labels = _tensor_pair(model, *validation)
        if (labels is None) != (val_labels is None):
            raise ValueError(
                "Training and validation must both use implicit targets or both supply explicit targets"
            )
        sample, val_sample = next(iter(inputs.values())), next(iter(val_inputs.values()))
        if sample.ndim != val_sample.ndim or sample.shape[:-2] != val_sample.shape[:-2]:
            raise ValueError("Validation must match training batch axes and layout")
        combined = {
            name: torch.cat((inputs[name], val_inputs[name]), dim=-2) for name in model.input_names
        }
        truth_labels = (
            None
            if labels is None
            else {
                name: torch.cat((labels[name], cast(TensorPorts, val_labels)[name]), dim=-2)
                for name in model.output_names
            }
        )
        length, cutoff = _time(next(iter(combined.values()))), n_train
        train = refit = _tensor_fit(model, inputs, labels)
        # Origin-aligned validation labels cannot describe its first lead rows.
        # Avoid relying on unsupported future-label suffixes in training targets.
        first = cutoff + (max(model._prediction_leads) if labels is not None else 0)
    validation_span = length - first
    chosen_horizon = max(1, validation_span // 2) if horizon is None else horizon
    plan = model._feedback_plan()
    if cutoff < plan.required_history:
        raise ValueError(f"Training history needs at least {plan.required_history} samples")
    candidates = list(range(first, length - chosen_horizon + 1))
    origins = []
    records = []
    for index in _choose(candidates, n_origins):
        history = _slice(combined, 0, index)
        steps = plan.future_steps(chosen_horizon)
        future = (
            {
                model.input_names[i]: combined[model.input_names[i]][..., index : index + steps, :]
                for i in plan.driver_indices
            }
            if plan.driver_indices and steps
            else None
        )
        if truth_labels is None:
            truth = {
                model.output_names[0]: combined[model.input_names[0]][
                    ..., index : index + chosen_horizon, :
                ]
            }
        else:
            truth = {
                name: truth_labels[name][..., index - lead : index - lead + chosen_horizon, :]
                for name, lead in zip(model.output_names, model._prediction_leads)
            }
        if any(not bool(torch.isfinite(value).all()) for value in truth.values()):
            raise ValueError("Validation truth must contain only finite values")
        device = next(iter(combined.values())).device
        origins.append(
            ForecastOrigin(
                history,
                future,
                truth,
                torch.arange(index, index + chosen_horizon, device=device),
                chosen_horizon,
            )
        )
        records.append(
            {"requested_start": index, "last_observed": index - 1, "first_scored": index, "skip": 0}
        )
    report = {
        "source": "tensor",
        "training_observations": cutoff,
        "development_observations": n_train,
        "first_validation_index": first,
        "horizon": chosen_horizon,
        "origins": records,
    }
    return train, refit, tuple(origins), report


def _trajectory_kwargs(model: Model, dt: float | None) -> dict[str, Any]:
    return dict(
        input_names=model.input_names,
        input_widths=model._input_widths,
        output_names=model.output_names,
        output_widths=model._output_widths,
        dt=dt,
        method=model.resample_method,
    )


def _trajectory_fit(model: Model, prepared: PreparedTrajectoryPorts, implicit: bool) -> PreparedFit:
    reference = module_reference(model.graph)
    options: dict[str, Any] = (
        {} if reference is None else {"dtype": reference.dtype, "device": reference.device}
    )
    return PreparedFit(
        {
            name: torch.tensor(value, **options)
            for name, value in zip(model.input_names, prepared.inputs)
        },
        {
            name: torch.tensor(value, **options)
            for name, value in zip(model.output_names, prepared.targets)
        },
        prepared.supervised_start,
        prepared.supervised_stop,
        implicit,
        prepared.context,
        prepared.output_systems,
    )


def _trajectory_map(value: Any, names: tuple[str, ...], label: str) -> dict[str, Any]:
    return dict(zip(names, _named_values(value, names, label)))


def _raw_slice(
    values: Mapping[str, Any], cutoff: float, *, inclusive: bool = False
) -> dict[str, Any]:
    cls = trajectory_type()
    result = {}
    for name, source in values.items():
        count = int(np.searchsorted(source.t, cutoff, side="right" if inclusive else "left"))
        if count == 0:
            raise ValueError(f"Trajectory {name!r} has no observations before its cutoff")
        result[name] = cls(
            np.asarray(source.t)[:count].copy(),
            np.asarray(source.y)[:count].copy(),
            system=source.system,
            meta=copy.deepcopy(source.meta),
            variables=source.variables,
        )
    return result


def _raw_append(training: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    cls = trajectory_type()
    result = {}
    for name, source in training.items():
        later = validation[name]
        if tuple(source.variables) != tuple(later.variables):
            raise ValueError(f"Validation component names/order differ for input {name!r}")
        if not len(source.t) or not len(later.t) or source.t[-1] >= later.t[0]:
            raise ValueError(
                "Explicit trajectory validation inputs must occur strictly after training inputs"
            )
        result[name] = cls(
            np.concatenate((source.t, later.t)),
            np.concatenate((source.y, later.y)),
            system=source.system,
            meta=copy.deepcopy(source.meta),
            variables=source.variables,
        )
    return result


def _trajectory_data(
    model: Model,
    training: Any,
    targets: Any,
    validation: tuple[Any, Any] | None,
    fraction: float,
    horizon: int | None,
    n_origins: int,
) -> tuple[PreparedFit, PreparedFit, tuple[ForecastOrigin, ...], dict[str, Any]]:
    input_sources = _trajectory_map(training, model.input_names, "training")
    implicit = targets is None
    full = prepare_trajectory_ports(
        training,
        targets=targets,
        prediction_leads=model._prediction_leads,
        **_trajectory_kwargs(model, model.resample_dt),
    )
    dt = full.context["clock"]["dt"]
    refit = _trajectory_fit(model, full, implicit)
    if validation is None:
        grid = full.times
        cutoff = math.floor((1 - fraction) * len(grid))
        if not 0 < cutoff < len(grid):
            raise ValueError("Training and validation must each contain observed samples")
        first_time = float(grid[cutoff])
        fitting_sources = _raw_slice(input_sources, first_time)
        target_sources = (
            input_sources if implicit else _trajectory_map(targets, model.output_names, "targets")
        )
        fitting_targets = None if implicit else _raw_slice(target_sources, first_time)
        fitted = prepare_trajectory_ports(
            fitting_sources,
            targets=fitting_targets,
            prediction_leads=model._prediction_leads,
            **_trajectory_kwargs(model, dt),
        )
        train = _trajectory_fit(model, fitted, implicit)
        truth_sources = target_sources
        all_sources = input_sources
    else:
        val_inputs, val_targets = validation
        if implicit != (val_targets is None):
            raise ValueError(
                "Training and validation must both use implicit targets or both supply explicit targets"
            )
        val_sources = _trajectory_map(val_inputs, model.input_names, "validation")
        all_sources = _raw_append(input_sources, val_sources)
        eval_full = prepare_trajectory_port_history(
            all_sources, context=full.context, **_trajectory_kwargs(model, dt)
        )
        grid = eval_full.times
        first_time = max(float(source.t[0]) for source in val_sources.values())
        cutoff = int(np.searchsorted(grid, first_time))
        train = refit
        truth_sources = (
            val_sources
            if implicit
            else _trajectory_map(val_targets, model.output_names, "validation targets")
        )
    if implicit:
        if len(model.input_names) != 1 or len(model.output_names) != 1:
            raise ValueError("Implicit trajectory tuning requires one input/output")
        truth_sources = {model.output_names[0]: truth_sources[model.input_names[0]]}
    final_time = float(grid[-1])
    truth_ports = tuple(
        _read_port(truth_sources[name], width, f"validation[{name!r}]", cutoff=final_time)
        for name, width in zip(model.output_names, model._output_widths)
    )
    if validation is not None:
        assert train.trajectory_context is not None
        for name, port in zip(model.output_names, truth_ports):
            expected = train.trajectory_context["outputs"][name]["variables"]
            if port.variables != expected:
                raise ValueError(f"Validation target components/order differ for output {name!r}")
    input_ports = tuple(
        _read_port(all_sources[name], width, f"training[{name!r}]")
        for name, width in zip(model.input_names, model._input_widths)
    )
    chosen_horizon = max(1, (len(grid) - cutoff) // 2) if horizon is None else horizon
    plan = model._feedback_plan()
    valid, timing = [], {}
    tolerance = 1e-9 * dt
    for index in range(cutoff, len(grid)):
        requested = float(grid[index])
        counts = [int(np.searchsorted(port.times, requested, side="left")) for port in input_ports]
        if any(count == 0 for count in counts):
            continue
        last = min(float(port.times[count - 1]) for port, count in zip(input_ports, counts))
        start = max(float(port.times[0]) for port in input_ports)
        history_count = math.floor(float(np.nextafter((last - start) / dt, math.inf))) + 1
        if history_count < plan.required_history:
            continue
        skip = max(0, math.ceil((requested - last - tolerance) / dt) - 1)
        first = last + (skip + 1) * dt
        end = first + (chosen_horizon - 1) * dt
        if (
            first < requested - tolerance
            or end > final_time + tolerance
            or any(
                not len(port.times)
                or first < port.times[0] - tolerance
                or end > port.times[-1] + tolerance
                for port in truth_ports
            )
        ):
            continue
        steps = plan.future_steps(chosen_horizon + skip)
        if steps and any(
            last + dt < input_ports[i].times[0] - tolerance
            or last + steps * dt > input_ports[i].times[-1] + tolerance
            for i in plan.driver_indices
        ):
            continue
        valid.append(index)
        timing[index] = (requested, last, skip)
    origins, records = [], []
    reference = module_reference(model.graph)
    options: dict[str, Any] = (
        {} if reference is None else {"dtype": reference.dtype, "device": reference.device}
    )
    for index in _choose(valid, n_origins):
        requested, _, skip = timing[index]
        raw_history = _raw_slice(all_sources, requested)
        prepared = prepare_trajectory_port_history(
            raw_history, context=train.trajectory_context, **_trajectory_kwargs(model, dt)
        )
        last = float(prepared.times[-1])
        rollout = chosen_horizon + skip
        times = last + dt * np.arange(skip + 1, rollout + 1, dtype=np.float64)
        history = {
            name: torch.tensor(value, **options)
            for name, value in zip(model.input_names, prepared.inputs)
        }
        truth = {
            name: torch.tensor(
                _sample(port, times, dt, model.resample_method, f"validation[{name!r}]"), **options
            )
            for name, port in zip(model.output_names, truth_ports)
        }
        driver_names = tuple(model.input_names[i] for i in plan.driver_indices)
        steps = plan.future_steps(rollout)
        future = None
        if driver_names and steps:
            arrays = prepare_trajectory_port_drivers(
                {name: all_sources[name] for name in driver_names},
                input_names=driver_names,
                input_widths=tuple(model._input_widths[i] for i in plan.driver_indices),
                last_time=last,
                dt=dt,
                steps=steps,
                method=model.resample_method,
                input_records=prepared.context["inputs"],
            )
            future = {
                name: torch.tensor(value, **options) for name, value in zip(driver_names, arrays)
            }
        origins.append(
            ForecastOrigin(history, future, truth, torch.from_numpy(times.copy()), rollout, skip)
        )
        records.append(
            {
                "requested_start": requested,
                "last_observed": last,
                "first_scored": float(times[0]),
                "skip": skip,
            }
        )
    report = {
        "source": "trajectory",
        "training_observations": _time(next(iter(train.inputs.values()))),
        "development_observations": len(full.times),
        "first_validation_time": first_time,
        "dt": dt,
        "horizon": chosen_horizon,
        "origins": records,
    }
    return train, refit, tuple(origins), report


def prepare_tuning_data(
    model: Model,
    training: Any,
    targets: Any = None,
    *,
    validation: tuple[Any, Any] | None = None,
    validation_fraction: float = 0.2,
    horizon: int | None = None,
    n_origins: int = 3,
) -> TuningData:
    """Prepare one chronological protocol without fitting or mutating ``model``.

    Automatic validation uses the last 20% and refits the winner on all supplied
    development observations. Explicit ``validation=(inputs, targets_or_None)``
    is held out even during final refitting. Its Tensor targets use the same
    origin alignment as training; scoring begins only when all heads have
    target support wholly within validation.

    Explicit Tensor labels beyond the observed physical window are discarded
    for both candidate fits and final refitting, matching trajectory semantics.
    The report states discarded row counts and actual scored origins. Default
    horizon is half the held-out span; all origins use its complete length.
    Score scales use this prototype's fixed absolute washout, population standard
    deviation of admitted training labels, unit scale for constant channels and
    a dtype epsilon floor for nonconstant channels.
    """
    if (
        isinstance(validation_fraction, (bool, np.bool_))
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(validation_fraction)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("validation_fraction must be a finite number between zero and one")
    n_origins = _integer(n_origins, "n_origins")
    horizon = None if horizon is None else _integer(horizon, "horizon")
    if validation is not None and (not isinstance(validation, tuple) or len(validation) != 2):
        raise ValueError("validation must be (inputs, targets_or_None)")
    signature = model_signature(model)
    clocked = is_trajectory(training) or has_trajectory_port(training)
    if validation is not None and clocked != (
        is_trajectory(validation[0]) or has_trajectory_port(validation[0])
    ):
        raise TypeError("Training and validation must both be tensors or both be trajectories")
    builder = _trajectory_data if clocked else _tensor_data
    train, refit, origins, report = builder(
        model, training, targets, validation, float(validation_fraction), horizon, n_origins
    )
    scales = _score_scales(train, model.washout)
    report.update(
        {
            "automatic_validation": validation is None,
            "refit_includes_validation": validation is None,
            "score_washout": model.washout,
            "score_scale": "training population std; constant=1; nonconstant floor=dtype epsilon",
            "prediction_leads": dict(zip(model.output_names, model._prediction_leads)),
            "train_supervision": {"start": train.supervised_start, "stop": train.supervised_stop},
            "refit_supervision": {"start": refit.supervised_start, "stop": refit.supervised_stop},
            "train_discarded_origin_rows": _time(next(iter(train.inputs.values())))
            - (train.supervised_stop - train.supervised_start),
            "refit_discarded_target_rows": {
                name: _time(next(iter(refit.inputs.values())))
                - (refit.supervised_stop - refit.supervised_start)
                for name in model.output_names
            },
        }
    )
    return TuningData(train, refit, origins, scales, report, signature)
