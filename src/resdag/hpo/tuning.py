"""Common, transactional tuning of fitted PyTorch reservoir models.

Optuna owns sampling and trial bookkeeping; joblib owns parallel processes.
The data adapter owns immutable chronological views. Candidates use the same
fit/forecast workflow as the final model, including named ports and units.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import platform
import random
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np
import optuna
import torch
from torch import nn

from resdag._contracts import module_reference
from resdag.core._hooks import forecast_as
from resdag.core.model import ESNModel
from resdag.core.workflow import Model, _washout
from resdag.data._trajectory import is_trajectory
from resdag.facade import ESN
from resdag.layers.readouts import ReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer
from resdag.models._recipes import (
    builtin_recipe,
    erdos_renyi_options,
    input_scaling_supported,
    sampled_options,
    workflow_options,
)

from ._parallel import run_jobs
from ._provenance import bind_protocol, class_identity, resolve_trial_settings
from ._provenance import callable_identity as _callable_identity
from ._provenance import canonical_config as _configuration
from ._trial_progress import ProgressCallback, TrialProgress
from ._tuning_data import TuningData, prepare_tuning_data
from ._tuning_scores import ScoreResult, StreamingForecastScore
from .pruners import resolve_pruner
from .resources import (
    MemoryBudget,
    MemoryContext,
    MemoryEstimate,
    Screening,
    _memory_context,
    estimate_memory,
)
from .storage import _close_storage, resolve_storage

__all__ = ["ScoreContext", "PrefixScoreContext", "TuningReport", "tune_model"]

Factory = Callable[[dict[str, Any], int], Model | ESN]
SearchSpace = Callable[[optuna.Trial], Mapping[str, Any]]
Preset = Literal["quick", "balanced", "thorough"]
_PRESETS = {"quick": (8, 1, 1), "balanced": (24, 3, 3), "thorough": (80, 5, 5)}


@dataclass(frozen=True)
class ScoreContext:
    """Original-unit tensors and fixed training scales for a custom objective.

    Return one finite scalar per seed/origin. The trial score is their median.
    Set ``direction='minimize'`` when returning an error rather than a reward.
    """

    predictions: dict[str, torch.Tensor]
    truth: dict[str, torch.Tensor]
    scales: dict[str, torch.Tensor]
    seed: int
    origin: int


@dataclass(frozen=True)
class PrefixScoreContext(ScoreContext):
    """Explicitly prefix-safe pruning objective with the complete target horizon.

    Predictions and truth contain only the admitted prefix. This callback
    controls pruning, while ``objective`` still scores the complete forecast.
    Keep its direction and scale comparable across matching resource steps.
    """

    completed_steps: int
    horizon: int


@dataclass(frozen=True)
class TuningReport:
    """Winning configuration, full Optuna study and reproducible validation facts."""

    study: optuna.Study
    best_params: dict[str, Any]
    best_trial: int
    score: float
    nrmse: float
    seeds: tuple[int, ...]
    refit_seed: int
    validation: dict[str, Any]
    protocol_hash: str
    attempts: int
    elapsed_seconds: float
    refit_seconds: float
    runs: tuple[dict[str, Any], ...]
    _template: Model | ESN = field(repr=False, compare=False)
    _installed_hash: str = field(repr=False, compare=False)
    resources: dict[str, Any] = field(default_factory=dict)


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _workflow(model: Model | ESN) -> Model:
    if isinstance(model, Model):
        return model
    if isinstance(model, ESN) and model.workflow is not None:
        return model.workflow
    raise TypeError("A candidate recipe must return a built rd.Model or rd.ESN")


def _synchronize(model: Model) -> None:
    reference = module_reference(model.graph)
    if reference is not None and reference.device.type == "cuda":
        torch.cuda.synchronize(reference.device)
    elif reference is not None and reference.device.type == "mps":
        torch.mps.synchronize()


def _template(model: Model | ESN, training: Any, seed: int) -> Model | ESN:
    report = model.__dict__.get("tuning_report")
    memo: dict[int, Any] = {} if report is None else {id(report): None}
    memo.update(
        {
            id(module.state): None
            for module in model.modules()
            if isinstance(module, BaseReservoirLayer) and module.state is not None
        }
    )
    with torch.inference_mode(False):
        template = copy.deepcopy(model, memo)
    template.__dict__.pop("tuning_report", None)
    if isinstance(template, ESN) and not template.is_built:
        sample = training.get("input") if isinstance(training, Mapping) else training
        if is_trajectory(sample):
            sample = torch.as_tensor(cast(Any, sample).y)
        if not isinstance(sample, torch.Tensor) or sample.ndim not in (2, 3):
            raise TypeError("ESN tuning requires a floating time-series tensor or trajectory")
        device = template._reference.device if template._device_pinned else sample.device
        with _isolated_seed(seed, device):
            template.build(
                sample.shape[-1],
                device=device,
                dtype=template._reference.dtype if template._dtype_pinned else sample.dtype,
            )
    return template


@contextmanager
def _isolated_seed(seed: int, device: torch.device) -> Iterator[None]:
    """Restore Python, NumPy and initialized Torch streams even on failed trials."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index not in cuda_devices:
            cuda_devices.append(index)
    try:
        with ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=cuda_devices))
            if torch.backends.mps.is_available():
                stack.enter_context(torch.random.fork_rng(devices=[0], device_type="mps"))
                torch.mps.manual_seed(seed)
            torch.default_generator.manual_seed(seed)
            for index in cuda_devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _automatic_options(model: Model | ESN) -> dict[str, Any] | None:
    if type(model) not in (Model, ESN):
        return None
    if isinstance(model, ESN) and (
        set(model._modules)
        - {"workflow", "_pending_input_transforms", "_pending_output_transforms"}
        or model._parameters
        or "forward" in model.__dict__
        or set(model._buffers) - {"_reference"}
        or any(
            getattr(model, key)
            for key in (
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
            )
        )
    ):
        return None
    if isinstance(model, ESN) and not model.is_built:
        return model._factory_options
    workflow = _workflow(model)
    if type(workflow) is not Model or set(workflow._modules) != {
        "graph",
        "input_scalers",
        "output_scalers",
    }:
        return None
    if workflow._parameters or "forward" in workflow.__dict__:
        return None
    if any(
        name != "_fitted"
        and not name.startswith(
            ("_history_", "_input_transform_sample_", "_output_transform_sample_")
        )
        for name in workflow._buffers
    ):
        return None
    if any(
        getattr(workflow, key)
        for key in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
    ):
        return None
    recipe = builtin_recipe(workflow.graph)
    return None if recipe is None else recipe["options"]


def _preset_recipe(template: Model | ESN) -> Factory:
    reference = module_reference(template.graph)
    assert reference is not None
    metadata = builtin_recipe(template.graph)
    assert metadata is not None
    options = copy.deepcopy(metadata["options"])
    options.pop("seed", None)
    settings_template = workflow_options(_workflow(template))
    device, dtype = reference.device, reference.dtype
    modes = {name: module.training for name, module in template.named_modules()}

    def build(params: dict[str, Any], seed: int) -> Model | ESN:
        settings = copy.deepcopy(settings_template)
        graph_params = {}
        for key, value in params.items():
            if key in settings:
                settings[key] = value
            else:
                graph_params[key] = value
        supplied = sampled_options(options, graph_params)
        supplied.update(device=device, dtype=dtype, seed=seed)
        if isinstance(template, ESN):
            supplied["alpha"] = supplied.pop("readout_alpha")
            for name in ("feedback_size", "output_size", "input_size"):
                supplied.pop(name, None)
            settings.pop("input_names")
            settings.pop("output_names")
            candidate: Model | ESN = ESN(input_size=template.input_size, **settings, **supplied)
        else:
            from resdag.models import classic_esn, ott_esn

            factory = {"classic": classic_esn, "ott": ott_esn}[metadata["family"]]
            candidate = Model(graph=factory(**supplied), **settings)
        candidate.train(template.training)
        for name, module in candidate.named_modules():
            if name in modes:
                module.training = modes[name]
        return candidate

    return build


def _fixed_recipe(template: Model | ESN) -> Factory:
    def build(params: dict[str, Any], seed: int) -> Model | ESN:
        candidate = copy.deepcopy(template)
        for key, value in params.items():
            if key == "washout":
                _workflow(candidate).washout = _washout(value)
            elif key == "alpha":
                heads = [
                    module
                    for module in candidate.graph.modules()
                    if isinstance(module, ReadoutLayer)
                ]
                if not heads or any(not hasattr(head, "alpha") for head in heads):
                    raise ValueError(
                        "alpha tuning requires readouts with an alpha setting; provide a factory"
                    )
                if not math.isfinite(value) or value < 0:
                    raise ValueError("alpha must be finite and non-negative")
                for head in heads:
                    head.alpha = value
            else:
                raise ValueError(
                    f"Fixed-DAG tuning supports alpha and washout, got {key!r}. "
                    "Supply factory(params, seed) to customize other settings or architecture."
                )
        return candidate

    return build


def _default_space(template: Model | ESN) -> SearchSpace:
    options = _automatic_options(template)
    recipe = builtin_recipe(_workflow(template).graph) if options is not None else None
    tune_input_to_readout = recipe is not None and recipe["family"] == "classic"
    heads = [
        module for module in _workflow(template).graph.modules() if isinstance(module, ReadoutLayer)
    ]
    tune_alpha = bool(heads) and all(hasattr(head, "alpha") for head in heads)
    if options is None and not tune_alpha:
        raise ValueError(
            "No default tunable readout alpha; provide search_space and, for custom settings, factory"
        )

    def space(trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = (
            {"alpha": trial.suggest_float("alpha", 1e-12, 1e-2, log=True)} if tune_alpha else {}
        )
        if options is not None:
            size = options["reservoir_size"]
            params.update(
                reservoir_size=trial.suggest_int(
                    "reservoir_size", max(8, size // 2), max(8, size * 2)
                ),
                spectral_radius=trial.suggest_float("spectral_radius", 0.05, 1.5),
                leak_rate=trial.suggest_float("leak_rate", 0.05, 1.0),
            )
            if tune_input_to_readout:
                params["input_to_readout"] = trial.suggest_categorical(
                    "input_to_readout", [True, False]
                )
            if input_scaling_supported(options):
                params["input_scaling"] = trial.suggest_float("input_scaling", 1e-3, 2.0, log=True)
            if options.get("bias", True) and options.get("bias_initializer") is None:
                params["bias_scaling"] = trial.suggest_float("bias_scaling", 1e-3, 1.0, log=True)
            topology = erdos_renyi_options(options)
            if topology is not None:
                available = params["reservoir_size"] - (0 if topology[2] else 1)
                params["mean_degree"] = trial.suggest_int("mean_degree", 3, min(12, available))
        return params

    return space


def _module_configs(model: Model) -> Any:
    module_keys = set(nn.Module().__dict__)
    runtime = {
        "_compiled_call_impl",
        "_compiled_scan_cache",
        "_state_user_set",
        "_noise_generators",
        "_sparse_weight_key",
        "_sparse_active",
        "_hh_structure_key",
        "_fitted_flag",
        "last_solve",
    }
    configurations: list[tuple[Any, ...]] = [("workflow", class_identity(type(model), set()))]
    implementations: dict[type[nn.Module], Any] = {}
    for group, root in (
        ("graph", model.graph),
        ("input", model.input_scalers),
        ("output", model.output_scalers),
    ):
        for name, module in root.named_modules():
            cls = type(module)
            if cls not in implementations:
                implementations[cls] = class_identity(cls, set())
            if module is model.graph:
                configurations.append((group, name, implementations[cls]))
                continue
            attributes = {
                key: value
                for key, value in vars(module).items()
                if key not in module_keys and key not in runtime
            }
            configurations.append(
                (
                    group,
                    name,
                    implementations[cls],
                    _configuration(attributes),
                )
            )
    return configurations


def _recipe_identity(template: Model | ESN) -> Any:
    workflow = _workflow(template)
    readouts = tuple(
        name for name, module in workflow.graph.named_modules() if isinstance(module, ReadoutLayer)
    )
    feature_state = {
        name: value
        for name, value in workflow.graph.state_dict().items()
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in readouts)
    }
    return (
        class_identity(type(template), set()),
        (
            _configuration(template._factory_options)
            if isinstance(template, ESN)
            else _configuration(feature_state)
        ),
        _module_configs(workflow),
        workflow.preprocessing,
        workflow.washout,
        workflow._transform_schema,
        _graph_structure(workflow),
    )


def _graph_structure(workflow: Model) -> Any:
    graph = workflow.graph
    if isinstance(graph, ESNModel):
        plan = graph._get_execution_plan(require_explicit_state=False)
        return (
            "symbolic",
            plan.input_features,
            plan.output_slots,
            tuple(
                (operation.name, operation.parents, operation.outputs)
                for operation in plan.operations
            ),
        )
    return (type(graph).__module__, type(graph).__qualname__, workflow._backend_schema)


def _protocol(data: TuningData, template: Model | ESN, **settings: Any) -> tuple[str, Any]:
    payload = {
        "version": 2,
        "contract": "resdag.model.autonomous",
        "views": _configuration(data),
        "model": _recipe_identity(template),
        "settings": settings,
        "environment": (platform.python_version(), torch.__version__, optuna.__version__),
    }
    encoded = repr(_configuration(payload))
    return hashlib.sha256(encoded.encode()).hexdigest(), payload


@dataclass
class _Evaluation:
    number: int
    params: dict[str, Any]
    score: float | None
    nrmse: float | None
    runs: tuple[dict[str, Any], ...]
    error: str | None = None
    pruned: bool = False
    pruning_reason: str | None = None
    intermediates: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class _TuningJob:
    number: int
    params: dict[str, Any]
    progress: ProgressCallback | None = None
    stage: Literal["full", "screen"] = "full"


@dataclass
class _Evaluator:
    factory: Factory
    data: TuningData
    seeds: tuple[int, ...]
    threshold: float
    aggregation: Literal["weakest", "mean"]
    objective: Callable[[ScoreContext], float] | None
    catch_exceptions: bool
    device: torch.device
    execution: Literal["auto", "fast", "history"]
    checkpoint_interval: int = 64
    pruning_objective: Callable[[PrefixScoreContext], float] | None = None

    def build(self, params: dict[str, Any], seed: int) -> Model | ESN:
        with torch.inference_mode(False):
            supplied = self.factory(copy.deepcopy(params), seed)
        if not isinstance(supplied, (Model, ESN)):
            raise TypeError("A candidate recipe must return rd.Model or rd.ESN")
        return _template(supplied, self.data.train.inputs, seed)

    def __call__(self, job: _TuningJob) -> _Evaluation:
        number, params = job.number, job.params
        runs: list[dict[str, Any]] = []
        intermediates: list[tuple[int, float]] = []
        current: dict[str, Any] | None = None
        try:
            for seed in self.seeds:
                with _isolated_seed(seed, self.device):
                    candidate = self.build(params, seed)
                    workflow = _workflow(candidate)
                    self.data.validate_model(workflow)
                    started = time.perf_counter()
                    self.data.fit(workflow)
                    _synchronize(workflow)
                    fit_seconds = time.perf_counter() - started
                    scales = self.data.scales_for(workflow)
                    for index, origin in enumerate(self.data.origins):
                        started = time.perf_counter()
                        horizon = origin.horizon - origin.skip
                        case = len(runs)
                        current = {
                            "seed": seed,
                            "origin": index,
                            "complete": False,
                            "completed_steps": 0,
                            "fit_seconds": fit_seconds if index == 0 else 0.0,
                        }
                        execution = self.execution
                        if execution == "auto":
                            execution = (
                                "history" if workflow._requires_complete_history() else "fast"
                            )
                        accumulator = StreamingForecastScore(
                            scales,
                            threshold=self.threshold,
                            aggregation=self.aggregation,
                        )
                        collected: dict[str, list[torch.Tensor]] = {
                            name: [] for name in workflow.output_names
                        }
                        truth = None
                        metrics: ScoreResult | None = None
                        with (
                            forecast_as(candidate),
                            origin.stream(
                                workflow,
                                chunk_size=self.checkpoint_interval,
                                execution=execution,
                                checkpoints=(
                                    tuple(
                                        range(origin.skip, origin.horizon, self.checkpoint_interval)
                                    )
                                    if origin.skip
                                    else None
                                ),
                            ) as chunks,
                        ):
                            for chunk in chunks:
                                predicted = chunk.named_predictions
                                if truth is None:
                                    truth = {
                                        name: value.to(predicted[name])
                                        for name, value in origin.truth.items()
                                    }
                                targets = {
                                    name: value[..., chunk.start : chunk.stop, :]
                                    for name, value in truth.items()
                                }
                                accumulator.update(predicted, targets)
                                current["completed_steps"] = accumulator.steps
                                if self.objective is not None or self.pruning_objective is not None:
                                    for name, value in predicted.items():
                                        collected[name].append(value.detach().clone())
                                if job.progress is not None and (
                                    self.objective is None or self.pruning_objective is not None
                                ):
                                    metrics = accumulator.summary()
                                    partial_score = metrics.score
                                    if self.pruning_objective is not None:
                                        partial_score = float(
                                            self.pruning_objective(
                                                PrefixScoreContext(
                                                    {
                                                        name: torch.cat(values, dim=-2)
                                                        for name, values in collected.items()
                                                    },
                                                    {
                                                        name: value[
                                                            ..., : accumulator.steps, :
                                                        ].clone()
                                                        for name, value in truth.items()
                                                    },
                                                    {
                                                        name: value.clone()
                                                        for name, value in scales.items()
                                                    },
                                                    seed,
                                                    index,
                                                    accumulator.steps,
                                                    horizon,
                                                )
                                            )
                                        )
                                    if not math.isfinite(partial_score):
                                        raise ValueError(
                                            "The pruning objective must return a finite scalar"
                                        )
                                    aggregate = statistics.median(
                                        [run["score"] for run in runs] + [partial_score]
                                    )
                                    step = case * horizon + accumulator.steps
                                    current.update(
                                        score=partial_score,
                                        nrmse=metrics.nrmse,
                                        head_horizons=metrics.head_horizons,
                                        head_errors=metrics.head_errors,
                                    )
                                    intermediates.append((step, aggregate))
                                    job.progress(aggregate, step)
                        if accumulator.steps != horizon:
                            raise RuntimeError(
                                "Forecast stream did not finish its declared horizon"
                            )
                        metrics = accumulator.summary()
                        assert truth is not None
                        score = (
                            metrics.score
                            if self.objective is None
                            else float(
                                self.objective(
                                    ScoreContext(
                                        {
                                            name: torch.cat(values, dim=-2)
                                            for name, values in collected.items()
                                        },
                                        {name: value.clone() for name, value in truth.items()},
                                        {name: value.clone() for name, value in scales.items()},
                                        seed,
                                        index,
                                    )
                                )
                            )
                        )
                        if not math.isfinite(score):
                            raise ValueError("The tuning objective must return a finite scalar")
                        _synchronize(workflow)
                        current.update(
                            score=score,
                            nrmse=metrics.nrmse,
                            head_horizons=metrics.head_horizons,
                            head_errors=metrics.head_errors,
                            forecast_seconds=time.perf_counter() - started,
                            complete=True,
                        )
                        runs.append(current)
                        current = None
                        step = (case + 1) * horizon
                        aggregate = statistics.median(run["score"] for run in runs)
                        # Full-forecast objectives preserve their full-data
                        # contract unless an explicit prefix objective is given.
                        if job.progress is None or (
                            self.objective is not None and self.pruning_objective is None
                        ):
                            intermediates.append((step, aggregate))
                            if job.progress is not None:
                                job.progress(aggregate, step)
            return _Evaluation(
                number,
                params,
                statistics.median(run["score"] for run in runs),
                statistics.median(run["nrmse"] for run in runs),
                tuple(runs),
                intermediates=tuple(intermediates),
            )
        except optuna.TrialPruned as error:
            if current is not None:
                current["forecast_seconds"] = time.perf_counter() - started
                runs.append(current)
            return _Evaluation(
                number,
                params,
                None,
                None,
                tuple(runs),
                pruned=True,
                pruning_reason=str(error),
                intermediates=tuple(intermediates),
            )
        except Exception as error:
            if not self.catch_exceptions:
                raise
            return _Evaluation(
                number, params, None, None, tuple(runs), f"{type(error).__name__}: {error}"
            )


def _install(target: Model | ESN, winner: Model | ESN, report: TuningReport) -> None:
    """Replace internals in one commit while preserving the outer Module's hooks."""
    if isinstance(target, Model):
        winner = _workflow(winner)
    elif not isinstance(winner, ESN):
        raise TypeError("ESN tuning requires its preset recipe; use Model for a custom DAG")
    winner.train(target.training)
    preserved = set(nn.Module().__dict__) - {
        "training",
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
    }
    state = {name: value for name, value in winner.__dict__.items() if name not in preserved}
    state["tuning_report"] = report
    state["_compiled_call_impl"] = None
    target.__dict__.update(state)


def _model_hash(model: Model | ESN) -> str:
    return hashlib.sha256(
        repr(
            _configuration(
                (
                    repr(model),
                    model.state_dict(),
                    _module_configs(_workflow(model)),
                    _graph_structure(_workflow(model)),
                )
            )
        ).encode()
    ).hexdigest()


def _run_trials(
    study: optuna.Study,
    evaluator: _Evaluator,
    space: SearchSpace,
    *,
    trials: int,
    seed: int,
    n_jobs: int,
    inner_max_num_threads: int,
    started: float,
    timeout: float | None,
    direction: str,
    screening: Screening | None,
    budget: MemoryBudget | None,
    context: MemoryContext | None,
    custom_factory: bool,
    estimator: Callable[[dict[str, Any], MemoryContext], MemoryEstimate | int] | None,
) -> tuple[int, dict[str, Any]]:
    """Parent-owned allocation, optional cohort ranking and finite job windows."""
    from joblib import effective_n_jobs

    workers = effective_n_jobs(n_jobs)
    report: dict[str, Any] = {
        "requested_jobs": n_jobs,
        "effective_jobs": workers,
        "max_running_jobs": 0,
        "screening": None if screening is None else asdict(screening),
        "memory_budget": None if budget is None else asdict(budget),
        "memory_skipped": 0,
        "screened": 0,
        "promoted": 0,
        "pruned": 0,
        "timed_out": False,
    }
    allocated: dict[int, optuna.Trial] = {}
    attempts = 0
    primary: BaseException | None = None
    progress: TrialProgress | None = None

    def expired() -> bool:
        return timeout is not None and time.perf_counter() - started >= timeout

    def finish(
        trial: optuna.Trial, state: optuna.trial.TrialState, value: float | None = None
    ) -> None:
        study.tell(trial, value, state=state)
        allocated.pop(trial.number, None)

    def reject(trial: optuna.Trial, reason: str) -> None:
        trial.set_user_attr("pruning_reason", reason)
        trial.set_user_attr("resdag_tune_valid", False)
        finish(trial, optuna.trial.TrialState.PRUNED)
        report["pruned"] += 1

    # A stage item holds no fitted model or forecast tensors in the parent.
    Item = tuple[optuna.Trial, dict[str, Any], MemoryEstimate | None]

    def execute(
        items: list[Item], worker: _Evaluator, stage: Literal["screen", "full"]
    ) -> list[tuple[Item, _Evaluation]]:
        completed = []
        pending = iter(items)
        next_item = next(pending, None)
        while next_item is not None and not expired():
            batch: list[Item] = []
            estimates: list[MemoryEstimate] = []
            while next_item is not None and len(batch) < workers:
                estimate = next_item[2]
                if budget is not None and estimate is not None:
                    if not budget.fits(estimate, estimates):
                        break
                    estimates.append(estimate)
                batch.append(next_item)
                next_item = next(pending, None)
            if not batch:
                raise RuntimeError("An inadmissible memory estimate reached the job scheduler")
            callbacks: list[ProgressCallback] = []
            try:
                jobs = []
                for trial, params, _ in batch:
                    callback = None
                    if progress is not None and stage == "full":
                        callback = progress.callback(trial)
                        callbacks.append(callback)
                    jobs.append(_TuningJob(trial.number, params, callback, stage))
                report["max_running_jobs"] = max(report["max_running_jobs"], len(batch))
                results = (
                    [worker(job) for job in jobs]
                    if len(batch) == 1
                    else run_jobs(
                        worker,
                        jobs,
                        n_jobs=len(batch),
                        inner_max_num_threads=inner_max_num_threads,
                    )
                )
                if progress is not None:
                    progress.check()
                for item, result in zip(batch, results, strict=True):
                    trial = item[0]
                    if trial.number != result.number:
                        raise RuntimeError("A worker returned another trial's evaluation")
                    trial.set_user_attr(
                        "runs" if stage == "full" else "screening_runs", result.runs
                    )
                    if result.error is not None:
                        trial.set_user_attr("error", result.error)
                        finish(trial, optuna.trial.TrialState.FAIL)
                    elif result.pruned:
                        reject(trial, result.pruning_reason or "Optuna pruning")
                    elif stage == "screen":
                        trial.set_user_attr("screening_score", result.score)
                        trial.set_user_attr("screening_nrmse", result.nrmse)
                        report["screened"] += 1
                        completed.append((item, result))
                    else:
                        if progress is None:
                            for step, value in result.intermediates:
                                trial.report(value, step=step)
                        trial.set_user_attr("nrmse", result.nrmse)
                        trial.set_user_attr("resdag_tune_valid", True)
                        trial.set_user_attr("evaluation_complete", True)
                        finish(trial, optuna.trial.TrialState.COMPLETE, result.score)
                        completed.append((item, result))
            finally:
                if progress is not None:
                    for callback in callbacks:
                        progress.release(callback)
        return completed

    try:
        with ExitStack() as stack:
            if type(study.pruner) is not optuna.pruners.NopPruner:
                progress = stack.enter_context(TrialProgress(parallel=workers > 1))
            while attempts < trials and not expired():
                cohort_limit = screening.cohort_size or trials if screening is not None else workers
                cohort: list[Item] = []
                for _ in range(min(cohort_limit, trials - attempts)):
                    if expired():
                        break
                    trial = study.ask()
                    allocated[trial.number] = trial
                    attempts += 1
                    with _isolated_seed((seed + trial.number) % (2**63), evaluator.device):
                        params = resolve_trial_settings(trial, space(trial))
                        estimate = (
                            None
                            if budget is None or context is None
                            else estimate_memory(
                                params, context, custom_factory=custom_factory, estimator=estimator
                            )
                        )
                    trial.set_user_attr("resolved_params", json.loads(json.dumps(params)))
                    trial.set_user_attr("resolved_params_literal", repr(params))
                    trial.set_user_attr("seeds", list(evaluator.seeds))
                    trial.set_user_attr("refit_seed", evaluator.seeds[0])
                    if estimate is not None:
                        trial.set_user_attr(
                            "memory_estimate",
                            {
                                "host_bytes": estimate.host_bytes,
                                "device_bytes": estimate.device_bytes,
                                "details": json.loads(
                                    json.dumps(dict(estimate.details), default=str, allow_nan=False)
                                ),
                            },
                        )
                        if budget is not None and not budget.fits(estimate):
                            report["memory_skipped"] += 1
                            reject(
                                trial,
                                "memory_budget: estimated candidate exceeds the planning limit",
                            )
                            continue
                    cohort.append((trial, params, estimate))
                if screening is None:
                    execute(cohort, evaluator, "full")
                else:
                    cheap = replace(
                        evaluator,
                        seeds=evaluator.seeds[: screening.seed_count],
                        data=replace(
                            evaluator.data, origins=evaluator.data.origins[: screening.origin_count]
                        ),
                    )
                    screened = execute(cohort, cheap, "screen")
                    screened.sort(
                        key=lambda pair: (
                            (
                                -cast(float, pair[1].score)
                                if direction == "maximize"
                                else cast(float, pair[1].score)
                            ),
                            pair[0][0].number,
                        )
                    )
                    keep = screening.promoted_count(len(screened))
                    for item, _ in screened[keep:]:
                        reject(item[0], "screening: not promoted to full evaluation")
                    promoted = [item for item, _ in screened[:keep]]
                    report["promoted"] += len(promoted)
                    execute(promoted, evaluator, "full")
                # A timeout stops new windows, including full promotion jobs.
                # Unfinished allocations remain explicit PRUNED attempts.
                for trial in list(allocated.values()):
                    reject(trial, "timeout: no further evaluation window was submitted")
            report["timed_out"] = expired()
    except BaseException as error:
        primary = error
        raise
    finally:
        first_cleanup_error: BaseException | None = None
        for trial in list(allocated.values()):
            try:
                current = next(t for t in study.trials if t.number == trial.number)
                if current.state == optuna.trial.TrialState.RUNNING:
                    finish(trial, optuna.trial.TrialState.FAIL)
            except BaseException as cleanup_error:
                if primary is not None:
                    primary.add_note(f"Trial cleanup also failed: {cleanup_error}")
                elif first_cleanup_error is None:
                    first_cleanup_error = cleanup_error
                else:
                    first_cleanup_error.add_note(f"Trial cleanup also failed: {cleanup_error}")
        if primary is None and first_cleanup_error is not None:
            raise first_cleanup_error
    return attempts, report


def tune_model(
    model: Model | ESN,
    training: Any,
    targets: Any = None,
    *,
    preset: Preset = "balanced",
    search_space: SearchSpace | None = None,
    factory: Factory | None = None,
    validation: tuple[Any, Any] | None = None,
    validation_fraction: float = 0.2,
    horizon: int | None = None,
    n_trials: int | None = None,
    seeds: Sequence[int] | None = None,
    seed: int = 0,
    n_origins: int | None = None,
    threshold: float = 0.2,
    aggregation: Literal["weakest", "mean"] = "weakest",
    objective: Callable[[ScoreContext], float] | None = None,
    pruning_objective: Callable[[PrefixScoreContext], float] | None = None,
    direction: Literal["maximize", "minimize"] = "maximize",
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: str | optuna.pruners.BasePruner | None = None,
    checkpoint_interval: int = 64,
    study: optuna.Study | None = None,
    storage: str | None = None,
    study_name: str | None = None,
    warm_start: optuna.Study | None = None,
    transfer_mode: Literal["reevaluate", "scores"] = "reevaluate",
    screening: bool | Screening | None = False,
    memory_budget: MemoryBudget | str | int | None = None,
    memory_estimator: Callable[[dict[str, Any], MemoryContext], MemoryEstimate | int] | None = None,
    n_jobs: int = 1,
    timeout: float | None = None,
    inner_max_num_threads: int = 1,
    catch_exceptions: bool = True,
    execution: Literal["auto", "fast", "history"] = "auto",
) -> Model | ESN:
    """Compare autonomous forecasts and install a fully refitted winner.

    ``factory(params, seed)`` builds custom candidate DAGs. Without a factory,
    custom models keep their weights/wiring and tune alpha/washout; presets
    rebuild automatically. New attempts, including failed configurations, count
    toward ``n_trials``. A timeout stops new windows; running fits and final
    refitting finish normally. Tune before constructing a PyTorch optimizer.
    """
    started = time.perf_counter()
    if preset not in _PRESETS:
        raise ValueError("preset must be 'quick', 'balanced' or 'thorough'")
    if factory is not None and isinstance(model, ESN):
        raise ValueError("ESN rebuilds itself automatically; use rd.Model for a custom factory")
    if aggregation not in ("weakest", "mean"):
        raise ValueError("aggregation must be 'weakest' or 'mean'")
    if execution not in ("auto", "fast", "history"):
        raise ValueError("execution must be 'auto', 'fast' or 'history'")
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be 'maximize' or 'minimize'")
    if objective is None and direction != "maximize":
        raise ValueError("The built-in horizon objective requires direction='maximize'")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive and finite")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    _count(inner_max_num_threads, "inner_max_num_threads")
    _count(checkpoint_interval, "checkpoint_interval")
    if not isinstance(catch_exceptions, bool):
        raise TypeError("catch_exceptions must be a boolean")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    trials, seed_count, origin_count = _PRESETS[preset]
    trials = trials if n_trials is None else _count(n_trials, "n_trials")
    origin_count = origin_count if n_origins is None else _count(n_origins, "n_origins")
    fixed_graph = factory is None and _automatic_options(model) is None
    if seeds is None:
        seeds = tuple(seed + i for i in range(1 if fixed_graph else seed_count))
    try:
        seeds = tuple(seeds)
    except TypeError as error:
        raise ValueError("seeds must be a nonempty sequence of integers") from error
    if not seeds or any(
        isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**63 for s in seeds
    ):
        raise ValueError("seeds must contain integers in [0, 2**63)")
    seed_bank = tuple(dict.fromkeys(seeds))
    if fixed_graph and len(seed_bank) != 1:
        raise ValueError("Several initialization seeds require a factory to rebuild a custom DAG")
    if study is not None and (
        storage is not None or study_name is not None or sampler is not None or pruner is not None
    ):
        raise ValueError("Pass an existing study or storage/study_name/sampler/pruner, not both")
    if warm_start is not None and not isinstance(warm_start, optuna.Study):
        raise TypeError("warm_start must be an Optuna study")
    if warm_start is not None and warm_start is study:
        raise ValueError("warm_start must be a separate source study")
    if transfer_mode not in ("reevaluate", "scores"):
        raise ValueError("transfer_mode must be 'reevaluate' or 'scores'")
    if search_space is not None and not callable(search_space):
        raise TypeError("search_space must be callable")
    if factory is not None and not callable(factory):
        raise TypeError("factory must be callable")
    if objective is not None and not callable(objective):
        raise TypeError("objective must be callable")
    if pruning_objective is not None and not callable(pruning_objective):
        raise TypeError("pruning_objective must be callable")
    if screening is True:
        screening = Screening()
    elif screening is False:
        screening = None
    elif screening is not None and not isinstance(screening, Screening):
        raise TypeError("screening must be False, True, or a Screening policy")
    budget = (
        memory_budget
        if isinstance(memory_budget, MemoryBudget) or memory_budget is None
        else MemoryBudget(memory_budget)
    )
    if memory_estimator is not None and not callable(memory_estimator):
        raise TypeError("memory_estimator must be callable")
    if memory_estimator is not None and budget is None:
        raise ValueError("memory_estimator requires memory_budget")
    if budget is not None and factory is not None and memory_estimator is None:
        raise ValueError("A custom factory with a memory budget requires memory_estimator")
    resolved_pruner: optuna.pruners.BasePruner
    if study is None and isinstance(pruner, str) and pruner.lower() == "threshold":
        if objective is not None:
            raise ValueError(
                "The common threshold preset is for positive horizon scores; "
                "supply a configured Optuna ThresholdPruner for a custom objective"
            )
        resolved_pruner = optuna.pruners.ThresholdPruner(
            lower=0.1, n_warmup_steps=2 * checkpoint_interval
        )
    else:
        resolved_pruner = resolve_pruner(pruner) if study is None else study.pruner
    # A huge lazy preset must not allocate its initial dense recurrence before
    # resource planning can even start. This is a known lower bound, not a
    # substituted smaller architecture or a claim about solver peak memory.
    if budget is not None and isinstance(model, ESN) and model.workflow is None:
        size = model._factory_options["reservoir_size"]
        sample = training.get("input") if isinstance(training, Mapping) else training
        if is_trajectory(sample):
            sample = torch.as_tensor(cast(Any, sample).y)
        element = model._reference.element_size()
        if not model._dtype_pinned and isinstance(sample, torch.Tensor):
            element = sample.element_size()
        lower = size * size * element
        destination = (
            model._reference.device
            if model._device_pinned or not isinstance(sample, torch.Tensor)
            else sample.device
        )
        minimum = MemoryEstimate(lower) if destination.type == "cpu" else MemoryEstimate(0, lower)
        if not budget.fits(minimum):
            raise ValueError(
                "The lazy ESN base template exceeds memory_budget before construction; "
                "start from a smaller base model or provide a custom construction recipe"
            )
    previous_report = model.__dict__.get("tuning_report")
    resuming = isinstance(previous_report, TuningReport) and (
        study is previous_report.study
        or (storage is not None and study_name == previous_report.study.study_name)
    )
    if resuming:
        assert isinstance(previous_report, TuningReport)
        if _model_hash(model) != previous_report._installed_hash:
            raise ValueError(
                "The model changed since tuning; use a fresh study for a changed model"
            )
        template = _template(previous_report._template, training, seed)
    else:
        template = _template(model, training, seed)
    workflow = _workflow(template)
    data = prepare_tuning_data(
        workflow,
        training,
        targets,
        validation=validation,
        validation_fraction=validation_fraction,
        horizon=horizon,
        n_origins=origin_count,
    )
    if screening is not None:
        screening.validate(len(seed_bank), len(data.origins))
    memory = (
        None
        if budget is None
        else _memory_context(
            workflow,
            data,
            preset_options=_automatic_options(template),
        )
    )
    if memory is not None and (objective is not None or pruning_objective is not None):
        reference = module_reference(workflow.graph)
        element = 4 if reference is None else reference.element_size()
        batch = next(iter(data.origins[0].truth.values()))
        batches = batch.shape[0] if batch.ndim == 3 else 1
        retained = (
            batches
            * max(origin.horizon for origin in data.origins)
            * sum(workflow._output_widths)
            * element
        )
        memory = replace(memory, retained_forecast_bytes=retained)
    if memory is not None and (
        execution == "history" or (execution == "auto" and workflow._requires_complete_history())
    ):
        element = torch.finfo(memory.dtype).bits // 8
        replay = max(
            sum(
                (value.shape[0] if value.ndim == 3 else 1)
                * (value.shape[-2] + origin.horizon)
                * value.shape[-1]
                for value in origin.history.values()
            )
            for origin in data.origins
        )
        memory = replace(
            memory,
            replay_history_bytes=3 * element * replay,
            forecast_samples=max(
                (value.shape[0] if value.ndim == 3 else 1) * (value.shape[-2] + origin.horizon)
                for origin in data.origins
                for value in origin.history.values()
            ),
        )
    automatic = _automatic_options(template)
    recipe = factory or (
        _preset_recipe(template) if automatic is not None else _fixed_recipe(template)
    )
    space = search_space or _default_space(template)
    fingerprint, _ = _protocol(
        data,
        template,
        seeds=seed_bank,
        threshold=threshold,
        aggregation=aggregation,
        direction=direction,
        factory=_callable_identity(factory),
        search_space=_callable_identity(search_space),
        objective=_callable_identity(objective),
        pruning_objective=_callable_identity(pruning_objective),
        memory_budget=None if budget is None else asdict(budget),
        memory_estimator=_callable_identity(memory_estimator),
        fixed_graph=fixed_graph,
        builtin_recipe=(
            None
            if automatic is None
            else {
                "version": 1,
                "family": cast(dict[str, Any], builtin_recipe(template.graph))["family"],
                "options": _configuration(automatic),
                "default_space_version": 2 if search_space is None else None,
            }
        ),
        execution=execution,
    )
    owned = None
    primary_error: BaseException | None = None
    try:
        if study is None:
            owned = resolve_storage(storage, n_workers=1)
            study = optuna.create_study(
                storage=owned,
                study_name=study_name,
                load_if_exists=study_name is not None,
                direction=direction,
                sampler=sampler or optuna.samplers.TPESampler(seed=seed),
                pruner=resolved_pruner,
            )
        if len(study.directions) != 1 or study.direction.name.lower() != direction:
            raise ValueError("Study direction must match the tuning objective")
        bind_protocol(study, fingerprint)
        if warm_start is not None:
            from .transfer import transfer_trials

            transfer_trials(study, warm_start, mode=transfer_mode, protocol_hash=fingerprint)
        study.set_user_attr(
            "resdag_tuning",
            {
                "version": 2,
                "seeds": list(seed_bank),
                "refit_seed": seed_bank[0],
                "validation": data.report,
                "threshold": threshold,
                "aggregation": aggregation,
                "direction": direction,
                "fixed_graph": fixed_graph,
            },
        )
        reference = module_reference(workflow.graph)
        evaluator = _Evaluator(
            recipe,
            data,
            seed_bank,
            threshold,
            aggregation,
            objective,
            catch_exceptions,
            torch.device("cpu") if reference is None else reference.device,
            execution,
            checkpoint_interval,
            pruning_objective,
        )
        attempts, resource_report = _run_trials(
            study,
            evaluator,
            space,
            trials=trials,
            seed=seed,
            n_jobs=n_jobs,
            inner_max_num_threads=inner_max_num_threads,
            started=started,
            timeout=timeout,
            direction=direction,
            screening=screening,
            budget=budget,
            context=memory,
            custom_factory=factory is not None,
            estimator=memory_estimator,
        )
        study.set_user_attr("resdag_tuning_resources", resource_report)
        valid = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.user_attrs.get("resdag_tune_valid")
            and trial.value is not None
            and math.isfinite(trial.value)
        ]
        if not valid:
            detail = next(
                (
                    trial.user_attrs.get("error", trial.user_attrs.get("pruning_reason"))
                    for trial in study.trials
                    if "error" in trial.user_attrs or "pruning_reason" in trial.user_attrs
                ),
                "no evaluation finished",
            )
            raise RuntimeError(f"No valid tuning candidates completed: {detail}")
        best = (max if direction == "maximize" else min)(
            valid, key=lambda trial: cast(float, trial.value)
        )
        if study.best_trial.number != best.number:
            raise ValueError(
                "Study contains a winning score outside this tuning protocol; use a fresh study"
            )
        best_params = ast.literal_eval(best.user_attrs["resolved_params_literal"])
        refit_started = time.perf_counter()
        with _isolated_seed(seed_bank[0], evaluator.device):
            winner = evaluator.build(best_params, seed_bank[0])
            data.fit(_workflow(winner), refit=True)
            _synchronize(_workflow(winner))
        refit_seconds = time.perf_counter() - refit_started
        report = TuningReport(
            study,
            best_params,
            best.number,
            cast(float, best.value),
            best.user_attrs["nrmse"],
            seed_bank,
            seed_bank[0],
            data.report,
            fingerprint,
            attempts,
            time.perf_counter() - started,
            refit_seconds,
            tuple(best.user_attrs["runs"]),
            template,
            _model_hash(winner),
            resource_report,
        )
        _close_storage(owned, keep_study_usable=True)
        _install(model, winner, report)
        owned = None
        return model
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_storage(owned, primary_error=primary_error, keep_study_usable=primary_error is None)
