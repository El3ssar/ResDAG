"""Common, transactional tuning of fitted PyTorch reservoir models.

Optuna owns sampling and trial bookkeeping; joblib owns parallel processes.
The data adapter owns immutable chronological views. Candidates use the same
fit/forecast workflow as the final model, including named ports and units.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import marshal
import math
import platform
import random
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, cast

import numpy as np
import optuna
import torch
from torch import nn

from resdag._contracts import module_reference
from resdag.core.model import ESNModel
from resdag.core.workflow import Model, _washout
from resdag.data._trajectory import is_trajectory
from resdag.facade import ESN, _configuration
from resdag.layers.readouts import ReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer

from ._parallel import run_jobs
from ._tuning_data import TuningData, prepare_tuning_data
from ._tuning_scores import score_forecast
from .storage import _close_storage, resolve_storage

__all__ = ["ScoreContext", "TuningReport", "tune_model"]

Factory = Callable[[dict[str, Any], int], Model | ESN]
SearchSpace = Callable[[optuna.Trial], Mapping[str, Any]]
Preset = Literal["quick", "balanced", "thorough"]
_PRESETS = {"quick": (8, 1, 1), "balanced": (24, 3, 3), "thorough": (80, 5, 5)}
_PROTOCOL_KEY = "resdag_tuning_protocol_v1"


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


def _preset_recipe(template: ESN) -> Factory:
    reference = module_reference(template.graph)
    assert reference is not None and template.input_size is not None
    options = copy.deepcopy(template._factory_options)
    options["alpha"] = options.pop("readout_alpha")
    options.pop("seed", None)
    workflow_options = copy.deepcopy(template._workflow_options)
    width, device, dtype = template.input_size, reference.device, reference.dtype

    def build(params: dict[str, Any], seed: int) -> ESN:
        supplied = copy.deepcopy(options)
        settings = copy.deepcopy(workflow_options)
        for key, value in params.items():
            if key in settings:
                settings[key] = value
            else:
                supplied[key] = value
        if "alpha" in params:
            readout_options = dict(supplied.get("readout_kwargs") or {})
            readout_options.pop("alpha", None)
            supplied["readout_kwargs"] = readout_options
        return ESN(input_size=width, device=device, dtype=dtype, seed=seed, **settings, **supplied)

    return build


def _fixed_recipe(template: Model) -> Factory:
    def build(params: dict[str, Any], seed: int) -> Model:
        candidate = copy.deepcopy(template)
        for key, value in params.items():
            if key == "washout":
                candidate.washout = _washout(value)
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
    heads = [
        module for module in _workflow(template).graph.modules() if isinstance(module, ReadoutLayer)
    ]
    tune_alpha = bool(heads) and all(hasattr(head, "alpha") for head in heads)
    if not isinstance(template, ESN) and not tune_alpha:
        raise ValueError(
            "No default tunable readout alpha; provide search_space and, for custom settings, factory"
        )

    def space(trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = (
            {"alpha": trial.suggest_float("alpha", 1e-10, 1e-2, log=True)} if tune_alpha else {}
        )
        if isinstance(template, ESN):
            size = template._factory_options["reservoir_size"]
            params.update(
                reservoir_size=trial.suggest_int(
                    "reservoir_size", max(8, size // 2), max(8, size * 2)
                ),
                spectral_radius=trial.suggest_float("spectral_radius", 0.2, 1.5),
                leak_rate=trial.suggest_float("leak_rate", 0.05, 1.0),
            )
        return params

    return space


def _dependency_identity(value: Any, seen: set[int]) -> Any:
    if inspect.ismodule(value):
        return ("module", value.__name__, getattr(value, "__version__", None))
    if inspect.isfunction(value) or inspect.ismethod(value) or isinstance(value, partial):
        return _callable_identity(value, seen)
    if isinstance(value, (Mapping, tuple, list)):
        if id(value) in seen:
            return ("recursive", type(value).__qualname__)
        seen = seen | {id(value)}
        if isinstance(value, Mapping):
            return tuple(
                (str(key), _dependency_identity(item, seen))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        return tuple(_dependency_identity(item, seen) for item in value)
    return _configuration(value)


def _callable_identity(function: Any, seen: set[int] | None = None) -> Any:
    if function is None:
        return None
    seen = set() if seen is None else seen
    if id(function) in seen:
        return (
            "recursive_callable",
            getattr(function, "__qualname__", type(function).__qualname__),
        )
    seen = seen | {id(function)}
    if isinstance(function, partial):
        return (
            "partial",
            _callable_identity(function.func, seen),
            _dependency_identity(function.args, seen),
            _dependency_identity(function.keywords, seen),
        )
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        code = getattr(function, "__code__", None)
        source = marshal.dumps(code).hex() if code is not None else repr(type(function))
    closure = getattr(function, "__closure__", None)
    dependencies: Mapping[str, Any] = {}
    if inspect.isfunction(function) or inspect.ismethod(function):
        dependencies = inspect.getclosurevars(function).globals
    return (
        getattr(function, "__module__", ""),
        getattr(function, "__qualname__", type(function).__qualname__),
        source,
        _dependency_identity(getattr(function, "__defaults__", None), seen),
        _dependency_identity(getattr(function, "__kwdefaults__", None), seen),
        _dependency_identity(
            tuple(cell.cell_contents for cell in closure) if closure else None, seen
        ),
        _configuration(getattr(function, "__dict__", None)),
        _configuration(getattr(function, "__self__", None)),
        _dependency_identity(dependencies, seen),
    )


def _module_configs(model: Model) -> Any:
    module_keys = set(nn.Module().__dict__)
    runtime = {
        "_compiled_call_impl",
        "_compiled_scan_cache",
        "_state_user_set",
        "_noise_generators",
        "_sparse_weight_key",
        "_sparse_active",
        "_fitted_flag",
        "last_solve",
    }
    configurations = []
    for group, root in (
        ("graph", model.graph),
        ("input", model.input_scalers),
        ("output", model.output_scalers),
    ):
        for name, module in root.named_modules():
            if module is model.graph:
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
                    type(module).__module__,
                    type(module).__qualname__,
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
        "version": 1,
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

    def build(self, params: dict[str, Any], seed: int) -> Model | ESN:
        with torch.inference_mode(False):
            supplied = self.factory(copy.deepcopy(params), seed)
        if not isinstance(supplied, (Model, ESN)):
            raise TypeError("A candidate recipe must return rd.Model or rd.ESN")
        return _template(supplied, self.data.train.inputs, seed)

    def __call__(self, job: tuple[int, dict[str, Any]]) -> _Evaluation:
        number, params = job
        runs: list[dict[str, Any]] = []
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
                        execution = self.execution
                        if execution == "auto":
                            execution = (
                                "history" if workflow._requires_complete_history() else "fast"
                            )
                        predictions = origin.forecast(workflow, execution=execution)
                        _synchronize(workflow)
                        forecast_seconds = time.perf_counter() - started
                        truth = {
                            name: value.to(predictions[name])
                            for name, value in origin.truth.items()
                        }
                        metrics = score_forecast(
                            predictions,
                            truth,
                            scales,
                            threshold=self.threshold,
                            aggregation=self.aggregation,
                        )
                        context = ScoreContext(
                            predictions,
                            {name: value.clone() for name, value in truth.items()},
                            {name: value.clone() for name, value in scales.items()},
                            seed,
                            index,
                        )
                        score = (
                            metrics.score
                            if self.objective is None
                            else float(self.objective(context))
                        )
                        if not math.isfinite(score):
                            raise ValueError("The tuning objective must return a finite scalar")
                        runs.append(
                            {
                                "seed": seed,
                                "origin": index,
                                "score": score,
                                "nrmse": metrics.nrmse,
                                "head_horizons": metrics.head_horizons,
                                "head_errors": metrics.head_errors,
                                "fit_seconds": fit_seconds if index == 0 else 0.0,
                                "forecast_seconds": forecast_seconds,
                            }
                        )
            return _Evaluation(
                number,
                params,
                statistics.median(run["score"] for run in runs),
                statistics.median(run["nrmse"] for run in runs),
                tuple(runs),
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


def _resolved_params(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep fixed values alongside sampled values in portable Optuna metadata."""
    if not isinstance(value, Mapping) or not all(isinstance(name, str) for name in value):
        raise TypeError("search_space must return a mapping with string keys")
    try:
        json.dumps(dict(value), allow_nan=False)
        return cast(dict[str, Any], ast.literal_eval(repr(dict(value))))
    except (TypeError, ValueError, SyntaxError) as error:
        raise TypeError(
            "search_space values must be finite JSON data; capture custom modules, "
            "initializers and callables in factory(params, seed) instead"
        ) from error


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
    direction: Literal["maximize", "minimize"] = "maximize",
    sampler: optuna.samplers.BaseSampler | None = None,
    study: optuna.Study | None = None,
    storage: str | None = None,
    study_name: str | None = None,
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
    if not isinstance(catch_exceptions, bool):
        raise TypeError("catch_exceptions must be a boolean")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    trials, seed_count, origin_count = _PRESETS[preset]
    trials = trials if n_trials is None else _count(n_trials, "n_trials")
    origin_count = origin_count if n_origins is None else _count(n_origins, "n_origins")
    fixed_graph = isinstance(model, Model) and factory is None
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
    if study is not None and (storage is not None or study_name is not None or sampler is not None):
        raise ValueError("Pass an existing study or storage/study_name/sampler, not both")
    if search_space is not None and not callable(search_space):
        raise TypeError("search_space must be callable")
    if factory is not None and not callable(factory):
        raise TypeError("factory must be callable")
    if objective is not None and not callable(objective):
        raise TypeError("objective must be callable")
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
    recipe = factory or (
        _preset_recipe(template) if isinstance(template, ESN) else _fixed_recipe(template)
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
        fixed_graph=fixed_graph,
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
                pruner=optuna.pruners.NopPruner(),
            )
        if len(study.directions) != 1 or study.direction.name.lower() != direction:
            raise ValueError("Study direction must match the tuning objective")
        previous = study.user_attrs.get(_PROTOCOL_KEY)
        evaluated = any(trial.state != optuna.trial.TrialState.WAITING for trial in study.trials)
        if previous != fingerprint and (previous is not None or evaluated):
            raise ValueError(
                "Study protocol differs (data, splits, seeds, task or objective); use a fresh study"
            )
        study.set_user_attr(_PROTOCOL_KEY, fingerprint)
        study.set_user_attr(
            "resdag_tuning",
            {
                "version": 1,
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
        )
        window = 1
        if n_jobs != 1:
            from joblib import effective_n_jobs

            window = effective_n_jobs(n_jobs)
        attempts = 0
        while attempts < trials and (timeout is None or time.perf_counter() - started < timeout):
            active = []
            window_error: BaseException | None = None
            try:
                jobs = []
                for _ in range(min(window, trials - attempts)):
                    trial = study.ask()
                    active.append(trial)
                    attempts += 1
                    with _isolated_seed((seed + trial.number) % (2**63), evaluator.device):
                        params = _resolved_params(space(trial))
                    jobs.append((trial.number, params))
                    trial.set_user_attr("resolved_params", json.loads(json.dumps(params)))
                    trial.set_user_attr("resolved_params_literal", repr(params))
                    trial.set_user_attr("seeds", list(seed_bank))
                    trial.set_user_attr("refit_seed", seed_bank[0])
                results = (
                    [evaluator(job) for job in jobs]
                    if n_jobs == 1
                    else run_jobs(
                        evaluator, jobs, n_jobs=n_jobs, inner_max_num_threads=inner_max_num_threads
                    )
                )
                for trial, result in zip(active, results, strict=True):
                    trial.set_user_attr("runs", result.runs)
                    if result.error is not None:
                        trial.set_user_attr("error", result.error)
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    else:
                        trial.set_user_attr("nrmse", result.nrmse)
                        trial.set_user_attr("resdag_tune_valid", True)
                        study.tell(trial, result.score)
                active = []
            except BaseException as error:
                window_error = error
                raise
            finally:
                for trial in active:
                    try:
                        current = next(t for t in study.trials if t.number == trial.number)
                        if current.state == optuna.trial.TrialState.RUNNING:
                            study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    except BaseException as cleanup_error:
                        if window_error is None:
                            raise
                        window_error.add_note(f"Trial cleanup also failed: {cleanup_error}")
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
                    trial.user_attrs["error"]
                    for trial in study.trials
                    if "error" in trial.user_attrs
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
