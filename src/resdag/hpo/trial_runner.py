"""Numerical raw-model evaluation with parent-owned Optuna bookkeeping.

The parent prepares settings/data and invokes user Trial callbacks. Managed
joblib workers fit, stream forecasts, and return ordinary numerical metadata;
they never receive a Study or open its storage. Checkpoint pruning stops future
forecast generation. Raw/custom losses may retain the growing prediction prefix
for their existing normalization and monitor semantics.
"""

import gc
import inspect
import json
import logging
import random
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Literal

import numpy as np
import optuna
import torch

from resdag.core import ESNModel
from resdag.core._modes import evaluating
from resdag.training import ESNTrainer

from ._provenance import RESOLVED_SETTINGS_KEY, fingerprint, resolve_trial_settings
from ._torch_losses import score_builtin_tensor_loss
from .losses import LossProtocol

__all__ = ["TrialRunner", "TrialCallback", "TrialEvaluationError"]

logger = logging.getLogger(__name__)


class TrialEvaluationError(RuntimeError):
    """An invalid evaluation that an HPO backend records as a failed trial."""


# Substring of the RuntimeError PyTorch raises when a *forked* worker touches
# CUDA after the parent already initialised it. This is a *systematic* failure —
# it recurs on every trial in every forked worker, so penalising it silently
# would burn the whole study budget producing ``penalty_value`` for every trial
# and return nonsense best_params. When a caught exception carries this marker
# we re-raise instead of penalising, so the study aborts loudly. The matched
# managed joblib backend avoids fork-based numerical workers.
_CUDA_FORK_REINIT_MARKER = "Cannot re-initialize CUDA in forked subprocess"


def _accepts_seed(func: Callable[..., Any]) -> bool:
    """Return ``True`` if ``func`` accepts a ``seed`` keyword argument.

    A ``model_creator`` accepts ``seed`` if its signature names a ``seed``
    parameter or declares ``**kwargs`` (variadic keyword).  When introspection
    fails (e.g. a C builtin without a signature) we conservatively assume it
    does not, so the seed is never injected blindly.

    Parameters
    ----------
    func : Callable
        The callable to introspect.

    Returns
    -------
    bool
        ``True`` when ``func`` can receive a ``seed=`` keyword argument.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if "seed" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# A per-trial callback receives the trial and a context dict.  It may call
# ``trial.report``/``trial.should_prune`` (acting as a pruning signal) but its
# own exceptions are always swallowed so it can never fail a trial.
TrialCallback = Callable[[optuna.Trial, dict[str, Any]], None]


@dataclass
class PreparedTrial:
    """Parent-prepared data/settings; contains no Trial or storage object."""

    number: int
    params: dict[str, Any]
    data: dict[str, Any]
    seed: int | None
    direction: optuna.study.StudyDirection
    rng: dict[str, Any]
    attrs: dict[str, Any]


@dataclass
class TrialResult:
    """Worker-owned numerical outcome and metadata for the parent to persist."""

    number: int
    params: dict[str, Any]
    state: Literal["complete", "pruned", "fail"]
    value: float | None
    attrs: dict[str, Any]
    error: str | None = None
    intermediate_values: dict[int, float] = field(default_factory=dict)
    reported_live: bool = False


@dataclass
class _EvaluationTrace:
    """Internal numeric trace; user Trial callbacks never receive this object."""

    number: int
    direction: optuna.study.StudyDirection
    progress: Callable[[float, int], None] | None
    user_attrs: dict[str, Any] = field(default_factory=dict)
    intermediate_values: dict[int, float] = field(default_factory=dict)

    @property
    def study(self) -> Any:
        return SimpleNamespace(direction=self.direction)

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value

    def report(self, value: float, step: int) -> None:
        self.intermediate_values[step] = value
        if self.progress is not None:
            self.progress(value, step)

    def should_prune(self) -> bool:
        # The parent progress callback already reports and checks its real Trial.
        return False


@contextmanager
def _rng_scope(device: torch.device | None) -> Iterator[None]:
    """Use standard Torch RNG guards and restore the caller's Python/NumPy state."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index not in cuda_devices:
            cuda_devices.append(index)
    try:
        with ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=cuda_devices))
            if torch.backends.mps.is_available():
                stack.enter_context(torch.random.fork_rng(devices=[0], device_type="mps"))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _capture_rng() -> dict[str, Any]:
    return {
        "torch": torch.random.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    torch.random.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def _unused_search_space(trial: Any) -> dict[str, Any]:
    raise RuntimeError("Search space is parent-owned and must not run in a worker")


def _unused_data_loader(trial: Any) -> dict[str, torch.Tensor]:
    raise RuntimeError("Data loading is parent-owned and must not run in a worker")


_REQUIRED_DATA_KEYS = ("warmup", "train", "target", "f_warmup", "val")

# Per-driver companion prefixes the runner threads through training and
# forecasting.  When ``drivers_keys`` is set, every declared key ``k`` must have
# all of ``warmup_k`` / ``train_k`` / ``f_warmup_k`` / ``forecast_k`` present in
# the data dict, or the driver would be silently dropped (training and
# forecasting would run feedback-only and produce a misleading score).
_DRIVER_PREFIXES = ("warmup_", "train_", "f_warmup_", "forecast_")


def _cleanup() -> None:
    """Reclaim memory between trials.

    Forces Python garbage collection and, when CUDA is available, empties the
    GPU memory cache to avoid out-of-memory errors across long studies.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_horizons(horizon: int, n_checkpoints: int) -> list[int]:
    """Build growing horizon checkpoints up to and including ``horizon``.

    The checkpoints are evenly spaced cut-off lengths used to slice a single
    forecast for intermediate ``trial.report`` calls.  The final checkpoint is
    always the full ``horizon`` so the reported value at the last checkpoint
    matches the returned objective.

    Parameters
    ----------
    horizon : int
        Full forecast horizon (number of autoregressive steps).
    n_checkpoints : int
        Desired number of checkpoints.  Values ``<= 1`` yield a single
        checkpoint at the full horizon (no intermediate reporting).

    Returns
    -------
    list of int
        Strictly increasing checkpoint lengths in ``[1, horizon]`` ending at
        ``horizon``.  De-duplicated when ``horizon < n_checkpoints``.
    """
    if horizon < 1:
        return [horizon]
    if n_checkpoints <= 1:
        return [horizon]

    n_checkpoints = min(n_checkpoints, horizon)
    raw = [round(horizon * (i + 1) / n_checkpoints) for i in range(n_checkpoints)]
    checkpoints: list[int] = []
    for c in raw:
        c = max(1, min(horizon, c))
        if not checkpoints or c > checkpoints[-1]:
            checkpoints.append(c)
    checkpoints[-1] = horizon
    return checkpoints


class TrialRunner:
    """Picklable callable that evaluates a single Optuna trial.

    An instance of this class behaves exactly like the objective closure that
    :func:`resdag.hpo.objective.build_objective` used to return: call it with an
    :class:`optuna.Trial` and it returns the (possibly clipped) loss to
    minimize. Managed joblib serializes only its numerical callbacks and
    prepared jobs; parent-only study/data callbacks remain in the caller.

    The runner generates forecast chunks up to growing horizon
    checkpoints, calling :meth:`optuna.Trial.report` and
    :meth:`optuna.Trial.should_prune` at each checkpoint so configured pruners
    can terminate diverging trials early.  Scoring stays on the forecast's
    device until the final scalar when the loss function supports tensor inputs,
    transparently falling back to NumPy otherwise.

    Parameters
    ----------
    model_creator : Callable[..., ESNModel]
        Function that creates a fresh model given hyperparameters.  Must accept
        all resolved model settings as keyword arguments. Joblib supports notebook
        factories and closures; parent-only callbacks are excluded from jobs.
    search_space : Callable[[Trial], dict[str, Any]]
        Function that defines the hyperparameter search space using Optuna's
        ``trial.suggest_*`` methods.  Returns a dict of hyperparameters.
    data_loader : Callable[[Trial], dict[str, Tensor]]
        Function that loads and returns training/validation data.  Must return a
        dict with keys ``"warmup"``, ``"train"``, ``"target"``, ``"f_warmup"``,
        ``"val"``.  Optionally include driver inputs (e.g. ``"warmup_driver"``).
    loss_fn : LossProtocol
        Loss function to evaluate model performance.  Receives ``(y_true,
        y_pred)`` — either both NumPy arrays or both ``torch.Tensor`` (see
        ``torch_scoring``) — and returns a scalar to minimize.
    targets_key : str, default="output"
        Name of the readout layer target in the targets dict.
    drivers_keys : list of str, optional
        Driver input keys in the data dict.  If provided, the matching
        ``warmup_*`` / ``train_*`` / ``f_warmup_*`` / ``forecast_*`` entries are
        threaded through training and forecasting.
    horizon_key : str, optional
        Key in the data dict specifying the forecast horizon.  If ``None``, uses
        ``val.shape[1]``.
    catch_exceptions : bool, default=True
        If ``True``, catch exceptions and return ``penalty_value`` instead of
        raising.  :class:`optuna.TrialPruned` is **always** re-raised so pruning
        works regardless of this flag.
    penalty_value : float, default=1e10
        Value returned when a trial fails and ``catch_exceptions`` is ``True``.
    monitor_losses : list of LossProtocol, optional
        Additional loss functions to compute and log as user attributes (not
        optimized on).
    monitor_params : dict, optional
        Per-monitor-loss keyword arguments keyed by loss-function name.
    device : torch.device, optional
        Device to place model and data on.  If ``None``, uses the model/data
        defaults.
    seed : int, optional
        Base seed for per-trial reproducibility.  Trial ``n`` uses
        ``seed + n`` to seed PyTorch, NumPy, and Python's :mod:`random`
        module, and the same per-trial seed is threaded into ``model_creator``
        (when it accepts a ``seed`` keyword) so the reservoir topology and
        input/feedback initializers — which build their own RNGs and ignore the
        legacy global NumPy state — also reproduce.
    clip_value : float, optional
        Upper bound for the objective value.  When set and the raw loss exceeds
        it, the returned value is clamped (or the trial pruned, see
        ``prune_on_clip``).  The unclipped value is always stored as the
        ``"raw_loss"`` user attribute.
    prune_on_clip : bool, default=False
        If ``True`` and ``clip_value`` is set, trials whose raw loss exceeds
        ``clip_value`` are pruned instead of returning the clipped value.
    n_checkpoints : int, default=5
        Number of growing horizon checkpoints at which to report and check for
        pruning.  Values ``<= 1`` disable intermediate reporting (a single
        report at the full horizon).
    torch_scoring : bool, default=True
        If ``True``, attempt to evaluate the loss on the forecast's device using
        ``torch.Tensor`` inputs, only moving the final scalar to the host.  If
        the loss function does not support tensors, the runner falls back to
        NumPy automatically.  Set ``False`` to always score in NumPy.
    trial_callbacks : list of TrialCallback, optional
        Per-trial callbacks invoked after each successful evaluation.  Each
        receives ``(trial, context)`` and may call
        ``trial.report``/``trial.should_prune`` to signal pruning.  Callback
        exceptions are logged and swallowed — a callback can **never** fail the
        trial.

    Notes
    -----
    The runner itself is picklable; whether a *given instance* pickles depends
    on its user-supplied callables (``model_creator``, ``search_space``,
    ``data_loader``, ``loss_fn``, monitor losses, trial callbacks).  Use
    module-level functions rather than lambdas/closures for those to keep the
    instance serializable for managed joblib distribution.
    """

    def __init__(
        self,
        model_creator: Callable[..., ESNModel],
        search_space: Callable[[optuna.Trial], dict[str, Any]],
        data_loader: Callable[[optuna.Trial], dict[str, torch.Tensor]],
        loss_fn: LossProtocol,
        targets_key: str = "output",
        drivers_keys: list[str] | None = None,
        horizon_key: str | None = None,
        catch_exceptions: bool = True,
        penalty_value: float = 1e10,
        monitor_losses: list[LossProtocol] | None = None,
        monitor_params: dict[str, dict[str, Any]] | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
        clip_value: float | None = None,
        prune_on_clip: bool = False,
        n_checkpoints: int = 5,
        torch_scoring: bool = True,
        trial_callbacks: list[TrialCallback] | None = None,
        error_policy: Literal["fail", "penalty"] = "penalty",
    ) -> None:
        self.model_creator = model_creator
        self.search_space = search_space
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.targets_key = targets_key
        self.drivers_keys = drivers_keys
        self.horizon_key = horizon_key
        self.catch_exceptions = catch_exceptions
        self.penalty_value = penalty_value
        self.monitor_losses = monitor_losses
        self.monitor_params = monitor_params
        self.device = device
        self.seed = seed
        self.clip_value = clip_value
        self.prune_on_clip = prune_on_clip
        self.n_checkpoints = n_checkpoints
        self.torch_scoring = torch_scoring
        self.trial_callbacks = trial_callbacks
        if error_policy not in ("fail", "penalty"):
            raise ValueError("error_policy must be 'fail' or 'penalty'")
        self.error_policy = error_policy

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Older serialized objectives used numeric penalties and did not store
        # an error policy. Preserve that behavior when resuming their workers.
        self.__dict__.setdefault("error_policy", "penalty")

    # ── Public API ───────────────────────────────────────────────────────
    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate ``trial`` and return the (possibly clipped) loss.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial to evaluate.

        Returns
        -------
        float
            Loss value for Optuna to minimize. The explicit penalty policy
            returns ``penalty_value`` for caught failures; the fail policy
            raises ``TrialEvaluationError`` for the backend to record.

        Raises
        ------
        optuna.TrialPruned
            When a horizon checkpoint or clipping triggers pruning.  Always
            raised regardless of ``catch_exceptions``.
        """
        try:
            return self._run(trial)
        except optuna.TrialPruned:
            # Pruning must propagate so Optuna records the trial as pruned.
            raise
        except Exception as e:  # noqa: BLE001 - intentional broad catch
            # Preserve a clear failure if external code invokes this runner in
            # an unsafe fork context. The managed joblib backend avoids it.
            if _CUDA_FORK_REINIT_MARKER in str(e):
                logger.error(
                    "Trial %s cannot initialize CUDA in this process context", trial.number
                )
                raise
            if self.catch_exceptions:
                logger.warning(f"Trial {trial.number} failed: {e}")
                trial.set_user_attr("error", str(e))
                if self.error_policy == "fail":
                    raise TrialEvaluationError(str(e)) from e
                return self._directional_penalty(trial)
            raise

    # ── Per-trial lifecycle ──────────────────────────────────────────────
    def _run(self, trial: optuna.Trial) -> float:
        prepared = self.prepare(trial)

        def report(value: float, step: int) -> None:
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at checkpoint {step}")

        return self.finalize(trial, self.evaluate(prepared, report))

    def prepare(self, trial: optuna.Trial) -> PreparedTrial:
        """Resolve settings/data once in the parent using the actual Optuna Trial."""
        with _rng_scope(self.device):
            trial_seed = self._seed_trial(trial)
            sampled_model_settings = self.search_space(trial)
            params = resolve_trial_settings(trial, sampled_model_settings)
            data = self.data_loader(trial)
            _validate_data_keys(data, drivers_keys=self.drivers_keys)
            # Suggestions made only by the data callback configure its slices,
            # not model_creator(**params). Keep them in Trial.params/provenance
            # while routing only declared model settings to the constructor.
            data_parameters = set(trial.params) - set(sampled_model_settings)
            params = {name: value for name, value in params.items() if name not in data_parameters}
            # Neither an in-place layer nor a failed sequential trial may alter
            # the caller's shared training/validation observations.
            data = {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in data.items()
            }
            attrs: dict[str, Any] = {
                RESOLVED_SETTINGS_KEY: repr(params),
                "resolved_params": json.loads(json.dumps(params)),
                "resdag_raw_data_hash": fingerprint(data),
                "resdag_raw_evaluation_number": trial.number,
                "resdag_raw_data_params": {name: trial.params[name] for name in data_parameters},
            }
            if trial_seed is not None:
                attrs["trial_seed"] = trial_seed
            if _accepts_seed(self.model_creator):
                model_seed = params.get("seed", trial_seed)
                if isinstance(model_seed, int):
                    attrs["model_seed"] = model_seed
            for key, value in attrs.items():
                trial.set_user_attr(key, value)
            return PreparedTrial(
                trial.number, params, data, trial_seed, trial.study.direction, _capture_rng(), attrs
            )

    def for_worker(self) -> "TrialRunner":
        """Exclude parent-only closures/resources from joblib serialization."""
        return TrialRunner(
            self.model_creator,
            _unused_search_space,
            _unused_data_loader,
            self.loss_fn,
            targets_key=self.targets_key,
            drivers_keys=self.drivers_keys,
            horizon_key=self.horizon_key,
            catch_exceptions=self.catch_exceptions,
            penalty_value=self.penalty_value,
            monitor_losses=self.monitor_losses,
            monitor_params=self.monitor_params,
            device=self.device,
            seed=self.seed,
            clip_value=self.clip_value,
            prune_on_clip=self.prune_on_clip,
            n_checkpoints=self.n_checkpoints,
            torch_scoring=self.torch_scoring,
            trial_callbacks=None,
            error_policy=self.error_policy,
        )

    def evaluate_job(
        self, job: tuple[PreparedTrial, Callable[[float, int], None] | None]
    ) -> TrialResult:
        return self.evaluate(*job)

    def evaluate(
        self,
        prepared: PreparedTrial,
        progress: Callable[[float, int], None] | None = None,
    ) -> TrialResult:
        """Fit/forecast/score without opening a study or touching its storage."""
        trace = _EvaluationTrace(
            prepared.number, prepared.direction, progress, dict(prepared.attrs)
        )
        primary_error: BaseException | None = None

        def outcome(
            state: Literal["complete", "pruned", "fail"],
            value: float | None,
            error: str | None = None,
        ) -> TrialResult:
            return TrialResult(
                prepared.number,
                prepared.params,
                state,
                value,
                trace.user_attrs,
                error,
                dict(trace.intermediate_values),
                progress is not None,
            )

        try:
            with _rng_scope(self.device):
                _restore_rng(prepared.rng)
                data = self._to_device(prepared.data)
                model = self._create_model(prepared.params, prepared.seed)
                if self.device is not None:
                    model = model.to(self.device)
                warmup, train = self._build_train_inputs(data)
                ESNTrainer(model).fit(
                    warmup_inputs=warmup,
                    train_inputs=train,
                    targets={self.targets_key: data["target"]},
                )
                raw_loss, predictions, truth = self._stream_forecast_score(model, data, trace)
                raw_loss = self._penalize_nonfinite(trace, raw_loss)
                trace.set_user_attr("raw_loss", raw_loss)
                loss = self._apply_clipping(trace, raw_loss)
                trace.set_user_attr("loss", loss)
                self._log_monitor_losses(trace, predictions, truth)
            return outcome("complete", loss)
        except optuna.TrialPruned as error:
            primary_error = error
            return outcome("pruned", None, str(error))
        except Exception as error:
            primary_error = error
            if not self.catch_exceptions or _CUDA_FORK_REINIT_MARKER in str(error):
                raise
            trace.set_user_attr("error", str(error))
            if self.error_policy == "penalty":
                return outcome("complete", self._directional_penalty(trace), str(error))
            return outcome("fail", None, str(error))
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                _cleanup()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Numerical cleanup also failed: {cleanup_error}")
                trace.set_user_attr("cleanup_error", str(cleanup_error))

    def finalize(self, trial: optuna.Trial, result: TrialResult) -> float:
        """Publish worker metadata and run user callbacks in the parent only."""
        for key, value in result.attrs.items():
            trial.set_user_attr(key, value)
        if not result.reported_live:
            for step, value in result.intermediate_values.items():
                trial.report(value, step)
        if result.state == "pruned":
            raise optuna.TrialPruned(result.error)
        if result.state == "fail":
            raise TrialEvaluationError(result.error)
        assert result.value is not None
        self._invoke_callbacks(
            trial,
            params=result.params,
            raw_loss=result.attrs.get("raw_loss", result.value),
            loss=result.value,
        )
        return result.value

    def _stream_forecast_score(
        self,
        model: ESNModel,
        data: dict[str, Any],
        trial: Any,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        from resdag.core._forecast_stream import raw_forecast_stream

        horizon = (
            int(data[self.horizon_key])
            if self.horizon_key and self.horizon_key in data
            else int(data["val"].shape[1])
        )
        if horizon < 1:
            raise ValueError("Forecast horizon must be positive")
        if horizon > data["val"].shape[1]:
            raise ValueError("Forecast horizon exceeds available validation truth")
        checkpoints = _checkpoint_horizons(horizon, self.n_checkpoints)
        warmup = (data["f_warmup"],) + tuple(
            data[f"f_warmup_{key}"] for key in self.drivers_keys or ()
        )
        drivers = tuple(data[f"forecast_{key}"] for key in self.drivers_keys or ()) or None
        pieces = []
        raw_loss = self.penalty_value
        checkpoint = 0
        with raw_forecast_stream(
            model, warmup, drivers, horizon=horizon, chunk_size=64, checkpoints=checkpoints
        ) as chunks:
            for chunk in chunks:
                prediction = chunk.outputs[0]
                if not torch.isfinite(prediction).all():
                    raise ValueError("Forecast contains non-finite values; the trial is invalid")
                pieces.append(prediction)
                if chunk.stop != checkpoints[checkpoint]:
                    continue
                predictions = torch.cat(pieces, dim=1)
                truth = data["val"][:, : chunk.stop]
                raw_loss = self._score(truth, predictions)
                raw_loss = self._penalize_nonfinite(trial, raw_loss)
                checkpoint += 1
                trial.set_user_attr(f"horizon@{checkpoint}", chunk.stop)
                trial.report(raw_loss, step=checkpoint)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at checkpoint {checkpoint}")
        return raw_loss, torch.cat(pieces, dim=1), data["val"][:, :horizon]

    def _seed_trial(self, trial: optuna.Trial) -> int | None:
        """Seed the global RNGs for a trial and return the per-trial seed.

        Derives a per-trial seed as ``self.seed + trial.number`` and applies it
        to PyTorch, NumPy (legacy global state), and Python's :mod:`random`
        module so every global RNG a user callback might draw from is pinned.
        The returned seed is *also* threaded explicitly into ``model_creator``
        (see :meth:`_create_model`) so the reservoir's topology and
        input/feedback initializers — which build their own
        ``np.random.default_rng`` / :class:`torch.Generator` and **do not** read
        the legacy global NumPy state — become a pure function of the per-trial
        seed.  Seeding ``torch.manual_seed`` here is what makes a string- or
        callable-form topology reproducible *even when* ``model_creator`` does
        not accept a ``seed`` (those generators derive their seed from torch's
        global RNG when none is passed).

        Parameters
        ----------
        trial : optuna.Trial
            The trial being evaluated; its ``number`` offsets the base seed.

        Returns
        -------
        int or None
            The per-trial seed (``self.seed + trial.number``), or ``None`` when
            no base ``seed`` was configured.
        """
        if self.seed is None:
            return None
        trial_seed = self.seed + trial.number
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed % (2**32))
        random.seed(trial_seed)
        return trial_seed

    def _create_model(self, params: dict[str, Any], trial_seed: int | None) -> ESNModel:
        """Build the trial's model, threading the per-trial seed when accepted.

        The reservoir's topology and input/feedback initializers construct their
        own RNGs and do **not** consume the legacy global NumPy state, so seeding
        ``np.random`` alone does not make them reproducible.  When a base
        ``seed`` is configured and ``model_creator`` accepts a ``seed`` keyword
        (a named parameter or ``**kwargs``), the per-trial seed is injected so
        those initializers become a pure function of ``(seed, trial.number)`` —
        independent of how much global RNG ``search_space`` / ``data_loader``
        consumed first.  An explicit ``seed`` already present in ``params`` (e.g.
        one sampled by ``search_space``) always wins and is left untouched.

        Parameters
        ----------
        params : dict
            Hyperparameters returned by ``search_space`` for this trial.
        trial_seed : int or None
            The per-trial seed from :meth:`_seed_trial`, or ``None`` when no
            base ``seed`` was configured.

        Returns
        -------
        ESNModel
            The freshly created model for this trial.
        """
        if trial_seed is not None and "seed" not in params and _accepts_seed(self.model_creator):
            params = {**params, "seed": trial_seed}
        return self.model_creator(**params)

    def _to_device(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move tensor entries of ``data`` to ``self.device`` (if set)."""
        if self.device is None:
            return data
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    def _build_train_inputs(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Assemble ``(warmup_inputs, train_inputs)`` with optional drivers.

        ``_validate_data_keys`` (run in :meth:`_run` before this) guarantees that
        every declared ``drivers_keys`` entry has its ``warmup_*`` / ``train_*``
        companions present, so the keys are threaded unconditionally — a missing
        driver is a hard error there, never a silent feedback-only fallback here.
        """
        warmup_inputs: tuple[torch.Tensor, ...] = (data["warmup"],)
        train_inputs: tuple[torch.Tensor, ...] = (data["train"],)

        if self.drivers_keys:
            for key in self.drivers_keys:
                warmup_inputs = warmup_inputs + (data[f"warmup_{key}"],)
                train_inputs = train_inputs + (data[f"train_{key}"],)

        return warmup_inputs, train_inputs

    def _forecast(
        self, model: ESNModel, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single forecast and return ``(preds, val)`` (full horizon)."""
        if self.horizon_key and self.horizon_key in data:
            horizon = int(data[self.horizon_key])
        else:
            horizon = int(data["val"].shape[1])

        f_warmup_inputs: tuple[torch.Tensor, ...] = (data["f_warmup"],)
        forecast_drivers_list: list[torch.Tensor] = []

        # ``_validate_data_keys`` guarantees the ``f_warmup_*`` / ``forecast_*``
        # companions exist for every declared driver, so they are threaded
        # unconditionally; the number of forecast drivers therefore always
        # matches ``drivers_keys`` (no silent omission before ``model.forecast``).
        if self.drivers_keys:
            for key in self.drivers_keys:
                f_warmup_inputs = f_warmup_inputs + (data[f"f_warmup_{key}"],)
                forecast_drivers_list.append(data[f"forecast_{key}"])

        with evaluating(model):
            preds = model.forecast(
                f_warmup_inputs,
                forecast_inputs=tuple(forecast_drivers_list) if forecast_drivers_list else None,
                horizon=horizon,
            )
        # ``forecast`` returns a tuple for multi-output models; feedback is
        # always the first output (matched against ``val`` below).
        if isinstance(preds, tuple):
            preds = preds[0]

        if not torch.isfinite(preds).all():
            raise ValueError("Forecast contains non-finite values; the trial is invalid")

        return preds, data["val"]

    def _evaluate_with_checkpoints(
        self, trial: optuna.Trial, preds: torch.Tensor, val: torch.Tensor
    ) -> float:
        """Score the *single* forecast at growing horizon checkpoints.

        The forecast is computed once; each checkpoint slices ``preds``/``val``
        to a prefix length and reports the prefix loss via
        :meth:`optuna.Trial.report`.  If the configured pruner signals pruning
        at any checkpoint, :class:`optuna.TrialPruned` is raised.  The loss at
        the final (full-horizon) checkpoint is returned as the raw loss.

        Reports are keyed by the **1-based checkpoint index** (``1, 2, …, n``),
        *not* the horizon length in timesteps.  A pruner's step-based knobs
        (e.g. :class:`~optuna.pruners.ThresholdPruner`'s ``n_warmup_steps`` or a
        successive-halving rung schedule) are therefore expressed in
        *checkpoints* and stay horizon-independent — a warmup of ``2`` skips the
        first two checkpoints whether the horizon is 100 or 10 000 steps.  The
        actual prefix length used to slice the forecast is recorded per report
        as the ``horizon@<idx>`` user attribute so the timestep mapping is still
        recoverable for diagnostics.

        Parameters
        ----------
        trial : optuna.Trial
            The trial being evaluated.
        preds, val : torch.Tensor
            Forecast and ground truth of shape ``(B, T, D)``.

        Returns
        -------
        float
            Raw (unclipped) loss over the full overlapping horizon.

        Raises
        ------
        optuna.TrialPruned
            If the pruner signals pruning at any checkpoint.
        """
        timesteps = min(preds.shape[1], val.shape[1])
        checkpoints = _checkpoint_horizons(timesteps, self.n_checkpoints)

        raw_loss = self.penalty_value
        for idx, prefix_len in enumerate(checkpoints, start=1):
            # Loss convention is ``loss_fn(y_true, y_pred)`` → (val, preds).
            raw_loss = self._score(val[:, :prefix_len, :], preds[:, :prefix_len, :])
            # Report by checkpoint *index* (horizon-independent step axis) so
            # pruner step knobs (n_warmup_steps, halving rungs) are portable;
            # record the concrete prefix length for diagnostics.
            trial.report(raw_loss, step=idx)
            trial.set_user_attr(f"horizon@{idx}", prefix_len)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at checkpoint {idx}/{len(checkpoints)} "
                    f"(horizon={prefix_len} steps, loss={raw_loss:.6f})"
                )
        return raw_loss

    def _penalize_nonfinite(self, trial: Any, raw_loss: float) -> float:
        """Reject a non-finite loss, or apply an explicitly chosen legacy penalty.

        Always retain the ``nonfinite_loss`` diagnostic. High-level studies use
        the fail policy so a divergent evaluation cannot become the winner.
        The penalty policy preserves low-level compatibility and flips the
        numeric penalty's sign for a maximization study.

        Parameters
        ----------
        trial : optuna.Trial
            The trial being evaluated.
        raw_loss : float
            The raw (unclipped) loss returned by the loss function.

        Returns
        -------
        float
            ``raw_loss`` unchanged when finite; otherwise the direction-aware
            penalty value (``penalty_value`` for minimize, ``-penalty_value`` for
            maximize).
        """
        if np.isfinite(raw_loss):
            return raw_loss

        trial.set_user_attr("nonfinite_loss", True)
        if self.error_policy == "fail":
            raise ValueError(f"Objective returned a non-finite loss: {raw_loss}")
        penalty = self._directional_penalty(trial)
        logger.warning(
            f"Trial {trial.number}: non-finite loss ({raw_loss}); "
            f"substituting penalty_value={penalty}."
        )
        return penalty

    def _directional_penalty(self, trial: Any) -> float:
        """Keep an explicit legacy penalty unfavorable in either direction."""
        if trial.study.direction == optuna.study.StudyDirection.MAXIMIZE:
            return -self.penalty_value
        return self.penalty_value

    def _apply_clipping(self, trial: Any, raw_loss: float) -> float:
        """Clamp or prune based on ``clip_value`` / ``prune_on_clip``."""
        if self.clip_value is not None and raw_loss > self.clip_value:
            if self.prune_on_clip:
                raise optuna.TrialPruned(
                    f"raw_loss {raw_loss:.6f} exceeds clip_value {self.clip_value}"
                )
            return self.clip_value
        return raw_loss

    def _log_monitor_losses(self, trial: Any, preds: torch.Tensor, val: torch.Tensor) -> None:
        """Compute monitor losses on the full horizon and log them."""
        if not self.monitor_losses:
            return

        timesteps = min(preds.shape[1], val.shape[1])
        preds_full = preds[:, :timesteps, :]
        val_full = val[:, :timesteps, :]
        monitor_params = self.monitor_params or {}

        for monitor_fn in self.monitor_losses:
            loss_name = getattr(monitor_fn, "__name__", str(monitor_fn))
            kwargs = monitor_params.get(loss_name, {})
            try:
                monitor_value = self._score(val_full, preds_full, monitor_fn, **kwargs)
                trial.set_user_attr(f"monitor_{loss_name}", monitor_value)
            except Exception as e:  # noqa: BLE001 - monitors must never fail trials
                logger.warning(f"Monitor loss {loss_name} failed: {e}")
                trial.set_user_attr(f"monitor_{loss_name}", None)

    def _invoke_callbacks(
        self,
        trial: optuna.Trial,
        *,
        params: dict[str, Any],
        raw_loss: float,
        loss: float,
    ) -> None:
        """Invoke per-trial callbacks; their exceptions can never fail a trial.

        A callback may raise :class:`optuna.TrialPruned` to act as a pruning
        signal — that is re-raised so Optuna records the trial as pruned.  Any
        other exception is logged and swallowed.
        """
        if not self.trial_callbacks:
            return

        context: dict[str, Any] = {"params": params, "raw_loss": raw_loss, "loss": loss}
        for callback in self.trial_callbacks:
            try:
                callback(trial, context)
            except optuna.TrialPruned:
                raise
            except Exception as e:  # noqa: BLE001 - callbacks must never fail trials
                cb_name = getattr(callback, "__name__", repr(callback))
                logger.warning(f"trial_callback {cb_name} failed (ignored): {e}")

    # ── Scoring ──────────────────────────────────────────────────────────
    def _score(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        loss_fn: LossProtocol | None = None,
        **kwargs: Any,
    ) -> float:
        """Evaluate a loss, preferring on-device tensors with a NumPy fallback.

        When ``torch_scoring`` is enabled the loss is first called with the
        ``torch.Tensor`` inputs (keeping CUDA tensors on-device until the final
        scalar).  Pure-NumPy losses raise on tensor inputs; in that case — and
        whenever ``torch_scoring`` is disabled — the tensors are detached, moved
        to host and converted to NumPy before re-evaluating.

        Parameters
        ----------
        y_true, y_pred : torch.Tensor
            Ground truth and prediction of shape ``(B, T, D)``.  The loss
            functions take ``(y_true, y_pred)`` positionally.
        loss_fn : LossProtocol, optional
            Loss to evaluate.  Defaults to ``self.loss_fn``.
        **kwargs
            Extra keyword arguments forwarded to the loss function.

        Returns
        -------
        float
            The scalar loss value.
        """
        fn = loss_fn if loss_fn is not None else self.loss_fn

        if self.torch_scoring:
            with torch.no_grad():
                builtin_result = score_builtin_tensor_loss(fn, y_true, y_pred, **kwargs)
            if builtin_result is not None:
                return float(builtin_result)
            try:
                with torch.no_grad():
                    # Intentionally feed tensors; pure-NumPy losses raise here
                    # and we fall through to the NumPy path below.
                    result = fn(y_true, y_pred, **kwargs)  # type: ignore[arg-type]
                return float(result)
            except Exception:  # noqa: BLE001 - tensor scoring unsupported → NumPy
                pass

        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        return float(fn(y_true_np, y_pred_np, **kwargs))


def _validate_data_keys(
    data: Any,
    drivers_keys: list[str] | None = None,
) -> None:
    """Validate the dictionary returned by the user-provided ``data_loader``.

    The data dict must contain the five required keys (``"warmup"``,
    ``"train"``, ``"target"``, ``"f_warmup"``, ``"val"``), each mapping to a 3-D
    ``torch.Tensor`` of shape ``(batch, timesteps, features)``.  When
    ``drivers_keys`` is non-empty, every declared key ``k`` must additionally
    supply *all four* companion entries (``warmup_k`` / ``train_k`` /
    ``f_warmup_k`` / ``forecast_k``); a partially- or fully-absent driver is
    rejected rather than silently dropped — a dropped driver would train and
    forecast feedback-only and yield a misleading score.

    Parameters
    ----------
    data : Any
        Value returned by ``data_loader(trial)``.  Must be a dict-like object.
    drivers_keys : list of str, optional
        Driver feature keys.  If provided, the corresponding ``warmup_*``,
        ``train_*``, ``f_warmup_*`` and ``forecast_*`` entries must all be
        present and are validated as 3-D tensors.

    Raises
    ------
    TypeError
        If ``data`` is not a dictionary.
    KeyError
        If one or more required keys are missing, or if a declared driver key
        lacks any of its four companion entries.  Missing names are listed
        alongside the keys the loader did return for easy typo spotting.
    ValueError
        If any required entry is not a ``torch.Tensor`` or is not 3-D.
    """
    if not isinstance(data, dict):
        raise TypeError(
            f"data_loader must return a dict, got {type(data).__name__}. "
            f"Expected keys: {list(_REQUIRED_DATA_KEYS)}."
        )

    missing = [k for k in _REQUIRED_DATA_KEYS if k not in data]
    if missing:
        raise KeyError(
            f"data_loader output is missing required keys: {missing}. "
            f"Required: {list(_REQUIRED_DATA_KEYS)}. "
            f"Got keys: {sorted(data.keys())}."
        )

    keys_to_check = list(_REQUIRED_DATA_KEYS)
    if drivers_keys:
        missing_drivers = [
            f"{prefix}{key}"
            for key in drivers_keys
            for prefix in _DRIVER_PREFIXES
            if f"{prefix}{key}" not in data
        ]
        if missing_drivers:
            raise KeyError(
                f"drivers_keys={list(drivers_keys)} declared, but the following "
                f"required driver entries are missing from data_loader output: "
                f"{missing_drivers}. Each driver key 'k' must supply all of "
                f"warmup_k, train_k, f_warmup_k and forecast_k (a misnamed or "
                f"absent driver is rejected rather than silently dropped). "
                f"Got keys: {sorted(data.keys())}."
            )
        keys_to_check.extend(
            f"{prefix}{key}" for key in drivers_keys for prefix in _DRIVER_PREFIXES
        )

    for key in keys_to_check:
        value = data[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"data['{key}'] must be a torch.Tensor, got {type(value).__name__}.")
        if value.dim() != 3:
            raise ValueError(
                f"data['{key}'] must be a 3-D tensor (batch, timesteps, features); "
                f"got shape {tuple(value.shape)}."
            )
