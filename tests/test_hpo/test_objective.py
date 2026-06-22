"""Tests for the HPO objective built by :func:`build_objective`.

The following behaviours of the :class:`resdag.hpo.runner.TrialRunner` returned
by :func:`build_objective` are pinned here:

* **Clip / prune + ``raw_loss``.**  ``clip_value`` clamps the returned objective
  to the bound while the unclipped value is preserved as the ``raw_loss`` user
  attribute; ``prune_on_clip`` instead raises :class:`optuna.TrialPruned` so a
  clip-exceeding trial is ``PRUNED`` (and excluded from the study's best value)
  rather than completing at the clamp.
* **Monitor losses.**  ``monitor_losses`` are computed and logged as
  ``monitor_<__name__>`` user attributes without affecting the optimized value,
  and ``monitor_params`` are remapped onto each monitor by the loss function's
  ``__name__`` (not the registry alias) and forwarded as kwargs.
* **Driver round-trip.**  ``drivers_keys`` thread the ``warmup_<key>`` /
  ``train_<key>`` / ``f_warmup_<key>`` / ``forecast_<key>`` data entries through
  training and forecasting, so an input-driven model produces a finite,
  non-penalty loss.
* **Per-trial seeding.**  Two studies built with the same base ``seed`` reproduce
  every trial value bit-for-bit; a different base seed diverges.
* **Non-finite loss handling.**  A non-finite (NaN/inf) loss is a *value*, not an
  exception, so it slips past the ``catch_exceptions`` / ``penalty_value``
  machinery in :class:`resdag.hpo.runner.TrialRunner` (built by
  :func:`resdag.hpo.objective.build_objective`).  Optuna then rejects the value
  and records the trial as ``FAIL`` — starving the sampler of the signal that the
  region is bad and excluding the trial from the study summary.  Diverged
  forecasts are the primary failure mode of the chaotic-system use case this
  module targets, so non-finite losses are common, not rare.  These tests pin the
  fix: a non-finite loss must yield a ``COMPLETE`` trial scored at
  ``penalty_value`` (direction-aware), with the original non-finiteness recorded
  as the ``nonfinite_loss`` user attribute.

* **Multi-output forecasts.**  :meth:`ESNModel.forecast` returns a *tuple* of
  tensors for multi-output models (the first output being the autoregression
  feedback).  The objective must normalize that tuple to the first output before
  indexing it (``preds.shape``), otherwise every multi-output trial raises
  ``AttributeError: 'tuple' object has no attribute 'shape'`` and — under the
  default ``catch_exceptions=True`` — silently completes at ``penalty_value``,
  turning the whole study into garbage.  These tests pin the guard: a 2-output
  model produces a ``COMPLETE`` trial with a finite, non-penalty loss scored on
  the first output.
"""

import numpy as np
import pytest
import torch

optuna = pytest.importorskip("optuna")

from resdag.core import ESNModel, reservoir_input  # noqa: E402
from resdag.hpo.objective import build_objective  # noqa: E402
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer, Power  # noqa: E402
from resdag.models import ott_esn  # noqa: E402


# ── Module-level callbacks (top-level so the runner stays picklable) ──────────
def model_creator(reservoir_size: int = 30, spectral_radius: float = 0.9):
    """Create a small Ott ESN for fast trials."""
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
    )


def search_space(trial):
    """Fixed search space so trials are cheap and deterministic."""
    return {"reservoir_size": 30, "spectral_radius": 0.9}


def data_loader(trial):
    """Deterministic synthetic data so trials are reproducible."""
    torch.manual_seed(0)
    data = torch.randn(1, 200, 3)
    return {
        "warmup": data[:, :20, :],
        "train": data[:, 20:120, :],
        "target": data[:, 21:121, :],
        "f_warmup": data[:, 120:140, :],
        "val": data[:, 140:180, :],  # 40-step horizon
    }


def nan_loss(y_true, y_pred, /, **kwargs):
    """Loss that always returns NaN (simulates a diverged forecast)."""
    return float("nan")


def inf_loss(y_true, y_pred, /, **kwargs):
    """Loss that always returns +inf (simulates a diverged forecast)."""
    return float("inf")


def _run_single_trial(loss_fn, *, direction: str = "minimize", penalty_value: float = 1e10):
    """Run a one-trial study with ``loss_fn`` and return the (study, trial)."""
    runner = build_objective(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        loss_fn=loss_fn,
        seed=1,
        penalty_value=penalty_value,
    )
    study = optuna.create_study(direction=direction)
    # ``catch=()`` ensures the trial is NOT rescued by Optuna's own exception
    # handling — any FAIL would be a genuine non-finite rejection.
    study.optimize(runner, n_trials=1, catch=())
    return study, study.trials[0]


class TestNonFiniteLossPenalty:
    """A non-finite loss completes with ``penalty_value`` instead of failing."""

    @pytest.mark.parametrize("loss_fn", [nan_loss, inf_loss], ids=["nan", "inf"])
    def test_nonfinite_loss_completes_with_penalty(self, loss_fn):
        """NaN/inf losses yield a COMPLETE trial valued at ``penalty_value``."""
        penalty = 1e10
        study, trial = _run_single_trial(loss_fn, penalty_value=penalty)

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value == penalty
        assert study.best_value == penalty

    @pytest.mark.parametrize("loss_fn", [nan_loss, inf_loss], ids=["nan", "inf"])
    def test_nonfinite_loss_records_user_attr(self, loss_fn):
        """The original non-finiteness is recorded as ``nonfinite_loss``."""
        _, trial = _run_single_trial(loss_fn)
        assert trial.user_attrs.get("nonfinite_loss") is True

    def test_maximize_direction_penalizes_negatively(self):
        """For a maximize study the penalty is sign-flipped to ``-penalty``."""
        penalty = 1e10
        study, trial = _run_single_trial(nan_loss, direction="maximize", penalty_value=penalty)

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value == -penalty
        assert study.best_value == -penalty

    def test_finite_loss_is_unchanged(self):
        """A finite loss passes through untouched and sets no penalty attr."""

        def finite_loss(y_true, y_pred, /, **kwargs):
            if isinstance(y_true, torch.Tensor):
                return float(torch.mean(torch.abs(y_true - y_pred)))
            return float(np.mean(np.abs(y_true - y_pred)))

        _, trial = _run_single_trial(finite_loss)

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value is not None
        assert np.isfinite(trial.value)
        assert "nonfinite_loss" not in trial.user_attrs


# ── Multi-output model callbacks (top-level so the runner stays picklable) ─────
def multi_output_model_creator(reservoir_size: int = 30, spectral_radius: float = 0.9):
    """Create a 2-output ESN whose ``forecast`` returns a tuple of tensors.

    The first output is the trained feedback readout (``name="output"``, the
    autoregression target the loss scores); the second is a parameterless
    ``Power`` branch derived from it, so the model needs only the single
    ``"output"`` target the runner supplies while still being genuinely
    multi-output (``forecast`` returns a 2-tuple).
    """
    inp = reservoir_input(3)
    res = ESNLayer(reservoir_size, feedback_size=3, spectral_radius=spectral_radius)(inp)
    cat = Concatenate()(inp, res)
    primary = CGReadoutLayer(3 + reservoir_size, 3, name="output")(cat)
    secondary = Power(2.0)(primary)
    return ESNModel(inp, [primary, secondary])


def mae_loss(y_true, y_pred, /, **kwargs):
    """Finite mean-absolute-error loss supporting tensor or NumPy inputs."""
    if isinstance(y_true, torch.Tensor):
        return float(torch.mean(torch.abs(y_true - y_pred)))
    return float(np.mean(np.abs(y_true - y_pred)))


class TestMultiOutputForecast:
    """Multi-output models complete with a finite loss scored on output 0."""

    def test_multi_output_forecast_is_a_tuple(self):
        """Sanity: the 2-output model's ``forecast`` returns a tuple of tensors.

        This guards the test's own premise — if the model ever became
        single-output, the regression below would pass vacuously.
        """
        model = multi_output_model_creator()
        data = data_loader(None)
        from resdag.training import ESNTrainer

        ESNTrainer(model).fit(
            warmup_inputs=(data["warmup"],),
            train_inputs=(data["train"],),
            targets={"output": data["target"]},
        )
        preds = model.forecast((data["f_warmup"],), horizon=data["val"].shape[1])

        assert isinstance(preds, tuple)
        assert len(preds) == 2
        assert all(isinstance(p, torch.Tensor) for p in preds)
        # First output is the feedback dimension (matched against ``val``).
        assert preds[0].shape[-1] == data["val"].shape[-1]

    def test_multi_output_trial_completes_with_finite_loss(self):
        """A 2-output model yields a COMPLETE trial with a finite, non-penalty loss.

        Without the tuple guard, ``forecast`` returns a tuple and the objective's
        ``preds.shape`` raises ``AttributeError``; under the default
        ``catch_exceptions=True`` the trial would silently complete at
        ``penalty_value``.  This asserts the loss is finite and *not* the penalty.
        """
        penalty = 1e10
        runner = build_objective(
            model_creator=multi_output_model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=1,
            penalty_value=penalty,
        )
        study = optuna.create_study(direction="minimize")
        # ``catch=()`` so any genuine AttributeError surfaces as a FAIL rather
        # than being rescued by Optuna's own exception handling.
        study.optimize(runner, n_trials=1, catch=())
        trial = study.trials[0]

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value is not None
        assert np.isfinite(trial.value)
        assert trial.value != penalty
        # No exception was caught and stashed by the runner.
        assert "error" not in trial.user_attrs

    def test_multi_output_does_not_raise_with_catch_disabled(self):
        """With ``catch_exceptions=False`` the tuple return must not raise.

        This is the strongest form of the guard check: it removes *both* safety
        nets (Optuna's ``catch`` and the runner's ``catch_exceptions``) so a
        tuple-indexing crash would propagate out of ``optimize`` as a hard error.
        """
        runner = build_objective(
            model_creator=multi_output_model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=1,
            catch_exceptions=False,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1, catch=())  # must not raise

        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE


# ── Constant-loss callbacks for clip / prune assertions ───────────────────────
def constant_loss_5(y_true, y_pred, /, **kwargs):
    """Deterministic loss that always returns ``5.0`` regardless of inputs.

    A fixed value makes the clip / prune thresholds exact: with ``clip_value``
    below ``5.0`` the trial clamps (or prunes) deterministically; above it the
    value passes through untouched.
    """
    return 5.0


class TestClipAndPrune:
    """``clip_value`` clamps the value and records ``raw_loss``; ``prune_on_clip`` prunes."""

    def _runner(self, *, clip_value=None, prune_on_clip=False):
        """Build a one-trial runner with a constant loss of ``5.0``."""
        return build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=constant_loss_5,
            seed=1,
            clip_value=clip_value,
            prune_on_clip=prune_on_clip,
        )

    def test_clip_clamps_value_and_records_raw_loss(self):
        """A raw loss above ``clip_value`` is clamped; ``raw_loss`` keeps the original."""
        runner = self._runner(clip_value=1.0)
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1, catch=())
        trial = study.trials[0]

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value == 1.0  # clamped to clip_value
        assert trial.user_attrs["raw_loss"] == 5.0  # unclipped value preserved
        assert trial.user_attrs["loss"] == 1.0  # logged loss is the clamped value

    def test_below_clip_passes_through_unchanged(self):
        """A raw loss at or below ``clip_value`` is returned verbatim."""
        runner = self._runner(clip_value=10.0)
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1, catch=())
        trial = study.trials[0]

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value == 5.0
        assert trial.user_attrs["raw_loss"] == 5.0

    def test_no_clip_value_leaves_raw_loss_equal_to_loss(self):
        """Without ``clip_value`` the returned loss equals the recorded ``raw_loss``."""
        runner = self._runner(clip_value=None)
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1, catch=())
        trial = study.trials[0]

        assert trial.value == 5.0
        assert trial.user_attrs["raw_loss"] == trial.user_attrs["loss"] == 5.0

    def test_prune_on_clip_prunes_exceeding_trial(self):
        """With ``prune_on_clip`` a clip-exceeding trial is PRUNED, not clamped."""
        runner = self._runner(clip_value=1.0, prune_on_clip=True)
        study = optuna.create_study(direction="minimize")
        # ``catch=()`` so a missed TrialPruned would surface as an error.
        study.optimize(runner, n_trials=1, catch=())
        trial = study.trials[0]

        assert trial.state == optuna.trial.TrialState.PRUNED
        # A pruned trial is excluded from the study's best value.
        with pytest.raises(ValueError):
            _ = study.best_value

    def test_prune_on_clip_keeps_below_threshold_trial(self):
        """``prune_on_clip`` does not prune a trial whose raw loss is within bound."""
        runner = self._runner(clip_value=10.0, prune_on_clip=True)
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1, catch=())
        trial = study.trials[0]

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value == 5.0


# ── Monitor-loss callbacks (top-level so the runner stays picklable) ──────────
def primary_loss(y_true, y_pred, /, **kwargs):
    """Finite MAE used as the optimized objective in monitor tests."""
    if isinstance(y_true, torch.Tensor):
        return float(torch.mean(torch.abs(y_true - y_pred)))
    return float(np.mean(np.abs(y_true - y_pred)))


def monitor_mse(y_true, y_pred, /, **kwargs):
    """Monitor loss with a stable ``__name__`` used to key ``monitor_params``.

    Returns mean-squared error; the optional ``offset`` kwarg (supplied via
    ``monitor_params`` keyed on this function's ``__name__``) is added to the
    result so the test can prove the kwargs were remapped and forwarded.
    """
    offset = kwargs.get("offset", 0.0)
    if isinstance(y_true, torch.Tensor):
        return float(torch.mean((y_true - y_pred) ** 2)) + offset
    return float(np.mean((y_true - y_pred) ** 2)) + offset


class TestMonitorLosses:
    """Monitor losses populate ``monitor_*`` attrs; ``monitor_params`` remap by name."""

    def _run(self, *, monitor_losses=None, monitor_params=None):
        """Run a one-trial study with the given monitor configuration."""
        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=primary_loss,
            seed=1,
            monitor_losses=monitor_losses,
            monitor_params=monitor_params,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1, catch=())
        return study.trials[0]

    def test_monitor_loss_populates_user_attr(self):
        """A monitor loss is logged as ``monitor_<__name__>`` and is finite."""
        trial = self._run(monitor_losses=[monitor_mse])

        key = f"monitor_{monitor_mse.__name__}"
        assert key in trial.user_attrs
        assert np.isfinite(trial.user_attrs[key])

    def test_monitor_loss_does_not_change_objective(self):
        """The optimized value is the primary loss, unaffected by the monitor."""
        with_monitor = self._run(monitor_losses=[monitor_mse])
        without_monitor = self._run(monitor_losses=None)

        # Same seed + same primary loss → identical objective value.
        assert with_monitor.value == without_monitor.value

    def test_monitor_params_remapped_by_function_name(self):
        """``monitor_params`` keyed by the loss ``__name__`` are forwarded as kwargs.

        ``monitor_mse`` adds its ``offset`` kwarg to the result, so a large offset
        keyed on the function name proves the remap and forwarding happened.
        """
        offset = 1000.0
        with_params = self._run(
            monitor_losses=[monitor_mse],
            monitor_params={monitor_mse.__name__: {"offset": offset}},
        )
        without_params = self._run(monitor_losses=[monitor_mse])

        key = f"monitor_{monitor_mse.__name__}"
        base = without_params.user_attrs[key]
        assert with_params.user_attrs[key] == pytest.approx(base + offset)

    def test_monitor_params_under_wrong_key_are_ignored(self):
        """Params keyed by anything other than the loss ``__name__`` are not applied.

        This pins the remap-by-``__name__`` contract: keying on a registry alias
        (e.g. ``"mse"``) rather than the function's real ``__name__`` is a no-op.
        """
        mislabelled = self._run(
            monitor_losses=[monitor_mse],
            monitor_params={"mse": {"offset": 1000.0}},  # wrong key
        )
        unparametrised = self._run(monitor_losses=[monitor_mse])

        key = f"monitor_{monitor_mse.__name__}"
        assert mislabelled.user_attrs[key] == pytest.approx(unparametrised.user_attrs[key])

    def test_real_registry_monitor_uses_canonical_name(self):
        """A registry loss is keyed by its real ``__name__``, not its alias.

        ``get_loss("efh")`` has ``__name__ == "expected_forecast_horizon"``; the
        logged attribute must use that canonical name, confirming the runner reads
        ``__name__`` rather than the ``"efh"`` alias the user passes to the loss
        registry.
        """
        from resdag.hpo import get_loss

        efh = get_loss("efh")
        trial = self._run(monitor_losses=[efh])

        assert f"monitor_{efh.__name__}" in trial.user_attrs
        assert "monitor_efh" not in trial.user_attrs


# ── Driven-model callbacks (top-level so the runner stays picklable) ──────────
def driven_model_creator(reservoir_size: int = 30, spectral_radius: float = 0.9):
    """Create an input-driven ESN: feedback (dim 3) + a 2-dim driver input.

    Mirrors the ``classic_esn`` topology but with a ``driver`` :class:`Input`
    wired into the reservoir's ``input_size``, so the model's ``forecast``
    requires a ``forecast_inputs`` driver series — exercising the ``drivers_keys``
    round-trip through training and forecasting.
    """
    feedback = reservoir_input(3)
    driver = reservoir_input(2)
    res = ESNLayer(
        reservoir_size,
        feedback_size=3,
        input_size=2,
        spectral_radius=spectral_radius,
    )(feedback, driver)
    cat = Concatenate()(feedback, res)
    readout = CGReadoutLayer(3 + reservoir_size, 3, name="output")(cat)
    return ESNModel([feedback, driver], readout)


def driven_data_loader(trial):
    """Synthetic data with the driver companions the ``drivers_keys`` flow needs.

    Alongside the five required feedback keys, supplies ``warmup_driver`` /
    ``train_driver`` / ``f_warmup_driver`` / ``forecast_driver`` (the
    ``<prefix>_<key>`` entries the runner threads through when
    ``drivers_keys=["driver"]``).
    """
    torch.manual_seed(0)
    fb = torch.randn(1, 200, 3)
    dr = torch.randn(1, 200, 2)
    return {
        "warmup": fb[:, :20, :],
        "train": fb[:, 20:120, :],
        "target": fb[:, 21:121, :],
        "f_warmup": fb[:, 120:140, :],
        "val": fb[:, 140:180, :],  # 40-step horizon
        "warmup_driver": dr[:, :20, :],
        "train_driver": dr[:, 20:120, :],
        "f_warmup_driver": dr[:, 120:140, :],
        "forecast_driver": dr[:, 140:180, :],  # spans the forecast horizon
    }


def misnamed_driver_data_loader(trial):
    """Driver data whose companions use a typo'd key (``drvr`` not ``driver``).

    Every feedback key is present and the driver tensors *are* supplied, but
    under ``warmup_drvr`` / ``train_drvr`` / ``f_warmup_drvr`` / ``forecast_drvr``
    instead of the ``*_driver`` names the runner expects for
    ``drivers_keys=["driver"]``.  Pre-fix this was silently dropped (the model
    trained feedback-only); now it must raise.
    """
    torch.manual_seed(0)
    fb = torch.randn(1, 200, 3)
    dr = torch.randn(1, 200, 2)
    return {
        "warmup": fb[:, :20, :],
        "train": fb[:, 20:120, :],
        "target": fb[:, 21:121, :],
        "f_warmup": fb[:, 120:140, :],
        "val": fb[:, 140:180, :],
        "warmup_drvr": dr[:, :20, :],  # typo: should be warmup_driver
        "train_drvr": dr[:, 20:120, :],
        "f_warmup_drvr": dr[:, 120:140, :],
        "forecast_drvr": dr[:, 140:180, :],
    }


class TestDriversKeysRoundTrip:
    """``drivers_keys`` thread driver inputs through training and forecasting."""

    def test_driven_trial_completes_with_finite_loss(self):
        """An input-driven model yields a COMPLETE trial with a finite, non-penalty loss."""
        penalty = 1e10
        runner = build_objective(
            model_creator=driven_model_creator,
            search_space=search_space,
            data_loader=driven_data_loader,
            loss_fn=mae_loss,
            drivers_keys=["driver"],
            seed=1,
            penalty_value=penalty,
            catch_exceptions=False,  # a driver wiring bug must surface, not penalize
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1, catch=())
        trial = study.trials[0]

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value is not None
        assert np.isfinite(trial.value)
        assert trial.value != penalty
        assert "error" not in trial.user_attrs


class TestDriversKeysValidation:
    """A declared driver key with no matching data entries raises, never drops silently."""

    def test_validate_data_keys_raises_on_misnamed_driver(self):
        """``_validate_data_keys`` rejects a declared driver whose entries are misnamed.

        The data dict has every feedback key and supplies the driver tensors, but
        under ``*_drvr`` rather than the ``*_driver`` names ``drivers_keys=["driver"]``
        expects.  The error must name the missing ``*_driver`` companions so the
        typo is obvious.
        """
        from resdag.hpo.objective import _validate_data_keys

        data = misnamed_driver_data_loader(None)
        with pytest.raises(KeyError) as excinfo:
            _validate_data_keys(data, drivers_keys=["driver"])

        message = str(excinfo.value)
        for expected in (
            "warmup_driver",
            "train_driver",
            "f_warmup_driver",
            "forecast_driver",
        ):
            assert expected in message

    def test_validate_data_keys_reports_only_the_missing_companion(self):
        """Only the absent companion is flagged; present ones are not listed.

        Dropping a single ``forecast_<key>`` (the entry an input-driven
        ``model.forecast`` *requires*) must be caught, while the three present
        companions are accepted — proving the check is per-companion, not all-or-nothing.
        """
        from resdag.hpo.objective import _validate_data_keys

        data = driven_data_loader(None)
        del data["forecast_driver"]
        with pytest.raises(KeyError, match="forecast_driver"):
            _validate_data_keys(data, drivers_keys=["driver"])

    def test_misnamed_driver_raises_through_the_runner(self):
        """End-to-end: a misnamed driver surfaces as an error, not a silent feedback-only run.

        With ``catch_exceptions=False`` the missing-companion ``KeyError`` must
        propagate out of ``optimize`` (``catch=()``) rather than the model
        training feedback-only and completing with a misleading score.
        """
        runner = build_objective(
            model_creator=driven_model_creator,
            search_space=search_space,
            data_loader=misnamed_driver_data_loader,
            loss_fn=mae_loss,
            drivers_keys=["driver"],
            seed=1,
            catch_exceptions=False,
        )
        study = optuna.create_study(direction="minimize")
        with pytest.raises(KeyError, match="forecast_driver"):
            study.optimize(runner, n_trials=1, catch=())


def seeded_model_creator(reservoir_size: int = 30, spectral_radius: float = 0.9, seed=None):
    """Seed-aware creator: threads ``seed`` into the reservoir initializers.

    Uses string-form ``topology`` and ``feedback_initializer`` so the recurrent
    matrix and feedback weights are drawn from their *own* RNGs — the exact
    initializers that ignore NumPy's legacy global state and torch's global RNG.
    Threading the per-trial ``seed`` through is therefore what makes a trial's
    objective a pure function of ``(base_seed, trial.number)``, so two studies
    with the same base seed reproduce and a different base seed diverges.
    """
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        topology="erdos_renyi",
        feedback_initializer="random",
        seed=seed,
    )


def stochastic_search_space(trial):
    """Search space with a genuine ``suggest_*`` so trials are non-degenerate.

    Sampling the spectral radius (rather than pinning it) gives each trial a
    distinct hyperparameter, so the per-trial seed has observable, trial-varying
    effects rather than collapsing every trial onto the same value.
    """
    return {
        "reservoir_size": 30,
        "spectral_radius": trial.suggest_float("spectral_radius", 0.7, 1.0),
    }


class TestPerTrialSeedDeterminism:
    """Two studies with the same base ``seed`` reproduce every trial value."""

    def _values(self, seed):
        """Run a 3-trial study with a seed-aware creator; return per-trial values."""
        runner = build_objective(
            model_creator=seeded_model_creator,
            search_space=stochastic_search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=seed,
        )
        # A fixed-seed sampler isolates the *runner's* per-trial seeding from the
        # sampler's own RNG, so any divergence is the runner's, not the sampler's.
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.RandomSampler(seed=0),
        )
        study.optimize(runner, n_trials=3, catch=())
        return [t.value for t in study.trials]

    def test_same_seed_reproduces_values(self):
        """Identical base seeds give identical per-trial values (and best_value)."""
        values_a = self._values(123)
        values_b = self._values(123)
        assert values_a == values_b
        assert min(values_a) == min(values_b)  # best_value reproduces

    def test_different_seed_diverges(self):
        """A different base seed changes at least one trial value."""
        assert self._values(123) != self._values(456)
