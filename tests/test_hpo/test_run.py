"""Tests for the HPO run_hpo function."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from resdag.hpo import get_study_summary, run_hpo
from resdag.models import ott_esn


def simple_model_creator(reservoir_size: int = 50, spectral_radius: float = 0.9):
    """Simple model creator for testing."""
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
    )


def simple_search_space(trial):
    """Simple search space for testing."""
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 20, 50, step=10),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
    }


def simple_data_loader(trial):
    """Simple data loader for testing."""
    torch.manual_seed(42)
    # Small synthetic data (B=1, T=100, D=3)
    data = torch.randn(1, 150, 3)
    return {
        "warmup": data[:, :20, :],
        "train": data[:, 20:70, :],
        "target": data[:, 21:71, :],  # Target shifted by 1
        "f_warmup": data[:, 70:90, :],
        "val": data[:, 90:100, :],
    }


def seeded_model_creator(reservoir_size: int = 30, spectral_radius: float = 0.9, seed=None):
    """Seed-aware creator: forwards ``seed`` into the reservoir initializers.

    Uses string-form ``topology`` and ``feedback_initializer`` so the recurrent
    matrix and feedback weights are drawn from their own RNGs — exactly the
    initializers that ignore NumPy's legacy global state.  Threading ``seed``
    through is what makes them reproducible per trial.
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


def rng_consuming_data_loader(trial):
    """Data loader that advances the global RNGs non-deterministically.

    Draws a random number of throwaway samples from torch / NumPy / ``random``
    *after* the deterministic data build, so the global RNG state at
    ``model_creator`` time depends on entropy the per-trial seed does not
    control.  Without the per-trial seed being threaded *directly* into
    ``model_creator``, the reservoir initializers would therefore vary run to
    run; this loader makes the reproducibility test meaningful rather than an
    artifact of a deterministic loader.
    """
    import random as _random

    torch.manual_seed(42)
    data = torch.randn(1, 150, 3)
    # Perturb every global RNG by a random amount drawn from a fresh,
    # unseeded generator so it differs across otherwise-identical runs.
    burn = int(torch.randint(0, 17, (1,), generator=None).item())
    for _ in range(burn + 1):
        torch.rand(1)
        np.random.rand()
        _random.random()
    return {
        "warmup": data[:, :20, :],
        "train": data[:, 20:70, :],
        "target": data[:, 21:71, :],
        "f_warmup": data[:, 70:90, :],
        "val": data[:, 90:100, :],
    }


class TestRunHPOBasic:
    """Basic tests for run_hpo function."""

    def test_basic_run(self):
        """Basic HPO run completes successfully."""
        study = run_hpo(
            model_creator=simple_model_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=2,
            verbosity=0,
        )
        assert len(study.trials) == 2
        assert study.best_value is not None

    def test_returns_study(self):
        """run_hpo returns an optuna Study."""
        import optuna

        study = run_hpo(
            model_creator=simple_model_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=1,
            verbosity=0,
        )
        assert isinstance(study, optuna.Study)

    def test_custom_loss_string(self):
        """Custom loss by string works."""
        for loss_name in ["efh", "forecast_horizon", "lyapunov", "standard", "soft_horizon"]:
            study = run_hpo(
                model_creator=simple_model_creator,
                search_space=simple_search_space,
                data_loader=simple_data_loader,
                n_trials=1,
                loss=loss_name,
                verbosity=0,
            )
            assert study.best_value is not None

    def test_custom_loss_callable(self):
        """Custom loss callable works."""

        def my_loss(y_true, y_pred):
            return float(np.mean(np.abs(y_true - y_pred)))

        study = run_hpo(
            model_creator=simple_model_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=1,
            loss=my_loss,
            verbosity=0,
        )
        assert study.best_value is not None


class TestRunHPOPersistence:
    """Test study persistence."""

    def test_sqlite_storage(self):
        """Study can be saved to SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_study.db"

            study = run_hpo(
                model_creator=simple_model_creator,
                search_space=simple_search_space,
                data_loader=simple_data_loader,
                n_trials=2,
                storage=f"sqlite:///{db_path}",
                study_name="test_persistence",
                verbosity=0,
            )

            assert db_path.exists()
            assert len(study.trials) == 2

    def test_resume_study(self):
        """Study can be resumed from storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_resume.db"
            storage_url = f"sqlite:///{db_path}"

            # Run first batch
            study1 = run_hpo(
                model_creator=simple_model_creator,
                search_space=simple_search_space,
                data_loader=simple_data_loader,
                n_trials=2,
                storage=storage_url,
                study_name="resume_test",
                verbosity=0,
            )
            assert len(study1.trials) == 2

            # Resume
            study2 = run_hpo(
                model_creator=simple_model_creator,
                search_space=simple_search_space,
                data_loader=simple_data_loader,
                n_trials=4,
                storage=storage_url,
                study_name="resume_test",
                verbosity=0,
            )
            # Should have 4 total (2 new + 2 existing)
            assert len(study2.trials) == 4


class TestRunHPOValidation:
    """Test input validation."""

    def test_invalid_n_trials(self):
        """Invalid n_trials raises ValueError."""
        with pytest.raises(ValueError):
            run_hpo(
                model_creator=simple_model_creator,
                search_space=simple_search_space,
                data_loader=simple_data_loader,
                n_trials=0,
                verbosity=0,
            )

    def test_non_callable_model_creator(self):
        """Non-callable model_creator raises TypeError."""
        with pytest.raises(TypeError):
            run_hpo(
                model_creator="not a callable",
                search_space=simple_search_space,
                data_loader=simple_data_loader,
                n_trials=1,
                verbosity=0,
            )


class TestStudySummary:
    """Test get_study_summary function."""

    def test_summary_format(self):
        """Summary is formatted correctly."""
        study = run_hpo(
            model_creator=simple_model_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=2,
            verbosity=0,
        )

        summary = get_study_summary(study)
        assert "Study Summary" in summary
        assert "Best Trial" in summary
        assert "Parameters" in summary
        assert "Top" in summary

    def test_summary_with_custom_top_n(self):
        """Summary respects top_n parameter."""
        study = run_hpo(
            model_creator=simple_model_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=3,
            verbosity=0,
        )

        summary = get_study_summary(study, top_n=2)
        assert "Top 2" in summary


class TestRunHPONoLoggingSideEffects:
    """AC4: run_hpo must not mutate the host application's root logger."""

    def test_does_not_call_basic_config(self):
        """run_hpo never calls logging.basicConfig (which mutates root logger)."""
        import logging
        from unittest import mock

        with mock.patch.object(logging, "basicConfig") as basic_config:
            run_hpo(
                model_creator=simple_model_creator,
                search_space=simple_search_space,
                data_loader=simple_data_loader,
                n_trials=1,
                verbosity=1,
            )
        basic_config.assert_not_called()

    def test_root_logger_untouched(self):
        """The root logger's level and handlers are unchanged after run_hpo."""
        import logging

        root = logging.getLogger()
        before_level = root.level
        before_handlers = list(root.handlers)

        run_hpo(
            model_creator=simple_model_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=1,
            verbosity=2,
        )

        assert root.level == before_level
        assert list(root.handlers) == before_handlers


class TestRunHPOLossParams:
    """Test loss_params functionality."""

    def test_loss_params_passed(self):
        """loss_params are passed to loss function."""
        study = run_hpo(
            model_creator=simple_model_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=1,
            loss="efh",
            loss_params={"threshold": 0.1, "softness": 0.01},
            verbosity=0,
        )
        assert study.best_value is not None

    def test_loss_params_lyapunov(self):
        """Lyapunov loss with custom lyapunov_t."""
        study = run_hpo(
            model_creator=simple_model_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=1,
            loss="lyapunov",
            loss_params={"lyapunov_t": 10},
            verbosity=0,
        )
        assert study.best_value is not None


class TestRunHPOReproducibility:
    """#273: a seeded study reproduces end to end, reservoir RNG included."""

    def _run(self, seed, model_creator=seeded_model_creator, data_loader=simple_data_loader):
        return run_hpo(
            model_creator=model_creator,
            search_space=simple_search_space,
            data_loader=data_loader,
            n_trials=4,
            seed=seed,
            verbosity=0,
        )

    def test_same_seed_identical_best_value(self):
        """AC1: two runs with the same seed give identical best_value."""
        study_a = self._run(42)
        study_b = self._run(42)
        assert study_a.best_value == study_b.best_value

    def test_same_seed_identical_trials(self):
        """Same seed reproduces every trial's params and value (not just the best)."""
        study_a = self._run(42)
        study_b = self._run(42)

        params_a = [t.params for t in study_a.trials]
        params_b = [t.params for t in study_b.trials]
        assert params_a == params_b

        values_a = [t.value for t in study_a.trials]
        values_b = [t.value for t in study_b.trials]
        assert values_a == values_b

    def test_different_seed_differs(self):
        """A different seed yields a different best_value."""
        study_a = self._run(42)
        study_c = self._run(123)
        assert study_a.best_value != study_c.best_value

    def test_reproducible_despite_global_rng_consumption(self):
        """The per-trial seed reaches the reservoir even when the data_loader
        burns the global RNGs by a run-varying amount.

        This is the crux of #273: the topology / feedback initializers build
        their own RNGs and ignore NumPy's legacy global state, so reproducibility
        must come from the per-trial seed threaded *into* model_creator, not from
        the pre-model global RNG state.
        """
        study_a = self._run(7, data_loader=rng_consuming_data_loader)
        study_b = self._run(7, data_loader=rng_consuming_data_loader)
        assert study_a.best_value == study_b.best_value
        assert [t.value for t in study_a.trials] == [t.value for t in study_b.trials]

    def test_seed_unaware_creator_still_runs(self):
        """A model_creator without a ``seed`` parameter is called unchanged.

        The seed is injected only when ``model_creator`` accepts it, so legacy
        creators keep working (backward compatibility).
        """
        study = self._run(42, model_creator=simple_model_creator)
        assert len(study.trials) == 4
        assert study.best_value is not None

    def test_explicit_seed_in_params_wins(self):
        """A ``seed`` returned by search_space is not overwritten by the trial seed."""

        def search_space_with_seed(trial):
            return {
                "reservoir_size": trial.suggest_int("reservoir_size", 20, 40, step=10),
                "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
                "seed": 999,  # pinned by the user; must survive injection
            }

        seen_seeds = []

        def recording_creator(reservoir_size, spectral_radius, seed=None):
            seen_seeds.append(seed)
            return seeded_model_creator(reservoir_size, spectral_radius, seed=seed)

        run_hpo(
            model_creator=recording_creator,
            search_space=search_space_with_seed,
            data_loader=simple_data_loader,
            n_trials=2,
            seed=42,
            verbosity=0,
        )
        assert seen_seeds, "model_creator was never called"
        assert all(s == 999 for s in seen_seeds)

    def test_no_seed_does_not_inject(self):
        """When no base seed is set, no ``seed`` kwarg is injected into the creator."""
        seen_seeds = []

        def recording_creator(reservoir_size, spectral_radius, seed="UNSET"):
            seen_seeds.append(seed)
            return seeded_model_creator(reservoir_size, spectral_radius, seed=None)

        run_hpo(
            model_creator=recording_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=2,
            seed=None,
            verbosity=0,
        )
        assert seen_seeds, "model_creator was never called"
        # The default sentinel survives → no injection happened.
        assert all(s == "UNSET" for s in seen_seeds)
