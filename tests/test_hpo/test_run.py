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
        for loss_name in ["efh", "horizon", "lyap", "standard", "discounted"]:
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
        """Lyapunov loss with custom LLE."""
        study = run_hpo(
            model_creator=simple_model_creator,
            search_space=simple_search_space,
            data_loader=simple_data_loader,
            n_trials=1,
            loss="lyap",
            loss_params={"lle": 0.5, "dt": 0.02},
            verbosity=0,
        )
        assert study.best_value is not None
