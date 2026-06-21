"""Tests for HPO warm-starting and cross-study transfer (issue #280).

Covers the public ``warm_start`` / ``transfer_from`` parameters of
:func:`resdag.hpo.run_hpo`, the :func:`resdag.hpo.export_best_config` helper, and
the underlying :mod:`resdag.hpo.transfer` building blocks.
"""

import logging
import tempfile
from pathlib import Path

import optuna
import torch
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState

from resdag.hpo import export_best_config, run_hpo
from resdag.hpo.transfer import apply_warm_start, transfer_trials
from resdag.models import ott_esn

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures: a tiny model creator / search space / data loader.
# ─────────────────────────────────────────────────────────────────────────────


def _model_creator(reservoir_size: int = 30, spectral_radius: float = 0.9) -> object:
    """Tiny ESN model creator for fast tests."""
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
    )


def _search_space(trial: optuna.Trial) -> dict:
    """Two-parameter search space."""
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 20, 50, step=10),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
    }


def _data_loader(trial: optuna.Trial) -> dict:
    """Small deterministic synthetic data."""
    torch.manual_seed(42)
    data = torch.randn(1, 150, 3)
    return {
        "warmup": data[:, :20, :],
        "train": data[:, 20:70, :],
        "target": data[:, 21:71, :],
        "f_warmup": data[:, 70:90, :],
        "val": data[:, 90:100, :],
    }


def _n_completed(study: optuna.Study) -> int:
    """Count COMPLETE trials in a study."""
    return len([t for t in study.trials if t.state == TrialState.COMPLETE])


def _make_prior_study(name: str = "prior") -> optuna.Study:
    """Build a small in-memory study with several COMPLETE trials."""
    return run_hpo(
        model_creator=_model_creator,
        search_space=_search_space,
        data_loader=_data_loader,
        n_trials=4,
        study_name=name,
        verbosity=0,
        seed=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# apply_warm_start (low-level)
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyWarmStart:
    """Unit tests for ``apply_warm_start``."""

    def test_enqueues_each_config(self) -> None:
        study = optuna.create_study()
        n = apply_warm_start(
            study,
            [
                {"reservoir_size": 30, "spectral_radius": 0.9},
                {"reservoir_size": 40, "spectral_radius": 1.0},
            ],
        )
        assert n == 2

    def test_empty_list_is_noop(self) -> None:
        study = optuna.create_study()
        assert apply_warm_start(study, []) == 0

    def test_skips_empty_dicts(self) -> None:
        study = optuna.create_study()
        assert apply_warm_start(study, [{}, {"x": 1.0}]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# transfer_trials (low-level)
# ─────────────────────────────────────────────────────────────────────────────


class TestTransferTrials:
    """Unit tests for ``transfer_trials``."""

    def test_transfers_all_completed(self) -> None:
        prior = _make_prior_study()
        n_prior = _n_completed(prior)

        dest = optuna.create_study()
        n = transfer_trials(dest, prior)

        assert n == n_prior
        assert _n_completed(dest) == n_prior

    def test_respects_param_name_filter(self) -> None:
        """Params outside the overlap set are dropped; trial still transfers."""
        prior = _make_prior_study()
        dest = optuna.create_study()

        # Only ``spectral_radius`` overlaps the (pretend) new search space.
        n = transfer_trials(dest, prior, param_names={"spectral_radius"})

        assert n == _n_completed(prior)
        for t in dest.trials:
            assert set(t.params.keys()) == {"spectral_radius"}

    def test_logs_mismatched_params(self, caplog) -> None:
        prior = _make_prior_study()
        dest = optuna.create_study()

        with caplog.at_level(logging.INFO, logger="resdag.hpo.transfer"):
            transfer_trials(dest, prior, param_names={"spectral_radius"})

        # ``reservoir_size`` is mismatched and must be reported at INFO.
        assert any("mismatched" in rec.message for rec in caplog.records)
        assert any("reservoir_size" in rec.getMessage() for rec in caplog.records)

    def test_skips_trial_with_no_overlap(self) -> None:
        """A source trial with zero overlapping params is skipped entirely."""
        prior = _make_prior_study()
        dest = optuna.create_study()

        n = transfer_trials(dest, prior, param_names={"nonexistent_param"})

        assert n == 0
        assert _n_completed(dest) == 0

    def test_only_completed_trials_transferred(self) -> None:
        """PRUNED / FAIL source trials are not copied."""
        source = optuna.create_study()
        good = optuna.trial.create_trial(
            state=TrialState.COMPLETE,
            value=1.0,
            params={"x": 0.5},
            distributions={"x": FloatDistribution(0.0, 1.0)},
        )
        pruned = optuna.trial.create_trial(
            state=TrialState.PRUNED,
            params={"x": 0.2},
            distributions={"x": FloatDistribution(0.0, 1.0)},
        )
        source.add_trial(good)
        source.add_trial(pruned)

        dest = optuna.create_study()
        n = transfer_trials(dest, source)
        assert n == 1
        assert _n_completed(dest) == 1

    def test_preserves_values(self) -> None:
        """Transferred trials keep their evaluated objective value."""
        source = optuna.create_study()
        source.add_trial(
            optuna.trial.create_trial(
                state=TrialState.COMPLETE,
                value=3.14,
                params={"x": 2.0, "y": 5},
                distributions={
                    "x": FloatDistribution(0.0, 10.0),
                    "y": IntDistribution(0, 10),
                },
            )
        )
        dest = optuna.create_study()
        transfer_trials(dest, source)

        assert _n_completed(dest) == 1
        assert dest.trials[0].value == 3.14

    def test_from_storage_path(self) -> None:
        """``transfer_from`` accepts a storage path holding one study."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "prior.db"
            url = f"sqlite:///{db}"
            run_hpo(
                model_creator=_model_creator,
                search_space=_search_space,
                data_loader=_data_loader,
                n_trials=3,
                storage=url,
                study_name="prior_on_disk",
                verbosity=0,
                seed=7,
            )

            dest = optuna.create_study()
            n = transfer_trials(dest, url)
            assert n == 3


# ─────────────────────────────────────────────────────────────────────────────
# export_best_config
# ─────────────────────────────────────────────────────────────────────────────


class TestExportBestConfig:
    """Unit tests for ``export_best_config``."""

    def test_returns_best_params(self) -> None:
        prior = _make_prior_study()
        cfg = export_best_config(prior)
        assert cfg == prior.best_params

    def test_round_trips_into_warm_start(self) -> None:
        """The exported config can be enqueued and is evaluated as a trial."""
        prior = _make_prior_study()
        best = export_best_config(prior)

        study = run_hpo(
            model_creator=_model_creator,
            search_space=_search_space,
            data_loader=_data_loader,
            n_trials=2,
            warm_start=[best],
            study_name="round_trip",
            verbosity=0,
            seed=2,
        )
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        assert any(
            t.params.get("reservoir_size") == best["reservoir_size"]
            and abs(t.params.get("spectral_radius", 1e9) - best["spectral_radius"]) < 1e-9
            for t in completed
        )

    def test_raises_without_completed_trials(self) -> None:
        study = optuna.create_study()
        try:
            export_best_config(study)
        except ValueError as exc:
            assert "no completed trials" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError")


# ─────────────────────────────────────────────────────────────────────────────
# run_hpo integration
# ─────────────────────────────────────────────────────────────────────────────


class TestRunHPOWarmStart:
    """Integration tests for ``warm_start`` / ``transfer_from`` via ``run_hpo``."""

    def test_warm_start_runs_before_sampled(self) -> None:
        """AC1: warm-start trials run before any sampler-proposed trial."""
        warm = {"reservoir_size": 50, "spectral_radius": 0.6}
        study = run_hpo(
            model_creator=_model_creator,
            search_space=_search_space,
            data_loader=_data_loader,
            n_trials=3,
            warm_start=[warm],
            study_name="ws_order",
            verbosity=0,
            seed=3,
        )
        # The very first trial must be the enqueued warm-start configuration.
        first = study.trials[0]
        assert first.params["reservoir_size"] == warm["reservoir_size"]
        assert abs(first.params["spectral_radius"] - warm["spectral_radius"]) < 1e-9

    def test_transfer_from_ingests_prior(self) -> None:
        """AC2: transfer_from ingests prior completed trials (by count)."""
        prior = _make_prior_study()
        n_prior = _n_completed(prior)

        study = run_hpo(
            model_creator=_model_creator,
            search_space=_search_space,
            data_loader=_data_loader,
            n_trials=2,
            transfer_from=prior,
            study_name="transfer_int",
            verbosity=0,
            seed=4,
        )
        # The n_prior prior COMPLETE trials were ingested verbatim, plus 2 new
        # trials were run on top (count by trial number to be pruning-robust).
        assert _n_completed(study) >= n_prior
        new_trials = [t for t in study.trials if t.number >= n_prior]
        assert len(new_trials) == 2

    def test_transfer_does_not_consume_budget(self) -> None:
        """Transferred seeds are not counted against ``n_trials`` (new runs)."""
        prior = _make_prior_study()
        n_prior = _n_completed(prior)

        study = run_hpo(
            model_creator=_model_creator,
            search_space=_search_space,
            data_loader=_data_loader,
            n_trials=3,
            transfer_from=prior,
            study_name="transfer_budget",
            verbosity=0,
            seed=5,
        )
        # n_prior transferred + exactly 3 new trials of any terminal state.
        new_trials = [t for t in study.trials if t.number >= n_prior]
        assert len(new_trials) == 3

    def test_transfer_filters_mismatched_params(self, caplog) -> None:
        """AC3: mismatched params are filtered out of transferred trials."""
        # Prior study has an extra ``extra_knob`` parameter the new space lacks.
        prior = optuna.create_study(study_name="prior_extra")
        for i in range(3):
            prior.add_trial(
                optuna.trial.create_trial(
                    state=TrialState.COMPLETE,
                    value=float(i),
                    params={
                        "reservoir_size": 30,
                        "spectral_radius": 0.8,
                        "extra_knob": 1.5,
                    },
                    distributions={
                        "reservoir_size": IntDistribution(20, 50, step=10),
                        "spectral_radius": FloatDistribution(0.5, 1.2),
                        "extra_knob": FloatDistribution(0.0, 2.0),
                    },
                )
            )

        with caplog.at_level(logging.INFO, logger="resdag.hpo.transfer"):
            study = run_hpo(
                model_creator=_model_creator,
                search_space=_search_space,
                data_loader=_data_loader,
                n_trials=1,
                transfer_from=prior,
                study_name="transfer_filter",
                verbosity=0,
                seed=6,
            )

        # ``extra_knob`` must not appear on any transferred trial.
        for t in study.trials:
            assert "extra_knob" not in t.params
        assert any("extra_knob" in rec.getMessage() for rec in caplog.records)

    def test_warm_start_and_transfer_combined(self) -> None:
        """Warm-start and transfer compose in a single run."""
        prior = _make_prior_study()
        n_prior = _n_completed(prior)
        best = export_best_config(prior)

        study = run_hpo(
            model_creator=_model_creator,
            search_space=_search_space,
            data_loader=_data_loader,
            n_trials=2,
            warm_start=[best],
            transfer_from=prior,
            study_name="combined",
            verbosity=0,
            seed=8,
        )
        # n_prior transferred seeds + exactly 2 new trials (any terminal state;
        # sampled trials may be pruned). The first new trial is the warm-start.
        new_trials = [t for t in study.trials if t.number >= n_prior]
        assert len(new_trials) == 2
        first_new = study.trials[n_prior]
        assert first_new.params["reservoir_size"] == best["reservoir_size"]


class TestRunHPOWarmStartMultiprocess:
    """AC3: both features work with ``n_workers > 1``."""

    def test_transfer_and_warm_start_multiprocess(self) -> None:
        prior = _make_prior_study()
        n_prior = _n_completed(prior)
        best = export_best_config(prior)

        with tempfile.TemporaryDirectory() as tmp:
            storage = str(Path(tmp) / "mp_study.log")
            study = run_hpo(
                model_creator=_model_creator,
                search_space=_search_space,
                data_loader=_data_loader,
                n_trials=4,
                warm_start=[best],
                transfer_from=prior,
                n_workers=2,
                storage=storage,
                study_name="mp_warm",
                verbosity=0,
                seed=9,
            )
            # Read everything from the journal-backed study *before* the temp
            # directory (and its .log file) is removed on context exit.
            trials = list(study.trials)

        # The n_prior transferred seeds survived the pre-fork ingest: they are
        # present as COMPLETE trials carrying values, distinct from the new run.
        seed_trials = [t for t in trials if t.number < n_prior]
        assert len(seed_trials) == n_prior
        assert all(t.state == TrialState.COMPLETE for t in seed_trials)

        # New trials were run on top of the seeds under multiple workers. Trials
        # may prune (pruned trials do not count towards the completed budget),
        # so assert progress was made rather than an exact count.
        new_trials = [t for t in trials if t.number >= n_prior]
        assert len(new_trials) >= 1

        # The warm-start config (enqueued pre-fork) was evaluated by a worker.
        assert any(
            t.params.get("reservoir_size") == best["reservoir_size"]
            and abs(t.params.get("spectral_radius", 1e9) - best["spectral_radius"]) < 1e-9
            for t in trials
            if t.state == TrialState.COMPLETE
        )
