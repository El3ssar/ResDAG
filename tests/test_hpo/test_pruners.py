"""Tests for first-class Optuna pruners in HPO (issue #278).

Covers:

- the ``{"asha", "hyperband", "median", "threshold", "none"}`` registry and
  :func:`resdag.hpo.resolve_pruner` resolution (keys, instances, ``None``, and
  error cases);
- that a **divergent** study run with a pruner records at least one ``PRUNED``
  trial and performs fewer intermediate-report steps than the same study with
  ``pruner=None`` — in **both** single- and multi-process modes;
- that :func:`resdag.hpo.get_study_summary` surfaces the ``PRUNED`` count;
- that ``pruner`` coexists with the existing ``clip_value`` / ``prune_on_clip``
  early-stopping path.

All model / search-space / data-loader callables are defined at *module level*
so they are picklable — required for the ``spawn`` start method used by the
multi-process backend.
"""

import pytest
import torch

optuna = pytest.importorskip("optuna")

from resdag.hpo import PRUNERS, get_study_summary, resolve_pruner, run_hpo  # noqa: E402
from resdag.models import ott_esn  # noqa: E402


# ── Picklable top-level fixtures (spawn-safe) ─────────────────────────────────
def _model_creator(reservoir_size: int = 40, spectral_radius: float = 0.9, seed=None):
    """Seed-aware creator whose reservoir diverges at large spectral radius."""
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        topology="erdos_renyi",
        feedback_initializer="random",
        seed=seed,
    )


def _divergent_search_space(trial):
    """Wide spectral-radius range so many configurations diverge early.

    A large spectral radius pushes the reservoir past the echo-state regime, so
    its autoregressive forecast blows up within the first few horizon
    checkpoints — exactly the early, large intermediate loss a pruner is meant
    to catch.
    """
    return {"spectral_radius": trial.suggest_float("spectral_radius", 0.8, 3.0)}


def _data_loader(trial):
    """Small deterministic multi-sine dataset (B=1, D=3)."""
    torch.manual_seed(0)
    t = torch.linspace(0, 30, 400)
    x = torch.stack([torch.sin(t), torch.cos(1.3 * t), torch.sin(0.7 * t)], dim=-1).unsqueeze(0)
    return {
        "warmup": x[:, :50],
        "train": x[:, 50:250],
        "target": x[:, 51:251],
        "f_warmup": x[:, 250:300],
        "val": x[:, 300:400],
    }


def _n_pruned(study: optuna.Study) -> int:
    """Number of ``PRUNED`` trials in *study*."""
    return len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])


def _n_report_steps(study: optuna.Study) -> int:
    """Total intermediate report steps across all trials.

    Each horizon checkpoint that a trial reaches contributes one entry to its
    ``intermediate_values``; a pruned trial stops early and therefore reports
    fewer steps, so the study-wide sum is the natural "work done" proxy.
    """
    return sum(len(t.intermediate_values) for t in study.trials)


# ── Registry / resolution ─────────────────────────────────────────────────────
class TestResolvePruner:
    """The pruner registry and :func:`resolve_pruner`."""

    def test_registry_keys(self):
        """The registry exposes exactly the documented keys."""
        assert set(PRUNERS) == {"asha", "hyperband", "median", "threshold", "none"}

    @pytest.mark.parametrize(
        "key, cls",
        [
            ("asha", optuna.pruners.SuccessiveHalvingPruner),
            ("hyperband", optuna.pruners.HyperbandPruner),
            ("median", optuna.pruners.MedianPruner),
            ("threshold", optuna.pruners.ThresholdPruner),
            ("none", optuna.pruners.NopPruner),
        ],
    )
    def test_string_keys_resolve(self, key, cls):
        """Each registry key resolves to the matching Optuna pruner class."""
        assert isinstance(resolve_pruner(key), cls)

    def test_case_insensitive(self):
        """Keys are matched case-insensitively."""
        assert isinstance(resolve_pruner("MEDIAN"), optuna.pruners.MedianPruner)

    def test_none_resolves_to_nop(self):
        """``None`` resolves to the no-op pruner (default behaviour)."""
        assert isinstance(resolve_pruner(None), optuna.pruners.NopPruner)

    def test_instance_passthrough(self):
        """A pre-built ``BasePruner`` is returned unchanged (fully configurable)."""
        custom = optuna.pruners.MedianPruner(n_startup_trials=7, n_warmup_steps=3)
        assert resolve_pruner(custom) is custom

    def test_unknown_key_raises(self):
        """An unknown string key raises ``ValueError`` listing the valid keys."""
        with pytest.raises(ValueError, match="Unknown pruner"):
            resolve_pruner("does_not_exist")

    def test_wrong_type_raises(self):
        """A non-str / non-pruner / non-None argument raises ``TypeError``."""
        with pytest.raises(TypeError):
            resolve_pruner(123)  # type: ignore[arg-type]


# ── End-to-end pruning behaviour ──────────────────────────────────────────────
class TestPruningSingleProcess:
    """A pruner early-stops divergent trials in single-process mode (AC2, AC3)."""

    def test_pruner_records_pruned_and_runs_fewer_steps(self):
        """AC2: ≥1 PRUNED trial and fewer report steps than ``pruner=None``."""
        baseline = run_hpo(
            _model_creator,
            _divergent_search_space,
            _data_loader,
            n_trials=12,
            seed=1,
            pruner=None,
            verbosity=0,
        )
        pruned = run_hpo(
            _model_creator,
            _divergent_search_space,
            _data_loader,
            n_trials=12,
            seed=1,
            pruner="median",
            verbosity=0,
        )

        assert _n_pruned(baseline) == 0
        assert _n_pruned(pruned) >= 1
        assert _n_report_steps(pruned) < _n_report_steps(baseline)

    def test_pruner_instance_accepted(self):
        """AC1: a configured ``BasePruner`` instance is honored end to end."""
        study = run_hpo(
            _model_creator,
            _divergent_search_space,
            _data_loader,
            n_trials=10,
            seed=2,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0),
            verbosity=0,
        )
        assert _n_pruned(study) >= 1


@pytest.mark.slow
class TestPruningMultiProcess:
    """AC3: the pruner is honored in multi-process mode too."""

    def test_pruner_records_pruned_multiprocess(self, tmp_path):
        """A divergent multi-process study with a pruner records ≥1 PRUNED trial
        and fewer report steps than the same study with ``pruner=None``."""
        baseline = run_hpo(
            _model_creator,
            _divergent_search_space,
            _data_loader,
            n_trials=12,
            n_workers=2,
            seed=1,
            pruner=None,
            storage=str(tmp_path / "baseline.log"),
            study_name="mp_baseline",
            verbosity=0,
        )
        pruned = run_hpo(
            _model_creator,
            _divergent_search_space,
            _data_loader,
            n_trials=12,
            n_workers=2,
            seed=1,
            pruner="median",
            storage=str(tmp_path / "pruned.log"),
            study_name="mp_pruned",
            verbosity=0,
        )

        assert _n_pruned(baseline) == 0
        assert _n_pruned(pruned) >= 1
        assert _n_report_steps(pruned) < _n_report_steps(baseline)


# ── Summary + coexistence with clip path ──────────────────────────────────────
class TestSummaryAndCoexistence:
    """``get_study_summary`` surfaces PRUNED; pruner coexists with clipping."""

    def test_summary_surfaces_pruned_count(self):
        """AC: the study summary reports the number of pruned trials."""
        study = run_hpo(
            _model_creator,
            _divergent_search_space,
            _data_loader,
            n_trials=12,
            seed=1,
            pruner="median",
            verbosity=0,
        )
        summary = get_study_summary(study)
        assert "Pruned:" in summary
        # The reported count matches the actual number of pruned trials.
        assert f"Pruned: {_n_pruned(study)}" in summary

    def test_pruner_coexists_with_clip(self):
        """The pruner and ``clip_value`` / ``prune_on_clip`` work together.

        Both early-stopping paths active at once must not crash and must still
        prune at least one trial.
        """
        study = run_hpo(
            _model_creator,
            _divergent_search_space,
            _data_loader,
            n_trials=12,
            seed=1,
            pruner="median",
            clip_value=1.0,
            prune_on_clip=True,
            verbosity=0,
        )
        assert study.best_value is not None
        assert _n_pruned(study) >= 1
