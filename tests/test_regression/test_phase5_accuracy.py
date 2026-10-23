"""Scientific-evidence guards for the isolated accuracy comparison runner."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "accuracy_regression.py"
SPEC = importlib.util.spec_from_file_location("resdag_accuracy_regression", PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _matched_reports():
    report = {
        "torch": "same",
        "numpy": "same",
        "device": "cpu",
        "dtype": "float64",
        "threads": 1,
        "interop_threads": 1,
        "raw_data_sha256": {"lorenz": "same-data"},
        "rows": [
            {
                "id": "classic-seed7-origin0",
                "origin_index": 2701,
                "history_steps": 500,
                "dt": 0.02,
                "lambda_max": 0.9056,
                "horizon_steps": 553,
            }
        ],
    }
    states = {
        "classic-seed7": {
            "recipe": {"spectral_radius": 0.9},
            "washout": 200,
            "observations": 2701,
            "initial_graph": {"weight": {"sha256": "same-initial-weights"}},
            "fitted_model": {"weight": {"sha256": "old-fitted-weights"}},
        }
    }
    return report, copy.deepcopy(report), states, copy.deepcopy(states)


@pytest.mark.parametrize("field", ["torch", "numpy", "dtype", "threads", "raw_data_sha256"])
def test_unmatched_environment_or_data_cannot_establish_accuracy_regression(field):
    before, after, old, new = _matched_reports()
    after[field] = "different"
    with pytest.raises(ValueError, match=f"Unmatched environment field: {field}"):
        RUNNER._assert_matched(before, after, old, new)


@pytest.mark.parametrize("field", ["recipe", "washout", "observations", "initial_graph"])
def test_changed_initial_weights_or_recipe_cannot_establish_causal_regression(field):
    before, after, old, new = _matched_reports()
    new["classic-seed7"][field] = "different"
    with pytest.raises(ValueError, match=f"Unmatched {field}"):
        RUNNER._assert_matched(before, after, old, new)


@pytest.mark.parametrize(
    "field", ["origin_index", "history_steps", "dt", "lambda_max", "horizon_steps"]
)
def test_forecast_clock_and_origins_must_match(field):
    before, after, old, new = _matched_reports()
    after["rows"][0][field] += 1
    with pytest.raises(ValueError, match=f"Unmatched {field}"):
        RUNNER._assert_matched(before, after, old, new)


def test_fitted_weight_changes_are_measured_instead_of_rejected_as_unmatched_setup():
    before, after, old, new = _matched_reports()
    new["classic-seed7"]["fitted_model"]["weight"]["sha256"] = "new-fitted-weights"
    RUNNER._assert_matched(before, after, old, new)


def test_independent_horizon_stops_at_first_failure_and_uses_fixed_feature_scales():
    truth = np.zeros((5, 2))
    scale = np.array([2.0, 20.0])
    prediction = np.array([[0, 0], [0.25, 2.5], [0.5, 5.0], [0, 0], [0, 0]])
    # Exact normalized RMS reaches 0.25 on row 2: strict '<' excludes it,
    # and later good samples cannot repair a broken autonomous prefix.
    assert RUNNER._independent_horizon(prediction, truth, scale, 0.25) == 2
    assert RUNNER._independent_horizon(prediction[:2], truth[:2], scale, 0.25) == 2
    prediction[0, 0] = np.nan
    assert RUNNER._independent_horizon(prediction, truth, scale, 0.25) == 0


def test_existing_source_archives_are_rejected_before_overwriting_data(tmp_path, monkeypatch):
    (tmp_path / "before").mkdir()
    existing = tmp_path / "lorenz.npy"
    existing.write_bytes(b"keep these observed samples")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(PATH), "--artifacts-dir", str(tmp_path), "--output", str(tmp_path / "report.json")],
    )
    with pytest.raises(ValueError, match="fresh artifacts-dir"):
        RUNNER.main()
    assert existing.read_bytes() == b"keep these observed samples"


def test_bit_identity_distinguishes_signed_zero_despite_zero_numeric_difference(tmp_path):
    import json

    before, after, old, new = _matched_reports()
    for label, report, states in (("before", before, old), ("after", after, new)):
        report["commit"] = label
        row = report["rows"][0]
        row.update(
            system="lorenz",
            family="classic",
            horizon_steps=2,
            fixed_training_scores={
                threshold: {"lyapunov_times": 0.0, "right_censored": False}
                for threshold in ("0.3", "0.5")
            },
        )
        (tmp_path / f"{label}-results.json").write_text(json.dumps(report))
        (tmp_path / f"{label}-states.json").write_text(json.dumps(states))
        values = np.zeros((2, 3))
        if label == "after":
            values[0, 0] = -0.0
        np.savez(tmp_path / f"{label}-predictions.npz", **{row["id"]: values})
    compared = RUNNER._compare(tmp_path)
    assert compared["maximum_forecast_difference"] == 0
    assert compared["all_forecasts_bit_identical"] is False
    assert compared["all_fixed_training_scores_identical"] is True


def test_tuned_protocol_keeps_all_test_origins_after_validation_and_spans_ten_lyapunov_times():
    spec = importlib.util.spec_from_file_location(
        "resdag_tuned_accuracy", PATH.with_name("tuned_accuracy.py")
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert not set(runner.TRAIN_SEEDS) & set(runner.TEST_SEEDS)
    assert len(runner.TRAIN_SEEDS) == 3
    assert len(runner.TEST_SEEDS) == 5
    for system in ("lorenz", "rossler"):
        protocol = runner._protocol(system)
        train, validation, test = (protocol[key] for key in ("training", "validation", "test"))
        horizon = protocol["horizon_steps"]
        assert train[0] == 0 and train[1] == validation[0] and validation[1] == test[0]
        assert protocol["horizon_lyapunov_times"] >= 10
        assert len(protocol["test_origins"]) == 3
        assert all(
            test[0] <= origin < origin + horizon <= test[1] for origin in protocol["test_origins"]
        )
        assert protocol["test_origins"][1] - protocol["test_origins"][0] == horizon


def _tuned_runner():
    spec = importlib.util.spec_from_file_location(
        "resdag_tuned_budget", PATH.with_name("tuned_accuracy.py")
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_full_budget_uses_no_pruning_and_counts_failed_candidates_honestly(tmp_path):
    optuna = pytest.importorskip("optuna")
    runner = _tuned_runner()
    study = runner._new_study(optuna, tmp_path, "balanced-budget")
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    runs = [{"complete": True, "completed_steps": 553} for _ in range(9)]
    study.add_trial(optuna.trial.create_trial(value=450, user_attrs={"runs": runs}))
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL,
            user_attrs={"error": "invalid candidate forecast"},
        )
    )
    evidence = runner._budget_evidence(study, 2, 9, 553)
    assert evidence["state_counts"] == {
        "COMPLETE": 1,
        "FAIL": 1,
        "PRUNED": 0,
        "RUNNING": 0,
        "WAITING": 0,
    }
    assert evidence["all_attempts_complete"] is False
    assert evidence["all_successful_trials_received_full_checks"] is True
    assert evidence["pruner"] == "NopPruner"
    # Matching the advertised seed count is insufficient: every origin must
    # also complete the entire forecast, including candidates already inaccurate.
    incomplete = [{"complete": True, "completed_steps": 128} for _ in range(9)]
    study.add_trial(optuna.trial.create_trial(value=10, user_attrs={"runs": incomplete}))
    with pytest.raises(ValueError, match="every full seed/origin forecast"):
        runner._budget_evidence(study, 3, 9, 553)


def test_full_budget_never_describes_pruned_or_missing_attempts_as_complete(tmp_path):
    optuna = pytest.importorskip("optuna")
    runner = _tuned_runner()
    study = runner._new_study(optuna, tmp_path, "pruned-budget")
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED))
    with pytest.raises(ValueError, match="attempt budget"):
        runner._budget_evidence(study, 1, 9, 553)
    with pytest.raises(ValueError, match="attempt budget"):
        runner._budget_evidence(study, 24, 9, 553)


def test_queued_known_recipe_is_inside_the_declared_optuna_search():
    optuna = pytest.importorskip("optuna")
    runner = _tuned_runner()
    known = runner._known()
    assert runner._search(optuna.trial.FixedTrial(known)) == known
    assert known["width"] == 400


def _recorded_report(name):
    import json

    return json.loads((PATH.parent / "results" / name).read_text())


def test_frozen_accuracy_report_summaries_and_identity_claims_recompute():
    import statistics

    report = _recorded_report("phase-five-accuracy-2026-10-23.json")
    before, after = report["before"]["rows"], report["after"]["rows"]
    assert len(before) == len(after) == report["forecast_count"] == 36
    assert [row["id"] for row in before] == [row["id"] for row in after]
    if report["all_forecasts_bit_identical"]:
        assert all(
            old["forecast_sha256"] == new["forecast_sha256"] for old, new in zip(before, after)
        )
    for summary in report["summaries"]:
        rows = [
            row
            for row in after
            if (row["system"], row["family"]) == (summary["system"], summary["family"])
        ]
        assert summary["forecast_count"] == len(rows)
        for threshold, recorded in summary["scores"].items():
            values = [row["fixed_training_scores"][threshold]["lyapunov_times"] for row in rows]
            assert recorded["median_lyapunov_times"] == statistics.median(values)
            assert recorded["min_lyapunov_times"] == min(values)
            assert recorded["max_lyapunov_times"] == max(values)
            assert recorded["right_censored_count"] == sum(
                row["fixed_training_scores"][threshold]["right_censored"] for row in rows
            )


def test_tuned_report_preserves_full_budgets_winning_trials_and_untouched_seed_bank():
    from collections import Counter

    report = _recorded_report("phase-five-tuned-accuracy-2026-10-23.json")
    winners = report["frozen_winners"]["winners"]
    assert set(winners) == {
        f"{system}-{family}" for system in ("lorenz", "rossler") for family in ("classic", "ott")
    }
    assert report["n_trials_per_study"] == 24 and report["quick"] is False
    assert len(report["train_seeds"]) == 3 and len(report["test_seeds"]) == 5
    assert not set(report["train_seeds"]) & set(report["test_seeds"])
    assert not set(report["selection_integrity"]["smoke_seeds"]) & set(report["test_seeds"])
    for name, winner in winners.items():
        protocol = report["data"][name.split("-")[0]]["protocol"]
        trials = winner["trials"]
        counts = Counter(trial["state"] for trial in trials)
        budget = report["validated_study_budgets"][name]
        assert len(trials) == winner["attempts"] == budget["attempts"] == 24
        assert counts["PRUNED"] == counts["RUNNING"] == counts["WAITING"] == 0
        assert budget["sampler"] == "TPESampler" and budget["pruner"] == "NopPruner"
        assert budget["state_counts"] == {state: counts[state] for state in budget["state_counts"]}
        assert budget["all_attempts_complete"] == (counts["COMPLETE"] == 24)
        for trial in trials:
            if trial["state"] == "COMPLETE":
                runs = trial["user_attrs"]["runs"]
                assert len(runs) == 9
                assert {run["seed"] for run in runs} == set(report["train_seeds"])
                assert {run["origin"] for run in runs} == {0, 1, 2}
                assert all(
                    run["complete"] and run["completed_steps"] == protocol["horizon_steps"]
                    for run in runs
                )
        best = next(trial for trial in trials if trial["number"] == winner["best_trial"])
        assert best["state"] == "COMPLETE"
        assert best["params"] == winner["params"]
        assert best["value"] == winner["validation_score_steps"]


def test_tuned_report_test_summaries_censoring_and_replay_claims_recompute():
    import statistics

    report = _recorded_report("phase-five-tuned-accuracy-2026-10-23.json")
    rows = report["after"]["rows"]
    old_rows = report["before"]["rows"]
    assert len(rows) == len(old_rows) == 150
    assert [row["id"] for row in rows] == [row["id"] for row in old_rows]
    if report["maximum_revision_forecast_difference"] == 0:
        assert all(
            old["forecast_sha256"] == new["forecast_sha256"] for old, new in zip(old_rows, rows)
        )
    for row in rows:
        protocol = report["data"][row["system"]]["protocol"]
        assert row["seed"] in report["test_seeds"]
        assert row["origin_index"] in protocol["test_origins"]
        assert row["origin_index"] >= protocol["validation"][1]
        assert row["origin_index"] + protocol["horizon_steps"] <= protocol["test"][1]
        assert protocol["horizon_lyapunov_times"] >= 10
        for score in row["scores"].values():
            assert score["lyapunov_times"] == pytest.approx(
                score["steps"] * protocol["dt"] * protocol["lambda_max"]
            )
            assert score["right_censored"] == (score["steps"] == protocol["horizon_steps"])
    for summary in report["summaries"]:
        chosen = [
            row
            for row in rows
            if (row["system"], row["family"], row["variant"])
            == (summary["system"], summary["family"], summary["variant"])
        ]
        assert len(chosen) == summary["count"] == 15
        for threshold, recorded in summary["scores"].items():
            values = [row["scores"][threshold]["lyapunov_times"] for row in chosen]
            assert recorded["median"] == statistics.median(values)
            assert recorded["min"] == min(values) and recorded["max"] == max(values)
            assert recorded["right_censored"] == sum(
                row["scores"][threshold]["right_censored"] for row in chosen
            )
            for seed, median in recorded["per_seed_medians"].items():
                assert median == statistics.median(
                    row["scores"][threshold]["lyapunov_times"]
                    for row in chosen
                    if row["seed"] == int(seed)
                )
