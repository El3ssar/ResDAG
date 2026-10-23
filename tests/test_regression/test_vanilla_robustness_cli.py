"""Independent trajectory isolation, metric oracles and real bounded CLI replay."""

import base64
import copy
import importlib.util
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "benchmarks/vanilla_robustness.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("vanilla_robustness_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_declares_independent_units_and_distinct_seed_banks(runner):
    plan = runner._plan(False, 24, 600, np)
    assert plan["validation_seeds"] != plan["final_seeds"]
    conditions = []
    for config in plan["systems"].values():
        assert len(config["validation_ics"]) == 3
        assert len(config["final_ics"]) == 5
        assert config["origin_count"] == 3
        assert config["horizon"] * config["dt"] * config["lambda_max"] >= 10
        conditions.extend(config["validation_ics"] + config["final_ics"])
    assert len({tuple(ic) for ic in conditions}) == 16
    assert np.isfinite(conditions).all()
    assert plan["pruning"] is False


def test_incumbents_are_only_predeclared_primary_recipes(runner):
    path = ROOT / "docs/assets/data/phase-five-independent-trajectories-2026-10-23.json"
    incumbents, origin = runner._incumbents(path, False)
    assert set(incumbents) == {"lorenz", "rossler"}
    assert incumbents["lorenz"]["reservoir_size"] == 1000
    assert incumbents["rossler"]["reservoir_size"] == 400
    assert all(params["input_to_readout"] is True for params in incumbents.values())
    assert origin["sha256"] == runner._hash(path.read_bytes())


def test_score_uses_training_scale_strict_crossings_and_retains_nonfinite_trace(runner):
    truth = torch.zeros(5, 3, dtype=torch.float64)
    scale = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    normalized = torch.tensor([0.1, 0.3, 0.4, float("nan"), 0.0], dtype=torch.float64)
    prediction = normalized[:, None] * scale
    config = {"dt": 0.02, "lambda_max": 0.9056}
    scores, trace = runner._score(prediction, truth, scale, config, torch)
    assert scores["0.3"]["steps"] == 1
    assert scores["0.5"]["steps"] == 3
    assert not scores["0.5"]["right_censored"]
    assert scores["0.3"]["lyapunov_times"] == pytest.approx(0.02 * 0.9056)
    decoded = zlib.decompress(base64.b64decode(trace["data"]))
    assert runner._hash(decoded) == trace["sha256"]
    np.testing.assert_allclose(np.frombuffer(decoded, dtype="<f8"), normalized.numpy())
    scores, _ = runner._score(truth, truth, scale, config, torch)
    assert all(item["right_censored"] and item["steps"] == 5 for item in scores.values())


def test_aggregation_does_not_replace_trajectory_medians_with_pooled_forecasts(runner):
    rows = []
    for trajectory, values in enumerate(((0, 0, 100), (1, 1, 100), (2, 2, 2))):
        for value in values:
            rows.append(
                {
                    "trajectory": trajectory,
                    "scores": {
                        key: {"lyapunov_times": value, "right_censored": False}
                        for key in ("0.3", "0.5")
                    },
                }
            )
    summary = runner._aggregate(rows)
    assert summary["0.3"]["median_of_trajectory_medians"] == 1
    assert summary["0.3"]["trajectory_medians"] == [0, 1, 2]
    assert (
        np.median([value for group in ((0, 0, 100), (1, 1, 100), (2, 2, 2)) for value in group])
        == 2
    )


def test_final_observations_require_frozen_winners_before_integration(runner, tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    with pytest.raises(RuntimeError, match="Freeze both winners"):
        runner._generate("lorenz", {}, [], final, None, None)
    assert not list(final.iterdir())


def test_dataset_tampering_is_rejected_before_loading(runner, tmp_path):
    path = tmp_path / "data.npy"
    np.save(path, np.zeros((10, 3)))
    datasets = [{"path": path, "sha256": runner._hash(path.read_bytes())}]
    path.write_bytes(b"changed")
    progress = {}
    with pytest.raises(RuntimeError, match="Dataset changed"):
        runner._evaluate({}, datasets, {}, [], None, torch, np, progress)
    assert progress["fits"] == progress["forecasts"] == []


def test_controlled_baselines_reject_incomplete_or_changed_reservoirs(runner):
    records = [
        {
            "trial": index,
            "params": {"input_to_readout": flag, "alpha": 1e-6},
            "evaluation": {
                "summary": {},
                "fits": [{"trajectory": 0, "seed": 1, "reservoir_sha256": "same"}],
            },
        }
        for index, flag in enumerate((True, False))
    ]
    with pytest.raises(RuntimeError, match="must complete"):
        runner._paired_baselines(records, {"0": "COMPLETE", "1": "WAITING"})
    assert runner._paired_baselines(records, {"0": "COMPLETE", "1": "COMPLETE"})[
        "same_reservoir_weights"
    ]
    changed = copy.deepcopy(records)
    changed[1]["evaluation"]["fits"][0]["reservoir_sha256"] = "different"
    with pytest.raises(RuntimeError, match="identical reservoir"):
        runner._paired_baselines(changed, {"0": "COMPLETE", "1": "COMPLETE"})


def test_existing_output_is_rejected_without_modification(tmp_path):
    output = tmp_path / "keep"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep")
    result = subprocess.run(
        [sys.executable, str(PATH), "--output", str(output), "--quick"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "output already exists" in result.stderr
    assert sentinel.read_text() == "keep"


def test_real_quick_cli_freezes_both_winners_before_final_runs(tmp_path):
    pytest.importorskip("tsdynamics")
    output = tmp_path / "real-smoke"
    result = subprocess.run(
        [sys.executable, str(PATH), "--output", str(output), "--quick", "--timeout", "120"],
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads((output / "preregistered.json").read_text())
    winners = json.loads((output / "frozen-winners.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert plan["kind"] == "execution smoke"
    assert winners["test_used_for_selection"] is False
    assert set(winners["winners"]) == {"lorenz", "rossler"}
    assert report["plan_sha256"] == winners["plan_sha256"]
    assert report["source_imports_verified"]
    for system in ("lorenz", "rossler"):
        validation = report["validation"][system]
        assert len(validation["trials"]) == 2
        assert {record["params"]["input_to_readout"] for record in validation["trials"]} == {
            False,
            True,
        }
        assert set(validation["states"].values()) == {"COMPLETE"}
        assert validation["controlled_feature_comparison"]["same_reservoir_weights"]
        for variant in ("incumbent", "winner"):
            evaluation = report["final"][system][variant]
            assert len(evaluation["fits"]) == len(evaluation["forecasts"]) == 1
            assert evaluation["fits"][0]["seed"] in plan["final_seeds"]
            assert evaluation["forecasts"][0]["finite"]
            config = plan["systems"][system]
            raw = np.load(output / report["final"][system]["datasets"][0]["path"])
            expected_scale = raw[config["washout"] + 1 : config["train_observations"]].std(0)
            np.testing.assert_allclose(evaluation["training_scales"][0]["scale"], expected_scale)
