"""Run the forecast experiment guide and independently check its scoring boundary."""

import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "17_forecast_experiment.py"
GUIDE = ROOT / "docs" / "start" / "forecast-experiment.md"


@pytest.fixture
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_forecast_experiment_guide_executes(
    tmp_path, monkeypatch, graphviz_rendering, single_threaded
):
    pytest.importorskip("tsdynamics")
    pytest.importorskip("optuna")
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    source = GUIDE.read_text()
    for match in re.finditer(r"```python\n(.*?)\n```", source, re.S):
        padding = "\n" * source[: match.start(1)].count("\n")
        exec(compile(padding + match.group(1), str(GUIDE), "exec"), namespace)
    assert set(namespace["results"]) == {"vanilla", "ott", "custom"}
    for result in namespace["results"].values():
        assert 0 <= result["valid_steps"] <= 500
        assert result["valid_lyapunov_times"] == pytest.approx(
            result["valid_steps"] * namespace["dt"] * 0.9056
        )
    assert (tmp_path / "custom.svg").read_text().lstrip().startswith("<?xml")
    assert (tmp_path / "custom.pt").is_file()
    assert namespace["actual"].variables == namespace["truth"].variables
    assert namespace["tuned"].tuning_report.attempts == 2
    assert namespace["tuned"].tuning_report.validation["refit_includes_validation"] is True
    assert len(namespace["tuned_prediction"]) == len(namespace["truth"])


@pytest.mark.parametrize("system", ["lorenz", "rossler"])
def test_runnable_experiment_both_systems(system, tmp_path, monkeypatch, single_threaded):
    pytest.importorskip("tsdynamics")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EXAMPLE),
            "--system",
            system,
            "--reservoir-size",
            "24",
            "--training",
            "500",
            "--washout",
            "30",
            "--history",
            "100",
            "--horizon",
            "50",
            "--diagram-format",
            "dot",
            "--output",
            str(tmp_path),
        ],
    )
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    report = json.loads((tmp_path / "scores.json").read_text())
    manifest = json.loads((tmp_path / "experiment.json").read_text())
    recipe = manifest["recipe"]
    assert manifest["schema_version"] == 1
    assert manifest["scores"] == report
    assert recipe["arguments"]["system"] == system
    assert recipe["arguments"]["horizon"] == 50
    assert recipe["integration"]["solver"] == "dop853"
    assert recipe["integration"]["rtol"] == 1e-10
    assert recipe["system"]["initial_condition"] == [1.0, 1.0, 1.0]
    assert recipe["system"]["parameters"] == (
        {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}
        if system == "lorenz"
        else {"a": 0.2, "b": 0.2, "c": 5.7}
    )
    assert recipe["data"]["splits"] == {
        "training": [0, 500],
        "history": [400, 500],
        "truth": [500, 550],
    }
    assert set(recipe["dependencies"]) >= {"resdag", "torch", "numpy", "tsdynamics"}
    namespace = runpy.run_path(str(EXAMPLE))
    assert namespace["_identity"](recipe) == manifest["experiment_id"]
    for name, digest in manifest["artifacts"].items():
        assert namespace["_file_hash"](tmp_path / name) == digest
    assert [row["model"] for row in report] == ["vanilla", "ott", "custom"]
    for row in report:
        assert row["fit_seconds"] > 0
        assert row["forecast_seconds"] > 0
        assert 0 <= row["valid_steps"] <= 50
        assert (tmp_path / f"{row['model']}.pt").is_file()
        assert (tmp_path / f"{row['model']}.dot").read_text().lstrip().startswith("digraph")
    # Reconstruct only clocked observations, without integration or fitting.
    import tsdynamics as ts

    monkeypatch.setattr(ts.systems.Lorenz, "run", lambda *a, **kw: pytest.fail("replay integrated"))
    monkeypatch.setattr(
        ts.systems.Rossler, "run", lambda *a, **kw: pytest.fail("replay integrated")
    )
    monkeypatch.setattr(namespace["rd"].Model, "fit", lambda *a, **kw: pytest.fail("replay fitted"))
    verified = namespace["verify_experiment"](tmp_path / "experiment.json")
    assert verified["verified_models"] == ["vanilla", "ott", "custom"]
    assert verified["forecasts"] == "exact replay"
    assert verified["timings"] == "recorded only; not verified"
    if system == "lorenz":
        # A fresh interpreter proves that in-memory models/data are not required.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), env.get("PYTHONPATH", "")))
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    (
                        "import runpy, sys, resdag as rd, tsdynamics as ts",
                        "def forbidden(*args, **kwargs): raise AssertionError('integration or fit during replay')",
                        "rd.Model.fit = forbidden",
                        "ts.systems.Lorenz.run = forbidden",
                        "ts.systems.Rossler.run = forbidden",
                        "sys.argv = [sys.argv[1], '--verify', sys.argv[2]]",
                        "runpy.run_path(sys.argv[0], run_name='__main__')",
                    )
                ),
                str(EXAMPLE),
                str(tmp_path / "experiment.json"),
            ],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        assert json.loads(child.stdout)["experiment_id"] == manifest["experiment_id"]
    # Corruption is rejected before rd.load can construct a model.
    monkeypatch.setattr(
        namespace["rd"], "load", lambda *a, **kw: pytest.fail("loaded tampered model")
    )
    artifact = tmp_path / "vanilla.pt"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact hash mismatch: vanilla.pt"):
        namespace["verify_experiment"](tmp_path / "experiment.json")


@pytest.fixture
def replay_data():
    """A small independent split oracle; no integration or model construction."""
    namespace = runpy.run_path(str(EXAMPLE))
    training = torch.arange(24, dtype=torch.float64).reshape(8, 3)
    times = 7.0 + torch.arange(11, dtype=torch.float64) * 0.25
    data = {
        "training_y": training,
        "training_t": times[:8],
        "history_y": training[-4:].clone(),
        "history_t": times[4:8].clone(),
        "truth_y": torch.arange(9, dtype=torch.float64).reshape(3, 3),
        "truth_t": times[8:],
    }
    predictions = {}
    for name in ("vanilla", "ott", "custom"):
        data[f"{name}_prediction"] = data["truth_y"].clone()
        predictions[f"{name}_prediction"] = namespace["_tensor_record"](data[f"{name}_prediction"])
    recipe = {
        "arguments": {"training": 8, "history": 4, "horizon": 3, "washout": 1, "dt": 0.25},
        "data": {
            "variables": ["x", "y", "z"],
            "actual_dt": 0.25,
            "splits": {"training": [0, 8], "history": [4, 8], "truth": [8, 11]},
            "tensors": {
                key: namespace["_tensor_record"](value)
                for key, value in data.items()
                if not key.endswith("_prediction")
            },
        },
    }
    return namespace, data, recipe, predictions


@pytest.mark.parametrize(
    "mutation", ["history_values", "history_clock", "truth_clock", "split", "dt"]
)
def test_replay_rejects_coherently_rehashed_misalignment(replay_data, mutation):
    namespace, data, recipe, predictions = replay_data
    validate = namespace["_validate_replay_data"]
    validate(data, recipe, predictions)
    if mutation == "split":
        recipe["data"]["splits"]["truth"] = [7, 10]
    elif mutation == "dt":
        recipe["arguments"]["dt"] = 0.5
    else:
        key = {
            "history_values": "history_y",
            "history_clock": "history_t",
            "truth_clock": "truth_t",
        }[mutation]
        data[key] = data[key] + 0.25
        recipe["data"]["tensors"][key] = namespace["_tensor_record"](data[key])
    # All tensor byte hashes agree with their descriptors. The independent
    # semantic oracle must still reject shifted clocks, values and split labels.
    with pytest.raises(ValueError, match="history|clock|split"):
        validate(data, recipe, predictions)


def test_replay_schema_and_path_validation_precede_model_loading(tmp_path, monkeypatch):
    namespace = runpy.run_path(str(EXAMPLE))
    monkeypatch.setattr(namespace["rd"], "load", lambda *a, **kw: pytest.fail("model loaded"))
    path = tmp_path / "experiment.json"
    path.write_text('{"schema_version": 999}')
    with pytest.raises(ValueError, match="schema_version"):
        namespace["verify_experiment"](path)
    path.write_text('{"schema_version": NaN}')
    with pytest.raises(ValueError, match="nonstandard JSON"):
        namespace["verify_experiment"](path)
    with pytest.raises(ValueError, match="escapes"):
        namespace["_artifact_path"](tmp_path, "../outside.pt")


def test_experiment_json_records_nonfinite_scores_as_null():
    namespace = runpy.run_path(str(EXAMPLE))
    training = np.arange(12.0).reshape(4, 3)
    scores = namespace["_score_record"](
        np.full((2, 3), np.inf), np.zeros((2, 3)), training, washout=1, dt=0.1, lambda_max=1.0
    )
    assert scores["valid_steps"] == 0
    assert scores["rmse"] is None
    assert json.loads(namespace["_json_text"](scores))["rmse"] is None


def test_training_scale_and_first_failure_are_used():
    score = runpy.run_path(str(EXAMPLE))["forecast_scores"]
    # Labels paired with washed-out inputs are deliberately extreme. Only labels
    # at indices washout+1 onward belong in the score scale; their std is one.
    training = np.array([[100.0], [100.0], [-1.0], [1.0]])
    truth = np.array([[1000.0], [0.0], [-1000.0], [0.0]])
    prediction = truth + np.array([[0.1], [0.3], [0.01], [0.01]])
    result = score(prediction, truth, training, washout=1, dt=0.25, lambda_max=0.2)
    assert result["valid_steps"] == 1
    assert result["valid_lyapunov_times"] == pytest.approx(0.05)
    assert result["reached_horizon"] is False
    assert result["secondary_valid_lyapunov_times_at_0_5"] == pytest.approx(0.2)


def test_full_prefix_is_reported_as_lower_bound():
    score = runpy.run_path(str(EXAMPLE))["forecast_scores"]
    truth = np.arange(6.0).reshape(3, 2)
    result = score(truth, truth, np.arange(12.0).reshape(6, 2), washout=1, dt=0.1, lambda_max=2)
    assert result["valid_steps"] == 3
    assert result["valid_lyapunov_times"] == pytest.approx(0.6)
    assert result["reached_horizon"] is True


def test_scores_reject_broadcasted_shapes():
    score = runpy.run_path(str(EXAMPLE))["forecast_scores"]
    with pytest.raises(ValueError, match="share a nonempty"):
        score(np.zeros((2, 1)), np.zeros((3, 1)), np.ones((6, 1)), washout=1, dt=0.1, lambda_max=1)
