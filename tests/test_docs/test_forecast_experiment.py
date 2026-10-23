"""Run the forecast experiment guide and independently check its scoring boundary."""

import json
import re
import runpy
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
    assert [row["model"] for row in report] == ["vanilla", "ott", "custom"]
    for row in report:
        assert row["fit_seconds"] > 0
        assert row["forecast_seconds"] > 0
        assert 0 <= row["valid_steps"] <= 50
        assert (tmp_path / f"{row['model']}.pt").is_file()
        assert (tmp_path / f"{row['model']}.dot").read_text().lstrip().startswith("digraph")


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
