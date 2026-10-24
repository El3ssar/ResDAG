"""Execute every shipped tutorial's real main body in a fresh interpreter.

Quick budgets exercise training/gradients/forecast/persistence, not scientific
quality or latency. Optional examples are skipped only when their dependencies
are absent; the CUDA benchmark itself reports a clean no-CUDA skip.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
COMPLETION = {
    "00_easiest_api.py": "Underlying model stays accessible",
    "01_quickstart.py": "5. Evaluate against ground truth",
    "02_premade_models.py": "6. Comparison (same data",
    "03_functional_api.py": "outputs: coords",
    "04_topologies_and_initializers.py": "separate input_initializer:",
    "05_training_paths.py": "C full BPTT",
    "06_forecasting.py": "aux             forecast:",
    "07_ensembles.py": "Ensemble spread is an empirical disagreement diagnostic",
    "08_save_load.py": "reloaded full model reproduces original outputs: True",
    "09_visualization.py": "Rendered files:",
    "10_hpo.py": "Refit best model -> forecast MSE@50",
    "11_gpu_benchmark.py": None,
    "12_feature_extractor.py": "shares recurrent weights: True",
    "13_windowed_forecast.py": "gap RMSE over",
    "14_streaming_dataloader.py": "fitted (is_fitted=True)",
    "15_end_to_end.py": "Checkpoint, PyTorch composition and chunk continuity verified.",
    "16_custom_reservoirs.py": "All families trained, forecast, restored, and passed upstream gradients.",
    "17_forecast_experiment.py": "Untuned single-seed examples.",
}


def _require_optional(name: str) -> None:
    if name == "09_visualization.py":
        pytest.importorskip("graphviz")
        if shutil.which("dot") is None:
            pytest.skip("Graphviz dot is needed to verify actual SVG/PNG rendering")
    elif name == "10_hpo.py":
        pytest.importorskip("optuna")
    elif name == "17_forecast_experiment.py":
        pytest.importorskip("tsdynamics")


def _run(
    name: str, arguments: list[str], folder: Path, *, matplotlib_config: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(matplotlib_config or folder / "matplotlib"),
        TMPDIR=str(folder),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    # Use the current interpreter/package environment without adding src to
    # PYTHONPATH, so this also works in isolated installed-wheel acceptance.
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / name), *arguments],
        cwd=folder,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


def test_every_numbered_example_has_an_execution_budget() -> None:
    discovered = {p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py")}
    assert discovered == COMPLETION.keys()


@pytest.mark.parametrize("name", COMPLETION, ids=lambda name: Path(name).stem)
def test_example_main_body(name: str, tmp_path: Path) -> None:
    _require_optional(name)
    arguments = ["--quick"]
    if name == "17_forecast_experiment.py":
        arguments = [
            "--reservoir-size",
            "16",
            "--training",
            "160",
            "--washout",
            "12",
            "--history",
            "32",
            "--horizon",
            "16",
            "--diagram-format",
            "dot",
            "--output",
            str(tmp_path / "experiment"),
        ]
    result = _run(name, arguments, tmp_path)
    marker = COMPLETION[name]
    if marker is not None:
        assert marker in result.stdout
    else:
        assert (
            "CUDA is not available on this machine — benchmark skipped." in result.stdout
            or "CPU and CUDA fit/forecast execution verified; no timings collected."
            in result.stdout
        )
        assert "GPU speedup" not in result.stdout
    if name == "08_save_load.py":
        assert "restored model reproduces original outputs: True" in result.stdout
        assert "Full forecast continuity verified with identical call shapes." in result.stdout
        assert {path.name for path in tmp_path.rglob("*.pt")} == {
            "model.pt",
            "checkpoint.pt",
            "warmed_up.pt",
            "model_full.pt",
        }
    elif name == "09_visualization.py":
        for suffix in ("ott_esn.svg", "ott_esn_detailed.png", "custom_dag.svg"):
            matches = list(tmp_path.rglob(suffix))
            assert len(matches) == 1 and matches[0].stat().st_size > 100
        assert list(tmp_path.rglob("ott_esn_detailed.png"))[0].read_bytes().startswith(b"\x89PNG")
    elif name == "17_forecast_experiment.py":
        manifest = tmp_path / "experiment" / "experiment.json"
        report = json.loads(manifest.read_text())
        assert [row["model"] for row in report["scores"]] == ["vanilla", "ott", "custom"]
        verified = json.loads(_run(name, ["--verify", str(manifest)], tmp_path).stdout)
        assert verified["forecasts"] == "exact replay"
        assert verified["verified_models"] == ["vanilla", "ott", "custom"]


@pytest.fixture(scope="session")
def matplotlib_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Share only the font cache, never model state or numerical inputs, between
    # fresh plot interpreters. Font discovery need not repeat for every plot.
    return tmp_path_factory.mktemp("example-matplotlib")


@pytest.mark.parametrize(
    "name",
    ["00_easiest_api.py", "01_quickstart.py", "06_forecasting.py", "13_windowed_forecast.py"],
)
def test_optional_plot_body(name: str, tmp_path: Path, matplotlib_config: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("optional matplotlib dependency is absent")
    result = _run(name, ["--quick", "--plot"], tmp_path, matplotlib_config=matplotlib_config)
    assert COMPLETION[name] in result.stdout
    assert "skipping plot" not in result.stdout
