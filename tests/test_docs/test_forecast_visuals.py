"""Recorded scientific visuals must retain their data, clocks and provenance."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLING = ROOT / "docs/_tooling/figures"


@pytest.fixture
def single_threaded():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _module(name):
    spec = importlib.util.spec_from_file_location(f"_doc_visual_{name}", TOOLING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _showcase_fixture():
    time = [0.1, 0.2, 0.3]
    prediction = np.array([[0.0, 0.0, 0.0], [0.6, 0.6, 0.6], [0.0, 0.0, 0.0]])
    case = {
        "variables": ["x", "y", "z"],
        "dt": 0.1,
        "lambda_max": 1.0,
        "time": time,
        "truth": np.zeros_like(prediction).tolist(),
        "prediction": prediction.tolist(),
        "training_scale": [2.0, 2.0, 2.0],
        "prediction_sha256": hashlib.sha256(prediction.astype("<f8").tobytes()).hexdigest(),
    }
    return {
        "schema_version": 1,
        "selection": {
            "trajectory": 0,
            "reservoir_seed": 7849201,
            "origin_index": 0,
            "predeclared": True,
        },
        "cases": {"lorenz": case, "rossler": dict(case)},
    }


def test_visual_loader_validates_identity_hash_clock_and_first_crossing(tmp_path):
    loader = _module("forecast_data")
    path = tmp_path / "showcase.json"
    record = _showcase_fixture()
    path.write_text(json.dumps(record))
    loaded = loader.load_showcase(path)
    assert loader.valid_steps(loaded["cases"]["lorenz"], 0.3) == 1
    assert loader.valid_steps(loaded["cases"]["lorenz"], 0.5) == 3

    record["cases"]["lorenz"]["prediction"][0][0] = 0.1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        loader.load_showcase(path)
    record = _showcase_fixture()
    record["cases"]["lorenz"]["time"] = [0.1, 0.3, 0.4]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="clock"):
        loader.load_showcase(path)
    record = _showcase_fixture()
    record["selection"]["trajectory"] = 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="selection"):
        loader.load_showcase(path)


def test_committed_display_cases_match_the_independent_study():
    loader = _module("forecast_data")
    showcase = loader.load_showcase()
    source = ROOT / "docs/assets/data" / showcase["source_study"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == showcase["source_study_sha256"]
    study = json.loads(source.read_text())
    protocol = study["preregistration"]["protocol"]
    assert protocol["kind"] == "independent-trajectory recipe search"
    for name, case in showcase["cases"].items():
        config = protocol["systems"][name]
        winner = study["results"]["final"][name]["winner"]
        row = next(
            row
            for row in winner["forecasts"]
            if (row["trajectory"], row["seed"], row["origin"]) == (0, 7849201, 0)
        )
        assert case["prediction_sha256"] == row["prediction_sha256"]
        assert len(case["time"]) == config["horizon"]
        assert case["dt"] == config["dt"]
        assert case["lambda_max"] == config["lambda_max"]
        np.testing.assert_array_equal(case["training_scale"], winner["training_scales"][0]["scale"])
        for threshold in (0.3, 0.5):
            assert loader.valid_steps(case, threshold) == row["scores"][str(threshold)]["steps"]


def test_recorded_optuna_diagnostics_preserve_states_params_and_scores(monkeypatch):
    pytest.importorskip("optuna")
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(TOOLING))
    hpo = _module("hpo")
    loader = _module("forecast_data")
    showcase = loader.load_showcase()
    source = ROOT / "docs/assets/data" / showcase["source_study"]
    original = json.loads(source.read_text())["results"]["validation"]["lorenz"]
    study = hpo.recorded_study(showcase=showcase)
    assert len(study.trials) == len(original["trials"])
    for actual, expected in zip(study.trials, original["trials"], strict=True):
        assert actual.params == expected["params"]
        assert actual.state.name == original["states"][str(expected["trial"])]
        if actual.state.name == "COMPLETE":
            assert (
                actual.value
                == expected["evaluation"]["summary"]["0.3"]["median_of_trajectory_medians"]
            )


def test_every_registered_initializer_has_a_current_inspectable_gallery_image():
    Image = pytest.importorskip("PIL.Image")

    from resdag.init.input_feedback.registry import _INPUT_FEEDBACK_REGISTRY
    from resdag.init.topology.registry import _TOPOLOGY_REGISTRY

    for family, registry in (
        ("topologies", _TOPOLOGY_REGISTRY),
        ("initializers", _INPUT_FEEDBACK_REGISTRY),
    ):
        for name in registry:
            path = ROOT / "docs/assets/figures" / family / f"{name}.png"
            assert path.is_file(), f"Regenerate the {family} gallery for {name}"
            with Image.open(path) as image:
                metadata = json.loads(image.info["Description"])
                assert metadata["component"] == name
                rows, columns = metadata["shape"]
                assert rows > 0 and columns > 0
                assert rows == columns if family == "topologies" else (rows, columns) == (80, 8)
                assert re.fullmatch(r"[0-9a-f]{64}", metadata["matrix_sha256"])
                assert image.width >= 300 and image.height >= 300


def test_standard_animation_retains_both_recorded_paths(monkeypatch, tmp_path):
    pytest.importorskip("tsdynamics")
    pytest.importorskip("matplotlib")
    from PIL import Image

    monkeypatch.syspath_prepend(str(TOOLING))
    hero = _module("hero_forecasts")
    case = _showcase_fixture()["cases"]["lorenz"]
    case["name"] = "Synthetic clock/plot check"
    plot = hero.forecast_plot(case)
    assert len(plot.layers) == 2
    plot.animate(duration=0.5, fps=6).trail(None)
    destination = tmp_path / "forecast.gif"
    plot.save(destination, size=(240, 160), dpi=60)
    with Image.open(destination) as image:
        assert image.n_frames >= 2
        assert image.size == (240, 160)
    import matplotlib.pyplot as plt

    plt.close("all")


def test_first_forecast_guide_executes(tmp_path, monkeypatch, single_threaded):
    monkeypatch.chdir(tmp_path)
    guide = ROOT / "docs/start/first-forecast.md"
    namespace = {"__name__": "__documented_example__"}
    for block in re.findall(r"^```python\n(.*?)^```", guide.read_text(), re.M | re.S):
        exec(compile(block, str(guide), "exec"), namespace)
    assert namespace["prediction"].shape == namespace["truth"].shape == (1000, 3)
    assert (tmp_path / "lorenz.pt").is_file()


def test_visualization_guide_writes_real_diagram_plots_and_animation(
    tmp_path,
    monkeypatch,
    graphviz_rendering,
    single_threaded,
):
    pytest.importorskip("tsdynamics")
    pytest.importorskip("matplotlib")
    from PIL import Image
    from tsdynamics.viz.spec import Plot

    original_save, original_animate = Plot.save, Plot.animate

    def bounded_save(self, path, **options):
        options.update(size=(240, 160), dpi=60)
        return original_save(self, path, **options)

    def bounded_animate(self, **options):
        return original_animate(self, duration=0.5, fps=6)

    monkeypatch.setattr(Plot, "save", bounded_save)
    monkeypatch.setattr(Plot, "animate", bounded_animate)
    monkeypatch.chdir(tmp_path)
    guide = ROOT / "docs/workflows/visualize.md"
    namespace = {"__name__": "__documented_example__"}
    for block in re.findall(r"^```python\n(.*?)^```", guide.read_text(), re.M | re.S):
        exec(compile(block, str(guide), "exec"), namespace)
    assert namespace["prediction"].y.shape == namespace["truth"].y.shape == (500, 3)
    np.testing.assert_allclose(namespace["prediction"].t, namespace["truth"].t, rtol=0, atol=1e-10)
    assert (tmp_path / "forecast-model.svg").read_text().lstrip().startswith("<?xml")
    assert (tmp_path / "forecast-model.dot").read_text().lstrip().startswith("digraph")
    diagram = ROOT / "docs/assets/figures/architectures/forecast_dag"
    assert (
        diagram.with_suffix(".dot").read_text().strip()
        == namespace["model"].plot_model(rankdir="TB", show_trainable=True).source.strip()
    )

    def labels(path):
        return [element.text for element in ET.parse(path).iter("{http://www.w3.org/2000/svg}text")]

    # The committed drawing must show the same ports, layers and feature widths
    # as the guide's actual SVG, rather than a similar older architecture.
    assert labels(diagram.with_suffix(".svg")) == labels(tmp_path / "forecast-model.svg")
    for name in ("forecast-x.png", "forecast-phase.png"):
        with Image.open(tmp_path / name) as image:
            assert image.size == (240, 160)
    with Image.open(tmp_path / "forecast.gif") as image:
        assert image.n_frames >= 2
    import matplotlib.pyplot as plt

    plt.close("all")
