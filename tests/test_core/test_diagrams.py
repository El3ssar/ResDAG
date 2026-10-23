"""Diagrams describe topology without executing the model or opening viewers."""

import builtins
import copy
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

import resdag as rd
from resdag.core._diagram import diagram_dot, plot_model


def _branched():
    x = rd.reservoir_input(1, dtype=torch.float64)
    features = rd.ESNLayer(4, feedback_size=1, seed=3, dtype=torch.float64)(x)
    left = rd.RidgeReadoutLayer(4, 1, name="left").double()(features)
    joined = rd.Concatenate()(features, left)
    right = rd.RidgeReadoutLayer(5, 1, name="right").double()(joined)
    return rd.Model(
        inputs={'state "quoted"': x},
        outputs={"first": left, "second": right},
        feedback=[rd.Feedback("second", 'state "quoted"', delay=3)],
        prediction_lead={"second": 2},
        washout=2,
    )


def test_named_branched_diagram_is_read_only_and_contains_feedback(tmp_path):
    model = _branched()
    model.graph(torch.ones(1, 5, 1, dtype=torch.float64))
    model.train()
    next(iter(model.graph.modules())).eval()
    modes = [module.training for module in model.modules()]
    state = model.get_state()
    checkpoint = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    handles = [
        m.register_forward_hook(lambda *args: pytest.fail("diagram executed a module"))
        for m in model.modules()
    ]
    try:
        destination = model.save(tmp_path / "model.dot", show_shapes=True, show_trainable=True)
    finally:
        for handle in handles:
            handle.remove()
    assert destination == tmp_path / "model.dot"
    source = destination.read_text()
    assert 'state \\"quoted\\"' in source
    assert "Concatenate" in source and "RidgeReadoutLayer" in source
    assert "delay=3; prediction lead=2" in source and 'style="dashed"' in source
    assert "(batch, time, 5)" in source
    assert '"op_0" -> "op_1"' in source
    assert '"op_0" -> "op_2"' in source
    assert modes == [module.training for module in model.modules()]
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    for name, tensor in state.items():
        torch.testing.assert_close(model.get_state()[name], tensor, rtol=0, atol=0)
    for name, value in checkpoint.items():
        actual = model.state_dict()[name]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        else:
            assert actual == value


@pytest.mark.parametrize(
    "format,signature", [("svg", b"<?xml"), ("png", b"\x89PNG"), ("pdf", b"%PDF")]
)
def test_actual_graphviz_rendering_honors_suffix_and_writes_one_file(
    tmp_path, format, signature, graphviz_rendering
):
    model = _branched()
    destination = tmp_path / f"diagram.{format}"
    with patch("graphviz.Source.view", side_effect=AssertionError("viewer must not open")):
        assert model.save(destination) == destination
    assert destination.read_bytes().startswith(signature)
    assert list(tmp_path.iterdir()) == [destination]


def test_dot_export_does_not_require_graphviz_python_or_executable(tmp_path, monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "graphviz":
            raise ImportError("disabled optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    path = _branched().save(tmp_path / "raw", format="dot")
    assert path == tmp_path / "raw.dot" and path.read_text().startswith("digraph")
    with pytest.raises(ImportError, match=r"resdag\[viz\]"):
        _branched().save(tmp_path / "render.svg")


def test_missing_graphviz_binary_has_actionable_error(tmp_path, monkeypatch):
    class MissingExecutableError(Exception):
        pass

    source = Mock()
    source.pipe.side_effect = MissingExecutableError("dot")
    monkeypatch.setitem(
        sys.modules,
        "graphviz",
        SimpleNamespace(
            Source=Mock(return_value=source), ExecutableNotFound=MissingExecutableError
        ),
    )
    with pytest.raises(RuntimeError, match="Graphviz system executable"):
        _branched().save(tmp_path / "render.png")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "name,format", [("x.bin", None), ("x.pt", "svg"), ("x.png", "pdf"), ("x", "jpeg")]
)
def test_bad_save_formats_fail_before_writing(tmp_path, name, format):
    with pytest.raises(ValueError, match="format/suffix"):
        _branched().save(tmp_path / name, format=format)
    assert list(tmp_path.iterdir()) == []


def test_raw_plot_infers_png_suffix_and_does_not_open_a_viewer(tmp_path, graphviz_rendering):
    model = _branched().graph
    with patch("graphviz.Source.view", side_effect=AssertionError("viewer must not open")):
        path = model.plot_model(save_path=tmp_path / "raw.png")
        source = model.plot_model()
    assert path == tmp_path / "raw.png"
    assert path.read_bytes().startswith(b"\x89PNG")
    assert source.source.startswith("digraph")


def test_preset_parallel_and_coupled_structures(tmp_path):
    preset = rd.ESN(input_size=1, reservoir_size=4, washout=2, seed=3)
    assert "ESNLayer" in preset.plot_model(format="dot")
    assert preset.save(tmp_path / "preset.dot") == tmp_path / "preset.dot"
    parallel = rd.Model(graph=rd.parallel_esn(6, 2, 1, reservoir_size=4), washout=2)
    source = diagram_dot(parallel)
    assert "Periodic patches" in source and "core=2, halo=1" in source
    assert "Stitch patch cores" in source and "shared reservoir" in source
    coupled = rd.Model(
        graph=rd.coupled_ensemble_esn(2, reservoir_size=4, feedback_size=1, output_size=1),
        washout=2,
    )
    source = diagram_dot(coupled)
    assert "Member 0" in source and "Member 1" in source
    assert "Aggregate: mean" in source
    assert source.count("feedback: delay=1") == 1


def test_coupled_graph_has_real_svg_rendering(tmp_path, graphviz_rendering):
    coupled = rd.Model(
        graph=rd.coupled_ensemble_esn(2, reservoir_size=4, feedback_size=1, output_size=1),
        washout=2,
    )
    assert plot_model(coupled, save_path=tmp_path / "coupled.svg").is_file()


def test_diagram_rejects_bad_layout_without_running():
    with pytest.raises(ValueError, match="rankdir"):
        diagram_dot(_branched(), rankdir="LR; injected=true")


@pytest.mark.parametrize("show_shapes", [False, True])
def test_scalar_intermediate_diagram_uses_metadata_without_forwarding(show_shapes):
    class Mean(nn.Module):
        def forward(self, values):
            return values.mean()

    class Scale(nn.Module):
        def forward(self, values, scale):
            return values * scale

    source = rd.reservoir_input(1, dtype=torch.float64)
    prediction = rd.RidgeReadoutLayer(1, 1, alpha=0.05).double()(Scale()(source, Mean()(source)))
    model = rd.Model(
        inputs={"state": source},
        outputs={"prediction": prediction},
        preprocessing="off",
        washout=2,
    )
    values = torch.arange(1.0, 21.0, dtype=torch.float64).reshape(-1, 1)
    model.fit(values, 2 * values + 1)
    handle = model.graph.register_forward_hook(lambda *args: pytest.fail("diagram ran graph"))
    try:
        source = diagram_dot(model, show_shapes=show_shapes)
    finally:
        handle.remove()
    assert "Mean" in source and "Scale" in source
    assert ("scalar" in source) == show_shapes


@pytest.mark.parametrize("show_shapes", [False, True])
def test_partitioned_collection_and_unpack_outputs_have_correct_edges(show_shapes):
    source = rd.reservoir_input(2, dtype=torch.float64)
    first, second = rd.FeaturePartitioner(partitions=2, overlap=0)(source)
    prediction = rd.RidgeReadoutLayer(2, 2, alpha=0.01).double()(rd.Concatenate()(first, second))
    model = rd.Model(
        inputs={"state": source},
        outputs={"prediction": prediction},
        preprocessing="off",
        washout=1,
    )
    model.fit(torch.arange(40.0, dtype=torch.float64).reshape(20, 2))
    source = diagram_dot(model, show_shapes=show_shapes)
    assert "FeaturePartitioner" in source and "UnpackLayer" in source
    assert "output 0; input 0" in source
    assert "output 1; input 1" in source
    assert ("collection/value" in source) == show_shapes


def test_sequential_label_keeps_each_invocation_of_a_shared_stateless_module():
    source = rd.reservoir_input(1, dtype=torch.float64)
    shared = nn.Linear(1, 1).double()
    prediction = rd.RidgeReadoutLayer(1, 1).double()(nn.Sequential(shared, shared)(source))
    model = rd.Model(inputs={"state": source}, outputs={"prediction": prediction}, washout=2)
    description = diagram_dot(model)
    assert "0: Linear" in description and "1: Linear" in description


def test_scaling_nodes_connect_raw_graph_and_original_unit_feedback_ports():
    source = diagram_dot(rd.ESN(input_size=1, reservoir_size=3, washout=2))
    assert "Input transform: input" in source
    assert "Inverse output transform: output" in source
    assert "FittedNormalization (standard)" in source
    assert '"input_0" -> "input_transform_0"' in source
    assert '"output_transform_0" -> "output_0"' in source
    assert '"output_0" -> "input_0" [label="feedback:' in source
