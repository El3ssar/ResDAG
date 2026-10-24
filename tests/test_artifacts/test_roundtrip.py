"""Independent artifact oracles across numerical components and graph families."""

import subprocess
import sys

import pytest
import torch
from torch import nn

import resdag as rd
from resdag.artifacts import ArtifactError, load, register_component, save_model
from resdag.layers import cells, readouts

CELL_SPECS = [
    (
        cells.ESNCell,
        dict(
            reservoir_size=5,
            feedback_size=2,
            input_size=1,
            seed=7,
            leak_rate=[0.2, 0.3, 0.4, 0.5, 0.6],
            noise=0.1,
        ),
    ),
    (cells.ES2NCell, dict(reservoir_size=5, feedback_size=2, input_size=1, seed=7, beta=0.6)),
    (
        cells.ResESNCell,
        dict(reservoir_size=5, feedback_size=2, input_size=1, seed=7, alpha=0.3, beta=0.8),
    ),
    (
        cells.EuSNCell,
        dict(
            reservoir_size=5,
            feedback_size=2,
            input_size=1,
            seed=7,
            epsilon=0.1,
            gamma=0.02,
            solver="rk4",
            substeps=2,
        ),
    ),
    (
        cells.GatedReservoirCell,
        dict(
            reservoir_size=5,
            feedback_size=2,
            input_size=1,
            seed=7,
            gate_feedback_initializer="normal",
            gate_input_initializer="uniform",
        ),
    ),
    (
        cells.LIFCell,
        dict(
            reservoir_size=5,
            feedback_size=2,
            input_size=1,
            seed=7,
            synaptic_current=True,
            tau_mem=7,
            tau_syn=3,
            trace_decay=0.6,
        ),
    ),
    (
        cells.CoRNNCell,
        dict(
            reservoir_size=5, feedback_size=2, input_size=1, seed=7, dt=0.2, gamma=0.7, epsilon=0.3
        ),
    ),
    (
        cells.LegendreDelayCell,
        dict(order=3, feedback_size=2, input_size=1, seed=7, theta=7, dt=0.2, memory_channels=2),
    ),
    (cells.NGCell, dict(input_dim=2, k=3, s=2, p=3, cumulative=True)),
    (
        cells.TorchRecurrentCell,
        dict(cell=nn.LSTMCell(3, 5), feedback_size=2, input_size=1, trainable=True),
    ),
]


@pytest.mark.parametrize("cls,options", CELL_SPECS, ids=[c.__name__ for c, _ in CELL_SPECS])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cell_equations_and_tensor_state_exact(tmp_path, cls, options, dtype):
    cell = cls(**options).to(dtype=dtype).eval()
    path = save_model(cell, tmp_path / "model.pt")
    restored = load(path)
    xs = [torch.randn(2, 2, dtype=dtype)] + (
        [torch.randn(2, 1, dtype=dtype)] if options.get("input_size") else []
    )
    state = cell.init_state(2, torch.device("cpu"), dtype)
    expected, next_state = cell(xs, state)
    actual, actual_state = restored(xs, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, next_state, rtol=0, atol=0)
    assert type(restored) is type(cell)
    for name, tensor in cell.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


READOUTS = [
    readouts.ReadoutLayer,
    readouts.AutoReadoutLayer,
    readouts.CGReadoutLayer,
    readouts.RidgeReadoutLayer,
    readouts.CholeskyReadoutLayer,
    readouts.SVDReadoutLayer,
    readouts.PinvReadoutLayer,
    readouts.IncrementalRidgeReadoutLayer,
    readouts.RLSReadout,
    readouts.FISTAReadoutLayer,
]


@pytest.mark.parametrize("cls", READOUTS, ids=lambda c: c.__name__)
def test_readout_constructor_configuration_and_gradients(tmp_path, cls):
    kwargs = {"name": "forecast", "trainable": True}
    if cls not in (readouts.ReadoutLayer, readouts.PinvReadoutLayer, readouts.RLSReadout):
        kwargs["alpha"] = 0.003
    if cls is readouts.AutoReadoutLayer:
        kwargs.update(solver="svd", rcond=1e-12, cpu_fallback=False)
    if cls is readouts.RLSReadout:
        kwargs.update(forgetting=0.9, delta=0.02)
    original = cls(5, 2, **kwargs).double()
    if cls is not readouts.ReadoutLayer:
        original.fit(
            torch.randn(24, 5, dtype=torch.float64), torch.randn(24, 2, dtype=torch.float64)
        )
    save_model(original, tmp_path / "readout.pt")
    restored = load(tmp_path / "readout.pt")
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(original(x), restored(x), rtol=0, atol=0)
    assert restored.is_fitted == original.is_fitted
    assert restored.name == original.name
    restored(x).sum().backward()
    assert restored.weight.grad is not None
    for name, value in kwargs.items():
        assert getattr(restored, name) == value


def _series():
    return torch.stack(
        (torch.sin(torch.arange(70.0) / 5), torch.cos(torch.arange(70.0) / 7)), dim=-1
    )


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("include_state", [False, True])
def test_preset_full_lifecycle_and_streaming(tmp_path, trusted, include_state):
    series = _series()
    original = rd.ESN(reservoir_size=9, washout=5, seed=5, leak_rate=0.6).fit(series)
    original.eval()
    original(series[:10])
    modes = {n: m.training for n, m in original.named_modules()}
    states = original.get_state()
    rng = torch.get_rng_state().clone()
    path = save_model(
        original,
        tmp_path / "model.pt",
        trusted=trusted,
        include_state=include_state,
        metadata={"name": "example"},
    )
    assert modes == {n: m.training for n, m in original.named_modules()}
    for name, state in original.get_state().items():
        torch.testing.assert_close(state, states[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    restored, meta = load(path, trusted=trusted, return_metadata=True)
    assert meta == {"name": "example"}
    assert isinstance(restored, rd.ESN)
    assert original._factory_options == restored._factory_options
    if include_state:
        torch.testing.assert_close(original(series[10:15]), restored(series[10:15]), rtol=0, atol=0)
    else:
        assert not restored.get_state()
    torch.testing.assert_close(
        original.forecast(horizon=8), restored.forecast(horizon=8), rtol=0, atol=0
    )


def test_noisy_stream_exact_next_chunk_and_modes(tmp_path):
    for trusted in (False, True):
        layer = rd.ESNLayer(7, feedback_size=2, noise=0.2, seed=5).train()
        layer(torch.randn(1, 6, 2))
        save_model(layer, tmp_path / "noise.pt", trusted=trusted)
        restored = load(tmp_path / "noise.pt", trusted=trusted)
        x = torch.randn(1, 7, 2)
        torch.testing.assert_close(layer(x), restored(x), rtol=0, atol=0)
        assert restored.training


def test_custom_registered_layer_and_component_version(tmp_path, isolated_component_registries):
    class Scale(nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = factor

        def forward(self, x):
            return self.factor * x

    register_component(
        "test.custom.Scale",
        Scale,
        encode=lambda m: {"factor": m.factor},
        decode=lambda c: Scale(**c),
    )
    x = rd.Input((2, 2))
    graph = rd.Model(
        inputs={"state": x}, outputs={"prediction": Scale(3)(x)}, preprocessing="off", washout=0
    )
    save_model(graph, tmp_path / "custom.pt")
    restored = load(tmp_path / "custom.pt")
    assert isinstance(restored.graph.outputs[0].layer, Scale)
    payload = torch.load(tmp_path / "custom.pt", weights_only=True)
    for record in payload["components"].values():
        if record["name"] == "test.custom.Scale":
            record["version"] = 2
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ArtifactError, match="Unsupported component version"):
        load(tmp_path / "bad.pt")


def test_unregistered_subclass_never_silently_becomes_parent(tmp_path):
    class Extra(rd.ESNLayer):
        pass

    with pytest.raises(ArtifactError, match="register_component"):
        save_model(Extra(3, feedback_size=1), tmp_path / "model.pt")


def test_trusted_notebook_class_fresh_process_and_default_refusal(tmp_path):
    class NotebookLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(3.0))

        def forward(self, x):
            return self.scale * x

    path = save_model(NotebookLayer(), tmp_path / "notebook.pt", trusted=True)
    with pytest.raises(ArtifactError, match="trusted=True"):
        load(path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from resdag.artifacts import load; import torch; m=load({str(path)!r},trusted=True); assert m(torch.ones(1)).item()==3",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_atomic_failed_save_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    save_model(nn.Linear(2, 1), path)
    old = path.read_bytes()

    def broken(*args, **kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError, match="disk write"):
        save_model(nn.Linear(2, 1), path)
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]
