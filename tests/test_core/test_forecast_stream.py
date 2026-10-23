"""Bounded forecast generation agrees with independent clocks and full execution."""

import copy

import pytest
import torch
from torch import nn

from resdag import ESNModel, Model, RidgeReadoutLayer, SVDReadoutLayer, reservoir_input
from resdag.core._feedback import Feedback
from resdag.core._forecast_capabilities import ForecastSequenceExecutor
from resdag.core._forecast_stream import forecast_stream, raw_forecast_stream
from resdag.ensemble import CoupledEnsembleESNModel
from resdag.layers import Concatenate
from resdag.layers.reservoirs import BaseReservoirLayer
from resdag.models import classic_esn, parallel_esn


def _ramp(*, lead=1, delay=None, device="cpu"):
    dtype = torch.float32 if device == "mps" else torch.float64
    x = reservoir_input(1, dtype=dtype, device=device)
    readout = RidgeReadoutLayer(
        1,
        1,
        alpha=0,
        solver="solve" if device == "mps" else "cholesky",
        use_float64=device != "mps",
        gram_dtype=torch.float32 if device == "mps" else None,
    ).to(device=device, dtype=dtype)
    model = Model(
        inputs={"state": x},
        outputs={"next": readout(x)},
        prediction_lead=lead,
        feedback=None if delay is None else [Feedback("next", "state", delay)],
        washout=0,
    )
    data = torch.arange(12, device=device, dtype=dtype)[:, None]
    return model.fit(data), data


def _collect(model, history=None, **kwargs):
    with forecast_stream(model, history, **kwargs) as stream:
        chunks = list(stream)
    return {
        name: torch.cat([chunk.named_predictions[name] for chunk in chunks], dim=-2)
        for name in model.output_names
    }, chunks


class _NonemptyUnitTransform(nn.Module):
    supports_framewise = True

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(1, dtype=torch.float64))

    def fit(self, training):
        return self

    def forward(self, value):
        assert value.shape[-2] > 0, "Valid framewise transforms need not accept empty sequences"
        return value * self.scale

    def inverse(self, value):
        return value / self.scale


@pytest.mark.parametrize("execution", ["fast", "history"])
@pytest.mark.parametrize("supplied", [False, True])
def test_zero_required_driver_frames_do_not_call_custom_preprocessing(execution, supplied):
    state, forcing = (reservoir_input(1, dtype=torch.float64) for _ in range(2))
    head = SVDReadoutLayer(2, 1, alpha=0).double()(Concatenate()(state, forcing))
    model = Model(
        inputs={"state": state, "driver": forcing},
        outputs={"prediction": head},
        prediction_lead=3,
        feedback=[Feedback("prediction", "state", 3)],
        input_transforms={"driver": _NonemptyUnitTransform()},
        washout=0,
    )
    data = torch.arange(30, dtype=torch.float64)[:, None]
    model.fit({"state": data, "driver": data * 2}, data + 3)
    kwargs = {"future_inputs": {"driver": data[:0]}} if supplied else {}
    expected = copy.deepcopy(model).forecast(horizon=2, execution=execution, **kwargs)
    actual, _ = _collect(model, horizon=2, execution=execution, chunk_size=1, **kwargs)
    torch.testing.assert_close(actual["prediction"], expected)


@pytest.mark.parametrize("lead,delay", [(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)])
@pytest.mark.parametrize("horizon,chunk_size", [(1, 1), (2, 1), (8, 3), (9, 64)])
@pytest.mark.parametrize("execution", ["fast", "history"])
def test_ramp_units_and_physical_clock_match_complete_forecast(
    lead, delay, horizon, chunk_size, execution
):
    model, data = _ramp(lead=lead, delay=delay)
    expected = copy.deepcopy(model).forecast(data[-5:], horizon=horizon, execution=execution)
    result, chunks = _collect(
        model, data[-5:], horizon=horizon, chunk_size=chunk_size, execution=execution
    )
    torch.testing.assert_close(result["next"], expected, rtol=1e-12, atol=1e-12)
    if lead == delay:
        torch.testing.assert_close(
            result["next"], torch.arange(12, 12 + horizon, dtype=data.dtype)[:, None]
        )
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [
        (start, min(start + chunk_size, horizon)) for start in range(0, horizon, chunk_size)
    ]


@pytest.mark.parametrize("execution", ["fast", "history"])
def test_named_nonfirst_feedback_distinct_leads_and_drivers_have_independent_clock_oracle(
    execution,
):
    state, forcing = (reservoir_input(1, dtype=torch.float64) for _ in range(2))
    features = Concatenate()(state, forcing)
    one = RidgeReadoutLayer(2, 1, alpha=0, name="one").double()(features)
    three = RidgeReadoutLayer(2, 1, alpha=0, name="three").double()(features)
    model = Model(
        inputs={"driver": forcing, "state": state},
        outputs={"three": three, "one": one},
        prediction_lead={"three": 3},
        feedback=[Feedback("three", "state", 3)],
        washout=0,
    )
    clock = torch.arange(20, dtype=torch.float64)[:, None]
    values = clock * (clock + 1) / 2
    drivers = clock + 1
    model.fit(
        {"state": values, "driver": drivers},
        {"one": values + drivers, "three": values + 3 * drivers + 3},
    )
    history = {"driver": drivers[:5], "state": values[:5]}
    actual, chunks = _collect(
        model,
        history,
        horizon=9,
        future_inputs={"driver": drivers[5:13]},
        execution=execution,
        chunk_size=4,
        checkpoints=[3, 7],
    )
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 3), (3, 7), (7, 9)]
    for head in actual.values():
        torch.testing.assert_close(head, values[5:14], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("family", ["symbolic", "parallel", "coupled"])
@pytest.mark.parametrize("execution", ["fast", "history"])
def test_values_final_states_and_normal_state_tensors_match_all_backends(family, execution):
    width = 4 if family == "parallel" else 1
    if family == "parallel":
        graph = parallel_esn(
            4, core_width=2, halo=1, reservoir_size=5, seed=31, dtype=torch.float64
        )
    else:
        members = [classic_esn(5, 1, 1, seed=seed, dtype=torch.float64) for seed in (31, 32)]
        graph = members[0] if family == "symbolic" else CoupledEnsembleESNModel(members)
    data = torch.randn(
        2, 30, width, generator=torch.Generator().manual_seed(71), dtype=torch.float64
    )
    model = Model(graph=graph, washout=2).fit(data)
    reference = copy.deepcopy(model)
    expected = reference.forecast(data[:, -7:], horizon=11, execution=execution)
    with torch.inference_mode():
        actual, _ = _collect(model, data[:, -7:], horizon=11, chunk_size=3, execution=execution)
    torch.testing.assert_close(actual[model.output_names[0]], expected, rtol=1e-12, atol=1e-12)
    assert model.get_state()
    for name, value in model.get_state().items():
        torch.testing.assert_close(value, reference.get_state()[name], rtol=1e-12, atol=1e-12)
        assert not value.is_inference()


def test_history_mode_replays_sequence_dependent_layer_across_chunk_boundaries():
    class Cumulative(nn.Module):
        def forward(self, value):
            return value.cumsum(1)

    x = reservoir_input(1, dtype=torch.float64)
    y = RidgeReadoutLayer(1, 1, alpha=0).double()(Cumulative()(x))
    data = torch.arange(1, 14, dtype=torch.float64)[:, None]
    model = Model(inputs={"x": x}, outputs={"y": y}, washout=0, preprocessing="off").fit(
        data, data.cumsum(0)
    )
    reference = copy.deepcopy(model).forecast(data[-4:], horizon=7, execution="history")
    actual, _ = _collect(model, data[-4:], horizon=7, chunk_size=2, execution="history")
    torch.testing.assert_close(actual["y"], reference)
    with pytest.raises(ValueError, match="one-step|framewise|fast"):
        with forecast_stream(model, horizon=7) as stream:
            next(stream)


@pytest.mark.parametrize("boundary", ["next", "legacy", "explicit", "empty"])
@pytest.mark.parametrize("reset", [False, True])
def test_raw_batched_continuation_and_seeds_match_full_forecast(boundary, reset):
    model = classic_esn(7, 1, 1, seed=17, dtype=torch.float64)
    data = torch.randn(2, 8, 1, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    model(data[:, :3])
    reference = copy.deepcopy(model)
    history = data[:, :0] if boundary == "empty" else data[:, 3:]
    kwargs = {"alignment": "legacy"} if boundary == "legacy" else {}
    if boundary in ("explicit", "empty"):
        kwargs["initial_feedback"] = torch.full((2, 1, 1), 0.3, dtype=torch.float64)
    expected = reference.forecast(history, horizon=13, reset=reset, **kwargs)
    with raw_forecast_stream(
        model, history, horizon=13, reset=reset, chunk_size=4, **kwargs
    ) as stream:
        chunks = list(stream)
    torch.testing.assert_close(
        torch.cat([chunk.outputs[0] for chunk in chunks], 1), expected, rtol=1e-12, atol=1e-12
    )
    for name, value in model.get_state().items():
        torch.testing.assert_close(value, reference.get_state()[name], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("alignment", ["next", "legacy"])
def test_raw_multihead_and_driver_chunks_match_independent_recurrence(alignment):
    feedback, driver = (reservoir_input(1, dtype=torch.float64) for _ in range(2))
    features = Concatenate()(feedback, driver)
    first = RidgeReadoutLayer(2, 1).double()
    second = RidgeReadoutLayer(2, 2).double()
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[0.5, 2.0]]).double())
        first.bias.fill_(1)
        second.weight.copy_(torch.tensor([[1.0, -1.0], [2.0, 3.0]]).double())
        second.bias.copy_(torch.tensor([0.0, -2.0]).double())
    model = ESNModel((feedback, driver), (first(features), second(features)))
    history = (
        torch.arange(6, dtype=torch.float64).reshape(2, 3, 1),
        torch.full((2, 3, 1), 0.25).double(),
    )
    drivers = torch.arange(22, dtype=torch.float64).reshape(2, 11, 1) / 10
    current = history[0][:, -1:] * 0.5 + history[1][:, -1:] * 2 + 1
    expected_first = [current] if alignment == "next" else []
    expected_second = (
        [
            torch.cat(
                (
                    history[0][:, -1:] - history[1][:, -1:],
                    2 * history[0][:, -1:] + 3 * history[1][:, -1:] - 2,
                ),
                -1,
            )
        ]
        if alignment == "next"
        else []
    )
    for step in range(11 - int(alignment == "next")):
        forcing = drivers[:, step : step + 1]
        expected_second.append(torch.cat((current - forcing, 2 * current + 3 * forcing - 2), -1))
        current = current * 0.5 + forcing * 2 + 1
        expected_first.append(current)
    with raw_forecast_stream(
        model,
        history,
        drivers,
        horizon=11,
        chunk_size=4,
        checkpoints=[3, 6, 10],
        alignment=alignment,
    ) as stream:
        chunks = list(stream)
    assert [chunk.stop for chunk in chunks] == [3, 6, 10, 11]
    torch.testing.assert_close(
        torch.cat([chunk.outputs[0] for chunk in chunks], 1), torch.cat(expected_first, 1)
    )
    torch.testing.assert_close(
        torch.cat([chunk.outputs[1] for chunk in chunks], 1), torch.cat(expected_second, 1)
    )


@pytest.mark.parametrize("raise_in_observer", [False, True])
def test_early_close_stops_actual_flat_generation_and_preserves_state_identity_and_modes(
    monkeypatch, raise_in_observer
):
    graph = classic_esn(5, 1, 1, seed=5, dtype=torch.float64)
    data = torch.arange(15, dtype=torch.float64)[:, None]
    model = Model(graph=graph, washout=1).fit(data)
    model.train()
    layers = [layer for layer in model.modules() if isinstance(layer, BaseReservoirLayer)]
    layers[0].eval()
    model(data[-4:])
    before = [(layer, layer.state, layer._state_user_set) for layer in layers]
    modes = [(layer, layer.training) for layer in model.modules()]
    flat = graph._get_flat_step()
    original = flat.step
    advances = []

    def counted(inputs, states):
        advances.append(1)
        return original(inputs, states)

    monkeypatch.setattr(flat, "step", counted)
    try:
        with forecast_stream(model, horizon=10**8, chunk_size=4) as stream:
            first = next(stream)
            assert first.stop == 4
            assert len(advances) == 3
            if raise_in_observer:
                raise RuntimeError("observer stopped")
    except RuntimeError as error:
        assert raise_in_observer and str(error) == "observer stopped"
    assert len(advances) == 3
    for layer, state, pinned in before:
        assert layer.state is state and layer._state_user_set == pinned
    assert [(layer, layer.training) for layer, _ in modes] == modes


def test_generic_early_close_uses_only_required_physical_prefix(monkeypatch):
    model, _ = _ramp(lead=3)
    original = ForecastSequenceExecutor.advance
    advances = []

    def counted(self, inputs):
        advances.append(1)
        return original(self, inputs)

    monkeypatch.setattr(ForecastSequenceExecutor, "advance", counted)
    with forecast_stream(model, horizon=10**8, chunk_size=4) as stream:
        assert next(stream).stop == 4
    assert len(advances) == 1


def test_raw_early_close_and_generation_exception_preserve_source_state_and_modes(monkeypatch):
    graph = classic_esn(5, 1, 1, seed=5, dtype=torch.float64)
    data = torch.arange(15, dtype=torch.float64).reshape(1, 15, 1)
    graph(data[:, :3])
    graph.set_state(graph.get_state())
    layers = [layer for layer in graph.modules() if isinstance(layer, BaseReservoirLayer)]
    before = [(layer, layer.state, layer._state_user_set) for layer in layers]
    graph.train()
    layers[0].eval()
    modes = [(layer, layer.training) for layer in graph.modules()]
    flat = graph._get_flat_step()
    original = flat.step
    advances = []

    def counted(inputs, states):
        advances.append(1)
        if len(advances) == 5:
            raise RuntimeError("fault inside generation")
        return original(inputs, states)

    monkeypatch.setattr(flat, "step", counted)
    with raw_forecast_stream(graph, data[:, -3:], horizon=10**8, chunk_size=4) as stream:
        assert next(stream).stop == 4
    assert len(advances) == 3
    advances.clear()
    with pytest.raises(RuntimeError, match="fault inside generation"):
        with raw_forecast_stream(graph, data[:, -3:], horizon=10**8, chunk_size=4) as stream:
            next(stream)
            next(stream)
    assert len(advances) == 5
    for layer, state, pinned in before:
        assert layer.state is state and layer._state_user_set == pinned
    assert [(layer, layer.training) for layer, _ in modes] == modes


def test_compiled_stream_uses_existing_chunk_engine_and_matches_eager(monkeypatch):
    # The eager Torch Dynamo backend verifies real capture without an expensive
    # platform compiler dependency in this focused behavioral test.
    compile_original = torch.compile
    captures = []

    def capture(function, **kwargs):
        captures.append(kwargs)
        return compile_original(function, backend="eager", fullgraph=True)

    monkeypatch.setattr(torch, "compile", capture)
    graph = classic_esn(4, 1, 1, seed=4, dtype=torch.float64)
    history = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1)
    expected = copy.deepcopy(graph).forecast(history, horizon=9)
    with raw_forecast_stream(graph, history, horizon=9, chunk_size=4, compile=True) as stream:
        chunks = list(stream)
    assert captures
    torch.testing.assert_close(torch.cat([chunk.outputs[0] for chunk in chunks], 1), expected)


@pytest.mark.parametrize("checkpoints", [[0], [3, 3], [4, 3], [8], [True], [1.5]])
def test_invalid_checkpoint_schedule_fails_before_execution(checkpoints):
    model, _ = _ramp()
    with pytest.raises(ValueError, match="checkpoint"):
        with forecast_stream(model, horizon=7, checkpoints=checkpoints):
            pytest.fail("Invalid checkpoints were accepted")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS is unavailable")
def test_native_mps_stream_preserves_device_and_original_units():
    model, data = _ramp(lead=2, device="mps")
    actual, chunks = _collect(model, horizon=7, chunk_size=3)
    assert all(chunk.outputs[0].device.type == "mps" for chunk in chunks)
    torch.testing.assert_close(
        actual["next"],
        torch.arange(12, 19, device="mps", dtype=data.dtype)[:, None],
        rtol=1e-4,
        atol=1e-4,
    )
