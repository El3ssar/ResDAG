"""Forecast hooks agree with ordinary PyTorch calls, units and explicit state."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.nn.modules import module as module_runtime

import resdag as rd
from resdag.core._forecast_capabilities import ForecastSequenceExecutor
from resdag.core._forecast_stream import forecast_stream
from resdag.core._hooks import forecast_as


def _identity():
    source = rd.reservoir_input(1, dtype=torch.float64)
    linear = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        linear.weight.fill_(1)
    return rd.ESNModel(source, linear(source)), linear


def _twice(module, args, output):
    return output * 2


def _plus_one(module, args):
    return (args[0] + 1, *args[1:])


def _ordinary_forecast(model, history, horizon):
    model.reset_state() if hasattr(model, "reset_state") else model.reset_reservoirs()
    current = model(history)[..., -1:, :]
    outputs = [current]
    for _ in range(horizon - 1):
        current = model(current)
        outputs.append(current)
    return torch.cat(outputs, dim=-2)


@pytest.mark.parametrize("location", ["root", "linear"])
def test_added_and_removed_pre_and_post_hooks_change_cached_forecasts(location):
    graph, linear = _identity()
    history = torch.ones(1, 2, 1, dtype=torch.float64)
    graph.forecast(history, horizon=3)
    lean = graph._get_flat_step()
    target = graph if location == "root" else linear
    before = target.register_forward_pre_hook(_plus_one)
    after = target.register_forward_hook(_twice)
    try:
        torch.testing.assert_close(
            graph.forecast(history, horizon=3),
            torch.tensor([[[4.0], [10.0], [22.0]]]).double(),
            rtol=0,
            atol=0,
        )
        assert not graph._get_flat_step().supports_compile
    finally:
        before.remove()
        after.remove()
    assert graph._get_flat_step() is lean
    torch.testing.assert_close(graph.forecast(history, horizon=3), torch.ones(1, 3, 1).double())


def test_native_kwargs_hooks_and_nested_container_hooks_are_preserved():
    source = rd.reservoir_input(1, dtype=torch.float64)
    block = nn.Sequential(nn.Identity(), nn.Identity())
    graph = rd.ESNModel(source, block(source))

    def pre(module, args, kwargs):
        return (args[0] + 1,), kwargs

    handles = [
        block.register_forward_pre_hook(pre, with_kwargs=True),
        block[1].register_forward_hook(_twice),
        block.register_forward_hook(
            lambda module, args, kwargs, output: output + 3, with_kwargs=True
        ),
    ]
    try:
        result = graph.forecast(torch.ones(1, 2, 1).double(), horizon=3)
        torch.testing.assert_close(result, torch.tensor([[[7.0], [19.0], [43.0]]]).double())
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("kind", ["pre", "post"])
def test_global_hooks_are_detected_after_cache_creation_and_removed_cleanly(kind):
    graph, linear = _identity()
    history = torch.ones(1, 2, 1).double()
    graph.forecast(history, horizon=3)
    lean = graph._get_flat_step()

    def hook(module, args, output):
        return output * 2 if module is linear else None

    handle = (
        module_runtime.register_module_forward_hook(hook)
        if kind == "post"
        else module_runtime.register_module_forward_pre_hook(
            lambda module, args: _plus_one(module, args) if module is linear else None
        )
    )
    try:
        expected = [2.0, 4.0, 8.0] if kind == "post" else [2.0, 3.0, 4.0]
        torch.testing.assert_close(
            graph.forecast(history, horizon=3), torch.tensor(expected).reshape(1, 3, 1).double()
        )
    finally:
        handle.remove()
    assert graph._get_flat_step() is lean


def test_backward_hooks_and_upstream_gradients_match_independent_ordinary_calls():
    graph, linear = _identity()
    reference, other = _identity()
    backward_calls = []
    for target in (linear, other):
        target.register_forward_pre_hook(_plus_one)
        target.register_forward_hook(_twice)
    linear.register_full_backward_hook(lambda module, grad_in, grad_out: backward_calls.append(1))
    data = torch.ones(1, 2, 1, dtype=torch.float64, requires_grad=True)
    original = data.detach().clone().requires_grad_()
    actual = graph.forecast(data, horizon=3, no_grad=False)
    expected = _ordinary_forecast(reference, original, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(data.grad, original.grad, rtol=0, atol=0)
    torch.testing.assert_close(linear.weight.grad, other.weight.grad, rtol=0, atol=0)
    assert len(backward_calls) == 3


def test_reservoir_hooks_in_explicit_execution_preserve_states_and_gradients():
    graph = rd.classic_esn(5, 1, 1, seed=3, dtype=torch.float64, trainable=True)
    reference = copy.deepcopy(graph)
    for model in (graph, reference):
        model.ESNLayer_1.detach_state_between_calls = False
        model.ESNLayer_1.register_forward_pre_hook(_plus_one)
        model.ESNLayer_1.register_forward_hook(_twice)
    data = torch.randn(2, 4, 1, dtype=torch.float64, requires_grad=True)
    original = data.detach().clone().requires_grad_()
    old = torch.ones(2, 5, dtype=torch.float64)
    graph.ESNLayer_1.set_state(old)
    stored = graph.ESNLayer_1.state
    state = torch.zeros(2, 5, dtype=torch.float64, requires_grad=True)
    reference_state = state.detach().clone().requires_grad_()
    reference.ESNLayer_1.set_state(reference_state)
    expected = reference(original)
    expected_final = reference.ESNLayer_1.state
    actual, final = graph.forward_stateless(data, states={"ESNLayer_1": state})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(final["ESNLayer_1"], expected_final, rtol=0, atol=0)
    assert graph.ESNLayer_1.state is stored and graph.ESNLayer_1._state_user_set
    (actual.sum() + final["ESNLayer_1"].sum()).backward()
    (expected.sum() + expected_final.sum()).backward()
    torch.testing.assert_close(data.grad, original.grad)
    torch.testing.assert_close(state.grad, reference_state.grad)
    for parameter, other in zip(graph.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, other.grad)


def test_hook_exceptions_restore_explicit_state_and_native_always_call_behavior():
    graph = rd.classic_esn(4, 1, 1, seed=7, dtype=torch.float64)
    graph.ESNLayer_1.set_state(torch.ones(1, 4).double())
    stored = graph.ESNLayer_1.state
    calls = []

    def fail(module, args, output):
        raise RuntimeError("hook failure")

    handle = graph.ESNLayer_1.register_forward_hook(fail)
    always = graph.register_forward_hook(lambda *args: calls.append(1), always_call=True)
    try:
        with pytest.raises(RuntimeError, match="hook failure"):
            graph.forward_stateless(torch.ones(1, 2, 1).double())
        assert graph.ESNLayer_1.state is stored
        assert graph.ESNLayer_1._state_user_set
        assert graph.ESNLayer_1.detach_state_between_calls
        assert calls == [1]
    finally:
        handle.remove()
        always.remove()


@pytest.mark.parametrize("facade", [False, True])
def test_workflow_and_facade_hooks_act_in_original_units(facade):
    data = torch.sin(torch.arange(60, dtype=torch.float64)[:, None] * 0.1)
    model = rd.ESN(8, input_size=1, washout=3, seed=7, dtype=torch.float64).fit(data)
    if not facade:
        model = model.workflow
    reference = copy.deepcopy(model)
    for value in (model, reference):
        value.register_forward_hook(lambda module, args, output: output + 0.25)
        workflow = value.workflow if isinstance(value, rd.ESN) else value
        workflow.input_scalers[0].register_forward_hook(lambda module, args, output: output * 0.9)
    history = data[-8:].unsqueeze(0)
    expected = _ordinary_forecast(reference, history, 5)
    actual = model.forecast(history, horizon=5)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    assert model.training and reference.training
    data = history.detach().clone().requires_grad_()
    other = history.detach().clone().requires_grad_()
    actual, final = model.forward_stateless(data)
    # Stateless prediction includes every teacher-forced timestamp.
    reference.reset_state()
    expected = reference(other)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(data.grad, other.grad)
    assert final


@pytest.mark.parametrize("execution", ["fast", "history"])
def test_named_delayed_feedback_and_streams_keep_hooked_original_units(execution):
    source = rd.reservoir_input(1, dtype=torch.float64)
    head = rd.RidgeReadoutLayer(1, 1, alpha=0).double()
    model = rd.Model(
        inputs={"state": source},
        outputs={"next": head(source)},
        washout=0,
        prediction_lead=2,
        feedback=[rd.Feedback("next", "state", delay=3)],
    )
    model.fit(torch.arange(8, dtype=torch.float64)[:, None])
    model.register_forward_hook(lambda module, args, output: output + 10)
    history = torch.arange(5, dtype=torch.float64)[:, None]
    expected = torch.tensor([[15.0], [16.0], [26.0], [27.0], [28.0], [38.0]]).double()
    torch.testing.assert_close(model.forecast(history, horizon=6, execution=execution), expected)
    with forecast_stream(model, history, horizon=6, chunk_size=2, execution=execution) as stream:
        actual = torch.cat([chunk.predictions for chunk in stream])
    torch.testing.assert_close(actual, expected)


def test_compilation_uses_warned_compatible_eager_path_only_with_hooks(monkeypatch):
    graph, linear = _identity()
    linear.register_forward_hook(_twice)
    monkeypatch.setattr(
        torch, "compile", lambda *args, **kwargs: pytest.fail("must preserve hooks eagerly")
    )
    with pytest.warns(RuntimeWarning, match="PyTorch hooks"):
        result = graph.forecast(torch.ones(1, 2, 1).double(), horizon=3, compile=True)
    torch.testing.assert_close(result, torch.tensor([[[2.0], [4.0], [8.0]]]).double())


def test_native_coupled_root_hooks_and_individual_outputs_use_the_same_inputs():
    first, _ = _identity()
    second, _ = _identity()
    model = rd.CoupledEnsembleESNModel([first, second])
    model.register_forward_pre_hook(_plus_one)
    model.register_forward_hook(_twice)
    result, individuals = model.forecast(
        torch.ones(1, 2, 1).double(), horizon=3, return_individuals=True
    )
    torch.testing.assert_close(result, torch.tensor([[[4.0], [10.0], [22.0]]]).double())
    for individual in individuals:
        torch.testing.assert_close(individual, result / 2)


@pytest.mark.parametrize("family", ["parallel", "coupled"])
def test_native_family_hooks_match_ordinary_forecast_and_stateless_gradients(family):
    if family == "parallel":
        model = rd.parallel_esn(
            input_dim=4,
            core_width=2,
            halo=1,
            reservoir_size=5,
            seed=7,
            dtype=torch.float64,
            readout="ridge",
        )
        values = torch.randn(2, 14, 4, dtype=torch.float64)
        model.fit(values[:, :3], values[:, 3:-1], values[:, 4:])
        width = 4
    else:
        model = rd.CoupledEnsembleESNModel(
            [
                rd.classic_esn(5, 1, 1, seed=3, dtype=torch.float64),
                rd.classic_esn(5, 1, 1, seed=7, dtype=torch.float64),
            ]
        )
        width = 1
    reference = copy.deepcopy(model)
    for value in (model, reference):
        value.register_forward_pre_hook(_plus_one)
        value.register_forward_hook(_twice)
        for reservoir in value.modules():
            if isinstance(reservoir, rd.ESNLayer):
                reservoir.register_forward_hook(lambda module, args, output: output * 0.9)
                reservoir.detach_state_between_calls = False
    history = torch.randn(2, 4, width, dtype=torch.float64)
    expected = _ordinary_forecast(reference, history, 4)
    actual = model.forecast(history, horizon=4)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    data = history.detach().clone().requires_grad_()
    original = data.detach().clone().requires_grad_()
    reference.reset_reservoirs()
    expected = reference(original)
    stored = {
        name: reservoir.state
        for name, reservoir in model.named_modules()
        if isinstance(reservoir, rd.ESNLayer)
    }
    actual, states = model.forward_stateless(data)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert states
    for name, reservoir in model.named_modules():
        if isinstance(reservoir, rd.ESNLayer):
            assert reservoir.state is stored[name]
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(data.grad, original.grad)


def test_tuning_stream_preserves_the_native_facade_hook_owner():
    pytest.importorskip("optuna")
    model = rd.ESN(6, input_size=1, washout=2, seed=7, dtype=torch.float64)
    model.register_forward_hook(lambda module, args, output: torch.full_like(output, 10))

    def objective(context):
        for predicted in context.predictions.values():
            torch.testing.assert_close(predicted, torch.full_like(predicted, 10), rtol=0, atol=0)
        return 1.0

    values = torch.sin(torch.arange(50, dtype=torch.float64)[:, None] * 0.1)
    model.tune(
        values,
        preset="quick",
        n_trials=1,
        horizon=3,
        n_origins=1,
        search_space=lambda trial: {"alpha": 1e-4},
        objective=objective,
        catch_exceptions=False,
    )
    assert model.tuning_report.score == 1
    torch.testing.assert_close(model.forecast(horizon=3), torch.full((3, 1), 10).double())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_native_mps_hooked_forecast_matches_ordinary_calls():
    values = torch.sin(torch.arange(30, device="mps", dtype=torch.float32)[:, None] * 0.1)
    model = rd.ESN(
        6,
        input_size=1,
        washout=2,
        seed=7,
        device="mps",
        dtype=torch.float32,
        readout="ridge",
        readout_kwargs={"solver": "solve", "use_float64": False},
    ).fit(values)
    reference = copy.deepcopy(model)
    for value in (model, reference):
        value.register_forward_pre_hook(_plus_one)
        value.register_forward_hook(_twice)
    history = values[-4:].unsqueeze(0)
    actual = model.forecast(history, horizon=4)
    expected = _ordinary_forecast(reference, history, 4)
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_named_multihead_hook_results_use_nonfirst_feedback_and_distinct_target_units():
    source = rd.reservoir_input(1, dtype=torch.float64)
    auxiliary = rd.RidgeReadoutLayer(1, 1, alpha=0, name="auxiliary").double()(source)
    prediction = rd.RidgeReadoutLayer(1, 1, alpha=0, name="prediction").double()(source)
    model = rd.Model(
        inputs={"state": source},
        outputs={"aux": auxiliary, "next": prediction},
        washout=0,
        feedback=[rd.Feedback("next", "state")],
    )
    values = torch.arange(8, dtype=torch.float64)[:, None]
    model.fit(values, targets={"aux": 2 * values + 1, "next": values + 1})
    model.register_forward_hook(
        lambda module, args, output: {"aux": output["aux"] + 20, "next": output["next"] + 10}
    )
    actual = model.forecast(values[:5], horizon=3)
    assert list(actual) == ["aux", "next"]
    torch.testing.assert_close(actual["next"], torch.tensor([[15.0], [26.0], [37.0]]).double())
    torch.testing.assert_close(actual["aux"], torch.tensor([[29.0], [51.0], [73.0]]).double())


@pytest.mark.parametrize("facade", [False, True])
def test_reservoirs_outside_declared_graph_are_rejected_before_forecast_hooks(facade):
    values = torch.sin(torch.arange(30, dtype=torch.float64)[:, None] * 0.1)
    model = rd.ESN(4, input_size=1, washout=2, seed=7, dtype=torch.float64).fit(values)
    if not facade:
        model = model.workflow
    model.add_module(
        "external_memory", rd.ESNLayer(3, feedback_size=1, seed=2, dtype=torch.float64)
    )
    calls = []
    model.register_forward_hook(lambda *args: calls.append(1))
    assert model(values[:3]).shape == (3, 1)
    calls.clear()
    for run in (
        lambda: model.forward_stateless(values[:3]),
        lambda: model.forecast(values[:3], horizon=2),
    ):
        with pytest.raises(ValueError, match="outside.*model.graph"):
            run()
        assert calls == []


@pytest.mark.parametrize("execution", ["fast", "history"])
def test_single_output_cast_hook_preserves_native_precision_in_forecasts_and_streams(execution):
    source = rd.reservoir_input(1, dtype=torch.float64)
    model = rd.Model(
        graph=rd.ESNModel(source, nn.Identity()(source)), preprocessing=None, washout=0
    )
    model.register_forward_hook(lambda module, args, output: output.float())
    history = torch.ones(1, 2, 1, dtype=torch.float64)
    expected = _ordinary_forecast(model, history, 4)
    actual = model.forecast(history, horizon=4, execution=execution)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with forecast_stream(model, history, horizon=4, chunk_size=2, execution=execution) as stream:
        streamed = torch.cat([chunk.predictions for chunk in stream], dim=1)
    assert streamed.dtype == torch.float32
    torch.testing.assert_close(streamed, expected, atol=0, rtol=0)


@pytest.mark.parametrize("facade", [False, True])
@pytest.mark.parametrize("execution", ["fast", "history"])
def test_paired_cast_hooks_reconcile_feedback_with_parameter_dtype(facade, execution):
    values = torch.sin(torch.arange(35, dtype=torch.float64)[:, None] * 0.1)
    model = rd.ESN(5, input_size=1, washout=2, dtype=torch.float64, seed=7).fit(values)
    if not facade:
        model = model.workflow
    model.register_forward_pre_hook(lambda module, args: (args[0].double(),))
    model.register_forward_hook(lambda module, args, output: output.float())
    reference = copy.deepcopy(model)
    history = values[-5:].unsqueeze(0)
    expected = _ordinary_forecast(reference, history, 4)
    actual = model.forecast(history, horizon=4, execution=execution)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    workflow = model.workflow if facade else model
    # Exercise the same owner context used by the tuning stream.
    with (
        forecast_as(model),
        forecast_stream(workflow, history, horizon=4, chunk_size=2, execution=execution) as stream,
    ):
        streamed = torch.cat([chunk.predictions for chunk in stream], dim=1)
    torch.testing.assert_close(streamed, expected, atol=0, rtol=0)


@pytest.mark.parametrize("execution", ["fast", "history"])
def test_native_prehooks_can_reconcile_mixed_feedback_and_driver_precision(execution):
    state = rd.reservoir_input(1, dtype=torch.float64)
    forcing = rd.reservoir_input(1, dtype=torch.float64)
    model = rd.Model(
        inputs={"state": state, "forcing": forcing},
        outputs={"next": state + forcing},
        preprocessing=None,
        washout=0,
    )
    model.register_forward_pre_hook(
        lambda module, args: ({name: value.double() for name, value in args[0].items()},)
    )
    model.register_forward_hook(lambda module, args, output: output.float())
    history = {name: torch.ones(1, 2, 1, dtype=torch.float64) for name in model.input_names}
    drivers = {"forcing": torch.arange(1, 4, dtype=torch.float64).reshape(1, 3, 1)}
    current = model(history)[:, -1:]
    expected = [current]
    for index in range(3):
        current = model({"state": current, "forcing": drivers["forcing"][:, index : index + 1]})
        expected.append(current)
    expected = torch.cat(expected, dim=1)
    actual = model.forecast(history, horizon=4, future_inputs=drivers, execution=execution)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with forecast_stream(
        model, history, horizon=4, future_inputs=drivers, chunk_size=2, execution=execution
    ) as stream:
        streamed = torch.cat([chunk.predictions for chunk in stream], dim=1)
    torch.testing.assert_close(streamed, expected, atol=0, rtol=0)


@pytest.mark.parametrize("execution", ["fast", "history"])
@pytest.mark.parametrize("hooked", [False, True])
@pytest.mark.parametrize("invalid", ["batch", "width", "frames", "ports"])
def test_hooked_and_lean_executors_reject_invalid_structure_before_dispatch(
    execution, hooked, invalid
):
    graph, linear = _identity()
    if hooked:
        linear.register_forward_hook(lambda module, args, output: output.float())
    calls = []
    if hooked:
        graph.register_forward_pre_hook(lambda *args: calls.append(1))
    executor = ForecastSequenceExecutor(graph, execution)
    executor.start((torch.ones(1, 2, 1, dtype=torch.float64),))
    before = len(calls)
    inputs = {
        "batch": (torch.ones(2, 1, 1, dtype=torch.float64),),
        "width": (torch.ones(1, 1, 2, dtype=torch.float64),),
        "frames": (torch.ones(1, 2, 1, dtype=torch.float64),),
        "ports": (),
    }[invalid]
    with pytest.raises(ValueError):
        executor.advance(inputs)
    assert len(calls) == before


@pytest.mark.parametrize("execution", ["fast", "history"])
def test_hook_free_executor_keeps_strict_feedback_placement(execution):
    graph, _ = _identity()
    executor = ForecastSequenceExecutor(graph, execution)
    executor.start((torch.ones(1, 2, 1, dtype=torch.float64),))
    with pytest.raises(ValueError, match="one frame"):
        executor.advance((torch.ones(1, 1, 1, dtype=torch.float32),))


def test_native_cast_hooks_preserve_raw_forecast_input_and_parameter_gradients():
    graph, linear = _identity()
    graph.register_forward_pre_hook(lambda module, args: (args[0].double(),))
    graph.register_forward_hook(lambda module, args, output: output.float())
    reference = copy.deepcopy(graph)
    data = torch.ones(1, 2, 1, dtype=torch.float64, requires_grad=True)
    other = data.detach().clone().requires_grad_()
    actual = graph.forecast(data, horizon=4, no_grad=False)
    expected = _ordinary_forecast(reference, other, 4)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(data.grad, other.grad, atol=0, rtol=0)
    reference_linear = next(
        module for module in reference.modules() if isinstance(module, nn.Linear)
    )
    torch.testing.assert_close(linear.weight.grad, reference_linear.weight.grad, atol=0, rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("facade", [False, True])
def test_native_fast_cast_hooks_can_route_feedback_across_devices(facade):
    values = torch.sin(torch.arange(30, dtype=torch.float32)[:, None] * 0.1)
    model = rd.ESN(5, input_size=1, washout=2, seed=7, readout="ridge").fit(values)
    if not facade:
        model = model.workflow
    model.register_forward_pre_hook(lambda module, args: (args[0].cpu(),))
    model.register_forward_hook(lambda module, args, output: output.to("mps"))
    reference = copy.deepcopy(model)
    history = values[-5:].unsqueeze(0)
    expected = _ordinary_forecast(reference, history, 4)
    actual = model.forecast(history, horizon=4)
    assert actual.device.type == expected.device.type == "mps"
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    workflow = model.workflow if facade else model
    with forecast_as(model), forecast_stream(workflow, history, horizon=4, chunk_size=2) as stream:
        streamed = torch.cat([chunk.predictions for chunk in stream], dim=1)
    assert streamed.device.type == "mps"
    torch.testing.assert_close(streamed, expected, atol=0, rtol=0)
