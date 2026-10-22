"""Explicit-state execution rejects hidden recurrent state before running nodes."""

import copy

import pytest
import torch
from torch import nn

from resdag import ESNModel, reservoir_input
from resdag.ensemble import CoupledEnsembleESNModel
from resdag.layers import ESNLayer


class _HiddenReservoir(nn.Module):
    def __init__(self, reservoir: ESNLayer) -> None:
        super().__init__()
        self.reservoir = reservoir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reservoir(x)


def _hidden_model(container: str = "sequential") -> tuple[ESNModel, nn.Module]:
    inp = reservoir_input(1, dtype=torch.float64)
    before = nn.Identity()
    reservoir = ESNLayer(1, feedback_size=1, seed=7, dtype=torch.float64)
    if container == "sequential":
        hidden = nn.Sequential(reservoir, nn.Identity())
    elif container == "nested":
        hidden = nn.Sequential(nn.Sequential(nn.Identity(), reservoir), nn.Identity())
    else:
        hidden = _HiddenReservoir(reservoir)
    return ESNModel(inp, hidden(before(inp))), before


@pytest.mark.parametrize(
    "entry",
    [
        "stateless",
        "stateless_compile",
        "adapter",
        "forecast_one",
        "forecast_many",
        "forecast_compile",
        "flat",
        "coupled_adapter",
        "coupled_forecast",
    ],
)
def test_hidden_reservoir_is_rejected_before_hooks_or_state_changes(entry: str) -> None:
    model, before = _hidden_model()
    x = torch.full((1, 3, 1), 0.1, dtype=torch.float64)
    model(x)
    stored = model.get_state()
    assert stored
    calls: list[None] = []
    handle = before.register_forward_hook(lambda *_: calls.append(None))
    try:
        with pytest.raises(ValueError, match="reservoir.*hidden inside.*functional graph node"):
            if entry in {"stateless", "stateless_compile"}:
                model.forward_stateless(x, compile=entry == "stateless_compile")
            elif entry == "adapter":
                model.as_module()
            elif entry == "forecast_one":
                model.forecast(x, horizon=1)
            elif entry == "forecast_many":
                model.forecast(x, horizon=3)
            elif entry == "forecast_compile":
                model.forecast(x, horizon=3, compile=True)
            elif entry == "flat":
                model._get_flat_step()
            elif entry == "coupled_adapter":
                CoupledEnsembleESNModel([model]).as_module()
            else:
                CoupledEnsembleESNModel([model]).forecast(x, horizon=1)
    finally:
        handle.remove()
    assert calls == []
    assert model.get_state().keys() == stored.keys()
    for name, state in model.get_state().items():
        torch.testing.assert_close(state, stored[name], rtol=0, atol=0)


@pytest.mark.parametrize("container", ["sequential", "nested", "custom"])
@pytest.mark.parametrize("coupled", [False, True])
def test_stateful_adapter_supports_hidden_reservoir_state_and_rejects_stateless_execution(
    container: str,
    coupled: bool,
) -> None:
    graph, before = _hidden_model(container)
    model = CoupledEnsembleESNModel([graph]) if coupled else graph
    reference = copy.deepcopy(model)
    adapter = model.as_module(stateful=True)
    x = torch.full((1, 3, 1), 0.1, dtype=torch.float64)
    first = adapter(x)
    torch.testing.assert_close(first, reference(x), rtol=0, atol=0)
    stored = adapter.get_state()
    for name, state in reference.get_state().items():
        torch.testing.assert_close(stored[name], state, rtol=0, atol=0)
    second = adapter(x)
    torch.testing.assert_close(second, reference(x), rtol=0, atol=0)
    assert not torch.allclose(second, first)
    adapter.reset_state()
    assert adapter.get_state() == {}
    adapter.set_state(stored)
    torch.testing.assert_close(adapter(x), second, rtol=0, atol=0)
    continued = adapter.get_state()
    calls: list[None] = []
    handle = before.register_forward_hook(lambda *_: calls.append(None))
    try:
        with pytest.raises(ValueError, match="hidden inside.*functional graph node"):
            adapter.forward_stateless(x)
    finally:
        handle.remove()
    assert calls == []
    for name, state in adapter.get_state().items():
        torch.testing.assert_close(state, continued[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "entry",
    [
        "stateless",
        "adapter",
        "stateful_adapter_stateless",
        "structural_plan_run",
        "flat",
        "forecast_one",
        "forecast_many",
        "coupled",
    ],
)
@pytest.mark.parametrize("flat_cached", [False, True])
def test_cached_execution_rejects_a_reservoir_appended_to_a_container(
    entry: str,
    flat_cached: bool,
) -> None:
    inp = reservoir_input(1, dtype=torch.float64)
    before = nn.Identity()
    container = nn.Sequential(nn.Identity())
    graph = ESNModel(inp, container(before(inp)))
    model = CoupledEnsembleESNModel([graph]) if entry == "coupled" else graph
    adapter = model.as_module(stateful=entry == "stateful_adapter_stateless")
    x = torch.full((1, 3, 1), 0.1, dtype=torch.float64)
    graph.forward_stateless(x)
    if flat_cached:
        graph._get_flat_step()
    structural_plan = graph._get_execution_plan(require_explicit_state=False)
    reservoir = ESNLayer(1, feedback_size=1, seed=7, dtype=torch.float64)
    container.append(reservoir)
    model(x)
    stored = model.get_state()
    assert stored
    calls: list[None] = []
    handle = before.register_forward_hook(lambda *_: calls.append(None))
    try:
        with pytest.raises(ValueError, match="hidden inside.*functional graph node"):
            if entry in {"stateless", "coupled"}:
                model.forward_stateless(x)
            elif entry == "adapter":
                adapter(x)
            elif entry == "stateful_adapter_stateless":
                adapter.forward_stateless(x)
            elif entry == "flat":
                graph._get_flat_step()
            elif entry == "structural_plan_run":
                structural_plan.run_sequence((x,))
            else:
                model.forecast(x, horizon=1 if entry == "forecast_one" else 3)
    finally:
        handle.remove()
    assert calls == []
    for name, state in model.get_state().items():
        torch.testing.assert_close(state, stored[name], rtol=0, atol=0)


@pytest.mark.parametrize("container", ["sequential", "nested", "custom"])
def test_structural_plan_describes_hidden_reservoir_but_cannot_execute_it(
    container: str,
) -> None:
    model, before = _hidden_model(container)
    x = torch.full((1, 3, 1), 0.1, dtype=torch.float64)
    calls: list[None] = []
    handle = before.register_forward_hook(lambda *_: calls.append(None))
    try:
        plan = model._get_execution_plan(require_explicit_state=False)
        assert len(plan.operations) == 2
        assert calls == [] and model.get_state() == {}
        with pytest.raises(ValueError, match="hidden inside.*functional graph node"):
            plan.run_sequence((x,))
        with pytest.raises(ValueError, match="hidden inside.*functional graph node"):
            model._get_execution_plan()
    finally:
        handle.remove()
    assert calls == [] and model.get_state() == {}


@pytest.mark.parametrize("container", ["sequential", "nested", "custom"])
def test_hidden_reservoir_error_identifies_the_module_and_preserves_stateful_forward(
    container: str,
) -> None:
    model, _ = _hidden_model(container)
    x = torch.full((1, 3, 1), 0.1, dtype=torch.float64)
    assert model.get_state() == {}
    with pytest.raises(ValueError, match="Expose each reservoir.*functional graph node") as error:
        model.forward_stateless(x)
    assert model.get_state() == {}
    reservoir_name = next(
        name for name, module in model.named_modules() if isinstance(module, ESNLayer)
    )
    assert reservoir_name in str(error.value)
    first = model(x)
    assert first.shape == x.shape
    assert model.get_state()
    model.reset_reservoirs()
    torch.testing.assert_close(model(x), first, rtol=0, atol=0)


def test_pointwise_container_preserves_explicit_state_and_chunk_gradient_parity() -> None:
    inp = reservoir_input(1, dtype=torch.float64)
    reservoir = ESNLayer(4, feedback_size=1, seed=9, dtype=torch.float64)
    readout = nn.Sequential(nn.Linear(4, 2, dtype=torch.float64), nn.Tanh())
    model = ESNModel(inp, readout(reservoir(inp)))
    chunked = copy.deepcopy(model)
    source = torch.randn(2, 7, 1, dtype=torch.float64, requires_grad=True)
    chunk_source = source.detach().clone().requires_grad_()
    full, full_state = model.forward_stateless(source)
    first, state = chunked.forward_stateless(chunk_source[:, :3])
    second, final_state = chunked.forward_stateless(chunk_source[:, 3:], states=state)
    combined = torch.cat((first, second), dim=1)
    torch.testing.assert_close(combined, full, rtol=0, atol=0)
    for name in full_state:
        torch.testing.assert_close(final_state[name], full_state[name], rtol=0, atol=0)
    full.square().sum().backward()
    combined.square().sum().backward()
    torch.testing.assert_close(chunk_source.grad, source.grad)
    for actual, expected in zip(chunked.parameters(), model.parameters()):
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad)
    assert model.get_state() == chunked.get_state() == {}
