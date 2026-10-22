"""Standard FX composition keeps the reservoir as one stateful module call."""

import copy

import pytest
import torch
from torch import fx, nn

import resdag as rd


class _ReservoirTracer(fx.Tracer):
    def is_leaf_module(self, module, qualified_name):
        return isinstance(module, (rd.Model, rd.ESN)) or super().is_leaf_module(
            module, qualified_name
        )


class _Pipeline(nn.Module):
    def __init__(self, model, named):
        super().__init__()
        self.before = nn.Linear(1, 1, dtype=torch.float64)
        self.reservoir = model
        self.after = nn.Linear(1, 1, dtype=torch.float64)
        self.named = named

    def forward(self, data):
        prediction = self.reservoir(self.before(data))
        if self.named:
            prediction = prediction["position"] + prediction["velocity"]
        return self.after(prediction)


def _build(named, preset=False):
    x = rd.reservoir_input(1, dtype=torch.float64)
    h = rd.ESNLayer(6, feedback_size=1, seed=19, dtype=torch.float64)(x)
    y = rd.SVDReadoutLayer(6, 1, alpha=1e-5).double()(h)
    outputs = {"position": y}
    if named:
        outputs["velocity"] = rd.SVDReadoutLayer(6, 1, alpha=1e-5).double()(h)
    model = (
        rd.ESN(6, input_size=1, washout=2, readout="svd", seed=19, dtype=torch.float64)
        if preset
        else rd.Model(inputs={"signal": x}, outputs=outputs, washout=2)
    )
    data = torch.sin(torch.arange(35, dtype=torch.float64) / 4)[:, None]
    if named:
        model.fit(data, {"position": data + 2, "velocity": 3 * data - 1})
    else:
        model.fit(data)
    return _Pipeline(model, named).eval(), data


@pytest.mark.parametrize("named,preset", [(False, False), (True, False), (False, True)])
def test_fx_leaf_capture_preserves_ownership_state_values_and_gradients(named, preset):
    pipeline, data = _build(named, preset)
    model = pipeline.reservoir
    model(data[:3])
    stored = {name: value.clone() for name, value in model.get_state().items()}
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    try:
        captured = fx.GraphModule(pipeline, _ReservoirTracer().trace(pipeline))
        assert calls == []  # tracing must not warm up or otherwise execute the RC
        for name, value in stored.items():
            torch.testing.assert_close(model.get_state()[name], value, rtol=0, atol=0)
        assert captured.reservoir is model
        assert [id(p) for p in captured.parameters()] == [id(p) for p in pipeline.parameters()]
        assert set(captured.state_dict()) == set(pipeline.state_dict())
        assert (
            sum(
                node.op == "call_module" and node.target == "reservoir"
                for node in captured.graph.nodes
            )
            == 1
        )

        first = data[-9:].unsqueeze(0).detach().requires_grad_()
        second = first.detach().clone().requires_grad_()
        model.reset_state()
        expected = pipeline(first)
        expected_state = copy.deepcopy(model.get_state())
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (first, pipeline.before.weight)
        )
        model.reset_state()
        actual = captured(second)
        actual_grads = torch.autograd.grad(actual.square().sum(), (second, captured.before.weight))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for observed, reference in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
            assert torch.isfinite(observed).all() and observed.abs().sum() > 0
        for name, value in expected_state.items():
            torch.testing.assert_close(model.get_state()[name], value, rtol=0, atol=0)

        # The captured graph retains the documented carry-between-calls policy.
        model.reset_state()
        full = captured(data[-9:].unsqueeze(0))
        model.reset_state()
        chunked = torch.cat(
            [captured(data[-9:-5].unsqueeze(0)), captured(data[-5:].unsqueeze(0))],
            dim=1,
        )
        torch.testing.assert_close(chunked, full, rtol=1e-12, atol=1e-12)
        captured.float()
        assert pipeline.reservoir is captured.reservoir
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert all(b.dtype == torch.float32 for b in model.buffers() if b.is_floating_point())
    finally:
        handle.remove()
