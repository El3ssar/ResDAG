"""Native MPS correctness for common parallel and coupled workflows."""

import copy
import warnings

import pytest
import torch

import resdag as rd


def _graph(kind):
    options = dict(
        reservoir_size=8,
        readout="cg",
        readout_alpha=0.05,
        readout_kwargs={"use_float64": False, "max_iter": 500, "tol": 1e-4},
        seed=9,
        dtype=torch.float32,
    )
    if kind == "parallel":
        return rd.parallel_esn(input_dim=4, core_width=2, halo=1, **options)
    return rd.coupled_ensemble_esn(
        2, model_factory=rd.classic_esn, feedback_size=4, output_size=4, **options
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS unavailable")
@pytest.mark.parametrize("kind", ["parallel", "coupled"])
def test_native_mps_fit_forecast_transfer_and_explicit_state_gradients(kind):
    data = torch.stack([torch.sin(torch.arange(70) / 5 + i) for i in range(4)], dim=-1)
    # Under-converged iterative fits must not count as successful device evidence.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cpu = rd.Model(graph=_graph(kind), washout=5).fit(data)
        accelerated = copy.deepcopy(cpu).to("mps")
        expected = cpu.forecast(horizon=4)
        actual = accelerated.forecast(horizon=4)
        assert actual.device.type == "mps"
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)

        cpu.reset_state()
        accelerated.reset_state()
        source = data[-7:].detach().requires_grad_()
        native_source = source.detach().to("mps").requires_grad_()
        reference, expected_state = cpu.forward_stateless(source)
        observed, actual_state = accelerated.forward_stateless(native_source)
        reference.square().mean().backward()
        observed.square().mean().backward()
        torch.testing.assert_close(observed.cpu(), reference, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(native_source.grad.cpu(), source.grad, atol=1e-4, rtol=1e-4)
        assert torch.isfinite(native_source.grad).all() and native_source.grad.abs().sum() > 0
        assert accelerated.get_state() == cpu.get_state() == {}
        assert actual_state.keys() == expected_state.keys()
        for name in actual_state:
            torch.testing.assert_close(
                actual_state[name].cpu(), expected_state[name], atol=1e-4, rtol=1e-4
            )

        fitted = rd.Model(graph=_graph(kind).to("mps"), washout=5).fit(data.to("mps"))
        prediction = fitted.forecast(horizon=4)
        assert prediction.device.type == "mps" and torch.isfinite(prediction).all()
        assert fitted.input_scalers[0]._stat0.device.type == "mps"
        assert fitted.output_scalers[0]._stat0.device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS unavailable")
@pytest.mark.parametrize("kind", ["parallel", "coupled"])
@pytest.mark.parametrize("source_device,destination_device", [("cpu", "mps"), ("mps", "cpu")])
@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_cross_device_checkpoint_assignment_keeps_derived_buffers_with_actual_parameters(
    kind, source_device, destination_device, assign, nested
):
    data = torch.stack([torch.sin(torch.arange(70) / 5 + i) for i in range(4)], dim=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        source_model = rd.Model(graph=_graph(kind), washout=5).fit(data).to(source_device)
    expected = source_model.forecast(horizon=4)
    destination_model = rd.Model(graph=_graph(kind), washout=0).to(destination_device)
    source = torch.nn.Sequential(source_model) if nested else source_model
    destination = torch.nn.Sequential(destination_model) if nested else destination_model
    checkpoint = copy.deepcopy(source.state_dict())
    destination.load_state_dict(checkpoint, assign=assign)
    expected_device = source_device if assign else destination_device
    assert next(destination_model.graph.parameters()).device.type == expected_device
    if kind == "parallel":
        assert destination_model.graph.patch_index.device.type == expected_device
        assert destination_model.graph.patch_index.dtype == torch.long
    prediction = destination_model.forecast(horizon=4)
    assert prediction.device.type == expected_device
    torch.testing.assert_close(prediction.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)

    # Further module conversions retain integer indices and usable field routing.
    destination_model.to("cpu")
    if kind == "parallel":
        assert destination_model.graph.patch_index.device.type == "cpu"
        assert destination_model.graph.patch_index.dtype == torch.long
    torch.testing.assert_close(
        destination_model.forecast(horizon=4), expected.cpu(), rtol=1e-4, atol=1e-4
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS unavailable")
def test_nested_partial_load_repairs_index_when_unrelated_sibling_key_is_missing():
    data = torch.stack([torch.sin(torch.arange(70) / 5 + i) for i in range(4)], dim=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        source_model = rd.Model(graph=_graph("parallel"), washout=5).fit(data)
    expected = source_model.forecast(horizon=4)
    source = torch.nn.ModuleDict({"other": torch.nn.Linear(1, 1), "workflow": source_model})
    destination_model = rd.Model(graph=_graph("parallel"), washout=0).to("mps")
    destination = torch.nn.ModuleDict(
        {"other": torch.nn.Linear(1, 1), "workflow": destination_model}
    )
    checkpoint = copy.deepcopy(source.state_dict())
    del checkpoint["other.bias"]
    incompatible = destination.load_state_dict(checkpoint, strict=False, assign=True)
    assert incompatible.missing_keys == ["other.bias"]
    assert incompatible.unexpected_keys == []
    assert next(destination_model.graph.parameters()).device.type == "cpu"
    assert destination_model.graph.patch_index.device.type == "cpu"
    assert destination_model.graph.patch_index.dtype == torch.long
    torch.testing.assert_close(destination_model.forecast(horizon=4), expected, rtol=0, atol=0)
