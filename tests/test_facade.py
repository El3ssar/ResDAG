"""The ESN preset is a standard Torch module sharing Model's entire workflow."""

import copy
import io

import numpy as np
import pytest
import torch
from torch import nn

import resdag as rd
from resdag.layers import ReadoutLayer
from resdag.models import classic_esn


def _series(width=2, length=70):
    t = torch.linspace(0, 6, length, dtype=torch.float64)
    return torch.stack([torch.sin(t + index / 3) for index in range(width)], -1)


def _esn(**options):
    return rd.ESN(9, washout=3, seed=14, **options)


def _equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal(actual[key], expected[key])
    else:
        assert actual == expected


def test_export_and_single_registered_workflow_with_raw_graph_access():
    from resdag.facade import ESN

    assert rd.ESN is ESN and "ESN" in rd.__all__
    model = _esn(input_size=2)
    assert isinstance(model, nn.Module) and isinstance(model.workflow, rd.Model)
    assert model.graph is model.workflow.graph
    assert set(model._modules) == {"workflow"}
    assert "model" not in model._modules and not hasattr(model, "model")
    assert all(key.startswith("workflow.") or key == "_extra_state" for key in model.state_dict())
    assert model.is_built and not model.is_fitted and model.input_size == 2


def test_lazy_fit_adopts_input_and_matches_independent_model_preset():
    series = _series()
    model = _esn()
    assert not model.is_built and model.input_size is None and list(model.parameters()) == []
    assert model.fit(series) is model
    reference = rd.Model(graph=classic_esn(9, 2, 2, seed=14, dtype=torch.float64), washout=3).fit(
        series
    )
    _equal(model.graph.state_dict(), reference.graph.state_dict())
    assert model.is_fitted and next(model.parameters()).dtype == torch.float64
    torch.testing.assert_close(
        model.forecast(horizon=8), reference.forecast(horizon=8), rtol=0, atol=0
    )
    model.reset_state()
    reference.reset_state()
    torch.testing.assert_close(model(series[:6]), reference(series[:6]), rtol=0, atol=0)


def test_eager_build_is_idempotent_and_optimizer_keeps_parameters_after_fit():
    model = _esn(input_size=2, dtype=torch.float64, readout_kwargs={"trainable": True})
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.eval()
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    assert model.build(2) is model
    assert model.fit(_series()) is model
    assert all(before is after for before, after in zip(parameters, model.parameters()))
    assert all(module.training is False for module in model.modules())
    data = _series()[:6].requires_grad_()
    model(data).square().sum().backward()
    optimizer.step()
    handle.remove()
    assert calls == [1] and data.grad is not None
    assert any(parameter.grad is not None for parameter in parameters)
    with pytest.raises(ValueError, match="input_size is fixed"):
        model.build(3)
    with pytest.raises(ValueError, match="use .to"):
        model.build(2, dtype=torch.float32)


def test_explicit_build_then_module_conversion_controls_lazy_placement():
    model = _esn().eval().double()
    assert model.build(2) is model
    assert next(model.parameters()).dtype == torch.float64
    assert not model.training and not model.workflow.training
    model.fit(_series())
    assert model.forecast(horizon=3).dtype == torch.float64
    pinned = _esn(dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        pinned.fit(_series())
    assert not pinned.is_built
    converted = _esn().float()
    with pytest.raises(ValueError, match="dtype"):
        converted.fit(_series())
    assert not converted.is_built
    converted.fit(_series().float())
    with pytest.raises(ValueError, match="dtype"):
        converted.fit(_series())


@pytest.mark.parametrize("batch", [None, 1, 2])
def test_prediction_and_forecast_preserve_exact_model_layout(batch):
    data = _series()
    if batch is not None:
        data = data.unsqueeze(0).expand(batch, -1, -1).clone()
    model = _esn().fit(data)
    prefix = () if batch is None else (batch,)
    assert model(data).shape == (*prefix, 70, 2)
    assert model.forecast(horizon=5).shape == (*prefix, 5, 2)
    assert model.forecast(horizon=5, return_warmup=True).shape == (*prefix, 8, 2)
    with pytest.raises(TypeError):
        model.forecast(5)


def test_state_carry_explicit_state_and_chunk_gradients():
    model = _esn().fit(_series())
    for layer in model.graph.modules():
        if isinstance(layer, rd.ESNLayer):
            layer.detach_state_between_calls = False
    data = _series()[:8].requires_grad_()
    model.reset_state()
    whole = model(data)
    model.reset_state()
    first = model(data[:3])
    state = model.get_state()
    last = model(data[3:])
    torch.testing.assert_close(torch.cat((first, last)), whole, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(whole.square().sum(), data)[0]
    b = torch.autograd.grad(torch.cat((first, last)).square().sum(), data)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    model.set_state(state)
    torch.testing.assert_close(model(data[3:]), last, rtol=0, atol=0)
    before = model.get_state()
    explicit, final = model.forward_stateless(data)
    p, middle = model.forward_stateless(data[:3])
    q, chunk_final = model.forward_stateless(data[3:], states=middle)
    torch.testing.assert_close(torch.cat((p, q)), explicit, rtol=1e-12, atol=1e-12)
    # Different projection batch shapes can change BLAS rounding across chunks.
    # Use the same float64 tolerance as the output/gradient comparisons above;
    # the stored-state immutability assertion below remains bit-exact.
    assert final.keys() == chunk_final.keys()
    for name in final:
        torch.testing.assert_close(final[name], chunk_final[name], rtol=1e-12, atol=1e-12)
    _equal(model.get_state(), before)


def test_failed_first_fit_stays_unbuilt_and_failed_refit_preserves_module():
    model = _esn().eval()
    handle = model.register_forward_hook(lambda *_: None)
    for data in (_series(length=3), torch.full((8, 2), float("nan"), dtype=torch.float64)):
        with pytest.raises(ValueError):
            model.fit(data)
        assert not model.is_built and list(model.parameters()) == [] and not model.training
    model.fit(_series())
    workflow = model.workflow
    before = copy.deepcopy(model.state_dict())
    model(_series()[:3])
    carried = model.get_state()
    with pytest.raises(ValueError):
        model.fit(_series(width=3))
    assert model.workflow is workflow
    _equal(model.state_dict(), before)
    _equal(model.get_state(), carried)
    handle.remove()


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_checkpoint_loads_matching_unbuilt_preset_and_keeps_predictions(assign, nested):
    source = _esn().fit(_series())
    destination = _esn().double()
    if nested:
        source, destination = nn.Sequential(source), nn.Sequential(destination)
    stream = io.BytesIO()
    torch.save(source.state_dict(), stream)
    stream.seek(0)
    destination.load_state_dict(torch.load(stream, weights_only=True), assign=assign)
    original = source[0] if nested else source
    restored = destination[0] if nested else destination
    assert restored.is_fitted and restored.input_size == 2
    torch.testing.assert_close(
        restored.forecast(horizon=4), original.forecast(horizon=4), rtol=0, atol=0
    )
    clone = copy.deepcopy(restored)
    torch.testing.assert_close(
        clone.forecast(horizon=4), restored.forecast(horizon=4), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "options",
    [{"spectral_radius": 0.8}, {"alpha": 1e-3}, {"activation": "identity"}, {"readout": "svd"}],
)
def test_checkpoint_rejects_different_preset_settings_before_weight_copy(options):
    source = _esn().fit(_series())
    destination = _esn(**options).double()
    before = copy.deepcopy(destination.state_dict())
    with pytest.raises(RuntimeError, match="architecture differs"):
        destination.load_state_dict(source.state_dict())
    assert not destination.is_built
    _equal(destination.state_dict(), before)


def test_checkpoint_unbuilt_and_width_validation():
    source, destination = _esn(), _esn()
    destination.load_state_dict(source.state_dict())
    assert not destination.is_built
    fitted = _esn().fit(_series())
    wrong = _esn(input_size=3, dtype=torch.float64)
    before = copy.deepcopy(wrong.state_dict())
    with pytest.raises(RuntimeError, match="input_size differs"):
        wrong.load_state_dict(fitted.state_dict())
    _equal(wrong.state_dict(), before)


def test_solver_and_custom_initializer_options_remain_accessible():
    calls = []

    def initializer(tensor):
        calls.append(tuple(tensor.shape))
        return torch.nn.init.constant_(tensor, 0.2)

    model = _esn(
        readout="ridge",
        readout_kwargs={"solver": "solve", "gram_dtype": torch.float64},
        feedback_initializer=initializer,
        leak_rate=0.4,
        topology="simple_cycle",
        readout_name="custom_internal_head",
    ).fit(_series())
    reservoir = next(layer for layer in model.graph.modules() if isinstance(layer, rd.ESNLayer))
    readout = next(layer for layer in model.graph.modules() if isinstance(layer, ReadoutLayer))
    assert reservoir.leak_rate == 0.4 and readout.solver == "solve"
    assert readout.name == "custom_internal_head"
    assert calls and torch.all(reservoir.cell.weight_feedback == 0.2)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_input_width_must_be_positive_integer(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        _esn(input_size=invalid)


def test_no_numpy_mirroring_or_integer_coercion_and_zero_washout_is_supported():
    for data in (_series().numpy(), torch.ones(20, 2, dtype=torch.int64)):
        with pytest.raises(TypeError, match="floating"):
            _esn().fit(data)
    model = rd.ESN(5, washout=0, seed=2).fit(_series())
    assert isinstance(model.forecast(horizon=2), torch.Tensor)
    assert model.forecast(horizon=2).shape == (2, 2)
    assert not hasattr(model, "feature_size")


def test_fitting_in_inference_mode_keeps_future_upstream_autograd():
    model = _esn()
    with torch.inference_mode():
        model.fit(_series())
    data = _series()[:4].requires_grad_()
    model(data).sum().backward()
    assert data.grad is not None and bool((data.grad != 0).any())
    assert all(not tensor.is_inference() for tensor in model.buffers())


def test_forecasting_periodic_signal_keeps_meaningful_accuracy():
    t = torch.linspace(0, 40, 1000, dtype=torch.float64)
    data = torch.stack((t.sin(), t.cos()), -1)
    model = rd.ESN(120, washout=50, alpha=1e-7, seed=4).fit(data[:800])
    prediction = model.forecast(horizon=200)
    assert torch.mean((prediction - data[800:]).square()).sqrt() < 0.1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_native_mps_conversion_before_lazy_fit_and_recurrent_gradients():
    model = (
        _esn(readout_kwargs={"solver": "solve", "use_float64": False, "gram_dtype": torch.float32})
        .float()
        .to("mps")
    )
    data = _series().float().to("mps")
    model.fit(data)
    assert next(model.parameters()).device.type == "mps"
    assert model.forecast(horizon=3).device.type == "mps"
    sample = data[:4].detach().requires_grad_()
    model.reset_state()
    model(sample).sum().backward()
    assert sample.grad is not None and sample.grad.device.type == "mps"


def test_trajectory_fit_has_same_physical_time_boundary_as_model():
    ts = pytest.importorskip("tsdynamics")
    series = _series()
    trajectory = ts.Trajectory(np.arange(len(series)) * 0.25, series.numpy(), variables=("x", "y"))
    model = _esn().fit(trajectory)
    forecast = model.forecast(horizon=3)
    assert isinstance(forecast, ts.Trajectory) and forecast.variables == ("x", "y")
    np.testing.assert_allclose(forecast.t, [17.5, 17.75, 18.0])
    tensor = model.forecast(horizon=3, return_type="tensor")
    torch.testing.assert_close(tensor, torch.from_numpy(forecast.y))


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_seeded_construction_restores_cpu_and_available_mps_rng_and_all_weights(eager, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    dtype = torch.float32 if device == "mps" else torch.float64
    before_cpu = torch.get_rng_state().clone()
    before_mps = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    a = _esn(input_size=2 if eager else None, dtype=dtype, device=device, preprocessing="off")
    if not eager:
        a.build(2)
    _equal(torch.get_rng_state(), before_cpu)
    if before_mps is not None:
        _equal(torch.mps.get_rng_state(), before_mps)
    torch.rand(17)
    if before_mps is not None:
        torch.rand(17, device="mps")
    surrounding_cpu = torch.get_rng_state().clone()
    surrounding_mps = torch.mps.get_rng_state().clone() if before_mps is not None else None
    b = _esn(input_size=2, dtype=dtype, device=device, preprocessing="off")
    _equal(a.state_dict(), b.state_dict())
    _equal(torch.get_rng_state(), surrounding_cpu)
    if surrounding_mps is not None:
        _equal(torch.mps.get_rng_state(), surrounding_mps)


def test_seed_isolation_restores_rng_when_preset_build_raises():
    model = _esn(topology="not_a_registered_topology")
    before_cpu = torch.get_rng_state().clone()
    before_mps = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    with pytest.raises(ValueError, match="Unknown topology"):
        model.build(2)
    assert not model.is_built
    _equal(torch.get_rng_state(), before_cpu)
    if before_mps is not None:
        _equal(torch.mps.get_rng_state(), before_mps)


@pytest.mark.parametrize("guide", ["reference/facade.md", "project/esn-migration.md"])
def test_new_esn_documentation_examples_execute(guide):
    import re
    from pathlib import Path

    path = Path(__file__).parents[1] / "docs" / guide
    namespace = {"__name__": "__documented_example__"}
    for source in re.findall(r"^```python\n(.*?)^```", path.read_text(), flags=re.M | re.S):
        exec(compile(source, str(path), "exec"), namespace)


def test_prediction_lead_feedback_and_history_execution_delegate_to_model():
    from resdag.core._feedback import Feedback

    route = Feedback(output="output", input="input", delay=3)
    series = torch.arange(50, dtype=torch.float64)[:, None]
    model = rd.ESN(
        4,
        washout=4,
        activation="identity",
        alpha=0,
        readout="svd",
        seed=7,
        prediction_lead=3,
        feedback=(route,),
    ).fit(series)
    assert model.prediction_leads == {"output": 3} and model.feedback == (route,)
    expected = torch.arange(50, 56, dtype=torch.float64)[:, None]
    fast = model.forecast(horizon=6)
    history = model.forecast(horizon=6, execution="history")
    torch.testing.assert_close(fast, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(history, expected, rtol=1e-10, atol=1e-10)
    restored = rd.ESN(4, washout=4, activation="identity", alpha=0, readout="svd", seed=7).double()
    restored.load_state_dict(model.state_dict())
    assert restored.prediction_leads == {"output": 3} and restored.feedback == (route,)
    torch.testing.assert_close(restored.forecast(horizon=6), expected, rtol=1e-10, atol=1e-10)
