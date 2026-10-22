"""Independent reversible preprocessing, ownership and persistence contracts."""

import copy
import io

import pytest
import torch
from torch import nn

import resdag as rd
from resdag.core._workflow_transforms import stage_transform, validate_transform_ownership


class Affine(nn.Module):
    supports_framewise = True

    def __init__(self, width=1, *, trainable=False):
        super().__init__()
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.gain = nn.Parameter(torch.ones(width), requires_grad=trainable)

    @torch.no_grad()
    def fit(self, data):
        self.center.copy_(data.mean((0, 1)))
        self.scale.copy_(data.std((0, 1), unbiased=False).clamp_min(0.01))
        return self

    def forward(self, data):
        return (data - self.center) / self.scale * self.gain

    def inverse(self, data):
        return data / self.gain * self.scale + self.center


class Asinh(nn.Module):
    supports_framewise = True

    def fit(self, data):
        return self

    def forward(self, data):
        return data.asinh()

    def inverse(self, data):
        return data.sinh()


class IntegerShift(nn.Module):
    supports_framewise = True

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.zeros(1, dtype=torch.int64))

    @torch.no_grad()
    def fit(self, data):
        self.shift.copy_(data.mean((0, 1)).round().to(torch.int64))
        return self

    def forward(self, data):
        return data - self.shift

    def inverse(self, data):
        return data + self.shift


def _model(
    *,
    input_transform=None,
    output_transform=None,
    preprocessing="off",
    washout=1,
    dtype=torch.float64,
):
    x = rd.reservoir_input(1, dtype=dtype)
    head = rd.RidgeReadoutLayer(1, 1, alpha=0, trainable=True).to(dtype=dtype)
    return rd.Model(
        inputs={"state": x},
        outputs={"prediction": head(x)},
        washout=washout,
        preprocessing=preprocessing,
        input_transforms=None if input_transform is None else {"state": input_transform},
        output_transforms=None if output_transform is None else {"prediction": output_transform},
    )


def _data(dtype=torch.float64):
    return torch.arange(1, 13, dtype=dtype)[:, None]


def _same(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, right[key], rtol=0, atol=0)
        else:
            assert value == right[key]


def test_standardize_overrides_global_off_fit_their_own_training_rows_and_preserve_instances():
    incoming, outgoing = rd.Standardize(1), rd.Standardize(1)
    model = _model(input_transform=incoming, output_transform=outgoing)
    with pytest.raises(RuntimeError, match="not fitted"):
        model(_data())
    model.fit(_data())
    assert model.input_scalers[0] is incoming and model.output_scalers[0] is outgoing
    torch.testing.assert_close(incoming.mean, _data()[1:-1].mean(0))
    torch.testing.assert_close(outgoing.mean, _data()[2:].mean(0))
    torch.testing.assert_close(model.forecast(horizon=4), torch.arange(13, 17).double()[:, None])


def test_unspecified_implicit_output_uses_original_input_rows_even_with_custom_input():
    custom = Affine()
    model = _model(input_transform=custom, preprocessing="standard").fit(_data())
    torch.testing.assert_close(custom.center, _data()[1:-1].mean(0))
    expected = _data()[1:-1].mean(0).reshape(1, 1, 1)
    torch.testing.assert_close(model.output_scalers[0]._stat0, expected)
    torch.testing.assert_close(model.forecast(horizon=3), torch.arange(13, 16).double()[:, None])


def test_identity_disables_one_port_fallback_without_custom_methods():
    identity = nn.Identity()
    model = _model(input_transform=identity, preprocessing="standard").fit(_data())
    assert model.input_scalers[0] is identity
    torch.testing.assert_close(model.forecast(horizon=3), torch.arange(13, 16).double()[:, None])


def test_nonlinear_output_and_affine_input_follow_physical_feedback_and_analytic_gradient():
    transform = Affine(trainable=True)
    model = _model(input_transform=transform, output_transform=Asinh(), washout=0)
    training = torch.linspace(-2, 2, 30, dtype=torch.float64)[:, None]
    model.fit(training, targets=(0.3 * training + 0.1).sinh())
    history = training[-4:]
    current = history[-1:]
    expected = []
    for _ in range(4):
        current = (0.3 * current + 0.1).sinh()
        expected.append(current)
    torch.testing.assert_close(
        model.forecast(history, horizon=4), torch.cat(expected), rtol=1e-12, atol=1e-12
    )
    source = history.detach().clone().requires_grad_()
    prediction = model(source)
    prediction.sum().backward()
    torch.testing.assert_close(
        source.grad, 0.3 * (0.3 * source.detach() + 0.1).cosh(), rtol=1e-12, atol=1e-12
    )
    assert transform.gain.grad is not None and bool((transform.gain.grad != 0).any())


def test_refitting_preserves_user_parameters_optimizer_references_and_transform_containers():
    incoming, outgoing = Affine(trainable=True), Affine(trainable=True)
    model = _model(input_transform=incoming, output_transform=outgoing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    parameters = tuple(model.parameters())
    containers = model.input_scalers, model.output_scalers
    model.fit(_data()).fit(_data() + 10)
    assert model.input_scalers is containers[0] and model.output_scalers is containers[1]
    assert tuple(id(value) for value in model.parameters()) == tuple(map(id, parameters))
    assert tuple(map(id, optimizer.param_groups[0]["params"])) == tuple(map(id, parameters))
    source = (_data()[:3] + 10).requires_grad_()
    optimizer.zero_grad()
    model(source).square().mean().backward()
    previous = incoming.gain.detach().clone()
    optimizer.step()
    assert not torch.equal(previous, incoming.gain)


class ModeAffine(Affine):
    supports_framewise = True

    def forward(self, data):
        return super().forward(data) * (2 if self.training else 1)

    def inverse(self, data):
        return super().inverse(data / (2 if self.training else 1))


def test_fit_and_forecast_use_transform_eval_and_restore_mixed_modes_including_failure():
    incoming = ModeAffine()
    outgoing = ModeAffine()
    model = _model(input_transform=incoming, output_transform=outgoing).fit(_data())
    model.train()
    outgoing.eval()
    modes = tuple(child.training for child in model.modules())
    torch.testing.assert_close(model.forecast(horizon=3), torch.arange(13, 16).double()[:, None])
    assert tuple(child.training for child in model.modules()) == modes

    def fail(*args):
        raise RuntimeError("deliberate forecast failure")

    handle = model.graph.register_forward_pre_hook(fail)
    # Custom transforms use explicit planner operations, so attach to a head.
    handle.remove()
    head = next(module for module in model.graph.modules() if isinstance(module, rd.ReadoutLayer))
    handle = head.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="deliberate forecast failure"):
            model.forecast(horizon=3)
    finally:
        handle.remove()
    assert tuple(child.training for child in model.modules()) == modes


class HiddenFlag(Affine):
    supports_framewise = True

    def __init__(self):
        super().__init__()
        self.ready = False

    def fit(self, data):
        super().fit(data)
        self.ready = True
        return self

    def forward(self, data):
        if not self.ready:
            raise RuntimeError("plain Python fitted flag")
        return super().forward(data)


class Resizing(Affine):
    def fit(self, data):
        self.center = data.new_zeros(2)
        return self


class Cumulative(Affine):
    supports_framewise = True

    def forward(self, data):
        return data.cumsum(1)

    def inverse(self, data):
        return torch.cat((data[:, :1], data[:, 1:] - data[:, :-1]), 1)


@pytest.mark.parametrize(
    "transform,error",
    [
        (HiddenFlag(), "registered tensor state"),
        (Resizing(), "fixed shapes"),
        (Cumulative(), "framewise"),
    ],
)
def test_invalid_learned_state_or_temporal_behavior_fails_before_live_fit_commit(transform, error):
    model = _model(input_transform=transform)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match=error):
        model.fit(_data())
    _same(model.state_dict(), before)
    assert not model.is_fitted


@pytest.mark.parametrize("alias", ["module", "parameter", "storage", "internal", "graph"])
def test_ambiguous_transform_state_aliases_are_rejected_at_construction(alias):
    incoming = Affine()
    if alias == "module":
        outgoing = incoming
    elif alias == "parameter":
        outgoing = Affine()
        outgoing.gain = incoming.gain
    elif alias == "storage":
        outgoing = Affine()
        outgoing.gain = nn.Parameter(incoming.gain.detach().view_as(incoming.gain))
        with pytest.raises(ValueError, match="independent storage"):
            _model(input_transform=incoming, output_transform=outgoing)
        return
    elif alias == "internal":
        incoming.alias = incoming.gain
        with pytest.raises(ValueError, match="unique registrations"):
            _model(input_transform=incoming)
        return
    else:
        source = rd.reservoir_input(1)
        head = rd.RidgeReadoutLayer(1, 1)
        incoming.gain = head.weight
        with pytest.raises(ValueError, match="independent registered state"):
            rd.Model(
                inputs={"x": source}, outputs={"y": head(source)}, input_transforms={"x": incoming}
            )
        return
    with pytest.raises(ValueError, match="independent registered state"):
        _model(input_transform=incoming, output_transform=outgoing)


@pytest.mark.parametrize("layout", ["expanded", "overlapping"])
def test_internally_overlapping_transform_state_rejects_before_live_fit_commit(layout):
    incoming = Affine()
    tensor = (
        torch.ones(1).expand(2)
        if layout == "expanded"
        else torch.ones(3).as_strided((2, 2), (1, 1))
    )
    incoming.extra = nn.Parameter(tensor.clone())
    model = _model(input_transform=incoming)
    # Changing state after construction must also fail before the graph changes.
    incoming.extra = nn.Parameter(tensor)
    before = copy.deepcopy(model.graph.state_dict())
    with pytest.raises(ValueError, match="nonoverlapping tensor storage"):
        model.fit(_data())
    _same(model.graph.state_dict(), before)
    assert not model.is_fitted


@pytest.mark.parametrize("layout", ["transpose", "slice"])
def test_independent_writable_transposed_and_sliced_transform_state_is_supported(layout):
    incoming = Affine()
    tensor = torch.ones(2, 3).T if layout == "transpose" else torch.ones(6)[::2]
    incoming.gain = nn.Parameter(tensor)
    validate_transform_ownership(nn.Identity(), (incoming,))


class WrongInverse(nn.Module):
    supports_framewise = True

    def fit(self, data):
        return self

    def forward(self, data):
        return data

    def inverse(self, data):
        return torch.zeros_like(data)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_max_finite_values_do_not_overflow_reversibility_tolerance(dtype):
    data = torch.full((1, 4, 1), torch.finfo(dtype).max, dtype=dtype)
    identity, _ = stage_transform(nn.Identity(), data)
    torch.testing.assert_close(identity(data), data, rtol=0, atol=0)
    with pytest.raises(ValueError, match="reversible"):
        stage_transform(WrongInverse(), data)


class WrongSmallFeatureInverse(WrongInverse):
    def inverse(self, data):
        result = data.clone()
        result[..., 1] = 0
        return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_large_feature_does_not_hide_inverse_error_in_smaller_feature(dtype):
    data = torch.tensor([torch.finfo(dtype).max, 1], dtype=dtype).expand(1, 4, 2)
    identity, _ = stage_transform(nn.Identity(), data)
    torch.testing.assert_close(identity(data), data, rtol=0, atol=0)
    with pytest.raises(ValueError, match="reversible"):
        stage_transform(WrongSmallFeatureInverse(), data)


class InPlace(nn.Module):
    supports_framewise = True

    def __init__(self, method):
        super().__init__()
        self.method = method

    def fit(self, data):
        return self

    def forward(self, data):
        return data.add_(1) if self.method == "forward" else data

    def inverse(self, data):
        return data.sub_(1)


@pytest.mark.parametrize("method", ["forward", "inverse"])
def test_inplace_preprocessing_rejects_before_changing_training_or_live_graph(method):
    model = _model(input_transform=InPlace(method))
    data = _data()
    original = data.clone()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="must not modify input tensors"):
        model.fit(data)
    torch.testing.assert_close(data, original, rtol=0, atol=0)
    _same(model.state_dict(), before)


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_custom_state_checkpoint_roundtrip_assign_and_inference_load(assign, nested):
    source_model = _model(
        input_transform=Affine(trainable=True), output_transform=Asinh(), washout=0
    )
    values = torch.linspace(-2, 2, 30, dtype=torch.float64)[:, None]
    source_model.fit(values, targets=(0.3 * values + 0.1).sinh())
    destination_model = _model(
        input_transform=Affine(trainable=True), output_transform=Asinh(), washout=0
    )
    original_module = destination_model.input_scalers[0]
    original_gain = original_module.gain
    source = nn.Sequential(source_model) if nested else source_model
    destination = nn.Sequential(destination_model) if nested else destination_model
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True, map_location="cpu")
    with torch.inference_mode():
        destination.load_state_dict(state, assign=assign)
    assert destination_model.input_scalers[0] is original_module
    assert (destination_model.input_scalers[0].gain is original_gain) == (not assign)
    torch.testing.assert_close(
        destination_model.forecast(horizon=3), source_model.forecast(horizon=3), rtol=0, atol=0
    )
    data = values[:3].detach().clone().requires_grad_()
    destination_model(data).sum().backward()
    assert data.grad is not None and bool(torch.isfinite(data.grad).all())


def test_integer_only_transform_state_loads_at_destination_floating_precision():
    source = _model(input_transform=IntegerShift()).fit(_data())
    target = _model(input_transform=IntegerShift(), dtype=torch.float32)
    target.load_state_dict(copy.deepcopy(source.state_dict()))
    assert target.input_scalers[0].shift.dtype == torch.int64
    assert target._input_transform_sample_0.dtype == torch.float32
    result = target.forecast(horizon=3)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.arange(13, 16).float()[:, None])
    data = _data(torch.float32)[:3].requires_grad_()
    target(data).sum().backward()
    torch.testing.assert_close(data.grad, torch.ones_like(data))


@pytest.mark.parametrize("corruption", ["class", "shape", "missing", "scale", "sample"])
def test_checkpoint_transform_preflight_is_before_graph_or_fitted_state_mutation(corruption):
    source = _model(input_transform=Affine()).fit(_data())
    destination = _model(input_transform=Affine()).fit(_data() + 10)
    state = copy.deepcopy(source.state_dict())
    if corruption == "class":
        state["_extra_state"]["transforms"][0][0]["modules"] = (("", "different.Transform"),)
    elif corruption == "shape":
        state["input_scalers.0.center"] = torch.zeros(2, dtype=torch.float64)
    elif corruption == "missing":
        del state["input_scalers.0.gain"]
    elif corruption == "scale":
        state["input_scalers.0.scale"].zero_()
    else:
        state["_input_transform_sample_0"].fill_(float("nan"))
    before = copy.deepcopy(destination.state_dict())
    with pytest.raises(RuntimeError, match="transform"):
        destination.load_state_dict(state)
    _same(destination.state_dict(), before)


def test_ordinary_custom_modules_fit_inside_inference_mode_with_later_autograd():
    model = _model(input_transform=Affine(trainable=True), output_transform=rd.Standardize(1))
    with torch.inference_mode():
        model.fit(_data())
    assert all(not value.is_inference() for value in model.buffers())
    source = _data()[:3].requires_grad_()
    model(source).sum().backward()
    assert source.grad is not None and model.input_scalers[0].gain.grad is not None


def test_inference_created_parameters_reject_explicitly_without_silent_data_repair():
    with torch.inference_mode():
        transform = Affine(trainable=True)
    with pytest.raises(ValueError, match="Parameters must be constructed outside inference_mode"):
        _model(input_transform=transform)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS unavailable")
def test_custom_transforms_native_mps_forecast_checkpoint_and_gradient_stay_device_local(
    monkeypatch,
):
    model = _model(
        input_transform=Affine(trainable=True),
        output_transform=Asinh(),
        washout=0,
        dtype=torch.float32,
    ).to("mps")
    head = next(
        module for module in model.graph.modules() if isinstance(module, rd.RidgeReadoutLayer)
    )
    head.use_float64 = False
    head.gram_dtype = torch.float32
    head.solver = "solve"
    values = torch.linspace(-2, 2, 30, device="mps")[:, None]
    model.fit(values, targets=(0.3 * values + 0.1).sinh())

    def forbidden(*args, **kwargs):
        raise AssertionError("Transform inference must remain on the native device")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "cpu", forbidden)
        context.setattr(torch.Tensor, "numpy", forbidden)
        prediction = model.forecast(horizon=3)
        source = values[:3].detach().clone().requires_grad_()
        model(source).sum().backward()
    assert prediction.device.type == "mps" and source.grad.device.type == "mps"
    cpu = copy.deepcopy(model).cpu()
    torch.testing.assert_close(prediction.cpu(), cpu.forecast(horizon=3), rtol=1e-4, atol=1e-4)
