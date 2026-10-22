"""Preset preprocessing registration, staging and optional trajectory boundaries."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

import resdag as rd


class Affine(nn.Module):
    supports_framewise = True

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.register_buffer("center", torch.zeros(1, dtype=torch.float64))

    @torch.no_grad()
    def fit(self, data):
        self.center.copy_(data.mean((0, 1)))
        return self

    def forward(self, data):
        return (data - self.center) * self.gain

    def inverse(self, data):
        return data / self.gain + self.center


def _preset(**options):
    return rd.ESN(8, washout=1, seed=3, dtype=torch.float64, **options)


def _data():
    return torch.linspace(-2, 3, 20, dtype=torch.float64)[:, None]


@pytest.mark.parametrize("eager", [False, True])
def test_original_transform_parameters_survive_build_and_fit_once(eager):
    incoming, outgoing = Affine(), Affine()
    model = _preset(
        input_size=1 if eager else None,
        input_transforms={"input": incoming},
        output_transforms={"output": outgoing},
    )
    assert sum(parameter is incoming.gain for parameter in model.parameters()) == 1
    assert sum(parameter is outgoing.gain for parameter in model.parameters()) == 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.fit(_data())
    assert model.workflow.input_scalers[0] is incoming
    assert model.workflow.output_scalers[0] is outgoing
    assert any(
        parameter is incoming.gain
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    torch.testing.assert_close(incoming.center, _data()[1:-1].mean(0))
    torch.testing.assert_close(outgoing.center, _data()[2:].mean(0))
    keys = tuple(model.state_dict())
    assert not any("pending" in key for key in keys)
    assert sum(key.endswith("input_scalers.0.gain") for key in keys) == 1
    value = _data()[:3].clone().requires_grad_()
    model.reset_state()
    model(value).sum().backward()
    assert value.grad is not None and incoming.gain.grad is not None
    assert outgoing.gain.grad is not None


def test_build_exposes_pending_originals_and_preserves_hooks_and_mode():
    custom = Affine()
    model = _preset(input_transforms={"input": custom})
    assert model.workflow is None
    assert model._pending_input_transforms["input"] is custom
    assert model.state_dict()["_pending_input_transforms.input.gain"] is not None
    model.eval()
    calls = []
    handle = custom.register_forward_hook(lambda module, args, result: calls.append(result.shape))
    identity = custom.gain
    model.build(1)
    assert model.workflow.input_scalers[0] is custom and custom.gain is identity
    assert not model.training and not model.workflow.training and not custom.training
    model.fit(_data())
    model(_data()[:2])
    assert calls
    handle.remove()


def test_failed_first_fit_preserves_unbuilt_original_registration_and_state():
    custom = Affine()
    model = _preset(input_transforms={"input": custom})
    before = copy.deepcopy(model.state_dict())
    bad = _data()
    bad[4] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.fit(bad)
    assert model.workflow is None and model.input_size is None
    assert model._pending_input_transforms["input"] is custom
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        else:
            assert model.state_dict()[key] == value


def test_fitted_checkpoint_loads_into_lazy_originals_without_replacing_parameters():
    source = _preset(input_transforms={"input": Affine()}, output_transforms={"output": Affine()})
    source.fit(_data())
    incoming, outgoing = Affine(), Affine()
    target = _preset(input_transforms={"input": incoming}, output_transforms={"output": outgoing})
    incoming_parameter, outgoing_parameter = incoming.gain, outgoing.gain
    target.load_state_dict(copy.deepcopy(source.state_dict()))
    assert target.is_built and target.is_fitted
    assert target.workflow.input_scalers[0] is incoming
    assert incoming.gain is incoming_parameter and outgoing.gain is outgoing_parameter
    torch.testing.assert_close(
        target.forecast(horizon=3), source.forecast(horizon=3), rtol=0, atol=0
    )


def test_preset_rejects_wrong_port_or_aliased_transforms_before_build():
    with pytest.raises(ValueError, match="declared port names"):
        _preset(input_transforms={"state": Affine()})
    with pytest.raises(TypeError, match="Module"):
        _preset(input_transforms={"input": object()})
    shared = Affine()
    with pytest.raises(ValueError, match="distinct module"):
        _preset(input_transforms={"input": shared}, output_transforms={"output": shared})


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_inference_first_fit_converts_originals_to_ordinary_autograd_state(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    custom = Affine().float()
    parameter = custom.gain
    model = rd.ESN(
        8,
        washout=1,
        seed=3,
        input_transforms={"input": custom},
    )
    dtype = torch.float64 if device == "cpu" else torch.float32
    data = _data().to(device=device, dtype=dtype)
    with torch.inference_mode():
        model.fit(data)
    assert model.workflow.input_scalers[0] is custom and custom.gain is parameter
    assert custom.center.device.type == device and custom.center.dtype == dtype
    assert not custom.center.is_inference() and not custom.gain.is_inference()
    source = data[:3].clone().requires_grad_()
    model(source).sum().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())


def test_lazy_fit_revalidates_original_state_before_candidate_deepcopy():
    custom = Affine()
    model = _preset(input_transforms={"input": custom})
    custom.center = torch.zeros(1).expand(2)
    with pytest.raises(ValueError, match="overlap|writable"):
        model.fit(torch.ones(8, 2).double())
    assert model.workflow is None and model._pending_input_transforms["input"] is custom


def test_lazy_named_trajectory_infers_width_and_preserves_custom_ownership():
    ts = pytest.importorskip("tsdynamics")
    times = 0.1 * np.arange(20)
    observed = ts.Trajectory(times, np.sin(times)[:, None], variables=("position",))
    custom = Affine()
    model = _preset(input_transforms={"input": custom})
    model.fit({"input": observed})
    assert model.input_size == 1 and model.workflow.input_scalers[0] is custom
    future = model.forecast(horizon=2)
    assert isinstance(future, ts.Trajectory)
    np.testing.assert_allclose(future.t, [2, 2.1], rtol=0, atol=1e-12)
