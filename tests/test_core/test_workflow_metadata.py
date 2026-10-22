"""Workflow checkpoint metadata is portable without optional-library objects."""

import io

import numpy as np
import pytest
import torch

import resdag as rd


def _model(**options):
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.RidgeReadoutLayer(1, 1, alpha=0.0).double()(x)
    return rd.Model(
        inputs={np.str_("state"): x}, outputs={np.str_("next"): y}, washout=0, **options
    )


def test_numpy_string_ports_and_sampling_method_become_safe_checkpoint_primitives():
    model = _model(resample_method=np.str_("linear")).fit(
        torch.arange(8, dtype=torch.float64).reshape(-1, 1)
    )
    assert type(model.input_names[0]) is str
    assert type(model.output_names[0]) is str
    assert type(model.resample_method) is str
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    model.load_state_dict(checkpoint)
    torch.testing.assert_close(model.forecast(horizon=2), torch.tensor([[8.0], [9.0]]).double())


def test_live_producing_torch_module_never_becomes_a_registered_model_child():
    ts = pytest.importorskip("tsdynamics")
    system = torch.nn.Linear(1, 1)
    trajectory = ts.Trajectory(
        np.arange(8) * 0.25, np.arange(8.0)[:, None], system=system, variables=("position",)
    )
    model = _model().fit(trajectory)
    assert set(model._modules) == {"graph", "input_scalers", "output_scalers"}
    assert model.forecast(horizon=2).system is system
    restored = _model()
    restored.load_state_dict(model.state_dict())
    assert restored.forecast(horizon=2).system is None
    model.fit(torch.arange(8, dtype=torch.float64).reshape(-1, 1))
    assert set(model._modules) == {"graph", "input_scalers", "output_scalers"}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_native_mps_trajectory_fitting_and_tensor_output_keep_physical_clock():
    ts = pytest.importorskip("tsdynamics")
    x = rd.reservoir_input(1, device="mps", dtype=torch.float32)
    hidden = rd.ESNLayer(6, feedback_size=1, seed=7, device="mps", dtype=torch.float32)(x)
    y = rd.CGReadoutLayer(6, 1, alpha=1e-3, use_float64=False).to("mps")(hidden)
    model = rd.Model(inputs={"state": x}, outputs={"next": y}, washout=2)
    times = 2.0 + np.arange(30) * 0.125
    observed = ts.Trajectory(times, np.sin(times)[:, None], variables=("position",))
    model.fit(observed)
    tensor = model.forecast(horizon=3, return_type="tensor")
    result = model.forecast(horizon=3)
    assert tensor.device.type == "mps"
    assert tensor.dtype == torch.float32 and result.y.dtype == np.float32
    torch.testing.assert_close(tensor.cpu(), torch.from_numpy(result.y), rtol=1e-6, atol=1e-6)
    expected_times = times[-1] + 0.125 * np.arange(1, 4)
    np.testing.assert_array_equal(result.t, expected_times)
    model.cpu().double()
    np.testing.assert_array_equal(model.forecast(horizon=3).t, expected_times)
