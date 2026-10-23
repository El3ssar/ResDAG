"""Complete artifacts retain optional trajectory clocks, names and physical units."""

import builtins
import copy
import subprocess
import sys

import numpy as np
import pytest
import torch

import resdag as rd
from resdag.data import _trajectory


@pytest.fixture
def ts():
    return pytest.importorskip("tsdynamics")


def _affine():
    source = rd.reservoir_input(1, dtype=torch.float64)
    prediction = rd.SVDReadoutLayer(1, 1, alpha=0).double()(source)
    return rd.Model(
        inputs={"state": source},
        outputs={"prediction": prediction},
        washout=2,
        resample_dt=0.25,
        resample_method="pchip",
    )


def _ramp(ts):
    times = 5 + 0.25 * np.arange(20)
    return ts.Trajectory(
        times,
        (3 * times + 11)[:, None],
        variables=("position",),
        system=torch.nn.Linear(1, 1),
        meta={"experiment": "measured", "dt": 999, "parameters": {"slope": np.float64(3)}},
    )


@pytest.mark.parametrize("trusted", [False, True])
def test_complete_artifact_defaults_to_trajectory_in_original_units(tmp_path, ts, trusted):
    observed = _ramp(ts)
    model = _affine().fit(observed)
    original = model.forecast(horizon=4)
    context = copy.deepcopy(model.get_extra_state()["trajectory"])
    path = model.save(tmp_path / "trajectory.pt", trusted=trusted)
    restored = rd.load(path, trusted=trusted, map_location="cpu")
    actual = restored.forecast(horizon=4)
    assert isinstance(actual, ts.Trajectory)
    assert actual.variables == ("position",)
    assert actual.system is None
    assert original.system is observed.system
    expected_times = observed.t[-1] + 0.25 * np.arange(1, 5)
    np.testing.assert_array_equal(actual.t, expected_times)
    np.testing.assert_allclose(actual.y[:, 0], 3 * expected_times + 11, rtol=0, atol=3e-12)
    np.testing.assert_array_equal(actual.y, original.y)
    assert actual.meta["source"] == original.meta["source"]
    assert restored.get_extra_state()["trajectory"] == context
    assert restored.resample_dt == 0.25 and restored.resample_method == "pchip"
    tensors = restored.forecast(horizon=4, return_type="tensor")
    torch.testing.assert_close(tensors, torch.from_numpy(actual.y), rtol=0, atol=0)
    assert tensors.device.type == "cpu" and tensors.dtype == torch.float64


def _position(times):
    return times * (times + 1) / 2


def _named_model():
    state, forcing = (rd.reservoir_input(1, dtype=torch.float64) for _ in range(2))
    features = rd.Concatenate()(state, forcing)
    near = rd.SVDReadoutLayer(2, 1, alpha=0, name="near_fit").double()(features)
    far = rd.SVDReadoutLayer(2, 1, alpha=0, name="far_fit").double()(features)
    return rd.Model(
        inputs={"state": state, "forcing": forcing},
        outputs={"near": near, "far": far},
        prediction_lead={"near": 1, "far": 2},
        feedback=[rd.Feedback("far", "state", delay=2)],
        washout=2,
    )


@pytest.mark.parametrize("trusted", [False, True])
def test_named_ports_different_leads_and_delayed_route_survive_complete_artifact(
    tmp_path, ts, trusted
):
    state_times = np.arange(13.0)
    force_times = np.arange(0.25, 12.26, 0.5)
    inputs = {
        "forcing": ts.Trajectory(force_times, (force_times + 1)[:, None], variables=("force",)),
        "state": ts.Trajectory(
            state_times, _position(state_times)[:, None], variables=("position",)
        ),
    }
    near_times, far_times = np.arange(6.0, 17.0), np.arange(5.0, 17.0)
    targets = {
        "far": ts.Trajectory(far_times, _position(far_times)[:, None], variables=("far_position",)),
        "near": ts.Trajectory(
            near_times, _position(near_times)[:, None], variables=("near_position",)
        ),
    }
    model = _named_model().fit(inputs, targets)
    future_times = np.array([13.0, 13.25, 14.5, 15.0])
    future = {
        "forcing": ts.Trajectory(future_times, (future_times + 1)[:, None], variables=("force",))
    }
    expected = model.forecast(horizon=4, future_inputs=future)
    model.save(tmp_path / "named.pt", trusted=trusted)
    restored = rd.load(tmp_path / "named.pt", trusted=trusted)
    assert restored.input_names == ("state", "forcing")
    assert restored.output_names == ("near", "far")
    assert restored.prediction_leads == {"near": 1, "far": 2}
    assert restored.feedback == (rd.Feedback("far", "state", delay=2),)
    actual = restored.forecast(horizon=4, future_inputs=future)
    tensors = restored.forecast(horizon=4, future_inputs=future, return_type="tensor")
    assert list(actual) == list(tensors) == ["near", "far"]
    for name in ("near", "far"):
        assert isinstance(actual[name], ts.Trajectory)
        assert actual[name].variables == (f"{name}_position",)
        np.testing.assert_array_equal(actual[name].t, [13.0, 14.0, 15.0, 16.0])
        np.testing.assert_array_equal(actual[name].t, expected[name].t)
        np.testing.assert_array_equal(actual[name].y, expected[name].y)
        np.testing.assert_allclose(
            actual[name].y[:, 0], _position(actual[name].t), rtol=0, atol=2e-10
        )
        torch.testing.assert_close(tensors[name], torch.from_numpy(actual[name].y), rtol=0, atol=0)


def _metadata_context():
    record = {
        "variables": ["position"],
        "source": {"metadata": {"experiment": "recorded elsewhere"}, "system": None},
        "observation": {"start_time": 5.0, "last_time": 9.75, "n_samples": 20},
        "resampled": False,
        "held_out_samples": 0,
    }
    return {
        "version": 2,
        "clock": {
            "start_time": 5.0,
            "last_time": 9.75,
            "dt": 0.25,
            "n_samples": 20,
            "reference_input": "state",
            "requested_dt": 0.25,
            "method": "pchip",
        },
        "inputs": {"state": copy.deepcopy(record)},
        "outputs": {"prediction": copy.deepcopy(record)},
        "supervision": {"start": 0, "stop": 19, "prediction_leads": {"prediction": 1}},
    }


@pytest.mark.parametrize("trusted", [False, True])
def test_portable_clock_metadata_and_tensor_forecast_need_no_optional_import(
    tmp_path, monkeypatch, trusted
):
    model = _affine().fit(torch.arange(20, dtype=torch.float64)[:, None])
    context = _metadata_context()
    checkpoint = model.state_dict()
    checkpoint["_extra_state"]["trajectory"] = context
    model.load_state_dict(checkpoint)
    original_import, original_import_module = (
        builtins.__import__,
        _trajectory.importlib.import_module,
    )

    def guarded(name, *args, **kwargs):
        if name.startswith("tsdynamics"):
            raise ImportError("optional dependency deliberately absent")
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        if name.startswith("tsdynamics"):
            raise ImportError("optional dependency deliberately absent")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(_trajectory.importlib, "import_module", guarded_import_module)
    model.save(tmp_path / "metadata.pt", trusted=trusted)
    restored = rd.load(tmp_path / "metadata.pt", trusted=trusted)
    assert restored.get_extra_state()["trajectory"] == context
    tensor = restored.forecast(horizon=3, return_type="tensor")
    torch.testing.assert_close(
        tensor, torch.tensor([[20.0], [21.0], [22.0]]).double(), atol=2e-12, rtol=0
    )
    with pytest.raises(ImportError, match=r"resdag\[dynamics\]"):
        restored.forecast(horizon=1)


def test_fresh_process_generic_load_preserves_trajectory_defaults_for_both_formats(tmp_path, ts):
    model = _affine().fit(_ramp(ts))
    for trusted in (False, True):
        model.save(tmp_path / f"fresh-{trusted}.pt", trusted=trusted)
    code = f"""import numpy as np
import torch
import tsdynamics as ts
import resdag as rd
for trusted in (False, True):
    model=rd.load({str(tmp_path)!r}+'/fresh-'+str(trusted)+'.pt', trusted=trusted)
    result=model.forecast(horizon=3)
    assert isinstance(result, ts.Trajectory)
    assert result.variables == ('position',)
    assert result.system is None
    np.testing.assert_array_equal(result.t, [10., 10.25, 10.5])
    np.testing.assert_allclose(result.y[:,0], 3*result.t+11, rtol=0, atol=3e-12)
    torch.testing.assert_close(model.forecast(horizon=3, return_type='tensor'), torch.from_numpy(result.y), rtol=0, atol=0)
"""
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
