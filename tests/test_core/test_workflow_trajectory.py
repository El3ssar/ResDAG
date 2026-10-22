"""Independent physical-time and lifecycle oracles for trajectory workflows."""

import copy
import io
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

import resdag as rd


@pytest.fixture
def ts():
    return pytest.importorskip("tsdynamics")


def _affine_model(*, washout=0, **options):
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.RidgeReadoutLayer(1, 1, alpha=0.0).double()(x)
    return rd.Model(inputs={"state": x}, outputs={"next": y}, washout=washout, **options)


def _ramp(ts, *, start=10.0, dt=0.25, count=8):
    times = start + dt * np.arange(count, dtype=np.float64)
    return ts.Trajectory(
        times,
        np.arange(count, dtype=np.float64)[:, None],
        variables=("position",),
        meta={"experiment": "measured ramp", "dt": 999.0},
    )


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        else:
            assert actual[name] == value


def test_trajectory_fit_returns_physical_next_samples_with_names_and_actual_axis(ts):
    observed = _ramp(ts)
    original_times, original_values = observed.t.copy(), observed.y.copy()
    model = _affine_model(washout=2).fit(observed)
    prediction = model.forecast(horizon=4)
    assert isinstance(prediction, ts.Trajectory)
    assert prediction.variables == ("position",)
    np.testing.assert_allclose(prediction.y, np.arange(8.0, 12.0)[:, None], rtol=0, atol=1e-11)
    np.testing.assert_allclose(prediction.t, [12.0, 12.25, 12.5, 12.75], rtol=0, atol=1e-12)
    # Input indices 2..6 are supervised: washout and final shifted target do not fit statistics.
    torch.testing.assert_close(model.input_scalers[0]._stat0, torch.tensor([[[4.0]]]).double())
    torch.testing.assert_close(
        model.input_scalers[0]._stat1, torch.tensor([[[np.sqrt(2.5)]]]).double()
    )
    np.testing.assert_array_equal(observed.t, original_times)
    np.testing.assert_array_equal(observed.y, original_values)
    assert observed.meta["dt"] == 999.0
    tensor = model.forecast(horizon=4, return_type="tensor")
    assert isinstance(tensor, torch.Tensor) and tensor.shape == (4, 1)
    torch.testing.assert_close(tensor, torch.from_numpy(prediction.y), rtol=0, atol=0)


def test_explicit_trajectory_history_and_warmup_dates_have_one_boundary_prediction(ts):
    observed = _ramp(ts)
    model = _affine_model().fit(observed)
    shorter = model.forecast(history=observed[:4], horizon=3)
    np.testing.assert_allclose(shorter.y, [[4.0], [5.0], [6.0]], rtol=0, atol=1e-11)
    np.testing.assert_allclose(shorter.t, [11.0, 11.25, 11.5], rtol=0, atol=1e-12)
    result = model.forecast(history=observed, horizon=4, return_warmup=True)
    np.testing.assert_allclose(result.y, np.arange(1.0, 12.0)[:, None], rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.t, 10.0 + 0.25 * np.arange(1, 12), rtol=0, atol=1e-12)
    assert len(result.t) == len(observed.t) - 1 + 4
    assert len(np.unique(result.t)) == len(result.t)


@pytest.mark.parametrize("method", ["linear", "pchip", "cubic"])
def test_irregular_default_grid_preserves_endpoints_and_count_with_physical_step(ts, method):
    observed_times = np.array([2.0, 2.2, 2.9, 4.0])
    observed = ts.Trajectory(
        observed_times, (3 * observed_times + 10)[:, None], variables=("position",)
    )
    model = _affine_model(resample_method=method).fit(observed)
    # Four equally spaced samples span [2, 4]: values 16,18,20,22 and dt=2/3.
    torch.testing.assert_close(model.input_scalers[0]._stat0, torch.tensor([[[18.0]]]).double())
    torch.testing.assert_close(model.input_scalers[0]._stat1, torch.tensor([[[2.0]]]).double())
    prediction = model.forecast(horizon=2)
    np.testing.assert_allclose(prediction.y, [[24.0], [26.0]], rtol=0, atol=1e-10)
    np.testing.assert_allclose(prediction.t, [4 + 2 / 3, 4 + 4 / 3], rtol=0, atol=1e-12)


def test_explicit_resampling_step_is_right_anchored_without_extrapolation(ts):
    observed_times = np.array([2.0, 2.2, 2.9, 4.0])
    observed = ts.Trajectory(
        observed_times, (3 * observed_times + 10)[:, None], variables=("position",)
    )
    model = _affine_model(resample_dt=0.75).fit(observed)
    # In-support grid [2.5,3.25,4] has values [17.5,19.75,22].
    torch.testing.assert_close(model.input_scalers[0]._stat0, torch.tensor([[[18.625]]]).double())
    prediction = model.forecast(horizon=2)
    np.testing.assert_allclose(prediction.y, [[24.25], [26.5]], rtol=0, atol=1e-10)
    np.testing.assert_allclose(prediction.t, [4.75, 5.5], rtol=0, atol=1e-12)


def test_new_history_uses_fitted_sampling_step_and_never_changes_cached_training_tail(ts):
    times = np.arange(8, dtype=np.float64) * 0.25
    training = ts.Trajectory(times, (4 * times + 3)[:, None], variables=("position",))
    model = _affine_model().fit(training)
    history_times = np.array([10.0, 10.2, 10.6, 11.0])
    history = ts.Trajectory(
        history_times, (4 * history_times + 3)[:, None], variables=("position",)
    )
    before = copy.deepcopy(model.state_dict())
    prediction = model.forecast(history=history, horizon=2)
    np.testing.assert_allclose(prediction.y, [[48.0], [49.0]], rtol=0, atol=1e-10)
    np.testing.assert_allclose(prediction.t, [11.25, 11.5], rtol=0, atol=1e-12)
    _assert_state_equal(model.state_dict(), before)
    cached = model.forecast(horizon=1)
    np.testing.assert_allclose(cached.y, [[11.0]], rtol=0, atol=1e-10)
    np.testing.assert_allclose(cached.t, [2.0], rtol=0, atol=1e-12)


def test_tensor_history_needs_timestamps_for_trajectory_output_and_tensor_fit_stays_tensor(ts):
    observed = _ramp(ts)
    values = torch.from_numpy(observed.y.copy())
    model = _affine_model().fit(values)
    assert isinstance(model.forecast(horizon=2), torch.Tensor)
    explicit = model.forecast(history=observed, horizon=2)
    assert isinstance(explicit, ts.Trajectory)
    np.testing.assert_allclose(explicit.t, [12.0, 12.25], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="[Tt]ime|[Tt]rajectory"):
        model.forecast(history=values, horizon=2, return_type="trajectory")
    fitted = _affine_model().fit(observed)
    with pytest.raises(ValueError, match="[Tt]ime|[Tt]rajectory"):
        fitted.forecast(history=values, horizon=2, return_type="trajectory")
    assert isinstance(
        fitted.forecast(history=values, horizon=2, return_type="tensor"), torch.Tensor
    )


def test_tensor_refit_changes_default_return_type_only_after_success(ts):
    model = _affine_model(washout=2).fit(_ramp(ts))
    checkpoint = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="washout"):
        model.fit(torch.arange(2, dtype=torch.float64).reshape(-1, 1))
    _assert_state_equal(model.state_dict(), checkpoint)
    assert isinstance(model.forecast(horizon=1), ts.Trajectory)
    model.fit(torch.arange(12, dtype=torch.float64).reshape(-1, 1))
    assert isinstance(model.forecast(horizon=1), torch.Tensor)
    with pytest.raises(ValueError, match="[Tt]ime|[Tt]rajectory"):
        model.forecast(horizon=1, return_type="trajectory")


def test_checkpoint_roundtrip_restores_time_names_and_resampling_options_with_map_location(ts):
    observed_times = np.array([2.0, 2.2, 2.9, 4.0])
    observed = ts.Trajectory(
        observed_times, (3 * observed_times + 10)[:, None], variables=("position",)
    )
    source = _affine_model(resample_dt=0.75, resample_method="pchip").fit(observed)
    expected = source.forecast(horizon=3)
    stream = io.BytesIO()
    torch.save(source.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, map_location="cpu", weights_only=True)
    restored = _affine_model()
    restored.load_state_dict(state)
    prediction = restored.forecast(horizon=3)
    assert isinstance(prediction, ts.Trajectory)
    assert prediction.variables == expected.variables
    np.testing.assert_array_equal(prediction.t, expected.t)
    np.testing.assert_allclose(prediction.y, expected.y, rtol=0, atol=0)
    _assert_state_equal(restored.state_dict(), source.state_dict())
    restored.float()
    converted = restored.forecast(horizon=3)
    np.testing.assert_allclose(converted.y, expected.y, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(converted.t, expected.t)


def test_legacy_v1_checkpoint_clears_previous_trajectory_context(ts):
    source = _affine_model().fit(torch.arange(12, dtype=torch.float64).reshape(-1, 1))
    legacy = copy.deepcopy(source.state_dict())
    for key in ("resample_dt", "resample_method", "trajectory"):
        legacy["_extra_state"].pop(key)
    for restored in (_affine_model(), _affine_model(resample_method="pchip").fit(_ramp(ts))):
        restored.load_state_dict(legacy)
        prediction = restored.forecast(horizon=3)
        assert isinstance(prediction, torch.Tensor)
        torch.testing.assert_close(prediction, source.forecast(horizon=3), rtol=0, atol=0)
        with pytest.raises(ValueError, match="[Tt]ime|[Tt]rajectory"):
            restored.forecast(horizon=1, return_type="trajectory")


@pytest.mark.parametrize(
    "torch_dtype,numpy_dtype",
    [
        (torch.bfloat16, np.float32),
        (torch.float16, np.float16),
        (torch.float32, np.float32),
        (torch.float64, np.float64),
    ],
)
def test_trajectory_numpy_boundary_promotes_only_bfloat16(ts, torch_dtype, numpy_dtype):
    model = _affine_model().fit(_ramp(ts)).to(dtype=torch_dtype)
    tensor = model.forecast(horizon=3, return_type="tensor")
    trajectory = model.forecast(horizon=3)
    assert tensor.dtype == torch_dtype
    assert trajectory.y.dtype == numpy_dtype
    expected = tensor.float() if torch_dtype == torch.bfloat16 else tensor
    np.testing.assert_array_equal(trajectory.y, expected.numpy())
    assert np.isfinite(trajectory.y).all()
    np.testing.assert_allclose(trajectory.t, [12.0, 12.25, 12.5], rtol=0, atol=1e-12)


@pytest.mark.parametrize("start, dt, horizon", [(0.0, 5e307, 2), (1e307, 8e307, 1)])
def test_overflowing_forecast_clock_fails_before_graph_hooks_or_state_changes(
    ts, start, dt, horizon
):
    observed = _ramp(ts, start=start, dt=dt, count=3)
    model = rd.Model(
        graph=rd.classic_esn(4, 1, 1, seed=7, dtype=torch.float64, readout="svd"), washout=0
    ).fit(observed)
    model(torch.ones(1, 3, 1, dtype=torch.float64))
    stored = model.get_state()
    calls = []
    handle = model.graph.register_forward_hook(lambda *_: calls.append(None))
    try:
        with pytest.raises(ValueError, match="[Tt]ime|finite|represent"):
            model.forecast(horizon=horizon)
    finally:
        handle.remove()
    assert calls == []
    _assert_state_equal(model.get_state(), stored)


@pytest.mark.parametrize("failure", ["timestamps", "solver"])
def test_failed_trajectory_refit_preserves_checkpoint_context_and_carried_state(
    ts, monkeypatch, failure
):
    x = rd.reservoir_input(1, dtype=torch.float64)
    hidden = rd.ESNLayer(6, feedback_size=1, seed=5, dtype=torch.float64)(x)
    y = rd.RidgeReadoutLayer(6, 1, alpha=1e-3).double()(hidden)
    model = rd.Model(inputs={"state": x}, outputs={"next": y}, washout=2).fit(_ramp(ts, count=24))
    model(torch.ones(1, 4, 1, dtype=torch.float64))
    checkpoint, carried = copy.deepcopy(model.state_dict()), model.get_state()
    replacement = _ramp(ts, start=100.0, dt=0.5, count=24)
    if failure == "timestamps":
        replacement.t[5] = replacement.t[4]
        error, match = ValueError, "[Tt]ime|increas|unique"
    else:

        def fail(*_args, **_kwargs):
            raise RuntimeError("test solver failure")

        monkeypatch.setattr(rd.RidgeReadoutLayer, "fit", fail)
        error, match = RuntimeError, "test solver failure"
    with pytest.raises(error, match=match):
        model.fit(replacement)
    _assert_state_equal(model.state_dict(), checkpoint)
    _assert_state_equal(model.get_state(), carried)


def test_actual_lorenz_trajectory_fits_and_forecasts_with_source_names_and_future_times(ts):
    system = ts.systems.Lorenz()
    trajectory = system.run(final_time=1.0, dt=0.02)
    training = trajectory[:41]
    model = rd.Model(
        graph=rd.classic_esn(8, 3, 3, seed=7, dtype=torch.float64, readout="svd"), washout=5
    ).fit(training)
    prediction = model.forecast(horizon=5)
    assert isinstance(prediction, ts.Trajectory)
    assert prediction.variables == ("x", "y", "z")
    assert prediction.y.shape == (5, 3) and np.isfinite(prediction.y).all()
    assert prediction.system is system
    np.testing.assert_allclose(
        prediction.t, training.t[-1] + 0.02 * np.arange(1, 6), rtol=0, atol=1e-12
    )
    # Portable metadata must not serialize the live producing-system object.
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
    restored = rd.Model(
        graph=rd.classic_esn(8, 3, 3, seed=7, dtype=torch.float64, readout="svd"), washout=5
    )
    restored.load_state_dict(checkpoint)
    replay = restored.forecast(horizon=5)
    assert replay.system is None
    assert replay.variables == prediction.variables
    np.testing.assert_array_equal(replay.t, prediction.t)
    np.testing.assert_allclose(replay.y, prediction.y, rtol=0, atol=0)
    assert replay.meta["source"]["system"]["name"] == "Lorenz"


def test_tensor_workflow_never_imports_optional_dependency_or_transfers_to_cpu():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import torch

        class NoTSDynamics(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'tsdynamics' or fullname.startswith('tsdynamics.'):
                    raise AssertionError('Tensor workflow imported optional TSDynamics')
                return None

        sys.meta_path.insert(0, NoTSDynamics())
        import resdag as rd
        assert not any(name == 'tsdynamics' or name.startswith('tsdynamics.') for name in sys.modules)

        def forbidden_transfer(*args, **kwargs):
            raise AssertionError('Tensor workflow transferred a tensor to CPU or NumPy')

        torch.Tensor.cpu = forbidden_transfer
        torch.Tensor.numpy = forbidden_transfer
        x = rd.reservoir_input(1, dtype=torch.float64)
        y = rd.RidgeReadoutLayer(1, 1, alpha=0.0).double()(x)
        model = rd.Model(inputs={'state': x}, outputs={'next': y}, washout=0)
        series = torch.arange(12, dtype=torch.float64).reshape(-1, 1)
        model.fit(series)
        source = series[:4].clone().requires_grad_()
        model(source).sum().backward()
        assert source.grad is not None and bool(torch.isfinite(source.grad).all())
        explicit, _ = model.forward_stateless(series[:4])
        assert explicit.shape == (4, 1)
        prediction = model.forecast(horizon=3)
        assert isinstance(prediction, torch.Tensor)
        assert torch.allclose(prediction, torch.tensor([[12.], [13.], [14.]], dtype=torch.float64))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_trajectory_checkpoint_load_and_tensor_forecast_need_no_optional_import(ts, tmp_path):
    source = _affine_model().fit(_ramp(ts))
    path = tmp_path / "trajectory-state.pt"
    torch.save(source.state_dict(), path)
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import torch

        class NoTSDynamics(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'tsdynamics' or fullname.startswith('tsdynamics.'):
                    raise AssertionError('Portable checkpoint imported optional TSDynamics')
                return None

        sys.meta_path.insert(0, NoTSDynamics())
        import resdag as rd

        def forbidden_transfer(*args, **kwargs):
            raise AssertionError('Tensor forecast transferred a tensor to CPU or NumPy')

        torch.Tensor.cpu = forbidden_transfer
        torch.Tensor.numpy = forbidden_transfer
        x = rd.reservoir_input(1, dtype=torch.float64)
        y = rd.RidgeReadoutLayer(1, 1, alpha=0.0).double()(x)
        restored = rd.Model(inputs={'state': x}, outputs={'next': y}, washout=0)
        state = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
        restored.load_state_dict(state)
        prediction = restored.forecast(horizon=3, return_type='tensor')
        assert isinstance(prediction, torch.Tensor)
        assert torch.allclose(prediction, torch.tensor([[8.], [9.], [10.]], dtype=torch.float64))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
