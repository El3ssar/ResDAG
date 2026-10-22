"""Optional trajectory clocks/units agree with independent tensor workflows."""

import copy

import numpy as np
import pytest
import torch

import resdag as rd


def _graph(kind):
    options = dict(reservoir_size=8, readout="svd", readout_alpha=0.05, seed=19)
    if kind == "parallel":
        return rd.parallel_esn(input_dim=4, core_width=2, halo=1, dtype=torch.float64, **options)
    return rd.coupled_ensemble_esn(
        2,
        model_factory=rd.classic_esn,
        feedback_size=4,
        output_size=4,
        dtype=torch.float64,
        **options,
    )


@pytest.mark.parametrize("kind", ["parallel", "coupled"])
@pytest.mark.parametrize("irregular", [False, True])
def test_trajectory_fit_clock_and_portable_context_match_tensor_oracle(kind, irregular):
    ts = pytest.importorskip("tsdynamics")
    target_times = np.linspace(15, 25, 50)
    times = target_times.copy()
    if irregular:
        times += 0.03 * np.sin(np.linspace(0, 2 * np.pi, 50))
    values = np.stack([10 * (i + 1) + np.sin(times + i) for i in range(4)], axis=-1)
    # Independent default resampling reference: unchanged endpoints and count.
    prepared = np.stack([np.interp(target_times, times, values[:, i]) for i in range(4)], axis=-1)
    graph = _graph(kind)
    expected = rd.Model(graph=copy.deepcopy(graph), washout=4).fit(torch.from_numpy(prepared))
    trajectory = ts.Trajectory(times, values, variables=("a", "b", "c", "d"))
    model = rd.Model(graph=graph, washout=4).fit(trajectory)
    reference = expected.forecast(horizon=5)
    prediction = model.forecast(horizon=5)
    assert isinstance(prediction, ts.Trajectory)
    assert prediction.variables == trajectory.variables
    dt = (times[-1] - times[0]) / (len(times) - 1)
    np.testing.assert_allclose(prediction.t, times[-1] + dt * np.arange(1, 6), atol=1e-12)
    np.testing.assert_allclose(prediction.y, reference.numpy(), rtol=1e-11, atol=1e-11)
    tensor = model.forecast(horizon=5, return_type="tensor")
    torch.testing.assert_close(tensor, reference, rtol=1e-11, atol=1e-11)

    restored = rd.Model(graph=_graph(kind), washout=0)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    after_load = restored.forecast(horizon=5)
    np.testing.assert_array_equal(after_load.t, prediction.t)
    np.testing.assert_allclose(after_load.y, prediction.y, rtol=0, atol=0)
    assert after_load.variables == prediction.variables
