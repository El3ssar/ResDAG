"""Compare diagnostics to actual cell derivatives and independent held-out fits."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resdag.layers import ESNLayer, EuSNLayer
from resdag.metrics import effective_spectral_radius, memory_capacity, reservoir_lyapunov
from resdag.metrics import lyapunov as lyap


def calibrated_reservoir():
    reservoir = ESNLayer(
        2,
        feedback_size=1,
        bias=False,
        spectral_radius=None,
        leak_rate=[0.2, 0.7],
        dtype=torch.float64,
        seed=17,
    )
    cell = reservoir.cell
    with torch.no_grad():
        cell.weight_hh.copy_(torch.diag(torch.tensor([0.6, 0.7], dtype=torch.float64)))
        cell.weight_feedback.copy_(torch.tensor([[0.4], [-0.3]], dtype=torch.float64))
    cell._recurrent_gain = torch.tensor([0.25, 0.5], dtype=torch.float64)
    cell._intrinsic_gain = torch.tensor([1.3, 0.8], dtype=torch.float64)
    cell._intrinsic_bias = torch.zeros(2, dtype=torch.float64)
    cell._has_recurrent_gain = True
    cell._has_intrinsic = True
    return reservoir


def test_effective_radius_matches_autograd_on_calibrated_vector_leak_cell():
    reservoir = calibrated_reservoir().eval()
    drive = torch.linspace(-0.7, 0.9, 9, dtype=torch.float64).reshape(1, -1, 1)
    cell = reservoir.cell
    state = cell.init_state(1, torch.device("cpu"), torch.float64)
    states, radii = [], []
    for time in range(drive.shape[1]):
        previous = state.detach().requires_grad_(True)

        def transition(h):
            return cell([drive[:, time]], h)[1]

        jacobian = torch.autograd.functional.jacobian(transition, previous)[0, :, 0, :]
        _, state = cell([drive[:, time]], state)
        states.append(state.detach())
        if time > 0:  # the first stored state has no supplied predecessor
            radii.append(torch.linalg.eigvals(jacobian).abs().max().item())
    assert effective_spectral_radius(reservoir, torch.stack(states, 1)) == pytest.approx(
        np.mean(radii), abs=1e-12
    )


def test_lyapunov_matches_closed_form_calibrated_vector_leak_spectrum():
    reservoir = calibrated_reservoir()
    # With zero drive and bias the orbit remains zero, so its Jacobian is constant.
    expected = np.sort(np.log([0.8 + 0.2 * 0.6 * 0.25 * 1.3, 0.3 + 0.7 * 0.7 * 0.5 * 0.8]))[::-1]
    actual = reservoir_lyapunov(reservoir, torch.zeros(1, 5, 1), k=2, transient=100, horizon=100)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("diagnostic", ["lyapunov", "radius", "memory"])
def test_diagnostics_preserve_caller_state_mode_weights_and_rng(diagnostic):
    reservoir = calibrated_reservoir().train()
    reservoir(torch.ones(2, 4, 1, dtype=torch.float64))
    original_state = reservoir.get_state().clone()
    original_buffers = {name: value.clone() for name, value in reservoir.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    if diagnostic == "lyapunov":
        reservoir_lyapunov(reservoir, torch.zeros(1, 5, 1), horizon=10)
    elif diagnostic == "radius":
        effective_spectral_radius(reservoir, torch.zeros(1, 5, 2))
    else:
        memory_capacity(reservoir, max_delay=4, n_samples=20, test_samples=30, washout=10)
    assert reservoir.training and reservoir.cell.training
    torch.testing.assert_close(reservoir.get_state(), original_state, rtol=0, atol=0)
    for name, value in reservoir.state_dict().items():
        torch.testing.assert_close(value, original_buffers[name], rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("diagnostic", ["lyapunov", "radius"])
def test_unvalidated_update_equations_are_explicitly_rejected(diagnostic):
    reservoir = EuSNLayer(5, feedback_size=1, seed=0)
    with pytest.raises(NotImplementedError, match="standard ESNCell"):
        if diagnostic == "lyapunov":
            reservoir_lyapunov(reservoir, torch.zeros(1, 5, 1), horizon=5)
        else:
            effective_spectral_radius(reservoir, torch.zeros(1, 5, 5))


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"transient": -1}, {"k": 0}])
def test_lyapunov_rejects_invalid_step_counts(kwargs):
    with pytest.raises(ValueError):
        reservoir_lyapunov(calibrated_reservoir(), torch.zeros(1, 5, 1), **kwargs)


def test_small_fit_does_not_report_in_sample_memory_capacity():
    reservoir = ESNLayer(5, feedback_size=1, bias=False, spectral_radius=0.9, seed=0)
    capacity = memory_capacity(
        reservoir, n_samples=10, max_delay=100, washout=20, test_samples=2000
    )
    # The previous same-sample fit/score returned over 50 for this 5-state model.
    assert 0 <= capacity < 5


def test_memory_curve_matches_independent_numpy_held_out_reference():
    reservoir = ESNLayer(
        1,
        feedback_size=1,
        bias=False,
        spectral_radius=None,
        activation="identity",
        dtype=torch.float64,
        seed=0,
    )
    with torch.no_grad():
        reservoir.cell.weight_hh.fill_(0.6)
        reservoir.cell.weight_feedback.fill_(0.3)
    train_samples, test_samples, washout, max_delay, alpha, seed = 25, 40, 5, 4, 0.03, 27
    generator = torch.Generator().manual_seed(seed)

    def reference_samples(count):
        start = washout + max_delay
        inputs = (
            2 * torch.rand(start + count, generator=generator, dtype=torch.float64) - 1
        ).numpy()
        state = 0.0
        states = []
        for value in inputs:
            state = 0.6 * state + 0.3 * value
            states.append(state)
        times = np.arange(start, start + count)
        targets = inputs[times[:, None] - np.arange(1, max_delay + 1)]
        return np.asarray(states)[start:, None], targets

    train_x, train_y = reference_samples(train_samples)
    test_x, test_y = reference_samples(test_samples)
    centered = train_x - train_x.mean(0)
    coefs = np.linalg.solve(
        centered.T @ centered + alpha * np.eye(1), centered.T @ (train_y - train_y.mean(0))
    )
    predictions = (test_x - train_x.mean(0)) @ coefs + train_y.mean(0)
    expected = [np.corrcoef(predictions[:, k], test_y[:, k])[0, 1] ** 2 for k in range(max_delay)]
    _, curve = memory_capacity(
        reservoir,
        n_samples=train_samples,
        test_samples=test_samples,
        washout=washout,
        max_delay=max_delay,
        alpha=alpha,
        seed=seed,
        return_curve=True,
    )
    np.testing.assert_allclose(curve, expected, atol=1e-14)


@pytest.mark.parametrize("current_namespace", [False, True])
def test_optional_analysis_namespace_dispatch(monkeypatch, current_namespace):
    calls = []

    def spectrum(system, **kwargs):
        calls.append(("system", kwargs))
        return SimpleNamespace(values=np.array([0.4, 0, -2]))

    def data(values, **kwargs):
        calls.append(("data", kwargs))
        return SimpleNamespace(lyapunov=0.3)

    api = SimpleNamespace(lyapunov_spectrum=spectrum, lyapunov_from_data=data)
    package = SimpleNamespace(analysis=api) if current_namespace else api
    monkeypatch.setattr(lyap, "_import_tsdynamics", lambda: package)
    assert lyap._lambda_from_system(object(), dt=0.1) == 0.4
    assert lyap._lambda_from_series(np.arange(100), dt=0.2) == 0.3
    assert calls == [("system", {"dt": 0.1}), ("data", {"dt": 0.2})]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_diagnostics_use_supported_cpu_precision_and_preserve_device():
    reservoir = ESNLayer(3, feedback_size=1, seed=4, device="mps")
    drive = torch.zeros(1, 5, 1, device="mps")
    states = reservoir(drive)
    assert np.isfinite(reservoir_lyapunov(reservoir, drive, horizon=5))
    assert np.isfinite(effective_spectral_radius(reservoir, states))
    assert np.isfinite(
        memory_capacity(reservoir, max_delay=3, n_samples=15, test_samples=30, washout=5)
    )
    assert reservoir.cell.weight_hh.device.type == "mps"
    assert reservoir.training


def test_unregularized_zero_capacity_handles_rank_deficient_states():
    reservoir = ESNLayer(2, feedback_size=1, bias=False, seed=1)
    with torch.no_grad():
        reservoir.cell.weight_hh.zero_()
        reservoir.cell.weight_feedback.zero_()
    assert (
        memory_capacity(reservoir, max_delay=3, n_samples=10, test_samples=20, washout=0, alpha=0)
        == 0
    )


@pytest.mark.parametrize("scale", [1e4, 1e8])
@pytest.mark.parametrize("alpha", [0.0, 1e-8])
def test_memory_capacity_handles_large_duplicate_states_without_mutating_caller(scale, alpha):
    reservoir = ESNLayer(
        16,
        feedback_size=1,
        bias=False,
        spectral_radius=None,
        activation="identity",
        dtype=torch.float64,
        seed=17,
    ).train()
    with torch.no_grad():
        reservoir.cell.weight_hh.zero_()
        reservoir.cell.weight_feedback.fill_(scale)
    reservoir(torch.ones(2, 3, 1, dtype=torch.float64))
    reservoir.cell.eval()  # A mixed mode configuration must also be preserved.
    original_state = reservoir.get_state().clone()
    original_weights = {name: value.clone() for name, value in reservoir.state_dict().items()}
    original_modes = [module.training for module in reservoir.modules()]
    original_rng = torch.random.get_rng_state().clone()
    n_samples, test_samples, max_delay, washout, seed = 10, 40, 4, 2, 31

    capacity, curve = memory_capacity(
        reservoir,
        n_samples=n_samples,
        test_samples=test_samples,
        max_delay=max_delay,
        washout=washout,
        alpha=alpha,
        seed=seed,
        return_curve=True,
    )

    # Every state is the same scaled current input. Any nonconstant linear
    # reconstruction has the current input's squared correlation with each
    # delayed target, independent of the fit's scale, intercept and ridge.
    generator = torch.Generator().manual_seed(seed)
    start = washout + max_delay
    torch.rand(start + n_samples, generator=generator, dtype=torch.float64)
    test_input = (
        2 * torch.rand(start + test_samples, generator=generator, dtype=torch.float64) - 1
    ).numpy()
    expected = [
        np.corrcoef(test_input[start:], test_input[start - delay : start + test_samples - delay])[
            0, 1
        ]
        ** 2
        for delay in range(1, max_delay + 1)
    ]
    assert np.isfinite(capacity)
    assert 0 <= capacity <= max_delay
    # At scale 1e8, the augmented system spans roughly thirteen orders of
    # magnitude; compare the observable rather than nearly cancelling weights.
    np.testing.assert_allclose(curve, expected, atol=1e-7)
    assert capacity == pytest.approx(np.sum(expected), abs=1e-7)
    torch.testing.assert_close(reservoir.get_state(), original_state, rtol=0, atol=0)
    for name, value in reservoir.state_dict().items():
        torch.testing.assert_close(value, original_weights[name], rtol=0, atol=0)
    assert [module.training for module in reservoir.modules()] == original_modes
    assert torch.equal(torch.random.get_rng_state(), original_rng)
