"""Independent numerical, memory and workflow checks for opt-in SciPy LSMR."""

import math
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg
import torch

import resdag as rd
from resdag.layers.readouts.lsmr_readout import _centered_operator


def _data(n=73, f=11, outputs=3):
    generator = torch.Generator().manual_seed(831)
    return (
        torch.randn(n, f, dtype=torch.float64, generator=generator),
        torch.randn(n, outputs, dtype=torch.float64, generator=generator),
    )


def _reference(x, y, alpha, bias):
    """Separate LAPACK SVD solve of an explicitly augmented design."""
    x, y = x.double().cpu().numpy(), y.double().cpu().numpy()
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    a, b = (x - xm, y - ym) if bias else (x, y)
    if alpha:
        a = np.vstack((a, math.sqrt(alpha) * np.eye(a.shape[1])))
        b = np.vstack((b, np.zeros((x.shape[1], y.shape[1]))))
    coefs = scipy.linalg.lstsq(a, b, lapack_driver="gelsd")[0]
    return torch.from_numpy(coefs), torch.from_numpy(ym - xm @ coefs) if bias else None


def _layer(f=11, outputs=3, **kwargs):
    options = {"alpha": 0.17, "atol": 1e-13, "btol": 1e-13, "max_iter": 500, **kwargs}
    return rd.LSMRReadoutLayer(f, outputs, **options).double()


@pytest.mark.parametrize("shape", [(73, 11), (11, 35)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 1e-9, 0.17])
def test_matches_independent_augmented_lapack_reference(shape, bias, alpha):
    x, y = _data(*shape)
    layer = _layer(shape[1], bias=bias, alpha=alpha)
    before_x, before_y = x.clone(), y.clone()
    layer.fit(x, y)
    coefficients, intercept = _reference(x, y, alpha, bias)
    torch.testing.assert_close(layer.weight.T, coefficients, rtol=2e-9, atol=2e-10)
    if bias:
        torch.testing.assert_close(layer.bias, intercept, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(x, before_x, atol=0, rtol=0)
    torch.testing.assert_close(y, before_y, atol=0, rtol=0)
    report = layer.last_solve
    assert isinstance(report, rd.LSMRSolveReport)
    assert report.alpha == alpha == layer.alpha
    assert report.solve_device == "cpu" and report.solve_dtype == torch.float64
    assert len(report.columns) == 3 and all(c.stop_code in (0, 1, 2, 4, 5) for c in report.columns)
    assert all(c.condition_estimate >= 0 for c in report.columns)
    with pytest.raises(FrozenInstanceError):
        report.alpha = 3


@pytest.mark.parametrize("alpha", [0.0, 1e-10, 0.1])
def test_rank_deficient_design_including_constant_column_matches_minimum_norm(alpha):
    x, y = _data(90, 4)
    x = torch.cat((x, x[:, :2], x.new_ones((90, 1))), -1)
    layer = _layer(7, alpha=alpha)
    layer.fit(x, y)
    # Collapse exact duplicate columns before solving. Their combined weight
    # has penalty alpha/2; split it equally for the unique minimum-norm fit.
    # This avoids amplifying LAPACK roundoff along null directions by 1/alpha.
    z, target = x[:, :4] - x[:, :4].mean(0), y - y.mean(0)
    penalty = torch.diag(torch.tensor([alpha / 2, alpha / 2, alpha, alpha]))
    combined = torch.linalg.solve(z.T @ z + penalty, z.T @ target)
    w = torch.cat((combined[:2] / 2, combined[2:], combined[:2] / 2, x.new_zeros(1, 3)))
    b = y.mean(0) - x.mean(0) @ w
    torch.testing.assert_close(layer(x), x @ w + b, atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(layer.weight.T, w, atol=2e-8, rtol=2e-7)
    assert layer.weight[:, -1].count_nonzero() == 0


@pytest.mark.parametrize("scale", [1e-200, 1e-20, 1.0, 1e100])
@pytest.mark.parametrize("bias", [False, True])
def test_target_units_do_not_create_false_zero_convergence(scale, bias):
    x, y = _data(73, 5)
    layer = _layer(5, bias=bias)
    layer.fit(x, y * scale)
    w, b = _reference(x, y, layer.alpha, bias)
    torch.testing.assert_close(layer.weight.T / scale, w, atol=2e-11, rtol=2e-10)
    if bias:
        torch.testing.assert_close(layer.bias / scale, b, atol=2e-11, rtol=2e-10)
    assert layer.weight.count_nonzero() == layer.weight.numel()


def test_large_offset_centering_is_applied_to_blocks_before_products():
    x, y = _data(180, 7)
    x += 1e9
    y += 1e6
    layer = _layer(7, chunk_size=13)
    layer.fit(x, y)
    w, b = _reference(x, y, layer.alpha, True)
    torch.testing.assert_close(layer.weight.T, w, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(layer.bias, b, rtol=2e-9, atol=2e-3)
    torch.testing.assert_close(layer(x), x @ w + b, rtol=1e-10, atol=2e-5)


def test_ill_conditioned_design_has_forceable_budget_and_condition_limit():
    x = torch.diag(torch.logspace(0, -6, 14, dtype=torch.float64))
    y = torch.ones(14, 1, dtype=torch.float64)
    layer = _layer(14, 1, bias=False, alpha=1e-12, conlim=0, atol=1e-14, btol=1e-14)
    layer.fit(x, y)
    expected, _ = _reference(x, y, layer.alpha, False)
    torch.testing.assert_close(layer.weight.T, expected, rtol=5e-5, atol=2e-6)
    limited = _layer(14, 1, bias=False, alpha=0, conlim=3)
    with pytest.raises(RuntimeError, match="condition limit"):
        limited.fit(x, y)
    assert not limited.is_fitted and limited.last_solve is None


@pytest.mark.parametrize("bias", [False, True])
def test_zero_design_and_zero_or_constant_outputs_are_valid(bias):
    x = torch.zeros(41, 7, dtype=torch.float64)
    y = torch.cat((x[:, :1], x.new_full((41, 1), 3)), -1)
    layer = _layer(7, 2, bias=bias)
    layer.fit(x, y)
    assert layer.weight.count_nonzero() == 0
    expected = y if bias else torch.zeros_like(y)
    torch.testing.assert_close(layer(x), expected, atol=0, rtol=0)


def test_iteration_failure_never_publishes_partial_outputs_or_fitted_flag(monkeypatch):
    x, y = _data()
    layer = _layer()
    previous = {name: value.clone() for name, value in layer.state_dict().items()}
    original = scipy.sparse.linalg.lsmr
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = list(original(*args, **kwargs))
        if calls == 2:
            result[1] = 7
        return tuple(result)

    monkeypatch.setattr(scipy.sparse.linalg, "lsmr", fail_second)
    with pytest.raises(RuntimeError, match="output 1.*iteration limit"):
        layer.fit(x, y)
    assert not layer.is_fitted and layer.last_solve is None
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, previous[name], rtol=0, atol=0)


def test_real_iteration_limit_and_refit_failure_preserve_previous_fit():
    x, y = _data()
    layer = _layer()
    layer.fit(x, y)
    weights, bias, report = layer.weight.clone(), layer.bias.clone(), layer.last_solve
    layer.max_iter = 1
    with pytest.raises(RuntimeError, match="iteration limit"):
        layer.fit(x, y)
    assert layer.is_fitted and layer.last_solve is report
    torch.testing.assert_close(layer.weight, weights, atol=0, rtol=0)
    torch.testing.assert_close(layer.bias, bias, atol=0, rtol=0)


@pytest.mark.parametrize("nonfinite", ["weight", "diagnostic"])
def test_nonfinite_scipy_results_are_rejected_before_fit_publication(monkeypatch, nonfinite):
    x, y = _data()
    original = scipy.sparse.linalg.lsmr

    def broken(*args, **kwargs):
        result = list(original(*args, **kwargs))
        if nonfinite == "weight":
            result[0][0] = np.nan
        else:
            result[6] = math.inf
        return tuple(result)

    monkeypatch.setattr(scipy.sparse.linalg, "lsmr", broken)
    layer = _layer()
    with pytest.raises(RuntimeError, match="no weights were published"):
        layer.fit(x, y)
    assert not layer.is_fitted


def test_norm_underflow_cannot_make_nonzero_design_look_solved():
    layer = _layer(2, 1, bias=False, alpha=0)
    x = torch.eye(2, dtype=torch.float64) * 1e-200
    with pytest.raises(RuntimeError, match="underflow"):
        layer.fit(x, torch.ones(2, 1, dtype=torch.float64))
    assert not layer.is_fitted


def test_operator_adjoint_and_scratch_bound_without_gram_or_full_centered_design(monkeypatch):
    x, y = _data(173, 11)
    centered = x.numpy() - x.numpy().mean(0)
    operator = _centered_operator(x.numpy(), x.numpy().mean(0), 7)
    v, u = np.arange(11.0), np.arange(173.0)
    np.testing.assert_allclose(operator @ v, centered @ v, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(operator.rmatvec(u), centered.T @ u, rtol=1e-13, atol=1e-10)
    assert np.dot(u, operator @ v) == pytest.approx(np.dot(operator.rmatvec(u), v), rel=1e-13)
    original_empty, original_subtract, original_lsmr = (
        np.empty,
        np.subtract,
        scipy.sparse.linalg.lsmr,
    )
    seen_rows = []

    def empty(shape, *args, **kwargs):
        if isinstance(shape, tuple):
            assert shape not in ((11, 11), (173, 11))
        return original_empty(shape, *args, **kwargs)

    def subtract(a, b, *args, **kwargs):
        if isinstance(a, np.ndarray) and a.ndim == 2:
            seen_rows.append(a.shape[0])
            assert a.shape[0] <= 7
        return original_subtract(a, b, *args, **kwargs)

    def lsmr(actual, rhs, **kwargs):
        assert isinstance(actual, scipy.sparse.linalg.LinearOperator)
        assert kwargs["damp"] == math.sqrt(0.17)
        return original_lsmr(actual, rhs, **kwargs)

    monkeypatch.setattr(np, "empty", empty)
    monkeypatch.setattr(np, "subtract", subtract)
    monkeypatch.setattr(scipy.sparse.linalg, "lsmr", lsmr)
    layer = _layer(chunk_size=7)
    layer.fit(x, y)
    assert seen_rows and max(seen_rows) == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -1},
        {"alpha": math.nan},
        {"alpha": math.inf},
        {"alpha": True},
        {"atol": -1},
        {"atol": math.nan},
        {"btol": math.inf},
        {"conlim": -1},
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"max_iter": True},
        {"chunk_size": 0},
        {"chunk_size": False},
        {"cpu_fallback": 1},
    ],
)
def test_invalid_configuration_is_rejected_early(kwargs):
    with pytest.raises(ValueError):
        _layer(**kwargs)


def _series(dtype=torch.float64, device="cpu"):
    t = torch.linspace(0.0, 8.0, 100, dtype=dtype, device=device)
    return torch.stack((t.sin(), t.cos()), -1)


def test_lsmr_native_preset_can_fit_without_solver_options():
    model = rd.ESN(12, input_size=2, washout=3, readout="lsmr", seed=17, dtype=torch.float64)
    model.fit(_series())
    assert model.is_fitted and torch.isfinite(model.forecast(horizon=3)).all()


@pytest.mark.parametrize("family", ["classic", "ott", "facade"])
def test_public_fit_forecast_portable_load_and_tune_preserve_solver(family, tmp_path):
    options = dict(
        readout="lsmr", readout_kwargs={"max_iter": 200, "chunk_size": 13}, dtype=torch.float64
    )
    if family == "facade":
        model = rd.ESN(12, input_size=2, washout=3, seed=17, **options)
    else:
        factory = rd.classic_esn if family == "classic" else rd.ott_esn
        model = rd.Model(graph=factory(12, 2, 2, seed=17, **options), washout=3)
    data = _series()
    model.fit(data)
    head = next(m for m in model.graph.modules() if isinstance(m, rd.LSMRReadoutLayer))
    assert head.last_solve.n_samples == len(data) - 4
    expected = model.forecast(history=data[-20:], horizon=5)
    assert torch.isfinite(expected).all()
    path = tmp_path / f"{family}.pt"
    model.save(path)
    restored = rd.load(path)
    restored_head = next(m for m in restored.graph.modules() if isinstance(m, rd.LSMRReadoutLayer))
    assert restored_head.max_iter == 200 and restored_head.chunk_size == 13
    assert restored_head.last_solve is None
    torch.testing.assert_close(
        restored.forecast(history=data[-20:], horizon=5), expected, atol=0, rtol=0
    )
    pytest.importorskip("optuna")
    restored.tune(
        data,
        preset="quick",
        n_trials=1,
        seeds=(7, 19),
        search_space=lambda trial: {"alpha": 1e-3},
        horizon=3,
    )
    assert restored_head is not next(
        m for m in restored.graph.modules() if isinstance(m, rd.LSMRReadoutLayer)
    )
    assert len(restored.tuning_report.runs) == 2
    assert torch.isfinite(restored.forecast(horizon=3)).all()


def test_fitted_layer_inside_trainable_torch_pipeline_preserves_upstream_gradients():
    x, y = _data(73, 5)
    head = _layer(5)
    head.fit(x, y)
    encoder = torch.nn.Linear(2, 5, dtype=torch.float64)
    pipeline = torch.nn.Sequential(encoder, head)
    inputs = torch.randn(3, 8, 2, dtype=torch.float64, requires_grad=True)
    pipeline(inputs).square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert encoder.weight.grad is not None and encoder.weight.grad.norm() > 0
    assert head.weight.grad is None
    head.trainable = True
    pipeline(inputs).sum().backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()


def test_fitted_model_inside_a_larger_pipeline_preserves_encoder_and_input_gradients():
    model = rd.ESN(
        12,
        input_size=2,
        washout=3,
        readout="lsmr",
        seed=17,
        dtype=torch.float64,
        readout_kwargs={"max_iter": 200},
    ).fit(_series())
    model.reset_state()
    encoder = torch.nn.Linear(2, 2, dtype=torch.float64)
    decoder = torch.nn.Linear(2, 1, dtype=torch.float64)
    pipeline = torch.nn.Sequential(encoder, model, decoder)
    inputs = torch.randn(2, 9, 2, dtype=torch.float64, requires_grad=True)
    pipeline(inputs).square().mean().backward()
    for gradient in (inputs.grad, encoder.weight.grad, decoder.weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0


def test_documented_lsmr_workflow_executes():
    guide = Path(__file__).resolve().parents[2] / "docs/reference/lsmr-readout.md"
    source = re.search(r"```python\n(.*?)\n```", guide.read_text(), flags=re.S).group(1)
    exec(compile(source, str(guide), "exec"), {"__name__": "__main__"})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_warned_cpu_bridge_and_rejection_leave_mps_parameters_on_device():
    data = _series(torch.float32, "mps")
    head = rd.LSMRReadoutLayer(2, 2, max_iter=100).to("mps")
    with pytest.warns(RuntimeWarning, match="CPU float64"):
        head.fit(data, data.roll(-1, 0))
    assert head.last_solve.cpu_bridge and head.last_solve.solve_device == "cpu"
    assert head.weight.device.type == "mps" and head.weight.dtype == torch.float32
    assert torch.isfinite(head(data)).all()
    rejected = rd.LSMRReadoutLayer(2, 2, cpu_fallback=False).to("mps")
    with pytest.raises(RuntimeError, match="cpu_fallback=False"):
        rejected.fit(data, data)
    assert not rejected.is_fitted
