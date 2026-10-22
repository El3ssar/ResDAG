"""Independent numerical and PyTorch contracts for automatic L2 readouts."""

import io
import math
import warnings
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import pytest
import torch

import resdag as rd
from resdag.layers.readouts import AutoReadoutLayer, ReadoutSolveReport
from resdag.models._builder import _make_readout


def _data(n=80, f=7, outputs=2, dtype=torch.float64):
    generator = torch.Generator().manual_seed(173)
    x = torch.randn(n, f, dtype=dtype, generator=generator)
    y = torch.randn(n, outputs, dtype=dtype, generator=generator)
    return x, y


def _reference(x, y, alpha, bias):
    """Independent CPU64 augmented-design LAPACK solve, never a Gram solve."""
    x, y = x.cpu().double(), y.cpu().double()
    xm, ym = x.mean(0), y.mean(0)
    if bias:
        x, y = x - xm, y - ym
    if alpha:
        x = torch.cat((x, math.sqrt(alpha) * torch.eye(x.shape[1], dtype=x.dtype)))
        y = torch.cat((y, y.new_zeros(x.shape[1], y.shape[1])))
    w = torch.linalg.lstsq(x, y, driver="gelsd").solution
    return w, ym - xm @ w if bias else None


@pytest.mark.parametrize("shape", [(80, 7), (7, 80)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 1e-4, 0.3])
def test_auto_matches_independent_augmented_reference(shape, bias, alpha):
    x, y = _data(*shape)
    layer = AutoReadoutLayer(shape[1], 2, alpha=alpha, bias=bias).double()
    layer.fit(x, y)
    expected, intercept = _reference(x, y, alpha, bias)
    torch.testing.assert_close(layer.weight.T, expected, rtol=2e-9, atol=2e-11)
    if bias:
        torch.testing.assert_close(layer.bias, intercept, rtol=2e-9, atol=2e-11)
    report = layer.last_solve
    assert isinstance(report, ReadoutSolveReport)
    assert report.alpha == alpha == layer.alpha
    assert report.solve_dtype == torch.float64
    assert report.solve_device == "cpu"
    assert report.n_samples == shape[0]
    assert report.n_features == shape[1]
    assert report.relative_optimality_residual < 1e-12


@pytest.mark.parametrize("solver", ["primal", "dual", "qr", "svd"])
@pytest.mark.parametrize("shape", [(80, 7), (7, 80)])
def test_forced_method_preserves_algorithm_and_objective(solver, shape):
    x, y = _data(*shape)
    layer = AutoReadoutLayer(shape[1], 2, alpha=0.2, solver=solver).double()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        layer.fit(x, y)
    expected, intercept = _reference(x, y, 0.2, True)
    torch.testing.assert_close(layer.weight.T, expected, rtol=2e-9, atol=2e-11)
    torch.testing.assert_close(layer.bias, intercept, rtol=2e-9, atol=2e-11)
    assert layer.last_solve.solver == solver
    assert layer.last_solve.alpha == 0.2


def test_primal_and_dual_choose_smaller_system():
    for shape, expected in [((80, 7), "primal"), ((7, 80), "dual")]:
        x, y = _data(*shape)
        layer = AutoReadoutLayer(shape[1], 2, bias=False, alpha=0.2).double()
        layer.fit(x, y)
        assert layer.last_solve.solver == expected
        assert layer.last_solve.roundoff_estimate is not None


@pytest.mark.parametrize("alpha", [0.0, 1e-18, 0.3])
def test_rank_deficiency_preserves_l2_solution(alpha):
    x, y = _data(40, 4)
    x = torch.cat((x, x[:, :2], x[:, :1] * 2), 1)
    layer = AutoReadoutLayer(7, 2, alpha=alpha).double()
    if 0 < alpha < 1e-10:
        with pytest.warns(RuntimeWarning, match="coefficients may be inaccurate"):
            layer.fit(x, y)
    else:
        layer.fit(x, y)
    expected, intercept = _reference(x, y, alpha, True)
    torch.testing.assert_close(layer(x), x @ expected + intercept, rtol=1e-8, atol=1e-10)
    if alpha == 0 or alpha >= 0.3:
        torch.testing.assert_close(layer.weight.T, expected, rtol=2e-8, atol=2e-10)
    assert layer.last_solve.alpha == alpha
    if alpha < 1e-10:
        assert layer.last_solve.solver == "svd"
        assert layer.last_solve.fallback_reasons
    if alpha > 0 and alpha < 1e-10:
        assert layer.last_solve.precision_warning is not None


def test_off_diagonal_ill_conditioning_cannot_hide_behind_cholesky_diagonal():
    x = torch.tensor([[1.0, 1e6], [0.0, 1.0]], dtype=torch.float64)
    y = torch.ones(2, 1, dtype=torch.float64)
    # The Cholesky diagonal ratio is one, despite very poor conditioning.
    factor = torch.linalg.cholesky(x.T @ x)
    torch.testing.assert_close(factor.diagonal(), torch.ones(2, dtype=x.dtype))
    layer = AutoReadoutLayer(2, 1, bias=False, alpha=0).double()
    layer.fit(x, y)
    expected, _ = _reference(x, y, 0, False)
    torch.testing.assert_close(layer.weight.T, expected, rtol=2e-10, atol=2e-9)
    assert layer.last_solve.solver == "svd"
    assert any("guard" in reason for reason in layer.last_solve.fallback_reasons)


@pytest.mark.parametrize("dtype,scale", [(torch.float32, 1e-30), (torch.float64, 1e-200)])
@pytest.mark.parametrize("solver", ["auto", "primal", "qr", "svd"])
def test_small_targets_are_not_mistaken_for_convergence(dtype, scale, solver):
    x, _ = _data(40, 3, 1, dtype=dtype)
    true = torch.tensor([[1.0], [-2.0], [0.5]], dtype=dtype) * scale
    y = x @ true
    layer = AutoReadoutLayer(3, 1, bias=False, alpha=0.1, solver=solver, solve_dtype=dtype).to(
        dtype
    )
    layer.fit(x, y)
    expected, _ = _reference(x, y, 0.1, False)
    assert layer.weight.count_nonzero() == 3
    torch.testing.assert_close(
        layer.weight.T.double() / scale, expected / scale, rtol=2e-5, atol=2e-6
    )
    assert math.isfinite(layer.last_solve.relative_optimality_residual)
    assert layer.last_solve.relative_optimality_residual < 1e-5


@pytest.mark.parametrize("alpha", [0.0, 1e-6])
def test_large_finite_design_avoids_gram_overflow(alpha):
    x = torch.diag(torch.tensor([1e155, 2e155], dtype=torch.float64))
    expected = torch.tensor([[2.0], [-3.0]], dtype=x.dtype)
    layer = AutoReadoutLayer(2, 1, bias=False, alpha=alpha).double()
    layer.fit(x, x @ expected)
    torch.testing.assert_close(layer.weight.T, expected, rtol=1e-12, atol=1e-12)
    assert layer.last_solve.solver in ("qr", "svd")
    assert layer.last_solve.alpha == alpha


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("target_scale", [1e-20, 1.0, 1e20])
def test_small_design_strong_regularization_matches_high_precision(dtype, target_scale):
    # An augmented LAPACK reference itself loses relative coefficient accuracy
    # when the top data block is tiny compared with sqrt(alpha) I. Use Decimal
    # normal equations here: this particular regularized 2x2 system is benign.
    x = torch.tensor([[1, 2], [3, -4], [5, 1]], dtype=dtype) * 1e-10
    y = torch.tensor([[0.2], [-0.3], [0.4]], dtype=dtype) * target_scale
    layer = AutoReadoutLayer(2, 1, bias=False, alpha=1e8, solve_dtype=dtype).to(dtype)
    layer.fit(x, y)
    with localcontext() as context:
        context.prec = 120
        rows = [[Decimal.from_float(float(item)) for item in row] for row in x]
        values = [Decimal.from_float(float(item)) for item in y[:, 0]]
        a = sum(row[0] ** 2 for row in rows) + Decimal("1e8")
        b = sum(row[0] * row[1] for row in rows)
        d = sum(row[1] ** 2 for row in rows) + Decimal("1e8")
        rhs0 = sum(row[0] * value for row, value in zip(rows, values, strict=True))
        rhs1 = sum(row[1] * value for row, value in zip(rows, values, strict=True))
        determinant = a * d - b * b
        expected = torch.tensor(
            [
                [float((d * rhs0 - b * rhs1) / determinant)],
                [float((a * rhs1 - b * rhs0) / determinant)],
            ],
            dtype=torch.float64,
        )
    scale = float(expected.abs().max())
    torch.testing.assert_close(
        layer.weight.T.double() / scale,
        expected / scale,
        rtol=2e-5 if dtype == torch.float32 else 1e-12,
        atol=1e-7,
    )
    assert layer.last_solve.solver == "primal"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_subnormal_gram_products_do_not_erase_representable_coefficients(device):
    x = torch.eye(2, dtype=torch.float32, device=device) * 1e-30
    y = torch.tensor([[1e-20], [-2e-20]], dtype=torch.float32, device=device)
    layer = AutoReadoutLayer(2, 1, bias=False, alpha=1e-30, solve_dtype=torch.float32).to(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer.fit(x, y)
    # The exact denominator is alpha +1e-60; its correction is below float64
    # precision here, so coefficients equal the targets to working accuracy.
    torch.testing.assert_close(
        layer.weight.T.cpu().double() / 1e-20, y.cpu().double() / 1e-20, rtol=1e-5, atol=1e-6
    )
    assert layer.last_solve.solver == "svd"
    assert any("underflow" in reason for reason in layer.last_solve.fallback_reasons)
    assert layer.last_solve.alpha == 1e-30
    if device == "mps":
        assert any("CPU float64" in str(item.message) for item in caught)
        assert layer.last_solve.solve_device == "cpu"


def test_forced_qr_rejects_lossy_regularizer_to_data_scale():
    x = torch.eye(2, dtype=torch.float64) * 1e-30
    layer = AutoReadoutLayer(2, 1, bias=False, alpha=1e-30, solver="qr").double()
    with pytest.raises(RuntimeError, match="data block.*svd"):
        layer.fit(x, torch.ones(2, 1, dtype=x.dtype))
    assert not layer.is_fitted


@pytest.mark.parametrize("dtype,scale", [(torch.float32, 1e-30), (torch.float64, 1e-200)])
def test_optimality_diagnostic_cannot_underflow_to_false_zero(dtype, scale):
    x = torch.eye(2, dtype=dtype) * scale
    y = torch.ones(2, 1, dtype=dtype) * scale
    layer = AutoReadoutLayer(2, 1, bias=False, alpha=0)
    residual, _ = layer._residuals(x, y, torch.zeros(2, 1, dtype=dtype))
    # At W=0, ||X.T Y|| / (||X|| ||Y||) =1/sqrt(2), independent of scale.
    assert residual == pytest.approx(1 / math.sqrt(2), rel=1e-6)


@pytest.mark.parametrize("bias", [False, True])
def test_zero_design_and_targets_have_finite_reports(bias):
    layer = AutoReadoutLayer(3, 2, alpha=0, bias=bias).double()
    layer.fit(torch.zeros(5, 3, dtype=torch.float64), torch.zeros(5, 2, dtype=torch.float64))
    assert layer.weight.count_nonzero() == 0
    assert layer.last_solve.relative_optimality_residual == 0
    assert layer.last_solve.relative_prediction_residual == 0


def test_forced_unsafe_normal_equations_warn_without_switching():
    x = torch.tensor([[1.0, 1e6], [0.0, 1.0]], dtype=torch.float64)
    layer = AutoReadoutLayer(2, 1, bias=False, alpha=0, solver="primal").double()
    with pytest.warns(RuntimeWarning, match="forced primal"):
        layer.fit(x, torch.ones(2, 1, dtype=x.dtype))
    assert layer.last_solve.solver == "primal"
    assert layer.alpha == 0


def test_forced_rank_deficient_qr_raises_without_switching_or_publishing():
    x = torch.ones(4, 3, dtype=torch.float64)
    layer = AutoReadoutLayer(3, 1, bias=False, alpha=0, solver="qr").double()
    before = layer.weight.clone()
    with pytest.raises(RuntimeError, match="rank deficient"):
        layer.fit(x, torch.ones(4, 1, dtype=x.dtype))
    torch.testing.assert_close(layer.weight, before)
    assert not layer.is_fitted
    assert layer.last_solve is None


def test_forced_wide_unregularized_qr_explains_svd_option():
    x, y = _data(3, 8)
    layer = AutoReadoutLayer(8, 2, bias=False, alpha=0, solver="qr").double()
    with pytest.raises(RuntimeError, match="underdetermined.*svd"):
        layer.fit(x, y)


@pytest.mark.parametrize("name", ["alpha", "rcond"])
@pytest.mark.parametrize("value", [-1, math.inf, -math.inf, math.nan])
def test_invalid_numeric_options_rejected_at_construction_and_live_solve(name, value):
    with pytest.raises(ValueError, match=name):
        AutoReadoutLayer(3, 2, **{name: value})
    layer = AutoReadoutLayer(3, 2)
    setattr(layer, name, value)
    x, y = _data(8, 3)
    with pytest.raises(ValueError, match=name):
        layer.solve(x, y)


@pytest.mark.parametrize(
    "options", [{"solver": "wrong"}, {"solve_dtype": torch.float16}, {"cpu_fallback": "yes"}]
)
def test_other_invalid_options_are_actionable(options):
    with pytest.raises((ValueError, TypeError), match=next(iter(options))):
        AutoReadoutLayer(3, 2, **options)


def test_rcond_zero_keeps_small_nonzero_unregularized_singular_values():
    x = torch.diag(torch.tensor([1.0, 1e-18], dtype=torch.float64))
    y = torch.tensor([[2.0], [3e-18]], dtype=x.dtype)
    layer = AutoReadoutLayer(2, 1, bias=False, alpha=0, solver="svd", rcond=0).double()
    layer.fit(x, y)
    torch.testing.assert_close(layer.weight, torch.tensor([[2.0, 3.0]], dtype=x.dtype))


def test_fit_keeps_parameters_gradients_and_weights_only_state_dict():
    x, y = _data(20, 4)
    layer = AutoReadoutLayer(4, 2, trainable=True).double()
    weight, bias = layer.weight, layer.bias
    layer.fit(x.reshape(2, 10, 4), y.reshape(2, 10, 2))
    assert layer.last_solve.n_samples == 20 and layer.last_solve.n_features == 4
    assert layer.weight is weight and layer.bias is bias
    assert layer.weight.requires_grad and layer.bias.requires_grad
    source = x.clone().requires_grad_()
    layer(source).square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert layer.weight.grad is not None
    buffer = io.BytesIO()
    torch.save(layer.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    restored = AutoReadoutLayer(4, 2, trainable=True).double()
    restored.load_state_dict(state)
    assert restored.is_fitted
    assert restored.last_solve is None
    torch.testing.assert_close(layer(x), restored(x))
    with pytest.raises(FrozenInstanceError):
        layer.last_solve.alpha = 100


def test_solve_does_not_publish_parameters_or_fitted_flag():
    layer = AutoReadoutLayer(3, 2).double()
    before = {key: value.clone() for key, value in layer.state_dict().items()}
    coefs, intercept = layer.solve(*_data(15, 3))
    assert coefs.shape == (3, 2) and intercept.shape == (2,)
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert not layer.is_fitted
    assert layer.last_solve is not None


def test_invalid_refit_preserves_existing_parameters_report_and_fitted_flag():
    layer = AutoReadoutLayer(3, 2).double()
    x, y = _data(15, 3)
    layer.fit(x, y)
    before = {key: value.clone() for key, value in layer.state_dict().items()}
    report = layer.last_solve
    x[0, 0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        layer.fit(x, y)
    assert layer.last_solve is report
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert layer.is_fitted


def test_exports_registry_and_custom_kwargs():
    assert rd.AutoReadoutLayer is AutoReadoutLayer
    assert rd.ReadoutSolveReport is ReadoutSolveReport
    layer = _make_readout(
        "auto",
        in_features=3,
        out_features=2,
        alpha=0.3,
        bias=False,
        name="forecast",
        readout_kwargs={"solver": "svd", "rcond": 0},
    )
    assert isinstance(layer, AutoReadoutLayer)
    assert layer.name == "forecast" and layer.solver == "svd" and layer.rcond == 0


_MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")


@_MPS
def test_real_mps_safe_float32_fit_is_native_without_cpu_bridge():
    x, y = _data(40, 3, dtype=torch.float32)
    x, y = x.to("mps"), y.to("mps")
    layer = AutoReadoutLayer(3, 2, alpha=1000, cpu_fallback=False).to("mps")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer.fit(x, y)
    expected, intercept = _reference(x, y, 1000, True)
    torch.testing.assert_close(layer.weight.T.cpu().double(), expected, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(layer.bias.cpu().double(), intercept, rtol=1e-4, atol=1e-7)
    assert layer.last_solve.solve_device.startswith("mps")
    assert layer.last_solve.solve_dtype == torch.float32
    assert layer.last_solve.solver == "primal"
    assert not caught


@_MPS
@pytest.mark.parametrize("solver", ["auto", "qr", "svd"])
def test_real_mps_stable_bridge_is_explicit_and_preserves_device_and_method(solver):
    x = torch.tensor([[1.0, 1e6], [0.0, 1.0]], dtype=torch.float32, device="mps")
    y = torch.ones(2, 1, device="mps")
    # QR needs a full-rank moderate-conditioning design when explicitly forced.
    if solver == "qr":
        x = torch.eye(2, device="mps")
    layer = AutoReadoutLayer(2, 1, bias=False, alpha=0, solver=solver).to("mps")
    weight = layer.weight
    with pytest.warns(RuntimeWarning, match="CPU float64"):
        layer.fit(x, y)
    assert layer.weight is weight and layer.weight.device.type == "mps"
    assert layer.last_solve.solve_device == "cpu"
    assert layer.last_solve.solve_dtype == torch.float64
    assert layer.last_solve.fallback_reasons
    assert layer.last_solve.alpha == 0
    if solver != "auto":
        assert layer.last_solve.solver == solver
    expected, _ = _reference(x, y, 0, False)
    torch.testing.assert_close(layer.weight.T.cpu().double(), expected, rtol=1e-5, atol=1e-7)


@_MPS
@pytest.mark.parametrize("options", [{"solver": "svd"}, {"solve_dtype": torch.float64}])
def test_mps_cpu_bridge_can_be_disabled(options):
    layer = AutoReadoutLayer(3, 2, cpu_fallback=False, **options).to("mps")
    x, y = _data(10, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="cpu_fallback=True"):
        layer.fit(x.to("mps"), y.to("mps"))
    assert not layer.is_fitted and layer.last_solve is None


@_MPS
def test_unimplemented_native_operation_bridges_without_forced_method_switch(monkeypatch):
    original = AutoReadoutLayer._gram_solve

    def unavailable_on_mps(self, x, y, *, dual):
        if x.device.type == "mps":
            raise NotImplementedError("simulated backend gap")
        return original(self, x, y, dual=dual)

    monkeypatch.setattr(AutoReadoutLayer, "_gram_solve", unavailable_on_mps)
    x, y = _data(10, 3, dtype=torch.float32)
    layer = AutoReadoutLayer(3, 2, solver="primal").to("mps")
    with pytest.warns(RuntimeWarning, match="CPU float64"):
        layer.fit(x.to("mps"), y.to("mps"))
    assert layer.last_solve.solver == "primal" and layer.last_solve.solve_device == "cpu"
