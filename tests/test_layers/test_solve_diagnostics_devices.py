"""Accuracy diagnostics retain supported precision and do not hide bad solves."""

import warnings

import pytest
import torch

from resdag import RidgeReadoutLayer, SVDReadoutLayer
from resdag.layers.readouts._solve_utils import (
    _normal_equations_residual,
    _warn_on_precision_loss,
)


@pytest.fixture(params=["cpu", "mps"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Native MPS unavailable")
    return request.param


@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
def test_scaled_residual_agrees_with_independent_float64_reference(device, scale):
    gram = torch.eye(4, dtype=torch.float32) * scale
    rhs = torch.ones(4, 3, dtype=torch.float32) * scale
    weights = torch.full((4, 3), 1.1, dtype=torch.float32)
    residual = gram.double() @ weights.double() - rhs.double()
    expected = float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(rhs.double()))
    actual = _normal_equations_residual(gram.to(device), weights.to(device), rhs.to(device), 0)
    assert actual == pytest.approx(expected, rel=5e-6, abs=1e-7)


def test_zero_rhs_distinguishes_exact_and_invalid_solutions(device):
    gram = torch.eye(3, device=device)
    rhs = torch.zeros(3, 1, device=device)
    assert _normal_equations_residual(gram, torch.zeros_like(rhs), rhs, 0) == 0
    assert _normal_equations_residual(gram, torch.ones_like(rhs), rhs, 0) == float("inf")


def test_nonfinite_residual_cannot_suppress_accuracy_warning(device):
    gram = torch.eye(3, device=device)
    weights = torch.full((3, 1), float("inf"), device=device)
    rhs = torch.ones(3, 1, device=device)
    assert _normal_equations_residual(gram, weights, rhs, 0) == float("inf")
    with pytest.warns(RuntimeWarning, match="residual=inf") as caught:
        _warn_on_precision_loss(
            gram=gram,
            coefs=weights,
            rhs=rhs,
            alpha_eff=0,
            cond_estimate=None,
            gram_dtype=torch.float32,
            layer_name="diagnostic-test",
        )
    if device == "mps":
        assert "Fit on CPU" in str(caught[0].message)
        assert "MPS residual is evaluated in float32" in str(caught[0].message)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS unavailable")
def test_native_mps_lu_ridge_fit_and_diagnostic_match_cpu_svd_reference():
    data = torch.randn(50, 4, generator=torch.Generator().manual_seed(33)) + 5
    target = data @ torch.tensor([[1.0, 2.0], [-1.0, 0.3], [2.0, -1.0], [0.5, 3.0]]) + 3
    expected = SVDReadoutLayer(4, 2, alpha=0.1).double()
    expected.fit(data.double(), target.double())
    native = RidgeReadoutLayer(
        4, 2, alpha=0.1, solver="solve", use_float64=False, gram_dtype=torch.float32
    ).to("mps")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        native.fit(data.to("mps"), target.to("mps"))
    assert native.is_fitted and native.weight.device.type == "mps"
    torch.testing.assert_close(native.weight.cpu().double(), expected.weight, atol=1e-5, rtol=1e-5)
    observed = native(data.to("mps"))
    torch.testing.assert_close(
        observed.cpu().double(), expected(data.double()), atol=1e-4, rtol=1e-5
    )
