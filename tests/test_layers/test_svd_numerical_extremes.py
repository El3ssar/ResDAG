"""Independent reference checks for finite SVD ridge filters and configuration."""

from decimal import Decimal, localcontext

import pytest
import torch

from resdag.layers import SVDReadoutLayer
from resdag.layers.readouts.svd_readout import _ridge_filter


def _decimal_filter(values: torch.Tensor, alpha: float) -> torch.Tensor:
    with localcontext() as context:
        context.prec = 120
        penalty = Decimal.from_float(alpha)
        result = []
        for value in values.tolist():
            singular = Decimal.from_float(value)
            result.append(float(singular / (singular * singular + penalty)))
    return torch.tensor(result, dtype=values.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alpha", [1e-300, 1e-40, 1e-6, 1e20, 1e80, 1e308])
def test_filter_matches_decimal_across_working_dtype_range(dtype, alpha):
    info = torch.finfo(dtype)
    values = torch.tensor(
        [0, info.tiny * info.eps, info.tiny, 1e-20, 0.125, 1, 2, 1e20, info.max / 2],
        dtype=dtype,
    )
    result = _ridge_filter(values, alpha)
    expected = _decimal_filter(values, alpha)
    assert result.dtype == dtype
    assert result.device == values.device
    assert not result.isnan().any()
    torch.testing.assert_close(result, expected, rtol=16 * info.eps, atol=4 * info.tiny * info.eps)


@pytest.mark.parametrize(
    "dtype,integer_dtype", [(torch.float32, torch.int32), (torch.float64, torch.int64)]
)
@pytest.mark.parametrize("alpha", [2.8679139182966226e-5, 1e-5, 0.0314159, 0.125, 0.3, 0.9, 2.0])
def test_subnormal_filter_does_not_amplify_intermediate_rounding(dtype, integer_dtype, alpha):
    smallest = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
    values = torch.tensor([0, 1, 2, 3, 7, 31, 101, 1009], dtype=dtype) * smallest
    expected = _decimal_filter(values, alpha)
    result = _ridge_filter(values, alpha)
    # An absolute decimal tolerance would hide amplified subnormal rounding.
    ulps = (result.view(integer_dtype) - expected.view(integer_dtype)).abs()
    assert int(ulps.max()) <= 1


@pytest.mark.parametrize("dtype,target_scale", [(torch.float32, 1e30), (torch.float64, 1e300)])
def test_subnormal_state_and_large_target_keep_only_final_filter_rounding(dtype, target_scale):
    info = torch.finfo(dtype)
    states = torch.tensor([[info.tiny * info.eps]], dtype=dtype)
    targets = torch.tensor([[target_scale]], dtype=dtype)
    alpha = 2.8679139182966226e-5
    layer = SVDReadoutLayer(1, 1, bias=False, alpha=alpha, use_float64=False).to(dtype)
    layer.fit(states, targets)
    with localcontext() as context:
        context.prec = 120
        singular = Decimal.from_float(states.item())
        target = Decimal.from_float(targets.item())
        penalty = Decimal.from_float(alpha)
        expected = float(singular * target / (singular * singular + penalty))
    # The final filter can itself be subnormal. Its unavoidable one-ULP
    # rounding is multiplied by the target; intermediate rounding must not
    # add tens of those ULPs before that multiplication.
    tolerance = 2 * (info.tiny * info.eps) * targets.item() + 16 * info.eps * abs(expected)
    assert abs(layer.weight.item() - expected) <= tolerance
    assert layer.alpha == alpha


@pytest.mark.parametrize("dtype,small", [(torch.float32, 1e-30), (torch.float64, 1e-250)])
def test_zero_alpha_and_rcond_retain_small_nonzero_singular_values(dtype, small):
    states = torch.diag(torch.tensor([small, 1.0, 0.0], dtype=dtype))
    coefficients = torch.tensor([[2.0], [-3.0], [4.0]], dtype=dtype)
    layer = SVDReadoutLayer(3, 1, bias=False, alpha=0, rcond=0, use_float64=False).to(dtype)
    layer.fit(states, states @ coefficients)
    torch.testing.assert_close(layer.weight, torch.tensor([[2.0, -3.0, 0.0]], dtype=dtype))
    assert layer.alpha == layer.rcond == 0


@pytest.mark.parametrize(
    "dtype,small,large", [(torch.float32, 1e-30, 1e20), (torch.float64, 1e-250, 1e100)]
)
def test_unrepresentable_final_coefficient_is_rejected(dtype, small, large):
    layer = SVDReadoutLayer(1, 1, bias=False, alpha=0, rcond=0, use_float64=False).to(dtype)
    before = layer.weight.detach().clone()
    with pytest.raises(RuntimeError, match="non-finite coefficients"):
        layer.fit(torch.tensor([[small]], dtype=dtype), torch.tensor([[large]], dtype=dtype))
    torch.testing.assert_close(layer.weight, before, rtol=0, atol=0)
    assert not layer.is_fitted


@pytest.mark.parametrize(
    "dtype,scale,alpha",
    [
        (torch.float32, 1e20, 1e-6),
        (torch.float64, 1e155, 1e-6),
        (torch.float32, 1e-20, 1e-40),
        (torch.float64, 1e-155, 1e-308),
    ],
)
def test_extreme_diagonal_fit_agrees_with_exact_scalar_ridge(dtype, scale, alpha):
    diagonal = torch.tensor([scale, 2 * scale, 0], dtype=dtype)
    states = torch.diag(diagonal)
    truth = torch.tensor([[2.0], [-3.0], [4.0]], dtype=dtype)
    targets = states @ truth
    layer = SVDReadoutLayer(3, 1, bias=False, alpha=alpha, use_float64=False).to(dtype)
    weight_id = id(layer.weight)
    layer.fit(states, targets)

    with localcontext() as context:
        context.prec = 120
        penalty = Decimal.from_float(alpha)
        expected = []
        for singular, target in zip(diagonal.tolist(), targets[:, 0].tolist()):
            singular = Decimal.from_float(singular)
            target = Decimal.from_float(target)
            expected.append(float(singular * target / (singular * singular + penalty)))
    reference = torch.tensor([expected], dtype=dtype)
    info = torch.finfo(dtype)
    torch.testing.assert_close(layer.weight, reference, rtol=64 * info.eps, atol=0)
    assert torch.isfinite(layer(states)).all()
    assert id(layer.weight) == weight_id
    assert layer.is_fitted
    assert layer.alpha == alpha


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 1e-4, 0.3])
def test_ordinary_fit_matches_independent_augmented_least_squares(dtype, bias, alpha):
    generator = torch.Generator().manual_seed(347)
    states = torch.randn(40, 6, dtype=dtype, generator=generator)
    states[:, -1] = states[:, 0]  # Exact rank deficiency in either parameter dtype.
    targets = torch.randn(40, 2, dtype=dtype, generator=generator)
    targets += torch.tensor([2.0, -3.0], dtype=dtype)
    layer = SVDReadoutLayer(6, 2, bias=bias, alpha=alpha).to(dtype)
    layer.fit(states, targets)

    design = states.double()
    if bias:
        design = torch.cat((design, torch.ones(40, 1, dtype=torch.float64)), dim=1)
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * alpha**0.5
    if bias:
        regularizer[-1, -1] = 0
    augmented = torch.cat((design, regularizer), dim=0)
    labels = torch.cat((targets.double(), torch.zeros(design.shape[1], 2)), dim=0)
    reference = torch.linalg.lstsq(augmented, labels, driver="gelsd", rcond=1e-15).solution
    expected = design @ reference
    torch.testing.assert_close(
        layer(states).double(),
        expected,
        rtol=5e-6 if dtype == torch.float32 else 1e-10,
        atol=5e-6 if dtype == torch.float32 else 1e-10,
    )


@pytest.mark.parametrize("name", ["alpha", "rcond"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_invalid_regularization_is_rejected_at_construction(name, value):
    with pytest.raises(ValueError, match=rf"{name} must be non-negative and finite"):
        SVDReadoutLayer(2, 1, **{name: value})


@pytest.mark.parametrize("method", ["fit", "solve"])
@pytest.mark.parametrize("name", ["alpha", "rcond"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_invalid_live_configuration_preserves_existing_fit(method, name, value):
    states = torch.eye(2, dtype=torch.float64)
    targets = torch.tensor([[2.0], [-3.0]], dtype=torch.float64)
    layer = SVDReadoutLayer(2, 1, bias=False).double()
    layer.fit(states, targets)
    before = {key: value.clone() for key, value in layer.state_dict().items()}
    parameter_id = id(layer.weight)
    setattr(layer, name, value)
    with pytest.raises(ValueError, match=rf"{name} must be non-negative and finite"):
        getattr(layer, method)(states, targets)
    for key, expected in before.items():
        torch.testing.assert_close(layer.state_dict()[key], expected, rtol=0, atol=0)
    assert id(layer.weight) == parameter_id
    assert layer.is_fitted


@pytest.mark.parametrize("alpha", [0.0, 1e-6])
def test_zero_design_has_zero_finite_solution_and_solve_does_not_publish(alpha):
    layer = SVDReadoutLayer(3, 2, bias=False, alpha=alpha).double()
    states = torch.zeros(5, 3, dtype=torch.float64)
    targets = torch.ones(5, 2, dtype=torch.float64)
    before = layer.weight.detach().clone()
    coefficients, intercept = layer.solve(states, targets)
    torch.testing.assert_close(coefficients, torch.zeros_like(coefficients), rtol=0, atol=0)
    assert intercept is None
    torch.testing.assert_close(layer.weight, before, rtol=0, atol=0)
    assert not layer.is_fitted


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_filter_preserves_native_mps_dtype_and_device():
    values = torch.tensor([0.0, 1e-3, 1.0, 1e20], dtype=torch.float32)
    result = _ridge_filter(values.to("mps"), 1e-6)
    assert result.device.type == "mps"
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.cpu(), _decimal_filter(values, 1e-6))
