"""Regression contracts shared by batch and streaming algebraic readouts."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from resdag.layers.readouts import (
    CGReadoutLayer,
    IncrementalRidgeReadoutLayer,
    PinvReadoutLayer,
    RidgeReadoutLayer,
    RLSReadout,
    SVDReadoutLayer,
)


@pytest.mark.parametrize(
    "readout_cls", [RidgeReadoutLayer, IncrementalRidgeReadoutLayer, RLSReadout]
)
@pytest.mark.parametrize(
    "state_shape,target_shape,match",
    [
        ((8,), (8, 1), "2D.*3D"),
        ((8, 2), (8,), "2D.*3D"),
        ((0, 2), (0, 1), "at least one sample"),
        ((2, 4, 2), (4, 2, 1), "batch and time"),
    ],
)
def test_fit_rejects_invalid_sample_axes(readout_cls, state_shape, target_shape, match) -> None:
    readout = readout_cls(2, 1)
    with pytest.raises(ValueError, match=match):
        readout.fit(torch.randn(state_shape), torch.randn(target_shape))
    assert not readout.is_fitted


@pytest.mark.parametrize(
    "readout_cls", [RidgeReadoutLayer, IncrementalRidgeReadoutLayer, RLSReadout]
)
def test_streaming_fit_rejects_invalid_axes_before_accumulating(readout_cls) -> None:
    if readout_cls is RidgeReadoutLayer:
        return
    readout = readout_cls(2, 1)
    with pytest.raises(ValueError, match="batch and time"):
        readout.partial_fit(torch.randn(2, 4, 2), torch.randn(4, 2, 1))
    assert readout.n_seen == 0


@pytest.mark.parametrize(
    "readout_cls", [RidgeReadoutLayer, CGReadoutLayer, SVDReadoutLayer, PinvReadoutLayer]
)
def test_targets_can_use_a_different_float_dtype(readout_cls) -> None:
    x = torch.randn(30, 2)
    y = (x.double() @ torch.tensor([[2.0], [-3.0]], dtype=torch.float64)) + 1.0
    readout = readout_cls(2, 1, gram_dtype=torch.float32)
    readout.fit(x, y)
    torch.testing.assert_close(readout(x).double(), y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bias", [False, True])
def test_zero_design_with_no_regularization_produces_finite_solution(bias: bool) -> None:
    readout = RidgeReadoutLayer(3, 1, alpha=0, bias=bias)
    x = torch.zeros(8, 3)
    y = torch.full((8, 1), 2.0)
    with pytest.warns(RuntimeWarning, match="recovered a finite solution"):
        readout.fit(x, y)
    expected = y if bias else torch.zeros_like(y)
    torch.testing.assert_close(readout(x), expected)


@pytest.mark.parametrize("readout_cls", [SVDReadoutLayer, PinvReadoutLayer])
def test_wide_decomposition_solvers_do_not_materialize_a_feature_gram(readout_cls) -> None:
    n_samples, n_features = 12, 128

    class RejectQuadraticFeatureStorage(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor):
                assert result.shape != (n_features, n_features), "unneeded quadratic allocation"
            return result

    x = torch.randn(n_samples, n_features, dtype=torch.float64)
    y = torch.randn(n_samples, 2, dtype=torch.float64)
    readout = readout_cls(n_features, 2).double()
    with RejectQuadraticFeatureStorage():
        readout.fit(x, y)
    torch.testing.assert_close(readout(x), y, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "readout_cls", [RidgeReadoutLayer, IncrementalRidgeReadoutLayer, RLSReadout]
)
def test_finite_solve_cannot_publish_weights_overflowing_parameter_dtype(readout_cls) -> None:
    torch.manual_seed(8)
    x = torch.randn(24, 2, dtype=torch.float64)
    y = x[:, :1] * 1e40
    readout = readout_cls(2, 1)
    with pytest.raises(RuntimeError, match="parameter dtype"):
        readout.fit(x, y)
    assert not readout.is_fitted
    assert torch.isfinite(readout.weight).all()
