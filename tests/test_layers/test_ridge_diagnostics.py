"""Actionable ridge diagnostics without changing solves or inventing accuracy claims."""

import warnings

import pytest
import torch

from resdag.layers.readouts import _solve_utils


def _diagnose(gram, coefs, rhs, *, gram_dtype, cond_estimate=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        _solve_utils._warn_on_precision_loss(
            gram=gram,
            coefs=coefs,
            rhs=rhs,
            alpha_eff=0.0,
            cond_estimate=cond_estimate,
            gram_dtype=gram_dtype,
            layer_name="RidgeReadoutLayer('forecast')",
        )
    return caught


@pytest.mark.parametrize("gram_dtype", [None, torch.float64])
def test_float64_rounding_sensitivity_does_not_recommend_existing_precision(gram_dtype):
    # Binary powers have an exact analytical Cholesky solve. Its tiny residual
    # coexists with a large condition estimate; neither is a forecast error.
    gram = torch.diag(torch.tensor([1.0, 2.0**-48], dtype=torch.float64))
    rhs = torch.tensor([[1.0], [2.0**-48]], dtype=torch.float64)
    with pytest.warns(RuntimeWarning) as caught:
        coefs = _solve_utils.solve_ridge(gram, rhs, 0.0, layer_name="ridge", gram_dtype=gram_dtype)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "already use float64" in message
    assert "gram_dtype=torch.float64" not in message
    assert "use_float64=True" not in message
    assert "will differ from the float64 result" not in message
    assert "not a measured prediction error" in message
    assert "est. precision loss=6.25e-02 (> threshold=0.001)" in message
    assert "normal-equations residual=0.00e+00 (<= threshold=0.001)" in message
    torch.testing.assert_close(coefs, torch.ones_like(coefs), rtol=0, atol=0)


@pytest.mark.parametrize(
    "formation_dtype,solve_dtype,formation_remedy,solve_remedy",
    [
        (torch.float32, torch.float32, True, True),
        (torch.float32, torch.float64, True, False),
        (torch.float64, torch.float32, False, True),
        (torch.float64, torch.float64, False, False),
    ],
)
def test_residual_only_failure_recommends_only_missing_precision(
    formation_dtype, solve_dtype, formation_remedy, solve_remedy
):
    # A perfectly conditioned identity Gram with deliberately wrong coefficients
    # tests residual reporting without falsely diagnosing an ill-conditioned Gram.
    gram = torch.eye(2, dtype=solve_dtype)
    coefs = torch.zeros(2, 1, dtype=solve_dtype)
    rhs = torch.ones_like(coefs)
    snapshots = [value.clone() for value in (gram, coefs, rhs)]
    caught = _diagnose(gram, coefs, rhs, gram_dtype=formation_dtype)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "RidgeReadoutLayer('forecast')" in message
    assert "normal-equations residual=1.00e+00 (> threshold=0.001)" in message
    assert "large normal-equations residual" in message
    assert "condition-based estimate indicates" not in message
    assert "Gram matrix is ill-conditioned" not in message
    assert ("gram_dtype=torch.float64" in message) is formation_remedy
    assert ("use_float64=True" in message) is solve_remedy
    assert "SVDReadoutLayer" in message
    for actual, expected in zip((gram, coefs, rhs), snapshots):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_large_residual_labels_a_small_precision_estimate_as_passing():
    gram = torch.eye(2, dtype=torch.float64)
    rhs = torch.ones(2, 1, dtype=torch.float64)
    caught = _diagnose(
        gram, torch.zeros_like(rhs), rhs, gram_dtype=torch.float64, cond_estimate=1.0
    )
    message = str(caught[0].message)
    assert "est. precision loss=2.22e-16 (<= threshold=0.001)" in message
    assert "normal-equations residual=1.00e+00 (> threshold=0.001)" in message
    assert "condition-based estimate indicates" not in message


def test_both_failed_diagnostics_are_identified_in_one_warning():
    gram = torch.diag(torch.tensor([1.0, 2.0**-48], dtype=torch.float64))
    rhs = torch.tensor([[1.0], [2.0**-48]], dtype=torch.float64)
    caught = _diagnose(
        gram,
        torch.zeros_like(rhs),
        rhs,
        gram_dtype=torch.float64,
        cond_estimate=2.0**48,
    )
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "est. precision loss=6.25e-02 (> threshold=0.001)" in message
    assert "normal-equations residual=1.00e+00 (> threshold=0.001)" in message
    assert "condition-based estimate indicates" in message
    assert "large normal-equations residual" in message


def test_healthy_diagnostics_remain_silent():
    gram = torch.eye(2, dtype=torch.float64)
    rhs = torch.ones(2, 1, dtype=torch.float64)
    assert not _diagnose(gram, rhs, rhs, gram_dtype=torch.float64, cond_estimate=1.0)


def test_diagnostics_at_exact_threshold_remain_silent(monkeypatch):
    # Binary powers avoid a decimal-rounding boundary. The existing rule is
    # strictly greater than the threshold, for both independent diagnostics.
    threshold = 2.0**-10
    monkeypatch.setattr(_solve_utils, "_RESIDUAL_WARN_TOL", threshold)
    gram = torch.ones(1, 1, dtype=torch.float64)
    rhs = torch.ones_like(gram)
    coefs = torch.full_like(rhs, 1 + threshold)
    caught = _diagnose(
        gram,
        coefs,
        rhs,
        gram_dtype=torch.float64,
        cond_estimate=threshold / torch.finfo(torch.float64).eps,
    )
    assert not caught


def test_nonfinite_residual_is_named_without_inventing_a_condition_diagnosis():
    gram = torch.eye(2, dtype=torch.float64)
    rhs = torch.ones(2, 1, dtype=torch.float64)
    caught = _diagnose(gram, torch.full_like(rhs, float("inf")), rhs, gram_dtype=torch.float64)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "normal-equations residual=inf (> threshold=0.001)" in message
    assert "condition-based estimate indicates" not in message
    assert "will differ" not in message


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_diagnostic_retains_cpu_float64_remedy_and_residual_qualification():
    gram = torch.eye(2, dtype=torch.float32, device="mps")
    rhs = torch.ones(2, 1, dtype=torch.float32, device="mps")
    caught = _diagnose(gram, torch.zeros_like(rhs), rhs, gram_dtype=torch.float32)
    message = str(caught[0].message)
    assert "Fit on CPU with float64 Gram formation and solving" in message
    assert "MPS residual is evaluated in float32" in message
    assert "gram_dtype=torch.float64" not in message
