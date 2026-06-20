"""Tests for normalize_data / denormalize_data round-trips and prepare_esn_data stats."""

import pytest
import torch

from resdag.utils import denormalize_data, normalize_data, prepare_esn_data
from resdag.utils.data import (
    denormalize_data as denormalize_data_data,
)
from resdag.utils.data import (
    normalize_data as normalize_data_data,
)
from resdag.utils.data import (
    prepare_esn_data as prepare_esn_data_data,
)

METHODS = ["minmax", "standard", "noncentered", "meanpreserving"]


def _sample_data() -> torch.Tensor:
    """Two batches of a 3D signal with a non-zero offset (exercises mean terms)."""
    torch.manual_seed(0)
    return torch.randn(2, 200, 3, dtype=torch.float64) * 5.0 + 20.0


@pytest.mark.parametrize("method", METHODS)
def test_denormalize_inverts_normalize(method: str) -> None:
    """``denormalize(normalize(x)) ≈ x`` for every normalization method."""
    data = _sample_data()
    normalized, stats = normalize_data(data, method=method)
    restored = denormalize_data(normalized, method, stats)

    assert restored.shape == data.shape
    assert torch.allclose(restored, data, atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_round_trip_preserves_dtype(method: str) -> None:
    """The inverse keeps the input dtype (float32 path)."""
    data = _sample_data().to(torch.float32)
    normalized, stats = normalize_data(data, method=method)
    restored = denormalize_data(normalized, method, stats)

    assert restored.dtype == torch.float32
    assert torch.allclose(restored, data, atol=1e-4)


def test_minmax_inverse_formula() -> None:
    """Spot-check the exact minmax inverse against the documented formula."""
    data = _sample_data()
    normalized, stats = normalize_data(data, method="minmax")
    expected = (normalized + 1) / 2 * stats["range"] + stats["min"]

    assert torch.allclose(denormalize_data(normalized, "minmax", stats), expected)


def test_standard_inverse_formula() -> None:
    data = _sample_data()
    normalized, stats = normalize_data(data, method="standard")
    expected = normalized * stats["std"] + stats["mean"]

    assert torch.allclose(denormalize_data(normalized, "standard", stats), expected)


def test_noncentered_inverse_formula() -> None:
    data = _sample_data()
    normalized, stats = normalize_data(data, method="noncentered")
    expected = normalized * stats["scale"]

    assert torch.allclose(denormalize_data(normalized, "noncentered", stats), expected)


def test_meanpreserving_inverse_formula() -> None:
    data = _sample_data()
    normalized, stats = normalize_data(data, method="meanpreserving")
    expected = (normalized - stats["mean"]) * stats["maxdev"] + stats["mean"]

    assert torch.allclose(denormalize_data(normalized, "meanpreserving", stats), expected)


def test_denormalize_with_external_stats() -> None:
    """Stats fitted on one tensor invert another tensor scaled by those stats."""
    data = _sample_data()
    _, stats = normalize_data(data, method="standard")
    other = _sample_data() + 3.0
    other_norm, _ = normalize_data(other, method="standard", stats=stats)

    assert torch.allclose(denormalize_data(other_norm, "standard", stats), other, atol=1e-8)


def test_denormalize_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown normalization method"):
        denormalize_data(torch.zeros(1, 1, 1), "bogus", {})  # type: ignore[arg-type]


def test_prepare_esn_data_default_returns_five_tuple() -> None:
    """``return_stats=False`` (default) keeps the legacy 5-tuple unchanged."""
    data = _sample_data()
    result = prepare_esn_data(data, warmup_steps=20, train_steps=80, val_steps=30, normalize=True)

    assert isinstance(result, tuple)
    assert len(result) == 5


def test_prepare_esn_data_return_stats_appends_stats() -> None:
    """``return_stats=True`` appends the fitted stats dict as a 6th element."""
    data = _sample_data()
    result = prepare_esn_data(
        data,
        warmup_steps=20,
        train_steps=80,
        val_steps=30,
        normalize=True,
        norm_method="minmax",
        return_stats=True,
    )

    assert len(result) == 6
    *_, stats = result
    assert isinstance(stats, dict)
    assert set(stats) == {"min", "range"}


def test_prepare_esn_data_return_stats_none_when_not_normalizing() -> None:
    """Stats element is ``None`` when ``normalize=False`` (nothing was fitted)."""
    data = _sample_data()
    *splits, stats = prepare_esn_data(
        data, warmup_steps=20, train_steps=80, val_steps=30, return_stats=True
    )

    assert len(splits) == 5
    assert stats is None


@pytest.mark.parametrize("method", METHODS)
def test_prepare_esn_data_round_trip_via_returned_stats(method: str) -> None:
    """Stats from ``prepare_esn_data`` invert its normalized splits exactly."""
    data = _sample_data()
    warmup, train, target, f_warmup, val, stats = prepare_esn_data(
        data,
        warmup_steps=20,
        train_steps=80,
        val_steps=30,
        normalize=True,
        norm_method=method,
        return_stats=True,
    )
    assert stats is not None

    # Recompute the raw splits with the same indexing and normalization off.
    raw = prepare_esn_data(data, warmup_steps=20, train_steps=80, val_steps=30)
    raw_warmup, raw_train, raw_target, raw_f_warmup, raw_val = raw

    assert torch.allclose(denormalize_data(warmup, method, stats), raw_warmup, atol=1e-8)
    assert torch.allclose(denormalize_data(train, method, stats), raw_train, atol=1e-8)
    assert torch.allclose(denormalize_data(target, method, stats), raw_target, atol=1e-8)
    assert torch.allclose(denormalize_data(f_warmup, method, stats), raw_f_warmup, atol=1e-8)
    assert torch.allclose(denormalize_data(val, method, stats), raw_val, atol=1e-8)


def test_exports_match_across_namespaces() -> None:
    """``resdag.utils`` and ``resdag.utils.data`` expose the same callables."""
    assert denormalize_data is denormalize_data_data
    assert normalize_data is normalize_data_data
    assert prepare_esn_data is prepare_esn_data_data
