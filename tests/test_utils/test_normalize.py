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


# ---------------------------------------------------------------------------
# Forward-transform reference values (hand-computed against a tiny tensor)
# ---------------------------------------------------------------------------


def _reference_data() -> torch.Tensor:
    """A small, fully hand-traceable (1, 4, 2) tensor.

    Column 0 spans ``[1, 7]`` (mean 4, max-abs 7); column 1 spans ``[-3, 9]``
    (mean 3, max-abs 9). The asymmetry around zero exercises the min/range,
    mean/std and mean-restoring branches distinctly.
    """
    return torch.tensor(
        [[[1.0, -3.0], [3.0, 1.0], [5.0, 5.0], [7.0, 9.0]]],
        dtype=torch.float64,
    )


def test_minmax_reference_values() -> None:
    """``minmax`` maps the per-feature min->-1 and max->+1 linearly."""
    data = _reference_data()
    normalized, stats = normalize_data(data, method="minmax")

    # col0: min=1, range=6 -> 2*(x-1)/6 - 1 ; col1: min=-3, range=12
    expected = torch.tensor(
        [
            [
                [2 * (1.0 - 1.0) / 6 - 1, 2 * (-3.0 + 3.0) / 12 - 1],
                [2 * (3.0 - 1.0) / 6 - 1, 2 * (1.0 + 3.0) / 12 - 1],
                [2 * (5.0 - 1.0) / 6 - 1, 2 * (5.0 + 3.0) / 12 - 1],
                [2 * (7.0 - 1.0) / 6 - 1, 2 * (9.0 + 3.0) / 12 - 1],
            ]
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(normalized, expected)
    assert torch.allclose(stats["min"].flatten(), torch.tensor([1.0, -3.0], dtype=torch.float64))
    assert torch.allclose(stats["range"].flatten(), torch.tensor([6.0, 12.0], dtype=torch.float64))
    # Range is exactly [-1, 1] per feature.
    assert torch.allclose(normalized.amin(dim=(0, 1)), torch.full((2,), -1.0, dtype=torch.float64))
    assert torch.allclose(normalized.amax(dim=(0, 1)), torch.full((2,), 1.0, dtype=torch.float64))


def test_standard_reference_values() -> None:
    """``standard`` subtracts the mean and divides by the (unbiased) std."""
    data = _reference_data()
    normalized, stats = normalize_data(data, method="standard")

    expected = (data - data.mean(dim=(0, 1), keepdim=True)) / data.std(dim=(0, 1), keepdim=True)
    assert torch.allclose(normalized, expected)
    assert torch.allclose(stats["mean"].flatten(), torch.tensor([4.0, 3.0], dtype=torch.float64))
    # Output has ~zero mean and ~unit (unbiased) std per feature.
    assert torch.allclose(
        normalized.mean(dim=(0, 1)), torch.zeros(2, dtype=torch.float64), atol=1e-12
    )
    assert torch.allclose(normalized.std(dim=(0, 1)), torch.ones(2, dtype=torch.float64))


def test_noncentered_reference_values() -> None:
    """``noncentered`` divides by the per-feature max-abs, leaving zero fixed."""
    data = _reference_data()
    normalized, stats = normalize_data(data, method="noncentered")

    # col0 max-abs = 7, col1 max-abs = 9
    expected = data / torch.tensor([7.0, 9.0], dtype=torch.float64)
    assert torch.allclose(normalized, expected)
    assert torch.allclose(stats["scale"].flatten(), torch.tensor([7.0, 9.0], dtype=torch.float64))
    # Max absolute value is exactly 1 per feature; zero stays zero.
    assert torch.allclose(normalized.abs().amax(dim=(0, 1)), torch.ones(2, dtype=torch.float64))


def test_meanpreserving_reference_values() -> None:
    """``meanpreserving`` scales deviations to [-1, 1] then restores the mean."""
    data = _reference_data()
    normalized, stats = normalize_data(data, method="meanpreserving")

    mean = data.mean(dim=(0, 1), keepdim=True)
    maxdev = (data - mean).abs().amax(dim=(0, 1), keepdim=True)
    expected = (data - mean) / maxdev + mean
    assert torch.allclose(normalized, expected)
    assert torch.allclose(stats["mean"].flatten(), torch.tensor([4.0, 3.0], dtype=torch.float64))
    assert torch.allclose(stats["maxdev"].flatten(), torch.tensor([3.0, 6.0], dtype=torch.float64))
    # The mean is preserved exactly by construction.
    assert torch.allclose(normalized.mean(dim=(0, 1)), mean.flatten())


# ---------------------------------------------------------------------------
# Zero-variance / zero-range edge cases (the guard branches)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_zero_variance_is_finite(method: str) -> None:
    """A constant feature must not divide by zero (the ``== 0 -> 1.0`` guards)."""
    const_value = 5.0
    data = torch.full((1, 10, 2), const_value, dtype=torch.float64)
    normalized, stats = normalize_data(data, method=method)

    assert torch.isfinite(normalized).all()

    if method == "minmax":
        # range guarded to 1 -> 2*(x - x)/1 - 1 == -1 everywhere.
        assert torch.allclose(stats["range"], torch.ones_like(stats["range"]))
        assert torch.allclose(normalized, torch.full_like(normalized, -1.0))
    elif method == "standard":
        # std guarded to 1 -> (x - x)/1 == 0 everywhere.
        assert torch.allclose(stats["std"], torch.ones_like(stats["std"]))
        assert torch.allclose(normalized, torch.zeros_like(normalized))
    elif method == "noncentered":
        # scale = |const| (non-zero here) -> x/|x| == 1 everywhere.
        assert torch.allclose(normalized, torch.ones_like(normalized))
    else:  # meanpreserving
        # maxdev guarded to 1; deviations are 0 so output == mean == const.
        assert torch.allclose(stats["maxdev"], torch.ones_like(stats["maxdev"]))
        assert torch.allclose(normalized, torch.full_like(normalized, const_value))


def test_noncentered_all_zero_scale_guarded() -> None:
    """An all-zero feature keeps ``noncentered`` finite via the scale guard."""
    data = torch.zeros(1, 8, 3, dtype=torch.float64)
    normalized, stats = normalize_data(data, method="noncentered")

    assert torch.allclose(stats["scale"], torch.ones_like(stats["scale"]))
    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized, torch.zeros_like(normalized))


# ---------------------------------------------------------------------------
# Stats round-trip: reproduces output and applies to held-out data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_passing_stats_reproduces_output(method: str) -> None:
    """Re-applying the returned stats reproduces the original normalized output."""
    data = _sample_data()
    normalized, stats = normalize_data(data, method=method)

    reproduced, stats_again = normalize_data(data, method=method, stats=stats)
    assert torch.allclose(reproduced, normalized, atol=1e-12)
    # Passing stats short-circuits recomputation: the dict is returned as-is.
    assert stats_again is stats


@pytest.mark.parametrize("method", METHODS)
def test_held_out_data_uses_fitted_stats(method: str) -> None:
    """Stats fitted on a training tensor transform unseen data with the same map.

    The held-out transform must equal applying the documented forward formula
    with the *fitted* stats (not stats recomputed from the held-out data).
    """
    train = _sample_data()
    _, stats = normalize_data(train, method=method)

    held_out = _sample_data() * 1.3 - 4.0  # different distribution
    applied, _ = normalize_data(held_out, method=method, stats=stats)

    if method == "minmax":
        manual = 2 * (held_out - stats["min"]) / stats["range"] - 1
    elif method == "standard":
        manual = (held_out - stats["mean"]) / stats["std"]
    elif method == "noncentered":
        manual = held_out / stats["scale"]
    else:  # meanpreserving
        manual = (held_out - stats["mean"]) / stats["maxdev"] + stats["mean"]

    assert torch.allclose(applied, manual, atol=1e-12)
    # And inverting with the same stats recovers the held-out values.
    assert torch.allclose(denormalize_data(applied, method, stats), held_out, atol=1e-8)


def test_normalize_unknown_method_raises() -> None:
    """The forward transform rejects an unknown method (both stat + apply paths)."""
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_data(_sample_data(), method="bogus")  # type: ignore[arg-type]


def test_apply_norm_unknown_method_raises() -> None:
    """``_apply_norm`` rejects an unknown method when stats are supplied."""
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_data(_sample_data(), method="bogus", stats={"min": torch.zeros(1)})  # type: ignore[arg-type]
