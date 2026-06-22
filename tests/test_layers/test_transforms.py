"""Transform-layer contracts.

Pins down the deterministic tensor transforms used inside premade models:
``Concatenate`` (feature-dim joining), ``FeaturePartitioner`` (circular feature
grouping), ``Power`` (uniform exponentiation), ``SelectiveDropout`` (per-feature
masking), ``SelectiveExponentiation`` (even/odd-index state augmentation),
``Standardize`` (per-feature z-score with fit/inverse), and the
``OutliersFilteredMean`` ensemble aggregator.

Every transform is exercised through its backward pass as well as its forward
pass: gradients must be finite and route correctly back to the inputs, since
end-to-end SGD through these ``nn.Module``s is a core library pillar. Where the
map is smooth and the inputs can be kept away from kinks, the analytic gradient
is cross-checked numerically with :func:`torch.autograd.gradcheck`.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from resdag.ensemble.aggregators import OutliersFilteredMean
from resdag.layers.transforms import (
    Concatenate,
    FeaturePartitioner,
    Power,
    SelectiveDropout,
    SelectiveExponentiation,
    Standardize,
)


class TestFeaturePartitioner:
    """Tests for FeaturePartitioner layer."""

    def test_single_partition_returns_input(self) -> None:
        """Test that single partition returns input unchanged."""
        layer = FeaturePartitioner(partitions=1, overlap=0)
        x = torch.randn(2, 10, 12)

        output = layer(x)

        assert len(output) == 1
        assert torch.allclose(output[0], x)

    def test_two_partitions_no_overlap(self) -> None:
        """Test partitioning into two with no overlap."""
        layer = FeaturePartitioner(partitions=2, overlap=0)
        x = torch.randn(2, 10, 12)

        output = layer(x)

        assert len(output) == 2
        assert output[0].shape == (2, 10, 6)  # 12 // 2 = 6
        assert output[1].shape == (2, 10, 6)

    def test_two_partitions_with_overlap(self) -> None:
        """Test partitioning into two with overlap."""
        layer = FeaturePartitioner(partitions=2, overlap=2)
        x = torch.randn(2, 10, 12)

        output = layer(x)

        assert len(output) == 2
        # Each partition: 12//2 + 2*2 = 6 + 4 = 10
        assert output[0].shape == (2, 10, 10)
        assert output[1].shape == (2, 10, 10)

    def test_circular_wrapping(self) -> None:
        """Test that circular wrapping works correctly."""
        layer = FeaturePartitioner(partitions=2, overlap=1)
        # Create simple tensor to verify wrapping
        x = torch.arange(12).reshape(1, 1, 12).float()

        output = layer(x)

        # First partition should start with last feature (11) due to wrapping
        assert output[0][0, 0, 0] == 11.0

        # Last partition should end with first feature (0) due to wrapping
        assert output[1][0, 0, -1] == 0.0

    def test_invalid_partition_size_raises_error(self) -> None:
        """Test that invalid partition size raises error."""
        layer = FeaturePartitioner(partitions=3, overlap=0)
        x = torch.randn(2, 10, 13)  # 13 not divisible by 3

        with pytest.raises(ValueError, match="must be divisible"):
            layer(x)

    def test_invalid_overlap_size_raises_error(self) -> None:
        """Test that overlap too large raises error."""
        layer = FeaturePartitioner(partitions=2, overlap=10)
        x = torch.randn(2, 10, 12)  # overlap=10 >= 12//2=6

        with pytest.raises(ValueError, match="must be smaller"):
            layer(x)

    def test_zero_partitions_raises_at_construction(self) -> None:
        """``partitions=0`` is rejected at construction, not as a forward ZeroDivisionError."""
        with pytest.raises(ValueError, match="partitions must be a positive integer"):
            FeaturePartitioner(partitions=0, overlap=0)

    def test_negative_partitions_raises_at_construction(self) -> None:
        """Negative partitions are rejected at construction with a clear message."""
        with pytest.raises(ValueError, match="partitions must be a positive integer"):
            FeaturePartitioner(partitions=-1, overlap=0)

    def test_negative_overlap_raises_at_construction(self) -> None:
        """Negative overlap is rejected at construction rather than accepted silently."""
        with pytest.raises(ValueError, match="overlap must be a non-negative integer"):
            FeaturePartitioner(partitions=2, overlap=-1)

    def test_single_partition_with_overlap_warns(self) -> None:
        """``partitions == 1`` with ``overlap > 0`` warns that the overlap is ignored."""
        with pytest.warns(UserWarning, match="ignored when partitions == 1"):
            layer = FeaturePartitioner(partitions=1, overlap=5)

        # The overlap is genuinely ignored: the single-partition path is a no-op.
        x = torch.randn(2, 10, 12)
        output = layer(x)
        assert len(output) == 1
        assert torch.allclose(output[0], x)

    def test_single_partition_without_overlap_does_not_warn(self) -> None:
        """``partitions == 1`` with ``overlap == 0`` is a valid config and warns nothing."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FeaturePartitioner(partitions=1, overlap=0)

    def test_multiple_partitions(self) -> None:
        """Test partitioning into multiple groups."""
        layer = FeaturePartitioner(partitions=4, overlap=1)
        x = torch.randn(3, 5, 16)

        output = layer(x)

        assert len(output) == 4
        # Each partition: 16//4 + 2*1 = 4 + 2 = 6
        for partition in output:
            assert partition.shape == (3, 5, 6)

    def test_2d_input_accepted(self) -> None:
        """2-D ``(batch, features)`` input is accepted (rank-agnostic on dim=-1).

        Needed so the layer works in the autoregressive ``forecast`` path, where
        the flattened engine feeds single-step slices.
        """
        layer = FeaturePartitioner(partitions=4, overlap=1)
        x2 = torch.randn(3, 16)

        parts2 = layer(x2)
        assert len(parts2) == 4
        for partition in parts2:
            assert partition.shape == (3, 6)  # 16//4 + 2*1
        # Each 2-D partition equals the 3-D path on a singleton-time slice.
        parts3 = layer(x2.unsqueeze(1))
        for p2, p3 in zip(parts2, parts3):
            assert torch.equal(p2, p3.squeeze(1))

    def test_gradients_route_to_wrapped_indices(self) -> None:
        """Each feature gets one gradient per slice it appears in (incl. wraps).

        Summing every partition output is a pure copy of the input, so a feature
        copied into ``c`` slices accumulates a gradient of exactly ``c``. With
        ``partitions=2, overlap=1`` over 12 features the slices are
        ``p0 = [x11, x0..x6]`` and ``p1 = [x5..x11, x0]``: the features at the
        partition boundary (``x0, x5, x6, x11``) land in two slices and so carry
        grad 2, while the interior features appear once and carry grad 1. This
        pins the exact circular-overlap routing, not just "grads are finite".
        """
        layer = FeaturePartitioner(partitions=2, overlap=1)
        x = torch.arange(12, dtype=torch.float64).reshape(1, 12).requires_grad_(True)

        outputs = layer(x)
        torch.stack([o.sum() for o in outputs]).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # Boundary features (0, 5, 6, 11) appear in two slices; the rest in one.
        expected = torch.ones(1, 12, dtype=torch.float64)
        expected[0, [0, 5, 6, 11]] = 2.0
        assert torch.equal(x.grad, expected)

    def test_no_overlap_gradient_is_one_everywhere(self) -> None:
        """With no overlap every feature lands in exactly one partition (grad 1)."""
        layer = FeaturePartitioner(partitions=3, overlap=0)
        x = torch.randn(2, 12, dtype=torch.float64, requires_grad=True)

        outputs = layer(x)
        torch.stack([o.sum() for o in outputs]).sum().backward()

        assert x.grad is not None
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_gradient_routes_only_to_wrapped_source(self) -> None:
        """A gradient on a wrapped slice position flows to its source feature.

        The first ``overlap`` columns of partition 0 are the *last* ``overlap``
        features wrapped around. Back-propagating a unit gradient through only
        that wrapped column must land on the original (last) feature, proving the
        circular routing is wired to the right source index.
        """
        layer = FeaturePartitioner(partitions=2, overlap=1)
        x = torch.arange(12, dtype=torch.float64).reshape(1, 12).requires_grad_(True)

        # Partition 0, column 0 is the wrapped copy of the last input feature.
        layer(x)[0][0, 0].backward()

        assert x.grad is not None
        expected = torch.zeros(1, 12, dtype=torch.float64)
        expected[0, -1] = 1.0
        assert torch.equal(x.grad, expected)

    def test_gradcheck(self) -> None:
        """Double-precision gradcheck over the partition list output."""
        layer = FeaturePartitioner(partitions=2, overlap=1)
        x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)

        # gradcheck wants a tuple of tensors; the layer returns a list.
        def wrapped(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tuple(layer(t))

        assert torch.autograd.gradcheck(wrapped, (x,), eps=1e-6, atol=1e-4)

    def test_extra_repr(self) -> None:
        """Test string representation."""
        layer = FeaturePartitioner(partitions=3, overlap=2)
        repr_str = layer.extra_repr()

        assert "partitions=3" in repr_str
        assert "overlap=2" in repr_str


class TestOutliersFilteredMean:
    """Tests for OutliersFilteredMean layer."""

    def test_z_score_method_basic(self) -> None:
        """A blatant outlier is filtered, so the output matches the inlier mean."""
        layer = OutliersFilteredMean(method="z_score", threshold=2.0)
        # Ten members tightly clustered near 1.0, plus one blatant outlier.
        torch.manual_seed(42)
        samples = []
        for _ in range(10):
            samples.append(torch.randn(2, 5, 4) * 0.05 + 1.0)  # Mean ~1.0
        samples.append(torch.ones(2, 5, 4) * 100.0)  # Obvious outlier
        x = torch.stack(samples, dim=0)

        output = layer(x)
        plain_mean = x.mean(dim=0)

        assert output.shape == (2, 5, 4)
        # Tight assertion: the outlier must actually be dropped. The result sits
        # at the inlier cluster (~1.0), nowhere near the contaminated plain mean
        # (~10.0). The loose `< 2.0` bound the original test used would have
        # passed even with nothing filtered, so check both ends.
        assert torch.all(torch.abs(output - 1.0) < 0.1)
        assert torch.all(plain_mean > 9.0)  # confirm the outlier really contaminates the plain mean

    def test_z_score_filters_outlier_small_ensembles(self) -> None:
        """A single strong outlier is filtered for N in {3, 4, 5} at the default threshold.

        The non-robust mean/std Z-score saturates at ``sqrt(N - 1)`` for a lone
        outlier, so a 3.0 cutoff could never flag anything until ``N >= 11``.
        The robust median/MAD modified Z-score must catch it even for tiny
        ensembles.
        """
        layer = OutliersFilteredMean(method="z_score")  # default threshold
        for n in (3, 4, 5):
            inliers = [torch.ones(2, 3, 4) for _ in range(n - 1)]  # norm 2.0
            outlier = torch.ones(2, 3, 4) * 1000.0  # norm 2000.0
            x = torch.stack(inliers + [outlier], dim=0)

            output = layer(x)

            # Outlier dropped -> output equals the inlier value (1.0), not the
            # contaminated plain mean (which is far larger).
            assert torch.allclose(
                output, torch.ones(2, 3, 4), atol=1e-4
            ), f"outlier not filtered for N={n}"

    def test_z_score_all_outliers_falls_back_to_plain_mean(self) -> None:
        """When every member is flagged, the layer returns the plain mean."""
        # All members identical -> zero dispersion -> the degenerate MAD path
        # keeps members at the median, so the mean is preserved exactly.
        layer = OutliersFilteredMean(method="z_score")
        samples = [torch.ones(2, 3, 4) * 7.0 for _ in range(4)]
        x = torch.stack(samples, dim=0)

        output = layer(x)

        assert torch.allclose(output, torch.ones(2, 3, 4) * 7.0)

    def test_z_score_preserves_inlier_spread(self) -> None:
        """Inliers with mild spread are kept; only the true outlier is dropped."""
        layer = OutliersFilteredMean(method="z_score")
        samples = [
            torch.ones(1, 1, 1) * 1.0,
            torch.ones(1, 1, 1) * 1.1,
            torch.ones(1, 1, 1) * 0.9,
            torch.ones(1, 1, 1) * 100.0,  # Outlier
        ]
        x = torch.stack(samples, dim=0)

        output = layer(x)

        # Mean of the three inliers (1.0, 1.1, 0.9) == 1.0; outlier excluded.
        assert torch.allclose(output, torch.ones(1, 1, 1) * 1.0, atol=1e-4)

    def test_default_threshold_is_method_aware(self) -> None:
        """Default threshold differs by method; explicit values are preserved."""
        assert OutliersFilteredMean(method="z_score").threshold == 3.5
        assert OutliersFilteredMean(method="iqr").threshold == 1.5
        assert OutliersFilteredMean(method="z_score", threshold=2.0).threshold == 2.0

    def test_iqr_method_basic(self) -> None:
        """Test IQR outlier filtering."""
        layer = OutliersFilteredMean(method="iqr", threshold=1.5)
        samples = [
            torch.ones(2, 5, 4) * 1.0,
            torch.ones(2, 5, 4) * 1.1,
            torch.ones(2, 5, 4) * 0.9,
            torch.ones(2, 5, 4) * 100.0,  # Outlier
        ]
        x = torch.stack(samples, dim=0)

        output = layer(x)

        assert output.shape == (2, 5, 4)

    def test_list_input(self) -> None:
        """Test that list of tensors works."""
        layer = OutliersFilteredMean(method="z_score", threshold=2.0)
        samples = [
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 4),
        ]

        output = layer(samples)

        assert output.shape == (2, 5, 4)

    def test_single_tensor_3d_input(self) -> None:
        """Test that single 3D tensor gets samples dimension added."""
        layer = OutliersFilteredMean(method="z_score", threshold=2.0)
        x = torch.randn(2, 5, 4)

        output = layer(x)

        assert output.shape == (2, 5, 4)

    def test_no_outliers_returns_mean(self) -> None:
        """Test that with no outliers, returns regular mean."""
        layer = OutliersFilteredMean(method="z_score", threshold=10.0)  # High threshold
        samples = [
            torch.ones(2, 5, 4) * 1.0,
            torch.ones(2, 5, 4) * 2.0,
            torch.ones(2, 5, 4) * 3.0,
        ]
        x = torch.stack(samples, dim=0)

        output = layer(x)

        # Should be mean: (1 + 2 + 3) / 3 = 2.0
        assert torch.allclose(output, torch.ones(2, 5, 4) * 2.0)

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unsupported method"):
            OutliersFilteredMean(method="invalid", threshold=2.0)

    def test_per_member_gradient_no_outliers_is_uniform(self) -> None:
        """With no outliers the layer is a plain mean: grad 1/N per member.

        The outlier mask is a hard ``< threshold`` comparison and carries no
        gradient, so with all members retained the aggregate is ``mean over
        samples`` and each member receives an equal share ``1/N`` of the upstream
        gradient.
        """
        n = 4
        layer = OutliersFilteredMean(method="z_score", threshold=10.0)  # keep all
        # All members identical -> nobody is an outlier; mean path is active.
        x = torch.ones(n, 2, 3, 4, dtype=torch.float64, requires_grad=True)

        layer(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.allclose(x.grad, torch.full_like(x, 1.0 / n))

    def test_per_member_gradient_skips_dropped_member(self) -> None:
        """A filtered-out member receives zero gradient; inliers split the rest.

        One blatant outlier is dropped, so it sits behind the (gradient-free)
        mask and gets no gradient, while each of the surviving ``N-1`` inliers
        receives ``1/(N-1)``.
        """
        layer = OutliersFilteredMean(method="z_score")  # default threshold
        inliers = [torch.ones(2, 3, 4, dtype=torch.float64) for _ in range(3)]
        outlier = torch.ones(2, 3, 4, dtype=torch.float64) * 1000.0
        x = torch.stack(inliers + [outlier], dim=0).requires_grad_(True)

        layer(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # Three inliers each carry 1/3; the dropped outlier carries 0.
        assert torch.allclose(x.grad[:3], torch.full((3, 2, 3, 4), 1.0 / 3, dtype=torch.float64))
        assert torch.allclose(x.grad[3], torch.zeros(2, 3, 4, dtype=torch.float64))

    def test_gradient_finite_list_input(self) -> None:
        """Per-member gradients flow back through the list-input path."""
        layer = OutliersFilteredMean(method="z_score", threshold=2.0)
        members = [torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]

        layer(members).sum().backward()

        for member in members:
            assert member.grad is not None
            assert torch.isfinite(member.grad).all()

    def test_gradcheck_no_outliers(self) -> None:
        """Gradcheck on a stable-mask config (high threshold -> always plain mean).

        A threshold large enough that no member is ever flagged makes the mask
        constant under the small ``gradcheck`` perturbations, so the map is the
        smooth ``mean`` and the analytic gradient matches the numeric one.
        """
        layer = OutliersFilteredMean(method="z_score", threshold=1e9)
        x = torch.randn(3, 2, 3, 4, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)

    def test_extra_repr(self) -> None:
        """Test string representation."""
        layer = OutliersFilteredMean(method="iqr", threshold=1.5)
        repr_str = layer.extra_repr()

        assert "method='iqr'" in repr_str
        assert "threshold=1.5" in repr_str


class TestSelectiveDropout:
    """Tests for SelectiveDropout layer."""

    def test_basic_dropout(self) -> None:
        """Test basic selective dropout."""
        mask = [False, True, False, True]  # Drop indices 1 and 3
        layer = SelectiveDropout(mask)
        x = torch.ones(2, 5, 4)

        output = layer(x)

        assert output.shape == (2, 5, 4)
        # Indices 1 and 3 should be zero
        assert torch.all(output[..., 1] == 0)
        assert torch.all(output[..., 3] == 0)
        # Indices 0 and 2 should be preserved
        assert torch.all(output[..., 0] == 1)
        assert torch.all(output[..., 2] == 1)

    def test_numpy_mask(self) -> None:
        """Test with numpy array mask."""
        mask = np.array([True, False, False, True])
        layer = SelectiveDropout(mask)
        x = torch.ones(2, 5, 4)

        output = layer(x)

        assert torch.all(output[..., 0] == 0)
        assert torch.all(output[..., 3] == 0)

    def test_torch_tensor_mask(self) -> None:
        """Test with torch tensor mask."""
        mask = torch.tensor([False, False, True, False])
        layer = SelectiveDropout(mask)
        x = torch.ones(2, 5, 4)

        output = layer(x)

        assert torch.all(output[..., 2] == 0)

    def test_all_dropped(self) -> None:
        """Test dropping all features."""
        mask = [True, True, True, True]
        layer = SelectiveDropout(mask)
        x = torch.ones(2, 5, 4)

        output = layer(x)

        assert torch.all(output == 0)

    def test_none_dropped(self) -> None:
        """Test dropping no features."""
        mask = [False, False, False, False]
        layer = SelectiveDropout(mask)
        x = torch.randn(2, 5, 4)

        output = layer(x)

        assert torch.allclose(output, x)

    def test_2d_input_accepted(self) -> None:
        """2-D ``(batch, features)`` input is accepted (rank-agnostic on dim=-1).

        Required so the layer can sit in the autoregressive ``forecast`` path,
        and consistent with ``ReadoutLayer`` / the other feature-wise transforms.
        """
        mask = [False, True, False, True]
        layer = SelectiveDropout(mask)
        x2 = torch.randn(5, 4)

        out2 = layer(x2)
        assert out2.shape == (5, 4)
        assert torch.all(out2[:, [1, 3]] == 0)
        assert torch.equal(out2[:, [0, 2]], x2[:, [0, 2]])
        # 2-D result equals the 3-D path on a singleton-time slice.
        x3 = x2.unsqueeze(1)
        assert torch.equal(layer(x3).squeeze(1), out2)

    @pytest.mark.parametrize("bad", [torch.ones(4), torch.ones(2, 3, 5, 4)])
    def test_invalid_input_rank_raises_error(self, bad: torch.Tensor) -> None:
        """Ranks other than 2-D / 3-D still raise."""
        layer = SelectiveDropout([False, True, False, True])
        with pytest.raises(ValueError, match="expects 2D"):
            layer(bad)

    def test_mismatched_feature_dim_raises_error(self) -> None:
        """Test that mismatched feature dim raises error."""
        mask = [False, True, False, True]  # Length 4
        layer = SelectiveDropout(mask)
        x = torch.ones(2, 5, 6)  # Features = 6

        with pytest.raises(ValueError, match="does not match"):
            layer(x)

    def test_invalid_mask_shape_raises_error(self) -> None:
        """Test that 2D mask raises error."""
        mask = [[False, True], [False, True]]  # 2D mask
        with pytest.raises(ValueError, match="must be 1D"):
            SelectiveDropout(mask)

    def test_state_dict_includes_mask(self) -> None:
        """Test that mask is saved in state_dict."""
        mask = [False, True, False, True]
        layer = SelectiveDropout(mask)

        state_dict = layer.state_dict()

        assert "mask" in state_dict
        assert torch.equal(state_dict["mask"], torch.tensor(mask))

    def test_gradient_zero_at_dropped_identity_elsewhere(self) -> None:
        """Gradient is zero at dropped indices and one at the kept indices.

        The layer is ``where(mask, 0, x)``: dropped positions are a constant zero
        (no upstream gradient flows in), kept positions are the identity (grad 1).
        """
        mask = [False, True, False, True]  # drop indices 1 and 3
        layer = SelectiveDropout(mask)
        x = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)

        layer(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        expected = torch.ones_like(x)
        expected[..., [1, 3]] = 0.0
        assert torch.equal(x.grad, expected)

    def test_gradient_all_dropped_is_zero(self) -> None:
        """Dropping every feature yields an all-zero gradient (constant output)."""
        layer = SelectiveDropout([True, True, True, True])
        x = torch.randn(2, 5, 4, requires_grad=True)

        layer(x).sum().backward()

        assert x.grad is not None
        assert torch.equal(x.grad, torch.zeros_like(x))

    def test_gradcheck(self) -> None:
        """Double-precision gradcheck for the masked-identity map."""
        layer = SelectiveDropout([False, True, False, True])
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)

    def test_extra_repr(self) -> None:
        """Test string representation."""
        mask = [False, True, True, False]  # 2 dropped out of 4
        layer = SelectiveDropout(mask)
        repr_str = layer.extra_repr()

        assert "features=4" in repr_str
        assert "dropped=2" in repr_str


class TestSelectiveExponentiation:
    """Tests for SelectiveExponentiation layer."""

    def test_even_index_exponentiates_even_positions(self) -> None:
        """Test that even index exponentiates even positions."""
        layer = SelectiveExponentiation(index=2, exponent=2.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        output = layer(x)

        # Index 2 is even, so even positions (0, 2) are squared
        expected = torch.tensor([[1.0, 2.0, 9.0, 4.0]])  # 3^2 = 9
        assert torch.allclose(output, expected)

    def test_odd_index_exponentiates_odd_positions(self) -> None:
        """Test that odd index exponentiates odd positions."""
        layer = SelectiveExponentiation(index=1, exponent=2.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        output = layer(x)

        # Index 1 is odd, so odd positions (1, 3) are squared
        expected = torch.tensor([[1.0, 4.0, 3.0, 16.0]])
        assert torch.allclose(output, expected)

    def test_different_exponents(self) -> None:
        """Test with different exponent values."""
        layer = SelectiveExponentiation(index=0, exponent=3.0)
        x = torch.tensor([[2.0, 2.0, 2.0, 2.0]])

        output = layer(x)

        # Index 0 is even, so even positions (0, 2) are cubed (2^3 = 8)
        expected = torch.tensor([[8.0, 2.0, 8.0, 2.0]])
        assert torch.allclose(output, expected)

    def test_3d_input(self) -> None:
        """Test with 3D tensor input."""
        layer = SelectiveExponentiation(index=0, exponent=2.0)
        x = torch.ones(2, 3, 4) * 2.0

        output = layer(x)

        assert output.shape == (2, 3, 4)
        # Even positions should be squared (4.0), odd unchanged (2.0)
        assert torch.allclose(output[..., 1], torch.ones(2, 3) * 2.0)  # Odd
        assert torch.allclose(output[..., 2], torch.ones(2, 3) * 4.0)  # Even

    def test_negative_values(self) -> None:
        """Test with negative values."""
        layer = SelectiveExponentiation(index=0, exponent=2.0)
        x = torch.tensor([[-2.0, -2.0, -2.0, -2.0]])

        output = layer(x)

        # Index 0 is even, so even positions (0, 2) are squared: (-2)^2 = 4
        expected = torch.tensor([[4.0, -2.0, 4.0, -2.0]])
        assert torch.allclose(output, expected)

    def test_fractional_exponent(self) -> None:
        """Test with fractional exponent."""
        layer = SelectiveExponentiation(index=2, exponent=0.5)
        x = torch.tensor([[4.0, 4.0, 9.0, 16.0]])

        output = layer(x)

        # Index 2 is even, so even positions (0, 2) get square root
        # sqrt(4) = 2, sqrt(9) = 3
        expected = torch.tensor([[2.0, 4.0, 3.0, 16.0]])
        assert torch.allclose(output, expected)

    def test_extra_repr(self) -> None:
        """Test string representation."""
        layer = SelectiveExponentiation(index=3, exponent=2.5)
        repr_str = layer.extra_repr()

        assert "index=3" in repr_str
        assert "exponent=2.5" in repr_str
        assert "sign_preserving=False" in repr_str
        assert "applies_to=odd_indices" in repr_str  # 3 % 2 == 1 (odd)

    def test_fractional_exponent_finite_gradients(self) -> None:
        """Fractional exponent must yield finite gradients at every position.

        Regression for the mask-multiply-then-pow formulation, which routed
        unselected positions through ``pow(0, e)`` and produced ``nan``/``inf``
        gradients for ``e < 1`` (issue #130, acceptance criterion 1).
        """
        layer = SelectiveExponentiation(index=2, exponent=0.5)
        x = torch.tensor([[4.0, 4.0, 9.0, 16.0]], requires_grad=True)

        output = layer(x)
        output.sum().backward()

        # Forward is unchanged: sqrt at even indices, identity elsewhere.
        assert torch.allclose(output, torch.tensor([[2.0, 4.0, 3.0, 16.0]]))
        # All four gradients finite (previously nan at masked indices 1 and 3).
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # d/dx 0.5*x^-0.5 at selected; identity (grad 1) at unselected.
        expected_grad = torch.tensor([[0.25, 1.0, 1.0 / 6.0, 1.0]])
        assert torch.allclose(x.grad, expected_grad)

    def test_negative_unselected_base_finite_gradients(self) -> None:
        """A negative *unselected* base must not poison the gradient.

        ``torch.where(mask, input.pow(e), input)`` evaluates ``pow`` for every
        element, so a negative unselected base under a fractional exponent would
        back-propagate ``0 * nan`` without input masking. Grad must stay finite.
        """
        layer = SelectiveExponentiation(index=0, exponent=0.5)
        x = torch.tensor([[4.0, -4.0, 9.0, -16.0]], requires_grad=True)

        output = layer(x)
        output.sum().backward()

        # Unselected (odd) negative bases pass through untouched.
        assert torch.allclose(output, torch.tensor([[2.0, -4.0, 3.0, -16.0]]))
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_negative_selected_base_default_mode_is_nan_forward(self) -> None:
        """Default mode documents torch.pow semantics: negative^fraction = nan.

        Issue #130 verified ``SelectiveExponentiation(index=0, exponent=0.5)``
        on ``[[-4, 2, -9, 3]]`` returns ``[[nan, 2, nan, 3]]``. Unselected
        positions stay finite.
        """
        layer = SelectiveExponentiation(index=0, exponent=0.5)
        x = torch.tensor([[-4.0, 2.0, -9.0, 3.0]])

        output = layer(x)

        # Selected (even) negative bases are nan; unselected pass through.
        assert torch.isnan(output[0, 0])
        assert torch.isnan(output[0, 2])
        assert output[0, 1] == 2.0
        assert output[0, 3] == 3.0

    def test_negative_selected_base_default_mode_unselected_grad_finite(self) -> None:
        """Even when selected bases are nan, unselected gradients stay finite."""
        layer = SelectiveExponentiation(index=0, exponent=0.5)
        x = torch.tensor([[-4.0, 2.0, -9.0, 3.0]], requires_grad=True)

        output = layer(x)
        output.nansum().backward()

        assert x.grad is not None
        # Unselected (odd) positions must have finite gradient.
        assert torch.isfinite(x.grad[0, [1, 3]]).all()

    def test_sign_preserving_negative_base_finite_forward(self) -> None:
        """Sign-preserving mode keeps negative selected bases finite.

        Acceptance criterion 2: negative selected bases with a fractional
        exponent no longer produce a nan forward.
        """
        layer = SelectiveExponentiation(index=0, exponent=0.5, sign_preserving=True)
        x = torch.tensor([[-4.0, 2.0, -9.0, 3.0]])

        output = layer(x)

        # sign(x) * |x|^0.5 at selected even indices, identity elsewhere.
        expected = torch.tensor([[-2.0, 2.0, -3.0, 3.0]])
        assert torch.allclose(output, expected)
        assert torch.isfinite(output).all()

    def test_sign_preserving_negative_base_finite_gradients(self) -> None:
        """Sign-preserving mode yields finite gradients on negative bases."""
        layer = SelectiveExponentiation(index=0, exponent=0.5, sign_preserving=True)
        x = torch.tensor([[-4.0, 2.0, -9.0, 3.0]], requires_grad=True)

        output = layer(x)
        output.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradcheck_fractional_exponent(self) -> None:
        """Double-precision gradcheck for the fractional-exponent path."""
        layer = SelectiveExponentiation(index=2, exponent=1.5)
        # Strictly positive bases keep the analytic gradient well-defined.
        x = torch.rand(2, 4, dtype=torch.float64, requires_grad=True) + 0.5

        assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_sign_preserving(self) -> None:
        """Gradcheck the sign-preserving path away from the x=0 kink."""
        layer = SelectiveExponentiation(index=0, exponent=1.5, sign_preserving=True)
        # Push bases away from 0 (the |x| kink) for a well-defined gradient.
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        x = (x + torch.sign(x) * 1.0).detach().requires_grad_(True)

        assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)

    def test_ott_path_exponent_two_unchanged(self) -> None:
        """Default ott_esn path (exponent=2.0) keeps forward and grad identical.

        Acceptance criterion 4. Verified against the analytic result of the old
        mask-multiply-then-pow formulation.
        """
        layer = SelectiveExponentiation(index=0, exponent=2.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

        output = layer(x)
        output.sum().backward()

        # Even indices squared, odd unchanged.
        assert torch.allclose(output, torch.tensor([[1.0, 2.0, 9.0, 4.0]]))
        # Selected grad = 2x; unselected grad = 1.
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.tensor([[2.0, 1.0, 6.0, 1.0]]))

    def test_zero_base_gradients_finite(self) -> None:
        """Selected zero bases give finite gradients (no pow(0, e) blow-up)."""
        layer = SelectiveExponentiation(index=0, exponent=2.0)
        x = torch.zeros(1, 4, requires_grad=True)

        output = layer(x)
        output.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_dtype_preserved(self) -> None:
        """Output dtype matches input dtype."""
        layer = SelectiveExponentiation(index=2, exponent=0.5)
        x = torch.tensor([[4.0, 4.0, 9.0, 16.0]], dtype=torch.float64)

        output = layer(x)

        assert output.dtype == torch.float64


class TestStandardize:
    """Tests for the Standardize per-feature z-score transform."""

    def test_default_is_identity(self) -> None:
        """Before fit, mean=0 and std=1 make forward an identity map."""
        layer = Standardize(num_features=3)
        x = torch.randn(2, 5, 3)

        assert torch.allclose(layer(x), x, atol=1e-7)

    def test_fit_computes_per_feature_stats(self) -> None:
        """fit stores the per-feature mean and population std."""
        layer = Standardize(num_features=3)
        # Per-feature constants offset and scaled so stats are predictable.
        base = torch.randn(4, 50, 3)
        x = base * torch.tensor([2.0, 5.0, 0.5]) + torch.tensor([1.0, -3.0, 10.0])

        layer.fit(x)

        flat = x.reshape(-1, 3)
        assert torch.allclose(layer.mean, flat.mean(dim=0), atol=1e-5)
        assert torch.allclose(layer.std, flat.std(dim=0, unbiased=False), atol=1e-5)

    def test_fit_returns_self_for_chaining(self) -> None:
        """fit returns the layer so calls can be chained."""
        layer = Standardize(num_features=2)
        x = torch.randn(3, 10, 2)

        assert layer.fit(x) is layer

    def test_forward_standardizes_to_zero_mean_unit_std(self) -> None:
        """After fit, the output has ~zero mean and ~unit std per feature."""
        layer = Standardize(num_features=3)
        x = torch.randn(8, 200, 3) * 4.0 + 7.0
        layer.fit(x)

        z = layer(x).reshape(-1, 3)

        assert torch.allclose(z.mean(dim=0), torch.zeros(3), atol=1e-5)
        assert torch.allclose(z.std(dim=0, unbiased=False), torch.ones(3), atol=1e-4)

    def test_inverse_round_trip(self) -> None:
        """inverse(forward(x)) reconstructs x to floating-point tolerance."""
        layer = Standardize(num_features=4)
        x = torch.randn(6, 30, 4) * 3.0 - 2.0
        layer.fit(x)

        assert torch.allclose(layer.inverse(layer(x)), x, atol=1e-5)

    def test_forward_inverse_round_trip_float64(self) -> None:
        """Round trip is exact within tight tolerance in double precision."""
        layer = Standardize(num_features=3).double()
        x = (torch.randn(5, 20, 3) * 10.0 + 1.0).double()
        layer.fit(x)

        assert torch.allclose(layer.inverse(layer(x)), x, atol=1e-10)

    def test_constant_feature_eps_stability(self) -> None:
        """A zero-variance feature does not produce nan/inf (eps floor)."""
        layer = Standardize(num_features=2)
        # Feature 0 is constant (std=0); feature 1 varies.
        x = torch.stack([torch.full((4, 10), 5.0), torch.randn(4, 10)], dim=-1)
        layer.fit(x)

        z = layer(x)
        assert torch.isfinite(z).all()
        # Centered constant feature collapses to 0 (within eps).
        assert torch.allclose(z[..., 0], torch.zeros(4, 10), atol=1e-6)
        # Round trip still recovers the original constant.
        assert torch.allclose(layer.inverse(z), x, atol=1e-5)

    def test_explicit_stats_at_construction(self) -> None:
        """Mean/std supplied at construction are used without fitting."""
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([2.0, 4.0])
        layer = Standardize(num_features=2, mean=mean, std=std, eps=0.0)
        x = torch.tensor([[3.0, 10.0]])

        # (3 - 1) / 2 = 1.0 ; (10 - 2) / 4 = 2.0
        assert torch.allclose(layer(x), torch.tensor([[1.0, 2.0]]))

    def test_gradient_flows_through_forward(self) -> None:
        """forward is differentiable; stored stats act as constants."""
        layer = Standardize(num_features=3)
        layer.fit(torch.randn(4, 20, 3) * 2.0 + 1.0)
        x = torch.randn(2, 5, 3, requires_grad=True)

        layer(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # d/dx (x - mean)/(std + eps) = 1/(std + eps), broadcast over leading dims.
        expected = (1.0 / (layer.std + layer.eps)).expand_as(x)
        assert torch.allclose(x.grad, expected, atol=1e-6)

    def test_gradient_flows_through_inverse(self) -> None:
        """inverse is differentiable with finite gradients."""
        layer = Standardize(num_features=3)
        layer.fit(torch.randn(4, 20, 3) * 3.0)
        x = torch.randn(2, 5, 3, requires_grad=True)

        layer.inverse(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        expected = (layer.std + layer.eps).expand_as(x)
        assert torch.allclose(x.grad, expected, atol=1e-6)

    def test_gradcheck_forward(self) -> None:
        """Double-precision gradcheck of the forward map."""
        layer = Standardize(num_features=3).double()
        layer.fit(torch.randn(4, 30, 3, dtype=torch.float64) * 2.0 + 1.0)
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)

    def test_stats_registered_as_buffers(self) -> None:
        """mean and std are buffers (in state_dict, not parameters)."""
        layer = Standardize(num_features=3)

        buffer_names = {name for name, _ in layer.named_buffers()}
        assert {"mean", "std"} <= buffer_names
        # No learnable parameters.
        assert list(layer.parameters()) == []

    def test_state_dict_round_trip(self) -> None:
        """Fitted stats survive a state_dict save/load."""
        layer = Standardize(num_features=3)
        layer.fit(torch.randn(4, 40, 3) * 5.0 + 2.0)

        clone = Standardize(num_features=3)
        clone.load_state_dict(layer.state_dict())

        assert torch.allclose(clone.mean, layer.mean)
        assert torch.allclose(clone.std, layer.std)

    def test_to_moves_buffers_dtype(self) -> None:
        """.to(dtype) / .double() move the stat buffers with the module."""
        layer = Standardize(num_features=3)
        layer.fit(torch.randn(4, 20, 3))

        layer = layer.double()

        assert layer.mean.dtype == torch.float64
        assert layer.std.dtype == torch.float64
        # Forward now operates in double precision end-to-end.
        x = torch.randn(2, 5, 3, dtype=torch.float64)
        assert layer(x).dtype == torch.float64

    def test_to_moves_buffers_device_cuda(self) -> None:
        """.to(device) moves the stat buffers (CUDA-gated)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        layer = Standardize(num_features=3)
        layer.fit(torch.randn(4, 20, 3))

        layer = layer.to("cuda")

        assert layer.mean.is_cuda
        assert layer.std.is_cuda

    def test_save_full_load_full_pipeline_round_trip(self) -> None:
        """Standardize stats travel with ESNModel.save_full / load_full."""
        import pytorch_symbolic as ps

        from resdag import CGReadoutLayer, ESNLayer, ESNModel

        norm = Standardize(num_features=3)
        norm.fit(torch.randn(4, 100, 3) * 6.0 - 1.0)

        inp = ps.Input((100, 3))
        normed = norm(inp)
        reservoir = ESNLayer(40, feedback_size=3)(normed)
        readout = CGReadoutLayer(40, 3, name="output")(reservoir)
        model = ESNModel(inp, readout)

        x = torch.randn(2, 100, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full.pt"
            model.save_full(path)
            restored = ESNModel.load_full(path)

            model.reset_reservoirs()
            restored.reset_reservoirs()
            assert torch.allclose(model(x), restored(x), atol=1e-5)

    def test_fit_feature_dim_mismatch_raises(self) -> None:
        """fit rejects inputs whose last dim is not num_features."""
        layer = Standardize(num_features=3)
        with pytest.raises(ValueError, match="does not match"):
            layer.fit(torch.randn(2, 5, 4))

    def test_forward_feature_dim_mismatch_raises(self) -> None:
        """forward rejects inputs whose last dim is not num_features."""
        layer = Standardize(num_features=3)
        with pytest.raises(ValueError, match="does not match"):
            layer(torch.randn(2, 5, 4))

    def test_inverse_feature_dim_mismatch_raises(self) -> None:
        """inverse rejects inputs whose last dim is not num_features."""
        layer = Standardize(num_features=3)
        with pytest.raises(ValueError, match="does not match"):
            layer.inverse(torch.randn(2, 5, 4))

    def test_invalid_num_features_raises(self) -> None:
        """num_features must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            Standardize(num_features=0)

    def test_invalid_mean_shape_raises(self) -> None:
        """A 2D mean is rejected."""
        with pytest.raises(ValueError, match="mean must be 1D"):
            Standardize(num_features=2, mean=torch.zeros(2, 2))

    def test_invalid_std_length_raises(self) -> None:
        """A std of the wrong length is rejected."""
        with pytest.raises(ValueError, match="std length"):
            Standardize(num_features=3, std=torch.ones(2))

    def test_extra_repr(self) -> None:
        """String representation reports num_features and eps."""
        layer = Standardize(num_features=5, eps=1e-6)
        repr_str = layer.extra_repr()

        assert "num_features=5" in repr_str
        assert "eps=1e-06" in repr_str


class TestConcatenate:
    """Forward and backward contracts for the Concatenate transform."""

    def test_forward_joins_feature_dim(self) -> None:
        """Inputs are concatenated along the last dimension."""
        layer = Concatenate()
        a = torch.randn(2, 5, 3)
        b = torch.randn(2, 5, 4)

        out = layer(a, b)

        assert out.shape == (2, 5, 7)
        assert torch.equal(out[..., :3], a)
        assert torch.equal(out[..., 3:], b)

    def test_gradient_splits_across_inputs(self) -> None:
        """Upstream gradient is split back to the matching input slices.

        ``cat`` simply views each input as a contiguous block of the output, so
        a per-feature upstream weight routes verbatim to the source tensor and
        each input sees only the slice it contributed.
        """
        layer = Concatenate()
        a = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)

        out = layer(a, b)
        # Distinct per-feature weights so a mis-routed gradient would be caught.
        weight = torch.arange(1, out.shape[-1] + 1, dtype=torch.float64)
        (out * weight).sum().backward()

        assert a.grad is not None and b.grad is not None
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
        # First 3 weights land on a; the remaining 4 land on b.
        assert torch.allclose(a.grad, weight[:3].expand_as(a))
        assert torch.allclose(b.grad, weight[3:].expand_as(b))

    def test_gradient_three_inputs(self) -> None:
        """Variadic input: every input receives a finite slice of the gradient."""
        layer = Concatenate()
        xs = [torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)]

        layer(*xs).sum().backward()

        for x in xs:
            assert x.grad is not None
            assert torch.equal(x.grad, torch.ones_like(x))

    def test_gradcheck(self) -> None:
        """Double-precision gradcheck across the variadic inputs."""
        layer = Concatenate()
        a = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(layer, (a, b), eps=1e-6, atol=1e-4)


class TestPower:
    """Forward and backward contracts for the Power transform."""

    def test_forward_squares_all_features(self) -> None:
        """Every element is raised to the exponent."""
        layer = Power(exponent=2.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        assert torch.allclose(layer(x), torch.tensor([[1.0, 4.0, 9.0, 16.0]]))

    def test_gradient_matches_power_rule(self) -> None:
        """Gradient is the analytic ``e * x**(e-1)`` and stays finite."""
        layer = Power(exponent=3.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64, requires_grad=True)

        layer(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # d/dx x^3 = 3 x^2.
        assert torch.allclose(x.grad, 3.0 * x.detach() ** 2)

    def test_gradient_fractional_exponent_positive_base(self) -> None:
        """Fractional exponent on strictly positive bases gives finite grads."""
        layer = Power(exponent=0.5)
        x = torch.tensor([[4.0, 9.0, 16.0]], dtype=torch.float64, requires_grad=True)

        layer(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # d/dx x^0.5 = 0.5 x^-0.5.
        assert torch.allclose(x.grad, 0.5 * x.detach() ** -0.5)

    def test_gradcheck_integer_exponent(self) -> None:
        """Double-precision gradcheck for an integer exponent (any real base)."""
        layer = Power(exponent=3.0)
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_fractional_exponent_positive_base(self) -> None:
        """Gradcheck the fractional path on strictly positive bases."""
        layer = Power(exponent=1.5)
        x = torch.rand(2, 4, dtype=torch.float64, requires_grad=True) + 0.5

        assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)

    def test_negative_base_integer_exponent_forward(self) -> None:
        """Negative base with an integer exponent is finite and sign-correct.

        Issue #208 acceptance criterion: a ``Power`` forward + gradient test
        including a negative base with an integer exponent. An even exponent
        maps negatives to positives; an odd exponent preserves the sign.
        """
        x = torch.tensor([[-2.0, 3.0, -4.0]])

        even = Power(exponent=2.0)(x)
        odd = Power(exponent=3.0)(x)

        assert torch.allclose(even, torch.tensor([[4.0, 9.0, 16.0]]))
        assert torch.allclose(odd, torch.tensor([[-8.0, 27.0, -64.0]]))
        assert torch.isfinite(even).all()
        assert torch.isfinite(odd).all()

    def test_negative_base_integer_exponent_gradient(self) -> None:
        """Gradient on a negative base with an integer exponent stays finite."""
        layer = Power(exponent=3.0)
        x = torch.tensor([[-2.0, 3.0, -4.0]], dtype=torch.float64, requires_grad=True)

        layer(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # d/dx x^3 = 3 x^2, valid for any real base.
        assert torch.allclose(x.grad, 3.0 * x.detach() ** 2)

    def test_negative_base_fractional_exponent_default_is_nan(self) -> None:
        """Default mode documents torch.pow: negative base ^ fraction = nan.

        Issue #208 verified ``Power(0.5)`` on ``[[-4.0, 4.0]]`` yields
        ``[[nan, 2.0]]``. This pins that caveat so the documented behaviour is
        regression-tested.
        """
        out = Power(exponent=0.5)(torch.tensor([[-4.0, 4.0]]))

        assert torch.isnan(out[0, 0])
        assert out[0, 1] == 2.0

    def test_zero_base_negative_exponent_default_is_inf(self) -> None:
        """Default mode documents torch.pow: zero base ^ negative = inf.

        Issue #208 verified ``Power(-1.0)`` on ``[[0.0, 2.0]]`` yields
        ``[[inf, 0.5]]``.
        """
        out = Power(exponent=-1.0)(torch.tensor([[0.0, 2.0]]))

        assert torch.isinf(out[0, 0])
        assert out[0, 1] == 0.5

    def test_sign_preserving_negative_base_finite_forward(self) -> None:
        """Sign-preserving mode keeps negative bases finite under a fraction.

        Issue #208 acceptance criterion: ``Power`` offers a sign-preserving
        mode. ``sign(x) * abs(x) ** exponent`` returns real, signed roots where
        the default :func:`torch.pow` returns ``nan``.
        """
        layer = Power(exponent=0.5, sign_preserving=True)
        x = torch.tensor([[-4.0, 4.0, -9.0, 9.0]])

        out = layer(x)

        assert torch.allclose(out, torch.tensor([[-2.0, 2.0, -3.0, 3.0]]))
        assert torch.isfinite(out).all()

    def test_sign_preserving_negative_base_finite_gradients(self) -> None:
        """Sign-preserving mode yields finite gradients on negative bases."""
        layer = Power(exponent=0.5, sign_preserving=True)
        x = torch.tensor([[-4.0, 4.0, -9.0, 9.0]], dtype=torch.float64, requires_grad=True)

        layer(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_sign_preserving_matches_default_on_positive_base(self) -> None:
        """For non-negative bases, both modes agree exactly."""
        x = torch.tensor([[0.5, 1.0, 2.0, 4.0]])

        plain = Power(exponent=1.5)(x)
        signed = Power(exponent=1.5, sign_preserving=True)(x)

        assert torch.allclose(plain, signed)

    def test_sign_preserving_odd_integer_matches_default(self) -> None:
        """Odd integer exponents are already sign-preserving; modes agree."""
        x = torch.tensor([[-2.0, 3.0, -4.0]])

        plain = Power(exponent=3.0)(x)
        signed = Power(exponent=3.0, sign_preserving=True)(x)

        assert torch.allclose(plain, signed)

    def test_gradcheck_sign_preserving(self) -> None:
        """Gradcheck the sign-preserving path away from the x=0 kink."""
        layer = Power(exponent=1.5, sign_preserving=True)
        # Push bases away from 0 (the |x| kink) for a well-defined gradient.
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        x = (x + torch.sign(x) * 1.0).detach().requires_grad_(True)

        assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)

    def test_extra_repr(self) -> None:
        """String representation reports the exponent and mode."""
        repr_str = Power(exponent=2.0).extra_repr()
        assert "exponent=2.0" in repr_str
        assert "sign_preserving=False" in repr_str
