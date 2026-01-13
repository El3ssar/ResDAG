"""Unit tests for custom layers."""

import numpy as np
import pytest
import torch

from resdag.layers.custom import (
    FeaturePartitioner,
    OutliersFilteredMean,
    SelectiveDropout,
    SelectiveExponentiation,
)


class TestFeaturePartitioner:
    """Tests for FeaturePartitioner layer."""

    def test_single_partition_returns_input(self):
        """Test that single partition returns input unchanged."""
        layer = FeaturePartitioner(partitions=1, overlap=0)
        x = torch.randn(2, 10, 12)

        output = layer(x)

        assert len(output) == 1
        assert torch.allclose(output[0], x)

    def test_two_partitions_no_overlap(self):
        """Test partitioning into two with no overlap."""
        layer = FeaturePartitioner(partitions=2, overlap=0)
        x = torch.randn(2, 10, 12)

        output = layer(x)

        assert len(output) == 2
        assert output[0].shape == (2, 10, 6)  # 12 // 2 = 6
        assert output[1].shape == (2, 10, 6)

    def test_two_partitions_with_overlap(self):
        """Test partitioning into two with overlap."""
        layer = FeaturePartitioner(partitions=2, overlap=2)
        x = torch.randn(2, 10, 12)

        output = layer(x)

        assert len(output) == 2
        # Each partition: 12//2 + 2*2 = 6 + 4 = 10
        assert output[0].shape == (2, 10, 10)
        assert output[1].shape == (2, 10, 10)

    def test_circular_wrapping(self):
        """Test that circular wrapping works correctly."""
        layer = FeaturePartitioner(partitions=2, overlap=1)
        # Create simple tensor to verify wrapping
        x = torch.arange(12).reshape(1, 1, 12).float()

        output = layer(x)

        # First partition should start with last feature (11) due to wrapping
        assert output[0][0, 0, 0] == 11.0

        # Last partition should end with first feature (0) due to wrapping
        assert output[1][0, 0, -1] == 0.0

    def test_invalid_partition_size_raises_error(self):
        """Test that invalid partition size raises error."""
        layer = FeaturePartitioner(partitions=3, overlap=0)
        x = torch.randn(2, 10, 13)  # 13 not divisible by 3

        with pytest.raises(ValueError, match="must be divisible"):
            layer(x)

    def test_invalid_overlap_size_raises_error(self):
        """Test that overlap too large raises error."""
        layer = FeaturePartitioner(partitions=2, overlap=10)
        x = torch.randn(2, 10, 12)  # overlap=10 >= 12//2=6

        with pytest.raises(ValueError, match="must be smaller"):
            layer(x)

    def test_multiple_partitions(self):
        """Test partitioning into multiple groups."""
        layer = FeaturePartitioner(partitions=4, overlap=1)
        x = torch.randn(3, 5, 16)

        output = layer(x)

        assert len(output) == 4
        # Each partition: 16//4 + 2*1 = 4 + 2 = 6
        for partition in output:
            assert partition.shape == (3, 5, 6)

    def test_extra_repr(self):
        """Test string representation."""
        layer = FeaturePartitioner(partitions=3, overlap=2)
        repr_str = layer.extra_repr()

        assert "partitions=3" in repr_str
        assert "overlap=2" in repr_str


class TestOutliersFilteredMean:
    """Tests for OutliersFilteredMean layer."""

    def test_z_score_method_basic(self):
        """Test Z-score outlier filtering."""
        layer = OutliersFilteredMean(method="z_score", threshold=2.0)
        # Create data with several samples and one obvious outlier
        torch.manual_seed(42)
        samples = []
        for i in range(10):
            samples.append(torch.randn(2, 5, 4) * 0.1 + 1.0)  # Mean ~1.0
        samples.append(torch.ones(2, 5, 4) * 100.0)  # Obvious outlier
        x = torch.stack(samples, dim=0)

        output = layer(x)

        assert output.shape == (2, 5, 4)
        # Output should be close to 1.0 (outlier filtered out)
        assert torch.all(torch.abs(output - 1.0) < 2.0)

    def test_iqr_method_basic(self):
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

    def test_list_input(self):
        """Test that list of tensors works."""
        layer = OutliersFilteredMean(method="z_score", threshold=2.0)
        samples = [
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 4),
        ]

        output = layer(samples)

        assert output.shape == (2, 5, 4)

    def test_single_tensor_3d_input(self):
        """Test that single 3D tensor gets samples dimension added."""
        layer = OutliersFilteredMean(method="z_score", threshold=2.0)
        x = torch.randn(2, 5, 4)

        output = layer(x)

        assert output.shape == (2, 5, 4)

    def test_no_outliers_returns_mean(self):
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

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unsupported method"):
            OutliersFilteredMean(method="invalid", threshold=2.0)

    def test_extra_repr(self):
        """Test string representation."""
        layer = OutliersFilteredMean(method="iqr", threshold=1.5)
        repr_str = layer.extra_repr()

        assert "method='iqr'" in repr_str
        assert "threshold=1.5" in repr_str


class TestSelectiveDropout:
    """Tests for SelectiveDropout layer."""

    def test_basic_dropout(self):
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

    def test_numpy_mask(self):
        """Test with numpy array mask."""
        mask = np.array([True, False, False, True])
        layer = SelectiveDropout(mask)
        x = torch.ones(2, 5, 4)

        output = layer(x)

        assert torch.all(output[..., 0] == 0)
        assert torch.all(output[..., 3] == 0)

    def test_torch_tensor_mask(self):
        """Test with torch tensor mask."""
        mask = torch.tensor([False, False, True, False])
        layer = SelectiveDropout(mask)
        x = torch.ones(2, 5, 4)

        output = layer(x)

        assert torch.all(output[..., 2] == 0)

    def test_all_dropped(self):
        """Test dropping all features."""
        mask = [True, True, True, True]
        layer = SelectiveDropout(mask)
        x = torch.ones(2, 5, 4)

        output = layer(x)

        assert torch.all(output == 0)

    def test_none_dropped(self):
        """Test dropping no features."""
        mask = [False, False, False, False]
        layer = SelectiveDropout(mask)
        x = torch.randn(2, 5, 4)

        output = layer(x)

        assert torch.allclose(output, x)

    def test_invalid_input_shape_raises_error(self):
        """Test that invalid input shape raises error."""
        mask = [False, True, False, True]
        layer = SelectiveDropout(mask)
        x = torch.ones(5, 4)  # 2D instead of 3D

        with pytest.raises(ValueError, match="Expected input shape"):
            layer(x)

    def test_mismatched_feature_dim_raises_error(self):
        """Test that mismatched feature dim raises error."""
        mask = [False, True, False, True]  # Length 4
        layer = SelectiveDropout(mask)
        x = torch.ones(2, 5, 6)  # Features = 6

        with pytest.raises(ValueError, match="does not match"):
            layer(x)

    def test_invalid_mask_shape_raises_error(self):
        """Test that 2D mask raises error."""
        mask = [[False, True], [False, True]]  # 2D mask
        with pytest.raises(ValueError, match="must be 1D"):
            SelectiveDropout(mask)

    def test_state_dict_includes_mask(self):
        """Test that mask is saved in state_dict."""
        mask = [False, True, False, True]
        layer = SelectiveDropout(mask)

        state_dict = layer.state_dict()

        assert "mask" in state_dict
        assert torch.equal(state_dict["mask"], torch.tensor(mask))

    def test_extra_repr(self):
        """Test string representation."""
        mask = [False, True, True, False]  # 2 dropped out of 4
        layer = SelectiveDropout(mask)
        repr_str = layer.extra_repr()

        assert "features=4" in repr_str
        assert "dropped=2" in repr_str


class TestSelectiveExponentiation:
    """Tests for SelectiveExponentiation layer."""

    def test_even_index_exponentiates_even_positions(self):
        """Test that even index exponentiates even positions."""
        layer = SelectiveExponentiation(index=2, exponent=2.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        output = layer(x)

        # Index 2 is even, so even positions (0, 2) are squared
        expected = torch.tensor([[1.0, 2.0, 9.0, 4.0]])  # 3^2 = 9
        assert torch.allclose(output, expected)

    def test_odd_index_exponentiates_odd_positions(self):
        """Test that odd index exponentiates odd positions."""
        layer = SelectiveExponentiation(index=1, exponent=2.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        output = layer(x)

        # Index 1 is odd, so odd positions (1, 3) are squared
        expected = torch.tensor([[1.0, 4.0, 3.0, 16.0]])
        assert torch.allclose(output, expected)

    def test_different_exponents(self):
        """Test with different exponent values."""
        layer = SelectiveExponentiation(index=0, exponent=3.0)
        x = torch.tensor([[2.0, 2.0, 2.0, 2.0]])

        output = layer(x)

        # Index 0 is even, so even positions (0, 2) are cubed (2^3 = 8)
        expected = torch.tensor([[8.0, 2.0, 8.0, 2.0]])
        assert torch.allclose(output, expected)

    def test_3d_input(self):
        """Test with 3D tensor input."""
        layer = SelectiveExponentiation(index=0, exponent=2.0)
        x = torch.ones(2, 3, 4) * 2.0

        output = layer(x)

        assert output.shape == (2, 3, 4)
        # Even positions should be squared (4.0), odd unchanged (2.0)
        assert torch.allclose(output[..., 1], torch.ones(2, 3) * 2.0)  # Odd
        assert torch.allclose(output[..., 2], torch.ones(2, 3) * 4.0)  # Even

    def test_negative_values(self):
        """Test with negative values."""
        layer = SelectiveExponentiation(index=0, exponent=2.0)
        x = torch.tensor([[-2.0, -2.0, -2.0, -2.0]])

        output = layer(x)

        # Index 0 is even, so even positions (0, 2) are squared: (-2)^2 = 4
        expected = torch.tensor([[4.0, -2.0, 4.0, -2.0]])
        assert torch.allclose(output, expected)

    def test_fractional_exponent(self):
        """Test with fractional exponent."""
        layer = SelectiveExponentiation(index=2, exponent=0.5)
        x = torch.tensor([[4.0, 4.0, 9.0, 16.0]])

        output = layer(x)

        # Index 2 is even, so even positions (0, 2) get square root
        # sqrt(4) = 2, sqrt(9) = 3
        expected = torch.tensor([[2.0, 4.0, 3.0, 16.0]])
        assert torch.allclose(output, expected)

    def test_extra_repr(self):
        """Test string representation."""
        layer = SelectiveExponentiation(index=3, exponent=2.5)
        repr_str = layer.extra_repr()

        assert "index=3" in repr_str
        assert "exponent=2.5" in repr_str
        assert "applies_to=odd_indices" in repr_str  # 3 % 2 == 1 (odd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
