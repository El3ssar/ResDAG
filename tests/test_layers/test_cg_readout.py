"""Unit tests for CGReadoutLayer."""

import pytest
import torch

from resdag.layers.readouts import CGReadoutLayer


def solve_ridge_closed_form(X: torch.Tensor, y: torch.Tensor, alpha: float):
    """Solve ridge regression using closed-form solution for comparison.

    Solves: (X.T @ X + alpha * I) @ w = X.T @ y

    Returns:
        coefs: (n_features, n_outputs)
        intercept: (n_outputs,)
    """
    # Center the data
    X_mean = X.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)

    X_centered = X - X_mean
    y_centered = y - y_mean

    # Solve using closed form: w = (X.T X + alpha I)^{-1} X.T y
    XtX = X_centered.T @ X_centered
    Xty = X_centered.T @ y_centered

    # Add regularization
    n_features = X_centered.shape[1]
    A = XtX + alpha * torch.eye(n_features, dtype=X.dtype, device=X.device)

    # Solve
    coefs = torch.linalg.solve(A, Xty)

    # Compute intercept
    intercept = (y_mean - X_mean @ coefs).squeeze(0)

    return coefs, intercept


class TestCGReadoutLayerInstantiation:
    """Test CGReadoutLayer instantiation and configuration."""

    def test_basic_instantiation(self):
        """Test creating CG readout with basic parameters."""
        readout = CGReadoutLayer(in_features=100, out_features=10)

        assert readout.in_features == 100
        assert readout.out_features == 10
        assert readout.bias is not None
        assert readout.max_iter == 100
        assert readout.tol == 1e-5

    def test_custom_cg_parameters(self):
        """Test custom CG solver parameters."""
        readout = CGReadoutLayer(in_features=50, out_features=5, max_iter=200, tol=1e-6)

        assert readout.max_iter == 200
        assert readout.tol == 1e-6

    def test_inherits_from_readout_layer(self):
        """Test that CGReadoutLayer inherits from ReadoutLayer."""
        from resdag.layers.readouts import ReadoutLayer

        readout = CGReadoutLayer(in_features=100, out_features=10)
        assert isinstance(readout, ReadoutLayer)

    def test_repr(self):
        """Test string representation."""
        readout = CGReadoutLayer(
            in_features=100, out_features=10, name="test_readout", max_iter=200
        )
        repr_str = repr(readout)

        assert "CGReadoutLayer" in repr_str
        assert "in_features=100" in repr_str
        assert "out_features=10" in repr_str
        assert "name='test_readout'" in repr_str
        assert "max_iter=200" in repr_str


class TestCGSolverAccuracy:
    """Test CG solver accuracy against closed-form solution."""

    def test_cg_matches_closed_form_single_output(self):
        """Test that CG solver matches closed-form solution for single output."""
        torch.manual_seed(42)

        # Generate synthetic data
        n_samples, n_features = 100, 20
        X = torch.randn(n_samples, n_features, dtype=torch.float64)
        y = torch.randn(n_samples, 1, dtype=torch.float64)
        alpha = 1e-3

        # Solve with CG
        readout_cg = CGReadoutLayer(
            in_features=n_features, out_features=1, max_iter=1000, tol=1e-10
        )
        coefs_cg, intercept_cg = readout_cg._solve_ridge_cg(X, y, alpha)

        # Solve with closed form
        coefs_cf, intercept_cf = solve_ridge_closed_form(X, y, alpha)

        # Compare coefficients
        assert torch.allclose(coefs_cg, coefs_cf, atol=1e-6, rtol=1e-5)

        # Compare intercepts
        assert torch.allclose(intercept_cg, intercept_cf, atol=1e-6, rtol=1e-5)

    def test_cg_matches_closed_form_multiple_outputs(self):
        """Test that CG solver matches closed-form for multiple outputs."""
        torch.manual_seed(42)

        # Generate synthetic data
        n_samples, n_features, n_outputs = 100, 20, 5
        X = torch.randn(n_samples, n_features, dtype=torch.float64)
        y = torch.randn(n_samples, n_outputs, dtype=torch.float64)
        alpha = 1e-3

        # Solve with CG
        readout_cg = CGReadoutLayer(
            in_features=n_features, out_features=n_outputs, max_iter=1000, tol=1e-10
        )
        coefs_cg, intercept_cg = readout_cg._solve_ridge_cg(X, y, alpha)

        # Solve with closed form
        coefs_cf, intercept_cf = solve_ridge_closed_form(X, y, alpha)

        # Compare coefficients
        assert torch.allclose(coefs_cg, coefs_cf, atol=1e-6, rtol=1e-5)

        # Compare intercepts
        assert torch.allclose(intercept_cg, intercept_cf, atol=1e-6, rtol=1e-5)

    def test_cg_with_different_regularization_strengths(self):
        """Test CG solver with various regularization strengths."""
        torch.manual_seed(42)

        n_samples, n_features, n_outputs = 50, 10, 3
        X = torch.randn(n_samples, n_features, dtype=torch.float64)
        y = torch.randn(n_samples, n_outputs, dtype=torch.float64)

        readout_cg = CGReadoutLayer(
            in_features=n_features, out_features=n_outputs, max_iter=1000, tol=1e-10
        )

        for alpha in [1e-6, 1e-4, 1e-2, 1.0]:
            coefs_cg, intercept_cg = readout_cg._solve_ridge_cg(X, y, alpha)
            coefs_cf, intercept_cf = solve_ridge_closed_form(X, y, alpha)

            assert torch.allclose(coefs_cg, coefs_cf, atol=1e-6, rtol=1e-5)
            assert torch.allclose(intercept_cg, intercept_cf, atol=1e-6, rtol=1e-5)

    def test_negative_alpha_raises_error(self):
        """Test that negative alpha raises ValueError."""
        readout = CGReadoutLayer(in_features=10, out_features=2)
        X = torch.randn(20, 10, dtype=torch.float64)
        y = torch.randn(20, 2, dtype=torch.float64)

        with pytest.raises(ValueError, match="Alpha must be non-negative"):
            readout._solve_ridge_cg(X, y, alpha=-1.0)


class TestCGReadoutFit:
    """Test CGReadoutLayer fit method."""

    def test_fit_2d_input(self):
        """Test fitting with 2D input."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3)
        X = torch.randn(100, 20)
        y = torch.randn(100, 5)

        readout.fit(X, y)

        assert readout.is_fitted
        assert readout.weight.shape == (5, 20)
        assert readout.bias.shape == (5,)

    def test_fit_3d_input(self):
        """Test fitting with 3D input (batch, time, features)."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3)
        X = torch.randn(4, 25, 20)  # (batch, time, features)
        y = torch.randn(4, 25, 5)

        readout.fit(X, y)

        assert readout.is_fitted
        assert readout.weight.shape == (5, 20)
        assert readout.bias.shape == (5,)

    def test_fit_updates_weights_and_bias(self):
        """Test that fit actually updates weights and bias."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=10, out_features=3, alpha=1e-3)

        # Store initial weights
        initial_weight = readout.weight.data.clone()
        initial_bias = readout.bias.data.clone()

        # Fit
        X = torch.randn(50, 10)
        y = torch.randn(50, 3)
        readout.fit(X, y)

        # Weights should have changed
        assert not torch.allclose(readout.weight.data, initial_weight)
        assert not torch.allclose(readout.bias.data, initial_bias)

    def test_fit_produces_accurate_predictions(self):
        """Test that fitted readout produces accurate predictions."""
        torch.manual_seed(42)

        readout_cg = CGReadoutLayer(
            in_features=20, out_features=5, max_iter=1000, tol=1e-10, alpha=1e-6
        )

        # Generate synthetic data with known relationship
        X = torch.randn(100, 20, dtype=torch.float32)
        true_weight = torch.randn(20, 5, dtype=torch.float32)
        true_bias = torch.randn(5, dtype=torch.float32)
        y = X @ true_weight + true_bias + torch.randn(100, 5) * 0.01  # Small noise

        # Fit with CG
        readout_cg.fit(X, y)

        # Fit with closed form for comparison
        X_64 = X.to(torch.float64)
        y_64 = y.to(torch.float64)
        coefs_cf, intercept_cf = solve_ridge_closed_form(X_64, y_64, 1e-6)

        # Compare predictions
        y_pred_cg = readout_cg(X)
        y_pred_cf = (X_64 @ coefs_cf + intercept_cf).to(torch.float32)

        assert torch.allclose(y_pred_cg, y_pred_cf, atol=1e-4, rtol=1e-3)

    def test_fit_with_mismatched_shapes_raises_error(self):
        """Test that mismatched input shapes raise ValueError."""
        readout = CGReadoutLayer(in_features=20, out_features=5)

        X = torch.randn(100, 20)
        y = torch.randn(50, 5)  # Different number of samples

        with pytest.raises(ValueError, match="Number of samples must match"):
            readout.fit(X, y)

    def test_fit_with_wrong_output_dim_raises_error(self):
        """Test that wrong output dimension raises ValueError."""
        readout = CGReadoutLayer(in_features=20, out_features=5)

        X = torch.randn(100, 20)
        y = torch.randn(100, 3)  # Should be 5, not 3

        with pytest.raises(ValueError, match="Target output dimension"):
            readout.fit(X, y)


class TestCGReadoutPredictions:
    """Test predictions after fitting."""

    def test_forward_after_fit(self):
        """Test forward pass after fitting."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3)
        X_train = torch.randn(100, 20)
        y_train = torch.randn(100, 5)

        readout.fit(X_train, y_train)

        # Forward pass on new data
        X_test = torch.randn(10, 20)
        y_pred = readout(X_test)

        assert y_pred.shape == (10, 5)

    def test_forward_3d_after_fit(self):
        """Test 3D forward pass after fitting."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3)
        X_train = torch.randn(4, 25, 20)
        y_train = torch.randn(4, 25, 5)

        readout.fit(X_train, y_train)

        # Forward pass on 3D data
        X_test = torch.randn(2, 10, 20)
        y_pred = readout(X_test)

        assert y_pred.shape == (2, 10, 5)


class TestCGReadoutConvergence:
    """Test CG solver convergence properties."""

    def test_convergence_with_low_tolerance(self):
        """Test that CG converges with tight tolerance."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, max_iter=2000, tol=1e-12)

        X = torch.randn(100, 20, dtype=torch.float64)
        y = torch.randn(100, 5, dtype=torch.float64)
        alpha = 1e-6

        coefs_cg, intercept_cg = readout._solve_ridge_cg(X, y, alpha)
        coefs_cf, intercept_cf = solve_ridge_closed_form(X, y, alpha)

        # Should be very close with tight tolerance
        assert torch.allclose(coefs_cg, coefs_cf, atol=1e-8, rtol=1e-7)

    def test_early_stopping_with_high_tolerance(self):
        """Test that CG stops early with loose tolerance."""
        torch.manual_seed(42)

        # This test mainly checks that it doesn't error with high tolerance
        readout = CGReadoutLayer(in_features=20, out_features=5, max_iter=10, tol=1e-2)

        X = torch.randn(100, 20, dtype=torch.float64)
        y = torch.randn(100, 5, dtype=torch.float64)

        coefs, intercept = readout._solve_ridge_cg(X, y, 1e-3)

        # Should still produce reasonable results
        assert coefs.shape == (20, 5)
        assert intercept.shape == (5,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCGReadoutGPU:
    """Test CGReadoutLayer on GPU."""

    def test_fit_on_gpu(self):
        """Test fitting on GPU."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3).cuda()
        X = torch.randn(100, 20).cuda()
        y = torch.randn(100, 5).cuda()

        readout.fit(X, y)

        assert readout.is_fitted
        assert readout.weight.is_cuda
        assert readout.bias.is_cuda

    def test_predictions_on_gpu(self):
        """Test predictions on GPU after fitting."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3).cuda()
        X_train = torch.randn(100, 20).cuda()
        y_train = torch.randn(100, 5).cuda()

        readout.fit(X_train, y_train)

        X_test = torch.randn(10, 20).cuda()
        y_pred = readout(X_test)

        assert y_pred.is_cuda
        assert y_pred.shape == (10, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
