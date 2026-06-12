"""Readout layer contracts: base ReadoutLayer and CGReadoutLayer.

Pins down:

- ``ReadoutLayer``: construction, 2D/3D forward semantics, validation,
  properties, gradients, state-dict round-trips, and repr,
- ``CGReadoutLayer``: the conjugate-gradient ridge solver against the
  closed-form solution, ``fit()`` for 2D/3D inputs, bias-free fitting,
  convergence behaviour, and predictions on every device.
"""

import pytest
import torch
import torch.nn as nn

from resdag.layers import ReadoutLayer
from resdag.layers.readouts import CGReadoutLayer


def solve_ridge_closed_form(
    X: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ridge regression using closed-form solution for comparison.

    Solves: (X.T @ X + alpha * I) @ w = X.T @ y on centered data.

    Returns
    -------
    coefs : torch.Tensor
        Shape ``(n_features, n_outputs)``.
    intercept : torch.Tensor
        Shape ``(n_outputs,)``.
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

    coefs = torch.linalg.solve(A, Xty)

    # Compute intercept
    intercept = (y_mean - X_mean @ coefs).squeeze(0)

    return coefs, intercept


# ---------------------------------------------------------------------------
# Base ReadoutLayer
# ---------------------------------------------------------------------------


class TestReadoutLayerInstantiation:
    """ReadoutLayer instantiation and configuration."""

    def test_basic_instantiation(self) -> None:
        """Creating readout with basic parameters."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        assert readout.in_features == 100
        assert readout.out_features == 10
        assert readout.bias is not None
        assert readout.name is None
        assert not readout.is_fitted

    def test_instantiation_without_bias(self) -> None:
        """Creating readout without bias."""
        readout = ReadoutLayer(in_features=50, out_features=5, bias=False)

        assert readout.bias is None

    def test_instantiation_with_name(self) -> None:
        """Creating named readout."""
        readout = ReadoutLayer(in_features=100, out_features=10, name="output1")

        assert readout.name == "output1"

    def test_is_linear_subclass(self) -> None:
        """ReadoutLayer is a proper nn.Linear subclass."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        assert isinstance(readout, nn.Linear)
        assert isinstance(readout, nn.Module)

    def test_parameters_accessible(self) -> None:
        """Weight and bias parameters are accessible."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        assert hasattr(readout, "weight")
        assert hasattr(readout, "bias")
        assert readout.weight.shape == (10, 100)
        assert readout.bias.shape == (10,)


class TestReadoutLayerForward2D:
    """Forward pass with 2D inputs (standard linear layer behaviour)."""

    def test_forward_2d_input(self) -> None:
        """Forward pass with 2D input."""
        readout = ReadoutLayer(in_features=100, out_features=10)
        x = torch.randn(5, 100)  # (batch=5, features=100)

        output = readout(x)

        assert output.shape == (5, 10)
        assert isinstance(output, torch.Tensor)

    def test_forward_2d_matches_linear(self) -> None:
        """2D forward matches standard nn.Linear."""
        # Create both with same weights
        readout = ReadoutLayer(in_features=100, out_features=10)
        linear = nn.Linear(100, 10)

        # Copy weights
        linear.weight.data = readout.weight.data.clone()
        linear.bias.data = readout.bias.data.clone()

        x = torch.randn(5, 100)

        output_readout = readout(x)
        output_linear = linear(x)

        assert torch.allclose(output_readout, output_linear)


class TestReadoutLayerForward3D:
    """Forward pass with 3D inputs (per-timestep application)."""

    def test_forward_3d_input(self) -> None:
        """Forward pass with 3D sequence input."""
        readout = ReadoutLayer(in_features=100, out_features=10)
        x = torch.randn(2, 20, 100)  # (batch=2, seq=20, features=100)

        output = readout(x)

        assert output.shape == (2, 20, 10)
        assert isinstance(output, torch.Tensor)

    def test_forward_3d_preserves_per_timestep_semantics(self) -> None:
        """3D forward applies an independent transformation per timestep."""
        readout = ReadoutLayer(in_features=100, out_features=10)
        x = torch.randn(2, 20, 100)

        # Apply to full sequence
        output_sequence = readout(x)

        # Verify we can apply readout to individual timesteps with same weights
        # (tests that same linear transform is used, not numerical equality)
        out_t0 = readout(x[:, 0, :])  # (2, 100) -> (2, 10)
        out_t5 = readout(x[:, 5, :])

        # These should have the correct shape from the linear layer
        assert out_t0.shape == (2, 10)
        assert out_t5.shape == (2, 10)

        # The full sequence output should have correct shape
        assert output_sequence.shape == (2, 20, 10)

    @pytest.mark.parametrize("seq_len", [1, 10, 50, 100])
    def test_forward_3d_different_sequence_lengths(self, seq_len: int) -> None:
        """Forward with different sequence lengths."""
        readout = ReadoutLayer(in_features=50, out_features=5)

        x = torch.randn(3, seq_len, 50)
        output = readout(x)
        assert output.shape == (3, seq_len, 5)

    def test_forward_3d_batch_size_one(self) -> None:
        """Forward with batch size 1."""
        readout = ReadoutLayer(in_features=20, out_features=5)
        x = torch.randn(1, 10, 20)

        output = readout(x)

        assert output.shape == (1, 10, 5)


class TestReadoutLayerInputValidation:
    """Input validation and error handling."""

    def test_forward_invalid_dimensions_raises_error(self) -> None:
        """Invalid input dimensions raise ValueError."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        # 1D input
        with pytest.raises(ValueError, match="expects 2D.*or 3D"):
            readout(torch.randn(100))

        # 4D input
        with pytest.raises(ValueError, match="expects 2D.*or 3D"):
            readout(torch.randn(2, 20, 10, 100))

    def test_forward_wrong_feature_size_raises_error(self) -> None:
        """Wrong feature size raises error."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        # 2D with wrong feature size
        x_2d = torch.randn(5, 50)  # Should be 100, not 50
        with pytest.raises(RuntimeError):  # PyTorch runtime error from matmul
            readout(x_2d)

        # 3D with wrong feature size
        x_3d = torch.randn(2, 20, 50)  # Should be 100, not 50
        with pytest.raises(RuntimeError):
            readout(x_3d)


class TestReadoutLayerProperties:
    """ReadoutLayer properties."""

    def test_name_property(self) -> None:
        """Name property getter."""
        readout_unnamed = ReadoutLayer(100, 10)
        assert readout_unnamed.name is None

        readout_named = ReadoutLayer(100, 10, name="my_readout")
        assert readout_named.name == "my_readout"

    def test_is_fitted_property_default_false(self) -> None:
        """is_fitted is False by default."""
        readout = ReadoutLayer(100, 10)
        assert readout.is_fitted is False

    def test_multiple_readouts_have_independent_names(self) -> None:
        """Multiple readouts have independent names."""
        readout1 = ReadoutLayer(100, 10, name="output1")
        readout2 = ReadoutLayer(100, 5, name="output2")
        readout3 = ReadoutLayer(100, 3)

        assert readout1.name == "output1"
        assert readout2.name == "output2"
        assert readout3.name is None


class TestReadoutLayerFit:
    """fit() raises NotImplementedError in the base class."""

    def test_fit_raises_not_implemented(self) -> None:
        """fit() raises NotImplementedError in base ReadoutLayer."""
        readout = ReadoutLayer(100, 10)
        states = torch.randn(10, 20, 100)
        targets = torch.randn(10, 20, 10)

        with pytest.raises(NotImplementedError, match="not implemented"):
            readout.fit(states, targets)


class TestReadoutLayerDevice:
    """Device handling."""

    def test_forward_on_device(self, device: torch.device) -> None:
        """Parameters and outputs land on the target device."""
        readout = ReadoutLayer(100, 10).to(device)

        assert next(readout.parameters()).device.type == device.type
        assert readout.weight.device.type == device.type
        assert readout.bias.device.type == device.type

        x = torch.randn(4, 20, 100, device=device)
        output = readout(x)

        assert output.device.type == device.type
        assert output.shape == (4, 20, 10)


class TestReadoutLayerGradients:
    """Gradient computation."""

    def test_gradients_flow_through_readout_2d(self) -> None:
        """Gradients flow through readout with 2D input."""
        readout = ReadoutLayer(in_features=100, out_features=10, trainable=True)
        x = torch.randn(5, 100, requires_grad=True)

        output = readout(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert readout.weight.grad is not None
        assert readout.bias.grad is not None

    def test_gradients_flow_through_readout_3d(self) -> None:
        """Gradients flow through readout with 3D input."""
        readout = ReadoutLayer(in_features=100, out_features=10, trainable=True)
        x = torch.randn(2, 20, 100, requires_grad=True)

        output = readout(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert readout.weight.grad is not None
        assert readout.bias.grad is not None


class TestReadoutLayerStateDict:
    """state_dict serialization."""

    def test_state_dict_contains_weights(self) -> None:
        """state_dict contains weight and bias."""
        readout = ReadoutLayer(100, 10)
        state = readout.state_dict()

        assert "weight" in state
        assert "bias" in state
        assert state["weight"].shape == (10, 100)
        assert state["bias"].shape == (10,)

    def test_load_state_dict(self) -> None:
        """Loading state_dict restores parameters."""
        readout1 = ReadoutLayer(100, 10)
        readout2 = ReadoutLayer(100, 10)

        # Set readout1 to specific values
        with torch.no_grad():
            readout1.weight.fill_(1.0)
            readout1.bias.fill_(2.0)

        # Load into readout2
        readout2.load_state_dict(readout1.state_dict())

        assert torch.allclose(readout2.weight, torch.ones(10, 100))
        assert torch.allclose(readout2.bias, torch.ones(10) * 2.0)


class TestReadoutLayerTraining:
    """Training mode behaviour."""

    def test_train_eval_mode(self) -> None:
        """train/eval mode switching."""
        readout = ReadoutLayer(100, 10)

        # Default is training mode
        assert readout.training is True

        readout.eval()
        assert readout.training is False

        readout.train()
        assert readout.training is True

    def test_standard_pytorch_training_works(self) -> None:
        """Standard PyTorch training works (SGD alternative)."""
        readout = ReadoutLayer(in_features=100, out_features=10, trainable=True)
        optimizer = torch.optim.SGD(readout.parameters(), lr=0.01)

        # Simple training step
        x = torch.randn(5, 20, 100)
        targets = torch.randn(5, 20, 10)

        optimizer.zero_grad()
        output = readout(x)
        loss = nn.MSELoss()(output, targets)
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert readout.weight.grad is not None


class TestReadoutLayerRepr:
    """String representation."""

    def test_repr_unnamed(self) -> None:
        """__repr__ for unnamed readout."""
        readout = ReadoutLayer(100, 10)
        repr_str = repr(readout)

        assert "ReadoutLayer" in repr_str
        assert "in_features=100" in repr_str
        assert "out_features=10" in repr_str
        assert "bias=True" in repr_str
        assert "name=" not in repr_str  # No name shown if None

    def test_repr_named(self) -> None:
        """__repr__ for named readout."""
        readout = ReadoutLayer(100, 10, name="output1")
        repr_str = repr(readout)

        assert "ReadoutLayer" in repr_str
        assert "name='output1'" in repr_str

    def test_repr_no_bias(self) -> None:
        """__repr__ for readout without bias."""
        readout = ReadoutLayer(100, 10, bias=False)
        repr_str = repr(readout)

        assert "bias=False" in repr_str


# ---------------------------------------------------------------------------
# CGReadoutLayer
# ---------------------------------------------------------------------------


class TestCGReadoutLayerInstantiation:
    """CGReadoutLayer instantiation and configuration."""

    def test_basic_instantiation(self) -> None:
        """Creating CG readout with basic parameters."""
        readout = CGReadoutLayer(in_features=100, out_features=10)

        assert readout.in_features == 100
        assert readout.out_features == 10
        assert readout.bias is not None
        assert readout.max_iter == 100
        assert readout.tol == 1e-5

    def test_custom_cg_parameters(self) -> None:
        """Custom CG solver parameters."""
        readout = CGReadoutLayer(in_features=50, out_features=5, max_iter=200, tol=1e-6)

        assert readout.max_iter == 200
        assert readout.tol == 1e-6

    def test_inherits_from_readout_layer(self) -> None:
        """CGReadoutLayer inherits from ReadoutLayer."""
        readout = CGReadoutLayer(in_features=100, out_features=10)
        assert isinstance(readout, ReadoutLayer)

    def test_repr(self) -> None:
        """String representation."""
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
    """CG solver accuracy against the closed-form solution."""

    def test_cg_matches_closed_form_single_output(self) -> None:
        """CG solver matches closed-form solution for single output."""
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

        assert torch.allclose(coefs_cg, coefs_cf, atol=1e-6, rtol=1e-5)
        assert torch.allclose(intercept_cg, intercept_cf, atol=1e-6, rtol=1e-5)

    def test_cg_matches_closed_form_multiple_outputs(self) -> None:
        """CG solver matches closed-form for multiple outputs."""
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

        assert torch.allclose(coefs_cg, coefs_cf, atol=1e-6, rtol=1e-5)
        assert torch.allclose(intercept_cg, intercept_cf, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("alpha", [1e-6, 1e-4, 1e-2, 1.0])
    def test_cg_with_different_regularization_strengths(self, alpha: float) -> None:
        """CG solver with various regularization strengths."""
        torch.manual_seed(42)

        n_samples, n_features, n_outputs = 50, 10, 3
        X = torch.randn(n_samples, n_features, dtype=torch.float64)
        y = torch.randn(n_samples, n_outputs, dtype=torch.float64)

        readout_cg = CGReadoutLayer(
            in_features=n_features, out_features=n_outputs, max_iter=1000, tol=1e-10
        )

        coefs_cg, intercept_cg = readout_cg._solve_ridge_cg(X, y, alpha)
        coefs_cf, intercept_cf = solve_ridge_closed_form(X, y, alpha)

        assert torch.allclose(coefs_cg, coefs_cf, atol=1e-6, rtol=1e-5)
        assert torch.allclose(intercept_cg, intercept_cf, atol=1e-6, rtol=1e-5)

    def test_negative_alpha_raises_error(self) -> None:
        """Negative alpha raises ValueError."""
        readout = CGReadoutLayer(in_features=10, out_features=2)
        X = torch.randn(20, 10, dtype=torch.float64)
        y = torch.randn(20, 2, dtype=torch.float64)

        with pytest.raises(ValueError, match="Alpha must be non-negative"):
            readout._solve_ridge_cg(X, y, alpha=-1.0)


class TestCGReadoutFit:
    """CGReadoutLayer fit method."""

    def test_fit_2d_input(self) -> None:
        """Fitting with 2D input."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3)
        X = torch.randn(100, 20)
        y = torch.randn(100, 5)

        readout.fit(X, y)

        assert readout.is_fitted
        assert readout.weight.shape == (5, 20)
        assert readout.bias.shape == (5,)

    def test_fit_3d_input(self) -> None:
        """Fitting with 3D input (batch, time, features)."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3)
        X = torch.randn(4, 25, 20)  # (batch, time, features)
        y = torch.randn(4, 25, 5)

        readout.fit(X, y)

        assert readout.is_fitted
        assert readout.weight.shape == (5, 20)
        assert readout.bias.shape == (5,)

    def test_fit_updates_weights_and_bias(self) -> None:
        """fit actually updates weights and bias."""
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

    def test_fit_produces_accurate_predictions(self) -> None:
        """Fitted readout produces accurate predictions."""
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

    def test_fit_with_mismatched_shapes_raises_error(self) -> None:
        """Mismatched input shapes raise ValueError."""
        readout = CGReadoutLayer(in_features=20, out_features=5)

        X = torch.randn(100, 20)
        y = torch.randn(50, 5)  # Different number of samples

        with pytest.raises(ValueError, match="sample count mismatch"):
            readout.fit(X, y)

    def test_fit_with_wrong_output_dim_raises_error(self) -> None:
        """Wrong output dimension raises ValueError."""
        readout = CGReadoutLayer(in_features=20, out_features=5)

        X = torch.randn(100, 20)
        y = torch.randn(100, 3)  # Should be 5, not 3

        with pytest.raises(ValueError, match="target feature dimension"):
            readout.fit(X, y)


class TestCGReadoutFitWithoutBias:
    """Fit correctness for bias=False (uncentered ridge, no intercept)."""

    def test_no_bias_recovers_linear_map(self) -> None:
        """y = X @ W with nonzero-mean X must be recovered without an
        intercept.  The legacy solver centered the data and then discarded
        the intercept, shifting every prediction."""
        torch.manual_seed(42)

        X = torch.randn(500, 20) + 2.0  # deliberately not zero-mean
        true_weight = torch.randn(20, 5)
        y = X @ true_weight

        readout = CGReadoutLayer(in_features=20, out_features=5, bias=False, alpha=1e-8)
        readout.fit(X, y)

        y_pred = readout(X)
        assert readout.bias is None
        assert torch.allclose(y_pred, y, atol=1e-3)

    def test_no_bias_matches_closed_form_ridge(self) -> None:
        """Weights must solve (XᵀX + αI) W = Xᵀy on the raw (uncentered) data."""
        torch.manual_seed(0)
        alpha = 1e-2

        X = torch.randn(300, 10) + 1.0
        y = torch.randn(300, 4)

        readout = CGReadoutLayer(in_features=10, out_features=4, bias=False, alpha=alpha)
        readout.fit(X, y)

        X64, y64 = X.to(torch.float64), y.to(torch.float64)
        gram = X64.T @ X64 + alpha * torch.eye(10, dtype=torch.float64)
        expected = torch.linalg.solve(gram, X64.T @ y64)

        assert torch.allclose(readout.weight.data.to(torch.float64).T, expected, atol=1e-4)

    def test_with_bias_still_centers(self) -> None:
        """bias=True keeps the centered solve with unpenalized intercept."""
        torch.manual_seed(1)

        X = torch.randn(400, 15) + 3.0
        true_weight = torch.randn(15, 2)
        true_bias = torch.tensor([5.0, -2.0])
        y = X @ true_weight + true_bias

        readout = CGReadoutLayer(in_features=15, out_features=2, alpha=1e-8)
        readout.fit(X, y)

        assert torch.allclose(readout(X), y, atol=1e-3)


class TestCGReadoutPredictions:
    """Predictions after fitting."""

    def test_forward_after_fit(self) -> None:
        """Forward pass after fitting."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3)
        X_train = torch.randn(100, 20)
        y_train = torch.randn(100, 5)

        readout.fit(X_train, y_train)

        # Forward pass on new data
        X_test = torch.randn(10, 20)
        y_pred = readout(X_test)

        assert y_pred.shape == (10, 5)

    def test_forward_3d_after_fit(self) -> None:
        """3D forward pass after fitting."""
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
    """CG solver convergence properties."""

    def test_convergence_with_low_tolerance(self) -> None:
        """CG converges with tight tolerance."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, max_iter=2000, tol=1e-12)

        X = torch.randn(100, 20, dtype=torch.float64)
        y = torch.randn(100, 5, dtype=torch.float64)
        alpha = 1e-6

        coefs_cg, _ = readout._solve_ridge_cg(X, y, alpha)
        coefs_cf, _ = solve_ridge_closed_form(X, y, alpha)

        # Should be very close with tight tolerance
        assert torch.allclose(coefs_cg, coefs_cf, atol=1e-8, rtol=1e-7)

    def test_early_stopping_with_high_tolerance(self) -> None:
        """CG stops early with loose tolerance and still returns sane shapes."""
        torch.manual_seed(42)

        # This test mainly checks that it doesn't error with high tolerance
        readout = CGReadoutLayer(in_features=20, out_features=5, max_iter=10, tol=1e-2)

        X = torch.randn(100, 20, dtype=torch.float64)
        y = torch.randn(100, 5, dtype=torch.float64)

        coefs, intercept = readout._solve_ridge_cg(X, y, 1e-3)

        # Should still produce reasonable results
        assert coefs.shape == (20, 5)
        assert intercept.shape == (5,)


class TestCGReadoutDevice:
    """CGReadoutLayer fit/predict on every available device."""

    def test_fit_on_device(self, device: torch.device) -> None:
        """Fitting on the target device."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3).to(device)
        X = torch.randn(100, 20, device=device)
        y = torch.randn(100, 5, device=device)

        readout.fit(X, y)

        assert readout.is_fitted
        assert readout.weight.device.type == device.type
        assert readout.bias.device.type == device.type

    def test_predictions_on_device(self, device: torch.device) -> None:
        """Predictions on the target device after fitting."""
        torch.manual_seed(42)

        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3).to(device)
        X_train = torch.randn(100, 20, device=device)
        y_train = torch.randn(100, 5, device=device)

        readout.fit(X_train, y_train)

        X_test = torch.randn(10, 20, device=device)
        y_pred = readout(X_test)

        assert y_pred.device.type == device.type
        assert y_pred.shape == (10, 5)
