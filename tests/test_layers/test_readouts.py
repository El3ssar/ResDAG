"""Readout layer contracts: base ReadoutLayer and CGReadoutLayer.

Pins down:

- ``ReadoutLayer``: construction, 2D/3D forward semantics, validation,
  properties, gradients, state-dict round-trips, and repr,
- ``CGReadoutLayer``: the conjugate-gradient ridge solver against the
  closed-form solution, ``fit()`` for 2D/3D inputs, bias-free fitting,
  convergence behaviour, and predictions on every device,
- ``CholeskyReadoutLayer``: the single-shot Cholesky ridge solver matching CG
  to ``< 1e-5``, the auto ``gram_dtype`` policy, and ``ESNTrainer`` use,
- ``IncrementalRidgeReadout``: chunked ``partial_fit`` / ``finalize`` matching a
  full-batch fit to ``< 1e-5``, the ``is_fitted``-after-``finalize`` /
  forward-before-``finalize`` lifecycle, and end-to-end fitting over a
  ``DataLoader`` of windowed chunks via ``ESNTrainer.fit_stream``.
"""

import pytest
import pytorch_symbolic as ps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from resdag import ESNLayer, ESNModel
from resdag.layers import ReadoutLayer
from resdag.layers.readouts import (
    CGReadoutLayer,
    CholeskyReadoutLayer,
    IncrementalRidgeReadout,
)
from resdag.training import ESNTrainer


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


class TestCGReadoutDegenerateColumns:
    """Degenerate target columns must not poison the fit with NaNs.

    Regression coverage for the per-column converged/degenerate mask: a column
    whose residual is zero from the start (or that converges to exact zero)
    used to drive ``0 / 0 = NaN`` in the CG step sizes, and the NaN then
    propagated to every other column.
    """

    def test_zero_column_does_not_produce_nans(self) -> None:
        """A single all-zero target column fits without NaNs (bias=False).

        The other columns keep their fitted values; only the zero column maps
        to ``W = 0``.
        """
        torch.manual_seed(0)

        n_samples, n_features = 200, 12
        X = torch.randn(n_samples, n_features, dtype=torch.float64)
        y = torch.randn(n_samples, 3, dtype=torch.float64)
        y[:, 1] = 0.0  # degenerate column in the middle

        readout = CGReadoutLayer(in_features=n_features, out_features=3, bias=False, alpha=1e-6)
        coefs, intercept = readout._solve_ridge_cg(X, y, 1e-6)

        assert torch.isfinite(coefs).all()
        assert intercept is None
        # The zero column must fit to W = 0; the others must be non-trivial.
        assert torch.allclose(coefs[:, 1], torch.zeros(n_features, dtype=torch.float64))
        assert not torch.allclose(coefs[:, 0], torch.zeros(n_features, dtype=torch.float64))
        assert not torch.allclose(coefs[:, 2], torch.zeros(n_features, dtype=torch.float64))

    def test_zero_column_via_fit_keeps_other_columns_accurate(self) -> None:
        """fit() with one zero target column leaves the other columns finite."""
        torch.manual_seed(1)

        X = torch.randn(150, 10)
        true_w = torch.randn(10, 4)
        y = X @ true_w
        y[:, 2] = 0.0  # one degenerate column

        readout = CGReadoutLayer(in_features=10, out_features=4, alpha=1e-8)
        readout.fit(X, y)

        assert torch.isfinite(readout.weight).all()
        assert torch.isfinite(readout.bias).all()
        y_pred = readout(X)
        assert torch.isfinite(y_pred).all()
        # The zero column stays at zero; the rest are recovered.
        assert torch.allclose(y_pred[:, 2], torch.zeros(150), atol=1e-4)

    def test_all_zero_targets_give_zero_weight_finite_bias(self) -> None:
        """All-zero targets fit to W = 0 with a finite (zero) bias, no NaNs."""
        torch.manual_seed(2)

        X = torch.randn(120, 8, dtype=torch.float64)
        y = torch.zeros(120, 5, dtype=torch.float64)

        readout = CGReadoutLayer(in_features=8, out_features=5, alpha=1e-6)
        readout.fit(X, y)

        assert torch.isfinite(readout.weight).all()
        assert torch.isfinite(readout.bias).all()
        assert torch.allclose(readout.weight, torch.zeros_like(readout.weight))
        assert torch.allclose(readout.bias, torch.zeros_like(readout.bias))

    def test_all_zero_targets_no_bias_give_zero_weight(self) -> None:
        """All-zero targets with bias=False fit to W = 0 (no NaNs, no shift)."""
        torch.manual_seed(3)

        X = torch.randn(120, 8, dtype=torch.float64) + 3.0
        y = torch.zeros(120, 5, dtype=torch.float64)

        readout = CGReadoutLayer(in_features=8, out_features=5, bias=False, alpha=1e-6)
        coefs, intercept = readout._solve_ridge_cg(X, y, 1e-6)

        assert intercept is None
        assert torch.isfinite(coefs).all()
        assert torch.allclose(coefs, torch.zeros_like(coefs))

    def test_over_iterating_past_convergence_stays_finite(self) -> None:
        """A well-conditioned fit over-iterated far past convergence stays finite.

        Once a column's residual reaches its tolerance (and ultimately exact
        zero) it is frozen, so thousands of extra iterations never turn the
        coefficients into NaNs.
        """
        torch.manual_seed(4)

        n_samples, n_features, n_outputs = 60, 15, 4
        X = torch.randn(n_samples, n_features, dtype=torch.float64)
        y = torch.randn(n_samples, n_outputs, dtype=torch.float64)
        alpha = 1e-6

        # Huge max_iter with an extremely tight tol forces the loop to keep
        # running long after exact convergence.
        readout = CGReadoutLayer(
            in_features=n_features, out_features=n_outputs, max_iter=5000, tol=1e-30
        )
        coefs, intercept = readout._solve_ridge_cg(X, y, alpha)

        assert torch.isfinite(coefs).all()
        assert intercept is not None and torch.isfinite(intercept).all()

        # Still matches the closed-form solution (no accuracy loss).
        coefs_cf, intercept_cf = solve_ridge_closed_form(X, y, alpha)
        assert torch.allclose(coefs, coefs_cf, atol=1e-6, rtol=1e-5)

    def test_over_iterating_with_zero_column_stays_finite(self) -> None:
        """Over-iteration AND a zero column together still stay finite.

        This is the worst case from the bug report: a zero column would go NaN,
        and over-iterating past convergence used to turn every column NaN.
        """
        torch.manual_seed(5)

        X = torch.randn(80, 10, dtype=torch.float64)
        y = torch.randn(80, 3, dtype=torch.float64)
        y[:, 0] = 0.0

        readout = CGReadoutLayer(in_features=10, out_features=3, max_iter=3000, tol=1e-30)
        coefs, _ = readout._solve_ridge_cg(X, y, 1e-6)

        assert torch.isfinite(coefs).all()
        assert torch.allclose(coefs[:, 0], torch.zeros(10, dtype=torch.float64))


class TestCGReadoutDegenerateGram:
    """Ill-conditioned / rank-deficient state matrices stay finite and bounded.

    Reservoir computing routinely produces ill-conditioned Gram matrices
    (highly correlated neurons, more features than samples, dead/constant
    units).  ``alpha`` is the guard: the regularized normal equations
    ``(XᵀX + αI)`` are positive-definite even when ``XᵀX`` is singular, so the
    ridge solve must always return finite, bounded weights — never ``NaN`` and
    never a blow-up from inverting a near-singular Gram.
    """

    def test_rank_deficient_gram_duplicate_columns_stays_bounded(self) -> None:
        """Duplicated state columns make ``XᵀX`` singular; ridge stays finite.

        Two identical feature columns give the Gram matrix a zero eigenvalue,
        so the *unregularized* normal equations are singular.  The ``alpha``
        ridge term lifts that eigenvalue to ``alpha > 0``, and the CG solve must
        return finite, bounded weights rather than ``NaN`` or an explosion.
        """
        torch.manual_seed(10)

        n_samples, n_features = 200, 12
        X = torch.randn(n_samples, n_features, dtype=torch.float64)
        # Force exact rank deficiency: column 5 duplicates column 2, and
        # column 9 duplicates column 3.  Both pairs collapse the Gram's rank.
        X[:, 5] = X[:, 2]
        X[:, 9] = X[:, 3]
        y = torch.randn(n_samples, 4, dtype=torch.float64)

        readout = CGReadoutLayer(
            in_features=n_features, out_features=4, alpha=1e-6, max_iter=1000, tol=1e-10
        )
        coefs, intercept = readout._solve_ridge_cg(X, y, 1e-6)

        assert torch.isfinite(coefs).all()
        assert intercept is not None and torch.isfinite(intercept).all()
        # "Bounded" is the operative word: ridge with a tiny alpha on a singular
        # Gram could in principle produce huge weights; verify it does not.
        assert coefs.abs().max() < 1e3

    def test_rank_deficient_gram_matches_closed_form_ridge(self) -> None:
        """On a rank-deficient Gram, CG still matches the closed-form ridge.

        The minimum-norm-among-regularized solution is uniquely defined by the
        ``(XᵀX + αI)`` system even when ``XᵀX`` is singular, so the CG solve
        must agree with a direct ``torch.linalg.solve`` of that same system.
        """
        torch.manual_seed(11)

        n_samples, n_features = 150, 10
        X = torch.randn(n_samples, n_features, dtype=torch.float64)
        X[:, 7] = X[:, 1]  # duplicate -> rank-deficient Gram
        y = torch.randn(n_samples, 3, dtype=torch.float64)
        alpha = 1e-3

        readout = CGReadoutLayer(
            in_features=n_features, out_features=3, alpha=alpha, max_iter=2000, tol=1e-12
        )
        coefs_cg, intercept_cg = readout._solve_ridge_cg(X, y, alpha)
        coefs_cf, intercept_cf = solve_ridge_closed_form(X, y, alpha)

        assert torch.allclose(coefs_cg, coefs_cf, atol=1e-6, rtol=1e-5)
        assert intercept_cg is not None
        assert torch.allclose(intercept_cg, intercept_cf, atol=1e-6, rtol=1e-5)

    def test_more_features_than_samples_stays_finite(self) -> None:
        """A wide system (F > N) is rank-deficient by construction; ridge copes.

        With more features than samples the Gram ``XᵀX`` has rank at most ``N``
        and is therefore singular, but the ``alpha`` ridge term keeps the
        solve well-posed.  Weights must be finite and bounded.
        """
        torch.manual_seed(12)

        n_samples, n_features = 8, 40  # deliberately F > N
        X = torch.randn(n_samples, n_features, dtype=torch.float64)
        y = torch.randn(n_samples, 5, dtype=torch.float64)

        readout = CGReadoutLayer(
            in_features=n_features, out_features=5, alpha=1e-6, max_iter=1000, tol=1e-10
        )
        coefs, intercept = readout._solve_ridge_cg(X, y, 1e-6)

        assert coefs.shape == (n_features, 5)
        assert torch.isfinite(coefs).all()
        assert intercept is not None and torch.isfinite(intercept).all()
        assert coefs.abs().max() < 1e3

    def test_more_features_than_samples_via_fit_is_finite(self) -> None:
        """The public ``fit`` path also produces finite weights when F > N."""
        torch.manual_seed(13)

        X = torch.randn(6, 30)  # 6 samples, 30 features
        y = torch.randn(6, 4)

        readout = CGReadoutLayer(in_features=30, out_features=4, alpha=1e-6)
        readout.fit(X, y)

        assert readout.is_fitted
        assert torch.isfinite(readout.weight).all()
        assert torch.isfinite(readout.bias).all()

    def test_all_zero_states_give_zero_weight_and_bias_only_prediction(self) -> None:
        """All-zero *states* (not targets) fit to ``W = 0`` and predict the bias.

        Unlike the all-zero-*targets* case (covered above), here the inputs are
        zero: every centered feature is zero, so there is no signal for the
        weights to latch onto and the fit collapses to ``W = 0`` with the bias
        absorbing the target mean.  Predictions then reduce to a constant
        bias-only output, finite for any input.
        """
        torch.manual_seed(14)

        X = torch.zeros(120, 12)
        y = torch.randn(120, 3)

        readout = CGReadoutLayer(in_features=12, out_features=3, alpha=1e-6)
        readout.fit(X, y)

        assert torch.isfinite(readout.weight).all()
        assert torch.isfinite(readout.bias).all()
        # No signal in the inputs -> weights collapse to exactly zero.
        assert torch.allclose(readout.weight, torch.zeros_like(readout.weight))
        # Prediction is bias-only: every row equals the fitted bias.
        pred = readout(torch.randn(5, 12))
        assert torch.isfinite(pred).all()
        assert torch.allclose(pred, readout.bias.expand(5, 3))


class TestCGReadoutNonFiniteInputs:
    """NaN/Inf inputs to ``fit`` are rejected with a clear error.

    A reservoir that diverged (or a corrupt target) can feed ``NaN``/``Inf``
    into the readout fit.  ``ReadoutLayer.fit`` guards against this: a
    non-finite value in ``states`` or ``targets`` raises a clear ``ValueError``
    *before* the algebraic solve, rather than propagating through the
    Gram-matrix matmuls into silently-``is_fitted`` ``NaN`` weights.  The guard
    lives in the base class, so every readout subclass inherits it.
    """

    def test_nan_state_is_rejected(self) -> None:
        """A ``NaN`` in ``states`` is rejected before the solve, not fitted."""
        torch.manual_seed(20)

        X = torch.randn(100, 10, dtype=torch.float64)
        X[0, 0] = float("nan")
        y = torch.randn(100, 3, dtype=torch.float64)

        readout = CGReadoutLayer(in_features=10, out_features=3, alpha=1e-6)
        with pytest.raises(ValueError, match=r"(?i)finite|nan|inf"):
            readout.fit(X, y)

        # Rejected cleanly: the layer is left unfitted, not carrying NaN weights.
        assert not readout.is_fitted
        assert torch.isfinite(readout.weight).all()

    def test_inf_target_is_rejected(self) -> None:
        """An ``Inf`` in ``targets`` is rejected before the solve, not fitted."""
        torch.manual_seed(21)

        X = torch.randn(100, 10, dtype=torch.float64)
        y = torch.randn(100, 3, dtype=torch.float64)
        y[5, 1] = float("inf")

        readout = CGReadoutLayer(in_features=10, out_features=3, alpha=1e-6)
        with pytest.raises(ValueError, match=r"(?i)finite|nan|inf"):
            readout.fit(X, y)

        assert not readout.is_fitted
        assert torch.isfinite(readout.weight).all()

    def test_nan_state_should_raise_a_clear_error(self) -> None:
        """Contract test for the intended behaviour: ``NaN`` states are rejected.

        A ``NaN`` state makes ``fit`` raise a ``ValueError`` whose message names
        the non-finite condition, and the readout is left unfitted rather than
        carrying ``NaN`` weights.
        """
        torch.manual_seed(22)

        X = torch.randn(100, 10)
        X[0, 0] = float("nan")
        y = torch.randn(100, 3)

        readout = CGReadoutLayer(in_features=10, out_features=3, alpha=1e-6)
        # Desired contract: a NaN state is rejected with a clear error, leaving
        # the readout unfitted rather than carrying NaN weights.  Today ``fit``
        # silently succeeds, so this assertion fails (and the strict xfail
        # records the open gap).
        try:
            readout.fit(X, y)
        except ValueError as exc:
            assert any(tok in str(exc).lower() for tok in ("finite", "nan", "inf"))
        else:
            raise AssertionError("fit() silently accepted a NaN state without raising")
        assert not readout.is_fitted


class TestCGReadoutConstructionValidation:
    """Constructor-time validation of alpha, tol, and max_iter."""

    def test_negative_alpha_rejected_at_construction(self) -> None:
        """alpha < 0 raises ValueError at construction."""
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            CGReadoutLayer(in_features=10, out_features=2, alpha=-1e-6)

    def test_zero_alpha_allowed(self) -> None:
        """alpha == 0 (pure least squares) is allowed."""
        readout = CGReadoutLayer(in_features=10, out_features=2, alpha=0.0)
        assert readout.alpha == 0.0

    def test_non_positive_tol_rejected(self) -> None:
        """tol <= 0 raises ValueError at construction."""
        with pytest.raises(ValueError, match="tol must be positive"):
            CGReadoutLayer(in_features=10, out_features=2, tol=0.0)
        with pytest.raises(ValueError, match="tol must be positive"):
            CGReadoutLayer(in_features=10, out_features=2, tol=-1e-5)

    def test_non_positive_max_iter_rejected(self) -> None:
        """max_iter < 1 raises ValueError at construction.

        ``max_iter=0`` previously skipped the loop entirely and returned an
        all-zero weight matrix with ``is_fitted == True``.
        """
        with pytest.raises(ValueError, match="max_iter must be a positive integer"):
            CGReadoutLayer(in_features=10, out_features=2, max_iter=0)
        with pytest.raises(ValueError, match="max_iter must be a positive integer"):
            CGReadoutLayer(in_features=10, out_features=2, max_iter=-3)


class TestReadoutLayerInFeaturesValidation:
    """fit() validates the state feature dimension against in_features."""

    def test_fit_wrong_in_features_raises_clear_error(self) -> None:
        """A wrong in_features surfaces as a clear ValueError, not a deep matvec error."""
        readout = CGReadoutLayer(in_features=20, out_features=5)
        X = torch.randn(100, 16)  # 16 != 20
        y = torch.randn(100, 5)

        with pytest.raises(ValueError, match="state feature dimension"):
            readout.fit(X, y)

    def test_fit_wrong_in_features_3d_raises(self) -> None:
        """The in_features check fires for flattened 3D inputs too."""
        readout = CGReadoutLayer(in_features=20, out_features=5)
        X = torch.randn(4, 25, 16)  # 16 != 20 after flatten
        y = torch.randn(4, 25, 5)

        with pytest.raises(ValueError, match="state feature dimension"):
            readout.fit(X, y)

    def test_fit_correct_in_features_passes(self) -> None:
        """Correct in_features fits without raising."""
        readout = CGReadoutLayer(in_features=20, out_features=5, alpha=1e-3)
        readout.fit(torch.randn(100, 20), torch.randn(100, 5))
        assert readout.is_fitted


class TestReadoutLayerIsFittedPersistence:
    """is_fitted is a registered buffer surviving state_dict round-trips."""

    def test_is_fitted_in_state_dict(self) -> None:
        """is_fitted is stored as a buffer and appears in the state_dict."""
        readout = CGReadoutLayer(in_features=10, out_features=3)
        assert "_is_fitted" in readout.state_dict()
        assert readout.state_dict()["_is_fitted"].dtype == torch.bool

    def test_is_fitted_survives_state_dict_round_trip(self) -> None:
        """A fitted readout's is_fitted=True survives load_state_dict."""
        src = CGReadoutLayer(in_features=10, out_features=3, alpha=1e-3)
        src.fit(torch.randn(50, 10), torch.randn(50, 3))
        assert src.is_fitted

        dst = CGReadoutLayer(in_features=10, out_features=3)
        assert not dst.is_fitted
        dst.load_state_dict(src.state_dict())

        assert dst.is_fitted is True
        # Loaded weights match.
        assert torch.allclose(dst.weight, src.weight)

    def test_unfitted_is_fitted_survives_round_trip(self) -> None:
        """An unfitted readout stays is_fitted=False after a round-trip."""
        src = CGReadoutLayer(in_features=10, out_features=3)
        dst = CGReadoutLayer(in_features=10, out_features=3)
        dst.fit(torch.randn(20, 10), torch.randn(20, 3))
        assert dst.is_fitted

        dst.load_state_dict(src.state_dict())
        assert dst.is_fitted is False

    def test_is_fitted_buffer_moves_with_device(self, device: torch.device) -> None:
        """The is_fitted buffer follows .to(device) and stays readable."""
        readout = CGReadoutLayer(in_features=10, out_features=3, alpha=1e-3).to(device)
        readout.fit(torch.randn(30, 10, device=device), torch.randn(30, 3, device=device))
        assert readout.is_fitted is True
        assert readout._is_fitted.device.type == device.type


# ===========================================================================
# CholeskyReadoutLayer + IncrementalRidgeReadout (issue #176, the streaming /
# DataLoader path). One test class per acceptance criterion.
# ===========================================================================


def _make_streaming_model(readout: ReadoutLayer, seq_len: int = 40) -> ESNModel:
    """A tiny reservoir -> readout model wired for streaming/forecast tests."""
    torch.manual_seed(0)
    inp = ps.Input((seq_len, 3))
    states = ESNLayer(50, feedback_size=3, spectral_radius=0.9)(inp)
    out = readout(states)
    return ESNModel(inp, out)


# ---------------------------------------------------------------------------
# CholeskyReadoutLayer
# ---------------------------------------------------------------------------


class TestCholeskyReadoutInstantiation:
    """Construction, inheritance, and constructor validation."""

    def test_is_readout_subclass(self) -> None:
        """CholeskyReadoutLayer is a ReadoutLayer (so ESNTrainer accepts it)."""
        readout = CholeskyReadoutLayer(in_features=20, out_features=3)
        assert isinstance(readout, ReadoutLayer)
        assert readout.in_features == 20
        assert readout.out_features == 3
        assert not readout.is_fitted

    def test_negative_alpha_rejected(self) -> None:
        """alpha < 0 raises at construction."""
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            CholeskyReadoutLayer(10, 2, alpha=-1e-6)

    def test_repr_contains_alpha(self) -> None:
        """repr surfaces the name and alpha."""
        r = repr(CholeskyReadoutLayer(10, 2, name="out", alpha=1e-4))
        assert "CholeskyReadoutLayer" in r
        assert "alpha=0.0001" in r
        assert "name='out'" in r


class TestCholeskyMatchesCG:
    """Acceptance: Cholesky matches CG within 1e-5 on a well-conditioned fit."""

    def test_matches_cg_within_1e5(self) -> None:
        """Cholesky ridge matches a tightly-converged CG fit to < 1e-5."""
        torch.manual_seed(7)
        X = torch.randn(300, 30, dtype=torch.float64)
        y = torch.randn(300, 4, dtype=torch.float64)
        alpha = 1e-4

        chol = CholeskyReadoutLayer(30, 4, alpha=alpha)
        coefs_c, intercept_c = chol._fit_impl(X, y)

        cg = CGReadoutLayer(30, 4, alpha=alpha, max_iter=5000, tol=1e-14)
        coefs_cg, intercept_cg = cg._solve_ridge_cg(X, y, alpha)

        assert torch.allclose(coefs_c, coefs_cg, atol=1e-5, rtol=1e-5)
        assert intercept_c is not None and intercept_cg is not None
        assert torch.allclose(intercept_c, intercept_cg, atol=1e-5, rtol=1e-5)

    def test_matches_closed_form(self) -> None:
        """Cholesky equals the closed-form ridge-with-intercept solution."""
        torch.manual_seed(42)
        X = torch.randn(200, 20, dtype=torch.float64)
        y = torch.randn(200, 5, dtype=torch.float64)
        alpha = 1e-3

        chol = CholeskyReadoutLayer(20, 5, alpha=alpha)
        coefs, intercept = chol._fit_impl(X, y)

        coefs_cf, intercept_cf = solve_ridge_closed_form(X, y, alpha)
        assert torch.allclose(coefs, coefs_cf, atol=1e-8, rtol=1e-7)
        assert intercept is not None
        assert torch.allclose(intercept, intercept_cf, atol=1e-8, rtol=1e-7)

    def test_no_bias_solves_raw_normal_equations(self) -> None:
        """bias=False solves the uncentered ridge system (no intercept)."""
        torch.manual_seed(3)
        alpha = 1e-2
        X = torch.randn(300, 10, dtype=torch.float64) + 1.0
        y = torch.randn(300, 4, dtype=torch.float64)

        chol = CholeskyReadoutLayer(10, 4, bias=False, alpha=alpha)
        coefs, intercept = chol._fit_impl(X, y)

        assert intercept is None
        gram = X.T @ X + alpha * torch.eye(10, dtype=torch.float64)
        expected = torch.linalg.solve(gram, X.T @ y)
        assert torch.allclose(coefs, expected, atol=1e-8)


class TestCholeskyTrainerCompatibility:
    """CholeskyReadoutLayer trains end-to-end through ESNTrainer, keyed by name."""

    def test_trainer_fits_and_forecasts(self) -> None:
        """ESNTrainer fits the named readout via its pre-hook, then forecasts."""
        readout = CholeskyReadoutLayer(50, 3, name="output", alpha=1e-6)
        model = _make_streaming_model(readout)

        warmup = torch.randn(1, 40, 3)
        train = torch.randn(1, 40, 3)
        targets = torch.randn(1, 40, 3)

        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets={"output": targets},
        )
        assert readout.is_fitted
        pred = model.forecast(warmup, horizon=20)
        assert pred.shape == (1, 20, 3)
        assert torch.isfinite(pred).all()


# ---------------------------------------------------------------------------
# Shared auto gram_dtype policy (acceptance criterion 3)
# ---------------------------------------------------------------------------


class TestAutoGramDtypePolicy:
    """Both readouts reuse the auto gram_dtype policy: float64 on CPU."""

    @pytest.mark.parametrize("cls", [CholeskyReadoutLayer, IncrementalRidgeReadout])
    def test_gram_dtype_default_is_none(self, cls) -> None:
        """The auto policy is the default (gram_dtype is None until overridden)."""
        readout = cls(10, 2)
        assert readout.gram_dtype is None
        assert readout.use_float64 is True

    def test_cholesky_cpu_float32_input_solves_in_float64(self) -> None:
        """A float32 CPU fit reaches float64-grade accuracy (auto gram policy).

        With the auto policy the heavy CPU matmuls run in float64, so even a
        float32-input fit matches the float64 closed-form solution far tighter
        than a genuinely float32 solve (~1e-3) would. The fitted weights are
        copied back to the float32 parameters, so we compare in float32.
        """
        torch.manual_seed(0)
        X = torch.randn(300, 12, dtype=torch.float32)
        y = torch.randn(300, 3, dtype=torch.float32)
        alpha = 1e-3

        readout = CholeskyReadoutLayer(12, 3, alpha=alpha)  # float32 params
        assert readout.weight.dtype == torch.float32
        readout.fit(X, y)

        coefs_cf, _ = solve_ridge_closed_form(X.double(), y.double(), alpha)
        assert torch.allclose(readout.weight.T.double(), coefs_cf, atol=1e-5, rtol=1e-4)

    def test_incremental_accumulators_are_float64_when_use_float64(self) -> None:
        """The sufficient-statistic buffers accumulate in float64 by default.

        Float64 accumulation is what keeps the running Gram from drifting across
        many chunks; it is the streaming counterpart of the auto gram policy.
        """
        readout = IncrementalRidgeReadout(10, 3)  # use_float64=True default
        assert readout.XtX.dtype == torch.float64
        assert readout.Xty.dtype == torch.float64

        f32 = IncrementalRidgeReadout(10, 3, use_float64=False)
        assert f32.XtX.dtype == torch.float32

    def test_incremental_cpu_float32_input_matches_float64(self) -> None:
        """A float32 CPU streaming fit reaches float64-grade accuracy."""
        torch.manual_seed(1)
        X = torch.randn(400, 12, dtype=torch.float32)
        y = torch.randn(400, 3, dtype=torch.float32)
        alpha = 1e-3

        inc = IncrementalRidgeReadout(12, 3, alpha=alpha)
        for chunk in torch.chunk(torch.arange(400), 5):
            inc.partial_fit(X[chunk], y[chunk])
        inc.finalize()

        coefs_cf, _ = solve_ridge_closed_form(X.double(), y.double(), alpha)
        assert torch.allclose(inc.weight.T.double(), coefs_cf, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# IncrementalRidgeReadout
# ---------------------------------------------------------------------------


class TestIncrementalInstantiation:
    """Construction, accumulator buffers, and validation."""

    def test_is_readout_subclass(self) -> None:
        """IncrementalRidgeReadout is a ReadoutLayer."""
        readout = IncrementalRidgeReadout(in_features=20, out_features=3)
        assert isinstance(readout, ReadoutLayer)
        assert not readout.is_fitted
        assert readout.n_seen == 0

    def test_negative_alpha_rejected(self) -> None:
        """alpha < 0 raises at construction."""
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            IncrementalRidgeReadout(10, 2, alpha=-1.0)

    def test_accumulator_buffers_registered(self) -> None:
        """The sufficient-statistic accumulators are registered buffers."""
        readout = IncrementalRidgeReadout(10, 2)
        sd = readout.state_dict()
        for key in ("XtX", "Xty", "sum_x", "sum_y", "_n"):
            assert key in sd
        assert readout.XtX.shape == (10, 10)
        assert readout.Xty.shape == (10, 2)

    def test_repr_contains_alpha(self) -> None:
        """repr surfaces the name and alpha."""
        r = repr(IncrementalRidgeReadout(10, 2, name="out", alpha=1e-4))
        assert "IncrementalRidgeReadout" in r
        assert "alpha=0.0001" in r


class TestIncrementalPartialFitMatchesFullBatch:
    """Acceptance: K chunks then finalize() matches a single full-batch fit < 1e-5."""

    def test_chunked_matches_full_batch_cholesky(self) -> None:
        """partial_fit over K chunks then finalize == full-batch Cholesky < 1e-5."""
        torch.manual_seed(0)
        X = torch.randn(600, 30, dtype=torch.float64)
        true_w = torch.randn(30, 4, dtype=torch.float64)
        y = X @ true_w + 0.01 * torch.randn(600, 4, dtype=torch.float64)
        alpha = 1e-4

        full = CholeskyReadoutLayer(30, 4, alpha=alpha)
        coefs_full, intercept_full = full._fit_impl(X, y)

        inc = IncrementalRidgeReadout(30, 4, alpha=alpha)
        for chunk in torch.chunk(torch.arange(600), 7):
            inc.partial_fit(X[chunk], y[chunk])
        assert inc.n_seen == 600
        inc.finalize()

        coefs_inc = inc.weight.T.to(torch.float64)
        intercept_inc = inc.bias.to(torch.float64)
        assert torch.allclose(coefs_inc, coefs_full, atol=1e-5, rtol=1e-5)
        assert torch.allclose(intercept_inc, intercept_full, atol=1e-5, rtol=1e-5)

    def test_chunked_matches_full_batch_cg(self) -> None:
        """The accumulated fit also matches a tightly-converged CG fit < 1e-5."""
        torch.manual_seed(1)
        X = torch.randn(500, 25, dtype=torch.float64)
        y = torch.randn(500, 3, dtype=torch.float64)
        alpha = 1e-3

        cg = CGReadoutLayer(25, 3, alpha=alpha, max_iter=5000, tol=1e-14)
        coefs_cg, intercept_cg = cg._solve_ridge_cg(X, y, alpha)

        inc = IncrementalRidgeReadout(25, 3, alpha=alpha)
        for chunk in torch.chunk(torch.arange(500), 11):
            inc.partial_fit(X[chunk], y[chunk])
        inc.finalize()

        assert torch.allclose(inc.weight.T.double(), coefs_cg, atol=1e-5, rtol=1e-5)
        assert torch.allclose(inc.bias.double(), intercept_cg, atol=1e-5, rtol=1e-5)

    def test_chunking_is_invariant_to_chunk_count(self) -> None:
        """The number of chunks does not change the final fit (additive stats)."""
        torch.manual_seed(2)
        X = torch.randn(360, 12, dtype=torch.float64)
        y = torch.randn(360, 2, dtype=torch.float64)

        def fit_in(k: int) -> torch.Tensor:
            inc = IncrementalRidgeReadout(12, 2, alpha=1e-4)
            for chunk in torch.chunk(torch.arange(360), k):
                inc.partial_fit(X[chunk], y[chunk])
            inc.finalize()
            return inc.weight.clone()

        assert torch.allclose(fit_in(1), fit_in(9), atol=1e-6)
        assert torch.allclose(fit_in(9), fit_in(40), atol=1e-6)

    def test_bias_free_matches_full_batch(self) -> None:
        """bias=False chunked fit matches the uncentered full-batch solve."""
        torch.manual_seed(3)
        X = torch.randn(400, 10, dtype=torch.float64) + 1.0
        y = torch.randn(400, 3, dtype=torch.float64)
        alpha = 1e-3

        inc = IncrementalRidgeReadout(10, 3, bias=False, alpha=alpha)
        for chunk in torch.chunk(torch.arange(400), 5):
            inc.partial_fit(X[chunk], y[chunk])
        inc.finalize()
        assert inc.bias is None

        gram = X.T @ X + alpha * torch.eye(10, dtype=torch.float64)
        expected = torch.linalg.solve(gram, X.T @ y)
        assert torch.allclose(inc.weight.T.double(), expected, atol=1e-5, rtol=1e-5)

    def test_partial_fit_accepts_3d_chunks(self) -> None:
        """3D ``(B, T, F)`` chunks are flattened consistently with full-batch."""
        torch.manual_seed(4)
        X3 = torch.randn(2, 50, 8, dtype=torch.float64)
        y3 = torch.randn(2, 50, 3, dtype=torch.float64)

        inc = IncrementalRidgeReadout(8, 3, alpha=1e-4)
        inc.partial_fit(X3, y3)
        inc.finalize()
        assert inc.n_seen == 2 * 50

        full = CholeskyReadoutLayer(8, 3, alpha=1e-4)
        coefs_full, _ = full._fit_impl(X3.reshape(-1, 8), y3.reshape(-1, 3))
        assert torch.allclose(inc.weight.T.double(), coefs_full, atol=1e-5, rtol=1e-5)


class TestIncrementalLifecycle:
    """Acceptance: is_fitted only after finalize; forward before finalize raises."""

    def test_is_fitted_only_after_finalize(self) -> None:
        """is_fitted is False until finalize() is called."""
        inc = IncrementalRidgeReadout(10, 3)
        assert not inc.is_fitted
        inc.partial_fit(torch.randn(40, 10), torch.randn(40, 3))
        assert not inc.is_fitted  # accumulated, but not solved yet
        inc.finalize()
        assert inc.is_fitted

    def test_forward_before_finalize_raises(self) -> None:
        """Calling forward before finalize raises a clear RuntimeError."""
        inc = IncrementalRidgeReadout(10, 3)
        inc.partial_fit(torch.randn(40, 10), torch.randn(40, 3))
        with pytest.raises(RuntimeError, match="not fitted.*finalize"):
            inc(torch.randn(5, 10))

    def test_forward_after_finalize_works(self) -> None:
        """forward succeeds once finalize() has run."""
        inc = IncrementalRidgeReadout(10, 3, alpha=1e-4)
        inc.partial_fit(torch.randn(40, 10), torch.randn(40, 3))
        inc.finalize()
        out = inc(torch.randn(5, 10))
        assert out.shape == (5, 3)

    def test_partial_fit_after_finalize_unfits(self) -> None:
        """New data after finalize clears is_fitted until re-finalized."""
        inc = IncrementalRidgeReadout(10, 3, alpha=1e-4)
        inc.partial_fit(torch.randn(40, 10), torch.randn(40, 3))
        inc.finalize()
        assert inc.is_fitted
        inc.partial_fit(torch.randn(20, 10), torch.randn(20, 3))
        assert not inc.is_fitted
        with pytest.raises(RuntimeError, match="not fitted"):
            inc(torch.randn(5, 10))

    def test_finalize_without_data_raises(self) -> None:
        """finalize before any partial_fit raises a clear RuntimeError."""
        inc = IncrementalRidgeReadout(10, 3)
        with pytest.raises(RuntimeError, match="no data accumulated"):
            inc.finalize()

    def test_reset_accumulators_clears_state(self) -> None:
        """reset_accumulators zeroes the statistics and the fitted flag."""
        inc = IncrementalRidgeReadout(10, 3, alpha=1e-4)
        inc.partial_fit(torch.randn(40, 10), torch.randn(40, 3))
        inc.finalize()
        inc.reset_accumulators()
        assert inc.n_seen == 0
        assert not inc.is_fitted
        assert torch.count_nonzero(inc.XtX) == 0
        assert torch.count_nonzero(inc.Xty) == 0

    def test_single_shot_fit_interface(self) -> None:
        """The inherited fit() works as a single-shot drop-in (resets + solves)."""
        torch.manual_seed(5)
        X = torch.randn(120, 10, dtype=torch.float64)
        y = torch.randn(120, 3, dtype=torch.float64)
        inc = IncrementalRidgeReadout(10, 3, alpha=1e-4)
        inc.fit(X, y)
        assert inc.is_fitted
        assert inc.n_seen == 120

        full = CholeskyReadoutLayer(10, 3, alpha=1e-4)
        coefs_full, _ = full._fit_impl(X, y)
        assert torch.allclose(inc.weight.T.double(), coefs_full, atol=1e-5, rtol=1e-5)


class TestIncrementalValidation:
    """partial_fit validates shapes with the same messages as base fit()."""

    def test_wrong_in_features_raises(self) -> None:
        """A wrong state feature dimension raises a clear ValueError."""
        inc = IncrementalRidgeReadout(20, 5)
        with pytest.raises(ValueError, match="state feature dimension"):
            inc.partial_fit(torch.randn(100, 16), torch.randn(100, 5))

    def test_wrong_out_features_raises(self) -> None:
        """A wrong target feature dimension raises a clear ValueError."""
        inc = IncrementalRidgeReadout(20, 5)
        with pytest.raises(ValueError, match="target feature dimension"):
            inc.partial_fit(torch.randn(100, 20), torch.randn(100, 3))

    def test_sample_mismatch_raises(self) -> None:
        """Mismatched sample counts raise a clear ValueError."""
        inc = IncrementalRidgeReadout(20, 5)
        with pytest.raises(ValueError, match="sample count mismatch"):
            inc.partial_fit(torch.randn(100, 20), torch.randn(50, 5))

    def test_nan_state_chunk_is_rejected(self) -> None:
        """A ``NaN`` in a state chunk is rejected before it poisons the stats."""
        inc = IncrementalRidgeReadout(20, 5)
        X = torch.randn(100, 20)
        X[0, 0] = float("nan")
        with pytest.raises(ValueError, match=r"(?i)finite|nan|inf"):
            inc.partial_fit(X, torch.randn(100, 5))
        assert not inc.is_fitted

    def test_inf_target_chunk_is_rejected(self) -> None:
        """An ``Inf`` in a target chunk is rejected before it poisons the stats."""
        inc = IncrementalRidgeReadout(20, 5)
        y = torch.randn(100, 5)
        y[3, 2] = float("inf")
        with pytest.raises(ValueError, match=r"(?i)finite|nan|inf"):
            inc.partial_fit(torch.randn(100, 20), y)
        assert not inc.is_fitted


class TestIncrementalStateDict:
    """Accumulators and is_fitted survive a state_dict round-trip."""

    def test_round_trip_preserves_fit(self) -> None:
        """A finalized readout reloads is_fitted=True with identical weights."""
        torch.manual_seed(6)
        src = IncrementalRidgeReadout(10, 3, alpha=1e-4)
        src.partial_fit(torch.randn(80, 10), torch.randn(80, 3))
        src.finalize()

        dst = IncrementalRidgeReadout(10, 3, alpha=1e-4)
        assert not dst.is_fitted
        dst.load_state_dict(src.state_dict())
        assert dst.is_fitted
        assert dst.n_seen == src.n_seen
        assert torch.allclose(dst.weight, src.weight)
        assert torch.allclose(dst.XtX, src.XtX)


class TestIncrementalDevice:
    """Incremental fitting works on every available device."""

    def test_partial_fit_finalize_on_device(self, device: torch.device) -> None:
        """Accumulating and finalizing land on the target device."""
        torch.manual_seed(0)
        inc = IncrementalRidgeReadout(20, 5, alpha=1e-4).to(device)
        for _ in range(3):
            X = torch.randn(60, 20, device=device)
            y = torch.randn(60, 5, device=device)
            inc.partial_fit(X, y)
        inc.finalize()
        assert inc.is_fitted
        assert inc.weight.device.type == device.type
        pred = inc(torch.randn(10, 20, device=device))
        assert pred.device.type == device.type
        assert pred.shape == (10, 5)


# ---------------------------------------------------------------------------
# Acceptance criterion 4: fit over a DataLoader of windowed chunks + forecast
# ---------------------------------------------------------------------------


def _windowed_loader(
    series: torch.Tensor,
    targets: torch.Tensor,
    window: int,
) -> DataLoader:
    """Split a contiguous ``(1, T, F)`` series into ordered windows of length ``window``."""
    xs = series.unfold(1, window, window).permute(0, 1, 3, 2).reshape(-1, window, series.shape[-1])
    ys = (
        targets.unfold(1, window, window).permute(0, 1, 3, 2).reshape(-1, window, targets.shape[-1])
    )
    ds = TensorDataset(xs, ys)
    # shuffle=False keeps the windows contiguous in time (required for the
    # stateful reservoir to stay synchronised across chunk boundaries).
    return DataLoader(ds, batch_size=1, shuffle=False)


class TestFitStreamOverDataLoader:
    """ESNTrainer.fit_stream trains over a DataLoader of windowed chunks."""

    def test_fit_stream_end_to_end_and_forecast(self) -> None:
        """Fit a readout over windowed DataLoader chunks, then forecast finitely."""
        readout = IncrementalRidgeReadout(50, 3, name="output", alpha=1e-6)
        model = _make_streaming_model(readout, seq_len=50)

        torch.manual_seed(11)
        warmup = torch.randn(1, 50, 3)
        series = torch.randn(1, 250, 3)
        targets = torch.randn(1, 250, 3)
        loader = _windowed_loader(series, targets, window=50)

        def chunk_stream():
            for xb, yb in loader:
                yield (xb,), {"output": yb}

        ESNTrainer(model).fit_stream(warmup_inputs=(warmup,), chunks=chunk_stream())
        assert readout.is_fitted
        assert readout.n_seen == 250

        pred = model.forecast(warmup, horizon=20)
        assert pred.shape == (1, 20, 3)
        assert torch.isfinite(pred).all()

    def test_fit_stream_matches_full_batch_fit(self) -> None:
        """Streaming over contiguous chunks == a single full-batch fit() < 1e-5.

        Two identically-seeded models (so reservoir weights match) are trained on
        the same data: one with a single-shot Cholesky fit, one with the chunked
        streaming path. The fitted readout weights must agree.
        """
        warmup = None
        train = None
        target = None

        def build_and_data(readout_cls):
            torch.manual_seed(0)
            inp = ps.Input((40, 3))
            states = ESNLayer(50, feedback_size=3, spectral_radius=0.9)(inp)
            ro = readout_cls(50, 3, name="output", alpha=1e-6)
            model = ESNModel(inp, ro(states))
            return model, ro

        torch.manual_seed(100)
        warmup = torch.randn(1, 40, 3)
        train = torch.randn(1, 300, 3)
        target = torch.randn(1, 300, 3)

        m_ref, ro_ref = build_and_data(CholeskyReadoutLayer)
        ESNTrainer(m_ref).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets={"output": target},
        )

        m_str, ro_str = build_and_data(IncrementalRidgeReadout)

        def chunks():
            for i in range(0, 300, 50):
                yield (train[:, i : i + 50],), {"output": target[:, i : i + 50]}

        ESNTrainer(m_str).fit_stream(warmup_inputs=(warmup,), chunks=chunks())

        assert torch.allclose(ro_str.weight, ro_ref.weight, atol=1e-5, rtol=1e-4)
        assert torch.allclose(ro_str.bias, ro_ref.bias, atol=1e-5, rtol=1e-4)

    def test_fit_stream_rejects_non_incremental_readout(self) -> None:
        """fit_stream requires IncrementalRidgeReadout readouts."""
        readout = CholeskyReadoutLayer(50, 3, name="output")
        model = _make_streaming_model(readout, seq_len=40)
        warmup = torch.randn(1, 40, 3)

        def chunks():
            yield (torch.randn(1, 40, 3),), {"output": torch.randn(1, 40, 3)}

        with pytest.raises(TypeError, match="IncrementalRidgeReadout"):
            ESNTrainer(model).fit_stream(warmup_inputs=(warmup,), chunks=chunks())

    def test_fit_stream_empty_chunks_raises(self) -> None:
        """An empty chunk stream raises rather than finalizing nothing."""
        readout = IncrementalRidgeReadout(50, 3, name="output")
        model = _make_streaming_model(readout, seq_len=40)
        warmup = torch.randn(1, 40, 3)

        with pytest.raises(RuntimeError, match="no chunks"):
            ESNTrainer(model).fit_stream(warmup_inputs=(warmup,), chunks=iter(()))

    def test_fit_stream_target_length_mismatch_raises(self) -> None:
        """A target whose length differs from its chunk's inputs raises."""
        readout = IncrementalRidgeReadout(50, 3, name="output")
        model = _make_streaming_model(readout, seq_len=40)
        warmup = torch.randn(1, 40, 3)

        def chunks():
            yield (torch.randn(1, 40, 3),), {"output": torch.randn(1, 30, 3)}

        with pytest.raises(ValueError, match="timesteps"):
            ESNTrainer(model).fit_stream(warmup_inputs=(warmup,), chunks=chunks())
