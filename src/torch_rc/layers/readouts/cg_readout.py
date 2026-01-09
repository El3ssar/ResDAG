"""Conjugate Gradient ReadoutLayer for ridge regression fitting."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import ReadoutLayer


class CGReadoutLayer(ReadoutLayer):
    """ReadoutLayer with Conjugate Gradient solver for ridge regression.

    This layer extends ReadoutLayer with an efficient Conjugate Gradient (CG)
    solver for fitting weights via ridge regression. The CG solver is:
    - Memory efficient (doesn't form normal equations explicitly)
    - GPU accelerated
    - Supports multiple outputs simultaneously

    Solves the regularized least squares problem:
        (X.T @ X + alpha * I) @ W = X.T @ Y

    The solver:
    - Centers the data automatically
    - Uses float64 precision for numerical stability
    - Computes bias term from centered data
    - Works with batched time-series data

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to use bias (default: True)
        name: Optional name for this readout
        trainable: Whether weights should be trainable (default: False)
        max_iter: Maximum CG iterations (default: 100)
        tol: Convergence tolerance (default: 1e-5)

    Example:
        >>> readout = CGReadoutLayer(in_features=100, out_features=10)
        >>> states = torch.randn(32, 50, 100)  # (batch, time, features)
        >>> targets = torch.randn(32, 50, 10)
        >>> readout.fit(states, targets, ridge=1e-6)
        >>> output = readout(states)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: Optional[str] = None,
        trainable: bool = False,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> None:
        """Initialize CGReadoutLayer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to use bias
            name: Optional name for identification
            trainable: Whether weights are trainable
            max_iter: Maximum iterations for CG solver
            tol: Convergence tolerance for CG solver
        """
        super().__init__(in_features, out_features, bias, name, trainable)
        self.max_iter = max_iter
        self.tol = tol

    def _solve_ridge_cg(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve Ridge Regression using Conjugate Gradient for multiple outputs.

        This method solves the regularized least squares problem:
            (X.T @ X + alpha * I) @ W = X.T @ Y

        Each output dimension is solved independently using a vectorized
        implementation of the Conjugate Gradient (CG) method. Scalar updates
        (alpha_cg, beta) are computed per output column using broadcasting.

        The data is centered automatically, and the corresponding bias term is returned.

        Args:
            X: Input tensor of shape (n_samples, n_features)
            y: Target tensor of shape (n_samples, n_outputs)
            alpha: L2 regularization strength (must be non-negative)

        Returns:
            coefs: Weight matrix of shape (n_features, n_outputs)
            intercept: Bias vector of shape (n_outputs,)

        Raises:
            ValueError: If alpha is negative
        """
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")

        # Work in float64 for numerical stability
        X = X.to(torch.float64)
        y = y.to(torch.float64)

        # Center the data
        X_mean = X.mean(dim=0, keepdim=True)  # (1, n_features)
        y_mean = y.mean(dim=0, keepdim=True)  # (1, n_outputs)
        n = float(X.shape[0])

        # Gram matrix of centered X: (X - μ_X)^T (X - μ_X) = X^T X - n * μ_X^T μ_X
        XtX = X.T @ X - n * (X_mean.T @ X_mean)

        def matvec(w: torch.Tensor) -> torch.Tensor:
            """Matrix-vector product: (X^T X + alpha * I) @ w"""
            return XtX @ w + alpha * w

        def conjugate_gradient(
            A_func,
            B: torch.Tensor,
            max_iter: int,
            tol: float,
        ) -> torch.Tensor:
            """Solve A @ X = B using Conjugate Gradient.

            Args:
                A_func: Function computing A @ x
                B: Right-hand side tensor of shape (n_features, n_outputs)
                max_iter: Maximum iterations
                tol: Convergence tolerance

            Returns:
                X: Solution tensor of shape (n_features, n_outputs)
            """
            X = torch.zeros_like(B)
            R = B - A_func(X)
            P = R.clone()
            Rs_old = (R * R).sum(dim=0)  # (n_outputs,)

            for i in range(max_iter):
                # Check convergence
                if torch.all(Rs_old < tol**2):
                    break

                AP = A_func(P)
                alpha_cg = Rs_old / (P * AP).sum(dim=0)  # (n_outputs,)
                X = X + P * alpha_cg
                R = R - AP * alpha_cg
                Rs_new = (R * R).sum(dim=0)  # (n_outputs,)
                beta = Rs_new / Rs_old
                P = R + P * beta
                Rs_old = Rs_new

            return X

        # Right-hand side: X^T @ y - n * μ_X^T @ μ_y
        rhs = X.T @ y - n * (X_mean.T @ y_mean)

        # Solve using CG
        coefs = conjugate_gradient(matvec, rhs, self.max_iter, self.tol)

        # Compute intercept
        intercept = (y_mean - X_mean @ coefs).squeeze(0)  # (n_outputs,)

        return coefs, intercept

    def fit(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        ridge: float = 1e-6,
    ) -> None:
        """Fit readout weights using Conjugate Gradient ridge regression.

        This method fits the readout layer to map states to targets using
        ridge regression solved via Conjugate Gradient. The method:
        - Handles both 2D (n_samples, features) and 3D (batch, time, features) inputs
        - Centers data automatically
        - Computes optimal weights and bias
        - Updates layer parameters in-place

        Args:
            states: Input states of shape (batch, time, features) or (n_samples, features)
            targets: Target outputs of shape (batch, time, outputs) or (n_samples, outputs)
            ridge: Ridge regularization parameter (alpha). Must be non-negative.

        Raises:
            ValueError: If input shapes don't match or ridge is negative
        """
        # Handle 3D inputs by reshaping to 2D
        if states.dim() == 3:
            batch_size, seq_len, features = states.shape
            states = states.reshape(batch_size * seq_len, features)

        if targets.dim() == 3:
            batch_size, seq_len, outputs = targets.shape
            targets = targets.reshape(batch_size * seq_len, outputs)

        # Validate shapes
        if states.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Number of samples must match: states has {states.shape[0]}, "
                f"targets has {targets.shape[0]}"
            )

        if targets.shape[1] != self.out_features:
            raise ValueError(
                f"Target output dimension ({targets.shape[1]}) must match "
                f"out_features ({self.out_features})"
            )

        # Solve ridge regression with CG
        coefs, intercept = self._solve_ridge_cg(states, targets, ridge)

        # Convert back to original dtype and update parameters
        with torch.no_grad():
            # coefs is (in_features, out_features), but nn.Linear expects (out_features, in_features)
            self.weight.copy_(coefs.T.to(self.weight.dtype))
            if self.bias is not None:
                self.bias.copy_(intercept.to(self.bias.dtype))

        # Mark as fitted
        self._is_fitted = True

    def __repr__(self) -> str:
        """String representation."""
        name_str = f", name='{self._name}'" if self._name is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f"{name_str}, "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol}"
            f")"
        )
