"""
Conjugate Gradient Readout Layer
================================

This module provides :class:`CGReadoutLayer`, which extends
:class:`ReadoutLayer` with an efficient Conjugate Gradient solver
for ridge regression fitting.

See Also
--------
resdag.layers.readouts.ReadoutLayer : Base readout layer.
resdag.training.ESNTrainer : Trainer for fitting readout layers.
"""

import torch

from .base import ReadoutLayer


class CGReadoutLayer(ReadoutLayer):
    """
    Readout layer with Conjugate Gradient ridge regression solver.

    This layer extends :class:`ReadoutLayer` with an efficient Conjugate
    Gradient (CG) solver for fitting weights via ridge regression. The CG
    solver is:

    - Memory efficient (doesn't form full normal equations matrix)
    - GPU accelerated
    - Numerically stable (uses float64 internally)
    - Supports batched time-series data

    The solver finds weights W that minimize:

    .. math::

        ||XW - Y||^2 + \\alpha ||W||^2

    where :math:`\\alpha` is the regularization strength.

    Parameters
    ----------
    in_features : int
        Size of input features (reservoir state dimension).
    out_features : int
        Size of output features (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term.
    name : str, optional
        Name for this readout layer. Used for identification in
        multi-readout architectures and by :class:`ESNTrainer`.
    trainable : bool, default=False
        If True, weights are trainable via backpropagation.
    alpha : float, default=1e-6
        L2 regularization strength. Must be non-negative. Larger values
        provide more regularization (smoother outputs, less overfitting).
    max_iter : int, default=100
        Maximum number of CG iterations.
    tol : float, default=1e-5
        Convergence tolerance for CG solver. Iterations stop when
        residual norm squared is below ``tol**2``.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.nn.Parameter or None
        Bias vector of shape ``(out_features,)``, or None if ``bias=False``.
    alpha : float
        L2 regularization strength.
    max_iter : int
        Maximum CG iterations.
    tol : float
        Convergence tolerance.

    Examples
    --------
    Basic usage:

    >>> readout = CGReadoutLayer(in_features=100, out_features=10, alpha=1e-6)
    >>> states = torch.randn(32, 50, 100)  # (batch, time, features)
    >>> targets = torch.randn(32, 50, 10)
    >>> readout.fit(states, targets)
    >>> output = readout(states)
    >>> print(output.shape)
    torch.Size([32, 50, 10])

    With custom regularization:

    >>> readout = CGReadoutLayer(100, 10, alpha=1e-4)  # Stronger regularization
    >>> readout.fit(states, targets)

    See Also
    --------
    ReadoutLayer : Base readout layer class.
    resdag.training.ESNTrainer : Trainer that uses this for fitting.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        max_iter: int = 100,
        tol: float = 1e-5,
        alpha: float = 1e-6,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha

    def _solve_ridge_cg(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve ridge regression using Conjugate Gradient method."""
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")

        # Work in float64 for numerical stability
        X = X.to(torch.float64)
        y = y.to(torch.float64)

        # Center the data
        X_mean = X.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        n = float(X.shape[0])

        # Gram matrix of centered X
        XtX = X.T @ X - n * (X_mean.T @ X_mean)

        def matvec(w: torch.Tensor) -> torch.Tensor:
            """Matrix-vector product: (X^T X + alpha * I) @ w."""
            return XtX @ w + alpha * w

        def conjugate_gradient(
            A_func,
            B: torch.Tensor,
            max_iter: int,
            tol: float,
        ) -> torch.Tensor:
            """Solve A @ X = B using Conjugate Gradient."""
            X = torch.zeros_like(B)
            R = B - A_func(X)
            P = R.clone()
            Rs_old = (R * R).sum(dim=0)

            for _ in range(max_iter):
                if torch.all(Rs_old < tol**2):
                    break

                AP = A_func(P)
                alpha_cg = Rs_old / (P * AP).sum(dim=0)
                X = X + P * alpha_cg
                R = R - AP * alpha_cg
                Rs_new = (R * R).sum(dim=0)
                beta = Rs_new / Rs_old
                P = R + P * beta
                Rs_old = Rs_new

            return X

        # Right-hand side
        rhs = X.T @ y - n * (X_mean.T @ y_mean)

        # Solve using CG
        coefs = conjugate_gradient(matvec, rhs, self.max_iter, self.tol)

        # Compute intercept
        intercept = (y_mean - X_mean @ coefs).squeeze(0)

        return coefs, intercept

    def fit(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Fit readout weights using Conjugate Gradient ridge regression.

        Fits the readout layer to map input states to target outputs using
        L2-regularized least squares (ridge regression), solved via the
        Conjugate Gradient method.

        The method automatically:

        - Handles both 2D and 3D input tensors
        - Centers data for numerical stability
        - Computes optimal weights and bias
        - Updates layer parameters in-place

        Parameters
        ----------
        inputs : torch.Tensor
            Input states of shape ``(batch, time, features)`` or
            ``(n_samples, features)``. For 3D inputs, data is flattened
            to ``(batch * time, features)``.
        targets : torch.Tensor
            Target outputs of shape ``(batch, time, outputs)`` or
            ``(n_samples, outputs)``. Must have same number of samples
            as inputs after flattening.

        Raises
        ------
        ValueError
            If number of samples doesn't match between inputs and targets,
            or if target output dimension doesn't match ``out_features``.

        Notes
        -----
        After calling ``fit()``, the ``is_fitted`` property returns True.

        Examples
        --------
        Fit on batched time-series data:

        >>> readout = CGReadoutLayer(100, 10)
        >>> states = torch.randn(4, 200, 100)  # (batch, time, features)
        >>> targets = torch.randn(4, 200, 10)
        >>> readout.fit(states, targets)
        >>> print(readout.is_fitted)
        True

        Fit on flattened data:

        >>> states_flat = torch.randn(800, 100)  # (n_samples, features)
        >>> targets_flat = torch.randn(800, 10)
        >>> readout.fit(states_flat, targets_flat)
        """
        # Handle 3D inputs by reshaping to 2D
        if inputs.dim() == 3:
            batch_size, seq_len, features = inputs.shape
            inputs = inputs.reshape(batch_size * seq_len, features)

        if targets.dim() == 3:
            batch_size, seq_len, outputs = targets.shape
            targets = targets.reshape(batch_size * seq_len, outputs)

        # Validate shapes
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Number of samples must match: states has {inputs.shape[0]}, "
                f"targets has {targets.shape[0]}"
            )

        if targets.shape[1] != self.out_features:
            raise ValueError(
                f"Target output dimension ({targets.shape[1]}) must match "
                f"out_features ({self.out_features})"
            )

        # Solve ridge regression with CG
        coefs, intercept = self._solve_ridge_cg(inputs, targets, self.alpha)

        # Update parameters
        with torch.no_grad():
            self.weight.copy_(coefs.T.to(self.weight.dtype))
            if self.bias is not None:
                self.bias.copy_(intercept.to(self.bias.dtype))

        self._is_fitted = True

    def __repr__(self) -> str:
        """Return string representation."""
        name_str = f", name='{self._name}'" if self._name is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f"{name_str}, "
            f"alpha={self.alpha}, "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol}"
            f")"
        )
