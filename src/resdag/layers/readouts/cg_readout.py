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

from .base_readout import ReadoutLayer


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
    use_float64 : bool, default=True
        If ``True`` (default), CG runs in ``float64`` for numerical
        stability and the result is cast back to the layer's parameter
        dtype.  Set to ``False`` to stay in the input dtype (typically
        ``float32``) when the doubled memory footprint matters and the
        inputs are already well-scaled — for example, very large reservoirs
        on GPU.

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
        use_float64: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.use_float64 = use_float64

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve ridge regression via Conjugate Gradient on flattened inputs.

        Parameters
        ----------
        states : torch.Tensor
            Flattened state matrix of shape ``(N, in_features)``.
        targets : torch.Tensor
            Flattened target matrix of shape ``(N, out_features)``.

        Returns
        -------
        coefs : torch.Tensor
            Coefficient matrix of shape ``(in_features, out_features)``.
        intercept : torch.Tensor
            Intercept vector of shape ``(out_features,)``.
        """
        return self._solve_ridge_cg(states, targets, self.alpha)

    def _solve_ridge_cg(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve ridge regression using Conjugate Gradient method."""
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")

        original_dtype = X.dtype
        # Upcast to float64 for numerical stability unless the caller opted out.
        # For very large reservoirs the doubled memory footprint can be the
        # bottleneck and the inputs are often already well-scaled.
        if self.use_float64:
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

        return coefs.to(original_dtype), intercept.to(original_dtype)

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
