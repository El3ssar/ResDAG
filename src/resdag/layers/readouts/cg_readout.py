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

from collections.abc import Callable

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
        Maximum number of CG iterations. Must be a positive integer.
    tol : float, default=1e-5
        Convergence tolerance for the CG solver. Must be positive. Each output
        column is considered converged when its residual-norm-squared falls
        below ``tol**2``.
    use_float64 : bool, default=True
        If ``True`` (default), the CG iterations on the small
        ``(in_features, in_features)`` system run in ``float64`` and the
        result is cast back to the layer's parameter dtype.  Cheap on every
        device — the expensive ``(N, F)`` Gram-formation matmuls stay in
        the input dtype regardless (see ``gram_dtype``).
    gram_dtype : torch.dtype, optional
        Dtype for forming the Gram matrix and right-hand side (the heavy
        ``(N, F)`` matmuls).  Default ``None`` is automatic: ``float64`` on
        CPU (cheap there), the input dtype on CUDA — float64 matmuls run at
        1/32–1/64 speed on consumer GPUs and are the classic reason ESN
        training measures *slower* on GPU than CPU.  Pass
        ``torch.float64`` to force full-precision Gram formation on any
        device (needed only for badly scaled states, e.g. unnormalized
        inputs concatenated into the readout; prefer normalizing the data).

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
        gram_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be a positive integer, got {max_iter}.")
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.use_float64 = use_float64
        self.gram_dtype = gram_dtype

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        intercept : torch.Tensor or None
            Intercept vector of shape ``(out_features,)``, or ``None`` when
            the layer was built with ``bias=False`` (the ridge problem is
            then solved without centering, i.e. with no intercept).
        """
        return self._solve_ridge_cg(states, targets, self.alpha)

    def _solve_ridge_cg(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve ridge regression using Conjugate Gradient method.

        When the layer has a bias, the data is centered and an unpenalized
        intercept is recovered afterwards (standard ridge-with-intercept).
        When built with ``bias=False`` the raw, uncentered normal equations
        are solved instead — centering without applying the intercept at
        predict time would systematically shift every prediction.

        Precision strategy: the heavy ``(N, F)`` matmuls that form the Gram
        matrix and right-hand side run in ``gram_dtype`` (default: the input
        dtype — float64 here is catastrophically slow on consumer GPUs),
        while the CG iterations on the small ``(F, F)`` system run in
        float64 when ``use_float64=True``. Pass ``gram_dtype=torch.float64``
        to also form the Gram in full precision (legacy behaviour; only
        matters for severely ill-conditioned state matrices).
        """
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")

        fit_intercept = self.bias is not None

        original_dtype = X.dtype
        solve_dtype = torch.float64 if self.use_float64 else X.dtype
        # Auto gram dtype: full precision where it's cheap (CPU), input
        # dtype where float64 throughput is crippled (consumer CUDA).
        gram_dtype = self.gram_dtype
        if gram_dtype is None:
            gram_dtype = solve_dtype if X.device.type == "cpu" else X.dtype
        if gram_dtype != X.dtype:
            X = X.to(gram_dtype)
            y = y.to(gram_dtype)

        if fit_intercept:
            # Center the data; the intercept is recovered after the solve.
            X_mean = X.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)
            n = float(X.shape[0])
            # Gram matrix of centered X
            XtX = X.T @ X - n * (X_mean.T @ X_mean)
        else:
            XtX = X.T @ X
        XtX = XtX.to(solve_dtype)

        def matvec(w: torch.Tensor) -> torch.Tensor:
            """Matrix-vector product: (X^T X + alpha * I) @ w."""
            return XtX @ w + alpha * w

        def conjugate_gradient(
            A_func: Callable[[torch.Tensor], torch.Tensor],
            B: torch.Tensor,
            max_iter: int,
            tol: float,
        ) -> torch.Tensor:
            """Solve ``A @ X = B`` column-by-column with Conjugate Gradient.

            The system is solved for every output column simultaneously; each
            column carries its own residual and step size. Three guards keep a
            degenerate column from poisoning the whole fitted weight matrix
            with ``NaN`` (the bug this method fixes):

            - **Per-column converged/degenerate mask.** A column is *frozen*
              once its residual reaches zero — either it started at zero (a
              trivial/all-zero target column) or it converged to exact zero.
              Frozen columns take a zero step and a zero ``beta`` and so never
              evaluate ``0 / 0``; an all-zero target column therefore fits to
              ``W = 0`` (no ``NaN``) instead of poisoning every other column.
            - **Floored ``alpha`` denominator.** ``(P · AP)`` is replaced by
              ``1`` wherever it is exactly zero before dividing, so a column
              whose search direction collapses (``denom == 0``) yields a finite
              step rather than ``NaN``. (For the symmetric positive-definite
              normal-equations operator solved here ``denom`` is only zero when
              ``P`` itself is zero — i.e. the column is already converged — so
              the floored step does not perturb the solution; the guard is
              defensive.)
            - **Floored ``beta`` divisor.** ``Rs_old`` is floored before
              forming ``beta = Rs_new / Rs_old`` so a column that converges to
              exactly zero never produces ``0 / 0`` for ``beta`` either.

            Convergence per column uses the residual-norm-squared test
            ``||r||^2 <= tol^2``. (Making ``tol`` scale-invariant is tracked
            separately as ``readouts-cg-relative-tolerance``; this ticket keeps
            the legacy stopping precision so existing fits are unchanged and
            only adds the degenerate-column ``NaN`` guards.)

            Returns
            -------
            torch.Tensor
                Solution ``X`` of the same shape as ``B``.
            """
            X = torch.zeros_like(B)
            R = B - A_func(X)
            P = R.clone()
            Rs_old = (R * R).sum(dim=0)

            tol_sq = tol * tol
            # ``frozen`` is the NaN guard: a column whose residual is exactly
            # zero (a zero/trivial target column, or one that hit exact
            # convergence) must not divide. A zero target column has
            # ``Rs_old == 0`` from the start and is frozen immediately.
            frozen = Rs_old == 0

            for it in range(max_iter):
                # The convergence test forces a device->host sync; checking
                # every iteration serializes the GPU. Every 10 is plenty —
                # at most 9 extra cheap (F, F) matvecs past convergence.
                if it % 10 == 0 and bool(torch.all((Rs_old <= tol_sq) | frozen)):
                    break

                AP = A_func(P)
                denom = (P * AP).sum(dim=0)
                # Replace an exactly-zero denominator with 1 so a collapsed
                # search direction (denom == 0) divides to a finite step rather
                # than NaN. For an SPD operator denom == 0 implies P == 0 (the
                # column has converged), so this only ever fires defensively.
                safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)
                alpha_cg = torch.where(frozen, torch.zeros_like(Rs_old), Rs_old / safe_denom)
                X = X + P * alpha_cg
                R = R - AP * alpha_cg
                Rs_new = (R * R).sum(dim=0)

                # Floor Rs_old before forming beta; frozen columns are masked to
                # beta = 0 regardless, so a converged/zero column never hits 0/0.
                safe_rs_old = torch.where(Rs_old == 0, torch.ones_like(Rs_old), Rs_old)
                beta = torch.where(frozen, torch.zeros_like(Rs_new), Rs_new / safe_rs_old)
                P = R + P * beta
                Rs_old = Rs_new
                # Freeze any column that has now reached exact zero residual so
                # its next iteration can't divide 0/0.
                frozen = frozen | (Rs_old == 0)

            return X

        # Right-hand side
        if fit_intercept:
            rhs = X.T @ y - n * (X_mean.T @ y_mean)
        else:
            rhs = X.T @ y
        rhs = rhs.to(solve_dtype)

        # Solve using CG
        coefs = conjugate_gradient(matvec, rhs, self.max_iter, self.tol)

        if fit_intercept:
            intercept = (y_mean.to(solve_dtype) - X_mean.to(solve_dtype) @ coefs).squeeze(0)
            return coefs.to(original_dtype), intercept.to(original_dtype)
        return coefs.to(original_dtype), None

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
