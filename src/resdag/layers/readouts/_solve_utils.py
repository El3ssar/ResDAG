"""
Shared Readout Solve Utilities
==============================

Precision, centering, and intercept-recovery helpers shared by every
algebraic readout solver (:class:`CGReadoutLayer`,
:class:`RidgeReadoutLayer`, :class:`SVDReadoutLayer`,
:class:`PinvReadoutLayer`).

The ridge-with-intercept problem every readout solves is

.. math::

    \\min_{W,\\,b} \\; \\lVert X W + b - Y \\rVert_2^2 + \\alpha \\lVert W \\rVert_2^2 ,

where the intercept :math:`b` is *not* penalised. The standard route is to
center :math:`X` and :math:`Y`, solve the penalised problem on the centered
data (which has no intercept), then recover :math:`b = \\bar{Y} - \\bar{X} W`.
When a readout is built with ``bias=False`` the raw, uncentered normal
equations are solved instead — centering without applying the intercept at
predict time would systematically shift every prediction.

This module factors that boilerplate out of the individual solvers so the
centering / precision logic lives in exactly one place. The solvers differ
only in *how* they invert the normal-equations system; everything around the
inversion is shared here.

Functions
---------
resolve_solve_dtype
    Pick the dtype the small ``(F, F)`` inversion runs in.
resolve_gram_dtype
    Pick the dtype the heavy ``(N, F)`` Gram-formation matmuls run in.
build_ridge_problem
    Center (when fitting an intercept), cast to ``gram_dtype``, and return a
    :class:`RidgeProblem` carrying the data and centering means.
recover_intercept
    Recover the unpenalised intercept from a coefficient matrix.

See Also
--------
resdag.layers.readouts.CGReadoutLayer : Conjugate-gradient ridge readout.
resdag.layers.readouts.RidgeReadoutLayer : Cholesky / solve direct readout.
resdag.layers.readouts.SVDReadoutLayer : SVD filter-factor readout.
resdag.layers.readouts.PinvReadoutLayer : lstsq / pseudo-inverse readout.
"""

from dataclasses import dataclass

import torch


def resolve_solve_dtype(input_dtype: torch.dtype, use_float64: bool) -> torch.dtype:
    """Pick the dtype the small ``(F, F)`` system is inverted in.

    Parameters
    ----------
    input_dtype : torch.dtype
        Dtype of the incoming state matrix.
    use_float64 : bool
        If ``True``, the inversion runs in ``float64`` regardless of the
        input dtype, then the result is cast back. This is cheap on every
        device — the system is only ``(F, F)`` — and is the default for
        numerical stability.

    Returns
    -------
    torch.dtype
        ``torch.float64`` when ``use_float64`` is set, else ``input_dtype``.
    """
    return torch.float64 if use_float64 else input_dtype


def resolve_gram_dtype(
    input_dtype: torch.dtype,
    device: torch.device,
    solve_dtype: torch.dtype,
    gram_dtype: torch.dtype | None,
) -> torch.dtype:
    """Pick the dtype the heavy ``(N, F)`` Gram-formation matmuls run in.

    The ``X.T @ X`` / ``X.T @ y`` products dominate the cost of an algebraic
    fit. Running them in ``float64`` is free on CPU but catastrophically slow
    on consumer GPUs (1/32–1/64 throughput), so the default is device-aware.

    Parameters
    ----------
    input_dtype : torch.dtype
        Dtype of the incoming state matrix.
    device : torch.device
        Device the state matrix lives on.
    solve_dtype : torch.dtype
        Dtype the small inversion runs in (see :func:`resolve_solve_dtype`).
    gram_dtype : torch.dtype or None
        Explicit override. When ``None`` (the default), full precision is used
        where it is cheap (CPU → ``solve_dtype``) and the input dtype is kept
        where ``float64`` throughput is crippled (CUDA → ``input_dtype``).

    Returns
    -------
    torch.dtype
        The resolved Gram-formation dtype.
    """
    if gram_dtype is not None:
        return gram_dtype
    return solve_dtype if device.type == "cpu" else input_dtype


@dataclass
class RidgeProblem:
    """Container for a prepared (optionally centered) ridge problem.

    Attributes
    ----------
    X : torch.Tensor
        State matrix in ``gram_dtype``, of shape ``(N, F)``. Not centered —
        centering is folded into :attr:`gram` / :attr:`rhs` analytically so
        the raw ``X`` can still be reused by SVD-style solvers.
    y : torch.Tensor
        Target matrix in ``gram_dtype``, of shape ``(N, T)``.
    gram : torch.Tensor
        The (optionally centered) Gram matrix ``XᵀX`` in ``solve_dtype``, of
        shape ``(F, F)``.
    rhs : torch.Tensor
        The (optionally centered) right-hand side ``Xᵀy`` in ``solve_dtype``,
        of shape ``(F, T)``.
    x_mean : torch.Tensor or None
        Column means of ``X`` of shape ``(1, F)`` when fitting an intercept,
        else ``None``.
    y_mean : torch.Tensor or None
        Column means of ``y`` of shape ``(1, T)`` when fitting an intercept,
        else ``None``.
    n_features : int
        Feature dimension ``F``.
    solve_dtype : torch.dtype
        Dtype the small inversion runs in.
    original_dtype : torch.dtype
        The input dtype to cast results back to.
    fit_intercept : bool
        Whether an intercept is being fitted.
    """

    X: torch.Tensor
    y: torch.Tensor
    gram: torch.Tensor
    rhs: torch.Tensor
    x_mean: torch.Tensor | None
    y_mean: torch.Tensor | None
    n_features: int
    solve_dtype: torch.dtype
    original_dtype: torch.dtype
    fit_intercept: bool


def build_ridge_problem(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    fit_intercept: bool,
    use_float64: bool,
    gram_dtype: torch.dtype | None,
) -> RidgeProblem:
    """Prepare the (optionally centered) normal-equations system.

    This performs the shared work every algebraic ridge solver needs before
    the actual inversion: resolve the working dtypes, cast the heavy matmul
    operands, and form the (centered, when fitting an intercept) Gram matrix
    ``XᵀX`` and right-hand side ``Xᵀy``.

    When ``fit_intercept`` is ``True`` the centered Gram and right-hand side
    are formed analytically — ``XᵀX - n x̄ᵀx̄`` and ``Xᵀy - n x̄ᵀȳ`` — which
    avoids materialising a centered copy of ``X`` and lets SVD-style solvers
    work from the same prepared problem.

    Parameters
    ----------
    X : torch.Tensor
        Flattened state matrix of shape ``(N, F)``.
    y : torch.Tensor
        Flattened target matrix of shape ``(N, T)``.
    fit_intercept : bool
        Whether to center the data and recover an unpenalised intercept later.
    use_float64 : bool
        Whether the small ``(F, F)`` inversion should run in ``float64``.
    gram_dtype : torch.dtype or None
        Explicit override for the Gram-formation dtype, or ``None`` for the
        device-aware default (see :func:`resolve_gram_dtype`).

    Returns
    -------
    RidgeProblem
        The prepared problem, carrying ``gram`` and ``rhs`` in ``solve_dtype``
        plus the centering means and bookkeeping needed to finalise the fit.
    """
    original_dtype = X.dtype
    solve_dtype = resolve_solve_dtype(original_dtype, use_float64)
    resolved_gram_dtype = resolve_gram_dtype(original_dtype, X.device, solve_dtype, gram_dtype)

    if resolved_gram_dtype != X.dtype:
        X = X.to(resolved_gram_dtype)
        y = y.to(resolved_gram_dtype)

    x_mean: torch.Tensor | None = None
    y_mean: torch.Tensor | None = None
    if fit_intercept:
        x_mean = X.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        n = float(X.shape[0])
        gram = X.T @ X - n * (x_mean.T @ x_mean)
        rhs = X.T @ y - n * (x_mean.T @ y_mean)
    else:
        gram = X.T @ X
        rhs = X.T @ y

    return RidgeProblem(
        X=X,
        y=y,
        gram=gram.to(solve_dtype),
        rhs=rhs.to(solve_dtype),
        x_mean=x_mean,
        y_mean=y_mean,
        n_features=X.shape[1],
        solve_dtype=solve_dtype,
        original_dtype=original_dtype,
        fit_intercept=fit_intercept,
    )


def recover_intercept(
    problem: RidgeProblem,
    coefs: torch.Tensor,
) -> torch.Tensor | None:
    """Recover the unpenalised intercept for a fitted coefficient matrix.

    Parameters
    ----------
    problem : RidgeProblem
        The prepared problem returned by :func:`build_ridge_problem`.
    coefs : torch.Tensor
        Coefficient matrix of shape ``(F, T)`` in ``problem.solve_dtype``.

    Returns
    -------
    torch.Tensor or None
        Intercept vector of shape ``(T,)``, or ``None`` when the problem was
        built with ``fit_intercept=False``.
    """
    if not problem.fit_intercept:
        return None
    assert problem.y_mean is not None and problem.x_mean is not None
    y_mean = problem.y_mean.to(problem.solve_dtype)
    x_mean = problem.x_mean.to(problem.solve_dtype)
    return (y_mean - x_mean @ coefs).squeeze(0)


def finalize_fit(
    problem: RidgeProblem,
    coefs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Recover the intercept and cast coefficients back to the input dtype.

    Parameters
    ----------
    problem : RidgeProblem
        The prepared problem returned by :func:`build_ridge_problem`.
    coefs : torch.Tensor
        Coefficient matrix of shape ``(F, T)`` in ``problem.solve_dtype``.

    Returns
    -------
    coefs : torch.Tensor
        Coefficient matrix of shape ``(F, T)`` cast to the original input
        dtype.
    intercept : torch.Tensor or None
        Intercept vector of shape ``(T,)`` cast to the original input dtype,
        or ``None`` when no intercept is fitted.
    """
    intercept = recover_intercept(problem, coefs)
    coefs_out = coefs.to(problem.original_dtype)
    if intercept is None:
        return coefs_out, None
    return coefs_out, intercept.to(problem.original_dtype)
