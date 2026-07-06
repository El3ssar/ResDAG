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
lmnt_gram_add
    The closed-form LMNT (Linearized Multi-Noise Training) noise-covariance
    term added to the Gram diagonal — the exact analytic expectation of fitting
    the readout on state-noise-perturbed reservoir states (Wikner et al. 2024).
apply_lmnt_regularizer
    Add the LMNT diagonal term to a prepared :class:`RidgeProblem`'s Gram in
    place, so the closed-form solve costs one solve, not many noisy copies.
solve_ridge
    Robustly solve the regularised normal equations ``(gram + αI) W = rhs``:
    Cholesky first, escalated scale-aware jitter on failure, then an
    ``eigh``-clamped fallback, plus a cheap post-solve accuracy guard.
recover_intercept
    Recover the unpenalised intercept from a coefficient matrix.

Noise-injection / LMNT regularization
-------------------------------------
Training the readout on **noise-perturbed** reservoir states is a standard,
tunable regulariser aimed at *closed-loop* (autonomous-rollout) robustness, a
different target than plain L2 ridge: the injected noise mimics the feedback
error the model sees when it free-runs on its own output, trading a little
one-step accuracy for closed-loop stability (Jaeger's classic state-noise idea;
Wikner et al. 2024, *Stabilizing machine learning prediction of dynamics*, Chaos
34, 043105).  Whether it nets out positive is problem- and :math:`\\sigma`-
dependent — it can help or hurt, and :math:`\\sigma` must be tuned per problem;
it is not a free forecast-quality win.  What *is* robustly guaranteed is the
mechanism below: LMNT computes the **exact expected noise-regularised readout**
in one solve.

Explicit noise training perturbs each state row :math:`r_i \\to r_i + \\xi_i`
with :math:`\\xi_i \\sim \\mathcal{N}(0, \\sigma^2 I_F)` (targets unchanged) and
refits.  Averaging over many noise draws, the *expected* noisy normal-equations
Gram is

.. math::

    \\mathbb{E}\\big[(R+\\Xi)^\\top (R+\\Xi)\\big]
        = R^\\top R + \\mathbb{E}[\\Xi^\\top\\Xi]
        = R^\\top R + N \\sigma^2 I_F ,

while the right-hand side :math:`\\mathbb{E}[(R+\\Xi)^\\top y] = R^\\top y` is
unchanged (the noise is zero-mean and independent of both :math:`R` and
:math:`y`).  **LMNT** replaces the Monte-Carlo average of :math:`M` noisy fits
with this analytic expectation: add the closed-form term to the Gram diagonal
and solve *once*.  With an intercept the states are centered first, which
removes one degree of freedom, so the centered pure-noise Gram expectation is
:math:`(N-1)\\sigma^2 I_F` (verified analytically and by Monte-Carlo); the raw
``bias=False`` term is :math:`N\\sigma^2 I_F`.  Either way the cost is a single
diagonal update, so LMNT :math:`\\approx` many-noise-copy training at the price
of one solve.

See Also
--------
resdag.layers.readouts.CGReadoutLayer : Conjugate-gradient ridge readout.
resdag.layers.readouts.RidgeReadoutLayer : Cholesky / solve direct readout.
resdag.layers.readouts.SVDReadoutLayer : SVD filter-factor readout.
resdag.layers.readouts.PinvReadoutLayer : lstsq / pseudo-inverse readout.
"""

import warnings
from dataclasses import dataclass

import torch

# Multiplier ``k`` for the escalated, scale-aware jitter used when the initial
# Cholesky factorisation fails. The retry uses
# ``alpha_eff = max(alpha, k * eps(gram_dtype) * trace(gram) / F)`` — enough to
# push the rounding-noise-negative eigenvalues of a rank-deficient centered Gram
# back above zero without materially perturbing a well-conditioned solve.
_JITTER_K = 16.0

# Relative normal-equations residual above which the post-solve accuracy guard
# warns. ``float32`` Gram formation in the ill-conditioned regime pushes this
# well past ``1e-3`` (up to O(1)); a healthy solve sits far below it.
_RESIDUAL_WARN_TOL = 1e-3


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

    **Accuracy first.** Reservoir readouts are ill-conditioned by construction
    (spectral radius near 1, raw inputs concatenated back in), and
    ``cond(XᵀX) = cond(X)²`` amplifies a ``float32`` Gram's ~1e-7 entry error
    past unity — a ``float64`` *solve* cannot recover precision destroyed at
    Gram *formation*. On GPU this silently produced wildly wrong (often
    non-finite, after autoregression) readouts for the premade concatenated-
    input models. So the default forms the Gram in ``solve_dtype`` (``float64``)
    on **every** device. The extra cost is a one-time ``O(N F²)`` matmul — a
    small fraction of a fit, dwarfed by the reservoir sequence pass — and
    correctness is the north star.

    Users with a large, provably well-conditioned fit who want the ~1/32–1/64
    GPU ``float64`` throughput back can pass ``gram_dtype=torch.float32``
    explicitly and opt into the speed/accuracy trade knowingly.

    Parameters
    ----------
    input_dtype : torch.dtype
        Dtype of the incoming state matrix.
    device : torch.device
        Device the state matrix lives on (unused now that the default is
        device-independent; kept for signature stability and explicit-override
        callers).
    solve_dtype : torch.dtype
        Dtype the small inversion runs in (see :func:`resolve_solve_dtype`).
    gram_dtype : torch.dtype or None
        Explicit override. When ``None`` (the default), the Gram is formed in
        ``solve_dtype`` (``float64`` unless ``use_float64=False``) regardless of
        device. Pass ``torch.float32`` to opt into faster, lower-precision GPU
        Gram formation.

    Returns
    -------
    torch.dtype
        The resolved Gram-formation dtype.
    """
    if gram_dtype is not None:
        return gram_dtype
    return solve_dtype


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
    n_samples : int
        Sample count ``N`` (rows of ``X`` after flattening).  Needed by the
        LMNT noise-covariance term, whose scale is :math:`N\\sigma^2`.
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
    n_samples: int
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
        n_samples=X.shape[0],
        solve_dtype=solve_dtype,
        original_dtype=original_dtype,
        fit_intercept=fit_intercept,
    )


def lmnt_gram_add(n_samples: int, state_noise: float, *, fit_intercept: bool) -> float:
    r"""Closed-form LMNT noise-covariance term added to the Gram diagonal.

    Training a ridge readout on state-noise-perturbed reservoir states —
    :math:`r_i \to r_i + \xi_i`, :math:`\xi_i \sim \mathcal{N}(0, \sigma^2 I)`,
    targets fixed — is a standard, tunable regulariser targeting **closed-loop
    robustness**: the injected noise mimics the feedback error the model sees
    when it free-runs on its own output (whether it improves a given rollout is
    problem- and :math:`\sigma`-dependent — tune :math:`\sigma` per problem).
    Averaged over many noise draws the noisy Gram's expectation is
    :math:`R^\top R` plus a scalar on the diagonal, and this function returns that
    scalar so **a single solve reproduces the many-noise-copy fit exactly** — its
    analytic expectation, not an approximation (**LMNT** — Linearized Multi-Noise
    Training; Wikner et al. 2024, *Stabilizing machine learning prediction of
    dynamics*, Chaos 34, 043105).

    The coefficient depends on whether an intercept is fitted (which centers the
    states, removing one degree of freedom):

    - ``fit_intercept=False`` (raw Gram): :math:`N \sigma^2` — the expectation
      of :math:`\Xi^\top\Xi`.
    - ``fit_intercept=True`` (centered Gram): :math:`(N-1) \sigma^2` — the
      expectation of the centered pure-noise Gram
      :math:`\Xi^\top\Xi - N\,\bar\xi^\top\bar\xi`.

    Both are the *exact* analytic expectation of the corresponding explicit-noise
    procedure (verified against Monte-Carlo), so LMNT is not an approximation of
    noise training but its closed form.

    Parameters
    ----------
    n_samples : int
        Number of state rows ``N`` the Gram was formed from.
    state_noise : float
        Per-feature noise standard deviation :math:`\sigma \ge 0`.  ``0`` returns
        ``0.0`` (no regularisation — byte-identical to a plain ridge fit).
    fit_intercept : bool
        Whether the Gram is centered (an intercept is being recovered).

    Returns
    -------
    float
        The scalar to add to every Gram diagonal entry
        (:math:`N\sigma^2` raw, :math:`(N-1)\sigma^2` centered).

    Raises
    ------
    ValueError
        If ``state_noise`` is negative.
    """
    if state_noise < 0:
        raise ValueError(f"state_noise must be non-negative, got {state_noise}.")
    if state_noise == 0.0:
        return 0.0
    # Centering an N-row block removes one dof from the pure-noise Gram, so its
    # expected diagonal is (N-1)sigma^2 rather than N sigma^2.
    effective_n = (n_samples - 1) if fit_intercept else n_samples
    # Guard the degenerate N<=1 case: a single (centered) sample carries no
    # noise-covariance signal, so the term is zero rather than negative.
    effective_n = max(effective_n, 0)
    return float(effective_n) * float(state_noise) * float(state_noise)


def apply_lmnt_regularizer(problem: RidgeProblem, state_noise: float) -> None:
    r"""Add the closed-form LMNT term to a prepared problem's Gram, in place.

    Mutates ``problem.gram`` by adding :func:`lmnt_gram_add` to its diagonal —
    the analytic expectation of the noise-perturbed Gram — so the subsequent
    :func:`solve_ridge` call yields the LMNT (many-noise-copy-equivalent)
    readout at the cost of one solve.  A no-op when ``state_noise == 0`` (the
    default everywhere), which keeps every noiseless fit byte-identical.

    The right-hand side is intentionally **not** touched: the expected noisy RHS
    equals the noiseless :math:`R^\top y` (the noise is zero-mean and independent
    of the targets), so only the Gram gains the noise covariance.

    Parameters
    ----------
    problem : RidgeProblem
        A problem prepared by :func:`build_ridge_problem`.  Its ``gram`` is
        updated in place.
    state_noise : float
        Per-feature noise standard deviation :math:`\sigma \ge 0`.

    Raises
    ------
    ValueError
        If ``state_noise`` is negative.
    """
    add = lmnt_gram_add(problem.n_samples, state_noise, fit_intercept=problem.fit_intercept)
    if add == 0.0:
        return
    eye = torch.eye(problem.n_features, dtype=problem.gram.dtype, device=problem.gram.device)
    problem.gram.add_(add * eye)


def build_noisy_ridge_problem(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    fit_intercept: bool,
    use_float64: bool,
    gram_dtype: torch.dtype | None,
    state_noise: float,
    n_draws: int,
    generator: torch.Generator | None = None,
) -> RidgeProblem:
    r"""Prepare a ridge problem from **explicitly** noise-perturbed states.

    The Monte-Carlo counterpart of the closed-form
    :func:`apply_lmnt_regularizer`: instead of adding the analytic
    :math:`N\sigma^2` term to the Gram, it draws ``n_draws`` i.i.d. Gaussian
    perturbations :math:`R \to R + \Xi`, :math:`\Xi \sim \mathcal{N}(0,
    \sigma^2 I)`, forms the (centered, when fitting an intercept) Gram and
    right-hand side for each draw, and averages them.  This is the
    definitely-correct baseline — Jaeger's state-noise training — and LMNT
    converges to it as ``n_draws`` grows (the two agree to Monte-Carlo error).

    Both the Gram *and* the right-hand side are averaged over draws for an
    honest Monte-Carlo estimate, even though the RHS expectation is the
    noiseless :math:`R^\top y`; averaging it keeps the finite-``n_draws`` result
    faithful to literally refitting on noisy states rather than silently folding
    in the analytic simplification.

    Parameters
    ----------
    X : torch.Tensor
        Flattened state matrix of shape ``(N, F)``.
    y : torch.Tensor
        Flattened target matrix of shape ``(N, T)``.
    fit_intercept : bool
        Whether to center each noisy draw and recover an unpenalised intercept.
    use_float64 : bool
        Whether the small ``(F, F)`` inversion should run in ``float64``.
    gram_dtype : torch.dtype or None
        Explicit override for the Gram-formation dtype, or ``None`` for the
        device-aware default (see :func:`resolve_gram_dtype`).
    state_noise : float
        Per-feature noise standard deviation :math:`\sigma > 0`.
    n_draws : int
        Number of Monte-Carlo noise draws to average (``>= 1``).
    generator : torch.Generator, optional
        RNG for the noise draws.  ``None`` uses the process-global RNG.

    Returns
    -------
    RidgeProblem
        The prepared problem whose ``gram`` / ``rhs`` are the noise-averaged
        normal equations.  The stored means (used to recover the intercept) are
        the *noiseless* column means, whose expectation matches the averaged
        noisy means (the added noise is zero-mean).

    Raises
    ------
    ValueError
        If ``state_noise`` is negative or ``n_draws < 1``.
    """
    if state_noise < 0:
        raise ValueError(f"state_noise must be non-negative, got {state_noise}.")
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}.")

    original_dtype = X.dtype
    solve_dtype = resolve_solve_dtype(original_dtype, use_float64)
    resolved_gram_dtype = resolve_gram_dtype(original_dtype, X.device, solve_dtype, gram_dtype)

    Xg = X.to(resolved_gram_dtype)
    yg = y.to(resolved_gram_dtype)

    n = float(Xg.shape[0])
    gram_acc = torch.zeros(Xg.shape[1], Xg.shape[1], dtype=Xg.dtype, device=Xg.device)
    rhs_acc = torch.zeros(Xg.shape[1], yg.shape[1], dtype=Xg.dtype, device=Xg.device)
    for _ in range(n_draws):
        noise = torch.randn(Xg.shape, dtype=Xg.dtype, device=Xg.device, generator=generator)
        Xn = Xg + state_noise * noise
        if fit_intercept:
            xbar = Xn.mean(dim=0, keepdim=True)
            ybar = yg.mean(dim=0, keepdim=True)
            gram_acc += Xn.T @ Xn - n * (xbar.T @ xbar)
            rhs_acc += Xn.T @ yg - n * (xbar.T @ ybar)
        else:
            gram_acc += Xn.T @ Xn
            rhs_acc += Xn.T @ yg
    gram_acc /= float(n_draws)
    rhs_acc /= float(n_draws)

    # Intercept recovery uses the noiseless means: E[noisy mean] = noiseless
    # mean (zero-mean noise), and using the exact noiseless means avoids adding
    # a second, unnecessary source of Monte-Carlo variance to the intercept.
    x_mean: torch.Tensor | None = None
    y_mean: torch.Tensor | None = None
    if fit_intercept:
        x_mean = Xg.mean(dim=0, keepdim=True)
        y_mean = yg.mean(dim=0, keepdim=True)

    return RidgeProblem(
        X=Xg,
        y=yg,
        gram=gram_acc.to(solve_dtype),
        rhs=rhs_acc.to(solve_dtype),
        x_mean=x_mean,
        y_mean=y_mean,
        n_features=Xg.shape[1],
        n_samples=Xg.shape[0],
        solve_dtype=solve_dtype,
        original_dtype=original_dtype,
        fit_intercept=fit_intercept,
    )


def _normal_equations_residual(
    gram: torch.Tensor,
    coefs: torch.Tensor,
    rhs: torch.Tensor,
    alpha: float,
) -> float:
    """Relative residual ``||(gram + αI) W - rhs|| / ||rhs||`` in ``float64``.

    A cheap post-solve accuracy check: it is one ``(F, F) @ (F, T)`` matmul,
    negligible next to the factorisation itself, and computed in ``float64`` so
    it exposes precision lost when the Gram was formed in ``float32``.

    Parameters
    ----------
    gram : torch.Tensor
        The (already regularisation-free) Gram matrix of shape ``(F, F)``.
    coefs : torch.Tensor
        The solved coefficient matrix of shape ``(F, T)``.
    rhs : torch.Tensor
        The right-hand side of shape ``(F, T)``.
    alpha : float
        The (effective) regularisation strength added to the diagonal.

    Returns
    -------
    float
        The relative residual, or ``0.0`` when ``rhs`` is all zeros.
    """
    g = gram.to(torch.float64)
    w = coefs.to(torch.float64)
    r = rhs.to(torch.float64)
    resid = g @ w + alpha * w - r
    rhs_norm = torch.linalg.norm(r)
    if rhs_norm == 0:
        return 0.0
    return float((torch.linalg.norm(resid) / rhs_norm).item())


def solve_ridge(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    alpha: float,
    *,
    layer_name: str,
    gram_dtype: torch.dtype | None = None,
    check_accuracy: bool = True,
) -> torch.Tensor:
    """Robustly solve the regularised normal equations ``(gram + αI) W = rhs``.

    This is the single shared solve path for every Gram-based readout
    (:class:`CholeskyReadoutLayer`, :class:`RidgeReadoutLayer`,
    :class:`IncrementalRidgeReadoutLayer`). It resolves the GPU-default ``float32``
    Gram footgun: on a rank-deficient centered Gram, ``float32`` rounding noise
    nudges zero eigenvalues slightly negative, past the default ``alpha=1e-6``
    jitter, so a bare :func:`torch.linalg.cholesky` raises a raw
    ``LinAlgError`` on CUDA while the identical ``float64`` CPU solve succeeds.

    The strategy is:

    1. Try a Cholesky factorisation of ``gram + αI`` (fast, exact for an SPD
       system) and back-substitute.
    2. On :class:`torch.linalg.LinAlgError`, retry once with escalated,
       scale-aware jitter ``α_eff = max(α, k · eps(dtype) · tr(gram) / F)`` —
       just enough to lift the rounding-noise-negative eigenvalues back above
       zero. A single warning naming the layer is emitted.
    3. If that still fails, fall back to a symmetric-eigendecomposition solve
       with eigenvalues clamped to ``≥ α_eff`` (always succeeds; robust to
       exact rank deficiency).

    **Accuracy guard.** After the solve, two cheap checks run (both reusing
    quantities the factorisation already produced, so no extra ``O(F³)`` work):

    - a **precision-loss estimate** ``κ̂(A) · eps(gram_dtype)`` from the
      factor's diagonal (Cholesky) or clamped eigenvalues (eigh). When the
      Gram was formed in ``float32`` this crosses ``~1e-3`` around
      ``cond(X) ≈ 3e3`` — exactly the cliff where a ``float32`` Gram starts
      costing real accuracy (``cond(XᵀX) = cond(X)²`` pushes the ``float32``
      entry error past unity) — and a ``float64`` solve cannot recover
      precision destroyed at Gram formation.
    - a **normal-equations residual** ``‖(gram+αI)W − rhs‖ / ‖rhs‖`` in
      ``float64`` (one ``(F, F) @ (F, T)`` matmul), which catches a genuinely
      under-satisfied solve (a ``float32`` *solve* dtype, or a heavily clamped
      eigh fallback).

    Either check tripping emits one warning naming the layer and suggesting
    ``gram_dtype=torch.float64`` or :class:`SVDReadoutLayer`. The Gram is now
    formed in ``float64`` by default on every device (see
    :func:`resolve_gram_dtype`), so this guard mostly fires when a user has
    explicitly opted into ``gram_dtype=torch.float32`` for speed.

    Parameters
    ----------
    gram : torch.Tensor
        The (optionally centered) Gram matrix ``XᵀX`` of shape ``(F, F)``, in
        the solve dtype. Regularisation is added here, so pass it *without*
        ``αI`` already on the diagonal.
    rhs : torch.Tensor
        The (optionally centered) right-hand side ``Xᵀy`` of shape ``(F, T)``,
        in the solve dtype.
    alpha : float
        The L2 regularisation strength (non-negative).
    layer_name : str
        Human-readable identifier for the calling layer, used in warnings.
    gram_dtype : torch.dtype, optional
        The dtype the Gram was *formed* in (``float32`` on CUDA by default).
        Used only by the precision-loss estimate to weight the condition
        number by the right machine epsilon; defaults to ``gram.dtype`` when
        not supplied.
    check_accuracy : bool, default=True
        Whether to run the cheap post-solve accuracy guard. Left on by default;
        both checks reuse the factorisation and cost a single small matmul.

    Returns
    -------
    torch.Tensor
        The coefficient matrix ``W`` of shape ``(F, T)`` in the solve dtype.
    """
    n_features = gram.shape[0]
    eye = torch.eye(n_features, dtype=gram.dtype, device=gram.device)
    a = gram + alpha * eye

    alpha_eff = alpha
    # ``cond_estimate`` is a cheap lower bound on cond(A) extracted from whatever
    # factorisation succeeded — the Cholesky factor's diagonal or the eigh
    # spectrum — so the precision guard needs no extra decomposition.
    cond_estimate: float | None = None
    escalated = False
    try:
        factor = torch.linalg.cholesky(a)
        coefs = torch.cholesky_solve(rhs, factor)
        cond_estimate = _cond_from_cholesky(factor)
    except torch.linalg.LinAlgError:
        escalated = True
        used_eigh = False
        # ``gram + alpha*I`` is not positive-definite: the Gram has an
        # eigenvalue below ``-alpha``. For a real ``XᵀX`` that can only be
        # rounding noise (a genuine Gram is PSD), so lift the diagonal by a
        # scale-aware floor keyed to the Gram's *magnitude* — ``max|diag|`` is a
        # cheap proxy for ‖gram‖ and dominates the ~``eps·‖gram‖`` noise floor
        # that produced the negative eigenvalue. (A ``float64`` Gram is PSD to
        # ~``eps64·‖gram‖`` and reaches this branch only for a genuinely
        # indefinite input.)
        eps = torch.finfo(gram.dtype).eps
        max_diag = float(gram.diagonal().abs().max().item())
        jitter = _JITTER_K * eps * max_diag
        alpha_eff = max(alpha, jitter)
        a_jittered = gram + alpha_eff * eye
        try:
            factor = torch.linalg.cholesky(a_jittered)
            coefs = torch.cholesky_solve(rhs, factor)
            cond_estimate = _cond_from_cholesky(factor)
        except torch.linalg.LinAlgError:
            # Still indefinite: solve on the PSD projection via an
            # eigendecomposition. Negative eigenvalues are rounding-noise null
            # directions, so clamp them to zero (NOT sign-invert them) before
            # adding the ridge floor — the coefficient of a null direction
            # becomes ``(Vᵀrhs) / alpha_eff`` (bounded), exactly ridge on the
            # PSD part, never ``1/λ`` with a wrong sign for a spurious negative
            # eigenvalue. Guaranteed finite.
            used_eigh = True
            evals, evecs = torch.linalg.eigh(gram)
            clamped = torch.clamp(evals, min=0.0) + alpha_eff
            coefs = evecs @ ((evecs.T @ rhs) / clamped.unsqueeze(1))
            cond_estimate = float((clamped.max() / clamped.min()).item())
        how = "an eigendecomposition (PSD-projected)" if used_eigh else "a jittered Cholesky"
        escalated_note = (
            f"escalated the ridge jitter from {alpha:g} to {alpha_eff:g}"
            if alpha_eff > alpha
            else f"kept alpha={alpha:g}"
        )
        warnings.warn(
            f"{layer_name}: the Gram matrix was not positive-definite for a "
            f"plain Cholesky at alpha={alpha:g} (an eigenvalue fell below "
            f"-alpha); recovered a finite solution via {how} and {escalated_note}. "
            f"This is almost always a float32-Gram rounding artifact on GPU — the "
            f"default now forms the Gram in float64, so if you see this you likely "
            f"passed gram_dtype=torch.float32. Drop that override (or use "
            f"SVDReadoutLayer) for a genuinely rank-deficient feature map.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Skip the precision guard when the jitter path already fired: that warning
    # is the actionable one, and a second (redundant) accuracy warning is noise.
    if check_accuracy and not escalated:
        _warn_on_precision_loss(
            gram=gram,
            coefs=coefs,
            rhs=rhs,
            alpha_eff=alpha_eff,
            cond_estimate=cond_estimate,
            gram_dtype=gram_dtype if gram_dtype is not None else gram.dtype,
            layer_name=layer_name,
        )

    return coefs


def _cond_from_cholesky(factor: torch.Tensor) -> float:
    """Cheap lower-bound estimate of ``cond(A)`` from its Cholesky factor.

    For ``A = L Lᵀ`` the ratio of the largest to smallest diagonal entry of
    ``L`` squared lower-bounds ``cond(A)``. It underestimates the true
    condition number (typically by ~1–2 orders of magnitude) but tracks it
    closely enough on a log scale to flag the ``float32``-Gram precision cliff,
    and it is free — the factor is already computed.

    Parameters
    ----------
    factor : torch.Tensor
        The lower-triangular Cholesky factor ``L`` of shape ``(F, F)``.

    Returns
    -------
    float
        The estimated condition number ``(max|Lᵢᵢ| / min|Lᵢᵢ|)²``.
    """
    diag = factor.diagonal().abs()
    dmin = diag.min()
    if dmin == 0:
        return float("inf")
    return float(((diag.max() / dmin) ** 2).item())


def _warn_on_precision_loss(
    *,
    gram: torch.Tensor,
    coefs: torch.Tensor,
    rhs: torch.Tensor,
    alpha_eff: float,
    cond_estimate: float | None,
    gram_dtype: torch.dtype,
    layer_name: str,
) -> None:
    """Emit (at most) one warning when the solve likely lost real accuracy.

    Combines two cheap signals: the estimated precision loss
    ``κ̂(A) · eps(gram_dtype)`` (predicts error baked in at Gram formation, the
    dominant ``float32``-on-GPU footgun) and the self-consistency residual
    ``‖(gram+αI)W − rhs‖ / ‖rhs‖`` (catches an under-satisfied solve). See
    :func:`solve_ridge` for the parameter meanings.
    """
    est_error = None
    if cond_estimate is not None:
        est_error = cond_estimate * torch.finfo(gram_dtype).eps

    residual = _normal_equations_residual(gram, coefs, rhs, alpha_eff)

    if (est_error is not None and est_error > _RESIDUAL_WARN_TOL) or (
        residual > _RESIDUAL_WARN_TOL
    ):
        detail = f"residual={residual:.2e}"
        if est_error is not None:
            detail = f"est. precision loss={est_error:.2e}, {detail}"
        warnings.warn(
            f"{layer_name}: the ridge solve is losing accuracy ({detail} > "
            f"{_RESIDUAL_WARN_TOL:g}). The Gram matrix is ill-conditioned and "
            f"forming it in {gram_dtype} (the GPU default is float32) destroys "
            f"precision the solve cannot recover — the fitted readout will "
            f"differ from the float64 result. Pass gram_dtype=torch.float64 "
            f"for an accurate readout, or use SVDReadoutLayer.",
            RuntimeWarning,
            stacklevel=3,
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
