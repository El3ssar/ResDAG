"""
Direct Ridge Readout Layer
==========================

This module provides `RidgeReadoutLayer`, which solves the ridge
normal equations directly with a Cholesky factorisation (`'cholesky'`) or a
general LU solve (`'solve'`) instead of the iterative Conjugate Gradient of
`CGReadoutLayer`.

See Also
--------
resdag.layers.readouts.ReadoutLayer : Base readout layer.
resdag.layers.readouts.CGReadoutLayer : Iterative CG ridge readout.
resdag.layers.readouts.SVDReadoutLayer : SVD filter-factor readout.
resdag.layers.readouts.PinvReadoutLayer : lstsq / pseudo-inverse readout.
"""

import torch

from ._solve_utils import (
    _warn_on_precision_loss,
    apply_lmnt_regularizer,
    build_noisy_ridge_problem,
    build_ridge_problem,
    finalize_fit,
    solve_ridge,
)
from .base import ReadoutLayer


class RidgeReadoutLayer(ReadoutLayer):
    """Readout layer with a direct ridge-regression solver.

    For the readout widths typical of reservoir computing
    (`in_features = 100`–`2000`) the `(F, F)` Gram matrix factorises
    exactly and almost instantly. A direct Cholesky solve is both faster and
    *exact* (up to floating point), whereas `CGReadoutLayer` only
    approximates the solution and can under-converge on the ill-conditioned
    concatenated-input readouts produced by `ott_esn` / `power_augmented`.

    The solver minimises the same ridge objective as `CGReadoutLayer`,

    $$
    \\lVert X W - Y \\rVert_2^2 + \\alpha \\lVert W \\rVert_2^2 ,
    $$

    by forming the regularised normal equations `(XᵀX + αI) W = Xᵀy` and
    solving them with one of:

    - `solver='cholesky'` (default) — `torch.linalg.cholesky` followed by
      `torch.linalg.cholesky_solve`. Exploits the symmetric positive
      definiteness of `XᵀX + αI` in exact arithmetic for `alpha > 0`.
      Finite-precision Gram formation can still lose positive definiteness.
    - `solver='solve'` — `torch.linalg.solve` (LU). It
      does not require positive definiteness, so it supports `alpha == 0`
      on a full-rank Gram.

    Solver-selection guide
    ----------------------
    - **Default to** `RidgeReadoutLayer(solver='cholesky')` for any
      well-conditioned readout with `alpha > 0` when an explicit
      normal-equations solve is desired. Runtime and accuracy depend on the
      workload, precision and hardware. `AutoReadoutLayer` checks
      numerical safety and can use the original design matrix instead.
    - Use `solver='solve'` when you need `alpha == 0` (pure least squares)
      and the Gram is full rank but you don't want the SVD machinery.
    - If the Gram is **rank deficient** (high-degree NG-RC feature maps, more
      features than samples), a Cholesky/LU solve will fail or be unstable —
      reach for `SVDReadoutLayer` (filter factors) or
      `PinvReadoutLayer` (lstsq / pinv) instead.

    Noise-injection / LMNT regularization
    -------------------------------------
    Pass `state_noise > 0` to apply **state-noise regularization**, a standard,
    tunable tool for **closed-loop (autonomous-rollout) robustness** rather than
    only weight magnitude.  Fitting on state-noise-perturbed reservoir states
    mimics the feedback error the model sees when it free-runs on its own output
    — the injected noise trades a little one-step accuracy for closed-loop
    stability (Jaeger's state-noise idea; Wikner et al. 2024, *Stabilizing
    machine learning prediction of dynamics*, Chaos 34, 043105).  **It is not a
    free or automatic forecast-quality win:** whether the trade nets out positive
    is problem- and $\\sigma$-dependent — it can help *or* hurt, and
    $\\sigma$ must be tuned per problem.  The shipped default is `0.0`
    (off); opt in and tune.

    The robust guarantee is the **mechanism**, not a VPT number: by default
    (`noise_draws=None`) the closed-form **LMNT** (Linearized Multi-Noise
    Training) path computes the *exact analytic expectation* of the
    noise-perturbed Gram — a single scalar on the diagonal ($N\\sigma^2$
    raw, $(N-1)\\sigma^2$ centered). One solve of the expected normal
    equations is the infinite-draw limit of the explicit-noise path. This is
    not the expectation of separately fitted coefficients or chaotic forecasts.
    Set `noise_draws=M` to average the normal equations of `M` explicit
    Gaussian-perturbed copies before solving once (seeded by `noise_seed`).
    This Monte-Carlo path converges to the analytic path as `M` grows. `state_noise` is
    **independent** of `alpha`: they add on the diagonal
    ($\\alpha + N\\sigma^2$) but noise training targets rollout robustness
    while `alpha` targets weight norm.

    Parameters
    ----------
    in_features : int
        Size of input features (reservoir state dimension).
    out_features : int
        Size of output features (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term. When `True` the data is centered and
        an unpenalised intercept is recovered; when `False` the raw normal
        equations are solved with no intercept.
    name : str, optional
        Name for this readout layer. Used for identification in multi-readout
        architectures and as the key into `targets` by `ESNTrainer`.
    trainable : bool, default=False
        If `True`, weights are trainable via backpropagation.
    alpha : float, default=1e-6
        L2 regularization strength. Must be non-negative. `alpha == 0` is
        only safe with `solver='solve'` on a full-rank Gram.
    solver : {'cholesky', 'solve'}, default='cholesky'
        Direct solver to use for the normal equations.
    state_noise : float, default=0.0
        Standard deviation $\\sigma$ of the Gaussian state noise the readout
        is regularised against for closed-loop stability (see the class
        docstring). `0.0` (default) disables it — the fit is then byte-identical
        to a plain ridge solve. Must be non-negative.
    noise_draws : int, optional
        When `state_noise > 0`: `None` (default) uses the closed-form LMNT
        term (one solve, exact expectation); an integer `M >= 1` instead
        averages `M` noisy-state normal equations, then solves once (Monte-Carlo). Ignored when
        `state_noise == 0`.
    noise_seed : int, optional
        Seed for the explicit-noise (`noise_draws`) Gaussian draws, for
        reproducibility. `None` uses the process-global RNG. Ignored for the
        closed-form LMNT path (which is deterministic).
    use_float64 : bool, default=True
        If `True` the small `(F, F)` factorisation runs in `float64` and
        the result is cast back to the parameter dtype.
    gram_dtype : torch.dtype, optional
        Dtype for forming the Gram matrix and right-hand side (the heavy
        `(N, F)` matmuls). Default `None` uses the solve dtype (`float64`
        with `use_float64=True`) on every device. An explicit
        `gram_dtype=torch.float32` can speed up well-conditioned fits on
        CUDA, at the cost of precision when forming the normal equations.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape `(out_features, in_features)`.
    bias : torch.nn.Parameter or None
        Bias vector of shape `(out_features,)`, or `None` if `bias=False`.
    alpha : float
        L2 regularization strength.
    solver : str
        The selected direct solver.
    state_noise : float
        Standard deviation of the closed-loop state-noise regulariser.

    Examples
    --------
    >>> readout = RidgeReadoutLayer(in_features=200, out_features=3, alpha=1e-6)
    >>> states = torch.randn(8, 50, 200)  # (batch, time, features)
    >>> targets = torch.randn(8, 50, 3)
    >>> readout.fit(states, targets)
    >>> output = readout(states)
    >>> output.shape
    torch.Size([8, 50, 3])

    Closed-form LMNT noise regularisation for closed-loop robustness (the
    `state_noise` value below is *illustrative* — tune it per problem, and
    expect it to help or hurt depending on the system and $\\sigma$):

    >>> readout = RidgeReadoutLayer(200, 3, alpha=1e-8, state_noise=1e-3)

    See Also
    --------
    CGReadoutLayer : Iterative Conjugate Gradient ridge readout.
    SVDReadoutLayer : SVD filter-factor readout for rank-deficient Gram.
    PinvReadoutLayer : lstsq / pseudo-inverse readout.
    resdag.training.ESNTrainer : Trainer that uses this for fitting.
    """

    _VALID_SOLVERS = ("cholesky", "solve")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        alpha: float = 1e-6,
        solver: str = "cholesky",
        state_noise: float = 0.0,
        noise_draws: int | None = None,
        noise_seed: int | None = None,
        use_float64: bool = True,
        gram_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if solver not in self._VALID_SOLVERS:
            raise ValueError(f"solver must be one of {self._VALID_SOLVERS}, got {solver!r}.")
        if state_noise < 0:
            raise ValueError(f"state_noise must be non-negative, got {state_noise}.")
        if noise_draws is not None and noise_draws < 1:
            raise ValueError(f"noise_draws must be >= 1 when given, got {noise_draws}.")
        self.alpha = alpha
        self.solver = solver
        self.state_noise = state_noise
        self.noise_draws = noise_draws
        self.noise_seed = noise_seed
        self.use_float64 = use_float64
        self.gram_dtype = gram_dtype

    def solve(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve ridge regression directly on flattened inputs.

        Parameters
        ----------
        states : torch.Tensor
            Flattened state matrix of shape `(N, in_features)`.
        targets : torch.Tensor
            Flattened target matrix of shape `(N, out_features)`.

        Returns
        -------
        coefs : torch.Tensor
            Coefficient matrix of shape `(in_features, out_features)`.
        intercept : torch.Tensor or None
            Intercept vector of shape `(out_features,)`, or `None` when the
            layer was built with `bias=False`.
        """
        if self.state_noise > 0 and self.noise_draws is not None:
            # Explicit Monte-Carlo noise training: average M noisy normal equations.
            generator: torch.Generator | None = None
            if self.noise_seed is not None:
                generator = torch.Generator(device=states.device)
                generator.manual_seed(int(self.noise_seed))
            problem = build_noisy_ridge_problem(
                states,
                targets,
                fit_intercept=self.bias is not None,
                use_float64=self.use_float64,
                gram_dtype=self.gram_dtype,
                state_noise=self.state_noise,
                n_draws=self.noise_draws,
                generator=generator,
            )
        else:
            problem = build_ridge_problem(
                states,
                targets,
                fit_intercept=self.bias is not None,
                use_float64=self.use_float64,
                gram_dtype=self.gram_dtype,
            )
            # Closed-form LMNT: fold the analytic noise-covariance term into the
            # Gram diagonal (no-op when state_noise == 0, keeping the fit exact).
            apply_lmnt_regularizer(problem, self.state_noise)

        layer_name = (
            f"{type(self).__name__}('{self._name}')"
            if self._name is not None
            else type(self).__name__
        )

        if self.solver == "cholesky":
            coefs = solve_ridge(
                problem.gram,
                problem.rhs,
                self.alpha,
                layer_name=layer_name,
                gram_dtype=problem.X.dtype,
            )
        else:  # "solve" — general LU. Does not need positive-definiteness, so
            # it tolerates alpha == 0 on a full-rank Gram; it still benefits
            # from the shared post-solve accuracy residual guard.
            eye = torch.eye(
                problem.n_features,
                dtype=problem.solve_dtype,
                device=problem.gram.device,
            )
            coefs = torch.linalg.solve(problem.gram + self.alpha * eye, problem.rhs)
            _warn_on_precision_loss(
                gram=problem.gram,
                coefs=coefs,
                rhs=problem.rhs,
                alpha_eff=self.alpha,
                cond_estimate=None,
                gram_dtype=problem.X.dtype,
                layer_name=layer_name,
            )

        return finalize_fit(problem, coefs)

    def __repr__(self) -> str:
        """Return string representation."""
        name_str = f", name='{self._name}'" if self._name is not None else ""
        noise_str = ""
        if self.state_noise > 0:
            mode = "lmnt" if self.noise_draws is None else f"draws={self.noise_draws}"
            noise_str = f", state_noise={self.state_noise} ({mode})"
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f"{name_str}, "
            f"alpha={self.alpha}, "
            f"solver='{self.solver}'"
            f"{noise_str}"
            f")"
        )
