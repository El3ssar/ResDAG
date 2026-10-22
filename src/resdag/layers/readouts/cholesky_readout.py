"""
Cholesky Readout Layer
======================

This module provides :class:`CholeskyReadoutLayer`, a direct ridge readout that
factorises the regularised normal equations ``(XᵀX + αI)`` with
:func:`torch.linalg.cholesky` and back-substitutes with
:func:`torch.cholesky_solve`.

It is the readout the streaming / minibatch path needs: the same closed-form
ridge solve that :class:`IncrementalRidgeReadoutLayer` performs in
:meth:`~IncrementalRidgeReadoutLayer.finalize`, packaged as a standalone
single-shot readout. It reuses the shared centering + auto ``gram_dtype``
policy (the solve dtype, float64 by default, on every device) to preserve
accuracy for ill-conditioned reservoir states.

See Also
--------
resdag.layers.readouts.ReadoutLayer : Base readout layer.
resdag.layers.readouts.CGReadoutLayer : Iterative CG ridge readout.
resdag.layers.readouts.RidgeReadoutLayer : Cholesky / LU direct readout (#70).
resdag.layers.readouts.IncrementalRidgeReadoutLayer : Streaming partial_fit readout.
"""

import torch

from ._solve_utils import (
    apply_lmnt_regularizer,
    build_noisy_ridge_problem,
    build_ridge_problem,
    finalize_fit,
    solve_ridge,
)
from .base import ReadoutLayer


class CholeskyReadoutLayer(ReadoutLayer):
    """Readout layer that solves ridge regression via a Cholesky factorisation.

    The regularised normal equations ``(XᵀX + αI) W = Xᵀy`` are formed (with
    analytic centering when a bias is fitted) and solved with
    :func:`torch.linalg.cholesky` followed by :func:`torch.cholesky_solve`. The
    ``(F, F)`` system is symmetric positive definite for ``alpha > 0``, so the
    Cholesky factorisation is the fastest, most numerically stable exact solve
    for the readout widths typical of reservoir computing
    (``in_features = 100``–``2000``).

    Unlike :class:`CGReadoutLayer`, which only *approximates* the solution
    iteratively, this layer returns the exact ridge solution (up to floating
    point) in a single shot. It is the single-batch counterpart of the
    streaming :class:`IncrementalRidgeReadoutLayer`: both ultimately Cholesky-solve
    the same regularised Gram system, so a full-batch ``CholeskyReadoutLayer``
    fit and an accumulated ``IncrementalRidgeReadoutLayer`` fit over the same data
    agree to within floating-point tolerance.

    The solver minimises the same ridge objective as :class:`CGReadoutLayer`,

    .. math::

        \\lVert X W - Y \\rVert_2^2 + \\alpha \\lVert W \\rVert_2^2 .

    Noise-injection / LMNT regularization
    -------------------------------------
    Pass ``state_noise > 0`` to apply **state-noise regularization** — a
    standard, tunable tool for **closed-loop (autonomous-rollout) robustness** —
    by training on state-noise-perturbed reservoir states (Jaeger's state noise;
    Wikner et al. 2024, Chaos 34, 043105). It trades a little one-step accuracy
    for closed-loop stability; it is **not an automatic forecast-quality win**
    (whether it helps or hurts is problem- and :math:`\\sigma`-dependent, so
    :math:`\\sigma` must be tuned per problem — the default is off). The robust
    guarantee is the mechanism: the default ``noise_draws=None`` uses the
    closed-form **LMNT** term — the *exact analytic expectation* of the noisy
    Gram. Solving these expected normal equations is the infinite-copy limit of
    explicit-noise training; it does not average separately fitted coefficients.
    Set ``noise_draws=M`` (seeded by ``noise_seed``) to average the normal
    equations of ``M`` noisy copies, then solve once. See :class:`RidgeReadoutLayer` for the
    full discussion. ``0.0`` (default) is byte-identical to a plain fit.

    Parameters
    ----------
    in_features : int
        Size of input features (reservoir state dimension).
    out_features : int
        Size of output features (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term. When ``True`` the data is centered and
        an unpenalised intercept is recovered; when ``False`` the raw normal
        equations are solved with no intercept.
    name : str, optional
        Name for this readout layer. Used for identification in multi-readout
        architectures and as the key into ``targets`` by :class:`ESNTrainer`.
    trainable : bool, default=False
        If ``True``, weights are trainable via backpropagation.
    alpha : float, default=1e-6
        L2 regularization strength. Must be non-negative. ``alpha > 0``
        guarantees the Gram is positive definite; with ``alpha == 0`` the
        factorisation succeeds only on a full-rank Gram (reach for
        :class:`SVDReadoutLayer` if the Gram may be rank deficient).
    state_noise : float, default=0.0
        Standard deviation :math:`\\sigma` of the closed-loop state-noise
        regulariser (see the class docstring). ``0.0`` (default) is byte-identical
        to a plain fit. Must be non-negative. Adds ``alpha > 0`` positive
        definiteness insurance too — the LMNT term is a positive diagonal add.
    noise_draws : int, optional
        ``None`` (default) uses the closed-form LMNT term; an integer ``M >= 1``
        averages ``M`` noisy-state normal equations before one solve.
        Ignored when ``state_noise == 0``.
    noise_seed : int, optional
        Seed for the explicit-noise (``noise_draws``) draws. Ignored for the
        deterministic LMNT path.
    use_float64 : bool, default=True
        If ``True`` the small ``(F, F)`` factorisation runs in ``float64`` and
        the result is cast back to the parameter dtype.
    gram_dtype : torch.dtype, optional
        Dtype for forming the Gram matrix and right-hand side (the heavy
        ``(N, F)`` matmuls). Default ``None`` uses the solve dtype (``float64``
        with ``use_float64=True``) on every device. An explicit
        ``gram_dtype=torch.float32`` can speed up well-conditioned fits on
        CUDA, at the cost of precision when forming the normal equations.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.nn.Parameter or None
        Bias vector of shape ``(out_features,)``, or ``None`` if ``bias=False``.
    alpha : float
        L2 regularization strength.

    Examples
    --------
    >>> readout = CholeskyReadoutLayer(in_features=200, out_features=3, alpha=1e-6)
    >>> states = torch.randn(8, 50, 200)  # (batch, time, features)
    >>> targets = torch.randn(8, 50, 3)
    >>> readout.fit(states, targets)
    >>> output = readout(states)
    >>> output.shape
    torch.Size([8, 50, 3])

    See Also
    --------
    CGReadoutLayer : Iterative Conjugate Gradient ridge readout.
    RidgeReadoutLayer : Direct ridge readout with a ``solver`` switch.
    IncrementalRidgeReadoutLayer : Streaming partial_fit / finalize counterpart.
    resdag.training.ESNTrainer : Trainer that uses this for fitting.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        alpha: float = 1e-6,
        state_noise: float = 0.0,
        noise_draws: int | None = None,
        noise_seed: int | None = None,
        use_float64: bool = True,
        gram_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if state_noise < 0:
            raise ValueError(f"state_noise must be non-negative, got {state_noise}.")
        if noise_draws is not None and noise_draws < 1:
            raise ValueError(f"noise_draws must be >= 1 when given, got {noise_draws}.")
        self.alpha = alpha
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
        """Solve ridge regression via a Cholesky factorisation on flattened inputs.

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
            Intercept vector of shape ``(out_features,)``, or ``None`` when the
            layer was built with ``bias=False``.
        """
        if self.state_noise > 0 and self.noise_draws is not None:
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
            apply_lmnt_regularizer(problem, self.state_noise)

        layer_name = (
            f"{type(self).__name__}('{self._name}')"
            if self._name is not None
            else type(self).__name__
        )
        coefs = solve_ridge(
            problem.gram,
            problem.rhs,
            self.alpha,
            layer_name=layer_name,
            gram_dtype=problem.X.dtype,
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
            f"alpha={self.alpha}"
            f"{noise_str}"
            f")"
        )
