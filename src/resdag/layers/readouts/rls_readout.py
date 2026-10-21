"""
Recursive Least Squares Readout Layer
=====================================

This module provides :class:`RLSReadout`, a *true-online* readout that updates
its weights one sample at a time with the **Recursive Least Squares** (RLS)
algorithm and an exponential forgetting factor :math:`\\lambda`. Unlike the
batch ridge readouts (:class:`RidgeReadoutLayer`, :class:`CholeskyReadoutLayer`),
which materialise the whole state matrix and solve the normal equations once,
and unlike :class:`IncrementalRidgeReadoutLayer`, which accumulates the ridge
sufficient statistics chunk-by-chunk and solves once at :meth:`finalize`, RLS
carries an **inverse-correlation matrix** :math:`P` and refines the weights
recursively at every step. No linear system is ever assembled or factorised.

RLS is the online sibling of ridge regression: with :math:`\\lambda = 1`
(growing window, no forgetting) it converges to the ordinary least-squares /
ridge(:math:`\\alpha \\to 0`) solution; with :math:`\\lambda < 1` it exponentially
down-weights old samples and tracks a **non-stationary** target that batch ridge
cannot follow.

Update equations
----------------
For feature vector :math:`x_t \\in \\mathbb{R}^F` and target
:math:`d_t \\in \\mathbb{R}^O`, with weight matrix :math:`W \\in \\mathbb{R}^{F
\\times O}` (the transpose of the :class:`torch.nn.Linear` ``weight``) and inverse
correlation :math:`P \\in \\mathbb{R}^{F \\times F}`:

.. math::

    g_t &= P_{t-1}\\, x_t \\\\
    k_t &= \\frac{g_t}{\\lambda + x_t^\\top g_t} \\\\
    e_t &= d_t - W_{t-1}^\\top x_t \\quad\\text{(a-priori error)} \\\\
    W_t &= W_{t-1} + k_t\\, e_t^\\top \\\\
    P_t &= \\frac{1}{\\lambda}\\left(P_{t-1} - k_t\\, g_t^\\top\\right)

:math:`P` is initialised to :math:`I / \\delta` (equivalently :math:`\\delta =
\\alpha`, a soft ridge prior — a large :math:`\\delta` / small init :math:`P`
regularises the early, data-starved updates). Because the rank-1 :math:`P`
update is symmetric in exact arithmetic, :math:`P` only drifts off symmetry by
round-off; it is therefore re-symmetrised **periodically** (and once at the end
of every pass, so it is exactly symmetric afterwards) rather than every step,
keeping the per-step cost at the intended :math:`O(F^2)`.

See Also
--------
resdag.layers.readouts.IncrementalRidgeReadoutLayer : Streaming ridge via
    accumulated sufficient statistics + a single ``finalize`` solve.
resdag.layers.readouts.RidgeReadoutLayer : Batch closed-form ridge readout.
resdag.layers.readouts.ReadoutLayer : Base readout layer.

References
----------
.. [1] S. Haykin, "Adaptive Filter Theory," 5th ed., Pearson, 2014, ch. 10
   (Recursive Least-Squares Adaptive Filtering).
"""

import torch

from ._solve_utils import resolve_solve_dtype
from .base import ReadoutLayer


class RLSReadout(ReadoutLayer):
    r"""True-online Recursive Least Squares (RLS) readout with forgetting.

    Refines the readout weights one sample at a time from a running
    inverse-correlation matrix :math:`P`, instead of solving a batch linear
    system. This makes it the natural readout for **streaming** and
    **non-stationary** reservoir computing: samples can arrive indefinitely, and
    a forgetting factor :math:`\lambda < 1` lets the readout track a target whose
    statistics drift over time.

    Update equations
    ----------------
    For each feature vector :math:`x_t \in \mathbb{R}^F` with target
    :math:`d_t \in \mathbb{R}^O`, writing :math:`W \in \mathbb{R}^{F \times O}`
    for the transpose of the ``nn.Linear`` ``weight`` and :math:`P \in
    \mathbb{R}^{F \times F}` for the inverse correlation:

    .. math::

        g_t &= P_{t-1}\, x_t \\
        k_t &= g_t / (\lambda + x_t^\top g_t) \\
        e_t &= d_t - W_{t-1}^\top x_t \\
        W_t &= W_{t-1} + k_t\, e_t^\top \\
        P_t &= (P_{t-1} - k_t\, g_t^\top) / \lambda

    The a-priori error :math:`e_t` uses the *current* weights (before this
    step's update), which is what makes the recursion equivalent to an
    exponentially-weighted least-squares solve at every step. The rank-1
    :math:`P` update preserves symmetry in exact arithmetic, so :math:`P` is
    re-symmetrised *periodically* — every :attr:`_RESYM_EVERY` steps plus once at
    the end of each pass — to control floating-point drift without paying a
    full transpose-copy on every step.

    Relationship to ridge
    ----------------------
    - :math:`\lambda = 1` (the default) is *growing-window* RLS: every past
      sample keeps unit weight, so after enough data the fit converges to the
      **ordinary least-squares** solution — equivalently ridge with
      :math:`\alpha \to 0`, the :math:`\delta` prior washing out as evidence
      accumulates. A fully-streamed :math:`\lambda = 1` fit therefore agrees with
      a batch :class:`RidgeReadoutLayer` (small ``alpha``) fit on the same data
      to a tight tolerance.
    - :math:`\lambda < 1` forgets: sample :math:`t - \tau` is weighted
      :math:`\lambda^{\tau}`, giving an effective memory of :math:`1 / (1 -
      \lambda)` samples. Use it when the input/target relation is
      non-stationary; there is **no** batch-ridge equivalent.

    Because RLS is inherently sequential (each :math:`P_t`, :math:`W_t` depends
    on its predecessor), :meth:`partial_fit` applies the updates in row-major
    stream order: a ``(B, T, F)`` batch is flattened to ``(B * T, F)`` and
    consumed one step at a time, exactly as :class:`ReadoutLayer.fit` flattens.

    Parameters
    ----------
    in_features : int
        Size of input features ``F`` (reservoir state dimension).
    out_features : int
        Size of output features ``O`` (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term. When ``True`` a constant ``1`` feature
        is appended to every input internally, so the intercept is estimated by
        the *same* RLS recursion as the slopes (and, like the slopes, is subject
        to the :math:`\delta` prior — it is not separately un-penalised as in the
        centered batch solvers). When ``False`` no intercept is fitted.
    name : str, optional
        Name for this readout layer. Used for identification in multi-readout
        architectures and as the key into ``targets`` by :class:`ESNTrainer`.
    trainable : bool, default=False
        If ``True``, weights are trainable via backpropagation *after* fitting.
    forgetting : float, default=1.0
        The forgetting factor :math:`\lambda \in (0, 1]`. ``1.0`` is
        growing-window RLS (converges to ordinary least squares); values ``< 1``
        exponentially forget old samples to track non-stationarity. Must be in
        ``(0, 1]``.
    delta : float, optional
        Initialises :math:`P = I / \delta` (default ``1e-2`` when neither
        ``delta`` nor ``alpha`` is given). Acts as a soft ridge prior on the
        early, data-starved updates: a **larger** ``delta`` shrinks the initial
        weights harder (a stronger prior), a **smaller** ``delta`` trusts the
        first samples more. Must be positive. Aliased by ``alpha`` — see below.
    alpha : float, optional
        Alias for ``delta`` (the two name the same ridge-prior knob:
        :math:`P_0 = I / \alpha`). Ridge readouts spell this regularisation
        strength ``alpha``, so it is accepted here too for API symmetry. Passing
        both ``alpha`` and a non-default ``delta`` raises. Must be positive.
    use_float64 : bool, default=True
        If ``True`` the ``P`` recursion and the weight updates run in
        ``float64`` (the fitted parameters are cast back to the layer dtype).
        The rank-1 :math:`P` update accumulates round-off over many steps, so
        double precision materially improves stability; it is cheap because the
        per-step work is only ``O(F^2)``.
    device : torch.device, optional
        Device for the layer parameters and the ``P`` buffer.
    dtype : torch.dtype, optional
        Dtype for the ``nn.Linear`` parameters.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.nn.Parameter or None
        Bias vector of shape ``(out_features,)``, or ``None`` if ``bias=False``.
    P : torch.Tensor
        The ``(F', F')`` inverse-correlation buffer, where ``F' = F + 1`` when a
        bias is fitted (the appended constant feature) and ``F`` otherwise. A
        persistent buffer, so it rides along with ``state_dict`` / ``.to(...)``.
    forgetting : float
        The forgetting factor :math:`\lambda`.
    delta : float
        The :math:`P` initialisation constant.
    n_seen : int
        Number of samples applied across all :meth:`partial_fit` calls since the
        last reset.

    Examples
    --------
    Stream a stationary problem to convergence, then predict:

    >>> readout = RLSReadout(in_features=200, out_features=3, forgetting=1.0)
    >>> for states_chunk, targets_chunk in chunks:  # doctest: +SKIP
    ...     readout.partial_fit(states_chunk, targets_chunk)
    >>> output = readout(states_chunk)  # doctest: +SKIP

    Fresh online fit in one call (resets ``P`` first):

    >>> readout.fit_online(states, targets)  # doctest: +SKIP

    See Also
    --------
    IncrementalRidgeReadoutLayer : Streaming ridge via sufficient statistics.
    RidgeReadoutLayer : Batch closed-form ridge readout.
    resdag.training.ESNTrainer : Trainer that uses this for fitting.

    References
    ----------
    .. [1] S. Haykin, "Adaptive Filter Theory," 5th ed., Pearson, 2014, ch. 10.
    """

    # Declared so type checkers know the registered buffer is a Tensor.
    P: torch.Tensor
    _n: torch.Tensor

    #: Symmetrise ``P`` every this many RLS steps (plus once at the end of every
    #: pass). The rank-1 inverse-correlation update is symmetric in *exact*
    #: arithmetic, so ``P`` only drifts off symmetry by round-off — a slow O(eps)
    #: accumulation. Re-symmetrising every step (a full transpose-copy per step)
    #: dominated the per-step cost at reservoir scale ``F`` (it tripled the
    #: two-matvec + rank-1 budget); doing it periodically keeps the per-step work
    #: at the intended ``O(F^2)`` while the end-of-pass symmetrisation guarantees
    #: ``P`` is exactly symmetric after any ``_rls_update``, for any sample count.
    _RESYM_EVERY: int = 32

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        *,
        forgetting: float = 1.0,
        delta: float | None = None,
        alpha: float | None = None,
        use_float64: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if device is not None or dtype is not None:
            # nn.Linear does not forward device/dtype in the resdag readout
            # signature, so move the freshly-created parameters explicitly. This
            # keeps the constructor signature aligned with the family's
            # device/dtype ergonomics without materialising a second Linear.
            self.to(device=device, dtype=dtype)

        if not 0.0 < forgetting <= 1.0:
            raise ValueError(f"forgetting (lambda) must be in (0, 1], got {forgetting}.")
        # ``alpha`` is an alias for ``delta`` (both parameterise P_0 = I / delta).
        # ``delta`` defaults to ``None`` (sentinel for "unset") so passing both
        # explicitly is detectable and rejected; the documented default is 1e-2.
        if alpha is not None and delta is not None:
            raise ValueError(
                "Pass either 'alpha' or 'delta' to RLSReadout, not both "
                f"(got alpha={alpha}, delta={delta}); they name the same "
                "P-initialisation knob P_0 = I / delta."
            )
        if alpha is not None:
            delta = alpha
        if delta is None:
            delta = 1e-2
        if delta <= 0.0:
            raise ValueError(f"delta (alpha) must be positive, got {delta}.")

        self.forgetting = forgetting
        self.delta = delta
        self.use_float64 = use_float64

        # The RLS recursion runs over an augmented feature space when a bias is
        # fitted: a constant 1 is appended to every input so the intercept is
        # learned by the same recursion. P therefore has size F' = F (+ 1).
        self._p_dim = in_features + 1 if self.bias is not None else in_features
        recur_dtype = resolve_solve_dtype(self.weight.dtype, use_float64)
        self.register_buffer(
            "P",
            torch.eye(self._p_dim, dtype=recur_dtype, device=self.weight.device) / self.delta,
        )
        self.register_buffer("_n", torch.zeros((), dtype=torch.long))

    @property
    def n_seen(self) -> int:
        """int : Number of samples applied across all ``partial_fit`` calls."""
        return int(self._n.item())

    @property
    def _recur_dtype(self) -> torch.dtype:
        """torch.dtype : The dtype the P recursion / weight updates run in."""
        return self.P.dtype

    @torch.no_grad()
    def reset_state(self) -> None:
        """Reset the recursion: ``P = I / delta``, zero the weights and sample count.

        Returns the layer to its freshly-constructed online state — :attr:`P` is
        reinitialised to ``I / delta``, the ``weight`` / ``bias`` parameters are
        zeroed, the sample count is cleared, and :attr:`is_fitted` becomes
        ``False``. Call this before starting a fresh stream; :meth:`fit_online`
        calls it for you.
        """
        eye = torch.eye(self._p_dim, dtype=self.P.dtype, device=self.P.device)
        self.P.copy_(eye / self.delta)
        self.weight.zero_()
        if self.bias is not None:
            self.bias.zero_()
        self._n.zero_()
        self._set_fitted(False)

    @torch.no_grad()
    def _rls_update(self, states: torch.Tensor, targets: torch.Tensor) -> None:
        """Apply the RLS recursion over a flattened ``(N, F)`` / ``(N, O)`` stream.

        Runs the canonical rank-1 RLS update once per sample, in row order,
        reading and writing :attr:`P` and the (augmented) weight matrix in
        place. The augmented weight carries the slopes in its first ``F`` rows
        and, when a bias is fitted, the intercept in the last row.

        Parameters
        ----------
        states : torch.Tensor
            Flattened state matrix of shape ``(N, in_features)``.
        targets : torch.Tensor
            Flattened target matrix of shape ``(N, out_features)``.
        """
        lam = self.forgetting
        dtype = self.P.dtype
        device = self.P.device
        fit_intercept = self.bias is not None

        x_all = states.to(dtype=dtype, device=device)
        d_all = targets.to(dtype=dtype, device=device)
        n = x_all.shape[0]

        # Augmented weight W in (F', O) layout (nn.Linear stores (O, F)).
        w = self.weight.detach().to(dtype).T.contiguous()  # (F, O)
        if fit_intercept:
            assert self.bias is not None
            bias_row = self.bias.detach().to(dtype).unsqueeze(0)  # (1, O)
            w = torch.cat([w, bias_row], dim=0)  # (F + 1, O)
            ones = torch.ones(n, 1, dtype=dtype, device=device)
            x_all = torch.cat([x_all, ones], dim=1)  # (N, F + 1)

        p = self.P
        # Reused scratch buffer for the periodic symmetrisation (holds P.T), so
        # the symmetrise step never allocates a fresh clone. Cheap because it is
        # only touched every ``_RESYM_EVERY`` steps, not every step.
        p_t = torch.empty_like(p)
        resym_every = self._RESYM_EVERY

        def _symmetrise() -> None:
            # P <- (P + P.T) / 2, in place, using the reused P.T buffer.
            p_t.copy_(p.T)
            p.add_(p_t)
            p.mul_(0.5)

        # RLS is inherently sequential: each step's P / W depend on the previous
        # step's, so the loop cannot be vectorised across time. Each step is,
        # however, pure tensor algebra (O(F'^2)), so the Python overhead is a
        # small constant per step.
        for t in range(n):
            x = x_all[t]  # (F',)
            d = d_all[t]  # (O,)

            g = p @ x  # (F',)
            denom = lam + torch.dot(x, g)
            k = g / denom  # Kalman gain (F',)

            # A-priori error using the CURRENT weights (before this update).
            pred = w.T @ x  # (O,)
            e = d - pred  # (O,)

            # Rank-1 weight update: W[:, o] += k * e[o].
            w.addr_(k, e)  # w += outer(k, e)

            # Rank-1 inverse-correlation update with forgetting.
            p.addr_(k, g, alpha=-1.0)  # P -= outer(k, g)
            p.div_(lam)
            # Periodically re-symmetrise to bleed off round-off asymmetry. The
            # rank-1 update is symmetric in exact arithmetic, so P only drifts by
            # O(eps) per step; doing this every ``resym_every`` steps (rather than
            # every step) keeps the per-step cost at the intended O(F'^2).
            if (t + 1) % resym_every == 0:
                _symmetrise()

        # End-of-pass symmetrisation: guarantees P is *exactly* symmetric after
        # the pass regardless of ``n`` (so chunked partial_fit and single-shot
        # fit both leave a symmetric P), at the cost of one extra transpose-copy
        # per pass rather than per step.
        if n > 0:
            _symmetrise()

        # Finiteness guard: a diverging recursion (forgetting too small,
        # ill-conditioned / unscaled inputs, or float32 accumulation) can drive P
        # or W to NaN/Inf while still "completing" the pass. Fail loudly here
        # rather than silently reporting is_fitted=True with a poisoned P.
        if not torch.isfinite(p).all() or not torch.isfinite(w).all():
            readout_id = f"'{self._name}'" if self._name is not None else type(self).__name__
            raise RuntimeError(
                f"{type(self).__name__} ({readout_id}): the RLS recursion diverged — "
                f"the inverse-correlation matrix P or the weights became non-finite "
                f"(NaN/Inf) during this pass. Likely causes: the forgetting factor "
                f"(lambda={self.forgetting}) is too small, the inputs are "
                f"ill-conditioned or unscaled, or the recursion is running in "
                f"float32 (use_float64={self.use_float64}). Increase forgetting toward "
                f"1.0, raise delta (P_0 = I / delta), scale/whiten the inputs, or keep "
                f"use_float64=True."
            )

        # Copy the fitted (augmented) weights back into the nn.Linear params.
        if fit_intercept:
            assert self.bias is not None
            self._copy_coefficients(w[:-1], w[-1])
        else:
            self._copy_coefficients(w, None)

        self._n += n

    @torch.no_grad()
    def partial_fit(self, states: torch.Tensor, targets: torch.Tensor) -> "RLSReadout":
        """Apply the RLS recursion over one chunk, continuing the current stream.

        Flattens a ``(B, T, F)`` chunk to ``(N, F)`` (row-major, matching
        :class:`ReadoutLayer.fit`), validates shapes and finiteness, and runs the
        rank-1 RLS update once per sample in stream order. State (:attr:`P` and
        the weights) is **carried across calls**, so feeding a long sequence in
        chunks is identical to feeding it whole — call :meth:`reset_state` (or
        :meth:`fit_online`) to start a fresh stream.

        Parameters
        ----------
        states : torch.Tensor
            State chunk of shape ``(B, T, F_in)`` or ``(N, F_in)``.
        targets : torch.Tensor
            Target chunk of shape ``(B, T, F_out)`` or ``(N, F_out)``.

        Returns
        -------
        RLSReadout
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If ``states`` and ``targets`` disagree on the sample dimension after
            flattening, if the state feature dimension does not match
            ``in_features``, if the target feature dimension does not match
            ``out_features``, or if ``states`` or ``targets`` contain non-finite
            (NaN or Inf) values.
        """
        states, targets = self._validate_chunk(states, targets, "partial_fit")
        self._rls_update(states, targets)
        # RLS refines the actual weights every step, so the layer is usable for
        # inference the moment any data has been applied.
        self._set_fitted(True)
        return self

    @torch.no_grad()
    def fit_online(self, states: torch.Tensor, targets: torch.Tensor) -> "RLSReadout":
        """Reset the recursion and stream-fit the whole ``(states, targets)`` pair.

        A readable one-call entry point that mirrors
        :meth:`IncrementalRidgeReadoutLayer`'s ergonomics: it clears any prior
        recursion state (:meth:`reset_state`) and then :meth:`partial_fit`\\ s the
        full input. Use it for a fresh online fit over a single sequence; use
        :meth:`partial_fit` directly to continue an existing stream.

        Parameters
        ----------
        states : torch.Tensor
            State matrix of shape ``(B, T, F_in)`` or ``(N, F_in)``.
        targets : torch.Tensor
            Target matrix of shape ``(B, T, F_out)`` or ``(N, F_out)``.

        Returns
        -------
        RLSReadout
            ``self``, so calls can be chained.
        """
        self.reset_state()
        return self.partial_fit(states, targets)

    def _validate_chunk(
        self, states: torch.Tensor, targets: torch.Tensor, method: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten to ``(N, F)`` and validate a state/target chunk.

        Mirrors the guards in :class:`ReadoutLayer.fit` /
        :class:`IncrementalRidgeReadoutLayer.partial_fit`: 3D→2D flattening,
        sample-count and feature-dimension agreement, and a non-finite check.

        Parameters
        ----------
        states : torch.Tensor
            State chunk of shape ``(B, T, F_in)`` or ``(N, F_in)``.
        targets : torch.Tensor
            Target chunk of shape ``(B, T, F_out)`` or ``(N, F_out)``.
        method : str
            The calling method name, used in error messages.

        Returns
        -------
        states : torch.Tensor
            Flattened states of shape ``(N, in_features)``.
        targets : torch.Tensor
            Flattened targets of shape ``(N, out_features)``.

        Raises
        ------
        ValueError
            On a shape mismatch or a non-finite input (see :meth:`partial_fit`).
        """
        return self._prepare_fit_inputs(states, targets, method)

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-shot batch fit: reset the recursion and stream the whole pass.

        Provided so :class:`RLSReadout` is a drop-in :class:`ReadoutLayer` —
        ``fit(states, targets)`` (and hence :class:`ESNTrainer`) works exactly
        like the other readouts. This is the *honest* single-shot behaviour for
        an online algorithm: it resets ``P`` and applies the RLS recursion over
        the entire (already-flattened) batch once, in stream order. For
        :math:`\\lambda = 1` and enough samples this recovers the ordinary
        least-squares fit; it is **not** a closed-form ridge solve.

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
        # The base ``fit`` has already flattened + validated, but reset the
        # recursion so a batch ``fit`` is a clean, order-independent restart.
        self.reset_state()
        self._rls_update(states, targets)
        self._set_fitted(True)
        # ``_rls_update`` already wrote the parameters; hand the base ``fit`` the
        # same values so its copy-back is a harmless no-op.
        coefs = self.weight.detach().T.to(states.dtype)
        intercept = None if self.bias is None else self.bias.detach().to(states.dtype)
        return coefs, intercept

    def __repr__(self) -> str:
        """Return string representation."""
        name_str = f", name='{self._name}'" if self._name is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f"{name_str}, "
            f"forgetting={self.forgetting}, "
            f"delta={self.delta}"
            f")"
        )
