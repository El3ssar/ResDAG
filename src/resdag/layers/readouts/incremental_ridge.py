"""
Incremental Ridge Readout Layer
===============================

This module provides :class:`IncrementalRidgeReadout`, a readout that fits ridge
regression *incrementally* over a stream of state/target chunks. Instead of
materialising every state in memory and solving once (the single-shot path of
:class:`CGReadoutLayer` / :class:`CholeskyReadoutLayer`), it accumulates the
ridge **sufficient statistics** — the Gram matrix ``XᵀX``, the cross-term
``Xᵀy``, the column sums of ``X`` and ``y``, and the sample count ``n`` — into
registered buffers via :meth:`~IncrementalRidgeReadout.partial_fit`, then solves
once in :meth:`~IncrementalRidgeReadout.finalize`.

This is the readout half of the ``DataLoader`` / streaming story: a sequence too
long to fit in memory, or a :class:`torch.utils.data.DataLoader` of windowed
chunks, can be fed chunk-by-chunk with no loss of accuracy. Because the
sufficient statistics are exact and additive, the accumulated fit matches a
single full-batch CG / Cholesky fit over the concatenated data to within
floating-point tolerance.

See Also
--------
resdag.layers.readouts.CholeskyReadoutLayer : Single-shot Cholesky counterpart.
resdag.layers.readouts.CGReadoutLayer : Iterative CG ridge readout.
resdag.training.ESNTrainer : Trainer with a streaming ``fit_stream`` path.
"""

import sys

import torch

from ._solve_utils import resolve_gram_dtype, resolve_solve_dtype
from .base import ReadoutLayer

# ``pytorch_symbolic`` frames that invoke a layer's ``forward`` to trace output
# shapes while *building* the graph. During that trace the readout is not yet
# fitted, so the forward guard must stand down (the traced output is only used
# to record shapes, never for inference).
_PS_BUILD_FRAMES = frozenset({"apply_module", "_recalculate_value"})


def _in_symbolic_build() -> bool:
    """Return ``True`` if a ``pytorch_symbolic`` graph-build trace is in progress.

    Walks a few caller frames (cheap, bounded) looking for the
    ``pytorch_symbolic`` functions that call a layer's ``forward`` to record
    output shapes. Used only to let an *unfitted* :class:`IncrementalRidgeReadout`
    survive graph construction; genuine inference never matches these frames.
    """
    frame = sys._getframe(2) if hasattr(sys, "_getframe") else None
    depth = 0
    while frame is not None and depth < 12:
        if frame.f_code.co_name in _PS_BUILD_FRAMES and "pytorch_symbolic" in (
            frame.f_globals.get("__name__", "")
        ):
            return True
        frame = frame.f_back
        depth += 1
    return False


class IncrementalRidgeReadout(ReadoutLayer):
    """Readout layer that fits ridge regression incrementally over chunks.

    The classic readout fit requires every state in memory at once. This layer
    instead accumulates the ridge **sufficient statistics** as chunks arrive:

    - ``XtX`` — the un-centered Gram matrix ``Σ xᵢ xᵢᵀ`` of shape ``(F, F)``,
    - ``Xty`` — the cross term ``Σ xᵢ yᵢᵀ`` of shape ``(F, T)``,
    - ``sum_x`` / ``sum_y`` — the column sums of ``X`` / ``y`` (for the
      un-penalised intercept),
    - ``n`` — the running sample count.

    These statistics are **additive**, so accumulating them chunk-by-chunk and
    solving once is algebraically identical to forming the whole Gram in a
    single pass. :meth:`finalize` assembles the centered, regularised normal
    equations from the running statistics and Cholesky-solves them, recovering
    the same coefficients (within floating point) that a full-batch
    :class:`CholeskyReadoutLayer` / :class:`CGReadoutLayer` fit would produce.

    The ridge objective is identical to the single-shot readouts,

    .. math::

        \\lVert X W - Y \\rVert_2^2 + \\alpha \\lVert W \\rVert_2^2 ,

    with the intercept (when ``bias=True``) recovered from the running means and
    left un-penalised, exactly as in the shared ``_solve_utils`` centering.

    Lifecycle
    ---------
    1. (Optionally) :meth:`reset_accumulators` to clear any prior statistics.
    2. Call :meth:`partial_fit` once per chunk to accumulate statistics.
    3. Call :meth:`finalize` once to solve and write the fitted weights.

    :attr:`is_fitted` becomes ``True`` **only** after :meth:`finalize`; calling
    :meth:`forward` before that raises a clear :class:`RuntimeError`.

    Parameters
    ----------
    in_features : int
        Size of input features (reservoir state dimension).
    out_features : int
        Size of output features (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term. When ``True`` the running means center
        the data and an un-penalised intercept is recovered; when ``False`` the
        raw normal equations are solved with no intercept.
    name : str, optional
        Name for this readout layer. Used for identification in multi-readout
        architectures and as the key into ``targets`` by :class:`ESNTrainer`.
    trainable : bool, default=False
        If ``True``, weights are trainable via backpropagation.
    alpha : float, default=1e-6
        L2 regularization strength. Must be non-negative.
    use_float64 : bool, default=True
        If ``True`` the accumulator buffers and the small ``(F, F)`` solve run
        in ``float64`` and the fitted weights are cast back to the parameter
        dtype.
    gram_dtype : torch.dtype, optional
        Dtype the per-chunk Gram matmuls run in. Default ``None`` is automatic:
        ``float64`` on CPU (cheap there), the input dtype on CUDA (``float64``
        matmuls are crippled on consumer GPUs). Pass ``torch.float64`` to force
        full-precision accumulation on any device. The device-aware default is
        resolved from the first chunk seen by :meth:`partial_fit`.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.nn.Parameter or None
        Bias vector of shape ``(out_features,)``, or ``None`` if ``bias=False``.
    alpha : float
        L2 regularization strength.
    n_seen : int
        Number of samples accumulated so far (across all chunks).

    Examples
    --------
    Fit over three chunks of a long sequence then finalize:

    >>> readout = IncrementalRidgeReadout(in_features=200, out_features=3)
    >>> for states_chunk, targets_chunk in chunks:  # doctest: +SKIP
    ...     readout.partial_fit(states_chunk, targets_chunk)
    >>> readout.finalize()  # doctest: +SKIP
    >>> output = readout(states_chunk)  # doctest: +SKIP

    See Also
    --------
    CholeskyReadoutLayer : Single-shot Cholesky ridge readout.
    CGReadoutLayer : Iterative Conjugate Gradient ridge readout.
    resdag.training.ESNTrainer.fit_stream : Trainer streaming-over-DataLoader path.
    """

    # Declared so type checkers know the registered buffers are Tensors.
    XtX: torch.Tensor
    Xty: torch.Tensor
    sum_x: torch.Tensor
    sum_y: torch.Tensor
    _n: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        alpha: float = 1e-6,
        use_float64: bool = True,
        gram_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        self.alpha = alpha
        self.use_float64 = use_float64
        self.gram_dtype = gram_dtype

        # When True the forward guard is bypassed: the layer may run unfitted.
        # Used internally during graph construction (where ``pytorch_symbolic``
        # traces shapes by calling ``forward`` once) and during the trainer's
        # streaming accumulation pass (where ``forward`` only needs to flow a
        # value downstream while the pre-hook accumulates statistics). Never a
        # public attribute — genuine inference always goes through the guard.
        self._allow_unfitted_forward = False

        # Accumulator dtype: float64 when use_float64 (cheap on the small
        # (F, F) / (F, T) buffers and avoids drift across many chunks).
        acc_dtype = resolve_solve_dtype(self.weight.dtype, use_float64)
        # Persistent buffers so accumulation state survives state_dict
        # round-trips and rides along with .to(device) / .to(dtype).
        self.register_buffer("XtX", torch.zeros(in_features, in_features, dtype=acc_dtype))
        self.register_buffer("Xty", torch.zeros(in_features, out_features, dtype=acc_dtype))
        self.register_buffer("sum_x", torch.zeros(in_features, dtype=acc_dtype))
        self.register_buffer("sum_y", torch.zeros(out_features, dtype=acc_dtype))
        self.register_buffer("_n", torch.zeros((), dtype=torch.long))

    @property
    def n_seen(self) -> int:
        """int : Number of samples accumulated across all ``partial_fit`` calls."""
        return int(self._n.item())

    @torch.no_grad()
    def reset_accumulators(self) -> None:
        """Clear all accumulated statistics and the fitted flag.

        Returns the layer to its freshly-constructed state: the sufficient
        statistics are zeroed, the sample count is reset to ``0``, and
        :attr:`is_fitted` becomes ``False``. Call this before re-fitting a layer
        that was previously finalized.
        """
        self.XtX.zero_()
        self.Xty.zero_()
        self.sum_x.zero_()
        self.sum_y.zero_()
        self._n.zero_()
        self._is_fitted.fill_(False)

    @torch.no_grad()
    def partial_fit(self, states: torch.Tensor, targets: torch.Tensor) -> "IncrementalRidgeReadout":
        """Accumulate the ridge sufficient statistics for one chunk.

        Flattens ``(B, T, F)`` chunks to ``(N, F)``, validates shapes against
        the layer's feature dimensions, and adds the chunk's contribution to the
        running ``XtX`` / ``Xty`` / ``sum_x`` / ``sum_y`` / ``n`` accumulators.
        Call once per ``DataLoader`` batch (or per window of a long sequence);
        the statistics are additive so chunk boundaries do not affect the final
        fit.

        Calling ``partial_fit`` after :meth:`finalize` keeps accumulating onto
        the existing statistics and clears the fitted flag — the layer must be
        re-:meth:`finalize`\\ d before it can be used for inference again. Use
        :meth:`reset_accumulators` to start a fresh fit.

        Parameters
        ----------
        states : torch.Tensor
            State chunk of shape ``(B, T, F_in)`` or ``(N, F_in)``.
        targets : torch.Tensor
            Target chunk of shape ``(B, T, F_out)`` or ``(N, F_out)``.

        Returns
        -------
        IncrementalRidgeReadout
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
        if states.dim() == 3:
            b, t, f = states.shape
            states = states.reshape(b * t, f)
        if targets.dim() == 3:
            b, t, f = targets.shape
            targets = targets.reshape(b * t, f)

        readout_id = f"'{self._name}'" if self._name is not None else type(self).__name__
        if states.shape[0] != targets.shape[0]:
            raise ValueError(
                f"{type(self).__name__}.partial_fit({readout_id}): sample count mismatch. "
                f"States have {states.shape[0]} samples after flattening, "
                f"targets have {targets.shape[0]}."
            )
        if states.shape[1] != self.in_features:
            raise ValueError(
                f"{type(self).__name__}.partial_fit({readout_id}): state feature dimension "
                f"({states.shape[1]}) does not match readout in_features "
                f"({self.in_features})."
            )
        if targets.shape[1] != self.out_features:
            raise ValueError(
                f"{type(self).__name__}.partial_fit({readout_id}): target feature dimension "
                f"({targets.shape[1]}) does not match readout out_features "
                f"({self.out_features})."
            )

        # Reject non-finite chunks before accumulating: a NaN/Inf here would
        # poison the running XtX / Xty statistics for the whole fit, not just
        # this chunk. Mirrors the guard in ReadoutLayer.fit.
        if not torch.isfinite(states).all():
            raise ValueError(
                f"{type(self).__name__}.partial_fit({readout_id}): states contain "
                f"non-finite values (NaN or Inf). Incremental fitting requires "
                f"finite inputs — clean or impute the reservoir states (a diverged "
                f"reservoir is the usual cause) before fitting."
            )
        if not torch.isfinite(targets).all():
            raise ValueError(
                f"{type(self).__name__}.partial_fit({readout_id}): targets contain "
                f"non-finite values (NaN or Inf). Incremental fitting requires "
                f"finite inputs — clean or impute the targets before fitting."
            )

        # Per-chunk Gram matmuls run in the device-aware gram dtype (float64 on
        # CPU, input dtype on CUDA) and accumulate into the float64 buffers.
        solve_dtype = resolve_solve_dtype(states.dtype, self.use_float64)
        gram_dtype = resolve_gram_dtype(states.dtype, states.device, solve_dtype, self.gram_dtype)
        x = states.to(gram_dtype)
        y = targets.to(gram_dtype)

        acc_dtype = self.XtX.dtype
        self.XtX += (x.T @ x).to(acc_dtype)
        self.Xty += (x.T @ y).to(acc_dtype)
        self.sum_x += x.sum(dim=0).to(acc_dtype)
        self.sum_y += y.sum(dim=0).to(acc_dtype)
        self._n += states.shape[0]

        # Any prior fit is stale once new data arrives.
        self._is_fitted.fill_(False)
        return self

    @torch.no_grad()
    def finalize(self) -> None:
        """Solve the accumulated ridge system and write the fitted weights.

        Assembles the centered (when ``bias=True``), regularised normal
        equations ``(XᵀX + αI) W = Xᵀy`` from the running sufficient statistics
        and solves them with a Cholesky factorisation. Recovers the un-penalised
        intercept from the running means, copies the result into the layer's
        ``weight`` / ``bias`` parameters, and sets :attr:`is_fitted` to ``True``.

        Raises
        ------
        RuntimeError
            If no samples have been accumulated (``partial_fit`` was never
            called, or :meth:`reset_accumulators` was called since).

        Notes
        -----
        ``finalize`` does **not** clear the accumulators, so it can be called
        again after additional :meth:`partial_fit` calls to refine the fit on
        more data. Call :meth:`reset_accumulators` to start over.
        """
        n = self.n_seen
        if n == 0:
            readout_id = f"'{self._name}'" if self._name is not None else type(self).__name__
            raise RuntimeError(
                f"{type(self).__name__}.finalize({readout_id}): no data accumulated. "
                f"Call partial_fit(states, targets) at least once before finalize()."
            )

        fit_intercept = self.bias is not None
        solve_dtype = self.XtX.dtype
        original_dtype = self.weight.dtype
        nf = float(n)

        if fit_intercept:
            # Centered Gram / RHS from the running statistics:
            #   XᵀX - n x̄ᵀx̄ = XtX - (Σx)ᵀ(Σx) / n
            #   Xᵀy - n x̄ᵀȳ = Xty - (Σx)ᵀ(Σy) / n
            x_mean = (self.sum_x / nf).unsqueeze(0)  # (1, F)
            y_mean = (self.sum_y / nf).unsqueeze(0)  # (1, T)
            gram = self.XtX - nf * (x_mean.T @ x_mean)
            rhs = self.Xty - nf * (x_mean.T @ y_mean)
        else:
            gram = self.XtX
            rhs = self.Xty

        eye = torch.eye(self.in_features, dtype=solve_dtype, device=gram.device)
        a = gram + self.alpha * eye
        factor = torch.linalg.cholesky(a)
        coefs = torch.cholesky_solve(rhs, factor)  # (F, T)

        self.weight.copy_(coefs.T.to(original_dtype))
        if fit_intercept and self.bias is not None:
            intercept = (y_mean - x_mean @ coefs).squeeze(0)
            self.bias.copy_(intercept.to(self.bias.dtype))

        self._is_fitted.fill_(True)

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-shot fit: accumulate one chunk then finalize.

        Provided so :class:`IncrementalRidgeReadout` is a drop-in
        :class:`ReadoutLayer` — ``fit(states, targets)`` works exactly like the
        other readouts (it resets the accumulators, accumulates the whole batch,
        and solves). The hook returns the fitted coefficients so the base
        ``fit`` copy-back path is reused.

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
        self.reset_accumulators()
        self.partial_fit(states, targets)
        self.finalize()
        # ``finalize`` already wrote the parameters and flipped is_fitted; hand
        # the base ``fit`` the same values so its copy-back is a harmless no-op.
        coefs = self.weight.T.to(states.dtype)
        intercept = None if self.bias is None else self.bias.to(states.dtype)
        return coefs, intercept

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the fitted linear transformation to ``input``.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape ``(B, F)`` or ``(B, T, F)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, F_out)`` or ``(B, T, F_out)``.

        Raises
        ------
        RuntimeError
            If :meth:`finalize` has not been called since the last
            ``partial_fit`` / construction. The error names the layer and points
            at ``finalize()`` so the streaming lifecycle is unambiguous.

        Notes
        -----
        Two internal call sites legitimately run ``forward`` while unfitted and
        bypass the guard via the private ``_allow_unfitted_forward`` flag:
        ``pytorch_symbolic`` graph construction (which traces output shapes by
        calling ``forward`` once with placeholder data) and the trainer's
        streaming accumulation pass (where the readout's forward only flows a
        value downstream while a pre-hook accumulates statistics). Genuine
        inference always goes through the guard.
        """
        if not self._allow_unfitted_forward and not self.is_fitted and not _in_symbolic_build():
            readout_id = f"'{self._name}'" if self._name is not None else type(self).__name__
            raise RuntimeError(
                f"{type(self).__name__}({readout_id}) is not fitted: call finalize() after "
                f"partial_fit(...) before forward/inference. "
                f"({self.n_seen} sample(s) accumulated so far.)"
            )
        return super().forward(input)

    def __repr__(self) -> str:
        """Return string representation."""
        name_str = f", name='{self._name}'" if self._name is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f"{name_str}, "
            f"alpha={self.alpha}"
            f")"
        )
