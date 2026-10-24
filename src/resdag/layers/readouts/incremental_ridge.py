"""
Incremental Ridge Readout Layer
===============================

This module provides `IncrementalRidgeReadoutLayer`, a readout that fits
ridge regression *incrementally* over a stream of state/target chunks. Instead of
materialising every state in memory and solving once (the single-shot path of
`CGReadoutLayer` / `CholeskyReadoutLayer`), it accumulates the
ridge **sufficient statistics** — centered cross-products, running means,
column sums and the sample count — into
registered buffers via `IncrementalRidgeReadoutLayer.partial_fit`, then
solves once in `IncrementalRidgeReadoutLayer.finalize`.

The old name `IncrementalRidgeReadout` is retained as a deprecated alias.

This is the readout half of the `DataLoader` / streaming story: a sequence too
long to fit in memory, or a `torch.utils.data.DataLoader` of windowed
chunks, can be fed chunk-by-chunk with no loss of accuracy. Because the
sufficient statistics are merged with stable centered covariance updates, the fit matches a
single full-batch CG / Cholesky fit over the concatenated data to within
floating-point tolerance.

See Also
--------
resdag.layers.readouts.CholeskyReadoutLayer : Single-shot Cholesky counterpart.
resdag.layers.readouts.CGReadoutLayer : Iterative CG ridge readout.
resdag.training.ESNTrainer : Trainer with a streaming `fit_stream` path.
"""

import sys
import warnings
from collections.abc import Callable
from typing import Any

import torch

from ._solve_utils import (
    _apply_with_solver_precision,
    resolve_gram_dtype,
    resolve_solve_dtype,
    solve_ridge,
)
from .base import ReadoutLayer

# ``pytorch_symbolic`` frames that invoke a layer's ``forward`` to trace output
# shapes while *building* the graph. During that trace the readout is not yet
# fitted, so the forward guard must stand down (the traced output is only used
# to record shapes, never for inference).
_PS_BUILD_FRAMES = frozenset({"apply_module", "_recalculate_value"})


def _in_symbolic_build() -> bool:
    """Return `True` if a `pytorch_symbolic` graph-build trace is in progress.

    Walks a few caller frames (cheap, bounded) looking for the
    `pytorch_symbolic` functions that call a layer's `forward` to record
    output shapes. Used only to let an *unfitted*
    `IncrementalRidgeReadoutLayer` survive graph construction; genuine
    inference never matches these frames.
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


class IncrementalRidgeReadoutLayer(ReadoutLayer):
    """Readout layer that fits ridge regression incrementally over chunks.

    The classic readout fit requires every state in memory at once. This layer
    instead accumulates the ridge **sufficient statistics** as chunks arrive:

    - `XtX` / `Xty` — centered cross-products when `bias=True` and
      raw products when `bias=False`,
    - `_mean_x` / `_mean_y` — stable running means for the intercept,
    - `sum_x` / `sum_y` — retained column sums,
    - `n` — the running sample count.

    Centered chunk statistics are merged using the Chan covariance formula, so
    accumulating them chunk-by-chunk and solving once is algebraically identical
    to forming the whole Gram in a
    single pass. `finalize` assembles the centered, regularised normal
    equations from the running statistics and Cholesky-solves them, recovering
    the same coefficients (within floating point) that a full-batch
    `CholeskyReadoutLayer` / `CGReadoutLayer` fit would produce.

    The ridge objective is identical to the single-shot readouts,

    $$
    \\lVert X W - Y \\rVert_2^2 + \\alpha \\lVert W \\rVert_2^2 ,
    $$

    with the intercept (when `bias=True`) recovered from the running means and
    left un-penalised, exactly as in the shared `_solve_utils` centering.

    Lifecycle
    ---------
    1. (Optionally) `reset_accumulators` to clear any prior statistics.
    2. Call `partial_fit` once per chunk to accumulate statistics.
    3. Call `finalize` once to solve and write the fitted weights.

    `is_fitted` becomes `True` **only** after `finalize`; calling
    `forward` before that raises a clear `RuntimeError`.

    Parameters
    ----------
    in_features : int
        Size of input features (reservoir state dimension).
    out_features : int
        Size of output features (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term. When `True` the running means center
        the data and an un-penalised intercept is recovered; when `False` the
        raw normal equations are solved with no intercept.
    name : str, optional
        Name for this readout layer. Used for identification in multi-readout
        architectures and as the key into `targets` by `ESNTrainer`.
    trainable : bool, default=False
        If `True`, weights are trainable via backpropagation.
    alpha : float, default=1e-6
        L2 regularization strength. Must be non-negative.
    use_float64 : bool, default=True
        If `True` the accumulator buffers and the small `(F, F)` solve run
        in `float64` and the fitted weights are cast back to the parameter
        dtype. Parameter casts such as `.float()` preserve float64 working
        statistics. MPS cannot hold these statistics; use CPU or explicitly
        opt out with `use_float64=False` for supported MPS operations.
    gram_dtype : torch.dtype, optional
        Dtype the per-chunk Gram matmuls run in. Default `None` is automatic:
        the solve dtype (`float64` with `use_float64=True`) on every device.
        Pass `torch.float32` explicitly for faster, lower-precision Gram
        formation on CUDA when the feature matrix is well-conditioned.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape `(out_features, in_features)`.
    bias : torch.nn.Parameter or None
        Bias vector of shape `(out_features,)`, or `None` if `bias=False`.
    alpha : float
        L2 regularization strength.
    n_seen : int
        Number of samples accumulated so far (across all chunks).

    Examples
    --------
    Fit over three chunks of a long sequence then finalize:

    >>> readout = IncrementalRidgeReadoutLayer(in_features=200, out_features=3)
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
    _mean_x: torch.Tensor
    _mean_y: torch.Tensor
    _statistics_version: torch.Tensor
    _n: torch.Tensor

    _PRECISION_BUFFERS = ("XtX", "Xty", "sum_x", "sum_y", "_mean_x", "_mean_y")

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
        self.register_buffer("_mean_x", torch.zeros(in_features, dtype=acc_dtype))
        self.register_buffer("_mean_y", torch.zeros(out_features, dtype=acc_dtype))
        # Version 1 stores centered cross-products when bias=True. Legacy
        # checkpoints had no version and stored cancellation-prone raw moments.
        self.register_buffer("_statistics_version", torch.ones((), dtype=torch.long))
        self.register_buffer("_n", torch.zeros((), dtype=torch.long))

        # Dtype the per-chunk Gram matmuls actually ran in (resolved from the
        # first chunk in ``partial_fit``). Recorded so ``finalize``'s accuracy
        # guard weights its condition estimate by the right machine epsilon —
        # the precision loss happens at the float32 matmul, before the float64
        # accumulation, so the accumulator dtype would understate it.
        self._chunk_gram_dtype: torch.dtype | None = None

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> "IncrementalRidgeReadoutLayer":
        _apply_with_solver_precision(
            self,
            fn,
            self._PRECISION_BUFFERS,
            use_float64=self.use_float64,
            recurse=recurse,
        )
        return self

    def _upgrade_raw_statistics(self, state: dict[str, Any], prefix: str = "") -> None:
        if prefix + "_statistics_version" in state:
            if int(state[prefix + "_statistics_version"].item()) != 1:
                raise RuntimeError("Unsupported incremental ridge statistics version")
            return
        keys = ("XtX", "Xty", "sum_x", "sum_y", "_n")
        if not all(prefix + name in state for name in keys):
            return
        count = int(state[prefix + "_n"].item())
        xx = state[prefix + "XtX"]
        dtype = resolve_solve_dtype(self.weight.dtype, self.use_float64)
        mean_x = state[prefix + "sum_x"].to(dtype=dtype) / max(count, 1)
        mean_y = state[prefix + "sum_y"].to(dtype=dtype) / max(count, 1)
        state[prefix + "XtX"] = xx.to(dtype=dtype)
        state[prefix + "Xty"] = state[prefix + "Xty"].to(dtype=dtype)
        state[prefix + "sum_x"] = state[prefix + "sum_x"].to(dtype=dtype)
        state[prefix + "sum_y"] = state[prefix + "sum_y"].to(dtype=dtype)
        if self.bias is not None and count:
            state[prefix + "XtX"] = xx.to(dtype=dtype) - count * torch.outer(mean_x, mean_x)
            state[prefix + "Xty"] = state[prefix + "Xty"].to(dtype=dtype) - count * torch.outer(
                mean_x, mean_y
            )
        state[prefix + "_mean_x"] = mean_x
        state[prefix + "_mean_y"] = mean_y
        state[prefix + "_statistics_version"] = torch.ones((), dtype=torch.long, device=xx.device)
        if count:
            warnings.warn(
                "Upgraded legacy incremental ridge raw-moment statistics. Precision "
                "already lost in old accumulators cannot be recovered; reset and refit "
                "before continuing highly offset or ill-conditioned streams.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        self._upgrade_raw_statistics(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._upgrade_raw_statistics(self._buffers)
        self.__dict__.setdefault("_chunk_gram_dtype", None)

    @property
    def n_seen(self) -> int:
        """int : Number of samples accumulated across all `partial_fit` calls."""
        return int(self._n.item())

    @torch.no_grad()
    def reset_accumulators(self) -> None:
        """Clear all accumulated statistics and the fitted flag.

        Returns the layer to its freshly-constructed state: the sufficient
        statistics are zeroed, the sample count is reset to `0`, and
        `is_fitted` becomes `False`. Call this before re-fitting a layer
        that was previously finalized.
        """
        self.XtX.zero_()
        self.Xty.zero_()
        self.sum_x.zero_()
        self.sum_y.zero_()
        self._mean_x.zero_()
        self._mean_y.zero_()
        self._n.zero_()
        self._chunk_gram_dtype = None
        self._set_fitted(False)

    @torch.no_grad()
    def partial_fit(
        self, states: torch.Tensor, targets: torch.Tensor
    ) -> "IncrementalRidgeReadoutLayer":
        """Accumulate the ridge sufficient statistics for one chunk.

        Flattens `(B, T, F)` chunks to `(N, F)`, validates shapes against
        the layer's feature dimensions, and adds the chunk's contribution to the
        running `XtX` / `Xty` / `sum_x` / `sum_y` / `n` accumulators.
        Call once per `DataLoader` batch (or per window of a long sequence);
        the centered statistics are merged so chunk boundaries do not affect the final
        fit.

        Calling `partial_fit` after `finalize` keeps accumulating onto
        the existing statistics and clears the fitted flag — the layer must be
        re-`finalize`\\ d before it can be used for inference again. Use
        `reset_accumulators` to start a fresh fit.

        Parameters
        ----------
        states : torch.Tensor
            State chunk of shape `(B, T, F_in)` or `(N, F_in)`.
        targets : torch.Tensor
            Target chunk of shape `(B, T, F_out)` or `(N, F_out)`.

        Returns
        -------
        IncrementalRidgeReadoutLayer
            `self`, so calls can be chained.

        Raises
        ------
        ValueError
            If `states` and `targets` disagree on the sample dimension after
            flattening, if the state feature dimension does not match
            `in_features`, if the target feature dimension does not match
            `out_features`, or if `states` or `targets` contain non-finite
            (NaN or Inf) values.
        """
        states, targets = self._prepare_fit_inputs(states, targets, "partial_fit")

        # Form each chunk in the requested Gram dtype (float64 by default),
        # then merge into buffers in their independently configured working dtype.
        solve_dtype = resolve_solve_dtype(states.dtype, self.use_float64)
        gram_dtype = resolve_gram_dtype(states.dtype, states.device, solve_dtype, self.gram_dtype)
        self._chunk_gram_dtype = gram_dtype
        x = states.to(gram_dtype)
        y = targets.to(gram_dtype)

        acc_dtype = self.XtX.dtype
        if self.bias is not None:
            chunk_mean_x = x.mean(dim=0)
            chunk_mean_y = y.mean(dim=0)
            centered_x = x - chunk_mean_x
            centered_y = y - chunk_mean_y
            delta_x = chunk_mean_x.to(acc_dtype) - self._mean_x
            delta_y = chunk_mean_y.to(acc_dtype) - self._mean_y
            old_count = self._n.to(acc_dtype)
            total = old_count + states.shape[0]
            correction = old_count * states.shape[0] / total
            self.XtX.add_((centered_x.T @ centered_x).to(acc_dtype))
            self.Xty.add_((centered_x.T @ centered_y).to(acc_dtype))
            self.XtX.addmm_(delta_x[:, None], (correction * delta_x)[None, :])
            self.Xty.addmm_(delta_x[:, None], (correction * delta_y)[None, :])
            self._mean_x.add_(delta_x * (states.shape[0] / total))
            self._mean_y.add_(delta_y * (states.shape[0] / total))
        else:
            self.XtX += (x.T @ x).to(acc_dtype)
            self.Xty += (x.T @ y).to(acc_dtype)
        self.sum_x += x.sum(dim=0).to(acc_dtype)
        self.sum_y += y.sum(dim=0).to(acc_dtype)
        self._n += states.shape[0]

        # Any prior fit is stale once new data arrives.
        self._set_fitted(False)
        return self

    @torch.no_grad()
    def finalize(self) -> None:
        """Solve the accumulated ridge system and write the fitted weights.

        Assembles the centered (when `bias=True`), regularised normal
        equations `(XᵀX + αI) W = Xᵀy` from the running sufficient statistics
        and solves them with a Cholesky factorisation. Recovers the un-penalised
        intercept from the running means, copies the result into the layer's
        `weight` / `bias` parameters, and sets `is_fitted` to `True`.

        Raises
        ------
        RuntimeError
            If no samples have been accumulated (`partial_fit` was never
            called, or `reset_accumulators` was called since).

        Notes
        -----
        `finalize` does **not** clear the accumulators, so it can be called
        again after additional `partial_fit` calls to refine the fit on
        more data. Call `reset_accumulators` to start over.
        """
        n = self.n_seen
        if n == 0:
            readout_id = f"'{self._name}'" if self._name is not None else type(self).__name__
            raise RuntimeError(
                f"{type(self).__name__}.finalize({readout_id}): no data accumulated. "
                f"Call partial_fit(states, targets) at least once before finalize()."
            )

        fit_intercept = self.bias is not None
        gram = self.XtX
        rhs = self.Xty

        layer_name = (
            f"{type(self).__name__}('{self._name}')"
            if self._name is not None
            else type(self).__name__
        )
        coefs = solve_ridge(  # (F, T)
            gram,
            rhs,
            self.alpha,
            layer_name=layer_name,
            gram_dtype=self._chunk_gram_dtype,
        )

        intercept = None
        if fit_intercept and self.bias is not None:
            intercept = self._mean_y - self._mean_x @ coefs
        self._copy_coefficients(coefs, intercept)

        self._set_fitted(True)

    def solve(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-shot fit: accumulate one chunk then finalize.

        Provided so `IncrementalRidgeReadoutLayer` is a drop-in
        `ReadoutLayer` — `fit(states, targets)` works exactly like the
        other readouts (it resets the accumulators, accumulates the whole batch,
        and solves). The hook returns the fitted coefficients so the base
        `fit` copy-back path is reused.

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
        self.reset_accumulators()
        self.partial_fit(states, targets)
        self.finalize()
        # ``finalize`` already wrote the parameters and flipped is_fitted; hand
        # the base ``fit`` the same values so its copy-back is a harmless no-op.
        coefs = self.weight.T.to(states.dtype)
        intercept = None if self.bias is None else self.bias.to(states.dtype)
        return coefs, intercept

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the fitted linear transformation to `input`.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(B, F)` or `(B, T, F)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(B, F_out)` or `(B, T, F_out)`.

        Raises
        ------
        RuntimeError
            If `finalize` has not been called since the last
            `partial_fit` / construction. The error names the layer and points
            at `finalize()` so the streaming lifecycle is unambiguous.

        Notes
        -----
        Two internal call sites legitimately run `forward` while unfitted and
        bypass the guard via the private `_allow_unfitted_forward` flag:
        `pytorch_symbolic` graph construction (which traces output shapes by
        calling `forward` once with placeholder data) and the trainer's
        streaming accumulation pass (where the readout's forward only flows a
        value downstream while a pre-hook accumulates statistics). Genuine
        inference always goes through the guard.

        Under `torch.compile` (the compiled forecast fast path captures this
        `forward` directly and calls it per autoregressive step) the entire
        Python guard is skipped: it would otherwise force a dynamo graph break
        on every step and silently drop the model to the ~14x-slower eager
        rollout. Fittedness is still enforced **before** the compiled loop by the
        eager teacher-forced warmup pass that `forecast` always runs first: it
        executes the full graph — including this readout's eager `forward` —
        which trips the guard for a genuinely-unfitted readout. So skipping the
        per-step guard under compile removes a redundant check, not a safety net.
        The fittedness check itself reads the sync-free `_fitted_flag` mirror
        (see `resdag.layers.readouts.ReadoutLayer.is_fitted`), never the
        `_is_fitted` buffer's `.item()`, so even the eager per-step path
        pays no device->host sync.
        """
        # Compiled fast path: never let dynamo trace the guard (a graph break
        # per step forfeits the compiled forecast). The eager warmup pass that
        # forecast runs first already trips the guard for an unfitted readout,
        # so skipping the per-step guard under compile is safe here.
        if torch.compiler.is_compiling():
            return super().forward(input)

        # Eager path: the fitted bool read is sync-free; the frame walk (the
        # expensive part) only runs when genuinely unfitted, i.e. rarely.
        if not self._fitted_flag and not self._allow_unfitted_forward and not _in_symbolic_build():
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


class IncrementalRidgeReadout(IncrementalRidgeReadoutLayer):
    """Deprecated alias for `IncrementalRidgeReadoutLayer`.

    Retained for backward compatibility.  Instantiating this class emits a
    `DeprecationWarning` and constructs an
    `IncrementalRidgeReadoutLayer`; the two are otherwise identical
    (`IncrementalRidgeReadout` is a subclass, so
    `isinstance(IncrementalRidgeReadout(...), IncrementalRidgeReadoutLayer)`
    holds).

    **Deprecated.**

    Use `IncrementalRidgeReadoutLayer` instead.
    `IncrementalRidgeReadout` will be removed in a future release.

    See Also
    --------
    IncrementalRidgeReadoutLayer : The canonical class this aliases.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "IncrementalRidgeReadout is deprecated and will be removed in a future "
            "release; use IncrementalRidgeReadoutLayer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
