"""
NG-RC Reservoir Layer
=====================

This module provides the Next Generation Reservoir Computer (NG-RC) sequence
layer:

- :class:`NGReservoir` — stateful sequence layer wrapping :class:`NGCell`.

See Also
--------
resdag.layers.cells.ngrc_cell : Single-step NG-RC cell (NGCell).
resdag.layers.reservoirs.base_reservoir : Abstract base (BaseReservoirLayer).
"""

from typing import Any, cast

import torch

from resdag.layers.cells.ngrc_cell import NGCell

from .base_reservoir import BaseReservoirLayer


class NGReservoir(BaseReservoirLayer):
    """
    Stateful sequence layer for Next Generation Reservoir Computing.

    Wraps :class:`NGCell` and processes a full input sequence by scanning
    causally over the time axis, maintaining an internal FIFO delay buffer as
    state.

    This is a feedforward feature map — there are no recurrent weight matrices
    and no learnable parameters.  Gradients flow through it to upstream
    modules if needed.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input vector.
    k : int, default=2
        Number of delay taps (including current input).
    s : int, default=1
        Spacing between delay taps, in timesteps.
    p : int, default=2
        Polynomial degree for nonlinear feature construction.  Only monomials
        of *exactly* degree ``p`` are emitted.  When ``p == 1`` and
        ``include_linear`` is ``True`` the nonlinear block is omitted, because
        the degree-1 monomials are identical to the linear features (see
        :class:`~resdag.layers.cells.ngrc_cell.NGCell`).
    include_constant : bool, default=True
        Whether to prepend a constant ``1.0`` to the feature vector.
    include_linear : bool, default=True
        Whether to include the linear delay-embedded features in the output.

    Attributes
    ----------
    cell : NGCell
        The inner single-step cell.
    state : torch.Tensor or None
        Current delay buffer of shape ``(batch, state_size, input_dim)``,
        or ``None`` if not yet initialized.

    Notes
    -----
    The delay buffer needs ``warmup_length = (k-1)*s`` steps to fill.  Steps
    before that produce valid features, but those features contain zeros from
    unfilled buffer slots.  The layer returns all steps; the caller can discard
    the first ``warmup_length`` outputs if desired.

    State management is inherited from :class:`BaseReservoirLayer` with the
    exception of :meth:`set_state`, which is overridden to validate the 3-D
    buffer shape ``(batch, state_size, input_dim)``.

    Examples
    --------
    Basic usage:

    >>> layer = NGReservoir(input_dim=3)
    >>> x = torch.randn(4, 100, 3)   # (batch=4, seq_len=100, d=3)
    >>> features = layer(x)
    >>> features.shape
    torch.Size([4, 100, 28])

    Double-scroll configuration (odd-symmetry, Eq. 10):

    >>> layer = NGReservoir(
    ...     input_dim=2, k=2, s=1, p=3,
    ...     include_constant=False, include_linear=True
    ... )
    >>> x = torch.randn(1, 50, 2)
    >>> features = layer(x)
    >>> features.shape
    torch.Size([1, 50, 24])

    See Also
    --------
    resdag.layers.cells.ngrc_cell.NGCell : The wrapped single-step cell.
    resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer : Base class.
    resdag.layers.readouts.cg_readout.CGReadoutLayer : Algebraic readout.
    """

    def __init__(
        self,
        input_dim: int,
        k: int = 2,
        s: int = 1,
        p: int = 2,
        include_constant: bool = True,
        include_linear: bool = True,
    ) -> None:
        cell = NGCell(
            input_dim=input_dim,
            k=k,
            s=s,
            p=p,
            include_constant=include_constant,
            include_linear=include_linear,
        )
        super().__init__(cell)

    # ------------------------------------------------------------------
    # Forward (vectorized whole-sequence fast path)
    # ------------------------------------------------------------------

    def forward_stateless(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pure whole-sequence NG-RC feature map: thread ``state`` in and out.

        Overrides the per-step loop of
        :meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.forward_stateless`
        with the vectorized
        :meth:`~resdag.layers.cells.ngrc_cell.NGCell.forward_sequence`, which
        builds every delay-embedded and monomial feature in one pass.  Like the
        base method it never reads or writes :attr:`state` and contains no
        per-timestep Python loop or data-dependent branch, so it slots into the
        compile-scan / vmap / export paths.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            ``inputs[0]`` is the input sequence of shape
            ``(batch, timesteps, input_dim)``.  Any further elements are
            ignored — NG-RC has no driving inputs.
        state : torch.Tensor
            Incoming delay buffer of shape ``(batch, state_size, input_dim)``.

        Returns
        -------
        outputs : torch.Tensor
            NG-RC features for all timesteps, shape
            ``(batch, timesteps, feature_dim)``.
        new_state : torch.Tensor
            Updated delay buffer of shape ``(batch, state_size, input_dim)``.
        """
        cell = cast(NGCell, self.cell)  # narrow from Tensor | Module for the type checker
        return cell.forward_sequence(inputs[0], state)

    def forward(
        self,
        feedback: torch.Tensor,
        *driving_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process an input sequence through the NG-RC feature map in one pass.

        NG-RC is a feedforward map, so the whole sequence is vectorized: there
        is **no per-timestep Python loop** on this path.  The output is
        numerically identical to scanning the cell step-by-step (``0.0``
        max-abs-diff, including the warmup region); the per-step
        :meth:`~resdag.layers.cells.ngrc_cell.NGCell.forward` is retained for
        streaming / stateful use and continues seamlessly after a batch call.
        This is a thin stateful wrapper over the pure
        :meth:`forward_stateless`, mirroring the base layer.

        Parameters
        ----------
        feedback : torch.Tensor
            Input sequence of shape ``(batch, timesteps, input_dim)``.
        *driving_inputs : torch.Tensor
            Ignored — NG-RC has no driving inputs.  Accepted for interface
            parity with :class:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer`.

        Returns
        -------
        torch.Tensor
            NG-RC features for all timesteps, shape
            ``(batch, timesteps, feature_dim)``.

        Raises
        ------
        ValueError
            If ``feedback`` is not 3-D.

        Notes
        -----
        The layer maintains internal state across forward calls (the FIFO delay
        buffer).  Use :meth:`reset_state` to clear it between independent
        sequences.  As in the base class, the stored state is detached from the
        autograd graph at call boundaries when
        :attr:`detach_state_between_calls` is ``True`` (truncated BPTT).
        """
        if feedback.dim() != 3:
            raise ValueError(f"Feedback must be 3D (B, T, F), got shape {feedback.shape}")

        batch_size = feedback.shape[0]
        self._maybe_init_state(batch_size, feedback.device, feedback.dtype)

        # mypy: state is guaranteed non-None after _maybe_init_state.
        assert self.state is not None
        outputs, new_state = self.forward_stateless([feedback, *driving_inputs], self.state)

        # Truncated BPTT at call boundaries (mirrors BaseReservoirLayer.forward):
        # gradients flow through the returned features within this call, but the
        # stored state must not keep the graph alive for a later forward+backward.
        # The data-dependent ``grad_fn`` branch lives here, never inside the pure
        # ``forward_stateless``.
        if self.detach_state_between_calls and new_state.grad_fn is not None:
            new_state = new_state.detach()

        self.state = new_state
        return outputs

    # ------------------------------------------------------------------
    # Properties (delegated to inner cell for convenience)
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        """Dimensionality of each input vector."""
        return cast(NGCell, self.cell).input_dim

    @property
    def feature_dim(self) -> int:
        """Total dimension of the output feature vector."""
        return cast(NGCell, self.cell).feature_dim

    @property
    def warmup_length(self) -> int:
        """
        Number of timesteps required to fill the delay buffer.

        Steps before this index contain zeros from unfilled buffer slots.
        Outputs are still produced for all timesteps; the caller may discard
        the first ``warmup_length`` outputs if needed.
        """
        return self.cell.state_size  # == (k-1)*s

    # ------------------------------------------------------------------
    # Convenience delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute lookups to the wrapped cell."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        modules = self.__dict__.get("_modules")
        if modules is not None and "cell" in modules:
            try:
                return getattr(modules["cell"], name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return repr(self.cell)
