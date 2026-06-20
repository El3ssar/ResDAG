"""
NG-RC Cell
==========

This module provides the Next Generation Reservoir Computer (NG-RC) cell:

- :class:`NGCell` — single-timestep feature construction via time-delayed inputs
  and polynomial monomials; implements the LSTMCell-like API adapter so that
  NG-RC slots into the same DAG architecture as ESNCell/ESNLayer.

Reference
---------
Gauthier et al., "Next Generation Reservoir Computing" (arXiv:2106.07688v2).
Equations 5, 6, 9, 10.

See Also
--------
resdag.layers.reservoirs.ngrc : Sequence wrapper (NGReservoir).
resdag.layers.cells.esn_cell : Concrete ESN cell (ESNCell).
"""

import itertools
import warnings
from typing import cast

import torch

from .base_cell import ReservoirCell


class NGCell(ReservoirCell):
    """
    Single-timestep NG-RC feature construction cell.

    This is NOT a recurrent cell — it owns no weight matrices and has no
    recurrent dynamics.  The "state" is a FIFO delay buffer of the last
    ``(k-1)*s`` input vectors.  The cell wraps feedforward feature
    construction (time-delayed inputs + polynomial monomials) behind a
    ``forward(x, state) -> (features, new_state)`` interface so that it
    composes with the same DAG infrastructure as :class:`ESNCell`.

    Parameters
    ----------
    input_dim : int
        Dimensionality of a single input vector.
    k : int, default=2
        Number of delay taps (including the current input).  ``k=1`` means
        only the current input is used (no delay buffer).
    s : int, default=1
        Spacing between delay taps, in timesteps.
    p : int, default=2
        Polynomial degree for nonlinear feature construction.  All unique
        monomials of *exactly* degree ``p`` from the ``D = input_dim * k``
        linear features are included.

        .. note::
            When ``p == 1`` the degree-1 monomials *are* the linear features
            ``O_lin``.  To avoid emitting two identical blocks (which makes the
            readout design matrix rank-deficient), the nonlinear block is
            **omitted** whenever ``p == 1`` and ``include_linear`` is ``True``.
            With ``p == 1`` and ``include_linear=False`` the degree-1 monomials
            are kept, since they are then the only delay-embedded features.
    include_constant : bool, default=True
        Whether to prepend a constant ``1.0`` feature to the output vector.
        Set ``True`` for Lorenz63 forecasting (Eq. 9).
    include_linear : bool, default=True
        Whether to include the linear delay-embedded features ``O_lin`` in
        the output.  Set ``True`` for both Lorenz63 and double-scroll (Eq. 9,
        10).

    Attributes
    ----------
    feature_dim : int
        Total dimension of the output feature vector ``O_total``::

            D = input_dim * k
            n_nonlin = C(D + p - 1, p)
            # The nonlinear block is dropped when it would duplicate O_lin:
            emit_nonlin = not (p == 1 and include_linear)
            feature_dim = (
                int(include_constant)
                + int(include_linear) * D
                + int(emit_nonlin) * n_nonlin
            )

    state_size : int
        Number of rows in the delay buffer: ``(k - 1) * s``.  When ``k=1``
        this is 0 and the buffer is empty.
    monomial_indices : torch.Tensor
        Long tensor of shape ``(n_monomials, p)`` containing the column
        indices into ``O_lin`` for each monomial.  Registered as a buffer.
    delay_indices : torch.Tensor
        Long tensor of shape ``(k-1,)`` containing the row indices into the
        delay buffer for extracting the delay taps.  Registered as a buffer.

    Examples
    --------
    Basic usage (k=2, s=1, p=2):

    >>> cell = NGCell(input_dim=3)
    >>> x = torch.randn(4, 3)                          # (batch=4, d=3)
    >>> state = cell.init_state(4, x.device, x.dtype)  # (4, 1, 3)
    >>> features, new_state = cell([x], state)
    >>> features.shape
    torch.Size([4, 28])
    >>> new_state.shape
    torch.Size([4, 1, 3])

    No-delay mode (k=1):

    >>> cell = NGCell(input_dim=3, k=1)
    >>> state = cell.init_state(1, 'cpu', torch.float32)
    >>> features, new_state = cell([torch.randn(1, 3)], state)
    >>> features.shape
    torch.Size([1, 10])

    See Also
    --------
    resdag.layers.reservoirs.ngrc.NGReservoir : Sequence wrapper for this cell.
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
        super().__init__()

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if s < 1:
            raise ValueError(f"s must be >= 1, got {s}")
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")

        self.input_dim = input_dim
        self.k = k
        self.s = s
        self.p = p
        self.include_constant = include_constant
        self.include_linear = include_linear

        # D = total linear feature dimension
        linear_feature_dim = input_dim * k

        # Precompute delay-tap row indices into the buffer (shape: (k-1,)).
        # For j = 1, ..., k-1:  X_{i - j*s} lives at row (k-1-j)*s of the
        # buffer (which holds inputs chronologically oldest-first).
        if k > 1:
            delay_idx = torch.tensor([(k - 1 - j) * s for j in range(1, k)], dtype=torch.long)
        else:
            delay_idx = torch.zeros(0, dtype=torch.long)
        self.register_buffer("delay_indices", delay_idx)

        # Precompute monomial index tuples (shape: (n_monomials, p)).
        raw_indices = list(itertools.combinations_with_replacement(range(linear_feature_dim), p))
        monomial_idx = torch.tensor(raw_indices, dtype=torch.long)  # (n_monomials, p)
        self.register_buffer("monomial_indices", monomial_idx)

        n_monomials = len(raw_indices)

        # When p == 1 the degree-1 monomials are exactly the columns of O_lin,
        # so emitting both O_lin and O_nonlin would duplicate every linear
        # column and make the readout design matrix rank-deficient.  Drop the
        # nonlinear block in that case; keep it when include_linear is False
        # (the degree-1 monomials are then the only delay-embedded features).
        self._emit_nonlinear: bool = not (p == 1 and include_linear)

        # Total output dimension
        self._feature_dim: int = (
            int(include_constant)
            + int(include_linear) * linear_feature_dim
            + int(self._emit_nonlinear) * n_monomials
        )

        if self._feature_dim > 10_000:
            warnings.warn(
                f"NGCell: feature_dim={self._feature_dim} exceeds 10,000. "
                f"Consider reducing k, p, or input_dim to avoid combinatorial explosion. "
                f"(linear_feature_dim={linear_feature_dim}, n_monomials={n_monomials})",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Total dimension of the O_total output vector."""
        return self._feature_dim

    @property
    def output_size(self) -> int:
        """Dimensionality of the per-step output (equals feature_dim)."""
        return self._feature_dim

    @property
    def state_size(self) -> int:
        """Number of rows in the delay buffer: ``(k-1)*s``."""
        return (self.k - 1) * self.s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return a zero-filled initial delay buffer.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.
        device : torch.device or str
            Target device for the buffer.
        dtype : torch.dtype
            Target dtype for the buffer.

        Returns
        -------
        torch.Tensor
            Zero tensor of shape ``(batch_size, state_size, input_dim)``.
            When ``k=1`` the second dimension is 0 (empty buffer).
        """
        return torch.zeros(batch_size, self.state_size, self.input_dim, device=device, dtype=dtype)

    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the NG-RC feature vector for a single timestep.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Per-timestep input slices.  ``inputs[0]`` is the current input
            of shape ``(batch, input_dim)``.  Additional elements are ignored
            (NG-RC has no driving inputs).
        state : torch.Tensor
            Delay buffer of shape ``(batch, state_size, input_dim)``.

        Returns
        -------
        features : torch.Tensor
            Output feature vector ``O_total`` of shape
            ``(batch, feature_dim)``.
        new_state : torch.Tensor
            Updated delay buffer of shape ``(batch, state_size, input_dim)``.
        """
        x = inputs[0]
        batch = x.shape[0]

        # ------------------------------------------------------------------
        # 1. Build O_lin (linear delay-embedded features, Eq. 5)
        # ------------------------------------------------------------------
        if self.k > 1:
            # Extract (k-1) delay taps from the buffer *before* update.
            # delay_indices selects rows so that taps[:,0,:] = X_{i-s},
            # taps[:,1,:] = X_{i-2s}, ..., taps[:,k-2,:] = X_{i-(k-1)s}.
            taps = state[:, self.delay_indices, :]  # (batch, k-1, input_dim)
            # O_lin = [X_i || X_{i-s} || ... || X_{i-(k-1)s}]
            o_lin = torch.cat([x.unsqueeze(1), taps], dim=1).reshape(batch, self.k * self.input_dim)
        else:
            # k=1: no delay taps, just current input
            o_lin = x  # (batch, input_dim)

        # ------------------------------------------------------------------
        # 2. Update delay buffer (FIFO: drop oldest row, append current x)
        # ------------------------------------------------------------------
        if self.state_size > 0:
            # Keep the last state_size rows after appending x.
            new_state = torch.cat([state, x.unsqueeze(1)], dim=1)[:, -self.state_size :, :]
        else:
            new_state = state  # empty tensor, no change

        # ------------------------------------------------------------------
        # 3. Nonlinear features (Eq. 6): all degree-p monomials from O_lin
        #
        # monomial_indices: (n_monomials, p)  — precomputed in __init__
        # o_lin[:, monomial_indices]: (batch, n_monomials, p) via advanced
        # indexing (broadcast over batch, gather over feature dim).
        # .prod(dim=-1): (batch, n_monomials)
        #
        # Skipped when p == 1 and include_linear is True, because the degree-1
        # monomials are exactly the columns of O_lin (see __init__).
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # 4. Assemble O_total = [c] ⊕ O_lin ⊕ O_nonlin (Eq. 9 / Eq. 10)
        # ------------------------------------------------------------------
        parts: list[torch.Tensor] = []
        if self.include_constant:
            parts.append(torch.ones(batch, 1, device=x.device, dtype=x.dtype))
        if self.include_linear:
            parts.append(o_lin)
        if self._emit_nonlinear:
            o_nonlin = o_lin[:, self.monomial_indices].prod(dim=-1)  # (batch, n_monomials)
            parts.append(o_nonlin)

        o_total = torch.cat(parts, dim=-1)  # (batch, feature_dim)

        return o_total, new_state

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute NG-RC features for a whole sequence in a single vectorized pass.

        This is the batch-forward fast path used by
        :meth:`~resdag.layers.reservoirs.ngrc.NGReservoir.forward`.  It is
        numerically identical to scanning :meth:`forward` step-by-step
        (``0.0`` max-abs-diff, including the warmup region), but contains no
        per-timestep Python loop: the delay-embedded linear features are built
        from shifted, front-zero-padded slices of the input, every monomial is
        gathered over the full ``(batch, timesteps, *)`` tensor at once, and the
        constant-ones block is allocated a single time.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape ``(batch, timesteps, input_dim)``.
        state : torch.Tensor
            Incoming delay buffer of shape ``(batch, state_size, input_dim)``,
            chronologically oldest-first.  This holds the ``(k-1)*s`` inputs
            immediately preceding ``x[:, 0]`` and supplies the delay taps for
            the first timesteps (zeros for a cold start).

        Returns
        -------
        features : torch.Tensor
            Output features ``O_total`` of shape
            ``(batch, timesteps, feature_dim)``.
        new_state : torch.Tensor
            Updated delay buffer of shape ``(batch, state_size, input_dim)``,
            holding the last ``state_size`` inputs of ``x`` (or, when the
            sequence is shorter than the buffer, carried-over history), so a
            subsequent streaming :meth:`forward` continues seamlessly.

        See Also
        --------
        NGCell.forward : Single-step counterpart driving the streaming path.
        """
        batch, seq_len, _ = x.shape
        pad = self.state_size  # == (k - 1) * s

        # ------------------------------------------------------------------
        # 1. Build O_lin for the whole sequence (Eq. 5).
        #
        # Prepend the incoming delay buffer (oldest-first) so the first
        # timesteps draw their delay taps from carried-over history; a cold
        # start contributes zeros, matching the per-step loop's empty buffer.
        # For delay tap j (0..k-1), X_{i-j*s} at output index i lives at
        # padded[:, pad + i - j*s, :], i.e. the length-T window starting at
        # column (pad - j*s).
        # ------------------------------------------------------------------
        if self.k > 1:
            padded = torch.cat([state, x], dim=1)  # (batch, pad + T, input_dim)
            taps = [
                padded[:, pad - j * self.s : pad - j * self.s + seq_len, :] for j in range(self.k)
            ]
            o_lin = torch.cat(taps, dim=-1)  # (batch, T, k * input_dim)
        else:
            # k=1: no delay taps, O_lin is just the current input.
            o_lin = x  # (batch, T, input_dim)

        # ------------------------------------------------------------------
        # 2. Updated delay buffer: the last ``pad`` rows of the padded stream
        #    (FIFO drop-oldest / append-current applied to the full sequence).
        # ------------------------------------------------------------------
        if pad > 0:
            new_state = torch.cat([state, x], dim=1)[:, -pad:, :]
        else:
            new_state = state  # empty buffer, unchanged

        # ------------------------------------------------------------------
        # 3. Assemble O_total = [c] + O_lin + O_nonlin (Eq. 9 / Eq. 10).
        #    The constant-ones block is allocated once for the whole sequence;
        #    monomials are gathered over the full (batch, T, *) tensor in one op.
        # ------------------------------------------------------------------
        parts: list[torch.Tensor] = []
        if self.include_constant:
            parts.append(torch.ones(batch, seq_len, 1, device=x.device, dtype=x.dtype))
        if self.include_linear:
            parts.append(o_lin)
        if self._emit_nonlinear:
            # o_lin[..., monomial_indices]: (batch, T, n_monomials, p)
            # .prod(dim=-1):                (batch, T, n_monomials)
            monomial_indices = cast(torch.Tensor, self.monomial_indices)
            o_nonlin = o_lin[..., monomial_indices].prod(dim=-1)
            parts.append(o_nonlin)

        o_total = torch.cat(parts, dim=-1)  # (batch, T, feature_dim)

        return o_total, new_state

    def validate_state(self, state: torch.Tensor) -> None:
        """
        Validate the 3-D delay-buffer layout used by NG-RC.

        Parameters
        ----------
        state : torch.Tensor
            Candidate state of shape ``(batch, state_size, input_dim)``.

        Raises
        ------
        ValueError
            If ``state`` is not 3-D or its non-batch dimensions do not match
            ``(state_size, input_dim)``.
        """
        expected = (self.state_size, self.input_dim)
        if state.dim() != 3 or state.shape[1:] != torch.Size(expected):
            raise ValueError(
                f"NGCell.validate_state: expected a 3-D delay buffer of shape "
                f"(batch, {self.state_size}, {self.input_dim}); "
                f"got tensor of shape {tuple(state.shape)}."
            )

    def __repr__(self) -> str:
        return (
            f"NGCell("
            f"input_dim={self.input_dim}, "
            f"k={self.k}, "
            f"s={self.s}, "
            f"p={self.p}, "
            f"feature_dim={self.feature_dim}"
            f")"
        )
