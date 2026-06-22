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
        Polynomial degree for nonlinear feature construction.  See ``cumulative``
        for the two supported degree conventions.

        **Exact-degree convention** (``cumulative=False``, the default): the
        nonlinear block contains all unique monomials of *exactly* degree ``p``
        from the ``D = input_dim * k`` linear features — there are
        ``C(D + p - 1, p)`` of them.  Lower-order cross terms (degrees
        ``2, ..., p-1``) are **not** included.  For example,
        ``NGCell(p=3, include_linear=True)`` emits ``constant + linear + cubic``
        with *no* quadratic terms.  This is the historical resdag behaviour and
        matches Gauthier et al. (arXiv:2106.07688v2), whose Lorenz63 (Eq. 9) and
        double-scroll (Eq. 10) bases each use a single polynomial degree.

        **Cumulative-degree convention** (``cumulative=True``): the nonlinear
        block contains every monomial of degree ``2, ..., p`` (or ``1, ..., p``
        when ``include_linear=False``), i.e. the full polynomial basis up to
        degree ``p``.  Many NVAR / NG-RC implementations (and configs ported
        from them) expect this "degree up to ``p``" basis.

        .. note::
            When ``p == 1`` the degree-1 monomials *are* the linear features
            ``O_lin``.  To avoid emitting two identical blocks (which makes the
            readout design matrix rank-deficient), the nonlinear block is
            **omitted** whenever ``p == 1`` and ``include_linear`` is ``True``.
            With ``p == 1`` and ``include_linear=False`` the degree-1 monomials
            are kept, since they are then the only delay-embedded features.  More
            generally, whenever ``include_linear`` is ``True`` the degree-1
            monomials are dropped from the nonlinear block (in both conventions),
            since they would duplicate ``O_lin``; the cumulative basis therefore
            spans degrees ``2, ..., p`` in that case.
    cumulative : bool, default=False
        Degree convention for the nonlinear block (see ``p``).  ``False`` keeps
        only exact-degree-``p`` monomials (default, unchanged behaviour);
        ``True`` includes all monomials of every degree up to ``p`` (the
        cumulative / "degree up to ``p``" basis).
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
            # Degrees emitted in the nonlinear block.  Degree 1 is excluded
            # whenever include_linear is True (those monomials == O_lin):
            d_lo = 2 if include_linear else 1
            degrees = range(d_lo, p + 1) if cumulative else [p]
            degrees = [g for g in degrees if not (g == 1 and include_linear)]
            n_nonlin = sum(C(D + g - 1, g) for g in degrees)
            feature_dim = (
                int(include_constant)
                + int(include_linear) * D
                + n_nonlin
            )

        In the default exact-degree mode (``cumulative=False``) this reduces to
        ``int(include_constant) + int(include_linear) * D + C(D + p - 1, p)``,
        with the ``C(...)`` term dropped only in the ``p == 1`` /
        ``include_linear=True`` duplicate-column case.

    state_size : int
        Number of rows in the delay buffer: ``(k - 1) * s``.  When ``k=1``
        this is 0 and the buffer is empty.
    monomial_indices : torch.Tensor
        Long tensor of shape ``(n_monomials, p)`` containing the column
        indices into ``O_lin`` for each monomial.  Registered as a buffer.
        Each row lists the ``O_lin`` columns multiplied together to form one
        monomial.  In exact-degree mode every row has ``p`` distinct index
        slots; in cumulative mode lower-degree monomials are **right-padded**
        with ``-1`` sentinels (gathered from a prepended ``1.0`` column at
        runtime) so all rows share the same width ``p``.
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
        cumulative: bool = False,
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
        self.cumulative = cumulative
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

        # ------------------------------------------------------------------
        # Decide which monomial degrees the nonlinear block emits.
        #
        # Exact-degree mode (cumulative=False): only degree p.
        # Cumulative mode (cumulative=True):    every degree 1..p.
        #
        # When include_linear is True the degree-1 monomials are exactly the
        # columns of O_lin, so emitting them again would duplicate every linear
        # column and make the readout design matrix rank-deficient.  Drop
        # degree 1 from the nonlinear block in that case (this generalises the
        # historical p==1 special case to both conventions).
        # ------------------------------------------------------------------
        degrees = range(1, p + 1) if cumulative else range(p, p + 1)
        emit_degrees = [g for g in degrees if not (g == 1 and include_linear)]

        # Build the monomial index rows.  Exact-degree mode keeps the historical
        # tight (n_monomials, p) layout with no padding.  Cumulative mode mixes
        # degrees, so every row is right-padded to width p with the -1 sentinel,
        # which is gathered from a prepended 1.0 column (the multiplicative
        # identity) at runtime — keeping a single vectorised gather/prod.
        raw_indices: list[tuple[int, ...]] = []
        for g in emit_degrees:
            raw_indices.extend(
                itertools.combinations_with_replacement(range(linear_feature_dim), g)
            )

        self._pad_with_ones: bool = cumulative and len(emit_degrees) > 1
        if self._pad_with_ones:
            # Right-pad short monomials with the -1 ones-sentinel to width p.
            padded = [list(idx) + [-1] * (p - len(idx)) for idx in raw_indices]
            monomial_idx = (
                torch.tensor(padded, dtype=torch.long)
                if padded
                else torch.zeros(0, p, dtype=torch.long)
            )
        else:
            # Tight layout: every monomial already has exactly ``width`` factors.
            width = emit_degrees[0] if emit_degrees else p
            monomial_idx = (
                torch.tensor([list(idx) for idx in raw_indices], dtype=torch.long)
                if raw_indices
                else torch.zeros(0, width, dtype=torch.long)
            )
        self.register_buffer("monomial_indices", monomial_idx)

        n_monomials = len(raw_indices)

        # The nonlinear block is omitted entirely only when no degrees survive
        # the degree-1 filter (i.e. p == 1 and include_linear is True).
        self._emit_nonlinear: bool = len(emit_degrees) > 0

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

    def _check_input_dim(self, x: torch.Tensor) -> None:
        """
        Validate that an input's feature dimension matches :attr:`input_dim`.

        The cell builds its delay-embedded and monomial features against a
        fixed ``input_dim``; a wrong feature width would otherwise surface as a
        cryptic ``torch.cat`` size-mismatch deep inside feature construction.
        This raises a clear :class:`ValueError` naming the expected width
        instead.

        Parameters
        ----------
        x : torch.Tensor
            Input slice whose trailing dimension is the feature width:
            ``(batch, input_dim)`` for the single-step path or
            ``(batch, timesteps, input_dim)`` for the sequence path.

        Raises
        ------
        ValueError
            If ``x.shape[-1]`` does not equal :attr:`input_dim`.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"NGCell expected input with feature dimension input_dim="
                f"{self.input_dim}, but got input of shape {tuple(x.shape)} "
                f"(feature dimension {x.shape[-1]})."
            )

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
            of shape ``(batch, input_dim)``.  NG-RC has no driving inputs, so
            only ``inputs[0]`` is consumed (the wrapping
            :class:`~resdag.layers.reservoirs.ngrc.NGReservoir` rejects driving
            inputs before they reach the cell).
        state : torch.Tensor
            Delay buffer of shape ``(batch, state_size, input_dim)``.

        Returns
        -------
        features : torch.Tensor
            Output feature vector ``O_total`` of shape
            ``(batch, feature_dim)``.
        new_state : torch.Tensor
            Updated delay buffer of shape ``(batch, state_size, input_dim)``.

        Raises
        ------
        ValueError
            If ``inputs[0]``'s trailing (feature) dimension does not equal
            :attr:`input_dim`.
        """
        x = inputs[0]
        self._check_input_dim(x)
        batch = x.shape[0]

        # ------------------------------------------------------------------
        # 1. Build O_lin (linear delay-embedded features, Eq. 5)
        # ------------------------------------------------------------------
        if self.k > 1:
            # Extract (k-1) delay taps from the buffer *before* update.
            # delay_indices selects rows so that taps[:,0,:] = X_{i-s},
            # taps[:,1,:] = X_{i-2s}, ..., taps[:,k-2,:] = X_{i-(k-1)s}.
            delay_indices = cast(torch.Tensor, self.delay_indices)
            taps = state[:, delay_indices, :]  # (batch, k-1, input_dim)
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
        # 3. Nonlinear features (Eq. 6): polynomial monomials from O_lin.
        #
        # Exact-degree mode emits only degree-p monomials; cumulative mode emits
        # every degree up to p.  The gather/product is delegated to
        # :meth:`_build_nonlinear`, which handles the padded cumulative layout.
        # The block is omitted only when p == 1 and include_linear is True
        # (the degree-1 monomials are then exactly the columns of O_lin).
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
            parts.append(self._build_nonlinear(o_lin))  # (batch, n_monomials)

        o_total = torch.cat(parts, dim=-1)  # (batch, feature_dim)

        return o_total, new_state

    def _build_nonlinear(self, o_lin: torch.Tensor) -> torch.Tensor:
        """
        Gather the polynomial monomials from the linear features ``O_lin``.

        Each row of :attr:`monomial_indices` lists the ``O_lin`` columns to
        multiply together for one monomial.  In exact-degree mode every row has
        ``p`` valid column indices, so the monomial block is a single advanced
        index plus a product over the last axis.  In cumulative mode the rows
        mix degrees and are right-padded with the ``-1`` sentinel; a ``1.0``
        column is prepended to ``O_lin`` (index ``0``, the multiplicative
        identity) and the indices shifted by ``+1`` so that ``-1`` selects the
        ones column and contributes nothing to the product.

        Parameters
        ----------
        o_lin : torch.Tensor
            Linear delay-embedded features of shape ``(..., D)`` (the leading
            dimensions are ``(batch,)`` for :meth:`forward` and
            ``(batch, timesteps)`` for :meth:`forward_sequence`).

        Returns
        -------
        torch.Tensor
            Monomial block of shape ``(..., n_monomials)``.
        """
        monomial_indices = cast(torch.Tensor, self.monomial_indices)
        if self._pad_with_ones:
            # Prepend a 1.0 identity column; shift -1 sentinels to that column.
            ones = torch.ones(*o_lin.shape[:-1], 1, device=o_lin.device, dtype=o_lin.dtype)
            o_aug = torch.cat([ones, o_lin], dim=-1)  # (..., 1 + D)
            gathered = o_aug[..., monomial_indices + 1]  # (..., n_monomials, p)
        else:
            gathered = o_lin[..., monomial_indices]  # (..., n_monomials, width)
        return gathered.prod(dim=-1)  # (..., n_monomials)

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

        Raises
        ------
        ValueError
            If ``x``'s trailing (feature) dimension does not equal
            :attr:`input_dim`.

        See Also
        --------
        NGCell.forward : Single-step counterpart driving the streaming path.
        """
        self._check_input_dim(x)
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
            parts.append(self._build_nonlinear(o_lin))  # (batch, T, n_monomials)

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
            f"cumulative={self.cumulative}, "
            f"feature_dim={self.feature_dim}"
            f")"
        )
