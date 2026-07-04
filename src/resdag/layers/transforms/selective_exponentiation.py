"""Selective exponentiation by feature index parity."""

import torch
import torch.nn as nn


class SelectiveExponentiation(nn.Module):
    """Exponentiate even or odd feature indices based on parity.

    If ``index`` is even, even positions in the last dimension are raised to
    ``exponent``; if ``index`` is odd, odd positions are exponentiated. Other
    elements are unchanged. Used in Ott-style state-augmented ESNs.

    The transform is implemented with a :func:`torch.where` gate so that
    unselected positions never enter the ``pow`` node: they pass through with a
    gradient of ``1`` and are never poisoned by the ``pow`` backward (which is
    ``inf``/``nan`` at ``x = 0`` for ``exponent < 1``). The base fed to ``pow``
    is also masked to a safe constant at unselected positions, which keeps
    unselected gradients finite even when those positions hold negative bases
    under a non-integer exponent.

    The parity mask depends only on ``(feature_dim, device)`` — never on the
    input values or dtype — so it is computed once per ``(dim, device)`` pair and
    cached. This keeps the autoregressive ``forecast`` loop (which calls this
    layer once per step) off the per-step ``arange``/parity/``ones_like``
    allocation path. The cache is a plain dict that stays out of ``state_dict``
    and ``named_buffers``; it is rebuilt lazily when a new dim or device is seen.
    For the common integer ``exponent=2`` with ``sign_preserving=False`` the
    forward reduces to ``where(mask, x * x, x)`` (``x * x`` is bit-identical to
    ``pow(x, 2)`` and finite everywhere, so the safe-base masking is
    unnecessary); the full safe-base path is retained for every other case.

    Parameters
    ----------
    index : int
        Parity selector: even ``index`` targets even indices; odd ``index``
        targets odd indices.
    exponent : float
        Power applied to the selected positions.
    sign_preserving : bool, default=False
        If ``True``, selected positions use the sign-preserving power
        ``sign(x) * abs(x) ** exponent``. This keeps the forward and backward
        passes finite for *negative selected bases* under a non-integer
        ``exponent`` (e.g. tanh reservoir states in ``[-1, 1]``). If ``False``
        (default), selected positions use plain :func:`torch.pow`, whose
        real-valued result is ``nan`` for a negative base with a non-integer
        exponent.

    Notes
    -----
    With ``sign_preserving=False`` and a non-integer ``exponent``, selected
    positions holding a *negative* value produce ``nan`` in both the forward
    and backward pass — this is the mathematically correct behaviour of a real
    ``pow`` and matches :func:`torch.pow`. Unselected positions are always
    finite. For integer exponents (including the default ``ott_esn`` path,
    ``exponent=2.0``) the two modes are identical for non-negative bases and
    differ only in sign for negative bases raised to an *odd* integer power.

    Examples
    --------
    >>> layer = SelectiveExponentiation(index=2, exponent=2.0)
    >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    >>> layer(x)
    tensor([[1., 2., 9., 4.]])

    Sign-preserving mode keeps negative selected bases finite under a
    fractional exponent:

    >>> layer = SelectiveExponentiation(index=0, exponent=0.5, sign_preserving=True)
    >>> x = torch.tensor([[-4.0, 2.0, -9.0, 3.0]])
    >>> layer(x)
    tensor([[-2.,  2., -3.,  3.]])
    """

    def __init__(self, index: int, exponent: float, sign_preserving: bool = False) -> None:
        """Store parity selector, exponent, and sign-preservation mode.

        Parameters
        ----------
        index : int
            Determines which feature indices are exponentiated (even vs odd).
        exponent : float
            Power applied to selected positions.
        sign_preserving : bool, default=False
            If ``True``, apply ``sign(x) * abs(x) ** exponent`` at selected
            positions so negative bases stay finite under a non-integer
            exponent.
        """
        super().__init__()
        self.index = index
        self.exponent = exponent
        self.sign_preserving = sign_preserving
        # Whether ``where(mask, x * x, x)`` reproduces the forward and gradient
        # exactly: only for a plain (non-sign-preserving) integer square. ``x * x``
        # is bit-identical to ``pow(x, 2)`` and finite for every real base, so the
        # safe-base masking is redundant there.
        self._is_square = (not sign_preserving) and float(exponent) == 2.0
        # Boolean parity masks cached per (feature_dim, device). Rebuilt lazily on
        # a new dim/device. Kept out of state_dict / named_buffers on purpose.
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _get_mask(self, dim: int, device: torch.device) -> torch.Tensor:
        """Return the cached boolean parity mask for ``(dim, device)``.

        The mask is dtype-independent (boolean) and depends only on the feature
        dimension and target parity, so it is computed once per ``(dim, device)``
        pair. Broadcasting against the input's dtype happens at use time.

        Parameters
        ----------
        dim : int
            Size of the last (feature) dimension.
        device : torch.device
            Device the input tensor lives on.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(dim,)`` — ``True`` at selected positions.
        """
        key = (dim, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            indices = torch.arange(dim, device=device)
            mask = (indices % 2) == (self.index % 2)
            self._mask_cache[key] = mask
        return mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply selective exponentiation along the last dimension.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(batch, ..., features)``.

        Returns
        -------
        torch.Tensor
            Same shape as *input*; selected indices are raised to ``exponent``,
            others are unchanged. Gradients at unselected positions are always
            finite; selected positions are finite when ``sign_preserving`` is
            set or the base/exponent combination is real-valued.
        """
        mask = self._get_mask(input.shape[-1], input.device)

        # Fast path: a plain integer square is finite for every real base, so the
        # gate reduces to ``where(mask, x * x, x)``. This is bit-identical to the
        # safe-base path in both forward and gradient (``x * x == pow(x, 2)``) and
        # skips the ``ones_like`` allocation and the extra ``where``/``pow`` nodes.
        if self._is_square:
            return torch.where(mask, input * input, input)

        # General path. Feed a safe constant (1.0) into ``pow`` at unselected
        # positions so the pow node never evaluates a value that would
        # back-propagate as 0 * nan through torch.where (e.g. a negative
        # unselected base with a fractional exponent). pow(1, e) == 1 with a
        # finite gradient, and the where gate discards this branch for unselected
        # positions anyway.
        safe_base = torch.where(mask, input, torch.ones_like(input))

        if self.sign_preserving:
            powed = safe_base.sign() * safe_base.abs().pow(self.exponent)
        else:
            powed = safe_base.pow(self.exponent)

        return torch.where(mask, powed, input)

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        parity = "even" if self.index % 2 == 0 else "odd"
        return (
            f"index={self.index}, exponent={self.exponent}, "
            f"sign_preserving={self.sign_preserving}, applies_to={parity}_indices"
        )
