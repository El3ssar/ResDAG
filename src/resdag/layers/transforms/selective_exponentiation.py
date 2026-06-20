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
        dim = input.shape[-1]

        indices = torch.arange(dim, device=input.device)
        target_parity = self.index % 2
        mask = (indices % 2) == target_parity

        # Feed a safe constant (1.0) into ``pow`` at unselected positions so the
        # pow node never evaluates a value that would back-propagate as 0 * nan
        # through torch.where (e.g. a negative unselected base with a fractional
        # exponent). pow(1, e) == 1 with a finite gradient, and the where gate
        # discards this branch for unselected positions anyway.
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
