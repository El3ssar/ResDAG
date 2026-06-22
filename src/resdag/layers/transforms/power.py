"""Element-wise power transform for reservoir feature augmentation."""

import torch
import torch.nn as nn


class Power(nn.Module):
    """Exponentiate every feature to a fixed power.

    Applies the chosen power element-wise along the last dimension. Used in
    power-augmented ESN architectures to enrich reservoir states before
    readout.

    Parameters
    ----------
    exponent : float
        Power applied to each element of the input tensor.
    sign_preserving : bool, default=False
        If ``True``, every element uses the sign-preserving power
        ``sign(x) * abs(x) ** exponent``. This keeps the forward and backward
        passes finite for *negative bases* under a non-integer ``exponent``
        (e.g. tanh reservoir states in ``[-1, 1]``, which include negatives and
        zeros). If ``False`` (default), elements use plain :func:`torch.pow`,
        whose real-valued result is ``nan`` for a negative base raised to a
        non-integer exponent, and ``inf`` for a zero base raised to a negative
        exponent.

    Notes
    -----
    With ``sign_preserving=False`` (the default), :func:`torch.pow` has two
    edge cases that silently corrupt readout inputs on common reservoir states:

    - A *negative base* with a *non-integer exponent* produces ``nan`` in both
      the forward and backward pass (the real ``pow`` is undefined there). For
      example, ``Power(0.5)`` on ``[[-4.0, 4.0]]`` yields ``[[nan, 2.0]]``.
    - A *zero base* with a *negative exponent* produces ``inf`` (division by
      zero). For example, ``Power(-1.0)`` on ``[[0.0, 2.0]]`` yields
      ``[[inf, 0.5]]``.

    Tanh reservoir states live in ``[-1, 1]`` and routinely include negative
    and zero values, so a non-integer exponent on raw states is a plausible
    augmentation choice that silently emits ``nan``/``inf`` with no diagnostic.
    Use ``sign_preserving=True`` to apply ``sign(x) * abs(x) ** exponent``,
    which stays finite for negative bases (the zero base with a negative
    exponent is still ``inf`` — ``abs(0) ** -1`` diverges — and is unaffected
    by sign preservation). For integer exponents the two modes are identical
    for non-negative bases and differ only in sign for negative bases raised to
    an *odd* integer power. Every integer exponent — even ones map negatives to
    non-negative values, odd ones (including the ``power_augmented`` default
    ``exponent=3.0``) preserve sign — is safe in either mode.

    Examples
    --------
    >>> layer = Power(exponent=2.0)
    >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    >>> layer(x)
    tensor([[ 1.,  4.,  9., 16.]])

    Sign-preserving mode keeps negative bases finite under a fractional
    exponent, where plain :func:`torch.pow` would return ``nan``:

    >>> layer = Power(exponent=0.5, sign_preserving=True)
    >>> x = torch.tensor([[-4.0, 4.0, -9.0, 9.0]])
    >>> layer(x)
    tensor([[-2.,  2., -3.,  3.]])
    """

    def __init__(self, exponent: float, sign_preserving: bool = False) -> None:
        """Store the exponent and sign-preservation mode for the forward pass.

        Parameters
        ----------
        exponent : float
            Value the input is raised to for every element.
        sign_preserving : bool, default=False
            If ``True``, apply ``sign(x) * abs(x) ** exponent`` so negative
            bases stay finite under a non-integer exponent.
        """
        super().__init__()
        self.exponent = exponent
        self.sign_preserving = sign_preserving

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Raise the input to ``self.exponent``.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(batch, ..., features)``.

        Returns
        -------
        torch.Tensor
            Same shape as *input*, with each element raised to ``exponent``.
            When ``sign_preserving`` is set, ``sign(x) * abs(x) ** exponent`` is
            used, which is finite for negative bases under a non-integer
            exponent (where plain :func:`torch.pow` returns ``nan``).
        """
        if self.sign_preserving:
            return input.sign() * input.abs().pow(self.exponent)
        return torch.pow(input, self.exponent)

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"exponent={self.exponent}, sign_preserving={self.sign_preserving}"
