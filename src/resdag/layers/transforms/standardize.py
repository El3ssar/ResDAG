"""Per-feature standardization transform with buffer-stored statistics.

Provides :class:`Standardize`, an ``nn.Module`` that centers and scales every
feature to zero mean and unit standard deviation. The per-feature ``mean`` and
``std`` are stored as registered buffers, so they travel with ``.to(device)``,
``.double()``, ``state_dict``/``load_state_dict``, and the
:meth:`~resdag.core.ESNModel.save_full`/``load_full`` round trip — letting
normalization live inside the model graph rather than ad-hoc user code.
"""

import torch
import torch.nn as nn


class Standardize(nn.Module):
    r"""Per-feature standardization (z-score) with a learnable-free inverse.

    Centers and scales each feature of the last dimension using stored
    statistics, applying ``(x - mean) / std`` in :meth:`forward` and the exact
    inverse ``x * std + mean`` in :meth:`inverse`. The ``mean`` and ``std``
    vectors are registered buffers of shape ``(num_features,)``, so they are
    part of the ``state_dict`` and move with ``.to()``/dtype casts and the
    :meth:`~resdag.core.ESNModel.save_full` round trip.

    Statistics may be supplied at construction or estimated from data with
    :meth:`fit`. Until either happens the layer initializes to ``mean = 0`` and
    ``std = 1`` (an identity transform).

    A small ``eps`` is added to ``std`` before division to avoid blow-up on
    constant (zero-variance) features; the same ``eps``-padded scale is used by
    :meth:`inverse`, so ``inverse(forward(x))`` reconstructs ``x`` to within
    floating-point tolerance for every feature.

    Parameters
    ----------
    num_features : int
        Size of the feature (last) dimension this layer standardizes.
    mean : torch.Tensor, optional
        Initial per-feature mean of shape ``(num_features,)``. Defaults to
        zeros (no centering until :meth:`fit` is called).
    std : torch.Tensor, optional
        Initial per-feature standard deviation of shape ``(num_features,)``.
        Defaults to ones (no scaling until :meth:`fit` is called).
    eps : float, default=1e-8
        Numerical-stability floor added to ``std`` before dividing. Guards
        against division by zero on constant features.

    Attributes
    ----------
    mean : torch.Tensor
        Registered buffer holding the per-feature mean ``(num_features,)``.
    std : torch.Tensor
        Registered buffer holding the per-feature standard deviation
        ``(num_features,)``.

    Examples
    --------
    Fit statistics from a batch, then standardize and invert:

    >>> import torch
    >>> from resdag.layers.transforms import Standardize
    >>>
    >>> layer = Standardize(num_features=3)
    >>> x = torch.randn(8, 100, 3) * 5.0 + 2.0  # (batch, time, features)
    >>> layer.fit(x)
    Standardize(num_features=3, eps=1e-08)
    >>> z = layer(x)  # ~zero mean, ~unit std per feature
    >>> torch.allclose(layer.inverse(z), x, atol=1e-5)
    True

    Inside a composable pipeline (statistics travel with ``save_full``/``.to``):

    >>> import pytorch_symbolic as ps
    >>> from resdag import ESNModel, ESNLayer, CGReadoutLayer, Standardize
    >>>
    >>> norm = Standardize(num_features=3)
    >>> norm.fit(torch.randn(4, 200, 3))
    Standardize(num_features=3, eps=1e-08)
    >>> inp = ps.Input((200, 3))
    >>> normed = norm(inp)
    >>> reservoir = ESNLayer(100, feedback_size=3)(normed)
    >>> readout = CGReadoutLayer(100, 3, name="output")(reservoir)
    >>> model = ESNModel(inp, readout)

    See Also
    --------
    Power : Per-feature exponentiation.
    SelectiveExponentiation : Parity-selective exponentiation.
    """

    # Class-level annotations so static checkers see the buffers as tensors
    # (``register_buffer`` otherwise widens them to ``Tensor | Module``).
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        num_features: int,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> None:
        """Register the per-feature mean/std buffers and the ``eps`` floor.

        Parameters
        ----------
        num_features : int
            Size of the feature dimension to standardize.
        mean : torch.Tensor, optional
            Initial per-feature mean of shape ``(num_features,)`` (default
            zeros).
        std : torch.Tensor, optional
            Initial per-feature standard deviation of shape
            ``(num_features,)`` (default ones).
        eps : float, default=1e-8
            Stability floor added to ``std`` before dividing.

        Raises
        ------
        ValueError
            If ``num_features`` is not positive, or if a supplied ``mean``/
            ``std`` is not a 1-D tensor of length ``num_features``.
        """
        super().__init__()

        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")

        self.num_features = num_features
        self.eps = eps

        mean_buffer = self._validate_stat(mean, "mean", torch.zeros(num_features))
        std_buffer = self._validate_stat(std, "std", torch.ones(num_features))

        self.register_buffer("mean", mean_buffer)
        self.register_buffer("std", std_buffer)

    def _validate_stat(
        self, value: torch.Tensor | None, name: str, default: torch.Tensor
    ) -> torch.Tensor:
        """Validate and coerce a per-feature statistic to a 1-D float tensor.

        Parameters
        ----------
        value : torch.Tensor or None
            Candidate statistic. ``None`` falls back to *default*.
        name : str
            Statistic name, used in error messages.
        default : torch.Tensor
            Tensor returned when *value* is ``None``.

        Returns
        -------
        torch.Tensor
            A detached, float-typed 1-D tensor of length ``num_features``.

        Raises
        ------
        ValueError
            If *value* is not 1-D or its length is not ``num_features``.
        """
        if value is None:
            return default

        tensor = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1D, but got shape {tuple(tensor.shape)}")
        if tensor.shape[0] != self.num_features:
            raise ValueError(
                f"{name} length ({tensor.shape[0]}) does not match "
                f"num_features ({self.num_features})"
            )
        return tensor.detach().clone()

    @torch.no_grad()
    def fit(self, input: torch.Tensor) -> "Standardize":
        """Estimate per-feature mean and std from data and store them.

        Statistics are reduced over every dimension except the last, so an
        input of shape ``(batch, time, features)`` (or any number of leading
        dims) yields per-feature vectors of length ``num_features``. The
        population standard deviation (``unbiased=False``) is used so a single
        observation does not produce a ``nan``. The buffers are updated in
        place and inherit the device/dtype of *input*.

        Parameters
        ----------
        input : torch.Tensor
            Data of shape ``(..., num_features)`` whose last dimension matches
            ``num_features``.

        Returns
        -------
        Standardize
            ``self``, so calls can be chained (e.g. ``layer.fit(x)(x)``).

        Raises
        ------
        ValueError
            If the last dimension of *input* does not equal ``num_features``.
        """
        if input.shape[-1] != self.num_features:
            raise ValueError(
                f"input feature dim ({input.shape[-1]}) does not match "
                f"num_features ({self.num_features})"
            )

        flat = input.reshape(-1, self.num_features).to(dtype=self.mean.dtype)
        self.mean.copy_(flat.mean(dim=0))
        self.std.copy_(flat.std(dim=0, unbiased=False))
        return self

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Standardize the last dimension as ``(x - mean) / (std + eps)``.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(..., num_features)``.

        Returns
        -------
        torch.Tensor
            Same shape as *input*, centered and scaled per feature. Gradients
            flow through unchanged (the stored statistics are constants).

        Raises
        ------
        ValueError
            If the last dimension of *input* does not equal ``num_features``.
        """
        if input.shape[-1] != self.num_features:
            raise ValueError(
                f"input feature dim ({input.shape[-1]}) does not match "
                f"num_features ({self.num_features})"
            )
        return (input - self.mean) / (self.std + self.eps)

    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        """Invert the transform as ``x * (std + eps) + mean``.

        The exact algebraic inverse of :meth:`forward`, using the same
        ``eps``-padded scale so that ``inverse(forward(x)) == x`` up to
        floating-point error. Use it to map standardized model outputs (e.g.
        forecasts) back to the original data scale.

        Parameters
        ----------
        input : torch.Tensor
            Standardized tensor of shape ``(..., num_features)``.

        Returns
        -------
        torch.Tensor
            Same shape as *input*, mapped back to the original feature scale.

        Raises
        ------
        ValueError
            If the last dimension of *input* does not equal ``num_features``.
        """
        if input.shape[-1] != self.num_features:
            raise ValueError(
                f"input feature dim ({input.shape[-1]}) does not match "
                f"num_features ({self.num_features})"
            )
        return input * (self.std + self.eps) + self.mean

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"num_features={self.num_features}, eps={self.eps}"
