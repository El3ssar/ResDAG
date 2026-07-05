"""Partial-square (symmetry-aware) state augmentation transform."""

import math

import torch
import torch.nn as nn


class PartialSquare(nn.Module):
    r"""Square a leading fraction (or count) of features, leave the rest.

    Squares the first ``k`` features of the last dimension in place and passes
    the remaining ``F - k`` features through unchanged, keeping the feature
    dimension the same. ``k`` is either a fixed integer count or a fraction
    ``eta`` of the feature dimension, ``k = floor(eta * F)``. Squaring only part
    of the state is a *symmetry-aware* augmentation: it injects the even
    nonlinearity needed to break the odd symmetry of a ``tanh`` reservoir while
    leaving a linear subspace intact, which improves conditioning versus
    squaring everything.

    .. math::

        g(x_i) =
        \begin{cases}
            x_i^2, & \text{if } i < k, \\
            x_i,   & \text{if } i \ge k.
        \end{cases}

    This is the ResDAG equivalent of ``PartialSquare`` in
    ReservoirComputing.jl (SciML/ReservoirComputing.jl, ``src/states.jl``),
    from Barbosa et al., "Symmetry-aware reservoir computing", *Phys. Rev. E*
    104, 045307 (2021), :doi:`10.1103/PhysRevE.104.045307`. RC.jl squares the
    leading ``floor(eta * N)`` entries of the (feature-major) state vector; here
    features live on the last axis, so the leading block of ``dim=-1`` is
    squared. The 0-based ResDAG threshold ``i < floor(eta * F)`` reproduces the
    1-based RC.jl condition ``idx <= floor(eta * N)`` element-for-element.

    Parameters
    ----------
    eta : float or int
        If a ``float`` in ``(0, 1]``, the *fraction* of leading features to
        square: ``k = floor(eta * F)`` is resolved per call against the runtime
        feature count. If an ``int``, the *count* of leading features to square
        directly (``k = eta``, clamped to ``F``).

    Examples
    --------
    Fractional ``eta`` squares a leading fraction of the features:

    >>> import torch
    >>> from resdag.layers.transforms import PartialSquare
    >>>
    >>> layer = PartialSquare(0.6)
    >>> x = torch.arange(10.0).reshape(1, 10)
    >>> layer(x)
    tensor([[ 0.,  1.,  4.,  9., 16., 25.,  6.,  7.,  8.,  9.]])

    An integer ``eta`` squares exactly that many leading features:

    >>> PartialSquare(3)(torch.tensor([[2.0, 2.0, 2.0, 2.0]]))
    tensor([[4., 4., 4., 2.]])

    See Also
    --------
    ExtendedSquare : Appends the square of every feature (doubles the dim).
    Power : Exponentiates every feature by a fixed power.
    SelectiveExponentiation : Parity-selective (even/odd index) exponentiation.
    """

    def __init__(self, eta: float | int) -> None:
        """Store and validate the fraction or count of leading features squared.

        Parameters
        ----------
        eta : float or int
            A ``float`` in ``(0, 1]`` (fraction of leading features to square)
            or a non-negative ``int`` (count of leading features to square).

        Raises
        ------
        ValueError
            If ``eta`` is a ``float`` outside ``(0, 1]`` or a negative ``int``.
        TypeError
            If ``eta`` is neither an ``int`` nor a ``float`` (``bool`` is
            rejected — it is an ``int`` subclass but a nonsensical selector).
        """
        super().__init__()
        # bool is an int subclass; reject it explicitly so True/False cannot
        # silently be read as the counts 1/0.
        if isinstance(eta, bool) or not isinstance(eta, (int, float)):
            raise TypeError(
                f"eta must be an int count or a float fraction, got {type(eta).__name__}"
            )

        self._is_fraction = isinstance(eta, float)
        if self._is_fraction:
            if not (0.0 < eta <= 1.0):
                raise ValueError(f"float eta must be in the interval (0, 1], got {eta}")
        else:
            if eta < 0:
                raise ValueError(f"int eta (feature count) must be non-negative, got {eta}")

        self.eta = eta

    def _threshold(self, num_features: int) -> int:
        """Resolve the number of leading features to square for this input.

        Parameters
        ----------
        num_features : int
            Size of the feature (last) dimension of the current input.

        Returns
        -------
        int
            The count ``k`` of leading features squared, clamped to
            ``[0, num_features]``. For a fractional ``eta`` this is
            ``floor(eta * num_features)``; for an integer ``eta`` it is ``eta``
            capped at ``num_features``.
        """
        if self._is_fraction:
            return math.floor(self.eta * num_features)
        return min(int(self.eta), num_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Square the leading ``k`` features; pass the rest through unchanged.

        The split point ``k`` is data-independent (it depends only on the static
        feature dimension and ``eta``), so the branch-free ``where(mask, x*x, x)``
        gate is torch.compile-friendly and rank-agnostic on the last dimension:
        ``(batch, features)``, ``(batch, time, features)`` and higher ranks are
        all accepted, covering both the full-sequence and single-step
        autoregressive ``forecast`` paths.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(..., features)``.

        Returns
        -------
        torch.Tensor
            Same shape as *input*; the first ``k`` features are squared, the
            remaining features are unchanged. Gradients are finite everywhere
            (``x * x`` has gradient ``2 * x`` on the squared block and ``1`` on
            the pass-through block).
        """
        num_features = input.shape[-1]
        k = self._threshold(num_features)
        indices = torch.arange(num_features, device=input.device)
        mask = indices < k
        return torch.where(mask, input * input, input)

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        kind = "fraction" if self._is_fraction else "count"
        return f"eta={self.eta} ({kind})"
