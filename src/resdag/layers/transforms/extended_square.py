"""Extended-square state augmentation transform."""

import torch
import torch.nn as nn


class ExtendedSquare(nn.Module):
    r"""Append the elementwise square of every feature: ``[x, x**2]``.

    Concatenates the input with its elementwise square along the last dimension,
    mapping ``x`` of shape ``(..., F)`` to ``[x, x ** 2]`` of shape
    ``(..., 2F)``. Doubling the state with squared copies breaks the odd
    symmetry of a ``tanh`` reservoir (which otherwise cannot represent even
    functions of its inputs), enriching the features seen by the readout.

    .. math::

        \vec{x} = \{x_1, x_2, \dots, x_N, x_1^2, x_2^2, \dots, x_N^2\}

    This is the ResDAG equivalent of ``ExtendedSquare`` in
    ReservoirComputing.jl (SciML/ReservoirComputing.jl, ``src/states.jl``),
    introduced by Herteux & Räth, "Breaking symmetries of the reservoir
    equations in echo state networks", *Chaos* 30, 123142 (2020),
    :doi:`10.1063/5.0028993`. RC.jl stacks the squared block below the linear
    block of the (feature-major) state; here features live on the last axis, so
    the squared block is concatenated along ``dim=-1``.

    Notes
    -----
    This layer has no configuration and no learnable parameters.

    Examples
    --------
    >>> import torch
    >>> from resdag.layers.transforms import ExtendedSquare
    >>>
    >>> layer = ExtendedSquare()
    >>> x = torch.tensor([[1.0, 2.0, 3.0]])
    >>> layer(x)
    tensor([[1., 2., 3., 1., 4., 9.]])

    The feature dimension is doubled:

    >>> ExtendedSquare()(torch.randn(2, 5, 4)).shape
    torch.Size([2, 5, 8])

    See Also
    --------
    Power : Exponentiates every feature in place (no concatenation).
    PartialSquare : Squares only a leading fraction of features in place.
    Pad : Appends a single constant column.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Concatenate ``input`` with its elementwise square along ``dim=-1``.

        Rank-agnostic on the feature (last) dimension: ``(batch, features)``,
        ``(batch, time, features)`` and any higher rank are all accepted, so the
        layer works in both the full-sequence and the single-step autoregressive
        ``forecast`` paths.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(..., features)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., 2 * features)`` — the original features
            followed by their elementwise squares. Gradients flow through both
            blocks (the squared block contributes ``2 * x`` to each feature's
            gradient) and stay finite for every real input.
        """
        return torch.cat([input, input * input], dim=-1)
