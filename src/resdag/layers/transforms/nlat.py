"""Nonlinear algebraic transforms (NLAT2 / NLAT3) of reservoir states.

Implements the T₂ and T₃ nonlinear algebraic transforms of Chattopadhyay,
Hassanzadeh & Subramanian (2020). Both replace selected features with products
of neighbouring features, in place (the feature dimension is unchanged). The
sibling T₁ transform (square every other feature) is already provided by
:class:`~resdag.layers.transforms.SelectiveExponentiation` with
``index=0, exponent=2`` — see its docstring for the NLAT1/Pathak equivalence.

Index convention
----------------
ReservoirComputing.jl (SciML/ReservoirComputing.jl, ``src/states.jl``) defines
these on a **1-based, feature-major** state vector and modifies *odd* Julia
indices. ResDAG features live on the **last (0-based) axis**, so a Julia index
``idx`` maps to the ResDAG feature ``i = idx - 1`` and RC.jl's "odd ``idx``"
becomes "even ``i``". The 0-based conditions below reproduce RC.jl's worked
examples element-for-element (verified against the ``states.jl`` docstrings):

- **NLAT2** — RC.jl ``for idx>1, isodd(idx): x[idx]=x[idx-1]*x[idx-2]``
  ⇒ for even ``i ≥ 2``: ``out[i] = x[i-1] * x[i-2]``.
- **NLAT3** — RC.jl ``for 1<idx<N, isodd(idx): x[idx]=x[idx-1]*x[idx+1]``
  ⇒ for even ``i`` with ``2 ≤ i ≤ N-2``: ``out[i] = x[i-1] * x[i+1]``.
"""

import torch
import torch.nn as nn


def _feature_parity_mask(num_features: int, device: torch.device) -> torch.Tensor:
    """Return a 0-based index vector for the feature dimension.

    Parameters
    ----------
    num_features : int
        Size of the feature (last) dimension.
    device : torch.device
        Device the mask should live on.

    Returns
    -------
    torch.Tensor
        Long tensor ``[0, 1, ..., num_features - 1]`` on *device*.
    """
    return torch.arange(num_features, device=device)


class NLAT2(nn.Module):
    r"""T₂ nonlinear algebraic transform: ``out[i] = x[i-1] * x[i-2]``.

    Replaces each even-0-based feature ``i ≥ 2`` with the product of its two
    preceding features and leaves every other feature unchanged, keeping the
    feature dimension the same. Introducing products of neighbouring states adds
    even nonlinear structure to a ``tanh`` reservoir without growing the state.

    .. math::

        \tilde{x}_i =
        \begin{cases}
            x_{i-1}\, x_{i-2}, & i \ge 2 \text{ and } i \text{ even (0-based)}; \\
            x_i,               & \text{otherwise}.
        \end{cases}

    This is the ResDAG equivalent of ``NLAT2`` in ReservoirComputing.jl
    (SciML/ReservoirComputing.jl, ``src/states.jl``), the T₂ algorithm of
    Chattopadhyay, Hassanzadeh & Subramanian, "Data-driven predictions of a
    multiscale Lorenz 96 chaotic system using machine-learning methods:
    reservoir computing, artificial neural network, and long short-term memory
    network", *Nonlinear Processes in Geophysics* 27, 373–389 (2020),
    :doi:`10.5194/npg-27-373-2020`. RC.jl's 1-based rule
    (``idx > 1`` and ``isodd(idx)``: ``x[idx] = x[idx-1] * x[idx-2]``) maps to
    the 0-based rule above; the two agree element-for-element on RC.jl's own
    worked example (see module docstring).

    The transform is *shape-static*: which features are rewritten depends only
    on the (static) feature dimension, never on the input values, so the forward
    is branch-free over the data and torch.compile-friendly.

    Notes
    -----
    This layer has no configuration and no learnable parameters.

    Examples
    --------
    >>> import torch
    >>> from resdag.layers.transforms import NLAT2
    >>>
    >>> layer = NLAT2()
    >>> x = torch.arange(10.0).reshape(1, 10)
    >>> layer(x)
    tensor([[ 0.,  1.,  0.,  3.,  6.,  5., 20.,  7., 42.,  9.]])

    See Also
    --------
    NLAT3 : T₃ transform using the surrounding neighbours ``x[i-1] * x[i+1]``.
    SelectiveExponentiation : T₁/NLAT1 (square every even-0-based feature).
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Rewrite even features ``i ≥ 2`` as ``x[i-1] * x[i-2]`` along ``dim=-1``.

        Rank-agnostic on the feature (last) dimension: ``(batch, features)``,
        ``(batch, time, features)`` and higher ranks are all accepted, so the
        layer works in both the full-sequence and single-step autoregressive
        ``forecast`` paths. Feature dimensions below ``3`` have no eligible
        position and pass through unchanged.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(..., features)``.

        Returns
        -------
        torch.Tensor
            Same shape as *input*; selected features hold the neighbour product
            ``x[i-1] * x[i-2]``, the rest are unchanged. Gradients are finite
            everywhere (each product is bilinear in the inputs).
        """
        num_features = input.shape[-1]
        if num_features < 3:
            return input

        # prod[..., j] = x[..., j+1] * x[..., j] for j in 0..N-3, which are the
        # products x[i-1] * x[i-2] aligned to output positions i = 2..N-1. Pad
        # two zeros at the front so it lines up with feature indices 0..N-1.
        prod = input[..., 1:-1] * input[..., :-2]
        prod = torch.nn.functional.pad(prod, (2, 0))

        idx = _feature_parity_mask(num_features, input.device)
        mask = (idx >= 2) & (idx % 2 == 0)
        return torch.where(mask, prod, input)


class NLAT3(nn.Module):
    r"""T₃ nonlinear algebraic transform: ``out[i] = x[i-1] * x[i+1]``.

    Replaces each even-0-based feature ``i`` in the interior (``2 ≤ i ≤ N-2``)
    with the product of its immediately preceding and following features and
    leaves every other feature unchanged, keeping the feature dimension the
    same. Like :class:`NLAT2` it injects even nonlinear structure into a
    ``tanh`` reservoir, but couples each feature to *both* of its neighbours.

    .. math::

        \tilde{x}_i =
        \begin{cases}
            x_{i-1}\, x_{i+1}, & 2 \le i \le N-2 \text{ and } i \text{ even (0-based)}; \\
            x_i,               & \text{otherwise}.
        \end{cases}

    This is the ResDAG equivalent of ``NLAT3`` in ReservoirComputing.jl
    (SciML/ReservoirComputing.jl, ``src/states.jl``), the T₃ algorithm of
    Chattopadhyay, Hassanzadeh & Subramanian, "Data-driven predictions of a
    multiscale Lorenz 96 chaotic system using machine-learning methods:
    reservoir computing, artificial neural network, and long short-term memory
    network", *Nonlinear Processes in Geophysics* 27, 373–389 (2020),
    :doi:`10.5194/npg-27-373-2020`. RC.jl's 1-based rule
    (``1 < idx < N`` and ``isodd(idx)``: ``x[idx] = x[idx-1] * x[idx+1]``) maps
    to the 0-based rule above; the last eligible feature is excluded when it has
    no right neighbour, matching RC.jl's ``idx < lastindex`` guard. The two
    agree element-for-element on RC.jl's own worked example (see module
    docstring).

    The transform is *shape-static*: which features are rewritten depends only
    on the (static) feature dimension, never on the input values, so the forward
    is branch-free over the data and torch.compile-friendly.

    Notes
    -----
    This layer has no configuration and no learnable parameters.

    Examples
    --------
    >>> import torch
    >>> from resdag.layers.transforms import NLAT3
    >>>
    >>> layer = NLAT3()
    >>> x = torch.arange(10.0).reshape(1, 10)
    >>> layer(x)
    tensor([[ 0.,  1.,  3.,  3., 15.,  5., 35.,  7., 63.,  9.]])

    See Also
    --------
    NLAT2 : T₂ transform using the two preceding neighbours ``x[i-1] * x[i-2]``.
    SelectiveExponentiation : T₁/NLAT1 (square every even-0-based feature).
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Rewrite interior even features as ``x[i-1] * x[i+1]`` along ``dim=-1``.

        Rank-agnostic on the feature (last) dimension: ``(batch, features)``,
        ``(batch, time, features)`` and higher ranks are all accepted, so the
        layer works in both the full-sequence and single-step autoregressive
        ``forecast`` paths. Feature dimensions below ``3`` have no eligible
        interior position and pass through unchanged.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(..., features)``.

        Returns
        -------
        torch.Tensor
            Same shape as *input*; selected interior features hold the neighbour
            product ``x[i-1] * x[i+1]``, the rest are unchanged. Gradients are
            finite everywhere (each product is bilinear in the inputs).
        """
        num_features = input.shape[-1]
        if num_features < 3:
            return input

        # prod[..., j] = x[..., j] * x[..., j+2] for j in 0..N-3, which are the
        # products x[i-1] * x[i+1] aligned to output positions i = 1..N-2. Pad
        # one zero at each end so it lines up with feature indices 0..N-1.
        prod = input[..., :-2] * input[..., 2:]
        prod = torch.nn.functional.pad(prod, (1, 1))

        idx = _feature_parity_mask(num_features, input.device)
        mask = (idx >= 2) & (idx <= num_features - 2) & (idx % 2 == 0)
        return torch.where(mask, prod, input)
