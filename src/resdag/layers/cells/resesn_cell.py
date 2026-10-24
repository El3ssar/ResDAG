"""
ResESN Cell
===========

This module provides the Residual Echo State Network (ResESN) reservoir cell:

- `ResESNCell` — single-timestep ResESN update with an orthogonal skip
  connection and decoupled skip / nonlinear weights; owns all parameters.

References
----------
A. Ceni and C. Gallicchio, "Residual Echo State Networks: Residual recurrent
neural networks with stable dynamics and fast learning," Neurocomputing (2024).
Shallow-model equation reviewed as Eq. (2) in A. Ceni et al., "Deep Residual
Echo State Networks," arXiv:2508.21172 (2025).

See Also
--------
resdag.layers.cells._skip_cell : Shared orthogonal-skip base class.
resdag.layers.cells.es2n_cell : ES2N — the convex-combination sibling.
"""

import math

from ._skip_cell import _SkipReservoirCell


class ResESNCell(_SkipReservoirCell):
    r"""
    Single-timestep Residual Echo State Network (ResESN) update.

    ResESN (Ceni & Gallicchio, 2024) generalises ES2N by **decoupling** the
    skip and nonlinear branch weights: where ES2N ties them into a convex
    combination $(1 - \beta,\ \beta)$, ResESN gives each branch an
    independent scaling coefficient.  Its shallow-model update (Eq. 2 of
    arXiv:2508.21172, adapted to resdag's separate feedback / driving-input
    streams) is

    $$
    x_t = \alpha\,O\,x_{t-1}
          + \beta\,\phi(W_r x_{t-1} + W_{fb} x_{fb,t} + W_{in} u_t + b)
    $$

    where $\phi$ is the activation, $O$ a **fixed random
    orthogonal** matrix (the residual branch), $W_r$ the
    spectral-radius-scaled frozen recurrent matrix, $W_{fb}$ /
    $W_{in}$ the feedback / input weights, $b$ the random bias, and
    $\alpha$ (skip) and $\beta$ (nonlinear) the two independent
    scaling coefficients that replace the leaky-integrator leak rate.

    Relationship to other models:

    * $\alpha = 1 - \beta$ recovers **ES2N** (the convex combination).
    * $O = I$ with $1 - \alpha = \beta$ recovers a **leaky ESN**.

    Like ES2N, ResESN takes **no** `leak_rate` (the two branch weights
    subsume it) and no train-mode `noise` (not part of the formulation).

    Parameters
    ----------
    reservoir_size, feedback_size, input_size, spectral_radius, bias, \
bias_scaling, activation, trainable, feedback_initializer, input_initializer, \
topology, orthogonal_scaling, seed
        See `resdag.layers.cells._skip_cell._SkipReservoirCell` and
        `resdag.layers.cells.esn_cell.ESNCell` for the shared meanings.
    alpha : float, default=0.5
        Scaling coefficient of the orthogonal skip (residual) branch.  Must be
        finite.  Larger values weight the linear memory branch more heavily; the
        Echo State Property holds under a contractivity condition analogous to a
        standard ESN's (see the paper).  A *live* knob validated on assignment.
    beta : float, default=0.9
        Scaling coefficient of the nonlinear branch.  Must be finite.  A *live*
        knob validated on assignment.

    Attributes
    ----------
    skip_matrix : torch.Tensor
        Frozen orthogonal skip matrix `O` (see the base class).
    weight_hh, weight_feedback, weight_input, bias_h
        Recurrent, feedback, input weights and bias (see
        `resdag.layers.cells.esn_cell.ESNCell`).

    Examples
    --------
    >>> import torch
    >>> cell = ResESNCell(reservoir_size=64, feedback_size=3, alpha=0.6, beta=0.4)
    >>> state = cell.init_state(2, torch.device("cpu"), torch.float32)
    >>> out, new_state = cell([torch.randn(2, 3)], state)
    >>> out.shape
    torch.Size([2, 64])

    See Also
    --------
    resdag.layers.reservoirs.resesn.ResESNLayer : Layer that sequences this cell.
    resdag.layers.cells.es2n_cell.ES2NCell : Convex-combination (ES2N) variant.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        alpha: float = 0.5,
        beta: float = 0.9,
        **kwargs: object,
    ) -> None:
        # Backing fields seeded before super().__init__ so the property setters
        # (shared by construction and live writes) have targets; re-assigned
        # below once the object is fully initialised.
        self._alpha: float = 0.5
        self._beta: float = 0.9
        super().__init__(reservoir_size=reservoir_size, feedback_size=feedback_size, **kwargs)  # type: ignore[arg-type]
        self.alpha = alpha
        self.beta = beta

    @property
    def alpha(self) -> float:
        """
        float : Scaling of the orthogonal skip (residual) branch.

        A *live* knob: assigning to it (directly or through
        `ResESNLayer.alpha = ...`) validates the value and takes effect on the
        next forward.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"alpha must be finite, got {value}")
        self._alpha = scalar

    @property
    def beta(self) -> float:
        """
        float : Scaling of the nonlinear branch.

        A *live* knob: assigning to it (directly or through
        `ResESNLayer.beta = ...`) validates the value and takes effect on the
        next forward.
        """
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"beta must be finite, got {value}")
        self._beta = scalar

    def _branch_weights(self) -> tuple[float, float]:
        """Return `(alpha, beta)` — the decoupled ResESN branch weights.

        The skip branch is weighted by `alpha` and the nonlinear branch by
        `beta`, matching Eq. 2 of arXiv:2508.21172.
        """
        return self._alpha, self._beta

    def __repr__(self) -> str:
        """Return string representation."""
        input_str = f", input_size={self.input_size}" if self.input_size is not None else ""
        return (
            f"ResESNCell("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"spectral_radius={self.spectral_radius}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}"
            f")"
        )
