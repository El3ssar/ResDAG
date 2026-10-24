"""
ES2N Cell
=========

This module provides the Edge-of-Stability State Network (ES2N) reservoir cell:

- `ES2NCell` — single-timestep ES2N update with an orthogonal skip
  connection; owns all parameters.

References
----------
A. Ceni and C. Gallicchio, "Edge of Stability Echo State Networks,"
arXiv:2308.02902 (2023/2025).

See Also
--------
resdag.layers.cells._skip_cell : Shared orthogonal-skip base class.
resdag.layers.cells.esn_cell : The leaky-ESN cell ES2N extends.
resdag.layers.cells.resesn_cell : Residual ESN — the decoupled-weight sibling.
"""

from ._skip_cell import _SkipReservoirCell


class ES2NCell(_SkipReservoirCell):
    r"""
    Single-timestep Edge-of-Stability State Network (ES2N) update.

    ES2N (Ceni & Gallicchio, arXiv:2308.02902) reconciles the Echo State
    Property with long memory by writing the reservoir state as a **convex
    combination** of a standard nonlinear ESN branch and a linear, input-free
    *orthogonal* branch.  Its update (paper Eq. 5, adapted to resdag's separate
    feedback / driving-input streams) is

    $$
    x_t = \beta\,\phi(W_r x_{t-1} + W_{fb} x_{fb,t} + W_{in} u_t + b)
          + (1 - \beta)\,O\,x_{t-1}
    $$

    where $\phi$ is the activation, $O$ is a **fixed random
    orthogonal** matrix (the skip / residual branch), $W_r$ is the
    spectral-radius-scaled frozen recurrent matrix, $W_{fb}$ / $W_{in}$
    the feedback / input weights, $b$ the random bias, and
    $\beta \in (0, 1]$ the *proximity* hyperparameter.

    The paper folds the feedback into the input term $u_t$; resdag keeps
    feedback and driving inputs as separate projections, so the projectable
    input part is $W_{fb} x_{fb,t} + W_{in} u_t + b$ and the recurrent
    part is $W_r x_{t-1}$ (inside $\phi$) plus $O x_{t-1}$
    (the skip).

    Interpretation of `beta`:

    * $\beta \to 0$ — the map approaches the orthogonal, norm-preserving
      linear map $x_t \approx O x_{t-1}$.  All Jacobian eigenvalues hover
      on the complex unit circle: the *edge of stability*, giving very long
      (near-lossless) memory (paper Thm. 3.5).
    * $\beta = 1$ — the skip vanishes and the update reduces to a standard
      **non-leaky** ESN $x_t = \phi(W_r x_{t-1} + \cdots)$.

    Because the proximity plays the memory-timescale role a leaky ESN gives to
    its leak rate, ES2N takes **no** `leak_rate` argument (and no train-mode
    `noise`, which is not part of the ES2N formulation).

    Parameters
    ----------
    reservoir_size, feedback_size, input_size, spectral_radius, bias, \
bias_scaling, activation, trainable, feedback_initializer, input_initializer, \
topology, orthogonal_scaling, seed
        See `resdag.layers.cells._skip_cell._SkipReservoirCell` and
        `resdag.layers.cells.esn_cell.ESNCell` for the shared meanings.
    beta : float, default=0.9
        Proximity to the edge of stability, in $(0, 1]$.  Smaller values
        weight the orthogonal skip more heavily (longer memory, closer to the
        edge of chaos); `1.0` recovers a standard non-leaky ESN.  A *live*
        knob validated on assignment (like a leak rate): `cell.beta = ...`
        takes effect on the next forward.

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
    >>> cell = ES2NCell(reservoir_size=64, feedback_size=3, beta=0.1)
    >>> state = cell.init_state(2, torch.device("cpu"), torch.float32)
    >>> out, new_state = cell([torch.randn(2, 3)], state)
    >>> out.shape
    torch.Size([2, 64])

    See Also
    --------
    resdag.layers.reservoirs.es2n.ES2NLayer : Layer that sequences this cell.
    resdag.layers.cells.resesn_cell.ResESNCell : Decoupled-weight variant.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        beta: float = 0.9,
        **kwargs: object,
    ) -> None:
        # ``beta`` is validated by its property setter (shared by the constructor
        # and live writes).  It is stored *before* super().__init__ so the setter
        # has its backing field ready, then re-validated below once the object is
        # fully initialised.
        self._beta: float = 0.9
        super().__init__(reservoir_size=reservoir_size, feedback_size=feedback_size, **kwargs)  # type: ignore[arg-type]
        self.beta = beta

    @property
    def beta(self) -> float:
        """
        float : Proximity to the edge of stability, in $(0, 1]$.

        A *live* knob: assigning to it (directly or through
        `ES2NLayer.beta = ...`) validates the value and takes effect on the
        next forward.  `beta` near 0 weights the orthogonal skip (long memory,
        edge of stability); `beta = 1` recovers a standard non-leaky ESN.
        """
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        scalar = float(value)
        if not 0.0 < scalar <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {value}")
        self._beta = scalar

    def _branch_weights(self) -> tuple[float, float]:
        """Return `(1 - beta, beta)` — the ES2N convex-combination weights.

        The skip branch is weighted by `1 - beta` and the nonlinear branch by
        `beta`, exactly matching paper Eq. 5.
        """
        return 1.0 - self._beta, self._beta

    def __repr__(self) -> str:
        """Return string representation."""
        input_str = f", input_size={self.input_size}" if self.input_size is not None else ""
        return (
            f"ES2NCell("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"spectral_radius={self.spectral_radius}, "
            f"beta={self.beta}"
            f")"
        )
