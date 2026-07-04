r"""
EuSN Cell (Euler State Network)
===============================

This module provides the Euler State Network (EuSN) reservoir cell:

- :class:`EuSNCell` — a :class:`~resdag.layers.cells.continuous_cell.ContinuousReservoirCell`
  whose vector field uses an **antisymmetric** recurrent matrix minus a small
  **diffusion** term, integrated by a single forward-Euler step.

References
----------
C. Gallicchio, "Euler State Networks: Non-dissipative Reservoir Computing,"
*Neurocomputing* 531 (2023): 214-228; arXiv:2203.09382.

See Also
--------
resdag.layers.cells.continuous_cell : Shared continuous-integration base.
resdag.layers.cells.esn_cell : The leaky-ESN cell whose weights this reuses.
resdag.init.matrices.antisymmetric : Antisymmetric matrix init (general use).
"""

import torch

from resdag.init.topology import estimate_spectral_radius
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.utils.general import SeedLike

from .continuous_cell import ContinuousReservoirCell


class EuSNCell(ContinuousReservoirCell):
    r"""
    Single-timestep Euler State Network (EuSN) update.

    EuSN (Gallicchio, arXiv:2203.09382) models the reservoir as an ODE

    .. math::

        \dot h(t) = \tanh\!\big(W_h\, h(t) + W_x\, x(t) + b\big),

    with :math:`W_h` **antisymmetric** (:math:`W_h = -W_h^\top`), and integrates
    it with a single forward-Euler step of size :math:`\varepsilon`, subtracting
    a small **diffusion** term :math:`\gamma I` from the recurrence for numerical
    stability:

    .. math::

        h_t = h_{t-1} + \varepsilon\,
              \tanh\!\big((W_h - \gamma I)\, h_{t-1} + W_x\, x_t + b\big).

    The antisymmetric :math:`W_h` is a rotation generator (purely imaginary
    eigenvalues), so :math:`x \mapsto W_h x` conserves the state norm; a small
    Euler step of the resulting flow is therefore **near-norm-preserving** — a
    stable, non-dissipative, long-memory regime with local Lyapunov exponents
    near zero (the "edge of stability").  The diffusion :math:`\gamma I` adds a
    tiny contraction that damps the forward-Euler discretization error and keeps
    the recurrence from drifting outward over long sequences.

    resdag construction
    -------------------
    resdag keeps feedback and driving inputs as *separate* projections, so the
    paper's :math:`W_x x_t + b` maps to resdag's precomputed drive
    :math:`W_{fb} x_{fb,t} + W_{in} u_t + b` (the inherited
    :meth:`~resdag.layers.cells.esn_cell.ESNCell.project_inputs`).  The
    antisymmetric :math:`W_h` is obtained **on the fly** from the frozen,
    spectral-radius-scaled recurrent matrix ``weight_hh`` (``= W_r``) as
    :math:`W_r - W_r^\top`; the stored ``W_r`` is left untouched (cleaner than
    baking antisymmetry into the buffer, and matching the paper's constraint
    exactly).  The vector field is therefore

    .. math::

        f(h, u) = \tanh\!\big((W_r - W_r^\top - \gamma I)\, h + u\big),

    with :math:`u` the precomputed drive and one Euler sub-step of size
    :math:`\varepsilon` (``dt=epsilon``, ``substeps=1``).

    Because the antisymmetric part :math:`W_r - W_r^\top` has purely imaginary
    eigenvalues, a layer-level ``spectral_radius`` on ``W_r`` still controls the
    *rotation rate* of the recurrence (the largest imaginary eigenvalue
    magnitude), so the usual scaling knob remains meaningful.

    Parameters
    ----------
    reservoir_size, feedback_size, input_size, bias, bias_scaling, activation, \
trainable, feedback_initializer, input_initializer, topology, seed
        See :class:`~resdag.layers.cells.esn_cell.ESNCell`.  ``activation``
        defaults to ``"tanh"`` (the EuSN nonlinearity); other activations are
        accepted for experimentation.
    epsilon : float, default=0.01
        Forward-Euler step size :math:`\varepsilon`.  Small values sit the
        reservoir near the edge of stability (near-norm-preserving); the paper
        sweeps :math:`\varepsilon \in \{10^{-5}, \dots, 1\}`.  Stored as the
        base ``dt``.
    gamma : float, default=0.01
        Diffusion coefficient :math:`\gamma \ge 0` subtracted as :math:`\gamma I`
        from the antisymmetric recurrence.  A small positive value damps the
        Euler discretization error; ``0.0`` recovers the purely norm-preserving
        (undamped) antisymmetric flow.  A *live* knob validated on assignment.
    spectral_radius : float, optional
        Target **rotation rate** of the *effective* antisymmetric operator
        :math:`W_r - W_r^\top` — its largest ``|imaginary eigenvalue|``, the
        quantity that actually drives the dynamics — **not** the raw ``W_r``.
        ``W_r`` is rescaled so the skew part hits this radius (see the notes on
        antisymmetric scaling below).  ``None`` leaves ``W_r`` unscaled.
    substeps : int, default=1
        Fixed integration sub-steps per interval (see
        :class:`ContinuousReservoirCell`).  ``1`` is the single forward-Euler
        step of the paper.
    solver : {'euler', 'rk4', 'torchdiffeq'}, default='euler'
        Integration scheme.  ``'euler'`` is the paper's method; ``'rk4'`` and
        the adaptive ``'torchdiffeq'`` integrate the *same* vector field more
        accurately (they agree with Euler in the small-``epsilon`` limit).

    Attributes
    ----------
    gamma : float
        Diffusion coefficient (live knob).
    epsilon : float
        Euler step size (alias of :attr:`dt`).
    weight_hh, weight_feedback, weight_input, bias_h
        Recurrent (``W_r``), feedback, input weights and bias (see
        :class:`~resdag.layers.cells.esn_cell.ESNCell`).

    Examples
    --------
    >>> import torch
    >>> cell = EuSNCell(reservoir_size=64, feedback_size=3, epsilon=0.01, gamma=0.01)
    >>> state = cell.init_state(2, torch.device("cpu"), torch.float32)
    >>> out, new_state = cell([torch.randn(2, 3)], state)
    >>> out.shape
    torch.Size([2, 64])

    See Also
    --------
    resdag.layers.reservoirs.eusn.EuSNLayer : Layer that sequences this cell.
    resdag.layers.cells.continuous_cell.ContinuousReservoirCell : Integration base.
    resdag.layers.cells.esn_cell.ESNCell : Standard leaky-ESN cell.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        epsilon: float = 0.01,
        gamma: float = 0.01,
        input_size: int | None = None,
        spectral_radius: float | None = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        activation: str = "tanh",
        substeps: int = 1,
        solver: str = "euler",
        torchdiffeq_method: str = "dopri5",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        trainable: bool = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
        seed: SeedLike = None,
    ) -> None:
        # ``gamma`` is validated by its property setter (shared with live writes);
        # stash the backing field before super().__init__ so the setter is ready,
        # then re-validate below once the object is fully initialised.
        self._gamma: float = 0.0
        # ``spectral_radius`` here targets the *effective* antisymmetric operator
        # ``W_r - W_r^T`` (the rotation rate that actually drives the dynamics),
        # NOT the raw ``W_r``.  The two differ substantially: for a dense random
        # ``W_r`` scaled to radius rho, the skew part ``W_r - W_r^T`` has a much
        # larger imaginary rate (~2x+), which would push the forward-Euler step
        # far past the stable edge and make the state grow without bound.  So we
        # pass ``spectral_radius=None`` to the parent (leaving ``W_r`` un-scaled)
        # and rescale ``weight_hh`` below so ``W_r - W_r^T`` hits the target.
        self._target_skew_radius = spectral_radius
        super().__init__(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            input_size=input_size,
            spectral_radius=None,
            bias=bias,
            bias_scaling=bias_scaling,
            activation=activation,
            trainable=trainable,
            dt=epsilon,
            substeps=substeps,
            solver=solver,
            torchdiffeq_method=torchdiffeq_method,
            rtol=rtol,
            atol=atol,
            feedback_initializer=feedback_initializer,
            input_initializer=input_initializer,
            topology=topology,
            seed=seed,
        )
        # Record the requested target on the base ``spectral_radius`` slot so the
        # ``__repr__`` and any introspection report the meaningful (skew-operator)
        # value rather than ``None``.
        self.spectral_radius = spectral_radius
        if spectral_radius is not None:
            self._scale_skew_spectral_radius(spectral_radius)
        self.gamma = gamma

    def _scale_skew_spectral_radius(self, target_radius: float) -> None:
        r"""Rescale ``weight_hh`` so ``W_r - W_r^T`` has spectral radius ``target_radius``.

        The EuSN dynamics are driven by the antisymmetric operator
        :math:`W_r - W_r^\top`, whose eigenvalues are purely imaginary; its
        spectral radius is the largest ``|imaginary eigenvalue|`` (the rotation
        rate).  Scaling ``W_r`` by a scalar scales the skew part by the same
        scalar, so we measure the skew part's radius once and rescale ``W_r`` to
        put it on target.  Left a no-op when the skew part is (numerically) zero
        (e.g. a symmetric ``W_r``, whose antisymmetric part vanishes).
        """
        with torch.no_grad():
            w = self.weight_hh.data
            skew = w - w.t()
            current = estimate_spectral_radius(skew)
            if current > 1e-8:
                self.weight_hh.data.mul_(target_radius / current)

    @property
    def epsilon(self) -> float:
        """float : Forward-Euler step size :math:`\\varepsilon` (alias of :attr:`dt`)."""
        return self.dt

    @property
    def gamma(self) -> float:
        r"""
        float : Diffusion coefficient :math:`\gamma \ge 0`.

        A *live* knob: assigning to it (directly or through
        ``EuSNLayer.gamma = ...``) validates the value and takes effect on the
        next forward.  ``0.0`` recovers the undamped antisymmetric flow.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        scalar = float(value)
        if scalar < 0.0:
            raise ValueError(f"gamma must be non-negative, got {value}")
        self._gamma = scalar

    def vector_field(self, state: torch.Tensor, projected_u: torch.Tensor) -> torch.Tensor:
        r"""EuSN field :math:`\tanh\big((W_r - W_r^\top - \gamma I)\,h + u\big)`.

        Antisymmetrizes the frozen recurrent matrix on the fly
        (:math:`W_r - W_r^\top`), subtracts the diffusion :math:`\gamma I`, adds
        the precomputed input drive ``projected_u``, and applies the activation
        (``tanh`` by default).

        Parameters
        ----------
        state : torch.Tensor
            Current state ``h``, shape ``(batch, reservoir_size)``.
        projected_u : torch.Tensor
            Precomputed input drive ``u = W_fb x_fb + W_in u + b`` for the
            interval, shape ``(batch, reservoir_size)``.

        Returns
        -------
        torch.Tensor
            :math:`\dot h`, shape ``(batch, reservoir_size)``.
        """
        # Antisymmetric recurrence W_h = W_r - W_r^T, applied as h (W_r - W_r^T)^T
        # = h W_r^T - h W_r via F.linear conventions.  Compute h @ W_h^T directly:
        # W_h^T = (W_r - W_r^T)^T = W_r^T - W_r, so h @ W_h^T = h @ W_r^T - h @ W_r.
        w = self.weight_hh
        recurrent = state @ w.t() - state @ w
        pre = recurrent - self._gamma * state + projected_u
        return self.activation_fn(pre)

    def __repr__(self) -> str:
        """Return string representation."""
        input_str = f", input_size={self.input_size}" if self.input_size is not None else ""
        return (
            f"EuSNCell("
            f"reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}"
            f"{input_str}, "
            f"epsilon={self.epsilon}, "
            f"gamma={self.gamma}, "
            f"spectral_radius={self.spectral_radius}, "
            f"solver={self.solver}"
            f")"
        )
