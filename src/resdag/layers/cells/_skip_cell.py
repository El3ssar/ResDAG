"""
Orthogonal-Skip Reservoir Cells (shared base)
==============================================

This private module provides :class:`_SkipReservoirCell`, the shared base for
the two edge-of-stability reservoir cells that augment the leaky-ESN update
with a **fixed random orthogonal skip connection**:

- :class:`~resdag.layers.cells.es2n_cell.ES2NCell` — Edge-of-Stability State
  Network (Ceni & Gallicchio, 2023): the skip and the nonlinear branch form a
  convex combination weighted by a single proximity parameter ``beta``.
- :class:`~resdag.layers.cells.resesn_cell.ResESNCell` — Residual ESN (Ceni &
  Gallicchio, 2024): the skip weight ``alpha`` and the nonlinear weight
  ``beta`` are decoupled.

Both share the ESN weight-ownership machinery — a spectral-radius-scaled frozen
recurrent matrix ``W_r``, feedback/input weights from the input-feedback
registry, a random bias, and per-parameter freezing driven by ``trainable`` —
so this base reuses :class:`~resdag.layers.cells.esn_cell.ESNCell`'s
initialization helpers by subclassing it.  What it adds is the frozen
orthogonal skip matrix ``O`` and the two-branch state update; concrete
subclasses supply only how the skip and nonlinear branches are combined.

See Also
--------
resdag.layers.cells.esn_cell.ESNCell : The leaky-ESN cell this base extends.
resdag.init.matrices.orthogonal : The Haar-random orthogonal matrix builder.
"""

import torch
import torch.nn.functional as F

from resdag.init.matrices.orthogonal import orthogonal_matrix
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.utils.general import SeedLike

from .esn_cell import ESNCell, _salt_seed

# Golden-ratio offset separating the orthogonal-skip draw from every other
# init stream (feedback / input / recurrent / bias).  Threading the raw layer
# seed straight into ``orthogonal_matrix`` would tie ``O`` to the recurrent
# draw; a distinct offset keeps the skip statistically independent while
# staying a deterministic function of ``seed``.  ``0x9E3779B9 * 3`` is the
# next Fibonacci-hashing multiple after the input-weight offset in ``esn_cell``.
_ORTHOGONAL_SEED_OFFSET = (0x9E3779B9 * 3) % (2**63 - 1)


class _SkipReservoirCell(ESNCell):
    """
    Shared base for reservoir cells with a fixed orthogonal skip connection.

    Extends :class:`~resdag.layers.cells.esn_cell.ESNCell` with a frozen random
    orthogonal matrix ``O`` (the residual / skip branch) and a two-branch state
    update

    .. math::

        x_t = a\\,O\\,x_{t-1} + b\\,\\phi(W_r x_{t-1} + W_{fb} x_{fb,t}
              + W_{in} u_t + \\beta_{bias})

    where the scalar branch weights :math:`a` (skip) and :math:`b` (nonlinear)
    are supplied by the concrete subclass via :meth:`_branch_weights`.  All of
    ``W_r`` (spectral-radius-scaled, frozen), ``W_fb`` / ``W_in`` (input-feedback
    registry) and the bias are owned and initialized exactly as in
    :class:`ESNCell`; the leaky-integrator mix of ``ESNCell`` is **not** used —
    the two-branch combination replaces it.

    Parameters
    ----------
    reservoir_size, feedback_size, input_size, spectral_radius, bias, \
bias_scaling, activation, trainable, feedback_initializer, input_initializer, \
topology, seed
        Identical in meaning to :class:`ESNCell` (see its docstring).  Note
        ``leak_rate`` and ``noise`` are intentionally absent: the orthogonal
        skip plays the role of the leaky-integrator memory term, so a separate
        leak rate would be redundant, and train-mode state noise is not part of
        the ES2N / ResESN formulation.
    orthogonal_scaling : float, default=1.0
        Gain applied to the orthogonal skip matrix ``O``.  With ``1.0`` the
        skip is exactly norm-preserving (all singular values equal 1), the
        regime in which the edge-of-stability property holds.  Other values
        scale the linear memory branch.

    recurrent_initializer : spec, optional
        First-class alias for ``topology``. Supply one spelling only. Names,
        tuples, callables and registered initializer objects are accepted.
    bias_initializer : spec, optional
        Input/feedback initializer grammar applied to a ``(1, reservoir_size)``
        view of the bias. Requires ``bias=True``; overrides ``bias_scaling``.
    initial_state_initializer : spec, optional
        Initializes a persistent ``(1, cell.state_size)`` reset template once.
        Each batch example receives an independent copy at reset. Zero by default.
    device : torch.device or str, optional
        Initial parameter location. Metal construction initializes on CPU before
        transferring to support initialization operations across backends.
    dtype : torch.dtype, optional
        Floating precision used during initialization, not only after conversion.

    Attributes
    ----------
    skip_matrix : torch.Tensor
        The frozen ``(reservoir_size, reservoir_size)`` orthogonal skip matrix
        ``O``, registered as a non-trainable buffer so it moves with ``.to()``
        / ``.double()`` and round-trips through save/load.

    See Also
    --------
    resdag.layers.cells.es2n_cell.ES2NCell : Convex-combination (ES2N) subclass.
    resdag.layers.cells.resesn_cell.ResESNCell : Decoupled-weight (ResESN) subclass.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        spectral_radius: float | None = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        activation: str = "tanh",
        trainable: bool = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
        orthogonal_scaling: float = 1.0,
        seed: SeedLike = None,
        *,
        recurrent_initializer: TopologySpec = None,
        bias_initializer: InitializerSpec = None,
        initial_state_initializer: InitializerSpec = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # ESNCell owns the recurrent/feedback/input/bias machinery; drive it with
        # a fixed leak_rate=1.0 (the skip supersedes leaky integration) and no
        # state noise (not part of the ES2N/ResESN formulation).  ``trainable``
        # is applied to *all* parameters at the end of ESNCell.__init__; the
        # orthogonal skip is a buffer, so it stays frozen regardless.
        self.orthogonal_scaling = orthogonal_scaling
        super().__init__(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            input_size=input_size,
            spectral_radius=spectral_radius,
            bias=bias,
            bias_scaling=bias_scaling,
            activation=activation,
            leak_rate=1.0,
            noise=0.0,
            trainable=trainable,
            feedback_initializer=feedback_initializer,
            input_initializer=input_initializer,
            topology=topology,
            seed=seed,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            initial_state_initializer=initial_state_initializer,
            device=device,
            dtype=dtype,
        )

        # Build the frozen orthogonal skip ``O`` on a stream offset from the
        # weight-init draws so it is independent of the recurrent matrix yet
        # still a deterministic function of ``seed``.  Registered as a
        # non-persistent-agnostic (default persistent=True) buffer so it saves,
        # loads, and follows ``.to()`` / ``.double()`` with the parameters.
        skip_seed = _salt_seed(self._seed_int, _ORTHOGONAL_SEED_OFFSET)
        skip = orthogonal_matrix(
            reservoir_size,
            gain=orthogonal_scaling,
            seed=skip_seed,
            dtype=self.weight_hh.dtype,
            device=self.weight_hh.device,
        )
        # Align the skip with the recurrent matrix's dtype/device so it sits
        # where the weights are before any ``.to()`` the caller may apply.
        skip = skip.to(dtype=self.weight_hh.dtype, device=self.weight_hh.device)
        self.register_buffer("skip_matrix", skip)
        self.skip_matrix: torch.Tensor

    # ------------------------------------------------------------------
    # Two-branch state update
    # ------------------------------------------------------------------

    def _branch_weights(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Return the ``(skip_weight, nonlinear_weight)`` scalar pair.

        Read fresh on every step so live writes to ``beta`` / ``alpha`` take
        effect on the next forward.  Concrete subclasses override this to encode
        their combination rule (ES2N: ``(1 - beta, beta)``; ResESN:
        ``(alpha, beta)``).

        Returns
        -------
        skip_weight, nonlinear_weight
            Scalars weighting the orthogonal skip branch and the nonlinear
            branch, respectively.
        """
        raise NotImplementedError

    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the next state for a single timestep.

        Realises

        .. math::

            x_t = a\\,O\\,x_{t-1} + b\\,\\phi(W_r x_{t-1} + \\text{projected})

        where ``projected`` = :meth:`~ESNCell.project_inputs` (``W_fb x_fb +
        W_in u + bias``) and ``(a, b)`` = :meth:`_branch_weights`.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            ``inputs[0]`` is the feedback slice; ``inputs[1]`` (optional) the
            driving-input slice.  Each ``(batch, feature_dim)``.
        state : torch.Tensor
            Current state, shape ``(batch, reservoir_size)``.

        Returns
        -------
        output : torch.Tensor
            Next state of shape ``(batch, reservoir_size)``.
        new_state : torch.Tensor
            Same tensor as ``output`` (state and output are identical).
        """
        projected = self.project_inputs(inputs)
        return self.step(projected, state)

    def step(
        self,
        projected_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent-only single-step update for the projected fast path.

        Consumes a precomputed input projection ``projected_t`` (``W_fb x_fb +
        W_in u + bias`` for this timestep) and applies the two-branch update.
        Branch-free and shape-static so it traces cleanly under
        :func:`torch.compile` (``fullgraph=True``), enabling the compiled
        forecast/teacher-forced engines.

        Parameters
        ----------
        projected_t : torch.Tensor
            Slice of :meth:`~ESNCell.project_inputs`, shape
            ``(batch, reservoir_size)``.
        state : torch.Tensor
            Current state, shape ``(batch, reservoir_size)``.

        Returns
        -------
        output : torch.Tensor
            Next state.
        new_state : torch.Tensor
            Same tensor as ``output``.
        """
        skip_weight, nonlinear_weight = self._branch_weights()
        # Nonlinear branch: phi(W_r x + projected).  ``addmm`` fuses the
        # recurrent matmul with the precomputed projection into one kernel.
        pre_activation = torch.addmm(projected_t, state, self.weight_hh.t())
        nonlinear = self.activation_fn(pre_activation)
        # Linear orthogonal skip branch: O x.
        skip = F.linear(state, self.skip_matrix)
        new_state = skip_weight * skip + nonlinear_weight * nonlinear
        return new_state, new_state
