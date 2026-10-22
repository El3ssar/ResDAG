r"""
Continuous-Time Reservoir Cells (shared base)
=============================================

This module provides :class:`ContinuousReservoirCell`, an intermediate abstract
base for reservoir cells whose state evolves by **numerically integrating an
ODE** :math:`\dot h = f(h, u)` rather than by a closed-form discrete map.  It is
the groundwork for the continuous / liquid-state family; the first concrete
subclass is the Euler State Network
(:class:`~resdag.layers.cells.eusn_cell.EuSNCell`).

The base owns the sub-stepping and the input projection; a subclass supplies
**only** the vector field :meth:`~ContinuousReservoirCell.vector_field` — the
:math:`\dot h = f(h, u)` term.  One input interval (one timestep of the driving
sequence) is integrated with ``substeps`` fixed sub-steps of size
``dt / substeps`` using forward **Euler** or classical **RK4**, holding the
input drive :math:`u` constant across the interval (the standard zero-order-hold
discretization used in reservoir computing).  An optional adaptive
**torchdiffeq** branch is available behind the ``resdag[ode]`` extra.

Design
------
* **Weight ownership** is inherited from
  :class:`~resdag.layers.cells.esn_cell.ESNCell` (feedback / input weights, the
  frozen recurrent matrix ``weight_hh``, the random bias, spectral-radius
  scaling, per-parameter freezing driven by ``trainable``, and seed threading),
  exactly as :class:`~resdag.layers.cells._skip_cell._SkipReservoirCell` does.
  The leaky-integrator mix of ``ESNCell`` is **not** used — the ODE integration
  replaces it (``leak_rate`` is fixed to ``1.0`` and ``noise`` to ``0.0``).
* **project_inputs** reuses ``ESNCell``'s projection verbatim
  (:math:`W_{fb} x_{fb} + W_{in} u + b` for the whole sequence in one batched
  matmul), so the compiled/chunked forecast + teacher-forced engines apply
  unchanged.
* **step** integrates one interval from the precomputed projection slice; its
  sub-step loop is a *static* Python ``range(substeps)``, so it traces cleanly
  under :func:`torch.compile` (``fullgraph=True``) for the fixed-step solvers.
* **output_from_state** defaults to the identity; subclasses whose readable
  output differs from the raw state override it.

Solvers
-------
``solver="euler"`` (default) and ``solver="rk4"`` need **no** extra dependency
and are the compilable path.  ``solver="torchdiffeq"`` lazily imports
``torchdiffeq`` (guarded :class:`ImportError` → ``pip install resdag[ode]``) and
uses an adaptive solver over each interval with a zero-order-hold input.  The
adaptive branch may not trace ``fullgraph=True``; the forecast engine then
degrades to the eager rollout, which is correct just slower.

See Also
--------
resdag.layers.cells.esn_cell.ESNCell : The cell whose weight machinery this reuses.
resdag.layers.cells.eusn_cell.EuSNCell : The first concrete continuous cell.
resdag.init.matrices.antisymmetric : Antisymmetric matrix init (EuSN recurrence).
"""

import math
from abc import abstractmethod

import torch

from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.utils.general import SeedLike

from .esn_cell import ESNCell

# Fixed-step solvers implemented in-tree (no dependency, compile-friendly) plus
# the optional adaptive branch.  Kept as a frozenset so an unknown solver name
# fails fast at construction with the valid set in the message.
_FIXED_STEP_SOLVERS = frozenset({"euler", "rk4"})
_VALID_SOLVERS = _FIXED_STEP_SOLVERS | {"torchdiffeq"}


class ContinuousReservoirCell(ESNCell):
    r"""
    Abstract base for reservoir cells that integrate :math:`\dot h = f(h, u)`.

    Extends :class:`~resdag.layers.cells.esn_cell.ESNCell` with fixed-step ODE
    integration over each input interval.  Concrete subclasses implement only
    :meth:`vector_field` (the :math:`\dot h = f(h, u)` term); this base supplies
    the sub-stepping (:meth:`step`), the input projection (inherited
    :meth:`~ESNCell.project_inputs`), and the sequence-level ``forward``.

    The state update integrates one input interval — one timestep of the driving
    sequence — with ``substeps`` fixed sub-steps of size ``dt / substeps``,
    holding the projected input drive :math:`u` constant across the interval
    (zero-order hold).  Writing :math:`g = \text{dt}/\text{substeps}` for the
    sub-step size, one **Euler** sub-step is

    .. math::

        h \leftarrow h + g \, f(h, u),

    and one classical **RK4** sub-step is the usual four-stage weighted average.
    The recurrent / feedback / input weights, bias, spectral scaling and
    freezing are all owned and initialized by :class:`ESNCell`.

    Parameters
    ----------
    reservoir_size, feedback_size, input_size, spectral_radius, bias, \
bias_scaling, activation, trainable, feedback_initializer, input_initializer, \
topology, seed
        Identical in meaning to :class:`~resdag.layers.cells.esn_cell.ESNCell`
        (see its docstring).  ``leak_rate`` and ``noise`` are intentionally
        absent — the ODE integration replaces leaky integration, and train-mode
        state noise is not part of the continuous-cell formulation.
    dt : float, default=1.0
        Length of one input interval in integration time.  The forward-Euler
        step size of the Euler State Network is ``dt`` with ``substeps=1``.
    substeps : int, default=1
        Number of fixed integration sub-steps per input interval.  Each has size
        ``dt / substeps``.  More sub-steps integrate the same interval more
        accurately; ``1`` is the single-step form used by EuSN.
    solver : {'euler', 'rk4', 'torchdiffeq'}, default='euler'
        Integration scheme.  ``'euler'`` and ``'rk4'`` are fixed-step, need no
        extra dependency, and trace ``fullgraph=True`` under
        :func:`torch.compile`.  ``'torchdiffeq'`` uses an adaptive solver over
        each interval (zero-order-hold input) and requires the optional
        ``resdag[ode]`` extra; it may not compile ``fullgraph=True``.
    torchdiffeq_method : str, default='dopri5'
        Adaptive method passed to ``torchdiffeq.odeint`` when
        ``solver='torchdiffeq'``.  Ignored otherwise.
    rtol, atol : float, default=1e-6, 1e-8
        Relative / absolute tolerances for the ``torchdiffeq`` adaptive solver.
        Ignored by the fixed-step solvers.

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
    dt : float
        Input-interval length (integration time).
    substeps : int
        Fixed sub-steps per interval.
    solver : str
        Active solver name.
    weight_hh, weight_feedback, weight_input, bias_h
        Recurrent, feedback, input weights and bias (see
        :class:`~resdag.layers.cells.esn_cell.ESNCell`).

    Notes
    -----
    ``vector_field`` receives the **precomputed** input drive ``projected_u``
    (``W_fb x_fb + W_in u + b`` for this interval), held constant across every
    sub-step — the caller does the projection once per interval, so the field
    itself only mixes in the recurrent / diffusion term that depends on ``h``.

    See Also
    --------
    resdag.layers.cells.eusn_cell.EuSNCell : Concrete Euler State Network cell.
    resdag.layers.cells.esn_cell.ESNCell : The leaky-ESN cell this base extends.
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
        dt: float = 1.0,
        substeps: int = 1,
        solver: str = "euler",
        torchdiffeq_method: str = "dopri5",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
        seed: SeedLike = None,
        *,
        recurrent_initializer: TopologySpec = None,
        bias_initializer: InitializerSpec = None,
        initial_state_initializer: InitializerSpec = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if substeps < 1:
            raise ValueError(f"substeps must be a positive integer, got {substeps}")
        if solver not in _VALID_SOLVERS:
            raise ValueError(f"Unknown solver {solver!r}. Supported: {sorted(_VALID_SOLVERS)}.")
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.solver = solver
        self.torchdiffeq_method = torchdiffeq_method
        self.rtol = float(rtol)
        self.atol = float(atol)

        # ESNCell owns the recurrent / feedback / input / bias machinery; drive
        # it with a fixed leak_rate=1.0 (the integrator replaces leaky mixing) and
        # no state noise (not part of the continuous-cell formulation).
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

    # ------------------------------------------------------------------
    # Live integration interval
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """Positive, finite integration interval; changes apply to the next step."""
        # Old full-model pickles stored this as a plain attribute.
        if "_dt" in self.__dict__:
            return float(self.__dict__["_dt"])
        return float(self.__dict__["dt"])

    @dt.setter
    def dt(self, value: float) -> None:
        scalar = float(value)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {value}")
        self._dt = scalar

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def vector_field(self, state: torch.Tensor, projected_u: torch.Tensor) -> torch.Tensor:
        r"""Return the ODE right-hand side :math:`\dot h = f(h, u)` for one interval.

        Called once per integration sub-step with the *same* ``projected_u``
        (zero-order hold across the interval).  The input drive is already
        projected (``W_fb x_fb + W_in u + b``), so the field only adds the
        recurrent / diffusion term that depends on ``state``.

        Parameters
        ----------
        state : torch.Tensor
            Current reservoir state, shape ``(batch, reservoir_size)``.
        projected_u : torch.Tensor
            Precomputed input drive for this interval, shape
            ``(batch, reservoir_size)``, constant across sub-steps.

        Returns
        -------
        torch.Tensor
            The time derivative :math:`\dot h`, shape ``(batch, reservoir_size)``.
        """
        ...

    def output_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """Map the integrated state to the readable per-step output.

        Defaults to the identity (output == state), matching :class:`ESNCell`.
        Subclasses whose readable output differs from the raw state override
        this.

        Parameters
        ----------
        state : torch.Tensor
            Integrated reservoir state, shape ``(batch, reservoir_size)``.

        Returns
        -------
        torch.Tensor
            Per-step output, shape ``(batch, output_size)``.
        """
        return state

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def _integrate_fixed(self, state: torch.Tensor, projected_u: torch.Tensor) -> torch.Tensor:
        """Integrate one interval with ``substeps`` fixed Euler / RK4 sub-steps.

        The sub-step loop is a *static* ``range(self.substeps)`` (a Python int
        fixed at construction), and the per-sub-step body is branch-free once the
        solver is chosen, so the whole thing unrolls cleanly under
        :func:`torch.compile` (``fullgraph=True``).

        Parameters
        ----------
        state : torch.Tensor
            Current state, shape ``(batch, reservoir_size)``.
        projected_u : torch.Tensor
            Constant input drive for the interval, shape
            ``(batch, reservoir_size)``.

        Returns
        -------
        torch.Tensor
            State after integrating the full interval.
        """
        h = state
        g = self.dt / self.substeps
        if self.solver == "rk4":
            for _ in range(self.substeps):
                k1 = self.vector_field(h, projected_u)
                k2 = self.vector_field(h + 0.5 * g * k1, projected_u)
                k3 = self.vector_field(h + 0.5 * g * k2, projected_u)
                k4 = self.vector_field(h + g * k3, projected_u)
                h = h + (g / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:  # "euler"
            for _ in range(self.substeps):
                h = h + g * self.vector_field(h, projected_u)
        return h

    def _integrate_torchdiffeq(
        self, state: torch.Tensor, projected_u: torch.Tensor
    ) -> torch.Tensor:
        """Integrate one interval with an adaptive ``torchdiffeq`` solver (ZOH input).

        Lazily imports ``torchdiffeq`` behind a guarded :class:`ImportError`
        pointing at the ``resdag[ode]`` extra.  The input drive ``projected_u``
        is held constant across the interval (zero-order hold), matching the
        fixed-step solvers; ``odeint`` is asked only for the endpoint ``t=dt``.

        Parameters
        ----------
        state : torch.Tensor
            Current state, shape ``(batch, reservoir_size)``.
        projected_u : torch.Tensor
            Constant input drive for the interval.

        Returns
        -------
        torch.Tensor
            State after integrating the full interval.

        Raises
        ------
        ImportError
            If ``torchdiffeq`` is not installed.
        """
        try:
            from torchdiffeq import odeint
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "solver='torchdiffeq' requires the optional 'torchdiffeq' dependency. "
                "Install it with:  pip install resdag[ode]  (or  uv sync --extra ode)."
            ) from exc

        def rhs(_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # Autonomous over the interval: the input drive is constant (ZOH).
            return self.vector_field(h, projected_u)

        t = torch.tensor([0.0, self.dt], device=state.device, dtype=state.dtype)
        solution = odeint(
            rhs, state, t, method=self.torchdiffeq_method, rtol=self.rtol, atol=self.atol
        )
        # solution has shape (len(t), batch, reservoir_size); take the endpoint.
        endpoint: torch.Tensor = solution[-1]
        return endpoint

    def _integrate(self, state: torch.Tensor, projected_u: torch.Tensor) -> torch.Tensor:
        """Dispatch to the fixed-step or adaptive integrator by :attr:`solver`."""
        if self.solver in _FIXED_STEP_SOLVERS:
            return self._integrate_fixed(state, projected_u)
        return self._integrate_torchdiffeq(state, projected_u)

    # ------------------------------------------------------------------
    # ReservoirCell interface (forward / step)
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Integrate one input interval and return ``(output, new_state)``.

        Projects the per-step inputs (``W_fb x_fb + W_in u + b``) and integrates
        the vector field over one interval of length :attr:`dt`.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            ``inputs[0]`` is the feedback slice ``(batch, feedback_size)``;
            ``inputs[1]`` (optional) the driving input ``(batch, input_size)``.
        state : torch.Tensor
            Current reservoir state, shape ``(batch, reservoir_size)``.

        Returns
        -------
        output : torch.Tensor
            Per-step output ``(batch, output_size)`` from
            :meth:`output_from_state`.
        new_state : torch.Tensor
            Integrated reservoir state ``(batch, reservoir_size)``.
        """
        projected = self.project_inputs(inputs)
        return self.step(projected, state)

    def step(
        self,
        projected_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Integrate one interval from a precomputed projection slice.

        Consumes ``projected_t`` (``W_fb x_fb + W_in u + b`` for this interval)
        and integrates the vector field over one interval.  For the fixed-step
        solvers the sub-step loop is static, so this traces cleanly under
        :func:`torch.compile` (``fullgraph=True``), enabling the compiled
        forecast / teacher-forced engines.

        Parameters
        ----------
        projected_t : torch.Tensor
            Slice of :meth:`~ESNCell.project_inputs`, shape
            ``(batch, reservoir_size)``.
        state : torch.Tensor
            Current reservoir state, shape ``(batch, reservoir_size)``.

        Returns
        -------
        output : torch.Tensor
            Per-step output ``(batch, output_size)``.
        new_state : torch.Tensor
            Integrated reservoir state ``(batch, reservoir_size)``.
        """
        new_state = self._integrate(state, projected_t)
        return self.output_from_state(new_state), new_state
